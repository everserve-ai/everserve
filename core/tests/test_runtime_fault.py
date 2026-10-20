# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Four CPU workers run the reference cores, the runtime and the KV copy end to end.

gloo forms their groups; the doubles in ``collective_doubles`` reduce, with the slot kernel's and
NCCL's failure semantics, so a survivor blocked on the killed rank is released by its change as in
production, never by a gloo timeout.

The test process plays the executor: it sends each step to every worker and, as the executor would,
SIGKILLs one worker and tells the survivors over the control channel. Output after the kill must
equal a no-fault run, token for token and logit for logit, whether the kill lands between steps or
in the middle of one.
"""

import hashlib
import json
import multiprocessing as mp
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from collective_doubles import AbortableGroup, Mesh, SlotDouble
from ports import free_ports

from everserve.comm.collective import Collective, ranks_of
from everserve.comm.group import ShrinkableGroup, ShrinkMethod
from everserve.comm.runtime import Runtime
from everserve.comm.shadow import form_groups, member_sets
from everserve.reference.arith.cores import LlamaCore, Qwen3MoeCore
from everserve.reference.arith.sources import DeterministicSource
from everserve.reference.control import FaultClient, FaultListener, broadcast_fault
from everserve.reference.kvcopy import KVCopy
from everserve.reference.store import start_server

pytestmark = pytest.mark.slow

W, SESS, M, STEPS, BS = 4, 8, 12, 24, 4
STUCK_S = 120
"""A rank that sends nothing for this long is stuck: past its own 90 s stack dump."""

QWEN = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=16,
    vocab_size=97,
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    num_experts=32,
    num_experts_per_tok=4,
    moe_intermediate_size=32,
)
LLAMA = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=8,
    vocab_size=97,
    rms_norm_eps=1e-5,
    rope_theta=5e5,
    intermediate_size=224,
    mlp_unit=8,
    rope_scaling={
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
    },
)


def shapes(cfg):
    h, d = cfg["hidden_size"], cfg["head_dim"]

    def f(n):
        if "embed_tokens" in n or n == "lm_head.weight":
            return (cfg["vocab_size"], h)
        if "q_norm" in n or "k_norm" in n:
            return (d,)
        if "norm" in n:
            return (h,)
        if "q_proj" in n:
            return (cfg["num_attention_heads"] * d, h)
        if "k_proj" in n or "v_proj" in n:
            return (cfg["num_key_value_heads"] * d, h)
        if "o_proj" in n:
            return (h, cfg["num_attention_heads"] * d)
        if "mlp.gate.weight" in n:
            return (cfg["num_experts"], h)
        inter = cfg.get("moe_intermediate_size") if "experts" in n else cfg.get("intermediate_size")
        if "gate_proj" in n or "up_proj" in n:
            return (inter, h)
        if "down_proj" in n:
            return (h, inter)
        raise KeyError(n)

    return f


def worker(kind, rank, conn, store_port, control_port, kv_store_addr, log_dir, shadow_ports):
    import faulthandler

    # a worker stuck for 90 s dumps every thread's stack: the evidence for a hang
    faulthandler.dump_traceback_later(
        90,
        repeat=True,
        file=open(Path(log_dir) / f"stack{rank}.txt", "w"),  # noqa: SIM115
    )
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    cfg = QWEN if kind == "ep" else LLAMA
    store = dist.TCPStore("127.0.0.1", store_port, None, False)
    device = torch.device("cpu")
    # gloo forms the groups; the doubles reduce, with the slot kernel's and NCCL's failure
    # semantics, so a survivor blocked on the killed rank is released by its change, as in
    # production, never by a gloo timeout
    mesh = Mesh(log_dir, rank)
    live = AbortableGroup(ShrinkableGroup.from_store(store, rank, W, "gloo"), mesh, "init")
    shadows = {}
    if shadow_ports:  # pre-form every survivor set beside the live group
        shadows = {
            members: AbortableGroup(
                ShrinkableGroup(pg, ranks_of(members), "gloo", None, ShrinkMethod.SHADOW),
                mesh,
                f"shadow-{'-'.join(map(str, members))}",
            )
            for members, pg in form_groups(rank, member_sets(W), "127.0.0.1", shadow_ports).items()
        }
    collective = Collective(rank, live, shadows, slot=SlotDouble(mesh, rank, range(W)))
    rt = Runtime(rank, W, device, collective, event_log=Path(log_dir))

    def on_fault(dead, sent_at):
        refused = collective.current
        rt.on_fault(dead, sent_at)
        # the adapter's part (decision 0025): a group the change swapped away from is released;
        # a shrunk one was aborted by its shrink
        if refused is not collective.current and isinstance(refused, AbortableGroup):
            refused.abort()

    listener = FaultListener(("127.0.0.1", control_port), on_fault, rt.identity)
    layers, d = cfg["num_hidden_layers"], cfg["head_dim"]
    ctx = {}
    caches = {}

    def attn(layer, q, k, v):
        # reference paged attention: write k/v at the step's slots, attend causally per session
        cache = caches[f"model.layers.{layer}.self_attn.attn"]
        slots, sess, pos = ctx["slots"], ctx["sess"], ctx["pos"]
        b, o = slots // BS, slots % BS
        cache[0, b, o] = k
        cache[1, b, o] = v
        kvc.save(f"model.layers.{layer}.self_attn.attn", cache, slots)
        t, nh, _ = q.shape
        out = torch.empty(t, nh * d, dtype=q.dtype)
        for i in range(t):
            ss = sess[i] * (M + STEPS + BS) + torch.arange(pos[i] + 1)
            keys, vals = cache[0, ss // BS, ss % BS], cache[1, ss // BS, ss % BS]
            a = F.scaled_dot_product_attention(
                q[i].unsqueeze(1), keys.transpose(0, 1), vals.transpose(0, 1), enable_gqa=True
            )
            out[i] = a.reshape(-1)
        return out

    core_cls = Qwen3MoeCore if kind == "ep" else LlamaCore
    core = core_cls(cfg, DeterministicSource(shapes(cfg)), device, rt, attn, rank, W)
    nb = SESS * (M + STEPS + BS) // BS + 1
    for layer in range(layers):
        caches[f"model.layers.{layer}.self_attn.attn"] = torch.zeros(
            2, nb, BS, core.cap, d, dtype=torch.bfloat16
        )
    kvc = KVCopy(kv_store_addr, core.slot_units)
    kvc.register(caches)
    rt.attempt_callbacks.append(kvc.clear)
    rt.commit_callbacks.append(kvc.commit)

    def on_recover(dead):
        info, restore = core.absorb(dead)
        kvc.restore(restore)
        return info

    rt.recover_callbacks.append(on_recover)
    conn.send("ready")
    while True:
        cmd = conn.recv()
        if cmd[0] == "exit":
            listener.close()
            return
        _, ids, pos, sess = cmd
        ids, pos, sess = torch.tensor(ids), torch.tensor(pos), torch.tensor(sess)
        ctx.update(slots=sess * (M + STEPS + BS) + pos, sess=sess, pos=pos)
        h = rt.run_step(lambda ids=ids, pos=pos: core.forward(ids, pos))
        last = h if cmd[0] == "step" else h.view(SESS, M, -1)[:, -1]
        lg = core.logits(last)
        digests = [hashlib.sha256(r.numpy().tobytes()).hexdigest()[:12] for r in lg]
        conn.send((lg.argmax(-1).tolist(), digests))


def run(kind, kill=None, events=None, shadows=False):
    """kill: None, ("between", step) or ("mid", step, delay_s). shadows: pre-form survivor sets."""
    tmp = tempfile.mkdtemp(prefix="everserve-")
    ctx = mp.get_context("spawn")
    # the listeners this process starts take port 0 and say which they took
    master = dist.TCPStore("127.0.0.1", 0, None, True, wait_for_workers=False)
    store_port = master.port
    kv_store = start_server(("127.0.0.1", 0))
    control_ports = free_ports(W)
    shadow_ports = free_ports(W) if shadows else []
    procs, conns = [], []
    for r in range(W):
        a, b = ctx.Pipe()
        p = ctx.Process(
            target=worker,
            args=(kind, r, b, store_port, control_ports[r], kv_store.address, tmp, shadow_ports),
            daemon=True,
        )
        p.start()
        procs.append(p)
        conns.append(a)
    for c in conns:
        assert c.recv() == "ready"
    ctl = {r: FaultClient(("127.0.0.1", control_ports[r])) for r in range(W)}
    g = torch.Generator().manual_seed(1)
    prompts = torch.randint(0, 97, (SESS, M), generator=g)
    alive = list(range(W))

    def send_all(msg):
        for r in alive:
            conns[r].send(msg)

    def evidence():
        # what every rank logged, and the stacks of any rank stuck past its 90 s dump
        for f in sorted(Path(tmp).glob("*")):
            if f.suffix in (".json", ".txt"):
                print(f"===== {f.name}\n{f.read_text()[-6000:]}")

    def recv_all():
        outs = []
        try:
            for r in alive:
                # past the workers' 90 s stack dump, so a stuck rank's stacks are on disk
                if not conns[r].poll(STUCK_S):
                    evidence()
                    pytest.fail(f"rank {r} sent nothing in {STUCK_S} s")
                outs.append(conns[r].recv())
        except EOFError:
            evidence()  # a worker died
            raise
        assert all(o == outs[0] for o in outs), "ranks disagree"
        return outs[0]

    def fault(dead):
        procs[dead].kill()
        procs[dead].join()
        alive.remove(dead)
        broadcast_fault([ctl[r] for r in alive], dead)

    send_all(
        (
            "prefill",
            prompts.flatten().tolist(),
            list(range(M)) * SESS,
            [s for s in range(SESS) for _ in range(M)],
        )
    )
    toks, digests = recv_all()
    seq, hs = [[t] for t in toks], [[h] for h in digests]
    cur = toks
    for step in range(1, STEPS + 1):
        send_all(("step", cur, [M + step - 1] * SESS, list(range(SESS))))
        if kill and kill[0] == "mid" and step == kill[1]:
            time.sleep(kill[2])
            fault(2)
        toks, digests = recv_all()
        for s in range(SESS):
            seq[s].append(toks[s])
            hs[s].append(digests[s])
        cur = toks
        if kill and kill[0] == "between" and step == kill[1]:
            fault(2)
    for r in alive:
        conns[r].send(("exit",))
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    kv_store.stop()
    del master
    if events is not None:
        for f in Path(tmp).glob("rank*.json"):
            events[f.name] = json.loads(f.read_text())
    return seq, hs


@pytest.fixture(scope="module")
def base():
    return {k: run(k) for k in ("ep", "tp")}


@pytest.mark.parametrize("kind", ["ep", "tp"])
def test_between_step_kill_is_bit_identical(base, kind):
    assert run(kind, ("between", 10)) == base[kind]


def test_shrink_through_a_preformed_survivor_set_is_bit_identical(base):
    ev = {}
    assert run("tp", ("mid", 12, 0.003), events=ev, shadows=True) == base["tp"]
    methods = {e["fields"]["method"] for v in ev.values() for e in v if e["kind"] == "shrink"}
    assert methods == {"shadow"}


@pytest.mark.parametrize("kind", ["ep", "tp"])
@pytest.mark.parametrize("delay", [0.0, 0.003, 0.01])
def test_mid_step_kill_is_bit_identical(base, kind, delay):
    ev = {}
    assert run(kind, ("mid", 12, delay), events=ev) == base[kind]
    survivors = {"rank0.json", "rank1.json", "rank3.json"}
    assert survivors <= set(ev)  # the dead rank may have traced its early steps before the kill
    kinds = {name: [e["kind"] for e in v] for name, v in ev.items() if name in survivors}
    assert all("shrink" in k and "recover" in k for k in kinds.values())
    if "rank2.json" in ev:
        assert "recover" not in [e["kind"] for e in ev["rank2.json"]]
    if delay == 0.0:  # killed while the step was running: every survivor re-ran it
        assert all("redo" in k for k in kinds.values())


def test_tokens_are_not_degenerate(base):
    seq, _ = base["ep"]
    assert len({tuple(s) for s in seq}) > 1 and len(set(seq[0])) > 1
