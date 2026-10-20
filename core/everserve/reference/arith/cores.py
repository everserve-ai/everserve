# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Reference model cores with placement-independent arithmetic.

Every rank of one engine core sees every token, so hidden states are replicated. Work is cut into
units whose shapes do not depend on the world size:

* attention units: one KV head with its query heads. A rank reserves ``cap`` head slots in its
  attention layer: its own units now, room for absorbed units after a loss;
* Qwen3-MoE: experts in fixed chunks, every decode token through every chunk (fixed shapes);
* Llama: MLP slices of a fixed width.

Decode (a small token count) puts every unit's partial output into its own slot of a buffer, the
buffer is all-reduced (one producer per element, so the sum is exact and independent of which rank
produced which slot), and the slots are summed in fp32 in a fixed order. The result is bit-identical
before and after a membership change. Prefill (a large token count) uses ordinary partial sums.

Attention is injected: ``attn(layer, q[T, cap*qh, d], k[T, cap, d], v[T, cap, d])`` returning
``[T, cap*qh*d]``:
the engine's paged attention in production and a reference implementation in tests.

These cores are the reference the production kernels are checked against, and the model code the
CPU tests run. They import no engine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 - torch convention

from everserve.comm.runtime import Runtime
from everserve.reference.arith.ops import inverse_frequencies, rms_norm, rope
from everserve.reference.arith.sources import WeightSource, fetch
from everserve.reference.plan import after_loss, capacity_after_one_loss, even_units, expert_chunks

DECODE_MAX_TOKENS = 64
"""Steps with at most this many tokens take the exact, slot-per-unit decode path."""

Attention = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
Config = Mapping[str, Any]


class ReferenceCore:
    """Shared embedding, norms and attention units. Subclasses add the MLP and the capacity rule."""

    def __init__(
        self,
        cfg: Config,
        source: WeightSource,
        device: torch.device,
        runtime: Runtime,
        attention: Attention,
        rank: int,
        world: int,
    ) -> None:
        """Load the rank-independent weights. Subclasses load the units they own.

        Args:
            cfg: the Hugging Face config as a mapping (``hidden_size``, ``num_hidden_layers``, ...).
            source: where weights come from.
            device: where the core runs.
            runtime: the membership runtime whose ``all_reduce`` the core uses.
            attention: the injected attention function.
            rank: this rank's original id.
            world: ranks at startup.
        """
        self.cfg, self.source, self.device, self.runtime, self.attention = (
            cfg,
            source,
            device,
            runtime,
            attention,
        )
        self.me, self.world = rank, world
        self.hidden = int(cfg["hidden_size"])
        self.layers = int(cfg["num_hidden_layers"])
        self.head_dim = int(cfg["head_dim"])
        self.kv_heads = int(cfg["num_key_value_heads"])
        self.q_per_kv = int(cfg["num_attention_heads"]) // self.kv_heads
        self.eps = float(cfg["rms_norm_eps"])
        self.inv_freq = inverse_frequencies(
            self.head_dim, float(cfg["rope_theta"]), cfg.get("rope_scaling")
        )
        g = self._load
        self.embed = g("model.embed_tokens.weight")
        self.lm_head = self.embed if cfg.get("tie_word_embeddings") else g("lm_head.weight")
        self.norm = g("model.norm.weight")
        self.layer_norms = [
            (
                g(f"model.layers.{i}.input_layernorm.weight"),
                g(f"model.layers.{i}.post_attention_layernorm.weight"),
            )
            for i in range(self.layers)
        ]
        self.cap = 0
        self._attn_weights: dict[int, list[tuple[torch.Tensor, ...]]] = {}
        self.slot_of: dict[int, int] = {}
        self.wqkv: list[torch.Tensor] = []
        self.wo: list[torch.Tensor] = []
        self.slot_unit = torch.empty(0, dtype=torch.long, device=device)

    def _load(self, name: str) -> torch.Tensor:
        return self.source.get(name).to(self.device)

    # ---------------------------------------------------------------- attention units
    def _load_attention_units(self, units: Sequence[int]) -> None:
        qn, d = self.q_per_kv * self.head_dim, self.head_dim
        for u in units:
            per_layer = []
            for i in range(self.layers):
                p = f"model.layers.{i}.self_attn."
                per_layer.append(
                    (
                        self.source.get(p + "q_proj.weight", rows=slice(u * qn, (u + 1) * qn)).to(
                            self.device
                        ),
                        self.source.get(p + "k_proj.weight", rows=slice(u * d, (u + 1) * d)).to(
                            self.device
                        ),
                        self.source.get(p + "v_proj.weight", rows=slice(u * d, (u + 1) * d)).to(
                            self.device
                        ),
                        self.source.get(p + "o_proj.weight", cols=slice(u * qn, (u + 1) * qn)).to(
                            self.device
                        ),
                    )
                )
            self._attn_weights[u] = per_layer
            free = [j for j in range(self.cap) if j not in self.slot_of.values()]
            self.slot_of[u] = free[0]
        self._stack_attention()

    def _stack_attention(self) -> None:
        """Per layer: ``Wqkv [cap, H, (qh+2)d]`` and ``Wo [cap, qh*d, H]``, zeros in empty slots.

        Every attention matmul then has the same shape before and after a loss.
        """
        qn, d, h = self.q_per_kv * self.head_dim, self.head_dim, self.hidden
        dtype = self.embed.dtype
        self.wqkv, self.wo = [], []
        for layer in range(self.layers):
            a = torch.zeros(self.cap, qn + 2 * d, h, dtype=dtype, device=self.device)
            o = torch.zeros(self.cap, qn, h, dtype=dtype, device=self.device)
            for u, j in self.slot_of.items():
                wq, wk, wv, wo = self._attn_weights[u][layer]
                a[j] = torch.cat([wq, wk, wv], 0)
                o[j] = wo.t()
            self.wqkv.append(a.transpose(1, 2).contiguous())
            self.wo.append(o)
        self.slot_unit = torch.full((self.cap,), -1, dtype=torch.long, device=self.device)
        for u, j in self.slot_of.items():
            self.slot_unit[j] = u

    def slot_units(self) -> list[tuple[int, int]]:
        """``[(head slot, unit), ...]`` for the units this rank holds, by slot."""
        return sorted((j, u) for u, j in self.slot_of.items())

    def _attention_block(
        self, layer: int, x: torch.Tensor, positions: torch.Tensor, decode: bool
    ) -> torch.Tensor:
        t, d, qh, cap = x.shape[0], self.head_dim, self.q_per_kv, self.cap
        qn = qh * d
        h = rms_norm(x, self.layer_norms[layer][0], self.eps)
        qkv = torch.bmm(h.unsqueeze(0).expand(cap, t, self.hidden), self.wqkv[layer])
        q = qkv[..., :qn].permute(1, 0, 2).reshape(t, cap, qh, d)
        k = qkv[..., qn : qn + d].permute(1, 0, 2)
        v = qkv[..., qn + d :].permute(1, 0, 2).contiguous()
        q, k = self._qk_norm(layer, q, k)
        q = rope(q.reshape(t, cap * qh, d), positions, self.inv_freq)
        k = rope(k.contiguous(), positions, self.inv_freq)
        o = self.attention(layer, q, k, v).view(t, cap, qn).permute(1, 0, 2)
        part = torch.bmm(o, self.wo[layer])
        if decode:
            slots = torch.zeros(self.kv_heads, t, self.hidden, dtype=x.dtype, device=x.device)
            live = self.slot_unit >= 0
            slots[self.slot_unit[live]] = part[live]
            self.runtime.all_reduce(slots)
            return x + slots.float().sum(0).to(x.dtype)
        live_idx = (self.slot_unit >= 0).nonzero().squeeze(-1)
        out = part[live_idx].float().sum(0).to(x.dtype)
        self.runtime.all_reduce(out)
        return x + out

    def _qk_norm(
        self, layer: int, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def _mlp_block(self, layer: int, x: torch.Tensor, decode: bool) -> torch.Tensor:
        raise NotImplementedError

    # ---------------------------------------------------------------- the forward
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Vocabulary logits in fp32 for hidden states ``h``."""
        return (h @ self.lm_head.t()).float()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Final hidden states for ``input_ids`` at ``positions``."""
        x = self.embed[input_ids]
        decode = x.shape[0] <= DECODE_MAX_TOKENS
        for layer in range(self.layers):
            x = self._attention_block(layer, x, positions, decode)
            x = self._mlp_block(layer, x, decode)
        return rms_norm(x, self.norm, self.eps)

    def absorb(self, dead: int) -> tuple[dict[str, Any], list[tuple[int, int]]]:
        """Take on this rank's share of ``dead``'s units.

        Returns:
            Facts for the event log, and ``[(unit, head slot), ...]`` for the attention units
            gained, which the caller restores KV for.
        """
        raise NotImplementedError


class Qwen3MoeCore(ReferenceCore):
    """Qwen3-MoE: GQA attention with q/k norms, experts in fixed chunks."""

    CHUNK = 4

    def __init__(
        self,
        cfg: Config,
        source: WeightSource,
        device: torch.device,
        runtime: Runtime,
        attention: Attention,
        rank: int,
        world: int,
    ) -> None:
        """Load this rank's expert chunks and attention units."""
        super().__init__(cfg, source, device, runtime, attention, rank, world)
        self.experts = int(cfg["num_experts"])
        self.top_k = int(cfg["num_experts_per_tok"])
        self.inter = int(cfg["moe_intermediate_size"])
        g = self._load
        self.q_norm = [g(f"model.layers.{i}.self_attn.q_norm.weight") for i in range(self.layers)]
        self.k_norm = [g(f"model.layers.{i}.self_attn.k_norm.weight") for i in range(self.layers)]
        self.gate = [g(f"model.layers.{i}.mlp.gate.weight") for i in range(self.layers)]
        self.chunks_owned = expert_chunks(self.experts, world, self.CHUNK)
        self.attn_owned = even_units(self.kv_heads, world)
        self.cap = self.attention_capacity(self.kv_heads, world, self.experts)
        self.chunks: list[tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]] = []
        self._load_chunks(self.chunks_owned[rank])
        self._load_attention_units(self.attn_owned[rank])

    @classmethod
    def attention_capacity(cls, kv_heads: int, world: int, experts: int) -> int:
        """Head slots a rank reserves, with expert load breaking ties in the attention plan."""
        if world == 1:
            return kv_heads
        chunks = expert_chunks(experts, world, cls.CHUNK)
        heads = even_units(kv_heads, world)
        worst = 0
        for dead in range(world):
            new_chunks, _ = after_loss(chunks, dead)

            def cost(r: int, o: dict, nc: dict = new_chunks) -> tuple[int, int]:
                return len(o[r]), len(nc[r])

            new_heads, _ = after_loss(heads, dead, cost)
            worst = max(worst, max(len(v) for v in new_heads.values()))
        return worst

    def plan_after(self, dead: int):  # noqa: ANN201 - four plans, documented in absorb
        """The expert and attention plans after losing ``dead``, and what each survivor gains."""
        new_chunks, gained_chunks = after_loss(self.chunks_owned, dead)
        new_heads, gained_heads = after_loss(
            self.attn_owned,
            dead,
            lambda r, o: (len(o[r]), len(new_chunks[r])),  # noqa: B023
        )
        return new_chunks, gained_chunks, new_heads, gained_heads

    def absorb(self, dead: int) -> tuple[dict[str, Any], list[tuple[int, int]]]:
        """Load the chunks and attention units this rank gains from ``dead``."""
        new_chunks, gained_chunks, new_heads, gained_heads = self.plan_after(dead)
        self.chunks_owned, self.attn_owned = new_chunks, new_heads
        self._load_chunks(gained_chunks[self.me])
        self._load_attention_units(gained_heads[self.me])
        info = {"chunks_gained": gained_chunks[self.me], "attn_gained": gained_heads[self.me]}
        return info, [(u, self.slot_of[u]) for u in gained_heads[self.me]]

    def _load_chunks(self, chunk_ids: Sequence[tuple[int, ...]]) -> None:
        if not chunk_ids:
            return
        names = [
            f"model.layers.{i}.mlp.experts.{e}.{w}_proj.weight"
            for ids in chunk_ids
            for i in range(self.layers)
            for e in ids
            for w in ("gate", "up", "down")
        ]
        tensors = fetch(self.source, names, self.device)
        for ids in chunk_ids:
            per_layer = []
            for i in range(self.layers):

                def weight(e: int, w: str, i: int = i) -> torch.Tensor:
                    return tensors[f"model.layers.{i}.mlp.experts.{e}.{w}_proj.weight"]

                gate_up = torch.stack(
                    [torch.cat([weight(e, "gate"), weight(e, "up")], 0).t() for e in ids]
                ).contiguous()
                down = torch.stack([weight(e, "down").t() for e in ids]).contiguous()
                per_layer.append((gate_up, down))
            self.chunks.append((torch.tensor(ids, device=self.device), per_layer))

    def _qk_norm(
        self, layer: int, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rms_norm(q, self.q_norm[layer], self.eps), rms_norm(k, self.k_norm[layer], self.eps)

    def _mlp_block(self, layer: int, x: torch.Tensor, decode: bool) -> torch.Tensor:
        t, h_dim = x.shape
        h = rms_norm(x, self.layer_norms[layer][1], self.eps)
        p = torch.softmax(h @ self.gate[layer].t(), dim=-1, dtype=torch.float)
        top_w, top_i = torch.topk(p, self.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        if decode:
            slot = torch.zeros(t, self.top_k, h_dim, dtype=x.dtype, device=x.device)
            for ids, per_layer in self.chunks:
                gate_up, down = per_layer[layer]
                c = gate_up.shape[0]
                a = torch.bmm(h.unsqueeze(0).expand(c, t, h_dim), gate_up)
                a = F.silu(a[..., : self.inter]) * a[..., self.inter :]
                o = torch.bmm(a, down)
                sel = (top_i.unsqueeze(-1) == ids).to(x.dtype)
                slot += (sel.unsqueeze(-1) * o.permute(1, 0, 2).unsqueeze(1)).sum(2)
            self.runtime.all_reduce(slot)
            return x + (slot.float() * top_w.unsqueeze(-1)).sum(1).to(x.dtype)
        out = torch.zeros(t, h_dim, dtype=torch.float, device=x.device)
        for ids, per_layer in self.chunks:
            gate_up, down = per_layer[layer]
            for j, e in enumerate(ids.tolist()):
                hit = top_i == e
                rows = hit.any(-1).nonzero().squeeze(-1)
                if rows.numel() == 0:
                    continue
                a = h[rows] @ gate_up[j]
                a = F.silu(a[:, : self.inter]) * a[:, self.inter :]
                weight = (top_w * hit).sum(-1)[rows].unsqueeze(-1)
                out.index_add_(0, rows, (a @ down[j]).float() * weight)
        out = out.to(x.dtype)
        self.runtime.all_reduce(out)
        return x + out


Tagged = tuple[str, int]
"""A Llama unit: ``("a", kv head)`` or ``("m", mlp slice)``."""


class LlamaCore(ReferenceCore):
    """Llama: GQA attention and a dense MLP cut into fixed-width slices."""

    MLP_UNIT = 512

    def __init__(
        self,
        cfg: Config,
        source: WeightSource,
        device: torch.device,
        runtime: Runtime,
        attention: Attention,
        rank: int,
        world: int,
    ) -> None:
        """Load this rank's attention units and MLP slices."""
        super().__init__(cfg, source, device, runtime, attention, rank, world)
        self.inter = int(cfg["intermediate_size"])
        self.unit = int(cfg.get("mlp_unit", self.MLP_UNIT))
        self.mlp_units = self.inter // self.unit
        self.owned = self.initial(self.kv_heads, self.mlp_units, world)
        self.cap = self.attention_capacity(self.kv_heads, self.mlp_units, world)
        self._mlp_weights: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        mine = self.owned[rank]
        self._load_attention_units([u for kind, u in mine if kind == "a"])
        self._load_mlp_units([b for kind, b in mine if kind == "m"])

    @staticmethod
    def initial(kv_heads: int, mlp_units: int, world: int) -> dict[int, list[Tagged]]:
        """The startup plan: heads and MLP slices split evenly, tagged by kind."""
        a, m = even_units(kv_heads, world), even_units(mlp_units, world)
        return {r: [("a", u) for u in a[r]] + [("m", b) for b in m[r]] for r in range(world)}

    @classmethod
    def attention_capacity(cls, kv_heads: int, mlp_units: int, world: int) -> int:
        """Head slots a rank reserves: the most heads any rank holds after any single loss."""
        if world == 1:
            return kv_heads
        return capacity_after_one_loss(
            cls.initial(kv_heads, mlp_units, world), count=lambda v: sum(k == "a" for k, _ in v)
        )

    def plan_after(self, dead: int) -> tuple[dict[int, list[Tagged]], dict[int, list[Tagged]]]:
        """The plan after losing ``dead`` and what each survivor gains."""
        return after_loss(self.owned, dead)

    def absorb(self, dead: int) -> tuple[dict[str, Any], list[tuple[int, int]]]:
        """Load the units this rank gains from ``dead``."""
        new, gained = self.plan_after(dead)
        self.owned = new
        mine = gained[self.me]
        heads = [u for kind, u in mine if kind == "a"]
        self._load_attention_units(heads)
        self._load_mlp_units([b for kind, b in mine if kind == "m"])
        return {"units_gained": mine}, [(u, self.slot_of[u]) for u in heads]

    def _load_mlp_units(self, blocks: Sequence[int]) -> None:
        for b in blocks:
            sl = slice(b * self.unit, (b + 1) * self.unit)
            per_layer = []
            for i in range(self.layers):
                p = f"model.layers.{i}.mlp."
                per_layer.append(
                    (
                        self.source.get(p + "gate_proj.weight", rows=sl).to(self.device),
                        self.source.get(p + "up_proj.weight", rows=sl).to(self.device),
                        self.source.get(p + "down_proj.weight", cols=sl).to(self.device),
                    )
                )
            self._mlp_weights[b] = per_layer

    def _mlp_block(self, layer: int, x: torch.Tensor, decode: bool) -> torch.Tensor:
        t, h_dim = x.shape
        h = rms_norm(x, self.layer_norms[layer][1], self.eps)
        if decode:
            slots = torch.zeros(self.mlp_units, t, h_dim, dtype=x.dtype, device=x.device)
            for b, per_layer in self._mlp_weights.items():
                wg, wu, wd = per_layer[layer]
                slots[b] = (F.silu(h @ wg.t()) * (h @ wu.t())) @ wd.t()
            self.runtime.all_reduce(slots)
            return x + slots.float().sum(0).to(x.dtype)
        out = torch.zeros(t, h_dim, dtype=torch.float, device=x.device)
        for per_layer in self._mlp_weights.values():
            wg, wu, wd = per_layer[layer]
            out += ((F.silu(h @ wg.t()) * (h @ wu.t())) @ wd.t()).float()
        out = out.to(x.dtype)
        self.runtime.all_reduce(out)
        return x + out
