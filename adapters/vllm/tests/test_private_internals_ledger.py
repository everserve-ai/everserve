# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""``PRIVATE-INTERNALS.md`` names every reach of the adapter into vLLM, and its anchors are live.

The ledger is the adapter's promise (``CONTRIBUTING.md`` rule 1): an adapter reaches an engine only
through documented surfaces or through lines it lists there. A record nobody checks drifts, and
this one did: before this test, rows cited adapter line numbers that pointed at other code. So the
ledger anchors functions, not lines, and this test parses the ledger and the adapter's source,
never vLLM (which is not installed where it runs), and fails on:

1. an adapter anchor ``file.py:symbol`` whose file is not under ``everserve_vllm`` or whose symbol
   is not a top-level function, class or module constant there, or a ``Class.method`` defined in
   that class's body. Resolved with ``ast``, so a stale anchor fails the row that carries it:
   rename ``serving.install`` to ``serving.put`` and A47's anchor ``serving.py:install`` no longer
   resolves, and the assertion names it;
2. a vLLM module the adapter imports (``import vllm.x``, ``from vllm.x import y``) that no pin
   anchor names as a path ``vllm/x.py`` or ``vllm/x/__init__.py``, except the public surface in
   :data:`PUBLIC` and :data:`PUBLIC_NAMES`; a ``torch._...`` import must be named as written;
3. a private attribute reach the regexes of :data:`REACHES` find in the source (``ps._TP``,
   ``._inflight_prefills``, ``sys._getframe`` and the rest) that the ledger does not name as a
   whole word;
4. a function that assigns onto a vLLM or torch object (``setattr(...)``, ``x.y = ...`` rooted at a
   name bound from a vLLM import or at ``torch.distributed``, or an instance write of one of the
   names in :data:`INSTANCE_WRITES` not rooted at ``self``) that the Patches section does not
   anchor, by itself or by its class.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[1] / "everserve_vllm"
LEDGER = Path(__file__).resolve().parents[1] / "PRIVATE-INTERNALS.md"

PUBLIC = {
    "vllm.config",  # set_current_vllm_config, the config context
    "vllm.distributed.kv_transfer.kv_connector.v1.base",  # the connector API
}
"""vLLM modules (and their submodules) that are its public surface; a reach needs no row."""

PUBLIC_NAMES = {"SamplingParams"}
"""Names ``from vllm import ...`` binds that are the package's API, not a submodule."""

REACHES = (
    "_abort_process_group",
    "_all_gather_out_place",
    "_all_reduce_out_place",
    "_aot_cache_dir",
    "_attention_ops",
    "_dummy_run",
    "_getframe",
    "_groups",
    "_inflight_prefills",
    "_init_message_queues",
    "_process_engine_step",
    "_reduce_scatter_out_place",
    "__commit_id__",
    "adaptive_long_prefill_threshold",
    "add_requests",
    "all_token_ids",
    "batch_queue",
    "broadcast_object",
    "capture_model",
    "compile_or_warm_up_model",
    "copy_to_uva",
    "create_from_handle",
    "cudagraph_manager",
    "death_writer",
    "decode_query_len",
    "device_communicator",
    "direct_register_custom_op",
    "drop_stale_output",
    "export_handle",
    "flashinfer_autotune",
    "for_sampler_warmup",
    "from_unique_id_bytes",
    "futures_queue",
    "get_block_ids",
    "get_class",
    "get_metadata",
    "get_pp_group",
    "global_rank",
    "group_ranks",
    "host_resident",
    "init_worker_distributed_environment",
    "intermediate_tensors",
    "is_failed",
    "kernel_warmup",
    "kv_block_zeroer",
    "kv_cache_dtype_str_to_dtype",
    "kv_cache_tensors",
    "last_sampled_tokens",
    "load_model",
    "local_cache_dir",
    "local_socket",
    "lock_workspace",
    "long_prefill_token_threshold",
    "max_chunks",
    "max_num_scheduled_tokens",
    "make_sibling_device_group",
    "mq_broadcaster",
    "ncclCommAbort",
    "ncclGetUniqueId",
    "ncclGetVersion",
    "num_actual_tokens",
    "num_in_flight_tokens",
    "num_output_placeholders",
    "num_scheduled_tokens",
    "num_stale_output_tokens",
    "num_tokens_with_spec",
    "peer_response_handles",
    "preempted_req_ids",
    "prefill_len",
    "pynccl_comm",
    "randomize_dummy_inputs",
    "rank_in_group",
    "remote_socket",
    "req_id_to_index",
    "resolve_flashinfer_autotune_file",
    "response_mqs",
    "ResponseStatus",
    "rpc_broadcast_mq",
    "scheduled_cached_reqs",
    "scheduled_new_reqs",
    "shutting_down",
    "slot_mapping",
    "spec_token_ids",
    "static_forward_context",
    "step_fn",
    "ubatch_runner",
    "unique_name",
    "unlock_workspace",
    "use_custom_op_call",
    "use_device_communicator",
    "VLLM_MQ_MAX_CHUNK_BYTES_MB",
    "wait_until_ready",
    "warmup_kernels",
    "worker_busy_loop",
    "worker_response_mq",
    "zero_block_ids",
)
"""vLLM attributes, methods and settings the adapter reaches as ``.name``, ``"name"`` or
``'name'``; each must be named in the ledger wherever the source reaches it."""

INSTANCE_WRITES = (
    "_all_gather_out_place",
    "_all_reduce_out_place",
    "_dummy_run",
    "_reduce_scatter_out_place",
    "all_reduce",
    "barrier",
    "broadcast",
    "broadcast_object",
    "capture_model",
    "death_writer",
    "device_communicator",
    "execute_model",
    "flashinfer_autotune",
    "for_sampler_warmup",
    "get_class",
    "group_ranks",
    "init_worker_distributed_environment",
    "kernel_warmup",
    "load_format",
    "load_model",
    "pynccl_comm",
    "rpc_broadcast_mq",
    "sample_tokens",
    "schedule",
    "step_fn",
    "unique_name",
    "use_custom_op_call",
    "warmup_kernels",
    "worker_response_mq",
)
"""Attributes of vLLM objects the adapter writes on instances it reached without a vLLM-bound
name in scope (a group off ``ps._TP``, a runner, a ``WorkerProc``); a write of one is a patch."""

ANCHOR = re.compile(
    r"`((?:models/)?[a-z_]+\.py):([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)`"
)
"""An adapter anchor as the ledger writes it: `file.py:symbol` or `file.py:Class.method`."""

GLOBAL_REACH = re.compile(r"(?:\bps|parallel_state)\.(_[A-Za-z_]+)|[\"'](_[A-Z][A-Z_]*)[\"']")
"""A module global of ``parallel_state`` reached by name, or named as a string for ``getattr``."""


# ----------------------------------------------------------------------------- the ledger
@dataclass(frozen=True)
class Row:
    section: str
    id: str
    cells: tuple[str, ...]

    @property
    def adapter_anchors(self) -> list[tuple[str, str]]:
        return [(m.group(1), m.group(2)) for m in ANCHOR.finditer(self.cells[1])]


def ledger_rows(text: str) -> list[Row]:
    """Every table row whose first cell is a row id, with the ``##`` section it is under."""
    rows: list[Row] = []
    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("| A"):
            continue
        cells = tuple(c.strip() for c in line.strip().strip("|").split("|"))
        rows.append(Row(section, cells[0], cells))
    return rows


def whole_word(name: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


# ----------------------------------------------------------------------------- the adapter
def adapter_files() -> list[Path]:
    return sorted(p for p in ADAPTER.rglob("*.py") if "__pycache__" not in p.parts)


def anchor_path(path: Path) -> str:
    """The anchor form of an adapter file: ``file.py``, or ``models/__init__.py``."""
    return path.relative_to(ADAPTER).as_posix()


def symbols(tree: ast.Module) -> set[str]:
    """Top-level functions, classes and constants, and ``Class.member`` for each class body."""
    out: set[str] = set()

    def names_of(target: ast.expr) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Tuple):
            return [n for t in target.elts for n in names_of(t)]
        return []

    def bound(node: ast.stmt) -> list[str]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [node.name]
        if isinstance(node, ast.Assign):
            return [n for t in node.targets for n in names_of(t)]
        if isinstance(node, ast.AnnAssign):
            return names_of(node.target)
        return []

    for node in tree.body:
        for name in bound(node):
            out.add(name)
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                for name in bound(member):
                    out.add(f"{node.name}.{name}")
    return out


def imported_vllm_modules(tree: ast.Module) -> list[tuple[int, list[str]]]:
    """Each vLLM import with the module paths that may name it, as ``(line, [paths])``.

    ``from vllm.a import b`` is ``vllm/a.py``, ``vllm/a/__init__.py`` or, ``b`` being a
    submodule, ``vllm/a/b.py``; one of them named in the ledger is enough.
    """
    out: list[tuple[int, list[str]]] = []

    def public(module: str) -> bool:
        return any(module == p or module.startswith(p + ".") for p in PUBLIC)

    def paths(module: str) -> list[str]:
        base = module.replace(".", "/")
        return [f"{base}.py", f"{base}/__init__.py"]

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] != "vllm" or public(node.module):
                continue
            for alias in node.names:
                if node.module == "vllm":
                    if alias.name in PUBLIC_NAMES:
                        continue
                    out.append((node.lineno, paths(f"vllm.{alias.name}")))
                else:
                    out.append(
                        (node.lineno, paths(node.module) + paths(f"{node.module}.{alias.name}"))
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "vllm" and alias.name != "vllm":
                    if not public(alias.name):
                        out.append((node.lineno, paths(alias.name)))
    return out


def imported_torch_private(tree: ast.Module) -> list[tuple[int, str]]:
    """Each ``torch._...`` module imported, by line."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("torch._"):
            out.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            out.extend((node.lineno, a.name) for a in node.names if a.name.startswith("torch._"))
    return out


def vllm_bound_names(tree: ast.Module) -> set[str]:
    """Names a vLLM import binds anywhere in the module, plus ``dist`` for torch.distributed."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "vllm":
            out.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "vllm":
                    out.add(a.asname or a.name.split(".")[0])
                if a.name == "torch.distributed" and a.asname:
                    out.add(a.asname)
    return out


def chain(node: ast.expr) -> list[str] | None:
    """``a.b.c`` or ``a.b[c]`` as ``["a", "b", "c"]`` (a subscript as its value's chain)."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        inner = chain(node.value)
        return None if inner is None else inner + [node.attr]
    if isinstance(node, ast.Subscript):
        return chain(node.value)
    return None


def patch_sites(tree: ast.Module, bound: set[str]) -> list[tuple[int, str, str]]:
    """Every assignment onto a vLLM or torch object: ``(line, enclosing symbol, what)``.

    The enclosing symbol is the top-level function or class, or ``Class.method`` for a method.
    """
    sites: list[tuple[int, str, str]] = []

    def targets(node: ast.stmt) -> list[ast.expr]:
        if isinstance(node, ast.Assign):
            out: list[ast.expr] = []
            for t in node.targets:
                out.extend(t.elts if isinstance(t, ast.Tuple) else [t])
            return out
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            return [node.target]
        return []

    def writes_vllm(target: ast.expr) -> str | None:
        c = chain(target)
        if c is None or len(c) < 2:
            return None
        dotted = ".".join(c)
        if c[0] in bound or c[:2] == ["torch", "distributed"]:
            return dotted
        if c[0] != "self" and c[-1] in INSTANCE_WRITES:
            return dotted
        return None

    def scan(body: list[ast.stmt], where: str) -> None:
        for node in body:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "setattr"
                ):
                    sites.append((sub.lineno, where, ast.unparse(sub)[:60]))
                elif isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for t in targets(sub):
                        what = writes_vllm(t)
                        if what is not None:
                            sites.append((sub.lineno, where, f"{what} ="))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan([node], node.name)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scan([member], f"{node.name}.{member.name}")
                else:
                    scan([member], node.name)
    return sites


def covered(file: str, where: str, anchors: set[tuple[str, str]]) -> bool:
    """Whether ``where`` in ``file`` is anchored, by itself or by its class."""
    if (file, where) in anchors:
        return True
    cls = where.split(".")[0]
    return "." in where and (file, cls) in anchors


# ----------------------------------------------------------------------------- the tests
def test_every_adapter_anchor_resolves():
    rows = ledger_rows(LEDGER.read_text())
    assert rows, "no rows read from the ledger"
    index = {
        anchor_path(p): symbols(ast.parse(p.read_text(), filename=str(p))) for p in adapter_files()
    }
    stale = []
    for row in rows:
        assert row.adapter_anchors, f"{row.id} ({row.section}) names no adapter anchor"
        for file, symbol in row.adapter_anchors:
            if file not in index:
                stale.append(f"{row.id}: `{file}:{symbol}`: no {file} under everserve_vllm")
            elif symbol not in index[file]:
                stale.append(f"{row.id}: `{file}:{symbol}`: {file} defines no {symbol}")
    assert not stale, "PRIVATE-INTERNALS.md anchors that no longer resolve:\n" + "\n".join(stale)


def test_every_imported_vllm_module_has_a_pin_anchor():
    ledger = LEDGER.read_text()
    missing = []
    for path in adapter_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for line, candidates in imported_vllm_modules(tree):
            if not any(c in ledger for c in candidates):
                missing.append(f"{anchor_path(path)}:{line}: one of {candidates}")
        for line, module in imported_torch_private(tree):
            if module not in ledger:
                missing.append(f"{anchor_path(path)}:{line}: {module}")
    assert not missing, (
        "modules the adapter imports with no pin anchor in PRIVATE-INTERNALS.md:\n"
        + "\n".join(missing)
    )


def test_every_private_attribute_reach_is_named():
    ledger = LEDGER.read_text()
    missing = []
    for path in adapter_files():
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for name in REACHES:
                if re.search(rf"(?:\.|[\"']){re.escape(name)}(?![A-Za-z0-9_])", line):
                    if not whole_word(name, ledger):
                        missing.append(f"{anchor_path(path)}:{n}: {name}")
            for m in GLOBAL_REACH.finditer(line):
                name = m.group(1) or m.group(2)
                if not whole_word(name, ledger):
                    missing.append(f"{anchor_path(path)}:{n}: parallel_state.{name}")
    assert not missing, (
        "private vLLM names the adapter reaches that PRIVATE-INTERNALS.md does not name:\n"
        + "\n".join(missing)
    )


def test_every_patch_is_in_the_patches_section():
    rows = ledger_rows(LEDGER.read_text())
    patch_rows = [r for r in rows if r.section == "Patches"]
    assert patch_rows, "the ledger has no Patches section"
    anchors = {a for r in patch_rows for a in r.adapter_anchors}
    unlisted = []
    for path in adapter_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        file = anchor_path(path)
        for line, where, what in patch_sites(tree, vllm_bound_names(tree)):
            if not covered(file, where, anchors):
                unlisted.append(f"{file}:{line} in {where}: {what}")
    assert not unlisted, (
        "assignments onto vLLM or torch objects whose function the Patches section of "
        "PRIVATE-INTERNALS.md does not anchor:\n" + "\n".join(unlisted)
    )
