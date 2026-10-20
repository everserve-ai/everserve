# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A model held in host memory once, and a rank's model loaded from it through vLLM's loader
registry: the copy's tensors are views into the buffers, the loader counts what the model took,
and a model the copy does not cover is refused."""

import json
import struct
import sys
import types

import pytest
import torch

from everserve_vllm import resident


def _safetensors(path, tensors, metadata=None):
    """Write ``tensors`` (name -> tensor) as one safetensors file at ``path``."""
    header = {}
    data = b""
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes() if t.numel() else b""
        header[name] = {
            "dtype": {torch.bfloat16: "BF16", torch.float32: "F32", torch.int64: "I64"}[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    if metadata:
        header["__metadata__"] = metadata
    raw_header = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + data)


def test_the_copy_holds_every_file_s_tensors_as_views_into_its_buffers(tmp_path):
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.tensor([1, -2, 3], dtype=torch.int64)
    c = torch.full((4,), 0.5, dtype=torch.bfloat16)
    _safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": a, "b": b}, {"format": "pt"})
    _safetensors(tmp_path / "model-00002-of-00002.safetensors", {"c": c})
    copy = resident.HostCopy.hold(tmp_path, pin=False)
    assert sorted(copy.entries) == ["a", "b", "c"] and copy.nbytes == 24 + 24 + 8
    assert torch.equal(copy.tensor("a"), a) and torch.equal(copy.tensor("b"), b)
    assert torch.equal(copy.tensor("c"), c) and copy.tensor("c").dtype == torch.bfloat16
    assert copy.tensor("a").data_ptr() == copy.buffers[0].data_ptr()  # a view, nothing moved
    assert [n for n, _ in copy.tensors()] == ["a", "b", "c"] and copy.stage_s >= 0


def test_the_copy_refuses_what_it_cannot_hold(tmp_path):
    with pytest.raises(FileNotFoundError, match="no safetensors file"):
        resident.HostCopy.hold(tmp_path, pin=False)
    header = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}  # 2 floats in 4 bytes
    with pytest.raises(ValueError, match="4 bytes for shape"):
        resident.parse_header(json.dumps(header).encode(), 0)
    header = {"x": {"dtype": "Q4", "shape": [2], "data_offsets": [0, 4]}}
    with pytest.raises(ValueError, match="dtype 'Q4'"):
        resident.parse_header(json.dumps(header).encode(), 0)


def test_the_loader_feeds_the_held_copy_to_the_model_and_records_what_it_took(
    tmp_path, monkeypatch
):
    _safetensors(
        tmp_path / "m.safetensors",
        {"w": torch.ones(2, 2, dtype=torch.float32), "v": torch.zeros(3, dtype=torch.bfloat16)},
    )
    monkeypatch.setattr(resident, "_COPY", None)
    monkeypatch.setattr(resident, "_INSTALLED", False)
    registered = {}

    def register_model_loader(load_format):
        def wrap(cls):
            registered[load_format] = cls
            return cls

        return wrap

    class BaseModelLoader:
        def __init__(self, load_config):
            self.load_config = load_config

    loader_mod = types.ModuleType("vllm.model_executor.model_loader")
    loader_mod.register_model_loader = register_model_loader
    base_mod = types.ModuleType("vllm.model_executor.model_loader.base_loader")
    base_mod.BaseModelLoader = BaseModelLoader
    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": loader_mod,
        "vllm.model_executor.model_loader.base_loader": base_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    resident.install()
    resident.install()  # once per process
    loader = registered[resident.LOAD_FORMAT]("load config")
    assert loader.load_config == "load config" and loader.download_model("model config") is None

    class Model:
        def __init__(self, params):
            self.params = params
            self.seen = []

        def named_parameters(self):
            return [(n, None) for n in self.params]

        def load_weights(self, weights):
            for name, t in weights:
                self.seen.append((name, tuple(t.shape), t.dtype))
            return {n for n, _, _ in self.seen} & set(self.params)

    with pytest.raises(RuntimeError, match="no model held in host memory"):
        loader.load_weights(Model(["w"]), None)
    from types import SimpleNamespace

    plain, fp8 = SimpleNamespace(quantization=None), SimpleNamespace(quantization="fp8")
    resident.hold(tmp_path, pin=False)
    model = Model(["w", "v"])
    loader.load_weights(model, plain)
    assert model.seen == [("w", (2, 2), torch.float32), ("v", (3,), torch.bfloat16)]
    assert resident.last_load()["bytes"] == 16 + 6 and resident.last_load()["tensors"] == 2
    with pytest.raises(ValueError, match="1 parameters not in the host copy: \\['u'\\]"):
        loader.load_weights(Model(["w", "u"]), plain)
    # a quantized model makes parameters (scales) the checkpoint lacks: vLLM checks none there
    loader.load_weights(Model(["w", "w_scale"]), fp8)


def _held_bytes(copy):
    """The bytes of host memory the copy's buffers occupy, as allocated."""
    return sum(b.untyped_storage().nbytes() for b in copy.buffers)


def test_the_copy_holds_each_file_s_tensor_bytes_and_no_more(tmp_path):
    """torch's pinned allocator holds 124.6 GB for a 72.7 GB model, each file in a power-of-two
    block."""
    _safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": torch.zeros(3)})
    _safetensors(
        tmp_path / "model-00002-of-00002.safetensors", {"b": torch.zeros(5, dtype=torch.int64)}
    )
    copy = resident.HostCopy.hold(tmp_path, pin=False)
    assert copy.nbytes == 12 + 40 and _held_bytes(copy) == copy.nbytes
    assert not copy.pinned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="page-locking needs a CUDA device")
def test_the_pinned_copy_is_locked_in_place_at_its_own_size_and_released(tmp_path):
    """Torch's pinned allocator would hold 1 MiB for this 1,000,012-byte file."""
    _safetensors(tmp_path / "model.safetensors", {"a": torch.zeros(250_003)})
    copy = resident.HostCopy.hold(tmp_path)
    assert copy.pinned and all(b.is_pinned() for b in copy.buffers)
    assert _held_bytes(copy) == copy.nbytes == 1_000_012
    on_device = copy.tensor("a").to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(on_device.cpu(), torch.zeros(250_003))
    copy.release()
    assert not copy.pinned and not any(b.is_pinned() for b in copy.buffers)
    copy.release()  # a second release has nothing to do
