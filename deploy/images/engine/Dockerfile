# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The engine's image: NVIDIA Dynamo's vLLM runtime with Everserve's adapter and its Dynamo
# integration, and the Everserve kernels built ahead of time for the runtime's torch. Every pod of
# a rendered instance runs it (integrations/dynamo). Built by build.sh beside this file, which
# builds DYNAMO_IMAGE first from Dynamo's own sources at DYNAMO_PIN; see README.md.

ARG DYNAMO_IMAGE

# the three packages, pure Python, as wheels
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS wheels
RUN pip install --no-cache-dir uv==0.11.26
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY core core
COPY adapters/vllm adapters/vllm
COPY integrations/dynamo integrations/dynamo
COPY LICENSE ./
RUN uv build --all-packages --wheel --out-dir /wheels

# the kernels, built against the runtime's own torch with the CUDA toolkit alone: no GPU here
FROM ${DYNAMO_IMAGE} AS kernels
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;10.0"
# the runtime image ends as its user (dynamo); what is installed here is installed as root
USER root
# the toolkit of the runtime's CUDA version, in this stage only and never in the image that runs:
# the compiler and every header torch's extension headers include (cuSPARSE, cuBLAS, cuSOLVER,
# cuRAND), whatever part of it the runtime image happens to carry (Dynamo's carries nvcc, not
# the libraries' headers)
RUN set -eu; \
    v="$(python3 -c 'import torch; print(torch.version.cuda.replace(".", "-"))')"; \
    apt-get update; \
    apt-get install -y --no-install-recommends "cuda-nvcc-$v" "cuda-cudart-dev-$v" \
      "cuda-cccl-$v" "libcublas-dev-$v" "libcusparse-dev-$v" "libcusolver-dev-$v" \
      "libcurand-dev-$v" g++; \
    rm -rf /var/lib/apt/lists/*
COPY --from=wheels /wheels /wheels
RUN python3 -m pip install --no-cache-dir --no-deps /wheels/everserve-*.whl
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" \
    python3 -m everserve.kernels --into /opt/everserve/kernels

FROM ${DYNAMO_IMAGE}
USER root
COPY --from=wheels /wheels /tmp/wheels
# the runtime's torch and vLLM stay as they are: the adapter is pinned to this vLLM, and the
# kernels to this torch
# (their versions read from the installed metadata, which a direct-URL or local-version install
# names as pip freeze does not)
RUN set -eu; \
    python3 -c "from importlib.metadata import version as v; \
print(f'torch=={v(\"torch\")}'); print(f'vllm=={v(\"vllm\")}')" | tee /tmp/held.txt; \
    python3 -c "from importlib.metadata import version as v; import sys; \
sys.exit(0 if v('vllm').split('+')[0] == '0.31.0' else f'vLLM {v(\"vllm\")}, not the pin 0.31.0')"; \
    python3 -m pip install --no-cache-dir -c /tmp/held.txt /tmp/wheels/*.whl; \
    rm -rf /tmp/wheels /tmp/held.txt
COPY --from=kernels /opt/everserve/kernels /opt/everserve/kernels
ENV EVERSERVE_KERNELS_DIR=/opt/everserve/kernels
# what a pod runs is importable, the adapter's flags render, and Dynamo's worker adds nothing to
# vLLM's additional_config that the renderer does not give every process, before the image is pushed
RUN python3 -c "import vllm, everserve_vllm, everserve_dynamo, dynamo.vllm; assert vllm.__version__.split('+')[0] == '0.31.0', vllm.__version__" \
 && python3 -m everserve_vllm.flags --head --guarantee exact > /dev/null \
 && python3 -m everserve_dynamo.vllm \
 && python3 -c "import json, everserve.kernels as k; s = json.load(open('/opt/everserve/kernels/stamp.json')); assert all(s[x] == v for x, v in k.stamp().items()), s"
# the runtime image's own user, as it ended: an engine pod never runs as root
USER dynamo
