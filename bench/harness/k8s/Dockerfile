# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The Kubernetes test's probe image: the core package, CPU torch, and stage_probe.py. A node pod of
# an instance without its engine (bench/harness/k8s/stage_probe.py). Built from the repository's
# root by bench/harness/build_images.sh.

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS wheels
RUN pip install --no-cache-dir uv==0.11.26
WORKDIR /src
COPY pyproject.toml uv.lock LICENSE ./
COPY core core
COPY adapters/vllm adapters/vllm
COPY integrations/dynamo integrations/dynamo
RUN uv build --package everserve --wheel --out-dir /wheels

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
COPY --from=wheels /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/everserve-*.whl && rm -rf /tmp/wheels
COPY bench/harness/k8s/stage_probe.py /probe/stage_probe.py
RUN python -c "import everserve.contract.daemon, everserve.podcontract"
USER 1000
