# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# The control plane's image: everserve-controller (the coordinator and the node daemon) and
# everserve-store. Base images are pinned by digest, the toolchain to rust-toolchain.toml's.
# Build: docker build -t everserve:dev .

FROM rust:1.97.1-slim-bookworm@sha256:2775a09d208ff0d7c1f50490c45b62db929e87ba1dcbc3f2132ac71a704bcdd3 AS build
# etcd-client generates etcd's gRPC types with protoc, which needs the well-known types
# (google/protobuf/*.proto) Debian ships apart, in libprotobuf-dev
RUN apt-get update \
 && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY proto proto
COPY wire wire
COPY telemetry telemetry
COPY controller controller
COPY store store
RUN cargo build --release --locked -p everserve-controller -p everserve-store

# no shell, no package manager; runs as uid 65532
FROM gcr.io/distroless/cc-debian12:nonroot@sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f
COPY --from=build /src/target/release/everserve-controller /src/target/release/everserve-store /usr/local/bin/
# by number, not name: a kubelet verifies runAsNonRoot only against a numeric user
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/everserve-controller"]
