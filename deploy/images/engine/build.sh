#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Build the engine's image on an amd64 Docker host: first NVIDIA Dynamo's vLLM runtime from
# Dynamo's own sources at DYNAMO_PIN, with Dynamo's own renderer, then this directory's Dockerfile
# over it. Run from the repository's root.
#
#   deploy/images/engine/build.sh --tag <registry>/everserve-engine:<tag> [--push]
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
tag=""
push=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) tag="$2"; shift 2 ;;
    --push) push=1; shift ;;
    *) echo "build.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$tag" ] || { echo "build.sh: --tag <registry>/everserve-engine:<tag>" >&2; exit 2; }

pin="$(grep -v '^#' "$here/DYNAMO_PIN" | tr -d '[:space:]')"
# Dynamo's runtime is kept beside the engine image in the same registry, so a build at the same pin
# takes it from there rather than compiling Dynamo again
dynamo_image="${tag%/*}/dynamo-vllm-runtime:${pin:0:12}"
work="${EVERSERVE_BUILD_DIR:-$HOME/.cache/everserve-build}/dynamo-${pin:0:12}"

if ! docker image inspect "$dynamo_image" >/dev/null 2>&1 && ! docker pull -q "$dynamo_image" 2>/dev/null; then
  # Dynamo's sources at the pinned commit, and nothing else: the commit is checked, not trusted
  if [ ! -d "$work/.git" ]; then
    mkdir -p "$work"
    git -C "$work" init -q
    git -C "$work" remote add origin https://github.com/ai-dynamo/dynamo.git
  fi
  git -C "$work" fetch -q --depth 1 origin "$pin"
  git -C "$work" checkout -q --force FETCH_HEAD
  [ "$(git -C "$work" rev-parse HEAD)" = "$pin" ] || { echo "build.sh: not $pin" >&2; exit 1; }
  # Dynamo's renderer needs its own two packages, kept out of the host's Python
  python3 -m venv "$work/.render"
  "$work/.render/bin/pip" install -q pyyaml jinja2
  (cd "$work" && "$work/.render/bin/python" container/render.py \
    --framework vllm --target runtime --cuda-version 13.0 --platform linux/amd64 \
    --output-short-filename)
  (cd "$work" && DOCKER_BUILDKIT=1 docker build -t "$dynamo_image" -f container/rendered.Dockerfile .)
  if [ "$push" = 1 ]; then
    docker push "$dynamo_image"
  fi
fi

DOCKER_BUILDKIT=1 docker build \
  --build-arg "DYNAMO_IMAGE=$dynamo_image" \
  -f "$here/Dockerfile" -t "$tag" "$root"
echo "built $tag on $dynamo_image (Dynamo $pin)"
if [ "$push" = 1 ]; then
  docker push "$tag"
fi
