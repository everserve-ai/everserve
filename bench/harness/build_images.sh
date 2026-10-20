#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Build and push the project's images on a rented Nebius CPU VM, then release the VM: the engine's
# (deploy/images/engine), the control plane's (the repository's Dockerfile) and the Kubernetes
# test's probe (bench/harness/k8s). The images are linux/amd64, Dynamo's runtime is compiled from
# source, and a laptop is neither. The VM is released when the build ends, whatever its outcome, and
# a watchdog releases it at --hours in any case. Name any of the three.
#
# The project must be in a region that allows CPU VMs: a tenant's GPU regions may allow none (its
# quotas: nebius quotas quota-allowance list --parent-id <tenant>, compute.instance.non-gpu.vcpu).
# The registry may be in another region.
#
#   EVERSERVE_NEBIUS_PROJECT=<project> bench/harness/build_images.sh \
#       [--engine <registry>/everserve-engine:<tag>] [--control <registry>/everserve:<tag>] \
#       [--probe <registry>/everserve-probe:<tag>] [--preset 32vcpu-128gb] [--hours 3] [--log <file>]
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
py="$root/.venv/bin/python"  # the harness needs Python 3.10 or later: the workspace's
neb="$py $here/nebius.py"
engine=""
control=""
probe=""
preset="32vcpu-128gb"
hours=3
logf="${TMPDIR:-/tmp}/build-images-$(date -u +%Y%m%dT%H%M%SZ).log"
while [ $# -gt 0 ]; do
  case "$1" in
    --engine) engine="$2"; shift 2 ;;
    --control) control="$2"; shift 2 ;;
    --probe) probe="$2"; shift 2 ;;
    --preset) preset="$2"; shift 2 ;;
    --hours) hours="$2"; shift 2 ;;
    --log) logf="$2"; shift 2 ;;
    *) echo "build_images.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
any="${engine:-${control:-$probe}}"
[ -n "$any" ] || { echo "build_images.sh: name --engine, --control or --probe" >&2; exit 2; }
registry_host="${any%%/*}"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$logf"; }

# an image is a commit: built from `git archive` of HEAD, which stamps the commit into the tree
# (export-subst, adapters/vllm/everserve_vllm/_source.py), so every process the image runs says
# what it runs (decision 0034). A tree with changes is refused, before anything is rented: an image
# of uncommitted code could not say what it is, and the coordinator refuses its spares for that.
paths=(pyproject.toml uv.lock LICENSE .dockerignore Dockerfile Cargo.toml Cargo.lock
  rust-toolchain.toml proto wire telemetry controller store core adapters/vllm integrations/dynamo
  deploy/images/engine bench/harness/k8s
  # the adapter's tests read these, run against the engine image (the images copy neither)
  docs/examples bench/harness/policies)
if [ -n "$(git -C "$root" status --porcelain --untracked-files=no -- "${paths[@]}")" ]; then
  echo "build_images.sh: the tree has uncommitted changes under the images' paths; commit them" >&2
  exit 2
fi
commit="$(git -C "$root" rev-parse HEAD)"

name="everserve-build-$(date -u +%H%M%S)"
id=""
# released on any exit, the create's own failure included: a VM whose create answer was lost is
# found by its name
#
# A release is tried again while it fails (nebius.py's release), and a VM still standing after that
# fails the build loudly: a release lost to a DNS timeout would otherwise leave the build VM running
# while the build exits 0, for the watchdog to end hours later and silently.
release() {
  local failed=0 i
  [ -n "$id" ] || id="$($neb list | "$py" -c "
import json, sys
print(' '.join(v['id'] for v in json.load(sys.stdin) if v['name'] == sys.argv[1]))" "$name")"
  for i in $id; do
    say "releasing $i"
    if ! $neb terminate "$i" --log "$logf" >/dev/null; then
      say "the build VM $i is still up: delete it by hand (its watchdog ends it at the latest)"
      failed=1
    fi
  done
  [ "$failed" = 0 ] || exit 1
}
trap release EXIT
trap 'exit 130' INT TERM  # a stopped driver exits, and so releases, rather than dying with its VM up
say "renting a build VM: cpu-d3 $preset as $name, released at the latest after $hours h"
refused="$(mktemp)"
if ! vm="$($neb deploy --name "$name" --platform cpu-d3 --preset "$preset" --disk 400 --log "$logf" \
    2>"$refused")"; then
  say "the VM was refused: $(grep -E 'Error|error' "$refused" | tail -1)"
  if grep -E "non-gpu.vcpu" "$refused" >/dev/null; then
    say "this project's region allows no CPU VMs: name a project in a region that does"
  fi
  exit 1
fi
id="$(echo "$vm" | "$py" -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
ip="$(echo "$vm" | "$py" -c 'import json,sys; print(json.load(sys.stdin)["public_ip"])')"
# the watchdog detaches into its own session: nothing that stops this script stops it
$neb watchdog "$id" --at "$("$py" -c "import time; print(time.time() + $hours * 3600)")" \
  --log "$logf" >/dev/null 2>&1
say "VM $id at $ip"

known="$(mktemp)"  # Nebius reuses public addresses: a host key from an earlier VM is not this one's
key="${EVERSERVE_SSH_KEY:-$HOME/.ssh/everserve}"  # the harness's key, which nebius.py deploy installs
ssh_() { ssh -i "$key" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$known" \
  -o ConnectTimeout=10 -o ServerAliveInterval=30 "everserve@$ip" "$@"; }
for _ in $(seq 60); do ssh_ true 2>/dev/null && break; sleep 5; done

say "docker on the VM"
# the image may carry Docker's own packages and their apt source already; Ubuntu's docker.io
# conflicts with them, so Docker's are what is installed where none is
ssh_ 'set -e; sudo apt-get update -q
pkgs="git python3-venv"
command -v docker >/dev/null || pkgs="$pkgs docker-ce docker-ce-cli containerd.io docker-buildx-plugin"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq $pkgs >/dev/null
sudo usermod -aG docker everserve' >>"$logf" 2>&1

say "the tree the images build from: commit $commit, as git archive stamps it"
git -C "$root" archive --format=tar.gz "$commit" -- "${paths[@]}" \
  | ssh_ 'mkdir -p /workspace/everserve && tar -C /workspace/everserve -xz' >>"$logf" 2>&1

login() {  # a short-lived token of the harness's account, never a key; it expires, so a push
  # after a long build logs in again first
  "${NEBIUS_BIN:-nebius}" --profile "${NEBIUS_PROFILE:-everserve}" iam get-access-token \
    | ssh_ "docker login $registry_host --username iam --password-stdin" >>"$logf" 2>&1
}
say "the registry login, with a short-lived token of the harness's account, never a key"
login

say "the registry takes a push before the builds are spent on one it would refuse"
check="${any%:*}:push-probe"
ssh_ "docker pull -q busybox:1.37 && docker tag busybox:1.37 $check && docker push -q $check" \
  >>"$logf" 2>&1 || { say "the registry refused $check; see $logf"; exit 1; }

# A push the registry refuses is tried again, with a fresh login, a bounded number of times: the
# build it follows takes most of an hour, and a refusal for authentication (the engine's push after
# the control plane's) is the token, not the image
PUSH_TRIES=3
push() {  # the image
  local try
  for try in $(seq 1 "$PUSH_TRIES"); do
    if ssh_ "docker push -q $1" >>"$logf" 2>&1; then
      say "pushed $1"
      return 0
    fi
    say "the registry refused $1 (try $try of $PUSH_TRIES)"
    [ "$try" -lt "$PUSH_TRIES" ] || break
    sleep $((10 * try))
    login
  done
  say "the push of $1 failed; its log is $logf"
  exit 1
}
build() {  # what, the command on the VM
  say "building $1"
  if ! ssh_ "cd /workspace/everserve && $2" >>"$logf" 2>&1; then
    say "the build of $1 failed; its log is $logf"
    exit 1
  fi
}
if [ -n "$control" ]; then
  build "$control" "DOCKER_BUILDKIT=1 docker build -t $control ."
  push "$control"
fi
if [ -n "$probe" ]; then
  build "$probe" "DOCKER_BUILDKIT=1 docker build -f bench/harness/k8s/Dockerfile -t $probe ."
  push "$probe"
fi
if [ -n "$engine" ]; then
  build "$engine" "deploy/images/engine/build.sh --tag $engine"
  # the adapter's tests against the vLLM the image carries, before it is pushed: the tests that
  # need vLLM are skipped anywhere else short of a GPU rental, and an image they fail is not one
  # to run (the tree is the commit the image is, mounted read-only; pytest in a throwaway layer)
  build "$engine's adapter tests" "docker run --rm --entrypoint /bin/sh \
    -v /workspace/everserve:/src:ro $engine -c 'cp -r /src /tmp/r && cd /tmp/r \
    && python3 -m pip install -q --target /tmp/pytest pytest jsonschema \
    && PYTHONPATH=/tmp/pytest python3 -m pytest -q -p no:cacheprovider -o faulthandler_timeout=300 \
       adapters/vllm/tests -m \"not slow\"'"
  # Dynamo's runtime beside it, as build.sh names it, so the next build at the pin takes it from
  # the registry rather than compiling Dynamo again; a push of what the registry holds is a no-op
  pin="$(grep -v '^#' "$root/deploy/images/engine/DYNAMO_PIN" | tr -d '[:space:]')"
  push "${engine%/*}/dynamo-vllm-runtime:${pin:0:12}"
  push "$engine"
fi
