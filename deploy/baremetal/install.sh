#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Install the Everserve control plane on one host: the binaries, the systemd units, the
# runtime directories, the example settings. Enables nothing; README.md says what to fill in and
# which units each node starts. Idempotent: existing settings under /etc/everserve are kept.
# Usage: sudo deploy/baremetal/install.sh [binaries-dir]   (default: target/release)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bins="${1:-$here/../../target/release}"
for b in everserve-controller everserve-store; do
  [[ -x "$bins/$b" ]] || { echo "install: $bins/$b is missing; build with cargo build --release" >&2; exit 2; }
done
if ! id everserve >/dev/null 2>&1; then
  useradd --system --home-dir /var/lib/everserve --shell /usr/sbin/nologin everserve
fi
install -m 0755 "$bins/everserve-controller" "$bins/everserve-store" /usr/local/bin/
install -d -m 0755 /etc/everserve /usr/share/doc/everserve
install -m 0644 "$here"/everserve-coord.service "$here"/everserve-node.service "$here"/everserve-store.service /etc/systemd/system/
install -m 0644 "$here/everserve.tmpfiles.conf" /etc/tmpfiles.d/everserve.conf
install -m 0644 "$here/README.md" /usr/share/doc/everserve/README.md
for f in coord node store; do
  [[ -e "/etc/everserve/$f.env" ]] || install -m 0640 -g everserve "$here/$f.env.example" "/etc/everserve/$f.env"
done
systemd-tmpfiles --create /etc/tmpfiles.d/everserve.conf
systemctl daemon-reload
cat <<MSG
installed. Next, on this host:
  1. put the policy at /etc/everserve/policy.json and the certificates under /etc/everserve/certs
     (everserve-controller certs --out /etc/everserve/certs --node <name> ... on one host, then
     copy each node its own .crt/.key and ca.crt);
  2. edit /etc/everserve/{coord,node,store}.env;
  3. systemctl enable --now everserve-store everserve-node   (every node)
     systemctl enable --now everserve-coord                  (the coordinator's host)
MSG
