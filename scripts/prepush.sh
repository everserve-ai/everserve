#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
# Push gate. Preflight with the slow tier, plus the doc build with warnings denied.
# Push only after this prints "prepush: green"; read it, then push in a separate command.
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/preflight.sh all --slow
RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps --quiet
echo "prepush: green"
