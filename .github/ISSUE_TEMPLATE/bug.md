---
name: Bug
about: Something behaves other than it says it does
labels: bug
---

**What happened**

**What you expected** (quote the documentation or decision record that says so, if one does)

**Where**: the component (control plane, store, vLLM adapter, Dynamo integration, chart), the
version or commit, the vLLM version (`adapters/vllm/PIN` is the one supported), the platform.

**How to reproduce**: commands, policy document, logs. A coordinator's `GET /v1/engines/{engine}`
and `.../events` answers say most of what a change did; attach them for anything involving a
change.
