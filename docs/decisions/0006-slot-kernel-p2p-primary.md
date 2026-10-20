# 0006: the peer-to-peer slot kernel is the production kernel; host-memory is the fallback

2026-10-04. Status: current.

**Context.** The harness's slot collective moved decode-size tensors through host-mapped memory
because the harness's four-GPU A40 pod had peer-to-peer disabled. Neocloud production hosts are NVLink and NVSwitch
systems where a PCIe bounce on every decode all-reduce costs steady-state latency.

**Decision.** `core/csrc` carries two variants of the slot collective, both reading the
device-visible membership slot on every replay: peer-to-peer device memory for hosts with P2P, and
host-mapped memory for hosts without. Topology selects the variant at startup. Both are phase 1.
No Triton prototype lands in the product tree.

**Consequences.** The 8-GPU H100 run is the first measurement of the P2P variant; the four-GPU A40 pod and CI
exercise the host-memory variant.
