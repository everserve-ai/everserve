# Attempt 4: no capacity, on Cort's word to keep waiting for L40S

After attempt 3, Cort chose to keep waiting for L40S, so that every probe stays on the same
hardware. Attempt 4 started on his word at 10:41Z, from `sg2-python` at `07d5b0f`. The probe is
unchanged since `d307c01`.

At 10:41:53Z the run asked for 3 × `gpu-l40s-a` `1gpu-8vcpu-32gb` in eu-north1. The node group got
no capacity, and mk8s's 900 s bound ended the attempt at 11:02:21Z (`driver-tail.txt`).

No GPU node was created, so none was billed. The cluster's control plane ran from 10:41:53Z to
11:14:01Z, 32 minutes, and was taken down. The project then listed no clusters and no instances.

**Running total of the failed attempts:** 117 minutes of control plane (26 + 33 + 26 + 32), and no
GPU.
