# Attempt 3 of 3: no capacity, so the platform question goes to Cort

The run went from `sg2-python` at `656bf43`. The probe is unchanged since `d307c01`; only records
were added.

At 07:11:50Z the run asked for 3 × `gpu-l40s-a` `1gpu-8vcpu-32gb` in eu-north1. The node group got
no capacity, and mk8s's 900 s bound ended the attempt at 07:32:42Z (`driver-tail.txt`).

No GPU node was created, so none was billed. The cluster's control plane ran from 07:11:50Z to
07:37:41Z, 26 minutes, and was taken down. The project then listed no clusters and no instances.

All three attempts found no L40S capacity in eu-north1, between 04:29Z and 07:32Z:

| Attempt | Control plane | GPU billed |
|---|---|---|
| 1 | 04:29:34Z to 04:55:19Z, 26 min | none |
| 2 | 05:47:53Z to 06:21:12Z, 33 min | none |
| 3 | 07:11:50Z to 07:37:41Z, 26 min | none |

Under the cap agreed with the peer, no further attempt is made until Cort decides the platform.
