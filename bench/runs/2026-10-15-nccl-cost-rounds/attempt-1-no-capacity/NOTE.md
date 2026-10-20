# Attempt 1 of 3 (after the hung run): no capacity

At 04:29:34Z the run asked for 3 × `gpu-l40s-a` `1gpu-8vcpu-32gb` in eu-north1. The node group
answered `ComputeInstanceOperationFailed`, and mk8s's capacity bound (900 s, `c28dca5`) ended the
attempt at 04:50:17Z (`driver-tail.txt`).

No GPU node was created, so none was billed. The cluster's control plane ran from 04:29:34Z to
04:55:19Z, 26 minutes, and was taken down. The project then listed no clusters and no instances.

Under the cap agreed with the peer, at most three attempts are made, about an hour apart. If all
three find no capacity, the platform question goes to Cort.
