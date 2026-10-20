# Attempt 2 of 3: no capacity

The run went from `sg2-python` at `d307c01`. Between the pre-registration's last commit
(`7044217`) and `d307c01`, one commit touches the probe's harness: `fcfa877`, which raises the
per-process bound in `dynamo_proof.sh` from 180 s to 300 s. `nccl_probe.py` is unchanged.

At 05:47:53Z the run asked for 3 × `gpu-l40s-a` `1gpu-8vcpu-32gb` in eu-north1. The node group got
no capacity, and mk8s's 900 s bound ended the attempt at 06:08:37Z (`driver-tail.txt`).

No GPU node was created, so none was billed. The cluster's control plane ran from 05:47:53Z to
06:21:12Z, 33 minutes, and was taken down. The project then listed no clusters and no instances.
