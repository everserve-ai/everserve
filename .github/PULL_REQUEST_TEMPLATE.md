**What is now true** (the commit subject says it; the body says why).

**Checks**
- [ ] `scripts/prepush.sh` is green.
- [ ] Docs moved with the code: a changed contract, field, flag or boundary updates its document
      in this change, and a new decision has its record.
- [ ] Every commit is signed off (`git commit -s`, the Developer Certificate of Origin).
- [ ] No GPU number is stated without its pre-registered record under `bench/runs/`.
- [ ] Nothing under `vllm/` is edited; a new private vLLM line is a new row in
      `adapters/vllm/PRIVATE-INTERNALS.md`.
