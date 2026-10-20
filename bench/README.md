# bench

Drills and measurements. Every number Everserve states comes from here.

- `runs/`: dated. A pre-registered run has `PREREGISTRATION.md` committed before it runs, then
  `results.json` and `RESULT.md` committed before any number is written in prose anywhere; an
  exploratory run says so at the top of its `RESULT.md`. A record keeps the evidence and not the
  dumps (`runs/README.md`, the rule in `harness/record.py`); raw output stays with whoever ran it.
- `harness/`: the drivers that run them: the Kubernetes and Dynamo proof driver, the VM drills,
  the image builds and the cloud helpers (`harness/README.md`).
- `drills/`: the failure modes, each proven by a run under `runs/` (`drills/README.md`).

Rules: state GPU spend before a run and get it approved; commit predictions before results;
a drill that tests something next to its claim is worse than none.
