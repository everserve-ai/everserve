# runs

Every number Everserve states comes from a directory here. A run is pre-registered before it
starts and written up after it ends, and the write-up never changes what was predicted.

A record names the commit it ran on by hash. Commits before the repository's publish
(2026-10-20) are in the private archive it was published from (`README.md`, History); the image
tags the records name were built from those commits.

## A run's directory

`YYYY-MM-DD-<name>/`, named for the day it ran:

- `PREREGISTRATION.md`, committed before the run: what runs, on what, what is predicted and
  what makes it pass, what it costs. A run without one is exploratory, and its `RESULT.md` says
  so in its first lines.
- `RESULT.md`, written after: the verdict first, then what ran, where each number was read, what
  was found, and what the run does not show.
- `results.json`, the figures the pre-registration asked for, where the run produced them;
  `predictions.json` where the predictions are machine-read.
- `run/` (or `run-N/` for a run with attempts): what of the run's output a record keeps.

`PLANNED/<name>/` holds a run written up before it has a day: its pre-registration, its
predictions, and the dry runs made while the harness was built (`dryrun-*/`, each with a
`NOTE.md`). It moves to a dated directory when the pre-registered run happens. The three under
it now (`p2-two-node`, `p3-resident`, `p3-resident-run2`) are the two-node and resident
batteries, whose dry runs decisions 0025 to 0032 cite by directory.

## What a record keeps

The rule is `bench/harness/record.py`, and `bench/harness/tests/test_record.py` fails when a
tracked file breaks it:

- kept whatever their size: every `.md`, `results.json`, `predictions.json`, `summary.json`,
  `verdict.json`, the harness's action log (`.jsonl`), measurements (`.csv`) and digests;
- kept when small (256 KiB): the API reads the harness acted on, captured lists, policies and
  load records (`.json`, `.txt`, `.yaml`);
- the driver's own `run.log` and `deploy.log`, up to 2 MiB;
- dropped: pod and process stdout (`out/`, `logs/`, every other `.log`), raw data (`data*/`),
  captures, per-session outputs and backed-up KV rows (`result.json`, `control.json`,
  `backup_*.json`), and a run's working files (`.bak`, `.kill`, `.armed`, `.start`, `.err`).

Before a record is committed, `python bench/harness/record.py <dir>` curates it and scrubs the
driving machine's paths, its temporary directories, the registry and project, and public
addresses, each to a placeholder. Raw output stays with whoever ran it; a `RESULT.md` that
cites a dropped file names what was read, not a file to open.

Credentials never land here: `.gitignore` keeps `certs/`, `kubeconfig`, keys and tokens out, and
the Kubernetes drivers write theirs to a directory outside the record.

## The proofs by number

The Kubernetes proofs are numbered in their records and in the harness. Before proof-15 the
runs were not recorded apart from `2026-10-13-k8s-e2e` (no GPUs) and `2026-10-14-k8s-e2e-refusal`.

| proof | directory | verdict |
|---|---|---|
| 15 | `2026-10-14-dynamo-proof` | pass: a node taken away under load, its rank taken by a spare in 412 ms |
| 16 | `2026-10-14-dynamo-proof-two-losses` | pass: two losses, each taken by a spare |
| 17 | `2026-10-14-dynamo-proof-pipe-leave` | pass: two losses by notice, the second from a spare that had taken the rank |
| 18 | `2026-10-14-dynamo-proof-s1` | pass: the two-loss proof under the coordinator's audiences |
| 19 | not recorded on its own; `2026-10-14-cell-s3` cites its 360 ms change | |
| 20 | `2026-10-14-dynamo-proof-crash` | fail: a node crashed under load |
| 21 | `2026-10-14-dynamo-proof-crash-2` | fail: crashed twice after the node fix, attempts a to c |
| 22 | `2026-10-15-dynamo-proof-crash-3` | pass: crashed twice, each loss taken by a spare |
| 23 | `2026-10-15-dynamo-proof-shared-groups` | pass: shared groups, a deletion then a crash |
| 24 | `2026-10-15-dynamo-proof-head-failover` (pre-registered, superseded by decision 0045) and `2026-10-15-dynamo-handoff-measure` (eight attempts) | the head's failover, since removed |
| 25 | `2026-10-15-dynamo-proof-gated-join` | pass: the gated blocking join, all eight predictions |
| 26 | `2026-10-19-dynamo-proof-known-gap` | pass: the worker-loss proof on the tree without the head's failover, all nine predictions |
| 27 | `2026-10-20-dynamo-proof-oss-polish` | pass on the record: the worker-loss proof on the open-source tree (the patch registry, the codec registry, the executor's shared change); prediction 11 falsified as worded by the designed abort's traceback |

The 2026-10-05 to 10-07 runs are the single-node and two-node batteries on rented GPU pods that
decisions 0009 to 0021 cite; the `2026-10-14-cell-*` runs are the synthetic control-plane cells
of the fleet-scale review (`docs/reviews/2026-10-14-control-plane-at-fleet-scale.md`); the
`2026-10-15-nccl-*` runs are the NCCL probes behind decision 0035's amendments.
