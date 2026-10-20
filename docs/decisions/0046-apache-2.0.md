# 0046: the repository is public under Apache-2.0

2026-10-19 (decided; applied the same day). Amends [0001](0001-monorepo.md), which called the
monorepo private.

**Context.** The repository was private while the product looked for a buyer, and every file
carried `LicenseRef-Everserve-Proprietary` as a placeholder (decision 0001, the industry-shapes
review). The engines it integrates with, vLLM and NVIDIA Dynamo, are Apache-2.0, and the adapter
contract was always meant to be proposed upstream (`docs/CONTRACT.md`). A runtime that asks an
operator to put it under a frontier model is read before it is trusted; a licence that forbids
reading forbids that.

**Decision.** Apache License, Version 2.0, for everything in the repository, from this commit.

- `LICENSE` is the licence text; `NOTICE` names the third-party material shipped with it (the
  DynamoGraphDeployment schema, NVIDIA's, Apache-2.0). Each Python package carries its own copy of
  `LICENSE` so a built wheel includes it.
- Every source file carries `SPDX-License-Identifier: Apache-2.0`, and `scripts/spdx-check.sh`
  fails on any other identifier, not only on a missing one.
- The crates declare `license = "Apache-2.0"` and stay unpublished on crates.io: the control
  plane ships as the image and the chart (`.github/workflows/release.yml`). The Python packages
  declare the same and may be published.
- The operator API's document names the licence (`info.license`), as its test requires.
- Contributions are accepted under the same licence with a Developer Certificate of Origin
  sign-off (`CONTRIBUTING.md`); there is no contributor licence agreement.

**Consequences.** The run records, decisions and reviews in this repository are public too, and
are held to the same bar as the code: nothing in them that an outside reader cannot follow or
should not see (the release checklist, `docs/RELEASE-CHECKLIST.md`). A dependency whose licence
obliges more than Apache-2.0 does not enter (`deny.toml`, CI's pip-audit).
