# 0001: one monorepo, splittable per package

2026-10-03. Status: amended by [0046](0046-apache-2.0.md) (2026-10-19); current as amended: the
repository is public under Apache-2.0, and "private" is struck from the title.

**Context.** Adapters move against engine pins and the core moves with them. Separate repos at
this stage cost coordination and buy nothing.

**Decision.** One repository. Every top-level package has its own manifest, tests, README and
CHANGELOG, and imports another package only through its published API. A split is
`git subtree split` plus a CI change.

**Consequences.** Shared CI; one version history; a package can leave when a customer pins it.
