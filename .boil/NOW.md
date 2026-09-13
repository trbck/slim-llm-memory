# NOW — slim-llm-memory

**Project:** candidate · stage L0 · north star: MRR@10 margin over plain BM25 on a real corpus, at acceptable latency
**Ladder:** 9/24 criteria green · next open: `.boil/` governance pointer added to CLAUDE.md
**Goal:** 5/5 checkboxes — Reranking that costs nothing on easy queries — run the cross-encoder only when the top of the ranking is actually contested.
**Measured:** milestones 5/5 green | delta 0 | current - (done) | spent $0.85/$5.00
**Loop:** 6 iterations · 0 actionable tickets
**Budget:** $0.85 of $5.00

## Brakes: CONTINUE
- all clear

## Actionable tickets
- none open — pick the next ladder criterion or close the goal

## Last session
**2026-09-13 — release-ready package**
Added LICENSE, `py.typed`, CHANGELOG, MANIFEST.in, `.github/workflows/ci.yml` (3.10–3.13 +
build/twine) and `release.yml` (tag `v*` -> PyPI trusted publishing). pyproject: SPDX
`license = "MIT"`, version single-sourced from `__version__`, `tomli` for 3.10 tests.
Two bugs found by testing the tarball instead of the checkout:
_(+7 more lines in log.md)_

---
**Next:** goal is green — run `boil-doctor.py --final` and hand off.
