# NOW — slim-llm-memory

**Project:** candidate · stage L0 · north star: MRR@10 margin over plain BM25 on a real corpus, at acceptable latency
**Ladder:** 6/24 criteria green · next open: The demo is repeatable from a clean checkout and documented in the README
**Goal:** 5/5 checkboxes — Reranking that costs nothing on easy queries — run the cross-encoder only when the top of the ranking is actually contested.
**Measured:** milestones 5/5 green | delta 0 | current - (done) | spent $0.85/$5.00
**Loop:** 6 iterations · 0 actionable tickets
**Budget:** $0.85 of $5.00

## Brakes: CONTINUE
- all clear

## Actionable tickets
- none open — pick the next ladder criterion or close the goal

## Last session
**2026-09-09 — charter + ladder (governance)**
Goal 1 closed: `boil-doctor.py --final` -> FINAL OK, 5/5 boxes with fresh EVIDENCE.
Wrote `.boil/charter.md` and `.boil/ladder.md`, clearing the UNGOVERNED flag. Track is
**life first, revisit business at L3**; north star is retrieval quality vs. plain BM25 on a
*real* corpus at bounded latency (currently negative — advisor's BM25 wins, ~1600x faster);
_(+13 more lines in log.md)_

---
**Next:** goal is green — run `boil-doctor.py --final` and hand off.
