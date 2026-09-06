# NOW — slim-llm-memory

**Project:** unknown · stage ? · north star: (unset)
**Ladder:** 0/0 criteria green
**Goal:** 5/5 checkboxes — Reranking that costs nothing on easy queries — run the cross-encoder only when the top of the ranking is actually contested.
**Measured:** milestones 5/5 green | delta 0 | current - (done) | spent $0.85/$5.00
**Loop:** 6 iterations · 0 actionable tickets
**Budget:** $0.85 of $5.00

## Brakes: CONTINUE
- all clear

## Actionable tickets
- none open — pick the next ladder criterion or close the goal

## Last session
**2026-09-04 — adaptive reranking (goal 1)**
5/5 milestones green, first-attempt pass rate 100%, $0.85 of a $5.00 budget.
M1 `should_rerank` · M2 cross-encoder knobs · M3 `Topic.ask` · M4 `Library.ask` · M5 bench.
Real-model bench (14 docs, 10 questions, `nomic-embed-text` + `bge-reranker-v2-m3`, CPU):
off 447 ms · auto 749 ms · always 3622 ms per query, MRR 1.00 for all three; auto skipped
_(+8 more lines in log.md)_

---
**Next:** goal is green — run `boil-doctor.py --final` and hand off.
