# boil status — slim-llm-memory

_generated 2026-09-04T15:08:05Z · session `boil-slim-llm-memory-20260904-170540`_

**· idle** · iteration `M1#1` · goal 0/5 green · tickets 0 done / 0 in progress / 0 open

**Goal:** Reranking that costs nothing on easy queries — run the cross-encoder only when the top of the ranking is actually contested.

## Milestones (the controller's ruler)

0/5 must-have green

| Milestone | Tier | Attempts | Result | Spend | Last counterexample |
|---|---|---:|---|---:|---|
| M1 | T1 | 1 | FAIL | $0.10 | AUDIT: write under protected path: tests/test_rerank_auto.py; monkey-p |
| M2 | T1 | 0 | - | $0.00 |  |
| M3 | T2 | 0 | - | $0.00 |  |
| M4 | T2 | 0 | - | $0.00 |  |
| M5 | T2 | 0 | - | $0.00 |  |

## Event log (newest first)

```
2026-09-04T15:08:05Z boil.score               M1        FAIL           AUDIT: write under protected path: tests/test_rerank_auto.py; monkey-patching in tests/test_rerank_auto.py: +def test_cross_encoder_passes_max_length_and_batch_size(monkeypatch):; monkey-patching in tests/test_rerank_auto.py: +    monkeypat
2026-09-04T15:05:40Z boil.prepare             M1        dispatch       should_rerank(scores, margin) decides from the top-two gap relative to the pool spread
```
