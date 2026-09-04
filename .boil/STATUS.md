# boil status — slim-llm-memory

_generated 2026-09-04T15:31:51Z · session `boil-slim-llm-memory-20260904-170540`_

**✅ done** · iteration `M5#1` · goal 5/5 green · tickets 0 done / 0 in progress / 0 open

**Goal:** Reranking that costs nothing on easy queries — run the cross-encoder only when the top of the ranking is actually contested.

## Milestones (the controller's ruler)

5/5 must-have green

| Milestone | Tier | Attempts | Result | Spend | Last counterexample |
|---|---|---:|---|---:|---|
| M1 | T1 | 1 | PASS | $0.10 |  |
| M2 | T1 | 1 | PASS | $0.10 |  |
| M3 | T2 | 1 | PASS | $0.20 |  |
| M4 | T2 | 1 | PASS | $0.20 |  |
| M5 | T2 | 1 | PASS | $0.25 |  |

## Event log (newest first)

```
2026-09-04T15:31:51Z boil.score               M5        PASS           EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py::test_bench_offline_reports_three_policies` -> exit 0 | 2026-09-04 | auto
2026-09-04T15:27:25Z boil.prepare             M5        dispatch       examples/04_rerank_bench.py reports MRR, reranker calls and latency for off / auto / always
2026-09-04T15:27:12Z boil.score               M4        PASS           EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py::test_library_auto_matches_topic` -> exit 0 | 2026-09-04 | auto
2026-09-04T15:20:54Z boil.prepare             M4        dispatch       Library.ask supports the same adaptive policy
2026-09-04T15:20:42Z boil.score               M3        PASS           EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py -k topic_auto` -> exit 0 | 2026-09-04 | auto
2026-09-04T15:11:33Z boil.prepare             M3        dispatch       Topic.ask(rerank=rr, rerank_margin=m) skips the reranker when the leader is clear, and says so
2026-09-04T15:11:22Z boil.score               M2        PASS           EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py::test_cross_encoder_passes_max_length_and_batch_size` -> exit 0 | 2026-09-04 | auto
2026-09-04T15:09:59Z boil.prepare             M2        dispatch       Reranker.cross_encoder accepts and uses max_length (default 256) and batch_size (default 32)
2026-09-04T15:09:46Z boil.score               M1        PASS           EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py::test_should_rerank_uses_relative_gap` -> exit 0 | 2026-09-04 | auto
2026-09-04T15:09:38Z boil.prepare             M1        dispatch       should_rerank(scores, margin) decides from the top-two gap relative to the pool spread
2026-09-04T15:08:05Z boil.score               M1        FAIL           AUDIT: write under protected path: tests/test_rerank_auto.py; monkey-patching in tests/test_rerank_auto.py: +def test_cross_encoder_passes_max_length_and_batch_size(monkeypatch):; monkey-patching in tests/test_rerank_auto.py: +    monkeypat
2026-09-04T15:05:40Z boil.prepare             M1        dispatch       should_rerank(scores, margin) decides from the top-two gap relative to the pool spread
```
