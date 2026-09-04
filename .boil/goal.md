# Goal

**One-line:** Reranking that costs nothing on easy queries — run the cross-encoder only when the top of the ranking is actually contested.

## Success checklist
- [x] `should_rerank(scores, margin)` decides from the top-two gap relative to the pool spread — EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py::test_should_rerank_uses_relative_gap` -> exit 0 | 2026-09-04 | auto {#M1}
- [x] `Reranker.cross_encoder` accepts and uses `max_length` (default 256) and `batch_size` (default 32) — EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py::test_cross_encoder_passes_max_length_and_batch_size` -> exit 0 | 2026-09-04 | auto {#M2}
- [x] `Topic.ask(rerank=rr, rerank_margin=m)` skips the reranker when the leader is clear, and says so — EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py -k topic_auto` -> exit 0 | 2026-09-04 | auto {#M3}
- [x] `Library.ask` supports the same adaptive policy — EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py::test_library_auto_matches_topic` -> exit 0 | 2026-09-04 | auto {#M4}
- [x] `examples/04_rerank_bench.py` reports MRR, reranker calls and latency for off / auto / always — EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q tests/test_rerank_auto.py::test_bench_offline_reports_three_policies` -> exit 0 | 2026-09-04 | auto {#M5}

## Requirements understanding

| Requirement | Interpretation | Acceptance signal | Confidence | Open uncertainty |
|---|---|---|---:|---|
| Adaptive rerank | Decide per query from the fused-score gap; no reranker call when the winner is clear | M1 + M3 + M4 checks green | 97 | the margin default is a guess; the bench prints the trade |
| Faster rerank | Shorter `max_length`, batched `predict` | M2 check green | 95 | quality cost of 256 tokens unmeasured on this CPU |
| Honest reporting | A result says whether reranking ran, and whether the policy declined | `rerank_skipped` asserted in M3/M4 | 99 | none |

## How the user will see this works

```
PYTHONPATH=. python examples/04_rerank_bench.py --offline --json
```
prints one row per policy (off / auto / always) with MRR, reranker calls and milliseconds.
The real-model demo runs the same script without `--offline`.

## Out of scope
- Changing the fusion formula or the default reranker model
- GPU, quantisation, or a different reranker family
- Tuning `FUSION_ALPHA` (separate goal)
