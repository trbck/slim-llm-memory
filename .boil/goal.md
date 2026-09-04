# Goal

**One-line:** Reranking that costs nothing on easy queries — run the cross-encoder only when the top of the ranking is actually contested.

## Success checklist
- [ ] `should_rerank(scores, margin)` decides from the top-two gap relative to the pool spread {#M1}
- [ ] `Reranker.cross_encoder` accepts and uses `max_length` (default 256) and `batch_size` (default 32) {#M2}
- [ ] `Topic.ask(rerank=rr, rerank_margin=m)` skips the reranker when the leader is clear, and says so {#M3}
- [ ] `Library.ask` supports the same adaptive policy {#M4}
- [ ] `examples/04_rerank_bench.py` reports MRR, reranker calls and latency for off / auto / always {#M5}

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
