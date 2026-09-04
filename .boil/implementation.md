# Implementation notes

Adaptive rerank sits between fusion and the reranker call in `Topic.ask`:

    candidates → fuse() → [(cand, score)] → should_rerank(scores, margin)?
                                             ├─ no  → top-k as fused        (rerank_skipped=True)
                                             └─ yes → reranker over 4·k     (rerank_skipped=False)

`Library.ask` reuses the same decision in `_merge` so routed and flat paths behave identically.
