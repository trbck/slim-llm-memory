# Log

## 2026-09-04 — adaptive reranking (goal 1)

5/5 milestones green, first-attempt pass rate 100%, $0.85 of a $5.00 budget.
M1 `should_rerank` · M2 cross-encoder knobs · M3 `Topic.ask` · M4 `Library.ask` · M5 bench.

Real-model bench (14 docs, 10 questions, `nomic-embed-text` + `bge-reranker-v2-m3`, CPU):
off 447 ms · auto 749 ms · always 3622 ms per query, MRR 1.00 for all three; auto skipped
the reranker on 10 of 10 questions. Offline (harder) corpus: off MRR 0.933, auto 1.000 with
3 of 10 reranked, always 1.000 with 10.

One process correction worth keeping: the ruler was compiled before it was committed, so the
first `score` failed its audit — the test file I had authored looked like a worker write
against `base_sha`. Fixed by committing the ruler first and re-freezing (identical check
hashes, new baseline). Next time: author → compile → **commit** → wire the guard → loop.

Next goal candidate: `auto` costs ~300 ms more than `off` because it still fetches the 4·k
pool before deciding. Worth measuring whether the pool can be sized lazily.
