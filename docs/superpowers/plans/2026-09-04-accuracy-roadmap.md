# Accuracy roadmap — hybrid retrieval, reranking, graph, sessions

**Goal:** close the gaps `notebooks/use_cases_demo.ipynb` measured, without changing the
requests-style surface: `topic(...).add/.ask/.answer`, `library(...).ask/.route`.
Every feature is gated by `evaluate()` numbers on the fixture questions.

**Rule for speed:** pure Python + numpy first; measure at 50k chunks; only then Cython or
Go for a hot loop that measures over budget (budget: hybrid query < 20 ms at 50k chunks,
index build < 5 s at 50k, or cached on disk).

## Interface (all additive)

```python
from slim_llm_memory import topic, library, evaluate

t = topic("nginx", chunk_words=120, overlap=20)        # heading-aware chunks with overlap
t.ask(q, k=5, mode="hybrid")                           # "hybrid" (default) | "dense" | "keyword"
t.ask(q, rerank=True)                                  # cross-encoder over the top-20 (optional extra)
t.answer(q, model=..., stream=False, rewrite=False)    # refuses when weak; validates [n] citations
r = evaluate(t, [("question", "expected term or doc"), ...], k=5)   # hit@1, hit@k, MRR, table repr

t.link("a.md", "b.md", relation="cites")               # graph edges; [[wikilinks]] become edges on add
t.related("a.md", k=5)                                 # 0.6·cosine + 0.4·graph
t.add(text, enrich=True)                               # local-LLM entities + relations into meta/graph
t.entities()                                           # {"Postgres": 3, ...}
t.ask(q, entity="Postgres")                            # filter by extracted entity

s = db.session("2026-09-04")                           # conversation memory as a topic
s.turn("user", "..."); s.turn("assistant", "...")
s.recall("what did we decide?")                        # → Result
s.history(n=10)
```

## Phases and files

| Phase | Files | Tests | Gate |
|---|---|---|---|
| 1a eval | `evals.py` | `test_evals.py` | reports hit@1/hit@k/MRR on fixtures |
| 1b hybrid | `keyword.py` (BM25, numpy CSR, disk cache `bm25.npz`), `topics.py` (`mode`, RRF) | `test_keyword.py`, `test_topic.py` | named questions rank 1 |
| 1c chunking | `chunking.py` (headings + overlap), `topics.py` | `test_chunking.py` | eval ≥ previous |
| 1d rewrite | `topics.py` (`answer(rewrite=)`) | mocked | — |
| 2a rerank | `rerank.py` (`Reranker.cross_encoder`, `.noop`), `topics.py` (`rerank=`) | mocked + optional real | eval ≥ previous |
| 2b multilingual | docs + notebook with `ollama:bge-m3` | notebook | EN→DE 3/3 |
| 2c answer hygiene | `topics.py` | mocked | refusal + citation check |
| 3a graph | `graph.py` (NetworkX, `graph.json`), `topics.py` (`link/related`) | `test_graph.py` | related() blends |
| 3b enrich | `enrich.py` (LLM JSON extraction), `topics.py` | mocked LLM | entities filter works |
| 4 sessions | `sessions.py`, `libraries.py` (`session()`) | `test_sessions.py` | recall works |
| demo | `notebooks/accuracy_demo.ipynb` | executed | before/after table |

## Results (2026-09-04, notebooks/accuracy_demo.ipynb, this repo's docs, 8 questions)

| retrieval | hit@1 | hit@5 | MRR |
|---|---|---|---|
| dense | 0.38 | 0.62 | 0.47 |
| keyword (BM25) | 0.25 | 0.88 | 0.50 |
| hybrid, linear fusion α=0.5 (default) | 0.38 | 0.88 | 0.56 |
| hybrid + cross-encoder rerank | 0.62 | 1.00 | 0.76 |

Tuning grid (chunk 120/160, overlap 0/20, α 0.3–0.6): no effect from chunking on this corpus; α=0.5 best.
Multilingual (bge-m3) EN→DE: 4/4 at rank 1 (was 1/3 with nomic-embed-text).
Speed at 50k chunks: BM25 build 7.2 s (cached per store version), BM25 query 2.8 ms; hybrid ask at 20k
chunks p50 6.6 ms / p95 10.3 ms. Cross-encoder ≈ 1 s per candidate on this loaded CPU → opt-in.
Verdict on Cython/Go: not needed; first candidate would be the BM25 tokeniser above ~500k chunks.
