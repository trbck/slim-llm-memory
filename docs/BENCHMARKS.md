# Benchmarks

Everything here was measured on one 8-core CPU machine with Ollama running on the CPU
(no GPU). Re-run the scripts to get numbers for your own hardware.

## Retrieval accuracy

Eight questions about this repo's own docs, from `notebooks/accuracy_demo.ipynb`. Four of
them include the product name, which pulls the intro passages up and makes them harder.

| retrieval | hit@1 | hit@5 | MRR |
|---|---|---|---|
| dense | 0.38 | 0.62 | 0.47 |
| hybrid (default) | 0.38 | 0.88 | 0.56 |
| hybrid + cross-encoder rerank | 0.62 | 1.00 | 0.76 |

Passages are split at headings, about 120 words each with a 20-word overlap
(`topic(..., chunk_words=120, overlap=20)`). The notebook also runs a grid over chunk size,
overlap and the fusion weight, and it confirms those defaults. Tune for your own documents
with `evaluate()`.

## Adaptive reranking

`examples/04_rerank_bench.py`: 14 documents, 10 questions, `nomic-embed-text` embeddings
and the `bge-reranker-v2-m3` cross-encoder.

| policy | MRR | hit@1 | reranker calls | ms/query |
|---|---|---|---|---|
| off | 1.00 | 1.00 | 0 | 447 |
| auto | 1.00 | 1.00 | 0 of 10 | 749 |
| always | 1.00 | 1.00 | 10 | 3622 |

`auto` gave the same answers as `always` and was 4.8× faster. It compares the leader's lead
over the runner-up with the spread of the candidate pool, and skips the reranker when that
gap is at least `rerank_margin` (default 0.15).

With `--offline` (no network, a harder setup where plain retrieval gets one question
wrong), `auto` reranked 3 of 10 questions and reached the same MRR as `always`.

## Question to context, and search speed

`examples/02_topic_context.py` and `notebooks/topic_context_demo.ipynb`, using this repo's
docs as the topic.

| Step | Cost | Where the time goes |
|------|------|---------------------|
| Question → context (33 chunks) | 1.2 to 1.5 s | Ollama embedding the question: >99.9 %. Search: 0.2 to 0.5 ms |
| Re-index after one edit | 1 embed call | 32 chunks unchanged, 1 re-embedded |
| Search, 1k × 768 | 0.3 ms p50 / 1.3 ms p95 | numpy matrix-vector product + argpartition |
| Search, 10k × 768 | 2.4 ms p50 / 7.3 ms p95 | |
| Search, 50k × 768 | 12 ms p50 / 22 ms p95 | |

At this size the search is never the slow part; the embedding call is. With a GPU or a
hosted embedder, question-to-context time drops to tens of milliseconds.

```bash
python examples/02_topic_context.py --fresh               # cold build, questions, scale table
python examples/02_topic_context.py --llm llama3.2:3b     # plus a grounded answer
```

## Topic routing

`examples/03_routing_bench.py`: at 500 topics × 200 chunks, searching only the topics whose
centroid is closest to the question cut the search from about 40 ms to about 2 ms.
