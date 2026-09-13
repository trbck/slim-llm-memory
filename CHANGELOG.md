# Changelog

## 0.1.1 — 2026-09-13

- Fix: `Topic.add()` raised `ImportError` on a plain install (no `[graph]` extra) when a
  document contained a `[[wikilink]]` to another document in the store. The documents were
  saved, but the call failed. Links are now skipped when networkx is missing.
- Docs: README rewritten, quickstart first; benchmark tables moved to `docs/BENCHMARKS.md`.

## 0.1.0 — 2026-09-13

First packaged release.

- `Memory`: persistent vector index — hash-skip upsert, cosine search, atomic flush.
- `topic()` / `library()`: per-topic stores and a database of topics with routing.
- Hybrid retrieval (BM25 + embeddings), adaptive cross-encoder reranking (`rerank="auto"`).
- Grounded answers with validated `[n]` citations; `evaluate()` for MRR/recall.
- Extras: `graph`, `rerank`, `obsidian`. `gemini` and `anthropic` are declared but not implemented yet.
