# Changelog

## 0.2.1 — 2026-09-14

Fixes from a review of 0.2.0, all with tests:

- MCP: a bad argument (unknown topic, empty text, `k < 1`) now reaches the client as a readable
  message instead of the SDK's generic "Error executing tool".
- `MemoryTools`: `recall`/`forget`/`answer` accept any spelling of a topic name that `remember`
  accepts (case, spaces, hyphens) and label hits and results with the topic's display name; `k` is
  validated; `dispatch()` rejects unknown and missing arguments by name; `answer()` no longer
  forwards arbitrary keyword arguments (a `stream=True` used to crash it).
- Topic names are slugged without a leading `_` or `.`, so an agent-chosen `_archive` or
  `.hidden` can no longer write a store into the library's own folders.
- Unnamed notes are keyed by 12 hex digits of their hash instead of 6 (silent overwrite of a
  different note after a few thousand notes).
- `Topic.add({doc: ""})` raises instead of silently deleting the doc's chunks; empty files in a
  folder still sync as before.
- Enrichment: when a chunk's text changes, the graph edges extracted from the old text are
  removed and the doc's `mentions` edges are rebuilt; a later `add(enrich=...)` re-extracts for
  chunks that carry no entities even when their text is unchanged.

## 0.2.0 — 2026-09-14

- New: `MemoryTools` — remember / recall / forget / answer / topics with JSON in and JSON out,
  plus `dispatch(name, args)`; `anthropic_tools()` and `openai_tools()` emit the tool definitions.
- New: `slim-memory-mcp`, an MCP server over the same five verbs (`pip install slim-llm-memory[mcp]`).
- New: `to_dict()` on `Result`, `Hit`, `Answer`, `Report`, `Route` and `Added`.
- New: `docs/llms.txt`, the whole API on one page for coding agents; shipped in the sdist.
- Fix: re-adding an unchanged document (re-indexing a folder, say) dropped the entities that
  `add(enrich=...)` had extracted for its chunks, because the meta-only update replaced the
  metadata wholesale. Entities now survive as long as the chunk text is unchanged.
- Docs: five use-case notebooks (`notebooks/1*_usecase_*.ipynb`): codebase Q&A, support ticket
  triage, agent memory across restarts, a judged semantic cache, notes housekeeping.

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
