# slim-llm-memory

Slim, fast, persistent **memory + retrieval for LLM apps**. Pure Python
where possible; numpy where it actually helps. Ollama for local
embeddings; cloud LLMs only for hard reasoning. ~1000 LOC, two hard
deps (numpy + httpx), drops into anything.

> **Status:** phase 1 — `Memory` core. See [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)
> for the full plan and what comes next (Gemini fallback, Tier router,
> Graph layer, ANN swap).

## Why

Vector DBs and full RAG frameworks are overkill for personal projects
and research code. At < 50k items, a single numpy array, a jsonl file,
and a content-hash for incremental updates is all you actually need.
This library is exactly that — but written carefully enough that you
can build serious things on it without hitting sharp corners.

When you outgrow it, the public API is **swap-compatible** with a real
vector store (faiss / SQLite-vss / Qdrant). The migration is local to
one file.

## Install

```bash
pip install slim-llm-memory          # numpy + httpx only
pip install slim-llm-memory[gemini]  # + Gemini cloud embedder (phase 2)
pip install slim-llm-memory[graph]   # + NetworkX graph layer (phase 4)
```

## 30-second tour

```python
from slim_llm_memory import Memory, Embedder

# Local Ollama for embeddings, persistent index in ./mymemory/
mem = Memory("./mymemory", Embedder.ollama("nomic-embed-text"))

# Add or update items — only changed texts are re-embedded
mem.upsert([
    {"id": "doc1", "text": "how to set up nginx", "meta": {"kind": "note"}},
    {"id": "doc2", "text": "milch kaufen",        "meta": {"kind": "shopping"}},
])

# Top-k semantic search — optional filters
hits = mem.search("nginx tutorial", k=5, kinds={"note"}, min_score=0.55)
for h in hits:
    print(h.id, h.score, h.text)

# Find duplicates by cosine similarity
clusters = mem.find_duplicates(threshold=0.86)

# Atomic persistence — safe to crash mid-anything
mem.flush()
```

`Embedder.noop()` exists for tests and offline development — same
interface, deterministic SHA-256 derived vectors, no network.

## What's in the box (phase 1)

| Module        | Purpose                                                      |
|---------------|--------------------------------------------------------------|
| `index.py`    | `Memory`, `Hit` — public API                                 |
| `store.py`    | Versioned manifest + atomic flush + fcntl lock + tombstones  |
| `embed.py`    | `Embedder.noop` (tests) + `Embedder.ollama` (local)          |
| `obs.py`      | Per-instance ring buffers + counters for `Memory.stats()`    |

Public API surface (the only thing callers see):

```
Memory(path, embedder)
  .upsert(items)              → {added, updated, skipped, embed_calls}
  .search(query, k, kinds, min_score)  → [Hit, ...]
  .neighbours(id, k, kinds)   → [Hit, ...]  (no embed call)
  .find_duplicates(threshold) → [[id, ...], ...]
  .update_text(id, text)      → bool
  .remove(id)                 → bool
  .stats()                    → dict (JSON-safe)
  .flush(force=False)         → bool
  .close(flush=True)
  context manager: `with Memory(...) as mem: ...`

Embedder.noop(dim=384)
Embedder.ollama(model="nomic-embed-text", base_url="http://localhost:11434", timeout=60)
```

## Persistence model

Files in your index directory:

```
items.vN.jsonl       one record per item: {id, text, hash, meta, ts, deleted?}
vectors.vN.npy       float32 ndarray, shape (N, dim) — row-aligned with items
manifest.json        atomic commit point; loading always honours its version pointer
.lock                advisory exclusive lock (one writer per directory)
```

A crash mid-flush leaves the **previous manifest version intact** — the
old files load cleanly. Garbage versioned files left behind by
crashes are ignored on next load.

## Performance

At p95 on a CPU with prenormalised float32 vectors:

| Items  | Pure-Python cosine | numpy linear scan (this lib) | faiss HNSW (phase 7) |
|--------|--------------------|------------------------------|----------------------|
| 1k     | 5–20 ms            | <1 ms                        | <1 ms                |
| 10k    | 50–200 ms          | 5 ms                         | <1 ms                |
| 50k    | 0.5–2 s            | 30 ms                        | 1–10 ms              |
| 100k+  | dead               | 100–500 ms                   | 1–10 ms              |

Phase 1 ships the numpy linear scan. When you outgrow it, swap the
storage backend behind the same `Memory.search()` signature.

## Tests + example

```bash
pytest                              # 44 tests, ~1.2 s, no network
PYTHONPATH=. python examples/01_minimal.py
```

## Migration paths

When the slim stack stops being enough, swap one file:

| Symptom                              | Replace                                    |
|--------------------------------------|--------------------------------------------|
| Search p95 > 100 ms at your scale    | `index.py` → faiss-cpu HNSW (same API)     |
| Need a 2nd writer process            | `store.py` → SQLite + sqlite-vss extension |
| > 1M items                           | both → Qdrant / Weaviate as a service      |
| Need real multi-hop graph queries    | future `graph.py` → Kùzu (embedded)        |
| Local LLM too slow / quality too low | future `tier.py` → drop L2; route L0 → L3  |

The whole point is: you don't outgrow it gradually. When you do, the
symptoms are obvious and the migration is local.

## License

MIT.
