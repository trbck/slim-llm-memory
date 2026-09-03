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
  .search_vector(vec, k, kinds, min_score) → [Hit, ...]  (pre-embedded query)
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

## Topic store: fast context for an LLM working on one topic

`topic()` is the "one numpy store per topic" shape with a `requests`-style
front door: open a store, put text in, get context out.

```python
from slim_llm_memory import topic

t = topic("nginx")                          # ~/.slim-llm-memory/topics/nginx, Ollama nomic-embed-text
t.add("docs/")                              # file, directory, raw text, or {name: text}; saved on return
r = t.ask("how do I enable TLS?")           # one embed call + one numpy scan
r                                           # hits with scores, embed ms, scan ms
r.context                                   # numbered block to prepend to an LLM prompt
t.answer("how do I enable TLS?")            # + a local Ollama chat model, grounded on r.context
```

`t.add` is incremental (unchanged chunks are never re-embedded), `t.forget(name)`
drops a doc, `embedder="noop"` runs offline for tests.

Several topics make a database. `library()` is a folder of topic stores;
`ask` embeds once and scans every topic, archiving is a folder move:

```python
from slim_llm_memory import library

db = library()                              # ~/.slim-llm-memory/topics
db.topic("nginx").add("docs/nginx/")
db.topic("cooking").add({"pasta.md": "..."})
db                                          # table of topics
db.ask("how do I enable TLS?")              # hits labelled by topic, merged by score
db.ask("...", topics=["nginx"])
db.route("how do I enable TLS?")             # stage 1 alone: topics ranked by centroid similarity
db.ask("...", route=True)                    # two-stage: route, then scan only the chosen topics
db.archive("cooking"); db.restore("cooking"); db.delete("cooking")
```

`ask` is exact (one concatenated scan) until the library holds more than
50k chunks, then it routes through topic centroids automatically; topics
within 0.05 of the best centroid are kept, and a prompt that matches no
topic falls back to the exact scan. `examples/03_routing_bench.py` has the
numbers: at 500 topics × 200 chunks, routing cuts the scan from ~40 ms to ~2 ms.

`notebooks/library_demo.ipynb` walks through it. `notebooks/use_cases_demo.ipynb`
measures four real use cases (grounded answers, paraphrase, languages, agent
session memory) and ends with an honest table of what is missing compared to a
full RAG stack, an ontology, and a vector database.

`notebooks/topic_context_demo.ipynb` (executed, 14 cells) and
`examples/02_topic_context.py` are the proof: this repo's docs as the
topic, live prompts with the latency split into embed vs scan, an
incremental update, an optional grounded LLM answer, and a synthetic scale
run. Measured on an 8-core CPU box (Ollama CPU-only):

| Step | Cost | Where the time goes |
|------|------|---------------------|
| Prompt → context (33 chunks) | 1.2–1.5 s | Ollama embed of the prompt: >99.9 %. Scan: 0.2–0.5 ms |
| Re-index after one edit | 1 embed call | 32 chunks hash-skipped, 1 re-embedded |
| Scan, 1k × 768 | 0.3 ms p50 / 1.3 ms p95 | numpy GEMV + argpartition |
| Scan, 10k × 768 | 2.4 ms p50 / 7.3 ms p95 | |
| Scan, 50k × 768 | 12 ms p50 / 22 ms p95 | |

The retrieval itself is never the bottleneck at this scale; the embedder is.
On a GPU or with a cloud embedder the prompt-to-context time drops to tens
of milliseconds and the scan numbers above are what remains.

```bash
PYTHONPATH=. python examples/02_topic_context.py --fresh                  # cold build + queries + scale
PYTHONPATH=. python examples/02_topic_context.py --llm llama3.2:3b        # + grounded answer
```

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
