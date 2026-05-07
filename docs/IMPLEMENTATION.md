# slim-llm-memory — Implementation Plan

> **Goal.** A drop-in **persistent memory + retrieval layer for LLM apps**
> that is fast, cheap, and slim. Pure Python where possible; numpy where
> it actually helps. Ollama for local embeddings and small models; cloud
> LLMs only for hard reasoning. Optional knowledge-graph layer.
>
> Single-machine, single-process, single-tenant. Scales cleanly to
> ~50–100k items per index without changing the public API. Beyond that
> you swap the storage backend, not the code that uses it.

This document is the contract. New work either follows it or amends it.
Audience: yourself in 4 weeks, and anyone else dropping the library into
a new project.

---

## 1. Scope

### In scope
- Persistent vector index over arbitrary `(id, text, metadata)` items.
- Incremental updates — only re-embed items whose content changed.
- Top-k semantic search.
- Pairwise duplicate clustering.
- Optional knowledge-graph layer over the same items.
- Tiered LLM routing: heuristic → local embed → local small LLM → cloud LLM.
- Observability primitives (ring-buffer of recent failures, simple stats).
- Atomic persistence — never leave the index in a half-written state.

### Out of scope
- Multi-tenant isolation.
- Multi-process concurrent writes (single writer enforced; multiple
  readers are fine).
- Real-time streaming pipelines (>100 writes/s sustained).
- ANN indexes (HNSW, IVF). When the linear scan stops being fast enough,
  swap the storage backend — public API stays.
- Distributed deployment.

### Non-goals (don't add these)
- A web service. Use it as a library.
- A schema layer beyond `metadata: dict[str, Any]`. Keep it free-form.
- A full graph database. NetworkX or Kùzu when graph is needed; nothing
  hand-rolled.
- A query language. Functions, not strings.

---

## 2. Stack at a glance

| Layer            | Choice                                | Why                                                              |
|------------------|---------------------------------------|------------------------------------------------------------------|
| Embeddings       | Ollama `nomic-embed-text` (local) or Gemini `text-embedding-004` (cloud) | Free, local, 768-dim, ~50 ms/batch on CPU. Cloud fallback works. |
| Vector storage   | `numpy.ndarray` mmap'd from a single `.npy`  | Loads in milliseconds, zero serialisation overhead.        |
| Metadata storage | `jsonl` (one record per line)         | Append-friendly, human-inspectable, no schema lock-in.           |
| Search           | `numpy.dot(matrix, q) / norms` linear scan | Sub-10 ms for 50k × 768. BLAS does the work; numba doesn't help. |
| Graph (optional) | NetworkX in-process, pickled          | Zero infra, fits 100k nodes. Kùzu only when multi-hop matters.   |
| Local small LLM  | Ollama `qwen2.5:3b` or `llama3.2:3b`  | 2–4 GB RAM, CPU-acceptable, multilingual.                        |
| Cloud LLM        | Claude Sonnet 4.6 / Gemini Flash      | Reasoning, planning, structured output.                          |
| Concurrency      | Single-writer file lock + atomic rename | Simple, correct, survives crashes.                             |

### Performance budget (target latency at p95)

| Operation                    | 1k items | 10k items | 50k items |
|------------------------------|----------|-----------|-----------|
| `embed(texts)` (local)       | 50 ms    | 50 ms     | 50 ms     |
| `embed(texts)` (cloud)       | 200 ms   | 200 ms    | 200 ms    |
| `search(query, k=10)`        | <1 ms    | 5 ms      | 30 ms     |
| `find_duplicates(threshold)` | 5 ms     | 200 ms    | 5 s ⚠️    |
| `reindex_incremental(N changed)` | 50 ms × N batches | same | same |

If any number bursts >10× the budget, that's the signal to graduate
the relevant subsystem (see §10).

---

## 3. Module layout

A single Python package, ~6 files, ~1000 LOC total. No external service.

```
slim_llm_memory/
├── __init__.py            # public re-exports
├── store.py               # numpy + jsonl persistence (~250 LOC)
├── embed.py               # Ollama / Gemini / no-op embedder (~150 LOC)
├── index.py               # search / dedup / cluster (~200 LOC)
├── graph.py               # optional NetworkX wrapper (~150 LOC)
├── llm.py                 # tier routing + small ollama wrapper (~200 LOC)
└── obs.py                 # ring buffers, error rate, stats (~80 LOC)
```

Optional: `tests/`, `examples/`, `benchmarks/`. No `Dockerfile`, no
service binaries — it's a library.

---

## 4. Public API (the only thing callers see)

```python
from slim_llm_memory import Memory, Embedder, Tier, Graph

# 1. construct (everything else flows from this)
mem = Memory(
    path="./mymemory",                    # directory; created if absent
    embedder=Embedder.ollama("nomic-embed-text"),
    # ↑ or Embedder.gemini("text-embedding-004", api_key=...)
    # ↑ or Embedder.noop() for tests
)

# 2. add / update items (incremental — only changed get re-embedded)
mem.upsert([
    {"id": "doc1", "text": "...",  "meta": {"kind": "note", "src": "..."}},
    ...
])

# 3. search
hits = mem.search("how do i set up nginx?", k=10, kinds={"note"}, min_score=0.55)
# → [{"id":"doc7", "score":0.81, "text":"...", "meta":{...}}, ...]

# 4. duplicates
clusters = mem.find_duplicates(threshold=0.86)
# → [["doc7", "doc92"], ["doc14", "doc55", "doc201"]]

# 5. update one item's text (triggers re-embed)
mem.update_text("doc1", "new text")

# 6. delete
mem.remove("doc1")

# 7. snapshot stats (for /health endpoints)
mem.stats()
# → {"items": 1234, "embed_dim": 768, "file_age_seconds": 88, ...}

# 8. flush to disk (atomic — safe to crash mid-anything)
mem.flush()


# ─────────── Tiered LLM routing ───────────
tier = Tier(
    local_embed=Embedder.ollama("nomic-embed-text"),
    local_llm=("ollama", "qwen2.5:3b"),
    cloud_llm=("anthropic", "claude-sonnet-4-6"),  # or ("gemini", "gemini-2.0-flash")
)

# Heuristic first; falls through to local; falls through to cloud only
# when needed.
result = tier.classify(text, choices=["bug", "question", "feature"])
result = tier.summarise(long_text, max_words=80)
result = tier.reason(prompt)            # always cloud — explicit


# ─────────── Optional graph layer ───────────
g = Graph(mem)                                           # shares storage path
g.link("doc1", "doc7", relation="cites", weight=1.0)
g.neighbours("doc1", relation="cites", depth=2)
g.related("doc1", k=10)                                  # vector + graph hybrid
```

That's the entire surface. Anything else is an internal helper.

---

## 5. Persistence model

### Files in `path/`

```
items.jsonl            # one record per item: {"id","text","hash","meta","ts"}
vectors.npy            # float32 array, shape (N, embed_dim) — row-aligned with items.jsonl
embedder.json          # {"name":"ollama:nomic-embed-text","dim":768} — sanity check
graph.pickle           # NetworkX DiGraph (only if Graph(mem) ever used)
.lock                  # exclusive write lock (fcntl), removed on close
```

### Invariants
- `vectors.npy` row `i` corresponds to `items.jsonl` line `i`. Always.
- `items.jsonl` is **append-only during a session**. Tombstones (`{"id":"X","deleted":true}`) handle removes; compaction happens at `flush()` when >20% tombstoned.
- `hash` = `sha1(text)[:16]`. Used to skip re-embed when identical text is upserted.
- Atomic write: write to `.tmp` → `os.replace()` → fsync. Never partial.
- Embedder change → entire index is rebuilt. Logged loudly.

### Why these choices
- `.npy` mmaps in microseconds and survives Python upgrades.
- `jsonl` is `tail`-friendly and inspectable without a tool.
- One pickle file for the graph because NetworkX serialisation is small
  and fast at our scale; doesn't need to align with vectors row-wise.

---

## 6. Phased implementation plan

Each phase is independently shippable and useful. A new project starts
with phase 1, adds phases as needed.

### Phase 1 — `Memory` core (vector index + persistence)
Files: `store.py`, `embed.py` (just Ollama + noop), `index.py`,
`obs.py`, `__init__.py`.

Acceptance:
- `Memory.upsert([…])` of 1000 items completes in <30 s on a cold index.
- `Memory.search("…", k=10)` p95 <30 ms at 10k items.
- A crash mid-`flush()` leaves the previous index intact.
- Re-running the same `upsert` is a no-op (hash dedup).
- `pytest tests/test_store.py` covers: persistence round-trip, hash
  skip, atomic crash recovery (kill mid-write, reopen, verify), embedder
  mismatch refuses to load.

Ship at: ~400 LOC, ~10 tests.

### Phase 2 — `Embedder.gemini` cloud fallback
Adds `embed.py::Embedder.gemini()`. Useful when local Ollama isn't
available (CI, serverless, low-RAM machines).

Acceptance:
- Same `Embedder` interface; swap is transparent to callers.
- Network failure surfaces as `EmbedderError`; partial batches are
  retried with exponential backoff.
- `pytest tests/test_embedder.py` mocks both backends.

### Phase 3 — Duplicate detection + clustering
Adds `Memory.find_duplicates(threshold)` to `index.py`.

Acceptance:
- Union-find clustering on cosine ≥ threshold.
- Returns clusters with ≥2 members; singletons omitted.
- For 10k items: <500 ms with `numpy.dot` (full pairwise upper triangle).
- For 50k items: warn in the docstring; suggest pre-filtering by metadata.

### Phase 4 — `Graph` layer
Files: `graph.py`. NetworkX DiGraph wrapper with vector-aware
helpers (`related()` does `mem.search() + g.neighbours()`).

Acceptance:
- `link / unlink / neighbours / related` work end-to-end.
- Graph survives `flush()` and reload.
- `g.related("doc1", k=10)` blends vector hits and 1-hop graph
  neighbours, deduped, ranked by `0.6*cosine + 0.4*graph_score`.

### Phase 5 — `Tier` LLM routing
Files: `llm.py`. Routes calls through heuristic → local embed →
local LLM → cloud LLM in declared order. Each step can short-circuit.

Acceptance:
- Tier can be configured at construction time; no global state.
- Local LLM call falls back to cloud on timeout (>5 s default).
- All call sites carry an `op` label that ends up in `obs` ring buffers.
- Cloud-only path (`tier.reason()`) doesn't even start the local LLM.

### Phase 6 — Observability
Files: `obs.py`. Ring buffer of recent embed/LLM failures, simple
counters (calls / hits / failures / cache_age). Exposed via
`Memory.stats()` and `Tier.stats()` for any host-app health endpoint.

Acceptance:
- No global state — everything hangs off the `Memory` / `Tier`
  instance.
- `stats()` returns a JSON-safe dict; safe to call from any thread.

Ship at: full ~1000 LOC, ~30 tests.

### Phase 7 (optional) — ANN backend swap
When an index outgrows linear scan (~100k+), drop in faiss IVF
behind the same `Memory.search` API. The contract is `search(query,
k) -> list[hit]`; implementation is opaque. Keep linear-scan as a
fallback for small indexes (faster cold-start).

---

## 7. Tier routing strategy (concrete)

The whole point of tiering is **latency + cost discipline**. Default
order, top to bottom:

| Tier | Latency | Cost | What it does                                              |
|------|---------|------|-----------------------------------------------------------|
| L0   | 0 ms    | 0    | Regex / heuristic. Catches the obvious.                   |
| L1   | 50 ms   | 0    | Local embed (Ollama) + linear scan retrieval.             |
| L2   | 0.5–3 s | 0    | Local 3–7B LLM (Ollama). Classification, summary, simple Q&A. |
| L3   | 1–10 s  | $    | Cloud LLM (Claude / Gemini). Reasoning, planning, structured output. |

Routing rules per operation type:

```
classify(text, choices)        → L0 → L2 → L3
extract(text, schema)          → L0 → L2 → L3
embed(texts)                   → L1
retrieve(query, k)             → L1
summarise(text, max_words)     → L2 → L3 (only if local truncates badly)
rerank(query, candidates)      → L2
reason(prompt)                 → L3 (always)
plan(goal, context)            → L3 (always)
```

Defaults that age well:
- **Local embed**: `nomic-embed-text` (768-dim, multilingual, current best
  free local embed).
- **Local LLM**: `qwen2.5:3b` (German + English good; 4 GB RAM). For
  English-heavy projects: `llama3.2:3b`.
- **Cloud LLM**: Claude Sonnet 4.6 for reasoning; Gemini 2.0 Flash for
  cheap structured output. Both have free tiers.

Don't tier `reason()` — explicit cloud call. Tiering reasoning produces
unpredictable quality.

---

## 8. Knowledge-graph layer

Use only when relations are explicit and worth traversing. Vector
similarity covers ~80% of "find related" needs; graph wins for
multi-hop, typed-edge, or constraint-walked queries.

### Design
- Backed by `networkx.DiGraph`. Each node id = `Memory` item id.
- Edges carry `relation: str`, `weight: float`, `meta: dict`.
- `Graph` doesn't own item data — it references `Memory.items` by id.
  This means `mem.remove("X")` should call `g.drop_node("X")` if the
  graph is in use. Do this as an explicit "linked" mode opt-in
  (`Graph(mem, link_lifecycle=True)`).
- Persistence: pickle `graph.pickle` next to `vectors.npy`.

### Hybrid retrieval
`g.related(id, k=10)` blends:
1. `mem.search(item_text, k=k*2)` (vector neighbours)
2. `g.neighbours(id, depth=1)` (1-hop edges)
3. Score = `α*cosine + (1-α)*edge_weight`, default α=0.6
4. Dedup, sort, return top-k

Tested at 100k nodes / 500k edges in ~30 ms on a laptop.

### When to graduate
- Multi-hop with constraints ("find all X cited by Y written after Z"):
  move to **Kùzu** (embedded graph DB, single binary, Cypher-lite).
  Same `Graph` interface, different backend.
- Distributed graph: out of scope. Use Neo4j or similar.

---

## 9. Tests + observability

### Tests (per phase, see §6)
- `test_store.py` — persistence, hash skip, atomic crash recovery,
  embedder-mismatch refusal.
- `test_embedder.py` — Ollama mock, Gemini mock, retry on 429.
- `test_index.py` — search ranking, kind/min_score filters, duplicates
  clustering correctness on synthetic vectors.
- `test_graph.py` — link/unlink, hybrid scoring, lifecycle linkage.
- `test_tier.py` — fallback chain, timeout-to-cloud, op labels propagate.
- `test_obs.py` — ring-buffer roll, stats shape.

Target: ≥30 tests, full pass under 5 s, zero network calls (all
mocked).

### Observability via `obs.py`
- Ring buffers (capacity 50): `embed_errors`, `llm_errors`,
  `slow_queries` (anything >100 ms).
- Counters: `embed_calls`, `embed_hits` (hash skipped),
  `search_calls`, `tier_l0/l1/l2/l3_hits`.
- `Memory.stats()` returns
  `{items, embed_dim, file_age_seconds, embed_calls, embed_errors_24h, ...}`.
- `Tier.stats()` returns per-tier hit counts so you can see how often
  cloud is actually invoked.

Caller wires these into their own `/health` endpoint or logs.

---

## 10. Migration paths (when to graduate which subsystem)

Each subsystem can be swapped independently because the public API is
narrow. Triggers and replacements:

| Symptom                                          | Subsystem | Replace with                                  |
|--------------------------------------------------|-----------|-----------------------------------------------|
| Search p95 > 100 ms at your scale                | `index.py` | `faiss-cpu` HNSW behind same `search()`      |
| 2nd writer process needs to write                | `store.py` | SQLite + `sqlite-vss` extension              |
| > 1M items                                       | both       | Qdrant / Weaviate as a service               |
| Embed model needs swapping mid-life              | `embed.py` | New `Embedder`; full reindex (logged loudly) |
| Graph needs multi-hop with constraints           | `graph.py` | Kùzu (embedded), same `Graph` interface      |
| Local LLM too slow / quality too low             | `llm.py`   | Drop L2; route L0 → L3 directly              |
| Multi-tenant / per-user isolation                | top-level  | One `Memory` per tenant; or graduate to a service |

The point of the slim stack: you don't outgrow it gradually — when you
do, the symptoms are obvious and the migration is local to one file.

---

## 11. Don't-do list

- **No global state.** Everything hangs off a `Memory` or `Tier`
  instance. Multiple instances in the same process must work.
- **No silent embedder change.** If `embedder.json` doesn't match the
  configured embedder, refuse to load and tell the user how to reindex.
- **No partial writes.** Always `.tmp → os.replace`.
- **No magic background threads.** Reindexing is explicit. The host app
  schedules it.
- **No hidden network calls.** `Embedder.noop()` exists for tests; cloud
  embedders never auto-fall-back to local without `Tier`.
- **No string query language.** Functions, not DSL.
- **No "soft" features that aren't covered by tests.** If it's not
  tested, it's not in the public API.

---

## 12. Reference dependency footprint

Hard:
- `numpy` (≥1.24)
- `httpx` (≥0.25) — for Ollama HTTP and cloud APIs

Soft (lazy-imported inside the modules that need them):
- `networkx` (only if `Graph` is used)
- `google-genai` (only if `Embedder.gemini()` is used)
- `anthropic` (only if `Tier(cloud_llm=("anthropic", …))`)
- `kuzu` (only if you graduate the graph backend)
- `faiss-cpu` (only if you graduate the search backend)

Total cold install (hard deps only): ~30 MB, no native binaries beyond
numpy's BLAS. Runs on Python 3.10+.

---

## 13. Repository layout for a new project using this

```
your-project/
├── slim_llm_memory/         # vendor or pip install -e .
├── tests/
├── examples/
│   ├── 01_minimal.py        # 20 lines: index a folder of markdown
│   ├── 02_tier.py           # 30 lines: classify with tiered routing
│   └── 03_graph.py          # 40 lines: hybrid retrieval over a graph
├── README.md                # quickstart + this doc as ./IMPLEMENTATION.md
└── pyproject.toml
```

`examples/01_minimal.py` should be **runnable in one command**, no
config, just `python examples/01_minimal.py` and it indexes its own
docstring. That's the credibility test for the API.

---

## 14. First commit checklist (for a fresh project)

- [ ] `pyproject.toml` with hard deps only (numpy + httpx).
- [ ] `slim_llm_memory/__init__.py` re-exports `Memory`, `Embedder`,
      `Tier`, `Graph`.
- [ ] `store.py` with atomic write + tombstones + reload.
- [ ] `embed.py` with `noop` + `ollama` only.
- [ ] `index.py` with `search` + `find_duplicates`.
- [ ] `obs.py` with ring buffers + counters.
- [ ] `tests/test_store.py` (8 tests, including atomic-crash recovery).
- [ ] `examples/01_minimal.py`.
- [ ] `README.md` linking to this doc.

That's phase 1. Ship it. Add phase 2+ as the consuming app needs them.
