# Obsidian Brain — design

**Date:** 2026-05-20
**Scope:** v1 — single user, single machine, single Obsidian vault, MCP into Claude Code.
**Status:** approved (brainstorm), ready for implementation planning.

## Goal

Give LLM workflows (Claude Code first) durable, low-latency recall over the
user's existing Obsidian vault. Editing a note makes its content searchable
within seconds; querying the brain from a Claude session returns hits in
under 200 ms. The brain can also stash mid-session findings into a quarantined
`inbox/` subfolder so they survive into future sessions.

This is the **C → core → A** loop from the decomposition:

```
notes folder ──► obsidian-brain ingest ──► slim-llm-memory ──► MCP server ──► Claude Code
```

D (HTTP/RAG endpoint) and B (Q&A shell) are explicitly out of scope for v1.
The chosen architecture leaves them as thin shells over the same core later.

## Non-goals (v1)

- Multi-user / multi-machine sync.
- Graph queries over `[[backlinks]]` (data is captured in `meta.links`, query
  layer comes in slim-llm-memory phase 4).
- LLM-generated tags / summaries / entity extraction during ingest.
- Editing existing curated notes from an LLM session.
- HTTP / OpenAI-compatible RAG endpoint.
- A standalone Q&A UI.

## Where it lives

A new subpackage inside the existing slim-llm-memory repo:

```
slim_llm_memory/
    apps/
        obsidian/
            __init__.py
            parser.py     # Obsidian markdown → Chunk list
            chunker.py    # adaptive chunking strategy
            watcher.py    # watchdog Observer + debounce
            ingest.py     # ingest CLI: full sweep + watch mode
            spool.py      # JSONL spool reader/writer
            mcp_server.py # stdio MCP server, six tools
            cli.py        # entry point: `slim-llm-obsidian ingest|mcp|stats`
```

Optional install extra: `pip install slim-llm-memory[obsidian]` pulls
`watchdog`, `pyyaml`, and `mcp` (the official Python SDK). The core lib's
"~1000 LOC, two hard deps" pitch is preserved — these deps are opt-in.

Console-script entry point in `pyproject.toml`:

```toml
[project.scripts]
slim-llm-obsidian = "slim_llm_memory.apps.obsidian.cli:main"
```

## Architecture

Two processes, one writer to the index. Decoupled via an on-disk JSONL spool.

```
┌────────────────────────────────────────────────────────────────────┐
│   ingest process                  │   mcp server process            │
│   (long-lived in --watch mode,    │   (long-lived, spawned by      │
│    or one-shot full-sweep)        │    Claude Code's MCP config)   │
│                                   │                                 │
│   ┌────────────┐                  │   ┌────────────────┐            │
│   │ watchdog   │──fs events──▶    │   │ MCP stdio loop │            │
│   │ + debounce │                  │   │                │            │
│   └─────┬──────┘                  │   └─────┬──────────┘            │
│         │                         │         │                       │
│         ▼                         │         │ every 5s              │
│   ┌────────────┐    writes ┌──────┴────┐    │ drain                 │
│   │ parser +   │──────────▶│  spool/   │◀───┘                       │
│   │ chunker    │  jsonl    │  *.jsonl  │                            │
│   └────────────┘           └──────┬────┘                            │
│                                   │      ┌─────────────────────┐    │
│                                   └─────▶│ Memory.upsert(...)  │    │
│                                          │ slim-llm-memory     │    │
│                                          │ index dir (single   │    │
│                                          │ fcntl writer = mcp) │    │
│                                          └─────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

**Why two processes, not one thread:** the MCP server's lifecycle is owned by
Claude Code (started/stopped on session). The watcher should run independently
of any Claude session — you want notes you save while no MCP server is running
to still be in the spool when one starts. The spool also gives us a crash-safe
queue: a `mcp` crash mid-upsert leaves spool files on disk; restart re-drains.

**Why single-writer to the index:** slim-llm-memory enforces `fcntl.flock`
per index directory. Routing all writes through `mcp` removes any lock fight.
Tradeoff: ~5 s extra ingest latency. Acceptable for the goal ("smarter Claude
in the next turn", not "indexed before keystroke").

**Filesystem layout:**

```
~/Vault/                          # the user's existing Obsidian vault
    Daily/  Projects/  People/ ...
    inbox/                        # quarantined LLM captures (only writer: mcp)
~/.obsidian-brain/
    config.toml                   # vault path, embedder, watch flag
    index/                        # slim-llm-memory store
        items.vN.jsonl  vectors.vN.npy  manifest.json  .lock
    spool/                        # pending upserts
        2026-05-20T12-00-00Z-{ulid}.jsonl
    logs/
        ingest.log  mcp.log
```

## Components

### parser.py

```python
@dataclass
class Chunk:
    id: str                   # f"{rel_path}#{section_idx}"
    text: str                 # the chunk body (markdown stripped of yaml frontmatter)
    meta: dict                # see below

def parse_file(vault_root: Path, abs_path: Path) -> list[Chunk]: ...
def parse_file_deleted(vault_root: Path, abs_path: Path) -> list[str]: ...
    # returns ids that should be removed
```

`meta` is the structured layer:

| key | value | source |
|---|---|---|
| `path` | vault-relative, posix-style | filesystem |
| `title` | str | YAML frontmatter `title` ⤳ first H1 ⤳ filename stem |
| `kind` | str | top-level folder (`Daily`, `Projects`, …); `inbox` for captures |
| `tags` | list[str] | YAML `tags` ∪ inline `#tag` (after stripping code blocks) |
| `links` | list[str] | `[[Note]]` targets, resolved to vault-relative path if possible |
| `heading_path` | list[str] | for H2-split chunks, the chain of headings down to this section |
| `section_idx` | int | 0 for "whole file", else 1..N for H2 sections / sliding windows |
| `mtime` | float | file mtime at parse time |
| `source` | `"vault"` \| `"inbox"` | which subtree this came from |

Edge cases the parser handles:
- YAML frontmatter that's invalid → parsed body unchanged, warning logged.
- `#tag` inside fenced code blocks → ignored (don't treat code as tag source).
- `[[Note]]` with section anchors (`[[Note#Heading]]`) → target normalised to `Note`.
- Files with no H1 and no frontmatter title → filename stem wins.

### chunker.py

```python
def chunk(text: str, *, max_tokens: int = 800, window: int = 400, overlap: int = 50) -> list[ChunkSlice]
```

Algorithm:

1. If `count_tokens(text) <= max_tokens` → one slice, `section_idx=0`.
2. Else split on `^##\s` boundaries → one slice per H2 section, `section_idx=1..N`, `heading_path=[file_title, h2]`.
3. If any resulting slice is still `> max_tokens` **or** the file has no H2s
   → fall back to sliding window: `window=400`, `overlap=50` tokens, `heading_path=[file_title]`.

Token counting: cheap whitespace tokenisation for v1 (off by ~15% vs.
real BPE, but fine for boundary decisions). Real `tiktoken` is a fat dep;
keep it out of the hot path.

### watcher.py

`watchdog.Observer` over `vault_root`, ignoring `.obsidian/`, `.git/`, and
hidden files. Events get debounced 2 s per path (a "save" often fires
multiple times). On debounce expiry: call `parser.parse_file` + write to spool.

Deletions: emit a spool entry of shape `{"op": "remove", "ids": [...]}`.

Renames: parser is keyed on path; rename emits a remove of the old path
followed by a normal upsert of the new path. Vector recomputation is wasted
on rename-only edits, but renames are rare.

### ingest.py + cli.py

```
slim-llm-obsidian ingest [--vault PATH] [--watch]
slim-llm-obsidian mcp    [--vault PATH]
slim-llm-obsidian stats  [--vault PATH]
```

- `ingest` (no flags): one-shot full sweep — walk the vault, emit spool
  entries for every file, exit. slim-llm-memory's own hash-skip on the
  mcp side ensures unchanged chunks don't re-embed.
- `ingest --watch`: full sweep first, then stay alive as a watchdog.
- `mcp`: starts the stdio MCP server. Always begins by draining any
  pending spool entries, then enters the stdio loop with a 5 s background
  drain task.
- `stats`: prints `Memory.stats()` plus spool depth + last-mtime, useful
  for the user's own /health-style checking.

Config: `~/.obsidian-brain/config.toml`:

```toml
vault = "~/Vault"
embedder = "ollama:nomic-embed-text"
embedder_url = "http://localhost:11434"
spool_drain_seconds = 5
flush_every_changes = 100
flush_every_seconds = 30
watcher_debounce_seconds = 2.0
log_level = "INFO"
```

### spool.py

JSONL files in `~/.obsidian-brain/spool/`. One file per debounce-flush
from the watcher (avoids contention on a single file). Schema per line:

```json
{"op": "upsert", "id": "...", "text": "...", "hash": "...", "meta": {...}}
{"op": "remove", "ids": ["..."]}
```

Drain protocol:
1. List `*.jsonl` files sorted by name (ULID prefix = chronological).
2. For each file: read lines, group by op, call `Memory.upsert` and
   `Memory.remove` in batches.
3. After successful batch, **rename** the file to `*.done` (not delete) and
   sweep `*.done` files older than 24 h periodically. Rename gives us a quick
   replay path if upsert silently failed somewhere.
4. Malformed JSON lines: log + skip + continue, don't fail the file.

### mcp_server.py

Uses the official `mcp` Python SDK with stdio transport. Six tools:

```python
@server.tool()
def search(query: str, k: int = 8,
           kinds: list[str] | None = None,
           min_score: float = 0.3) -> list[dict]: ...

@server.tool()
def get(path: str) -> dict: ...
    # returns {"path": ..., "title": ..., "text": full_file_text, "meta": ...}

@server.tool()
def related(path_or_id: str, k: int = 5) -> list[dict]: ...
    # If `path_or_id` is a full chunk id (contains `#`), use that vector.
    # If it's a vault-relative path: resolve to the file's first chunk id
    # (`path#0` for whole-file items, else `path#1` for H2-split items).
    # Either way: zero embedder calls — one numpy GEMV against the stored
    # vector.

@server.tool()
def by_tag(tags: list[str], k: int = 20) -> list[dict]: ...
    # in-memory filter over Memory.store.items, ts-desc

@server.tool()
def recent(n: int = 10, kind: str | None = None) -> list[dict]: ...

@server.tool()
def remember(text: str, tags: list[str] | None = None,
             title: str | None = None) -> dict: ...
    # writes ~/Vault/inbox/{ulid}-{slug}.md with YAML frontmatter
    # returns {"path": ..., "id": ..., "ingested": False}
```

`remember` only ever writes to `vault/inbox/`. Hard-coded; not configurable.
The path is validated to be inside the vault and inside the `inbox/` subtree.
If `vault/inbox/` doesn't exist, the server creates it on startup.

`related` exploits an underused property of slim-llm-memory: stored vectors
are already normalised, so neighbour search of an existing item is one
matrix-vector product with no embedder call. We add a small helper
`Memory.neighbours(item_id, k)` that does this directly.

`by_tag` and `recent` need a few-ms scan over `Memory.store.items`. Acceptable
at 50k items; if it ever bites, we add a tag-index and a ts-index as parallel
numpy arrays maintained alongside items (the same layout fix flagged for
search-mask vectorisation in the broader perf discussion).

## Data flow

**Edit → searchable.**

1. User saves `~/Vault/Projects/foo.md`.
2. Watchdog fires; debounce 2 s; `parser.parse_file` produces N chunks.
3. All chunks appended to a new `spool/{ts}-{ulid}.jsonl`.
4. Within 5 s, mcp server drains the spool, calls `Memory.upsert(chunks)`.
   slim-llm-memory's hash-skip dedupes — unchanged chunks don't re-embed.
   No watcher-side cache; one source of truth.
5. `Memory.flush()` is triggered every 30 s **or** after >100 changes,
   whichever first.

**p95 save-to-searchable ≈ 5–7 s.**

**Search.**

1. Claude Code calls `search("how do I undo a git commit", k=5)` via MCP.
2. mcp server calls `Memory.search(query, k, kinds, min_score)`.
3. Ollama embeds the query (~30 ms on CPU), numpy scans (<5 ms at 50k chunks).
4. Top-k Hits serialised back over stdio.

**p95 ≈ 80 ms**, well below the 200 ms bar.

**Capture (`remember`).**

1. Claude Code calls `remember("noteworthy snippet", tags=["claude-session"])`.
2. mcp server writes `~/Vault/inbox/{ulid}-{slug}.md` with YAML:
   ```yaml
   ---
   tags: [claude-session]
   source: claude
   ts: 2026-05-20T13:45:00Z
   ---
   noteworthy snippet
   ```
3. Watchdog (running in the separate ingest process) sees the new file,
   parses it, pushes a spool entry.
4. Same ingest path → searchable in ~5 s.

If the ingest process isn't running, the file is still on disk and will be
caught on the next `slim-llm-obsidian ingest --full-sweep`.

## Error handling

| Failure | Behaviour |
|---|---|
| Ollama unreachable | `Memory.upsert` raises `EmbedderError`. mcp catches, leaves spool file unrenamed, retries every 30 s, single warning per failure cluster (not per attempt). |
| Spool line malformed | Log + skip the line. Don't fail the rest of the file. |
| Vault path missing | `ingest` and `mcp` fail fast on startup with a clear message. |
| Two `mcp` processes | Second one fails immediately on `Memory`'s fcntl lock. |
| `mcp` crash mid-upsert | `Memory.flush` is atomic; spool files persist on disk; restart re-drains. The `*.done` rename is the commit point on the spool side, parallel to the manifest swap on the index side. |
| Non-UTF-8 file | Skipped with a warning. |
| Watchdog event storm (e.g. git pull rewriting 1000 files) | Debounce coalesces per path; spool batches are large but bounded; flushes throttle to once per 30 s minimum. |
| Long chunk (>2048 tokens, embedder limit) | Hard-clipped at the embedder's max input length; warning logged with chunk id so user can investigate. |
| `remember` path traversal attempt (`text` containing relative paths in title) | Title is slugified (a-z0-9-, max 60 chars); resulting path is asserted to be inside `vault/inbox/`. |

## Testing

- **Unit:** parser (golden tests against a small fixture vault: dailies,
  long literature note, frontmatter quirks, code-fenced `#tag`, escaped
  `[[brackets]]`). Chunker (adaptive boundaries, no-H2 fallback). Spool
  drain (corrupt-line tolerance, rename-on-success).
- **Integration:** end-to-end with `Embedder.noop()` for hermeticity —
  write fixture file, expect `search` to find it, assert latency <200 ms,
  assert `remember` round-trips through ingest.
- **Manual smoke:** real vault + real Ollama, Claude Code with MCP config
  pointing at `slim-llm-obsidian mcp`. Exercise all six tools from a Claude
  session.

## Open questions deferred to v2

- HTTP / OpenAI-compatible RAG endpoint (D consumer).
- Standalone Q&A shell (B consumer).
- LLM enrichment during ingest (summary, entity extraction, auto-tags).
  Will require a hook between `parse_file` and the spool write; design
  not pre-baked.
- Graph layer over `meta.links` — captured but not queried.
- ANN swap (faiss / SQLite-vss) — slim-llm-memory's documented migration
  path; not v1 work.
- Background flush triggers tuned by index size.

## Performance budget summary

| Operation | Target | Method |
|---|---|---|
| Single-file ingest | <100 ms parse + <30 ms embed per chunk | local; deterministic |
| Save → searchable (p95) | <7 s | 2 s debounce + 5 s drain + ε |
| `search` over 50k chunks (p95) | <100 ms total | Ollama embed + numpy scan |
| `related` over 50k chunks (p95) | <10 ms | no embed; one GEMV |
| Full vault re-sweep, 5k files | <10 min cold, <30 s warm | hash-skip on warm path |
