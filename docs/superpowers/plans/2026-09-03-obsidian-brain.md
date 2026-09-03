# Obsidian Brain v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Claude Code durable, sub-200 ms recall over the user's Obsidian vault via an MCP server, with a watcher that makes edits searchable within seconds and a `remember` tool that stashes captures into `vault/inbox/`.

**Architecture:** Two processes decoupled by an on-disk JSONL spool. The *ingest* process (watchdog + debounce, or a one-shot sweep) parses markdown into chunks and appends them to `~/.obsidian-brain/spool/*.jsonl`. The *mcp* process is the single writer to the slim-llm-memory index: it drains the spool every 5 s, serves six tools over stdio, and flushes every 100 changes or 30 s. All MCP-agnostic logic lives in a `Brain` class so it is testable with `Embedder.noop()` and no network.

**Tech Stack:** Python 3.10+, slim-llm-memory `Memory` (numpy + httpx), `watchdog` 6, `pyyaml` 6, `mcp` 2.x (`MCPServer`, stdio transport), `tomllib` (stdlib). Tests: pytest, `pytest-asyncio` for the MCP adapter.

**Spec:** `docs/specs/2026-05-20-obsidian-brain-design.md`

## Global Constraints

- Core library keeps "numpy + httpx only" as hard deps. All new deps go behind the `obsidian` optional extra: `watchdog`, `pyyaml`, `mcp`.
- `requires-python = ">=3.10"`. Use `tomllib` (3.11+) with `tomli` fallback is NOT acceptable as a hard dep: use `tomllib` if available, else parse with a tiny `key = "value"` fallback. (See Task 2.)
- Single writer to the index: only the `mcp` process ever constructs `Memory`. `ingest` never imports `Memory`.
- `remember` only ever writes inside `vault/inbox/`. Hard-coded, path-asserted.
- In `mcp` mode **nothing** may be written to stdout (it is the MCP transport). Logging goes to stderr + `~/.obsidian-brain/logs/mcp.log`.
- No global state: everything hangs off a `Brain` instance.
- Tests: zero network, `Embedder.noop()` only. Full suite must stay under 10 s.
- Python env for running tests/notebooks: `/home/trbck/miniconda3/envs/trading/bin/python` (has pytest, jupyter, mcp 2.1.1, watchdog 6.0.0, pyyaml 6.0.3).
- Deviation from spec, locked in here: spool lines are **per file**, not per chunk (`{"op":"file","path":...,"chunks":[...]}` and `{"op":"remove","path":...}`), because the brain must delete stale chunk ids when a file shrinks from 5 H2 sections to 3, and deletion events carry no chunk count. `parse_file_deleted` is therefore not needed. Hash is computed by `Memory`, not carried in the spool.
- Commit after every task with the trailer:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2
  ```

---

## File structure

```
slim_llm_memory/
    store.py                    # MODIFY: lock leak fix, amortised append buffer
    index.py                    # MODIFY: threshold=1.0 fix, Memory.neighbours()
    apps/
        __init__.py             # CREATE (empty)
        obsidian/
            __init__.py         # CREATE: re-export Brain, Chunk
            config.py           # CREATE: Config dataclass, load_config, make_embedder
            chunker.py          # CREATE: adaptive chunking, pure functions
            parser.py           # CREATE: Obsidian markdown → list[Chunk]
            spool.py            # CREATE: Spool (write / pending / read / mark_done / sweep_done)
            brain.py            # CREATE: Brain (drain, flush policy, six operations)
            watcher.py          # CREATE: Debouncer + VaultEventHandler + run_watch
            ingest.py           # CREATE: sweep(vault, spool) full walk
            mcp_server.py       # CREATE: build_server(brain) → MCPServer, serve()
            cli.py              # CREATE: argparse: ingest | mcp | stats
tests/
    test_store.py               # MODIFY: add lock-release-on-failed-load, append perf
    test_index.py               # MODIFY: add threshold=1.0 + neighbours tests
    obsidian/
        __init__.py             # CREATE (empty)
        conftest.py             # CREATE: fixture vault copy + noop brain factory
        fixtures/vault/         # CREATE: small golden vault (see Task 4)
        test_config.py
        test_chunker.py
        test_parser.py
        test_spool.py
        test_brain.py
        test_watcher.py
        test_ingest.py
        test_mcp_server.py
        test_integration.py
notebooks/verify_memory.ipynb   # COMMIT (currently untracked); re-run after Task 0
pyproject.toml                  # MODIFY: [obsidian] extra + console script
README.md                       # MODIFY: Obsidian Brain section
.gitignore                      # MODIFY: .ipynb_checkpoints/
```

---

### Task 0: Fix the three known Memory issues and commit the verification notebook

**Files:**
- Modify: `slim_llm_memory/store.py` (`Store.__init__` ~line 128, `add_item` ~line 242, `flush` compaction ~line 295)
- Modify: `slim_llm_memory/index.py` (`find_duplicates` ~line 257)
- Modify: `tests/test_store.py`, `tests/test_index.py`
- Modify: `.gitignore`
- Commit: `notebooks/verify_memory.ipynb`

**Interfaces:**
- Consumes: existing `Store`, `Memory`.
- Produces: `Store.vectors` remains a `(N, dim)` float32 ndarray (now a view onto a larger capacity buffer `Store._buf`); everything else unchanged.

Background (from `notebooks/verify_memory.ipynb`, 2026-09-03):
1. `Store.__init__` acquires `.lock` before `_load_or_init()`; if loading raises, the fd leaks and the directory stays locked for the process lifetime.
2. `Store.add_item` does `np.vstack` per item → O(N²) bulk upsert (25 s for 20k items).
3. `find_duplicates(threshold=1.0)` never matches: float32 dot of identical unit vectors is 0.99999988.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py`:

```python
# ─── lock released when load fails ───────────────────────────────────────

def test_lock_released_when_load_fails(tmp_path: Path):
    with _make_store(tmp_path, name="model-A", dim=8) as s:
        s.add_item(Item(id="a", text="x", hash="h"), _vec(1))
        s.flush()
    with pytest.raises(StoreError, match="embedder mismatch"):
        Store(tmp_path, embedder_name="model-B", embedder_dim=8)
    # The failed constructor must not keep the directory locked.
    with _make_store(tmp_path, name="model-A", dim=8) as s:
        assert [it.id for it in s.items] == ["a"]


# ─── amortised append ─────────────────────────────────────────────────────

def test_append_uses_amortised_buffer(tmp_path: Path):
    # Mechanism test (timing is flaky at dim 8): vectors must be a view onto a
    # geometrically grown capacity buffer, and in-place updates write through.
    with _make_store(tmp_path) as s:
        for i in range(100):
            s.add_item(Item(id=str(i), text="t", hash="h"), _vec(i))
        assert s.vectors.shape == (100, 8)
        assert s._buf.shape[0] >= 100 and np.shares_memory(s.vectors, s._buf)
        s.add_item(Item(id="5", text="u", hash="h2"), _vec(999))
        assert np.allclose(s._buf[5], _vec(999))
        s.flush()
    with _make_store(tmp_path) as s:
        assert s.vectors.shape == (100, 8)
        assert np.allclose(s.vectors[5], _vec(999))
        assert np.allclose(s.vectors[7], _vec(7))
```

Append to `tests/test_index.py`:

```python
def test_find_duplicates_threshold_one_matches_identical(mem: Memory):
    mem.upsert([
        {"id": "a", "text": "identical text", "meta": {}},
        {"id": "b", "text": "identical text", "meta": {}},
        {"id": "c", "text": "something else", "meta": {}},
    ])
    clusters = mem.find_duplicates(threshold=1.0)
    assert sorted(clusters[0]) == ["a", "b"]
    assert len(clusters) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/test_store.py::test_lock_released_when_load_fails tests/test_store.py::test_append_uses_amortised_buffer tests/test_index.py::test_find_duplicates_threshold_one_matches_identical -v`
Expected: 3 FAIL (lock: `StoreError: another writer holds the lock`; append: `AttributeError: 'Store' object has no attribute '_buf'`; dedup: `IndexError` on `clusters[0]`).

- [ ] **Step 3: Fix the lock leak**

In `slim_llm_memory/store.py`, `Store.__init__`, replace the trailing `self._load_or_init()` with:

```python
        try:
            self._load_or_init()
        except BaseException:
            self._lock.release()
            raise
```

- [ ] **Step 4: Amortised append buffer**

In `Store.__init__`, replace `self.vectors: np.ndarray = np.zeros((0, self.embedder_dim), dtype=np.float32)` with:

```python
        self._buf: np.ndarray = np.zeros((0, self.embedder_dim), dtype=np.float32)
        self.vectors: np.ndarray = self._buf[:0]
```

Add a helper method on `Store` (place after `_load_or_init`):

```python
    def _set_vectors(self, vectors: np.ndarray) -> None:
        """Adopt ``vectors`` as the backing buffer; ``self.vectors`` is a view of the live rows."""
        self._buf = np.ascontiguousarray(vectors, dtype=np.float32)
        self.vectors = self._buf[: self._buf.shape[0]]
```

In `_load_or_init`, replace `self.vectors = vectors` with `self._set_vectors(vectors)`.

Replace the append branch of `add_item`:

```python
        else:
            n = self.vectors.shape[0]
            if n == self._buf.shape[0]:
                new_cap = max(16, self._buf.shape[0] * 2)
                grown = np.empty((new_cap, self.embedder_dim), dtype=np.float32)
                grown[:n] = self._buf[:n]
                self._buf = grown
            self._buf[n] = vector
            self.vectors = self._buf[: n + 1]
            self.items.append(item)
            self._id_to_idx[item.id] = len(self.items) - 1
```

In `flush`, replace the compaction block's vector rebuild:

```python
            keep_idx = [i for i, it in enumerate(self.items) if not it.deleted]
            new_items = [self.items[i] for i in keep_idx]
            new_vectors = (
                self.vectors[keep_idx]
                if keep_idx else np.zeros((0, self.embedder_dim), dtype=np.float32)
            )
            self.items = new_items
            self._set_vectors(new_vectors)
            self._id_to_idx = {it.id: i for i, it in enumerate(new_items)}
```

`np.save(vectors_p, self.vectors.astype(np.float32, copy=False))` stays: saving a view writes only the live rows.

- [ ] **Step 5: Fix threshold=1.0**

In `slim_llm_memory/index.py`, `find_duplicates`, after the threshold validation add:

```python
        # float32 dot of identical unit vectors lands at 0.99999988, so a
        # literal 1.0 would never match. Tolerate float32 rounding.
        eff_threshold = min(threshold, 1.0 - 1e-6)
```

and change `cols = np.where(row >= threshold)[0]` to `cols = np.where(row >= eff_threshold)[0]`.

- [ ] **Step 6: Run the full suite**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q`
Expected: 47 passed.

- [ ] **Step 7: Re-run the verification notebook**

Add `.ipynb_checkpoints/` to `.gitignore`. Then (needs Ollama running; takes a few minutes):

Run: `cd /home/trbck/workspace/slim-llm-memory && PYTHONPATH=. /home/trbck/miniconda3/envs/trading/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/verify_memory.ipynb 2>&1 | tail -5`
Expected: exits 0. Then `grep -c '⚠️' notebooks/verify_memory.ipynb` should drop (the three "known" checks flip to ✅). If Ollama is not reachable, skip the execute step, commit the notebook as-is, and note it in the commit body.

- [ ] **Step 8: Commit**

```bash
git add slim_llm_memory/store.py slim_llm_memory/index.py tests/test_store.py tests/test_index.py .gitignore notebooks/verify_memory.ipynb
git commit -m "fix: release lock on failed load, amortised vector append, threshold=1.0 dedup

Found by notebooks/verify_memory.ipynb (committed here).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 1: `Memory.neighbours(item_id, k)`

**Files:**
- Modify: `slim_llm_memory/index.py` (add method after `search`)
- Modify: `tests/test_index.py`
- Modify: `README.md` public API block (add one line)

**Interfaces:**
- Produces: `Memory.neighbours(item_id: str, k: int = 10, kinds: set[str] | None = None) -> list[Hit]`. Zero embedder calls. Excludes `item_id` itself. Returns `[]` for unknown/tombstoned id.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_index.py`:

```python
def test_neighbours_uses_stored_vector_without_embedding(mem: Memory):
    mem.upsert([
        {"id": "a", "text": "nginx tls setup", "meta": {"kind": "note"}},
        {"id": "b", "text": "nginx tls setup", "meta": {"kind": "note"}},   # identical → cosine 1
        {"id": "c", "text": "milch kaufen", "meta": {"kind": "shopping"}},
    ])
    calls_before = mem.stats()["counters"].get("embed.calls", 0)
    hits = mem.neighbours("a", k=5)
    assert [h.id for h in hits][0] == "b"
    assert all(h.id != "a" for h in hits)
    assert mem.stats()["counters"].get("embed.calls", 0) == calls_before
    # kinds filter applies
    assert [h.id for h in mem.neighbours("a", k=5, kinds={"shopping"})] == ["c"]
    # unknown id → empty
    assert mem.neighbours("nope") == []
    mem.remove("b")
    assert all(h.id != "b" for h in mem.neighbours("a", k=5))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/test_index.py::test_neighbours_uses_stored_vector_without_embedding -v`
Expected: FAIL `AttributeError: 'Memory' object has no attribute 'neighbours'`

- [ ] **Step 3: Refactor `search` so the ranking is shared, then add `neighbours`**

In `slim_llm_memory/index.py`, extract the body of `search` after the query vector is computed into a private method, and call it from both:

```python
    def _rank(self, qv: np.ndarray, k: int, kinds: set[str] | None,
              min_score: float, exclude: str | None = None) -> list[Hit]:
        """Top-k over the store by dot product with the unit vector ``qv``."""
        scores = self.store.vectors @ qv  # shape (N,)
        mask = np.ones(scores.shape[0], dtype=bool)
        for i, it in enumerate(self.store.items):
            if it.deleted or it.id == exclude:
                mask[i] = False
            elif kinds is not None and (it.meta.get("kind") not in kinds):
                mask[i] = False
        if min_score > 0:
            mask &= scores >= min_score
        valid = np.where(mask)[0]
        if valid.size == 0:
            return []
        kk = min(k, valid.size)
        candidate_scores = scores[valid]
        if kk < valid.size:
            top_idx = np.argpartition(-candidate_scores, kk - 1)[:kk]
        else:
            top_idx = np.arange(valid.size)
        top_idx = top_idx[np.argsort(-candidate_scores[top_idx])]
        chosen = valid[top_idx]
        hits: list[Hit] = []
        for i in chosen:
            it = self.store.items[int(i)]
            hits.append(Hit(id=it.id, score=float(scores[i]), text=it.text, meta=dict(it.meta)))
        return hits

    def neighbours(self, item_id: str, k: int = 10, kinds: set[str] | None = None) -> list[Hit]:
        """Nearest stored items to an existing item. No embedder call — one GEMV."""
        idx = self.store._id_to_idx.get(item_id)
        if idx is None or k < 1:
            return []
        t0 = time.time()
        hits = self._rank(self.store.vectors[idx], k, kinds, min_score=0.0, exclude=item_id)
        self.obs.record_slow("neighbours", (time.time() - t0) * 1000)
        self.obs.count("neighbours.calls")
        return hits
```

`search` keeps its guards and embedding, then becomes:

```python
        qv = _normalise(np.asarray(qv_raw[0], dtype=np.float32))
        hits = self._rank(qv, k, kinds, min_score)
        duration_ms = (time.time() - t0) * 1000
        self.obs.record_slow("search", duration_ms)
        self.obs.count("search.calls")
        return hits
```

Add `.neighbours(id, k, kinds)  → [Hit, ...]  (no embed call)` to the README public API block.

- [ ] **Step 4: Run tests**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q`
Expected: 48 passed.

- [ ] **Step 5: Commit**

```bash
git add slim_llm_memory/index.py tests/test_index.py README.md
git commit -m "feat: Memory.neighbours — nearest items to a stored id without an embed call

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 2: Package scaffold, `[obsidian]` extra, config loading

**Files:**
- Modify: `pyproject.toml`
- Create: `slim_llm_memory/apps/__init__.py`, `slim_llm_memory/apps/obsidian/__init__.py`, `slim_llm_memory/apps/obsidian/config.py`
- Create: `tests/obsidian/__init__.py`, `tests/obsidian/test_config.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class Config:
      vault: Path
      home: Path                       # ~/.obsidian-brain
      embedder: str = "ollama:nomic-embed-text"
      embedder_url: str = "http://localhost:11434"
      spool_drain_seconds: float = 5.0
      flush_every_changes: int = 100
      flush_every_seconds: float = 30.0
      watcher_debounce_seconds: float = 2.0
      log_level: str = "INFO"
      @property index_dir -> Path; spool_dir -> Path; logs_dir -> Path

  def load_config(path: Path | None = None, *, vault: str | None = None, home: Path | None = None) -> Config
  def make_embedder(spec: str, url: str) -> Embedder   # "ollama:<model>" | "noop:<dim>"
  ```

- [ ] **Step 1: pyproject**

Add under `[project.optional-dependencies]`:

```toml
obsidian = ["watchdog>=4", "pyyaml>=6", "mcp>=2"]
```

Change `dev` to `"slim-llm-memory[test,gemini,graph,anthropic,obsidian]", "ruff>=0.6"`.

Add after `[project.urls]`:

```toml
[project.scripts]
slim-llm-obsidian = "slim_llm_memory.apps.obsidian.cli:main"
```

Add to `[tool.pytest.ini_options]`: `asyncio_mode = "auto"`.

- [ ] **Step 2: Write the failing test**

`tests/obsidian/__init__.py` empty. `tests/obsidian/test_config.py`:

```python
from pathlib import Path

import pytest

from slim_llm_memory.apps.obsidian.config import Config, load_config, make_embedder


def test_defaults_when_no_file(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml", vault=str(tmp_path / "Vault"), home=tmp_path / "home")
    assert cfg.vault == (tmp_path / "Vault")
    assert cfg.embedder == "ollama:nomic-embed-text"
    assert cfg.spool_drain_seconds == 5.0
    assert cfg.index_dir == tmp_path / "home" / "index"
    assert cfg.spool_dir == tmp_path / "home" / "spool"
    assert cfg.logs_dir == tmp_path / "home" / "logs"


def test_file_values_and_cli_override(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        'vault = "~/Vault"\nembedder = "noop:64"\nspool_drain_seconds = 1\n'
        'flush_every_changes = 7\nlog_level = "DEBUG"\n',
        encoding="utf-8",
    )
    cfg = load_config(p, home=tmp_path)
    assert cfg.vault == Path("~/Vault").expanduser()
    assert cfg.embedder == "noop:64"
    assert cfg.spool_drain_seconds == 1.0
    assert cfg.flush_every_changes == 7
    assert cfg.log_level == "DEBUG"
    # explicit vault wins over file
    cfg2 = load_config(p, vault=str(tmp_path / "Other"), home=tmp_path)
    assert cfg2.vault == tmp_path / "Other"


def test_missing_vault_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="vault"):
        load_config(tmp_path / "missing.toml", home=tmp_path)


def test_make_embedder():
    e = make_embedder("noop:32", "http://unused")
    assert e.name == "noop:32" and e.dim == 32
    o = make_embedder("ollama:nomic-embed-text", "http://localhost:11434")
    assert o.name == "ollama:nomic-embed-text"
    with pytest.raises(ValueError):
        make_embedder("gemini:x", "")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_config.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'slim_llm_memory.apps'`

- [ ] **Step 4: Implement**

`slim_llm_memory/apps/__init__.py`: empty file.

`slim_llm_memory/apps/obsidian/__init__.py`:

```python
"""Obsidian Brain — vault ingest + MCP server on top of slim-llm-memory.

Optional extra: ``pip install slim-llm-memory[obsidian]``.
See docs/specs/2026-05-20-obsidian-brain-design.md.
"""
```

`slim_llm_memory/apps/obsidian/config.py`:

```python
"""Config for the Obsidian Brain: ``~/.obsidian-brain/config.toml`` + CLI overrides."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from slim_llm_memory import Embedder

DEFAULT_HOME = Path("~/.obsidian-brain")


@dataclass
class Config:
    vault: Path
    home: Path = field(default_factory=lambda: DEFAULT_HOME.expanduser())
    embedder: str = "ollama:nomic-embed-text"
    embedder_url: str = "http://localhost:11434"
    spool_drain_seconds: float = 5.0
    flush_every_changes: int = 100
    flush_every_seconds: float = 30.0
    watcher_debounce_seconds: float = 2.0
    log_level: str = "INFO"

    @property
    def index_dir(self) -> Path:
        return self.home / "index"

    @property
    def spool_dir(self) -> Path:
        return self.home / "spool"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"


_LINE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$')


def _parse_toml(text: str) -> dict[str, Any]:
    """tomllib when available (3.11+); otherwise a flat ``key = value`` subset."""
    try:
        import tomllib
        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if val.startswith('"') and val.endswith('"'):
            out[key] = val[1:-1]
        elif val in ("true", "false"):
            out[key] = val == "true"
        else:
            try:
                out[key] = int(val)
            except ValueError:
                out[key] = float(val)
    return out


def load_config(
    path: Path | None = None,
    *,
    vault: str | None = None,
    home: Path | None = None,
) -> Config:
    home = (home or DEFAULT_HOME).expanduser()
    path = path or (home / "config.toml")
    data: dict[str, Any] = {}
    if path.exists():
        data = _parse_toml(path.read_text(encoding="utf-8"))

    vault_str = vault or data.get("vault")
    if not vault_str:
        raise ValueError(
            f"no vault configured: pass --vault or set `vault = \"~/Vault\"` in {path}"
        )

    kwargs: dict[str, Any] = {"vault": Path(str(vault_str)).expanduser(), "home": home}
    for f in fields(Config):
        if f.name in ("vault", "home") or f.name not in data:
            continue
        kwargs[f.name] = _cast(f.name, data[f.name])
    return Config(**kwargs)


_FLOAT_FIELDS = {"spool_drain_seconds", "flush_every_seconds", "watcher_debounce_seconds"}
_INT_FIELDS = {"flush_every_changes"}


def _cast(name: str, value: Any) -> Any:
    if name in _FLOAT_FIELDS:
        return float(value)
    if name in _INT_FIELDS:
        return int(value)
    return str(value)


def make_embedder(spec: str, url: str) -> Embedder:
    """``"ollama:<model>"`` → Ollama; ``"noop:<dim>"`` → deterministic test embedder."""
    kind, _, arg = spec.partition(":")
    if kind == "ollama" and arg:
        return Embedder.ollama(arg, base_url=url)
    if kind == "noop":
        return Embedder.noop(int(arg or 384))
    raise ValueError(f"unsupported embedder spec {spec!r} (use 'ollama:<model>' or 'noop:<dim>')")
```

- [ ] **Step 5: Run tests**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml slim_llm_memory/apps tests/obsidian
git commit -m "feat(obsidian): package scaffold, [obsidian] extra, config loading

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 3: Chunker

**Files:**
- Create: `slim_llm_memory/apps/obsidian/chunker.py`
- Create: `tests/obsidian/test_chunker.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class ChunkSlice:
      text: str
      section_idx: int          # 0 whole file; 1..N otherwise
      heading: str | None       # H2 text for H2 splits; None for whole-file / windows

  def count_tokens(text: str) -> int                 # whitespace tokens
  def chunk(text: str, *, max_tokens: int = 800, window: int = 400, overlap: int = 50) -> list[ChunkSlice]
  ```

- [ ] **Step 1: Write the failing tests**

`tests/obsidian/test_chunker.py`:

```python
from slim_llm_memory.apps.obsidian.chunker import ChunkSlice, chunk, count_tokens


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_is_one_slice():
    out = chunk("hello world", max_tokens=800)
    assert out == [ChunkSlice(text="hello world", section_idx=0, heading=None)]


def test_count_tokens_is_whitespace():
    assert count_tokens("a  b\nc\t d") == 4
    assert count_tokens("") == 0


def test_splits_on_h2_when_too_long():
    text = "intro line\n\n## Alpha\n" + words(300, "a") + "\n\n## Beta\n" + words(300, "b")
    out = chunk(text, max_tokens=500)
    assert [s.section_idx for s in out] == [1, 2, 3]
    assert out[0].heading is None and out[0].text == "intro line"
    assert out[1].heading == "Alpha" and out[1].text.startswith("## Alpha\n")
    assert out[2].heading == "Beta" and "b299" in out[2].text


def test_h2_split_skips_empty_preamble():
    text = "## Alpha\n" + words(300, "a") + "\n## Beta\n" + words(300, "b")
    out = chunk(text, max_tokens=500)
    assert [s.heading for s in out] == ["Alpha", "Beta"]
    assert [s.section_idx for s in out] == [1, 2]


def test_no_h2_falls_back_to_sliding_window():
    text = words(1000)
    out = chunk(text, max_tokens=800, window=400, overlap=50)
    assert all(s.heading is None for s in out)
    assert [s.section_idx for s in out] == [1, 2, 3]
    # windows step by window-overlap = 350; last window contains the final token.
    assert out[0].text.startswith("w0 ") and "w399" in out[0].text
    assert out[1].text.startswith("w350 ")
    assert out[-1].text.endswith("w999")
    assert all(count_tokens(s.text) <= 400 for s in out)


def test_oversized_h2_section_falls_back_to_window():
    text = "## Big\n" + words(1200) + "\n## Small\nx y z"
    out = chunk(text, max_tokens=800, window=400, overlap=50)
    assert all(s.heading is None for s in out)
    assert len(out) > 2


def test_h3_does_not_split():
    # Both H2 sections stay under max_tokens so the H2 path is taken; the H3
    # must not create a third slice.
    text = "## A\n" + words(300, "a") + "\n### sub\n" + words(100, "s") + "\n## B\n" + words(300, "b")
    out = chunk(text, max_tokens=500)
    assert [s.heading for s in out] == ["A", "B"]
    assert [s.section_idx for s in out] == [1, 2]
    assert "### sub" in out[0].text and "s99" in out[0].text


def test_window_preserves_original_whitespace():
    text = "\n".join(f"line{i}" for i in range(1000))
    out = chunk(text, max_tokens=800, window=400, overlap=50)
    assert "\n" in out[0].text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_chunker.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`slim_llm_memory/apps/obsidian/chunker.py`:

```python
"""Adaptive chunking: whole file → H2 sections → sliding token windows.

Token counting is whitespace tokenisation (≈15% off vs. BPE, fine for
boundary decisions; keeps tiktoken out of the hot path).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"\S+")
_H2 = re.compile(r"^## +(.*?)\s*$", re.MULTILINE)


@dataclass
class ChunkSlice:
    text: str
    section_idx: int
    heading: str | None


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _h2_sections(text: str) -> list[tuple[str | None, str]]:
    """Split on ``^## `` lines. Returns (heading, section_text) pairs; the
    preamble (heading None) is included only when non-blank."""
    matches = list(_H2.finditer(text))
    if not matches:
        return []
    out: list[tuple[str | None, str]] = []
    pre = text[: matches[0].start()].strip()
    if pre:
        out.append((None, pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), text[m.start():end].strip()))
    return out


def _windows(text: str, window: int, overlap: int) -> list[str]:
    spans = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    if not spans:
        return []
    step = max(1, window - overlap)
    out: list[str] = []
    start = 0
    while True:
        stop = min(start + window, len(spans))
        out.append(text[spans[start][0]: spans[stop - 1][1]])
        if stop >= len(spans):
            break
        start += step
    return out


def chunk(text: str, *, max_tokens: int = 800, window: int = 400, overlap: int = 50) -> list[ChunkSlice]:
    if count_tokens(text) <= max_tokens:
        return [ChunkSlice(text=text, section_idx=0, heading=None)]

    sections = _h2_sections(text)
    if sections and all(count_tokens(body) <= max_tokens for _, body in sections):
        return [
            ChunkSlice(text=body, section_idx=i, heading=heading)
            for i, (heading, body) in enumerate(sections, start=1)
        ]

    return [
        ChunkSlice(text=w, section_idx=i, heading=None)
        for i, w in enumerate(_windows(text, window, overlap), start=1)
    ]
```

- [ ] **Step 4: Run tests**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_chunker.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add slim_llm_memory/apps/obsidian/chunker.py tests/obsidian/test_chunker.py
git commit -m "feat(obsidian): adaptive chunker (whole → H2 → sliding window)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 4: Parser + fixture vault

**Files:**
- Create: `slim_llm_memory/apps/obsidian/parser.py`
- Create: `tests/obsidian/fixtures/vault/…` (below)
- Create: `tests/obsidian/conftest.py`, `tests/obsidian/test_parser.py`

**Interfaces:**
- Consumes: `chunker.chunk`, `chunker.ChunkSlice`.
- Produces:
  ```python
  @dataclass
  class Chunk:
      id: str        # f"{rel_path}#{section_idx}"
      text: str
      meta: dict     # path,title,kind,tags,links,heading_path,section_idx,mtime,source

  IGNORED_DIRS = {".obsidian", ".git", ".trash"}
  def is_vault_markdown(vault_root: Path, abs_path: Path) -> bool
  def rel_path(vault_root: Path, abs_path: Path) -> str            # posix, vault-relative
  def parse_text(vault_root: Path, rel: str, raw: str, mtime: float) -> list[Chunk]
  def parse_file(vault_root: Path, abs_path: Path) -> list[Chunk]  # [] + warning on non-UTF-8
  ```

- [ ] **Step 1: Create the fixture vault**

Create these files (exact content matters for the golden tests):

`tests/obsidian/fixtures/vault/Daily/2026-05-20.md`:
```markdown
Woke up early. Worked on #obsidian-brain and pinged [[People/Alice]].
Todo: read [[Long note#Section B]] again.
```

`tests/obsidian/fixtures/vault/People/Alice.md`:
```markdown
---
title: Alice Example
tags: [person, colleague]
---
# Alice

Works on infra. Knows nginx. #person
```

`tests/obsidian/fixtures/vault/Projects/Long note.md`: generate it exactly with this command (frontmatter, H1 + intro, then three H2 sections of 350 tokens each, so the file is ~1060 tokens and every section is under 800):

```bash
/home/trbck/miniconda3/envs/trading/bin/python - <<'EOF'
from pathlib import Path
p = Path("tests/obsidian/fixtures/vault/Projects/Long note.md")
p.parent.mkdir(parents=True, exist_ok=True)
parts = ["---\ntags:\n  - literature\n---\n# Long literature note\n\nIntro paragraph mentioning [[Alice]].\n"]
for name, prefix in [("Section A", "a"), ("Section B", "b"), ("Section C", "c")]:
    parts.append(f"\n## {name}\n\n" + " ".join(f"{prefix}{i}" for i in range(350)) + "\n")
p.write_text("".join(parts), encoding="utf-8")
EOF
```

`tests/obsidian/fixtures/vault/Projects/Quirks.md` (note the fenced python block is part of the file content):

````markdown
---
this is: [not: valid yaml
---
# Quirks

Inline #real-tag here but not this one:

```python
# not-a-tag in code
x = "#also-not"
```

Escaped \[[not a link]] and a real [[People/Alice|Alice]] link.
Heading-like ## in the middle of a line is fine. URL http://x.y/#frag stays.
````

`tests/obsidian/fixtures/vault/notitle.md`:
```markdown
Just a body with no title and no headings.
```

`tests/obsidian/fixtures/vault/.obsidian/workspace.json`: `{}`

`tests/obsidian/fixtures/vault/inbox/01J-capture.md`:
```markdown
---
tags: [claude-session]
source: claude
ts: 2026-05-20T13:45:00Z
---
noteworthy snippet
```

`tests/obsidian/conftest.py`:

```python
import shutil
from pathlib import Path

import pytest

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A writable copy of the fixture vault."""
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    return dst
```

- [ ] **Step 2: Write the failing tests**

`tests/obsidian/test_parser.py`:

```python
from pathlib import Path

from slim_llm_memory.apps.obsidian.parser import (
    Chunk, is_vault_markdown, parse_file, parse_text, rel_path,
)


def ids(chunks: list[Chunk]) -> list[str]:
    return [c.id for c in chunks]


def test_frontmatter_title_tags_and_kind(vault: Path):
    [c] = parse_file(vault, vault / "People" / "Alice.md")
    assert c.id == "People/Alice.md#0"
    assert c.meta["title"] == "Alice Example"
    assert c.meta["kind"] == "People"
    assert c.meta["source"] == "vault"
    assert sorted(c.meta["tags"]) == ["colleague", "person"]   # yaml ∪ inline, deduped
    assert c.meta["section_idx"] == 0
    assert c.meta["heading_path"] == ["Alice Example"]
    assert c.text.startswith("# Alice")          # frontmatter stripped
    assert isinstance(c.meta["mtime"], float)


def test_h1_title_and_links_resolved(vault: Path):
    [c] = parse_file(vault, vault / "Daily" / "2026-05-20.md")
    assert c.meta["title"] == "2026-05-20"       # no fm, no H1 → stem
    assert c.meta["kind"] == "Daily"
    assert c.meta["tags"] == ["obsidian-brain"]
    # [[People/Alice]] resolves to an existing file; [[Long note#Section B]] → anchor dropped,
    # resolved by stem search to Projects/Long note.md
    assert c.meta["links"] == ["People/Alice.md", "Projects/Long note.md"]


def test_long_note_splits_on_h2(vault: Path):
    chunks = parse_file(vault, vault / "Projects" / "Long note.md")
    assert ids(chunks) == [f"Projects/Long note.md#{i}" for i in range(1, 5)]
    assert chunks[0].meta["heading_path"] == ["Long literature note"]
    assert chunks[1].meta["heading_path"] == ["Long literature note", "Section A"]
    assert chunks[3].meta["heading_path"] == ["Long literature note", "Section C"]
    for c in chunks:
        assert c.meta["title"] == "Long literature note"
        assert c.meta["tags"] == ["literature"]
        assert c.meta["links"] == ["People/Alice.md"]   # file-level links on every chunk


def test_quirks_invalid_yaml_code_fence_escaped_links(vault: Path):
    [c] = parse_file(vault, vault / "Projects" / "Quirks.md")
    assert c.meta["title"] == "Quirks"
    assert c.text.startswith("---\nthis is:")          # invalid fm → body unchanged
    assert c.meta["tags"] == ["real-tag"]
    assert c.meta["links"] == ["People/Alice.md"]


def test_no_title_uses_stem_and_root_kind(vault: Path):
    [c] = parse_file(vault, vault / "notitle.md")
    assert c.meta["title"] == "notitle"
    assert c.meta["kind"] == "root"


def test_inbox_kind_and_source(vault: Path):
    [c] = parse_file(vault, vault / "inbox" / "01J-capture.md")
    assert c.meta["kind"] == "inbox" and c.meta["source"] == "inbox"
    assert c.meta["tags"] == ["claude-session"]


def test_is_vault_markdown_filters(vault: Path):
    assert is_vault_markdown(vault, vault / "Daily" / "2026-05-20.md")
    assert not is_vault_markdown(vault, vault / ".obsidian" / "workspace.json")
    assert not is_vault_markdown(vault, vault / ".obsidian" / "x.md")
    assert not is_vault_markdown(vault, vault / "Daily" / "img.png")
    assert not is_vault_markdown(vault, vault.parent / "outside.md")
    assert rel_path(vault, vault / "Daily" / "2026-05-20.md") == "Daily/2026-05-20.md"


def test_non_utf8_is_skipped(vault: Path):
    bad = vault / "bad.md"
    bad.write_bytes(b"\xff\xfe not utf8")
    assert parse_file(vault, bad) == []


def test_parse_text_empty_body_yields_no_chunks(vault: Path):
    assert parse_text(vault, "Daily/empty.md", "---\ntags: [x]\n---\n\n", 0.0) == []


def test_crlf_body_is_normalised(vault: Path):
    raw = "---\r\ntitle: X\r\n---\r\nline one\r\nline two\r\n"
    [c] = parse_text(vault, "Daily/crlf.md", raw, 0.0)
    assert "\r" not in c.text
    assert c.text == "line one\nline two"
    assert c.meta["title"] == "X"


def test_link_resolution_uses_one_vault_walk(vault: Path, monkeypatch):
    import pathlib
    from slim_llm_memory.apps.obsidian import parser as parser_mod
    parser_mod._stem_index_cache.clear()
    calls = {"n": 0}
    original = pathlib.Path.rglob

    def counting_rglob(self, pattern):
        calls["n"] += 1
        return original(self, pattern)

    monkeypatch.setattr(pathlib.Path, "rglob", counting_rglob)
    for _ in range(2):
        [c] = parse_file(vault, vault / "Daily" / "2026-05-20.md")
        assert c.meta["links"] == ["People/Alice.md", "Projects/Long note.md"]
    assert calls["n"] == 1   # one walk builds the index; the second parse hits the cache
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_parser.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Implement**

`slim_llm_memory/apps/obsidian/parser.py`:

```python
"""Obsidian markdown → ``Chunk`` list.

Structured metadata (title, kind, tags, links, heading_path, …) is the
same on every chunk of a file; only ``text``, ``section_idx`` and
``heading_path`` vary per chunk.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import chunk as _chunk

logger = logging.getLogger(__name__)

IGNORED_DIRS = {".obsidian", ".git", ".trash"}

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_TAG = re.compile(r"(?<![\w/#&])#([A-Za-z][\w/-]*)")
_WIKILINK = re.compile(r"(?<!\\)\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_H1 = re.compile(r"^# +(.+?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict[str, Any]


# ─── path helpers ─────────────────────────────────────────────────────────

def rel_path(vault_root: Path, abs_path: Path) -> str:
    return abs_path.resolve().relative_to(vault_root.resolve()).as_posix()


def is_vault_markdown(vault_root: Path, abs_path: Path) -> bool:
    try:
        rel = abs_path.resolve().relative_to(vault_root.resolve())
    except ValueError:
        return False
    if abs_path.suffix.lower() != ".md":
        return False
    return not any(part in IGNORED_DIRS or part.startswith(".") for part in rel.parts[:-1])


# ─── frontmatter / tags / links ───────────────────────────────────────────

def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    try:
        import yaml
        data = yaml.safe_load(m.group(1))
    except Exception as exc:  # invalid yaml → body unchanged
        logger.warning("invalid frontmatter ignored: %s", exc)
        return {}, raw
    if not isinstance(data, dict):
        return {}, raw
    return data, raw[m.end():]


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip().lstrip("#") for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(t).strip().lstrip("#") for t in value if str(t).strip()]
    return [str(value)]


_STEM_INDEX_TTL = 30.0
_stem_index_cache: dict[Path, tuple[float, dict[str, list[str]]]] = {}


def _stem_index(vault_root: Path) -> dict[str, list[str]]:
    """{file stem: [vault-relative paths]} for every vault note; one rglob, cached 30 s.

    Without this, every unresolved wikilink would walk the whole vault (O(links × files)
    on a full sweep). Links to notes created within the TTL stay unresolved until the
    file is next parsed — acceptable for watch mode."""
    root = vault_root.resolve()
    now = time.monotonic()
    cached = _stem_index_cache.get(root)
    if cached and now - cached[0] < _STEM_INDEX_TTL:
        return cached[1]
    index: dict[str, list[str]] = {}
    for p in root.rglob("*.md"):
        if is_vault_markdown(root, p):
            index.setdefault(p.stem, []).append(rel_path(root, p))
    _stem_index_cache[root] = (now, index)
    return index


def _resolve_link(vault_root: Path, target: str) -> str:
    """``[[People/Alice]]`` → ``People/Alice.md`` if that file exists; else look up
    ``<stem>.md`` in the cached vault index; else the raw target."""
    target = target.strip()
    direct = vault_root / f"{target}.md"
    if direct.is_file():
        return rel_path(vault_root, direct)
    cands = _stem_index(vault_root).get(Path(target).name)
    return sorted(cands)[0] if cands else target


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ─── main entry points ────────────────────────────────────────────────────

def parse_text(vault_root: Path, rel: str, raw: str, mtime: float) -> list[Chunk]:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")   # CRLF files must not leak \r into chunks
    fm, body = _split_frontmatter(raw)
    body = body.strip("\n")
    if not body.strip():
        return []

    stem = Path(rel).stem
    title = str(fm.get("title") or "").strip()
    if not title:
        h1 = _H1.search(body)
        title = h1.group(1).strip() if h1 else stem

    parts = rel.split("/")
    top = parts[0] if len(parts) > 1 else "root"
    source = "inbox" if top == "inbox" else "vault"

    scan = _FENCE.sub("", body)
    tags = _dedupe(_as_str_list(fm.get("tags")) + _INLINE_TAG.findall(scan))
    links = _dedupe([_resolve_link(vault_root, t) for t in _WIKILINK.findall(scan)])

    base_meta = {
        "path": rel, "title": title, "kind": top, "tags": tags, "links": links,
        "mtime": float(mtime), "source": source,
    }
    chunks: list[Chunk] = []
    for s in _chunk(body):
        heading_path = [title] + ([s.heading] if s.heading else [])
        meta = dict(base_meta, heading_path=heading_path, section_idx=s.section_idx)
        chunks.append(Chunk(id=f"{rel}#{s.section_idx}", text=s.text, meta=meta))
    return chunks


def parse_file(vault_root: Path, abs_path: Path) -> list[Chunk]:
    try:
        raw = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("skipping non-UTF-8 file: %s", abs_path)
        return []
    except FileNotFoundError:
        return []
    return parse_text(vault_root, rel_path(vault_root, abs_path), raw, abs_path.stat().st_mtime)
```

- [ ] **Step 5: Run tests**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_parser.py -v`
Expected: 9 passed. If `test_h1_title_and_links_resolved` link order differs, the fixture text order is `[[People/Alice]]` then `[[Long note#Section B]]`, so the expected list is correct; fix the regex, not the test.

- [ ] **Step 6: Commit**

```bash
git add slim_llm_memory/apps/obsidian/parser.py tests/obsidian
git commit -m "feat(obsidian): markdown parser with frontmatter, tags, wikilinks, fixture vault

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 5: Spool

**Files:**
- Create: `slim_llm_memory/apps/obsidian/spool.py`
- Create: `tests/obsidian/test_spool.py`

**Interfaces:**
- Produces:
  ```python
  def file_entry(path: str, chunks: list[Chunk]) -> dict   # {"op":"file","path":..,"chunks":[{"id","text","meta"},..]}
  def remove_entry(path: str) -> dict                      # {"op":"remove","path":..}

  class Spool:
      def __init__(self, directory: Path) -> None          # mkdir -p
      def write(self, entries: list[dict]) -> Path | None  # None when entries empty
      def pending(self) -> list[Path]                      # *.jsonl sorted by name
      def read(self, path: Path) -> list[dict]             # malformed lines logged + skipped
      def mark_done(self, path: Path) -> Path              # rename → .done
      def sweep_done(self, max_age_seconds: float = 86400) -> int
      def depth(self) -> int                               # number of pending files
  ```
  File names: `{UTC %Y%m%dT%H%M%S%f}Z-{8 hex}.jsonl` — lexicographic == chronological.

- [ ] **Step 1: Write the failing tests**

`tests/obsidian/test_spool.py`:

```python
import os
import time
from pathlib import Path

from slim_llm_memory.apps.obsidian.parser import Chunk
from slim_llm_memory.apps.obsidian.spool import Spool, file_entry, remove_entry


def test_write_read_roundtrip_in_order(tmp_path: Path):
    sp = Spool(tmp_path / "spool")
    e1 = file_entry("a.md", [Chunk(id="a.md#0", text="hi", meta={"path": "a.md"})])
    e2 = remove_entry("b.md")
    p1 = sp.write([e1])
    p2 = sp.write([e2])
    assert sp.write([]) is None
    assert sp.pending() == [p1, p2]
    assert sp.depth() == 2
    assert sp.read(p1) == [{"op": "file", "path": "a.md",
                            "chunks": [{"id": "a.md#0", "text": "hi", "meta": {"path": "a.md"}}]}]
    assert sp.read(p2) == [{"op": "remove", "path": "b.md"}]


def test_names_sort_chronologically(tmp_path: Path):
    sp = Spool(tmp_path)
    paths = [sp.write([remove_entry(str(i))]) for i in range(5)]
    assert sp.pending() == paths
    assert all(p.name.endswith(".jsonl") for p in paths)


def test_malformed_lines_are_skipped(tmp_path: Path):
    sp = Spool(tmp_path)
    p = tmp_path / "20260101T000000000000Z-deadbeef.jsonl"
    p.write_text('{"op":"remove","path":"x.md"}\nnot json\n\n{"op":"remove","path":"y.md"}\n', encoding="utf-8")
    assert [e["path"] for e in sp.read(p)] == ["x.md", "y.md"]


def test_mark_done_and_sweep(tmp_path: Path):
    sp = Spool(tmp_path)
    p = sp.write([remove_entry("x.md")])
    done = sp.mark_done(p)
    assert done.suffix == ".done" and done.exists() and not p.exists()
    assert sp.pending() == []
    # Fresh .done files are kept; old ones swept.
    assert sp.sweep_done(max_age_seconds=3600) == 0
    old = time.time() - 90000
    os.utime(done, (old, old))
    assert sp.sweep_done(max_age_seconds=86400) == 1
    assert not done.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_spool.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`slim_llm_memory/apps/obsidian/spool.py`:

```python
"""JSONL spool between the ingest process and the mcp (index writer) process.

One file per watcher flush / sweep batch. Line schema:

    {"op": "file",   "path": "Projects/foo.md", "chunks": [{"id","text","meta"}, ...]}
    {"op": "remove", "path": "Projects/foo.md"}

Drain protocol (see brain.py): read pending files in name order, apply,
rename to ``.done`` on success, sweep ``.done`` older than 24 h.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from .parser import Chunk

logger = logging.getLogger(__name__)


def file_entry(path: str, chunks: list[Chunk]) -> dict:
    return {
        "op": "file",
        "path": path,
        "chunks": [{"id": c.id, "text": c.text, "meta": c.meta} for c in chunks],
    }


def remove_entry(path: str) -> dict:
    return {"op": "remove", "path": path}


class Spool:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _new_name(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        return f"{ts}Z-{secrets.token_hex(4)}.jsonl"

    def write(self, entries: list[dict]) -> Path | None:
        if not entries:
            return None
        final = self.dir / self._new_name()
        tmp = final.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e, ensure_ascii=False))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)   # readers never see a half-written file
        return final

    def pending(self) -> list[Path]:
        return sorted(p for p in self.dir.glob("*.jsonl") if p.is_file())

    def depth(self) -> int:
        return len(self.pending())

    def read(self, path: Path) -> list[dict]:
        out: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d malformed spool line skipped: %s", path.name, line_no, exc)
                    continue
                if isinstance(obj, dict) and "op" in obj:
                    out.append(obj)
                else:
                    logger.warning("%s:%d spool line without op skipped", path.name, line_no)
        return out

    def mark_done(self, path: Path) -> Path:
        done = path.with_suffix(".done")
        os.replace(path, done)
        return done

    def sweep_done(self, max_age_seconds: float = 86400) -> int:
        cutoff = time.time() - max_age_seconds
        n = 0
        for p in self.dir.glob("*.done"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    n += 1
            except OSError:
                pass
        return n
```

- [ ] **Step 4: Run tests**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_spool.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add slim_llm_memory/apps/obsidian/spool.py tests/obsidian/test_spool.py
git commit -m "feat(obsidian): JSONL spool with atomic write, done-rename, sweep

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 6: Brain — drain, flush policy, six operations

**Files:**
- Create: `slim_llm_memory/apps/obsidian/brain.py`
- Create: `tests/obsidian/test_brain.py`
- Modify: `slim_llm_memory/apps/obsidian/__init__.py` (re-export `Brain`, `BrainError`, `Chunk`)
- Modify: `tests/obsidian/conftest.py` (add `brain` fixture)

**Interfaces:**
- Consumes: `Memory`, `Memory.neighbours` (Task 1), `Spool` (Task 5), `parse_file`/`rel_path`/`is_vault_markdown` (Task 4).
- Produces:
  ```python
  class BrainError(ValueError): ...

  class Brain:
      def __init__(self, vault: Path, memory: Memory, spool: Spool, *,
                   flush_every_changes: int = 100, flush_every_seconds: float = 30.0) -> None
      lock: threading.RLock             # every public method takes it
      def drain(self) -> dict           # {"files": n, "upserted": n, "removed": n, "embed_failed": bool}
      def maybe_flush(self, force: bool = False) -> bool
      def search(self, query: str, k: int = 8, kinds: list[str] | None = None, min_score: float = 0.3) -> list[dict]
      def get(self, path: str) -> dict                  # {"path","title","text","meta"}; BrainError if missing
      def related(self, path_or_id: str, k: int = 5) -> list[dict]
      def by_tag(self, tags: list[str], k: int = 20) -> list[dict]
      def recent(self, n: int = 10, kind: str | None = None) -> list[dict]
      def remember(self, text: str, tags: list[str] | None = None, title: str | None = None) -> dict
      def stats(self) -> dict          # Memory.stats() + spool_depth + last_drain_ts + last_flush_ts
      def close(self) -> None          # flush + memory.close()
  ```
  Hit dict shape (everywhere a list[dict] is returned):
  `{"id","score","path","title","kind","tags","heading_path","text"}` (score is `None` for `by_tag`/`recent`; `recent` returns one entry per file, using the file's lowest-`section_idx` chunk).

- [ ] **Step 1: Add the brain fixture**

Append to `tests/obsidian/conftest.py`:

```python
from slim_llm_memory import Embedder, Memory
from slim_llm_memory.apps.obsidian.brain import Brain
from slim_llm_memory.apps.obsidian.spool import Spool


@pytest.fixture
def brain(vault: Path, tmp_path: Path):
    home = tmp_path / "home"
    mem = Memory(home / "index", Embedder.noop(dim=64))
    b = Brain(vault, mem, Spool(home / "spool"), flush_every_changes=3, flush_every_seconds=1000)
    yield b
    b.close()
```

- [ ] **Step 2: Write the failing tests**

`tests/obsidian/test_brain.py`:

```python
import time
from pathlib import Path

import pytest

from slim_llm_memory import EmbedderError, Embedder, Memory
from slim_llm_memory.apps.obsidian.brain import Brain, BrainError
from slim_llm_memory.apps.obsidian.parser import parse_file
from slim_llm_memory.apps.obsidian.spool import Spool, file_entry, remove_entry


def spool_file(brain: Brain, rel: str) -> None:
    chunks = parse_file(brain.vault, brain.vault / rel)
    brain.spool.write([file_entry(rel, chunks)])


def test_drain_upserts_and_marks_done(brain: Brain):
    spool_file(brain, "People/Alice.md")
    spool_file(brain, "Projects/Long note.md")
    r = brain.drain()
    assert r["files"] == 2 and r["upserted"] == 5 and r["removed"] == 0 and r["embed_failed"] is False
    assert brain.spool.pending() == []
    assert len(list(brain.spool.dir.glob("*.done"))) == 2
    assert brain.stats()["items_open"] == 5
    assert brain.drain() == {"files": 0, "upserted": 0, "removed": 0, "embed_failed": False}


def test_drain_removes_stale_chunks_when_file_shrinks(brain: Brain):
    spool_file(brain, "Projects/Long note.md")
    brain.drain()
    assert brain.stats()["items_open"] == 4
    (brain.vault / "Projects" / "Long note.md").write_text("# Long literature note\n\nshort now\n", encoding="utf-8")
    spool_file(brain, "Projects/Long note.md")
    r = brain.drain()
    assert r["upserted"] == 1 and r["removed"] == 4
    assert brain.stats()["items_open"] == 1
    # Embedder.noop is hash-based: query with the exact chunk text, and min_score=0 (default 0.3 would drop it).
    assert brain.search("# Long literature note\n\nshort now", k=5, min_score=0.0)[0]["id"] == "Projects/Long note.md#0"


def test_drain_remove_op(brain: Brain):
    spool_file(brain, "People/Alice.md")
    brain.drain()
    brain.spool.write([remove_entry("People/Alice.md")])
    r = brain.drain()
    assert r["removed"] == 1 and brain.stats()["items_open"] == 0
    # removing an unknown path is a no-op, not an error
    brain.spool.write([remove_entry("nope.md")])
    assert brain.drain()["removed"] == 0


def test_drain_leaves_file_pending_on_embedder_failure(vault: Path, tmp_path: Path):
    class Flaky(Embedder):
        def __init__(self):
            self.name, self.dim, self.fail = "noop:64", 64, True
            self._inner = Embedder.noop(64)
        def embed(self, texts):
            if self.fail:
                raise EmbedderError("ollama down")
            return self._inner.embed(texts)
    emb = Flaky()
    b = Brain(vault, Memory(tmp_path / "idx", emb), Spool(tmp_path / "spool"))
    try:
        spool_file(b, "People/Alice.md")
        r = b.drain()
        assert r["embed_failed"] is True and r["upserted"] == 0
        assert b.spool.depth() == 1                     # not renamed
        emb.fail = False
        r = b.drain()
        assert r["embed_failed"] is False and r["upserted"] == 1 and b.spool.depth() == 0
    finally:
        b.close()


def test_flush_policy_every_n_changes(brain: Brain):
    # fixture: flush_every_changes=3
    spool_file(brain, "People/Alice.md")
    brain.drain()
    assert brain.memory.store._dirty is True            # 1 change < 3
    spool_file(brain, "Projects/Long note.md")          # +4 → 5 ≥ 3 → flushed
    brain.drain()
    assert brain.memory.store._dirty is False
    assert brain.stats()["last_flush_ts"] is not None


def test_flush_policy_by_time(vault: Path, tmp_path: Path):
    b = Brain(vault, Memory(tmp_path / "idx", Embedder.noop(64)), Spool(tmp_path / "spool"),
              flush_every_changes=1000, flush_every_seconds=0.05)
    try:
        spool_file(b, "People/Alice.md")
        b.drain()
        assert b.memory.store._dirty is True
        time.sleep(0.06)
        assert b.maybe_flush() is True
        assert b.memory.store._dirty is False
    finally:
        b.close()


def test_search_shape_and_filters(brain: Brain):
    for rel in ["People/Alice.md", "Daily/2026-05-20.md", "Projects/Long note.md"]:
        spool_file(brain, rel)
    brain.drain()
    # Embedder.noop is hash-based: only the exact chunk text scores ~1.0.
    hits = brain.search("# Alice\n\nWorks on infra. Knows nginx. #person", k=3, min_score=0.0)
    h = hits[0]
    assert set(h) == {"id", "score", "path", "title", "kind", "tags", "heading_path", "text"}
    assert h["path"] == "People/Alice.md" and h["title"] == "Alice Example"
    assert brain.search("anything", k=3, kinds=["Daily"], min_score=-1.0)[0]["kind"] == "Daily"
    assert brain.search("", k=3) == []


def test_get_returns_full_file(brain: Brain):
    spool_file(brain, "People/Alice.md")
    brain.drain()
    g = brain.get("People/Alice.md")
    assert g["title"] == "Alice Example" and g["text"].startswith("---\ntitle: Alice Example")
    assert g["meta"]["kind"] == "People"
    with pytest.raises(BrainError, match="not found"):
        brain.get("People/Nobody.md")
    with pytest.raises(BrainError, match="outside"):
        brain.get("../outside.md")


def test_related_by_path_and_by_id(brain: Brain):
    (brain.vault / "People" / "Alice2.md").write_text(
        "---\ntitle: Alice Example\ntags: [person, colleague]\n---\n# Alice\n\nWorks on infra. Knows nginx. #person\n",
        encoding="utf-8")
    for rel in ["People/Alice.md", "People/Alice2.md", "Daily/2026-05-20.md", "Projects/Long note.md"]:
        spool_file(brain, rel)
    brain.drain()
    calls = brain.memory.stats()["counters"]["embed.calls"]
    r = brain.related("People/Alice.md", k=2)
    assert r[0]["path"] == "People/Alice2.md"
    assert all(x["id"] != "People/Alice.md#0" for x in r)
    # H2-split file resolves to its first chunk (#1)
    r2 = brain.related("Projects/Long note.md", k=2)
    assert all(x["id"] != "Projects/Long note.md#1" for x in r2)
    r3 = brain.related("Projects/Long note.md#2", k=2)
    assert len(r3) == 2
    assert brain.memory.stats()["counters"]["embed.calls"] == calls   # zero embed calls
    with pytest.raises(BrainError, match="not found"):
        brain.related("nope.md")


def test_by_tag_and_recent(brain: Brain):
    for rel in ["People/Alice.md", "Daily/2026-05-20.md", "Projects/Long note.md", "inbox/01J-capture.md"]:
        spool_file(brain, rel)
    brain.drain()
    tagged = brain.by_tag(["person"], k=20)
    assert [t["path"] for t in tagged] == ["People/Alice.md"]
    both = brain.by_tag(["person", "literature"], k=20)
    assert {t["path"] for t in both} == {"People/Alice.md", "Projects/Long note.md"}
    assert all(t["score"] is None for t in both)
    assert len(brain.by_tag(["literature"], k=20)) == 1     # one entry per file, not per chunk
    rec = brain.recent(n=10)
    assert len(rec) == 4 and len({r["path"] for r in rec}) == 4
    mt = [r["mtime"] for r in rec]
    assert mt == sorted(mt, reverse=True)
    assert [r["path"] for r in brain.recent(n=5, kind="inbox")] == ["inbox/01J-capture.md"]


def test_remember_writes_inbox_only(brain: Brain):
    r = brain.remember("noteworthy snippet", tags=["claude-session"], title="../../etc/passwd")
    assert r["ingested"] is False
    assert r["path"].startswith("inbox/") and r["path"].endswith("-etc-passwd.md")
    assert r["id"] == r["path"] + "#0"
    p = brain.vault / r["path"]
    assert p.resolve().is_relative_to((brain.vault / "inbox").resolve())
    txt = p.read_text(encoding="utf-8")
    assert txt.startswith("---\n") and "tags: [claude-session]" in txt and "source: claude" in txt
    assert txt.rstrip().endswith("noteworthy snippet")
    assert "title: ../../etc/passwd" not in txt
    # no title → slug from first words
    r2 = brain.remember("Remember to buy milk tomorrow morning early")
    assert r2["path"].endswith("-remember-to-buy-milk-tomorrow-morning.md")
    with pytest.raises(BrainError):
        brain.remember("   ")


def test_inbox_created_on_init(vault: Path, tmp_path: Path):
    import shutil
    shutil.rmtree(vault / "inbox")
    b = Brain(vault, Memory(tmp_path / "idx", Embedder.noop(64)), Spool(tmp_path / "spool"))
    try:
        assert (vault / "inbox").is_dir()
    finally:
        b.close()


def test_missing_vault_fails_fast(tmp_path: Path):
    with pytest.raises(BrainError, match="vault"):
        Brain(tmp_path / "nope", Memory(tmp_path / "idx", Embedder.noop(64)), Spool(tmp_path / "spool"))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_brain.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Implement**

`slim_llm_memory/apps/obsidian/brain.py`:

```python
"""Brain — the MCP-agnostic core of the Obsidian Brain.

Owns the single ``Memory`` writer, drains the spool, applies the flush
policy, and implements the six operations the MCP server exposes. Every
public method takes ``self.lock`` so a background drain thread and the
tool calls never interleave inside ``Memory``.
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slim_llm_memory import EmbedderError, Hit, Memory

from .parser import is_vault_markdown
from .spool import Spool

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


class BrainError(ValueError):
    """User-facing failure (bad path, unknown item, bad argument)."""


def _hit_dict(item_id: str, score: float | None, text: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item_id,
        "score": None if score is None else round(float(score), 4),
        "path": meta.get("path"),
        "title": meta.get("title"),
        "kind": meta.get("kind"),
        "tags": list(meta.get("tags") or []),
        "heading_path": list(meta.get("heading_path") or []),
        "text": text,
    }


class Brain:
    def __init__(
        self,
        vault: Path,
        memory: Memory,
        spool: Spool,
        *,
        flush_every_changes: int = 100,
        flush_every_seconds: float = 30.0,
    ) -> None:
        self.vault = Path(vault).expanduser().resolve()
        if not self.vault.is_dir():
            memory.close(flush=False)
            raise BrainError(f"vault directory does not exist: {self.vault}")
        self.memory = memory
        self.spool = spool
        self.flush_every_changes = int(flush_every_changes)
        self.flush_every_seconds = float(flush_every_seconds)
        self.lock = threading.RLock()

        self.inbox = self.vault / "inbox"
        self.inbox.mkdir(exist_ok=True)

        self._changes_since_flush = 0
        self._started_at = time.time()
        self._last_flush_ts: float | None = None
        self._last_drain_ts: float | None = None
        self._embed_failing = False
        # path → set(chunk ids) for stale-chunk removal and path lookups.
        self._by_path: dict[str, set[str]] = {}
        for it in self.memory.store.items:
            if not it.deleted:
                self._by_path.setdefault(str(it.meta.get("path")), set()).add(it.id)

    # ─── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        with self.lock:
            self.memory.close(flush=True)

    # ─── drain + flush ────────────────────────────────────────────────────
    def drain(self) -> dict[str, Any]:
        """Apply every pending spool file in order. Stops at the first
        embedder failure and leaves that file pending for the next tick."""
        result = {"files": 0, "upserted": 0, "removed": 0, "embed_failed": False}
        with self.lock:
            for path in self.spool.pending():
                entries = self.spool.read(path)
                try:
                    for e in entries:
                        if e.get("op") == "file":
                            up, rm = self._apply_file(str(e.get("path")), e.get("chunks") or [])
                        elif e.get("op") == "remove":
                            up, rm = 0, self._remove_path(str(e.get("path")))
                        else:
                            logger.warning("unknown spool op %r in %s", e.get("op"), path.name)
                            continue
                        result["upserted"] += up
                        result["removed"] += rm
                except EmbedderError as exc:
                    if not self._embed_failing:
                        logger.warning("embedder failing, will retry: %s", exc)
                    self._embed_failing = True
                    result["embed_failed"] = True
                    break
                self.spool.mark_done(path)
                result["files"] += 1
            else:
                if self._embed_failing:
                    logger.info("embedder recovered")
                self._embed_failing = False
            self._last_drain_ts = time.time()
            self.spool.sweep_done()
            self.maybe_flush()
        return result

    def _apply_file(self, path: str, chunks: list[dict]) -> tuple[int, int]:
        new_ids = {c["id"] for c in chunks}
        stale = self._by_path.get(path, set()) - new_ids
        if chunks:
            self.memory.upsert([{"id": c["id"], "text": c["text"], "meta": c.get("meta") or {}} for c in chunks])
        removed = 0
        for sid in stale:
            if self.memory.remove(sid):
                removed += 1
        if new_ids:
            self._by_path[path] = new_ids
        else:
            self._by_path.pop(path, None)
        self._changes_since_flush += len(chunks) + removed
        return len(chunks), removed

    def _remove_path(self, path: str) -> int:
        removed = 0
        for sid in self._by_path.pop(path, set()):
            if self.memory.remove(sid):
                removed += 1
        self._changes_since_flush += removed
        return removed

    def maybe_flush(self, force: bool = False) -> bool:
        with self.lock:
            due_by_count = self._changes_since_flush >= self.flush_every_changes
            since = time.time() - (self._last_flush_ts or self._started_at)
            due_by_time = self.memory.store._dirty and since >= self.flush_every_seconds
            if not (force or due_by_count or due_by_time):
                return False
            wrote = self.memory.flush(force=force)
            self._changes_since_flush = 0
            self._last_flush_ts = time.time()
            return wrote

    # ─── operations ───────────────────────────────────────────────────────
    def search(self, query: str, k: int = 8, kinds: list[str] | None = None,
               min_score: float = 0.3) -> list[dict]:
        with self.lock:
            hits = self.memory.search(query, k=k, kinds=set(kinds) if kinds else None, min_score=min_score)
            return [_hit_dict(h.id, h.score, h.text, h.meta) for h in hits]

    def _safe_vault_path(self, rel: str) -> Path:
        p = (self.vault / rel).resolve()
        if not p.is_relative_to(self.vault):
            raise BrainError(f"path is outside the vault: {rel}")
        return p

    def _first_chunk_id(self, path: str) -> str | None:
        ids = self._by_path.get(path)
        if not ids:
            return None
        return min(ids, key=lambda i: int(i.rsplit("#", 1)[1]))

    def get(self, path: str) -> dict:
        with self.lock:
            p = self._safe_vault_path(path)
            if not p.is_file() or not is_vault_markdown(self.vault, p):
                raise BrainError(f"note not found: {path}")
            text = p.read_text(encoding="utf-8")
            first = self._first_chunk_id(path)
            meta: dict[str, Any] = {}
            if first is not None:
                meta = dict(self.memory.store.items[self.memory.store._id_to_idx[first]].meta)
            return {"path": path, "title": meta.get("title") or p.stem, "text": text, "meta": meta}

    def related(self, path_or_id: str, k: int = 5) -> list[dict]:
        with self.lock:
            item_id = path_or_id if "#" in path_or_id else self._first_chunk_id(path_or_id)
            if item_id is None or item_id not in self.memory.store._id_to_idx:
                raise BrainError(f"item not found in index: {path_or_id}")
            hits = self.memory.neighbours(item_id, k=k)
            return [_hit_dict(h.id, h.score, h.text, h.meta) for h in hits]

    def _open_items(self):
        return (it for it in self.memory.store.items if not it.deleted)

    def _one_per_file(self, items) -> list:
        """Keep the lowest section_idx chunk per path."""
        best: dict[str, Any] = {}
        for it in items:
            p = it.meta.get("path")
            cur = best.get(p)
            if cur is None or it.meta.get("section_idx", 0) < cur.meta.get("section_idx", 0):
                best[p] = it
        return list(best.values())

    def by_tag(self, tags: list[str], k: int = 20) -> list[dict]:
        wanted = {t.strip().lstrip("#") for t in (tags or []) if t and t.strip()}
        if not wanted:
            return []
        with self.lock:
            items = [it for it in self._open_items() if wanted & set(it.meta.get("tags") or [])]
            items = self._one_per_file(items)
            items.sort(key=lambda it: float(it.meta.get("mtime") or it.ts), reverse=True)
            return [_hit_dict(it.id, None, it.text, it.meta) for it in items[:k]]

    def recent(self, n: int = 10, kind: str | None = None) -> list[dict]:
        with self.lock:
            items = [it for it in self._open_items() if kind is None or it.meta.get("kind") == kind]
            items = self._one_per_file(items)
            items.sort(key=lambda it: float(it.meta.get("mtime") or it.ts), reverse=True)
            out = []
            for it in items[:n]:
                d = _hit_dict(it.id, None, it.text, it.meta)
                d["mtime"] = float(it.meta.get("mtime") or it.ts)
                out.append(d)
            return out

    def remember(self, text: str, tags: list[str] | None = None, title: str | None = None) -> dict:
        if not text or not text.strip():
            raise BrainError("text must not be empty")
        text = text.strip()
        slug_src = title if title and title.strip() else " ".join(text.split()[:6])
        slug = _SLUG.sub("-", slug_src.lower()).strip("-")[:60].strip("-") or "note"
        ts = datetime.now(timezone.utc)
        name = f"{ts.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}-{slug}.md"
        with self.lock:
            target = (self.inbox / name).resolve()
            if not target.is_relative_to(self.inbox.resolve()):
                raise BrainError("refusing to write outside vault/inbox")
            clean_tags = [t.strip().lstrip("#") for t in (tags or []) if t and t.strip()]
            fm = ["---", f"tags: [{', '.join(clean_tags)}]", "source: claude",
                  f"ts: {ts.strftime('%Y-%m-%dT%H:%M:%SZ')}"]
            if title and title.strip() and "\n" not in title:
                safe_title = title.strip().replace('"', "'")
                if not safe_title.startswith((".", "/")):
                    fm.append(f'title: "{safe_title}"')
            fm.append("---")
            target.write_text("\n".join(fm) + "\n" + text + "\n", encoding="utf-8")
            rel = target.relative_to(self.vault).as_posix()
            return {"path": rel, "id": f"{rel}#0", "ingested": False}

    def stats(self) -> dict:
        with self.lock:
            s = self.memory.stats()
            s.update({
                "vault": str(self.vault),
                "spool_depth": self.spool.depth(),
                "last_drain_ts": self._last_drain_ts,
                "last_flush_ts": self._last_flush_ts,
                "changes_since_flush": self._changes_since_flush,
                "embed_failing": self._embed_failing,
            })
            return s
```

Note on `remember` title: the test passes `title="../../etc/passwd"` and expects the slug `etc-passwd` and no raw title in the frontmatter. The `startswith((".", "/"))` guard drops it. Note `Path.is_relative_to` is 3.9+.

Update `slim_llm_memory/apps/obsidian/__init__.py` to add:

```python
from .brain import Brain, BrainError
from .parser import Chunk

__all__ = ["Brain", "BrainError", "Chunk"]
```

- [ ] **Step 5: Run tests**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_brain.py -v`
Expected: 13 passed. If `test_drain_upserts_and_marks_done` reports `upserted == 5`: Alice = 1 chunk, Long note = 4 chunks (preamble + 3 H2). If the preamble count differs, check Task 4's fixture has a non-empty intro before `## Section A`.

- [ ] **Step 6: Commit**

```bash
git add slim_llm_memory/apps/obsidian/brain.py slim_llm_memory/apps/obsidian/__init__.py tests/obsidian/conftest.py tests/obsidian/test_brain.py
git commit -m "feat(obsidian): Brain core — spool drain, flush policy, six operations

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 7: Full sweep + watcher (debounce, delete, rename)

**Files:**
- Create: `slim_llm_memory/apps/obsidian/ingest.py`
- Create: `slim_llm_memory/apps/obsidian/watcher.py`
- Create: `tests/obsidian/test_ingest.py`, `tests/obsidian/test_watcher.py`

**Interfaces:**
- Consumes: `parse_file`, `is_vault_markdown`, `rel_path`, `Spool`, `file_entry`, `remove_entry`.
- Produces:
  ```python
  # ingest.py
  def iter_vault_files(vault: Path) -> Iterator[Path]                 # sorted, ignores hidden dirs
  def sweep(vault: Path, spool: Spool, *, batch_size: int = 200) -> int   # files spooled

  # watcher.py
  class Debouncer:
      def __init__(self, delay: float, callback: Callable[[str], None]) -> None
      def touch(self, key: str) -> None        # (re)start the timer for key
      def cancel(self, key: str) -> None
      def flush(self) -> None                  # fire everything pending now (tests, shutdown)
      def pending(self) -> int

  class VaultHandler(FileSystemEventHandler):
      def __init__(self, vault: Path, spool: Spool, debounce_seconds: float = 2.0) -> None
      def emit_path(self, rel: str) -> None    # parse if exists → file_entry, else remove_entry
      # on_created / on_modified → touch; on_deleted → cancel + remove; on_moved → remove old + touch new

  def run_watch(vault: Path, spool: Spool, debounce_seconds: float = 2.0, stop: threading.Event | None = None) -> None
  ```

- [ ] **Step 1: Write the failing tests**

`tests/obsidian/test_ingest.py`:

```python
from pathlib import Path

from slim_llm_memory.apps.obsidian.ingest import iter_vault_files, sweep
from slim_llm_memory.apps.obsidian.spool import Spool


def test_iter_vault_files_skips_hidden_and_non_md(vault: Path):
    rels = [p.relative_to(vault).as_posix() for p in iter_vault_files(vault)]
    assert rels == sorted(rels)
    assert "Daily/2026-05-20.md" in rels and "inbox/01J-capture.md" in rels
    assert not any(r.startswith(".obsidian") for r in rels)


def test_sweep_spools_every_file_in_batches(vault: Path, tmp_path: Path):
    sp = Spool(tmp_path / "spool")
    n = sweep(vault, sp, batch_size=2)
    assert n == 6
    files = sp.pending()
    assert len(files) == 3
    entries = [e for f in files for e in sp.read(f)]
    assert all(e["op"] == "file" for e in entries)
    assert {e["path"] for e in entries} == {p.relative_to(vault).as_posix() for p in iter_vault_files(vault)}
    long = next(e for e in entries if e["path"] == "Projects/Long note.md")
    assert len(long["chunks"]) == 4
```

`tests/obsidian/test_watcher.py`:

```python
import threading
import time
from pathlib import Path

from slim_llm_memory.apps.obsidian.spool import Spool
from slim_llm_memory.apps.obsidian.watcher import Debouncer, VaultHandler, run_watch


def test_debouncer_coalesces_and_fires_once():
    fired: list[str] = []
    d = Debouncer(0.05, fired.append)
    for _ in range(5):
        d.touch("a")
    d.touch("b")
    assert d.pending() == 2
    time.sleep(0.15)
    assert sorted(fired) == ["a", "b"]
    assert d.pending() == 0


def test_debouncer_cancel_and_flush():
    fired: list[str] = []
    d = Debouncer(10.0, fired.append)
    d.touch("a"); d.touch("b"); d.cancel("a")
    d.flush()
    assert fired == ["b"] and d.pending() == 0


def test_handler_emit_upsert_then_remove(vault: Path, tmp_path: Path):
    sp = Spool(tmp_path / "spool")
    h = VaultHandler(vault, sp, debounce_seconds=0.01)
    h.emit_path("People/Alice.md")
    (vault / "People" / "Alice.md").unlink()
    h.emit_path("People/Alice.md")
    entries = [e for f in sp.pending() for e in sp.read(f)]
    assert [e["op"] for e in entries] == ["file", "remove"]
    assert entries[0]["chunks"][0]["id"] == "People/Alice.md#0"


def test_handler_ignores_non_vault_files(vault: Path, tmp_path: Path):
    from watchdog.events import FileModifiedEvent
    sp = Spool(tmp_path / "spool")
    h = VaultHandler(vault, sp, debounce_seconds=0.01)
    h.on_modified(FileModifiedEvent(str(vault / ".obsidian" / "workspace.json")))
    h.on_modified(FileModifiedEvent(str(vault / "Daily" / "pic.png")))
    h.debouncer.flush()
    assert sp.pending() == []


def test_run_watch_end_to_end(vault: Path, tmp_path: Path):
    sp = Spool(tmp_path / "spool")
    stop = threading.Event()
    t = threading.Thread(target=run_watch, args=(vault, sp, 0.05, stop), daemon=True)
    t.start()
    time.sleep(0.3)   # observer warm-up
    (vault / "Daily" / "new.md").write_text("# New\n\nfresh #daily\n", encoding="utf-8")
    time.sleep(0.5)
    (vault / "Daily" / "new.md").rename(vault / "Daily" / "renamed.md")
    time.sleep(0.5)
    (vault / "Daily" / "renamed.md").unlink()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=5)
    ops = [(e["op"], e["path"]) for f in sp.pending() for e in sp.read(f)]
    assert ("file", "Daily/new.md") in ops
    assert ("remove", "Daily/new.md") in ops
    assert ("file", "Daily/renamed.md") in ops
    assert ops[-1] == ("remove", "Daily/renamed.md")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_ingest.py tests/obsidian/test_watcher.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement ingest.py**

```python
"""One-shot full sweep: walk the vault and spool every markdown file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from .parser import is_vault_markdown, parse_file, rel_path
from .spool import Spool, file_entry

logger = logging.getLogger(__name__)


def iter_vault_files(vault: Path) -> Iterator[Path]:
    vault = Path(vault)
    for p in sorted(vault.rglob("*.md")):
        if p.is_file() and is_vault_markdown(vault, p):
            yield p


def sweep(vault: Path, spool: Spool, *, batch_size: int = 200) -> int:
    """Spool a ``file`` entry for every vault note. Returns files spooled.
    The mcp side's hash-skip makes re-sweeps cheap."""
    vault = Path(vault)
    batch: list[dict] = []
    n = 0
    for p in iter_vault_files(vault):
        batch.append(file_entry(rel_path(vault, p), parse_file(vault, p)))
        n += 1
        if len(batch) >= batch_size:
            spool.write(batch)
            batch = []
    if batch:
        spool.write(batch)
    logger.info("sweep: spooled %d files from %s", n, vault)
    return n
```

- [ ] **Step 4: Implement watcher.py**

```python
"""watchdog Observer over the vault + per-path debounce → spool entries."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .parser import is_vault_markdown, parse_file, rel_path
from .spool import Spool, file_entry, remove_entry

logger = logging.getLogger(__name__)


class Debouncer:
    """Per-key trailing-edge debounce on ``threading.Timer``."""

    def __init__(self, delay: float, callback: Callable[[str], None]) -> None:
        self.delay = float(delay)
        self.callback = callback
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _fire(self, key: str) -> None:
        with self._lock:
            self._timers.pop(key, None)
        try:
            self.callback(key)
        except Exception:
            logger.exception("debounce callback failed for %s", key)

    def touch(self, key: str) -> None:
        with self._lock:
            old = self._timers.pop(key, None)
            if old is not None:
                old.cancel()
            t = threading.Timer(self.delay, self._fire, args=(key,))
            t.daemon = True
            self._timers[key] = t
            t.start()

    def cancel(self, key: str) -> None:
        with self._lock:
            t = self._timers.pop(key, None)
        if t is not None:
            t.cancel()

    def flush(self) -> None:
        with self._lock:
            keys = list(self._timers)
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()
        for k in keys:
            self._fire(k)

    def pending(self) -> int:
        with self._lock:
            return len(self._timers)


class VaultHandler(FileSystemEventHandler):
    def __init__(self, vault: Path, spool: Spool, debounce_seconds: float = 2.0) -> None:
        super().__init__()
        self.vault = Path(vault).resolve()
        self.spool = spool
        self.debouncer = Debouncer(debounce_seconds, self.emit_path)

    # ─── spool emission ───────────────────────────────────────────────────
    def emit_path(self, rel: str) -> None:
        abs_path = self.vault / rel
        if abs_path.is_file():
            chunks = parse_file(self.vault, abs_path)
            self.spool.write([file_entry(rel, chunks)])
            logger.info("spooled %s (%d chunks)", rel, len(chunks))
        else:
            self.spool.write([remove_entry(rel)])
            logger.info("spooled remove %s", rel)

    # ─── watchdog events ──────────────────────────────────────────────────
    def _rel(self, src: str | bytes) -> str | None:
        p = Path(src.decode() if isinstance(src, bytes) else src)
        if not is_vault_markdown(self.vault, p):
            return None
        return rel_path(self.vault, p)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and (rel := self._rel(event.src_path)):
            self.debouncer.touch(rel)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and (rel := self._rel(event.src_path)):
            self.debouncer.touch(rel)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and (rel := self._rel(event.src_path)):
            self.debouncer.cancel(rel)
            self.emit_path(rel)   # file is gone → remove entry, no debounce

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if rel := self._rel(event.src_path):
            self.debouncer.cancel(rel)
            self.emit_path(rel)
        if rel := self._rel(event.dest_path):
            self.debouncer.touch(rel)


def run_watch(vault: Path, spool: Spool, debounce_seconds: float = 2.0,
              stop: threading.Event | None = None) -> None:
    """Block until ``stop`` is set (or KeyboardInterrupt)."""
    handler = VaultHandler(vault, spool, debounce_seconds)
    observer = Observer()
    observer.schedule(handler, str(handler.vault), recursive=True)
    observer.start()
    logger.info("watching %s (debounce %.1fs)", handler.vault, debounce_seconds)
    try:
        while stop is None or not stop.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join(timeout=5)
        handler.debouncer.flush()
```

Note: `is_vault_markdown` resolves paths, so a deleted file's `rel` still computes because `resolve()` does not require existence (strict=False).

- [ ] **Step 5: Run tests**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_ingest.py tests/obsidian/test_watcher.py -v`
Expected: 7 passed. `test_run_watch_end_to_end` depends on inotify timing; if it flakes, raise the sleeps to 0.8 s rather than weakening asserts.

- [ ] **Step 6: Commit**

```bash
git add slim_llm_memory/apps/obsidian/ingest.py slim_llm_memory/apps/obsidian/watcher.py tests/obsidian/test_ingest.py tests/obsidian/test_watcher.py
git commit -m "feat(obsidian): full sweep + watchdog watcher with per-path debounce

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 8: MCP server adapter

**Files:**
- Create: `slim_llm_memory/apps/obsidian/mcp_server.py`
- Create: `tests/obsidian/test_mcp_server.py`

**Interfaces:**
- Consumes: `Brain` (Task 6), `mcp.server.mcpserver.MCPServer` (mcp 2.x).
- Produces:
  ```python
  def build_server(brain: Brain, *, drain_seconds: float = 5.0, retry_seconds: float = 30.0) -> MCPServer
  def serve(brain: Brain, *, drain_seconds: float = 5.0) -> None   # blocking stdio loop
  ```
  Tools registered: `search`, `get`, `related`, `by_tag`, `recent`, `remember`, plus `stats`. `BrainError` → `ToolError` (message reaches the client); everything else is a crash.

- [ ] **Step 1: Write the failing tests**

`tests/obsidian/test_mcp_server.py`:

```python
import asyncio
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from slim_llm_memory.apps.obsidian.brain import Brain
from slim_llm_memory.apps.obsidian.ingest import sweep
from slim_llm_memory.apps.obsidian.mcp_server import build_server


def _payload(result) -> object:
    if getattr(result, "structured_content", None) is not None:
        sc = result.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(result.content[0].text)


async def test_tools_registered(brain: Brain):
    server = build_server(brain)
    names = {t.name for t in await server.list_tools()}
    assert {"search", "get", "related", "by_tag", "recent", "remember", "stats"} <= names


async def test_search_and_get_via_call_tool(brain: Brain):
    sweep(brain.vault, brain.spool)
    brain.drain()
    server = build_server(brain)
    # Embedder.noop is hash-based: only the exact chunk text scores ~1.0.
    hits = _payload(await server.call_tool("search", {"query": "# Alice\n\nWorks on infra. Knows nginx. #person", "k": 2, "min_score": 0.0}))
    assert hits[0]["path"] == "People/Alice.md"
    got = _payload(await server.call_tool("get", {"path": "People/Alice.md"}))
    assert got["title"] == "Alice Example"
    rem = _payload(await server.call_tool("remember", {"text": "from a test", "tags": ["t"]}))
    assert rem["path"].startswith("inbox/")


async def test_brain_error_becomes_tool_error(brain: Brain):
    from mcp.server.mcpserver.exceptions import ToolError
    server = build_server(brain)
    with pytest.raises(ToolError, match="outside"):
        await server.call_tool("get", {"path": "../x.md"})


async def test_lifespan_drains_spool_on_startup_and_periodically(brain: Brain):
    sweep(brain.vault, brain.spool)
    server = build_server(brain, drain_seconds=0.05)
    assert brain.spool.depth() == 1
    async with server.settings.lifespan(server):
        await asyncio.sleep(0.02)
        assert brain.spool.depth() == 0                 # initial drain
        sweep(brain.vault, brain.spool)
        await asyncio.sleep(0.2)
        assert brain.spool.depth() == 0                 # periodic drain
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_mcp_server.py -v`
Expected: FAIL `ModuleNotFoundError: ... mcp_server`

- [ ] **Step 3: Implement**

`slim_llm_memory/apps/obsidian/mcp_server.py`:

```python
"""stdio MCP server over a ``Brain``. Thin adapter: every tool is one Brain call.

Uses the official ``mcp`` SDK 2.x (``MCPServer``; FastMCP was renamed in 2.0).
A lifespan task drains the spool on startup and then every ``drain_seconds``
(``retry_seconds`` while the embedder is failing).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .brain import Brain, BrainError

logger = logging.getLogger(__name__)


def _guard(fn):
    """Translate BrainError → ToolError so the message reaches the client.
    ``functools.wraps`` sets ``__wrapped__`` so the SDK's ``inspect.signature``
    still sees the real parameters when it builds the input schema."""
    @functools.wraps(fn)
    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except BrainError as exc:
            raise ToolError(str(exc)) from exc
    return wrapped


async def _drain_loop(brain: Brain, drain_seconds: float, retry_seconds: float) -> None:
    while True:
        try:
            r = await asyncio.to_thread(brain.drain)
        except Exception:
            logger.exception("drain failed")
            r = {"embed_failed": True}
        await asyncio.sleep(retry_seconds if r.get("embed_failed") else drain_seconds)


def build_server(brain: Brain, *, drain_seconds: float = 5.0, retry_seconds: float = 30.0) -> MCPServer:

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[dict[str, Any]]:
        task = asyncio.create_task(_drain_loop(brain, drain_seconds, retry_seconds))
        try:
            yield {"brain": brain}
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            await asyncio.to_thread(brain.maybe_flush, True)

    server = MCPServer(
        "obsidian-brain",
        instructions=(
            "Semantic memory over the user's Obsidian vault. Use `search` for "
            "free-text recall, `get` to read a full note, `related` for neighbours "
            "of a note, `by_tag`/`recent` for browsing, and `remember` to stash a "
            "finding into vault/inbox/ for future sessions."
        ),
        lifespan=lifespan,
    )

    @server.tool()
    @_guard
    def search(query: str, k: int = 8, kinds: list[str] | None = None, min_score: float = 0.3) -> list[dict]:
        """Top-k semantic search over vault chunks. `kinds` filters by top-level folder."""
        return brain.search(query, k=k, kinds=kinds, min_score=min_score)

    @server.tool()
    @_guard
    def get(path: str) -> dict:
        """Return the full text + metadata of one note by vault-relative path."""
        return brain.get(path)

    @server.tool()
    @_guard
    def related(path_or_id: str, k: int = 5) -> list[dict]:
        """Nearest chunks to an existing note (path) or chunk id (`path#idx`). No embedding call."""
        return brain.related(path_or_id, k=k)

    @server.tool()
    @_guard
    def by_tag(tags: list[str], k: int = 20) -> list[dict]:
        """Notes carrying any of `tags`, newest first."""
        return brain.by_tag(tags, k=k)

    @server.tool()
    @_guard
    def recent(n: int = 10, kind: str | None = None) -> list[dict]:
        """Most recently modified notes, optionally restricted to one top-level folder."""
        return brain.recent(n=n, kind=kind)

    @server.tool()
    @_guard
    def remember(text: str, tags: list[str] | None = None, title: str | None = None) -> dict:
        """Write a capture into vault/inbox/. It becomes searchable once the ingest watcher picks it up."""
        return brain.remember(text, tags=tags, title=title)

    @server.tool()
    @_guard
    def stats() -> dict:
        """Index + spool health."""
        return brain.stats()

    return server


def serve(brain: Brain, *, drain_seconds: float = 5.0) -> None:
    build_server(brain, drain_seconds=drain_seconds).run(transport="stdio")
```

If `ToolError` is not at `mcp.server.mcpserver.exceptions`, find it with `grep -rn "class ToolError" $(python -c "import mcp,os;print(os.path.dirname(mcp.__file__))")`, then fix the import in both the module and the test. If the SDK rejects the wrapped signature (input schema shows `*a, **kw`), drop `_guard` and put the `try/except BrainError → ToolError` inside each tool body instead.

- [ ] **Step 4: Run tests**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_mcp_server.py -v`
Expected: 4 passed. If `_payload` fails to decode, print `result` once, adapt `_payload` to the real `CallToolResult` shape, and keep the assertions.

- [ ] **Step 5: Commit**

```bash
git add slim_llm_memory/apps/obsidian/mcp_server.py tests/obsidian/test_mcp_server.py
git commit -m "feat(obsidian): MCP stdio server with seven tools and background spool drain

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 9: CLI (`ingest | mcp | stats`) + logging

**Files:**
- Create: `slim_llm_memory/apps/obsidian/cli.py`
- Create: `tests/obsidian/test_cli.py`

**Interfaces:**
- Consumes: `load_config`, `make_embedder`, `sweep`, `run_watch`, `Brain`, `serve`, `Spool`.
- Produces: `main(argv: list[str] | None = None) -> int`. Subcommands:
  ```
  slim-llm-obsidian ingest [--vault PATH] [--config PATH] [--watch]
  slim-llm-obsidian mcp    [--vault PATH] [--config PATH]
  slim-llm-obsidian stats  [--vault PATH] [--config PATH]
  ```
  `--home PATH` (hidden, for tests) overrides `~/.obsidian-brain`.

- [ ] **Step 1: Write the failing tests**

`tests/obsidian/test_cli.py`:

```python
import json
from pathlib import Path

import pytest

from slim_llm_memory.apps.obsidian.cli import main
from slim_llm_memory.apps.obsidian.spool import Spool


def test_ingest_one_shot_spools_files(vault: Path, tmp_path: Path, capsys):
    home = tmp_path / "home"
    rc = main(["ingest", "--vault", str(vault), "--home", str(home)])
    assert rc == 0
    assert Spool(home / "spool").depth() == 1
    assert (home / "logs" / "ingest.log").exists()


def test_stats_prints_json_read_only(vault: Path, tmp_path: Path, capsys):
    home = tmp_path / "home"
    (home).mkdir()
    (home / "config.toml").write_text('embedder = "noop:32"\n', encoding="utf-8")
    assert main(["ingest", "--vault", str(vault), "--home", str(home)]) == 0
    rc = main(["stats", "--vault", str(vault), "--home", str(home)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["items_open"] == 0 and out["spool_depth"] == 1   # stats never writes the index


def test_missing_vault_fails_fast(tmp_path: Path, capsys):
    rc = main(["mcp", "--vault", str(tmp_path / "nope"), "--home", str(tmp_path / "home")])
    assert rc == 2
    assert "vault" in capsys.readouterr().err.lower()


def test_no_vault_configured(tmp_path: Path, capsys):
    rc = main(["ingest", "--home", str(tmp_path / "home")])
    assert rc == 2
    assert "--vault" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_cli.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`slim_llm_memory/apps/obsidian/cli.py`:

```python
"""``slim-llm-obsidian`` entry point: ingest | mcp | stats.

stdout is reserved for the MCP transport in ``mcp`` mode; all logging goes
to stderr and ``~/.obsidian-brain/logs/<cmd>.log``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config, load_config, make_embedder
from .spool import Spool


def _setup_logging(cfg: Config, name: str) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.FileHandler(cfg.logs_dir / f"{name}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ]
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO),
                        format=fmt, handlers=handlers, force=True)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slim-llm-obsidian", description="Obsidian Brain: vault ingest + MCP server")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("ingest", "sweep the vault into the spool (add --watch to stay alive)"),
                        ("mcp", "run the stdio MCP server (single index writer)"),
                        ("stats", "print index + spool stats as JSON")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--vault", help="vault path (overrides config.toml)")
        s.add_argument("--config", type=Path, help="config file (default ~/.obsidian-brain/config.toml)")
        s.add_argument("--home", type=Path, help=argparse.SUPPRESS)
        if name == "ingest":
            s.add_argument("--watch", action="store_true", help="after the sweep, keep watching for changes")
    return p


def _load(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config, vault=args.vault, home=args.home)
    if not cfg.vault.is_dir():
        raise ValueError(f"vault directory does not exist: {cfg.vault}")
    return cfg


def cmd_ingest(cfg: Config, watch: bool) -> int:
    from .ingest import sweep
    spool = Spool(cfg.spool_dir)
    n = sweep(cfg.vault, spool)
    logging.getLogger(__name__).info("swept %d files", n)
    if watch:
        from .watcher import run_watch
        run_watch(cfg.vault, spool, cfg.watcher_debounce_seconds)
    return 0


def cmd_mcp(cfg: Config) -> int:
    from slim_llm_memory import Memory
    from .brain import Brain
    from .mcp_server import serve
    memory = Memory(cfg.index_dir, make_embedder(cfg.embedder, cfg.embedder_url))
    brain = Brain(cfg.vault, memory, Spool(cfg.spool_dir),
                  flush_every_changes=cfg.flush_every_changes,
                  flush_every_seconds=cfg.flush_every_seconds)
    try:
        serve(brain, drain_seconds=cfg.spool_drain_seconds)
    finally:
        brain.close()
    return 0


def cmd_stats(cfg: Config) -> int:
    """Read-only: never opens the index for writing (the mcp process may hold the lock)."""
    manifest = cfg.index_dir / "manifest.json"
    out: dict = {"vault": str(cfg.vault), "index_dir": str(cfg.index_dir),
                 "items_open": 0, "index_version": None, "index_ts": None}
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        out.update({"items_open": m.get("n_open", 0), "index_version": m.get("version"),
                    "index_ts": m.get("ts"), "embedder": m.get("embedder")})
    spool = Spool(cfg.spool_dir)
    pending = spool.pending()
    out["spool_depth"] = len(pending)
    out["spool_oldest"] = pending[0].name if pending else None
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = _load(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _setup_logging(cfg, args.cmd)
    try:
        if args.cmd == "ingest":
            return cmd_ingest(cfg, args.watch)
        if args.cmd == "mcp":
            return cmd_mcp(cfg)
        return cmd_stats(cfg)
    except ValueError as exc:      # BrainError, StoreError-as-ValueError, config
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests + the whole suite**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q`
Expected: all passed (≈ 48 core + ≈ 45 obsidian), under 10 s. `logging.basicConfig(force=True)` prevents handler duplication across tests.

- [ ] **Step 5: Commit**

```bash
git add slim_llm_memory/apps/obsidian/cli.py tests/obsidian/test_cli.py
git commit -m "feat(obsidian): slim-llm-obsidian CLI (ingest|mcp|stats) with file+stderr logging

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

### Task 10: End-to-end integration test, docs, spec status

**Files:**
- Create: `tests/obsidian/test_integration.py`
- Modify: `README.md` (new "Obsidian Brain" section + Status line)
- Modify: `docs/specs/2026-05-20-obsidian-brain-design.md` (Status → implemented; spool schema note)
- Modify: `docs/IMPLEMENTATION.md` §13 (mention `apps/obsidian`)

- [ ] **Step 1: Write the integration test**

`tests/obsidian/test_integration.py`:

```python
"""Hermetic end-to-end: sweep → drain → search; remember → sweep → drain → search.
Mirrors the spec's Data flow section with Embedder.noop()."""

import time
from pathlib import Path

from slim_llm_memory.apps.obsidian.brain import Brain
from slim_llm_memory.apps.obsidian.ingest import sweep


def test_edit_to_searchable_and_capture_roundtrip(brain: Brain):
    assert sweep(brain.vault, brain.spool) == 6
    r = brain.drain()
    assert r["files"] == 1 and r["upserted"] >= 6

    # Embedder.noop is hash-based: only the exact chunk text scores ~1.0.
    t0 = time.perf_counter()
    hits = brain.search("# Alice\n\nWorks on infra. Knows nginx. #person", k=3, min_score=0.0)
    assert (time.perf_counter() - t0) < 0.2
    assert hits[0]["path"] == "People/Alice.md"

    # edit → re-sweep → hash-skip for unchanged, re-embed for changed
    (brain.vault / "Daily" / "2026-05-20.md").write_text("Completely different daily entry about kayaking.\n", encoding="utf-8")
    sweep(brain.vault, brain.spool)
    brain.drain()
    assert brain.search("Completely different daily entry about kayaking.", k=1, min_score=0.0)[0]["path"] == "Daily/2026-05-20.md"
    skipped = brain.memory.stats()["counters"]["upsert.skipped_text_unchanged"]
    assert skipped >= 5

    # capture → lands in inbox → next sweep+drain makes it searchable
    cap = brain.remember("kayak rental is cheaper on weekdays", tags=["claude-session"])
    assert cap["ingested"] is False
    sweep(brain.vault, brain.spool)
    brain.drain()
    hit = brain.search("kayak rental is cheaper on weekdays", k=1, min_score=0.0)[0]
    assert hit["path"] == cap["path"] and hit["kind"] == "inbox"
    assert brain.get(cap["path"])["meta"]["source"] == "inbox"

    # persistence: flush, reopen, still searchable
    brain.maybe_flush(force=True)
    stats = brain.stats()
    assert stats["dirty"] is False and stats["spool_depth"] == 0
```

- [ ] **Step 2: Run it**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest tests/obsidian/test_integration.py -v`
Expected: 1 passed.

- [ ] **Step 3: README section**

Change the Status line to: `> **Status:** phase 1 — \`Memory\` core, plus the first app: **Obsidian Brain** (\`slim_llm_memory.apps.obsidian\`).` and add before "## Tests + example":

````markdown
## Obsidian Brain (app)

Durable recall over your Obsidian vault for Claude Code, via MCP.
Design: [`docs/specs/2026-05-20-obsidian-brain-design.md`](docs/specs/2026-05-20-obsidian-brain-design.md).

```bash
pip install slim-llm-memory[obsidian]        # + watchdog, pyyaml, mcp

cat > ~/.obsidian-brain/config.toml <<EOF
vault = "~/Vault"
embedder = "ollama:nomic-embed-text"
EOF

slim-llm-obsidian ingest --watch             # terminal 1: sweep, then watch
claude mcp add obsidian-brain -- slim-llm-obsidian mcp   # register with Claude Code
slim-llm-obsidian stats                      # health: items, spool depth
```

Two processes, one index writer: `ingest` parses notes into a JSONL spool
under `~/.obsidian-brain/spool/`; `mcp` (started by Claude Code) drains it
every 5 s and serves `search`, `get`, `related`, `by_tag`, `recent`,
`remember`, `stats`. `remember` writes only to `vault/inbox/`.
````

- [ ] **Step 4: Spec + implementation doc**

In the spec header set `**Status:** implemented 2026-09-03 (see docs/superpowers/plans/2026-09-03-obsidian-brain.md)` and add under "### spool.py" a one-paragraph note: "Implementation note: lines are per *file* (`{"op":"file","path","chunks":[…]}` / `{"op":"remove","path"}`) so the brain can drop stale chunk ids when a note shrinks; `parse_file_deleted` was not needed. Hash is computed by `Memory`."

In `docs/IMPLEMENTATION.md` §13 add a line under `slim_llm_memory/`: `│   └── apps/obsidian/   # first consumer app: vault ingest + MCP server`.

- [ ] **Step 5: Full suite, then commit**

Run: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q`
Expected: all passed.

```bash
git add tests/obsidian/test_integration.py README.md docs/specs/2026-05-20-obsidian-brain-design.md docs/IMPLEMENTATION.md
git commit -m "docs(obsidian): README quickstart, spec status, end-to-end integration test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Bfp1r95yrUXn93gKURKeU2"
```

---

## Manual smoke (after Task 10, not automated)

1. `pip install -e ".[obsidian]"` in the trading env.
2. `slim-llm-obsidian ingest --vault ~/Vault` then `slim-llm-obsidian stats` → `spool_depth ≥ 1`.
3. `claude mcp add obsidian-brain -- /home/trbck/miniconda3/envs/trading/bin/slim-llm-obsidian mcp --vault ~/Vault`.
4. In a Claude Code session: call `stats` (spool drained, items_open > 0), `search`, `get`, `related`, `by_tag`, `recent`, `remember`; confirm the inbox file appears and, with `ingest --watch` running, becomes searchable within ~7 s.
5. Watch `~/.obsidian-brain/logs/mcp.log` for a single "embedder failing" warning if Ollama is stopped, and "embedder recovered" when it returns.

## Self-review notes

- Spec coverage: parser (T4), chunker (T3), watcher + renames/deletes (T7), ingest sweep + `--watch` (T7/T9), spool + drain protocol + `.done` sweep (T5/T6), six tools + `related` via `Memory.neighbours` (T1/T6/T8), `remember` inbox-only + slug + path assert (T6), flush policy 100/30 s (T6), config.toml (T2), logs dir + stderr (T9), error table: Ollama down → file left pending + single warning (T6/T8), malformed line (T5), vault missing → fail fast (T6/T9), second mcp → `Memory` lock (existing), non-UTF-8 skip (T4), event storm → per-path debounce + batched sweep (T7). Long-chunk hard clip (>2048 tokens) is handled implicitly: the chunker's 400-token windows never reach the embedder limit; no extra code.
- Not in scope, as the spec says: graph queries, LLM enrichment, HTTP endpoint, Q&A shell.
- Type consistency: `Chunk(id,text,meta)` (T4) ↔ `file_entry` (T5) ↔ `Brain._apply_file` reads `c["id"], c["text"], c["meta"]` (T6). `Brain.search(kinds: list[str])` (T6) ↔ MCP `search(kinds: list[str] | None)` (T8). `Config` field names (T2) ↔ `cli.py` usages (T9).
