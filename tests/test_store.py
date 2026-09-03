"""Tests for the persistence layer — Store directly.

Cover the contract from docs/IMPLEMENTATION.md §5:
  - persistence round-trip
  - hash-skip on identical content (Store layer just receives Items;
    skip is enforced one layer up by Memory, so we don't test it here)
  - atomic crash recovery via fault-injected os.replace
  - embedder mismatch refusal
  - tombstones + compaction at the >20% threshold
  - lock blocks a second writer
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from slim_llm_memory.store import Item, Store, StoreError


def _make_store(path: Path, name: str = "test:8", dim: int = 8) -> Store:
    return Store(path, embedder_name=name, embedder_dim=dim)


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ─── round-trip ───────────────────────────────────────────────────────────

def test_round_trip(tmp_path: Path):
    with _make_store(tmp_path) as s:
        s.add_item(Item(id="a", text="hello", hash="h1"), _vec(1))
        s.add_item(Item(id="b", text="world", hash="h2"), _vec(2))
        s.flush()
    with _make_store(tmp_path) as s:
        ids = [it.id for it in s.items]
        assert ids == ["a", "b"]
        assert s.vectors.shape == (2, 8)


def test_no_dirty_no_flush(tmp_path: Path):
    with _make_store(tmp_path) as s:
        assert s.flush() is False  # nothing dirty


def test_dirty_flush_returns_true(tmp_path: Path):
    with _make_store(tmp_path) as s:
        s.add_item(Item(id="a", text="x", hash="h"), _vec(1))
        assert s.flush() is True
        # second flush is a no-op
        assert s.flush() is False


# ─── embedder mismatch ────────────────────────────────────────────────────

def test_embedder_mismatch_refuses_to_load(tmp_path: Path):
    with _make_store(tmp_path, name="model-A", dim=8) as s:
        s.add_item(Item(id="a", text="x", hash="h"), _vec(1))
        s.flush()
    with pytest.raises(StoreError, match="embedder mismatch"):
        Store(tmp_path, embedder_name="model-B", embedder_dim=8)


def test_embedder_dim_mismatch_refuses_to_load(tmp_path: Path):
    with _make_store(tmp_path, name="model-A", dim=8) as s:
        s.add_item(Item(id="a", text="x", hash="h"), _vec(1, 8))
        s.flush()
    with pytest.raises(StoreError, match="dim"):
        Store(tmp_path, embedder_name="model-A", embedder_dim=16)


# ─── atomic crash recovery ───────────────────────────────────────────────

def test_crash_before_manifest_replace_keeps_previous_intact(tmp_path: Path, monkeypatch):
    # First commit — succeeds normally.
    with _make_store(tmp_path) as s:
        s.add_item(Item(id="a", text="alpha", hash="h1"), _vec(1))
        s.flush()

    # Second commit — simulate crash *before* the manifest os.replace.
    with _make_store(tmp_path) as s:
        s.add_item(Item(id="b", text="beta", hash="h2"), _vec(2))
        # Patch os.replace to blow up specifically on the manifest swap.
        original_replace = os.replace

        def _fault(src, dst):
            if str(dst).endswith("manifest.json"):
                raise RuntimeError("simulated crash mid-commit")
            return original_replace(src, dst)

        monkeypatch.setattr("slim_llm_memory.store.os.replace", _fault)
        with pytest.raises(RuntimeError, match="simulated"):
            s.flush()

    # Re-open: must see only the *previous* committed state.
    with _make_store(tmp_path) as s:
        ids = [it.id for it in s.items]
        assert ids == ["a"]


def test_garbage_versioned_files_are_ignored_on_load(tmp_path: Path):
    # Commit one version, then drop a stray future-version file in.
    with _make_store(tmp_path) as s:
        s.add_item(Item(id="a", text="x", hash="h"), _vec(1))
        s.flush()
    (tmp_path / "items.v999.jsonl").write_text("garbage\n", encoding="utf-8")
    np.save(tmp_path / "vectors.v999.npy", np.zeros((1, 8), dtype=np.float32))

    with _make_store(tmp_path) as s:
        # Manifest still points at v1, garbage v999 is ignored.
        assert [it.id for it in s.items] == ["a"]


# ─── tombstones + compaction ─────────────────────────────────────────────

def test_mark_deleted_tombstones(tmp_path: Path):
    with _make_store(tmp_path) as s:
        for i in range(5):
            s.add_item(Item(id=str(i), text=f"t{i}", hash=f"h{i}"), _vec(i))
        s.flush()
        assert s.mark_deleted("2") is True
        assert s.mark_deleted("nope") is False
        # Still in items list — tombstoned
        assert any(it.deleted for it in s.items)
        # Open indices skip the tombstone
        open_idx = list(s.open_indices())
        assert 2 not in open_idx


def test_compaction_triggers_above_threshold(tmp_path: Path):
    # 5 items, delete 2 → 40% tombstoned > 20% threshold → compaction on flush.
    with _make_store(tmp_path) as s:
        for i in range(5):
            s.add_item(Item(id=str(i), text=f"t{i}", hash=f"h{i}"), _vec(i))
        s.flush()
        s.mark_deleted("0")
        s.mark_deleted("3")
        assert s.needs_compaction() is True
        s.flush()
        # After compaction, no tombstones survive on disk.
        assert all(not it.deleted for it in s.items)
        assert len(s.items) == 3
        assert {it.id for it in s.items} == {"1", "2", "4"}


def test_compaction_does_not_trigger_below_threshold(tmp_path: Path):
    # 10 items, delete 1 → 10% tombstoned < 20% threshold → no compaction.
    with _make_store(tmp_path) as s:
        for i in range(10):
            s.add_item(Item(id=str(i), text=f"t{i}", hash=f"h{i}"), _vec(i))
        s.flush()
        s.mark_deleted("0")
        assert s.needs_compaction() is False
        s.flush()
        assert any(it.deleted for it in s.items)
        assert len(s.items) == 10  # still 10 rows


# ─── id reuse / update in place ──────────────────────────────────────────

def test_add_item_with_existing_id_updates_in_place(tmp_path: Path):
    with _make_store(tmp_path) as s:
        s.add_item(Item(id="a", text="v1", hash="h1"), _vec(1))
        s.add_item(Item(id="a", text="v2", hash="h2"), _vec(2))
        # Single item, latest version
        assert len(s.items) == 1
        assert s.items[0].text == "v2"


# ─── lock contention ─────────────────────────────────────────────────────

def test_second_writer_is_blocked(tmp_path: Path):
    s1 = _make_store(tmp_path)
    try:
        with pytest.raises(StoreError, match="lock"):
            _make_store(tmp_path)
    finally:
        s1.close()
    # After release, a new writer can attach.
    s2 = _make_store(tmp_path)
    s2.close()


# ─── stats shape ─────────────────────────────────────────────────────────

def test_stats_shape(tmp_path: Path):
    with _make_store(tmp_path) as s:
        s.add_item(Item(id="a", text="x", hash="h"), _vec(1))
        st = s.stats()
        assert st["items"] == 1
        assert st["embedder"] == "test:8"
        assert st["embed_dim"] == 8
        assert st["dirty"] is True
        s.flush()
        st = s.stats()
        assert st["dirty"] is False
        assert st["version"] == 1
        assert isinstance(st["file_age_seconds"], int)


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
