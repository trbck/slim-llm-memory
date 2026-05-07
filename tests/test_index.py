"""Tests for Memory — search, dedup, hash-skip, persistence, filters."""

from pathlib import Path

import pytest

from slim_llm_memory import Embedder, Memory


@pytest.fixture
def mem(tmp_path: Path):
    m = Memory(tmp_path, Embedder.noop(dim=128))
    yield m
    m.close()


def test_upsert_and_search_basics(mem: Memory):
    r = mem.upsert([
        {"id": "n1", "text": "how to set up nginx", "meta": {"kind": "note"}},
        {"id": "n2", "text": "configure tls in nginx", "meta": {"kind": "note"}},
        {"id": "s1", "text": "milch kaufen", "meta": {"kind": "shopping"}},
    ])
    assert r["added"] == 3
    assert r["updated"] == 0
    assert r["embed_calls"] == 1

    hits = mem.search("how to set up nginx", k=1)
    assert len(hits) == 1
    assert hits[0].id == "n1"
    assert hits[0].score >= 0.99  # exact text → near-1.0 cosine via noop embedder


def test_hash_skip_avoids_reembed(mem: Memory):
    mem.upsert([{"id": "x", "text": "same content", "meta": {}}])
    r = mem.upsert([{"id": "x", "text": "same content", "meta": {"kind": "updated"}}])
    assert r["added"] == 0
    assert r["updated"] == 0
    assert r["skipped"] == 1
    assert r["embed_calls"] == 0  # hash matched → no embed call
    # meta updated though
    hits = mem.search("same content", k=1)
    assert hits[0].meta == {"kind": "updated"}


def test_text_change_re_embeds(mem: Memory):
    mem.upsert([{"id": "x", "text": "v1", "meta": {}}])
    r = mem.upsert([{"id": "x", "text": "v2 different", "meta": {}}])
    assert r["updated"] == 1
    assert r["embed_calls"] == 1


def test_search_kind_filter(mem: Memory):
    mem.upsert([
        {"id": "n1", "text": "linux nginx tuning", "meta": {"kind": "note"}},
        {"id": "s1", "text": "linux nginx tuning", "meta": {"kind": "shopping"}},
    ])
    hits = mem.search("linux nginx tuning", k=5, kinds={"note"})
    assert {h.id for h in hits} == {"n1"}


def test_search_min_score_filter(mem: Memory):
    mem.upsert([{"id": "a", "text": "alpha", "meta": {}}])
    # Query with totally different text — noop embedder produces low cosine.
    hits = mem.search("xyzzy plugh", k=5, min_score=0.99)
    assert hits == []


def test_search_empty_query_returns_empty(mem: Memory):
    mem.upsert([{"id": "a", "text": "x", "meta": {}}])
    assert mem.search("", k=5) == []
    assert mem.search("   ", k=5) == []


def test_search_zero_k_returns_empty(mem: Memory):
    mem.upsert([{"id": "a", "text": "x", "meta": {}}])
    assert mem.search("x", k=0) == []


def test_search_on_empty_index_returns_empty(mem: Memory):
    assert mem.search("anything", k=10) == []


def test_remove(mem: Memory):
    mem.upsert([
        {"id": "a", "text": "alpha", "meta": {}},
        {"id": "b", "text": "beta", "meta": {}},
    ])
    assert mem.remove("a") is True
    assert mem.remove("a") is False  # already tombstoned
    hits = mem.search("alpha", k=5)
    assert all(h.id != "a" for h in hits)


def test_update_text(mem: Memory):
    mem.upsert([{"id": "x", "text": "old", "meta": {"kind": "note"}}])
    assert mem.update_text("x", "completely new content") is True
    # meta preserved
    hits = mem.search("completely new content", k=1)
    assert hits[0].meta == {"kind": "note"}


def test_persistence_round_trip(tmp_path: Path):
    e = Embedder.noop(dim=64)
    with Memory(tmp_path, e) as m:
        m.upsert([
            {"id": "a", "text": "alpha", "meta": {"kind": "note"}},
            {"id": "b", "text": "beta", "meta": {"kind": "shopping"}},
        ])
        m.flush()
    with Memory(tmp_path, Embedder.noop(dim=64)) as m:
        hits = m.search("alpha", k=1)
        assert hits[0].id == "a"
        assert m.stats()["items_open"] == 2


def test_find_duplicates_clusters_identical_texts(mem: Memory):
    # Three items with the *same* text → same noop vector → cosine 1.0.
    mem.upsert([
        {"id": "a", "text": "duplicate me", "meta": {}},
        {"id": "b", "text": "duplicate me", "meta": {}},
        {"id": "c", "text": "duplicate me", "meta": {}},
        {"id": "d", "text": "different content entirely", "meta": {}},
    ])
    clusters = mem.find_duplicates(threshold=0.99)
    assert len(clusters) == 1
    assert set(clusters[0]) == {"a", "b", "c"}


def test_find_duplicates_no_singletons(mem: Memory):
    mem.upsert([
        {"id": "a", "text": "alpha", "meta": {}},
        {"id": "b", "text": "beta", "meta": {}},
    ])
    clusters = mem.find_duplicates(threshold=0.99)
    # Different texts → no duplicates; singletons not returned.
    assert clusters == []


def test_find_duplicates_threshold_validation(mem: Memory):
    with pytest.raises(ValueError):
        mem.find_duplicates(threshold=0.0)
    with pytest.raises(ValueError):
        mem.find_duplicates(threshold=1.5)


def test_stats_shape(mem: Memory):
    mem.upsert([{"id": "a", "text": "x", "meta": {}}])
    st = mem.stats()
    # Store keys
    assert st["items_open"] == 1
    assert st["embed_dim"] == 128
    # Obs keys
    assert "counters" in st
    assert "uptime_seconds" in st


def test_invalid_upsert_inputs(mem: Memory):
    with pytest.raises(ValueError):
        mem.upsert([{"id": "", "text": "x"}])
    with pytest.raises(ValueError):
        mem.upsert([{"id": "x"}])  # no text
