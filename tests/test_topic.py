"""Tests for TopicStore + Memory.search_vector — offline, noop embedder."""

from pathlib import Path

import pytest

from slim_llm_memory import Embedder, Memory
from slim_llm_memory.topic import TopicStore, chunk_paragraphs


# ─── chunk_paragraphs ─────────────────────────────────────────────────────

def test_chunk_paragraphs_packs_to_budget():
    paras = [" ".join(f"w{i}" for i in range(50))] * 5      # 5 × 50 words
    text = "\n\n".join(paras)
    out = chunk_paragraphs(text, max_words=120)
    assert [len(c.split()) for c in out] == [100, 100, 50]
    assert all("\n\n" in c for c in out[:2])                   # paragraphs kept, joined


def test_chunk_paragraphs_glues_short_tail():
    text = " ".join(f"w{i}" for i in range(100)) + "\n\n# Heading only"
    out = chunk_paragraphs(text, max_words=120, min_words=20)
    assert len(out) == 1 and out[0].endswith("# Heading only")


def test_chunk_paragraphs_empty():
    assert chunk_paragraphs("") == []
    assert chunk_paragraphs("\n\n  \n") == []


# ─── Memory.search_vector ─────────────────────────────────────────────────

def test_search_vector_matches_search(tmp_path: Path):
    with Memory(tmp_path, Embedder.noop(64)) as mem:
        mem.upsert([{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}])
        qv = mem.embedder.embed(["alpha"])[0]
        via_vec = mem.search_vector(qv, k=2, min_score=0.0)
        via_txt = mem.search("alpha", k=2, min_score=0.0)
        assert [h.id for h in via_vec] == [h.id for h in via_txt] == ["a", "b"]
        assert abs(via_vec[0].score - 1.0) < 1e-5
        assert mem.search_vector([0.0] * 64, k=2, min_score=0.01) == []   # zero vector → all scores 0
        with pytest.raises(ValueError):
            mem.search_vector([1.0] * 3, k=2)                     # wrong dim


# ─── TopicStore ───────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path):
    s = TopicStore(tmp_path / "topic", Embedder.noop(64))
    yield s
    s.close()


DOCS = {
    "a.md": "para one about nginx\n\npara two about tls\n\npara three about certbot",
    "b.md": "milch kaufen\n\nbrot kaufen",
}


def test_add_docs_chunks_and_ids(store: TopicStore):
    r = store.add_docs(DOCS, max_words=3, min_words=1)
    assert r["chunks"] == 5 and r["added"] == 5 and r["removed"] == 0
    ids = {it.id for it in store.memory.store.items}
    assert ids == {"a.md#0", "a.md#1", "a.md#2", "b.md#0", "b.md#1"}
    assert store.memory.store.items[0].meta == {"kind": "doc", "doc": "a.md", "idx": 0}


def test_add_docs_is_incremental(store: TopicStore):
    store.add_docs(DOCS, max_words=3, min_words=1)
    r = store.add_docs(DOCS, max_words=3, min_words=1)
    assert r["skipped"] == 5 and r["embed_calls"] == 0            # nothing changed → no embed
    docs = dict(DOCS, **{"a.md": "para one about nginx\n\nchanged"})
    r = store.add_docs(docs, max_words=3, min_words=1)
    assert r["skipped"] == 3 and r["updated"] == 1 and r["removed"] == 1  # a#1 re-embedded, a#2 gone
    assert store.stats()["items_open"] == 4


def test_context_for_returns_block_and_timings(store: TopicStore):
    store.add_docs(DOCS, max_words=3, min_words=1)
    ctx = store.context_for("para two about tls", k=2, min_score=0.0)
    assert ctx.hits[0].id == "a.md#1"
    assert ctx.prompt.startswith("Context (retrieved for this prompt")
    assert "[1] (a.md, score 1.00)\npara two about tls" in ctx.prompt
    assert ctx.embed_ms >= 0 and ctx.scan_ms >= 0 and ctx.total_ms == ctx.embed_ms + ctx.scan_ms


def test_context_for_respects_word_budget(store: TopicStore):
    store.add_docs(DOCS, max_words=3, min_words=1)
    ctx = store.context_for("para two about tls", k=5, min_score=0.0, max_words=4)
    assert "[1]" in ctx.prompt and "[2]" not in ctx.prompt          # budget spent after hit 1


def test_context_for_empty_when_nothing_matches(store: TopicStore):
    store.add_docs(DOCS, max_words=3, min_words=1)
    ctx = store.context_for("totally unrelated", k=3, min_score=0.99)
    assert ctx.hits == [] and ctx.prompt == ""


def test_store_persists_across_reopen(tmp_path: Path):
    with TopicStore(tmp_path / "t", Embedder.noop(64)) as s:
        s.add_docs(DOCS, max_words=3, min_words=1)
        s.flush()
    with TopicStore(tmp_path / "t", Embedder.noop(64)) as s:
        assert s.stats()["items_open"] == 5
        assert s.context_for("milch kaufen", k=1, min_score=0.0).hits[0].id == "b.md#0"
