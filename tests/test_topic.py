"""Tests for the ``topic`` front door + Memory.search_vector — offline, noop embedder."""

from pathlib import Path

import pytest

from slim_llm_memory import Embedder, Memory, topic
from slim_llm_memory.topics import Added, Result, Topic, chunk_paragraphs


# ─── chunk_paragraphs ─────────────────────────────────────────────────────

def test_chunk_paragraphs_packs_to_budget():
    paras = [" ".join(f"w{i}" for i in range(50))] * 5      # 5 × 50 words
    out = chunk_paragraphs("\n\n".join(paras), max_words=120)
    assert [len(c.split()) for c in out] == [100, 100, 50]
    assert all("\n\n" in c for c in out[:2])


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
        assert mem.search_vector([0.0] * 64, k=2, min_score=0.01) == []
        with pytest.raises(ValueError):
            mem.search_vector([1.0] * 3, k=2)


# ─── topic() ──────────────────────────────────────────────────────────────

@pytest.fixture
def t(tmp_path: Path):
    tp = topic("Test Topic", path=tmp_path / "store", embedder="noop:64", chunk_words=3)
    yield tp
    tp.close()


A_MD = "para one about nginx\n\npara two about tls\n\npara three about certbot"
B_MD = "milch kaufen\n\nbrot kaufen"


def test_topic_factory_defaults_and_slug(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("slim_llm_memory.topics.DEFAULT_HOME", tmp_path / "home")
    tp = topic("My Topic!", embedder="noop:32")
    try:
        assert tp.path == tmp_path / "home" / "my-topic"
        assert tp.memory.embedder.name == "noop:32"
        assert len(tp) == 0 and tp.docs() == []
        assert "my-topic" in repr(tp)
    finally:
        tp.close()
    with pytest.raises(ValueError):
        topic("!!!", embedder="noop")                 # no path → name must slug to something
    with pytest.raises(ValueError):
        topic("x", path=tmp_path / "y", embedder="gemini:foo")


def test_add_text_file_dir_and_mapping(t: Topic, tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(A_MD, encoding="utf-8")
    (tmp_path / "docs" / "sub").mkdir()
    (tmp_path / "docs" / "sub" / "b.txt").write_text(B_MD, encoding="utf-8")
    (tmp_path / "docs" / "skip.bin").write_bytes(b"\x00")

    r = t.add(tmp_path / "docs")                       # directory → relative names
    assert isinstance(r, Added) and r and r.docs == 2 and r.chunks == 5 and r.embedded == 5
    assert t.docs() == ["a.md", "sub/b.txt"]

    r = t.add(str(tmp_path / "docs" / "a.md"), name="again.md")   # file with explicit name
    assert r.docs == 1 and "again.md" in t

    r = t.add("just a loose sentence")                 # raw text → note-<hash>
    assert r.chunks == 1 and any(d.startswith("note-") for d in t.docs())

    r = t.add({"c.md": "eins\n\nzwei", "d.md": "drei"})  # mapping
    assert r.docs == 2 and {"c.md", "d.md"} <= set(t.docs())
    assert "added 2 doc(s)" in repr(r)

    with pytest.raises(FileNotFoundError):
        t.add(tmp_path / "nope.md")
    with pytest.raises(ValueError):
        t.add("   ")


def test_add_is_incremental_and_removes_stale_chunks(t: Topic):
    t.add({"a.md": A_MD})
    r = t.add({"a.md": A_MD})
    assert not r and r.skipped == 3 and r.embedded == 0          # unchanged → no embed
    r = t.add({"a.md": "para one about nginx\n\nchanged"})
    assert r.embedded == 1 and r.skipped == 1 and r.removed == 1   # a#1 re-embedded, a#2 gone
    assert len(t) == 2
    assert t.forget("a.md") == 2 and len(t) == 0 and t.forget("a.md") == 0


def test_ask_returns_result_with_context_and_timings(t: Topic):
    t.add({"a.md": A_MD, "b.md": B_MD})
    r = t.ask("para two about tls", k=2, min_score=0.0)
    assert isinstance(r, Result) and len(r) == 2 and r
    assert r.top.id == "a.md#1"
    assert [h.id for h in r] == [h.id for h in r.hits]
    assert r.context.startswith("Context (retrieved for this prompt")
    assert "[1] (a.md, score 1.00)\npara two about tls" in r.context
    assert r.ms == r.embed_ms + r.scan_ms >= 0
    assert "2 hit(s)" in repr(r) and "a.md#1" in repr(r)


def test_ask_word_budget_and_no_hits(t: Topic):
    t.add({"a.md": A_MD, "b.md": B_MD})
    r = t.ask("para two about tls", k=5, min_score=0.0, max_words=4)
    assert "[1]" in r.context and "[2]" not in r.context
    r = t.ask("unrelated", min_score=0.99)
    assert not r and r.top is None and r.context == ""


def test_topic_persists_across_reopen(tmp_path: Path):
    with topic("p", path=tmp_path / "p", embedder="noop:64", chunk_words=3) as tp:
        tp.add({"a.md": A_MD, "b.md": B_MD})          # add() saves; no flush call needed
    with topic("p", path=tmp_path / "p", embedder="noop:64", chunk_words=3) as tp:
        assert len(tp) == 5 and tp.docs() == ["a.md", "b.md"]
        assert tp.ask("milch kaufen", k=1, min_score=0.0).top.id == "b.md#0"


def test_answer_calls_ollama_chat_with_context(t: Topic, monkeypatch):
    t.add({"a.md": A_MD})
    captured = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "  it is [1]  "}}

    def fake_post(url, json=None, timeout=None):
        captured["url"], captured["json"] = url, json
        return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)
    out = t.answer("para two about tls", model="m", min_score=0.0)
    assert out == "it is [1]"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "m"
    assert "para two about tls" in captured["json"]["messages"][1]["content"]


def test_topic_reopen_in_same_process_returns_open_object(tmp_path: Path):
    a = topic("r", path=tmp_path / "r", embedder="noop:64")
    b = topic("r", path=tmp_path / "r", embedder="noop:64")     # re-running a notebook cell
    assert b is a and not a.closed
    a.close()
    assert a.closed
    a.close()                                                    # idempotent
    c = topic("r", path=tmp_path / "r", embedder="noop:64")     # after close → a fresh object
    assert c is not a and not c.closed
    c.close()


def test_topic_open_elsewhere_gives_a_hint(tmp_path: Path):
    from slim_llm_memory.store import StoreError
    with Topic("x", path=tmp_path / "x", embedder="noop:64"):   # direct Topic() bypasses the registry
        with pytest.raises(StoreError, match="another process"):
            topic("x", path=tmp_path / "x", embedder="noop:64")
