"""answer() pipeline (mocked Ollama) and rerankers (noop) — offline."""

from pathlib import Path

import pytest

from slim_llm_memory import Reranker, library, topic
from slim_llm_memory import llm as llm_mod
from slim_llm_memory.llm import Answer, validate_citations
from slim_llm_memory.rerank import resolve

A_MD = "para one about nginx\n\npara two about tls\n\npara three about certbot"


@pytest.fixture
def t(tmp_path: Path):
    tp = topic("x", path=tmp_path / "x", embedder="noop:64", chunk_words=3, overlap=0)
    tp.add({"a.md": A_MD})
    yield tp
    tp.close()


# ─── citations / Answer ───────────────────────────────────────────────────

def test_validate_citations_drops_dangling_and_dedupes():
    clean, cited = validate_citations("It is manifest.json [1] and [3], not [9]. Also [1].", n_hits=3)
    assert clean == "It is manifest.json [1] and [3], not . Also [1]."
    assert cited == [1, 3]
    assert validate_citations("see [n2] and [n 1]", 3) == ("see [2] and [1]", [2, 1])


def test_answer_is_a_str_with_attributes():
    a = Answer("hello [1]", hits=[1], context="ctx", citations=[1], query="q")
    assert a == "hello [1]" and a.upper() == "HELLO [1]" and a.hits == [1] and not a.refused


# ─── grounded answer with a fake model ────────────────────────────────────

def test_answer_grounded_validates_citations_and_carries_hits(t, monkeypatch):
    seen = {}

    def fake_chat(model, messages, *, url, stream=False, timeout=600.0, options=None):
        seen["model"], seen["user"], seen["stream"] = model, messages[1]["content"], stream
        return "tls via certbot [1] and [7]."

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    a = t.answer("para two about tls", model="m", min_score=0.0, k=2)
    assert isinstance(a, Answer) and a == "tls via certbot [1] and ."
    assert a.citations == [1] and len(a.hits) == 2 and a.context in seen["user"] and seen["model"] == "m"
    assert "Question: para two about tls" in seen["user"] and not a.refused


def test_answer_refuses_without_calling_the_model(t, monkeypatch):
    monkeypatch.setattr(llm_mod, "chat", lambda *a, **k: pytest.fail("model must not be called"))
    a = t.answer("nothing matches this", min_score=0.99, mode="dense")
    assert a.refused and a == llm_mod.REFUSAL and a.hits == []
    b = t.answer("para two about tls", min_score=0.0, refuse_below=1.5)        # best cosine is 1.0 < 1.5
    assert b.refused and len(b.hits) > 0


def test_answer_rewrite_uses_rewritten_query_then_falls_back(t, monkeypatch):
    calls = []

    def fake_chat(model, messages, *, url, stream=False, timeout=600.0, options=None):
        calls.append(messages[0]["content"][:20])
        if messages[0]["content"].startswith("Rewrite"):
            return "para two about tls"                    # the rewrite hits an exact chunk
        return "answer [1]"

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    a = t.answer("please, could you tell me about TLS?", rewrite=True, min_score=0.0, k=1)
    assert a.query == "para two about tls" and a.hits[0].id == "a.md#1" and a == "answer [1]"
    assert calls[0].startswith("Rewrite") and len(calls) == 2


def test_answer_stream_yields_pieces(t, monkeypatch):
    monkeypatch.setattr(llm_mod, "chat", lambda *a, **k: iter(["par", "tial [1]"]) if k.get("stream") else "x")
    pieces = list(t.answer("para two about tls", min_score=0.0, stream=True))
    assert pieces == ["par", "tial [1]"]


def test_library_answer_uses_library_ask(tmp_path: Path, monkeypatch):
    with library(tmp_path / "lib", embedder="noop:64", chunk_words=3, overlap=0) as db:
        db.topic("a").add({"a.md": A_MD})
        monkeypatch.setattr(llm_mod, "chat", lambda *a, **k: "yes [1]")
        a = db.answer("para two about tls", min_score=0.0, k=1)
        assert a == "yes [1]" and a.hits[0].meta["topic"] == "a"


# ─── rerankers ────────────────────────────────────────────────────────────

def test_noop_reranker_scores_overlap():
    rr = Reranker.noop()
    assert rr.score("atomic commit point", ["the atomic commit point is manifest.json", "unrelated"]) == [1.0, 0.0]
    assert rr.score("", ["x"]) == [0.0] and rr.score("q", []) == []


def test_resolve_reranker():
    assert resolve(None) is None and resolve(False) is None
    assert resolve(Reranker.noop()).name == "noop"
    assert resolve(True).name.startswith("cross-encoder:")
    with pytest.raises(TypeError):
        resolve("yes")


def test_ask_with_reranker_reorders_and_annotates(t):
    rr = Reranker.noop()
    r = t.ask("certbot para three", k=2, mode="dense", min_score=-1.0, rerank=rr)
    assert r.top.id == "a.md#2" and r.top.meta["rerank"] == 1.0 and r.reranked == "noop"
    assert "reranked (noop)" in repr(r) and len(r) == 2
    assert all("rerank" in h.meta for h in r.hits)


def test_library_ask_with_reranker(tmp_path: Path):
    with library(tmp_path / "lib", embedder="noop:64", chunk_words=3, overlap=0) as db:
        db.topic("a").add({"a.md": A_MD})
        db.topic("b").add({"b.md": "milch kaufen\n\nbrot kaufen"})
        r = db.ask("brot kaufen", k=1, mode="dense", min_score=-1.0, rerank=Reranker.noop())
        assert r.top.id == "b/b.md#1" and r.reranked == "noop"
        r2 = db.ask("brot kaufen", k=1, min_score=-1.0, rerank=Reranker.noop(), route=True)
        assert r2.top.id == "b/b.md#1" and r2.routed is not None
