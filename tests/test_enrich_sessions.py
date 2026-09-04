"""Enrichment (mocked model) and sessions — offline."""

from pathlib import Path

import pytest

from slim_llm_memory import library, session, topic
from slim_llm_memory import enrich as enrich_mod
from slim_llm_memory import llm as llm_mod
from slim_llm_memory.enrich import parse_extraction


def test_parse_extraction_is_tolerant():
    raw = 'Sure! {"entities": ["Postgres", "pgbouncer", "postgres"], "relations": [["Postgres","Depends On","pgbouncer"], ["x"]]} thanks'
    out = parse_extraction(raw)
    assert out["entities"] == ["Postgres", "pgbouncer"]
    assert out["relations"] == [("Postgres", "depends_on", "pgbouncer")]
    assert parse_extraction("no json") == {"entities": [], "relations": []}
    assert parse_extraction('{"entities": "bad"}')["entities"] == []


FAKE = {
    "postgres": {"entities": ["Postgres", "pgbouncer"], "relations": [["Postgres", "uses", "pgbouncer"]]},
    "nginx": {"entities": ["nginx", "certbot"], "relations": [["nginx", "uses", "certbot"]]},
}


def fake_extract(model, text, *, url, timeout=600.0):
    key = "postgres" if "Postgres" in text else "nginx"
    fake_extract.calls.append(model)
    return {"entities": list(FAKE[key]["entities"]), "relations": [tuple(r) for r in FAKE[key]["relations"]]}


fake_extract.calls = []


def test_add_enrich_sets_entities_edges_and_filters(tmp_path: Path, monkeypatch):
    import slim_llm_memory.topics as T
    monkeypatch.setattr(T, "_extract", fake_extract)
    fake_extract.calls.clear()
    with topic("e", path=tmp_path / "e", embedder="noop:64", chunk_words=50, overlap=0) as t:
        r = t.add({"db.md": "Postgres pool exhausted; added pgbouncer.", "web.md": "nginx fronts it with certbot certs."},
                  enrich="m")
        assert r.embedded == 2 and fake_extract.calls == ["m", "m"]
        assert t.entities() == {"Postgres": 1, "certbot": 1, "nginx": 1, "pgbouncer": 1}
        assert set(t.neighbours("Postgres")) >= {("pgbouncer", "uses", 1.0), ("db.md", "mentions", 1.0)}
        assert t.neighbours("nginx", relation="uses") == [("certbot", "uses", 1.0)]
        only = t.ask("pool", k=5, min_score=-1.0, entity="postgres")           # case-insensitive
        assert [h.meta["doc"] for h in only] == ["db.md"]
        assert t.ask("pool", k=5, min_score=-1.0, entity="nothing") .hits == []
        t.add({"db.md": "Postgres pool exhausted; added pgbouncer."}, enrich="m")   # unchanged → no model call
        assert fake_extract.calls == ["m", "m"]
    with library(tmp_path / "lib", embedder="noop:64", chunk_words=50, overlap=0) as db:
        db.topic("a").add({"db.md": "Postgres pool exhausted; added pgbouncer."}, enrich=True)
        assert fake_extract.calls[-1] == "llama3.2:3b"
        assert db.ask("pool", k=3, min_score=-1.0, entity="Postgres").top.meta["topic"] == "a"
        assert db.ask("pool", k=3, min_score=-1.0, entity="Redis").hits == []


def test_extract_uses_json_format(monkeypatch):
    seen = {}

    def fake_chat(model, messages, *, url, stream=False, timeout=600.0, options=None, fmt=None):
        seen["fmt"], seen["model"] = fmt, model
        return '{"entities": ["A"], "relations": []}'

    monkeypatch.setattr(enrich_mod, "chat", fake_chat)
    assert enrich_mod.extract("m", "A is here")["entities"] == ["A"]
    assert seen == {"fmt": "json", "model": "m"}


# ─── sessions ─────────────────────────────────────────────────────────────

def test_session_turns_history_recall(tmp_path: Path):
    with session("refactor day", path=tmp_path / "s", embedder="noop:64") as s:
        assert len(s) == 0 and "0 turns" in repr(s)
        assert s.turn("user", "the flaky test was the shared tmp dir") == "00001-user"
        assert s.turn("assistant", "fixed by using tmp_path per test") == "00002-assistant"
        s.turn("user", "keep numpy until 100k chunks")
        assert len(s) == 3
        assert s.history(2) == [("user", "keep numpy until 100k chunks")] or \
            s.history(2)[-1] == ("user", "keep numpy until 100k chunks")
        assert [r for r, _ in s.history()] == ["user", "assistant", "user"]
        assert s.transcript(1) == "user: keep numpy until 100k chunks"
        r = s.recall("fixed by using tmp_path per test", k=1, min_score=0.0)
        assert r.top.meta["role"] == "assistant" and r.top.meta["turn"] == 2
        assert list(s)[0][0] == "user"
    with session("refactor day", path=tmp_path / "s", embedder="noop:64") as s:
        assert len(s) == 3                                                   # persisted
        assert s.turn("user", "next") == "00004-user"


def test_session_summary_calls_model(tmp_path: Path, monkeypatch):
    import slim_llm_memory.sessions as S
    monkeypatch.setattr(S, "chat", lambda model, messages, **kw: f"- summary of: {messages[1]['content'][:20]}")
    with session("x", path=tmp_path / "x", embedder="noop:64") as s:
        assert s.summary() == ""
        s.turn("user", "hello there")
        assert s.summary(model="m").startswith("- summary of: user: hello")


def test_library_sessions_are_separate_from_topics(tmp_path: Path):
    with library(tmp_path / "lib", embedder="noop:64") as db:
        db.topic("nginx").add({"a.md": "install nginx"})
        s = db.session("Monday standup")
        s.turn("user", "we ship monday")
        assert db.sessions() == ["Monday standup"] and [i.name for i in db.topics()] == ["nginx"]
        assert (db.path / "_sessions" / "monday-standup" / "manifest.json").exists()
        assert db.ask("we ship monday", k=1, min_score=0.0).top.meta["topic"] == "nginx"   # sessions not searched
        assert s.recall("we ship monday", k=1, min_score=0.0).top.meta["role"] == "user"
