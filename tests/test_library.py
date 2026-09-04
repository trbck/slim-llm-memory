"""Tests for ``library`` — a folder of topic stores. Offline, noop embedder."""

from pathlib import Path

import pytest

from slim_llm_memory import library
from slim_llm_memory.libraries import ARCHIVE_DIR, Library, TopicInfo
from slim_llm_memory.topics import Topic


@pytest.fixture
def db(tmp_path: Path):
    lib = library(tmp_path / "lib", embedder="noop:64", chunk_words=3)
    yield lib
    lib.close()


NGINX = {"setup.md": "install nginx\n\nenable tls with certbot", "tuning.md": "worker processes"}
COOK = {"pasta.md": "boil pasta\n\nadd sauce", "bread.md": "knead dough"}


def fill(db: Library) -> None:
    db.topic("nginx").add(NGINX)
    db.topic("Cooking at home").add(COOK)


def test_layout_is_one_folder_per_topic(db: Library):
    fill(db)
    assert (db.path / "nginx" / "manifest.json").exists()
    assert (db.path / "cooking-at-home" / "topic.json").exists()
    assert (db.path / ARCHIVE_DIR).is_dir()
    assert sorted(p.name for p in (db.path / "nginx").iterdir() if p.suffix == ".npy") == ["vectors.v1.npy"]


def test_topics_listing_and_repr(db: Library):
    assert db.topics() == [] and len(db) == 0 and "empty" in repr(db)
    fill(db)
    infos = db.topics()
    assert [i.name for i in infos] == ["Cooking at home", "nginx"]        # display names, sorted by slug
    assert all(isinstance(i, TopicInfo) and not i.archived for i in infos)
    assert {(i.slug, i.docs, i.chunks) for i in infos} == {("nginx", 2, 3), ("cooking-at-home", 2, 3)}
    assert "nginx" in db and "Cooking at home" in db and "nope" not in db
    assert "2 active, 0 archived" in repr(db) and "Cooking at home" in repr(db)


def test_topics_counts_from_disk_without_opening(tmp_path: Path):
    with library(tmp_path / "lib", embedder="noop:64", chunk_words=3) as db:
        fill(db)
    db2 = library(tmp_path / "lib", embedder="noop:64", chunk_words=3)     # nothing open yet
    infos = db2.topics()
    assert {(i.slug, i.docs, i.chunks) for i in infos} == {("nginx", 2, 3), ("cooking-at-home", 2, 3)}
    assert db2._open == {}                                                 # listing did not open stores
    db2.close()


def test_topic_returns_same_open_object_and_getitem(db: Library):
    a = db.topic("nginx")
    assert isinstance(a, Topic) and db["nginx"] is a and db.topic("NGINX") is a


def test_ask_fans_out_and_merges_by_score(db: Library):
    fill(db)
    r = db.ask("boil pasta", k=3, min_score=0.0)
    assert r.top.id == "Cooking at home/pasta.md#0"
    assert r.top.meta["topic"] == "Cooking at home" and r.top.meta["doc"] == "pasta.md"
    assert r.routed is None                                          # exact (unrouted) by default
    assert "[1] (Cooking at home/pasta.md, score 1.00)" in r.context
    assert len(r) == 3
    d = db.ask("boil pasta", k=3, min_score=0.0, mode="dense")
    assert [h.score for h in d] == sorted((h.score for h in d), reverse=True)   # dense: cosine order

    only = db.ask("boil pasta", k=3, min_score=0.0, topics=["nginx"])
    assert only and all(h.meta["topic"] == "nginx" for h in only)
    with pytest.raises(KeyError):
        db.ask("x", topics=["nope"])


def test_archive_restore_delete(db: Library):
    fill(db)
    dst = db.archive("Cooking at home")
    assert dst == db.path / ARCHIVE_DIR / "cooking-at-home" and dst.is_dir()
    assert [i.name for i in db.topics()] == ["nginx"]
    assert [i.name for i in db.topics(archived=True)] == ["Cooking at home"]
    assert "1 active, 1 archived" in repr(db)
    assert "Cooking at home" in db                                          # still known

    assert not db.ask("boil pasta", min_score=0.0, topics=None) or \
        all(h.meta["topic"] != "Cooking at home" for h in db.ask("boil pasta", min_score=0.0))
    r = db.ask("boil pasta", k=1, min_score=0.0, include_archived=True)
    assert r.top.meta["topic"] == "Cooking at home" and r.top.meta.get("archived") is True
    r = db.ask("boil pasta", k=1, min_score=0.0, topics=["Cooking at home"])   # explicit name works too
    assert r.top.meta["topic"] == "Cooking at home"

    with pytest.raises(KeyError, match="archived"):
        db.topic("Cooking at home")
    assert db.archive("Cooking at home") == dst                             # idempotent

    back = db.restore("Cooking at home")
    assert back == db.path / "cooking-at-home" and db.topics(archived=True) == []
    assert db.topic("Cooking at home").docs() == ["bread.md", "pasta.md"]   # data survived the round trip

    db.delete("Cooking at home")
    assert "Cooking at home" not in db and not back.exists()
    with pytest.raises(KeyError):
        db.delete("Cooking at home")


def test_library_persists_and_reopens(tmp_path: Path):
    with library(tmp_path / "lib", embedder="noop:64", chunk_words=3) as db:
        fill(db)
        db.archive("nginx")
    with library(tmp_path / "lib", embedder="noop:64", chunk_words=3) as db:
        assert [i.name for i in db.topics()] == ["Cooking at home"]
        assert [i.name for i in db.topics(archived=True)] == ["nginx"]
        assert db.ask("enable tls with certbot", k=1, min_score=0.0, include_archived=True).top.meta["topic"] == "nginx"


# ─── routing ──────────────────────────────────────────────────────────────

def test_centroid_is_unit_mean_and_cached(db: Library):
    import numpy as np
    t = db.topic("nginx")
    assert not t.centroid().any()                                      # empty topic → zeros
    t.add(NGINX)
    c = t.centroid()
    assert abs(np.linalg.norm(c) - 1.0) < 1e-5
    assert t.centroid() is c                                           # cached while unchanged
    t.add({"more.md": "reload nginx"})
    assert t.centroid() is not c                                       # add → recomputed


def test_route_ranks_topics_and_chooses_within_margin(db: Library):
    fill(db)
    r = db.route("boil pasta")
    assert [n for n, _ in r.ranked][0] == "Cooking at home"
    assert r.chosen[0] == "Cooking at home" and len(r.ranked) == 2
    assert r.embed_ms >= 0 and r.route_ms >= 0 and "route(" in repr(r)
    assert db.route("boil pasta", m=1).chosen == ["Cooking at home"]


def test_route_falls_back_to_everything_when_no_topic_fits(db: Library, monkeypatch):
    fill(db)
    monkeypatch.setattr("slim_llm_memory.libraries.ROUTE_MIN_SCORE", 0.99)
    r = db.route("boil pasta")
    assert set(r.chosen) == {"nginx", "Cooking at home"}                # weak best → exact fallback


def test_ask_exact_is_default_for_small_libraries(db: Library):
    fill(db)
    r = db.ask("boil pasta", k=3, min_score=0.0)
    assert r.routed is None                                           # exact, unrouted
    assert r.top.id == "Cooking at home/pasta.md#0"
    assert "routed" not in repr(r)


def test_ask_routed_matches_exact_and_reports_topics(db: Library):
    fill(db)
    exact = db.ask("boil pasta", k=2, min_score=0.0, route=False)
    routed = db.ask("boil pasta", k=2, min_score=0.0, route=True)
    assert [h.id for h in routed] == [h.id for h in exact]
    assert routed.routed == ["Cooking at home"] and routed.routed_of == 2
    assert set(routed.per_topic_ms) == {"Cooking at home"}
    assert "routed to 1/2 topics" in repr(routed)
    assert db.ask("boil pasta", k=2, min_score=0.0, route=1).routed == ["Cooking at home"]


def test_ask_auto_routes_above_threshold(db: Library, monkeypatch):
    fill(db)
    assert db.ask("boil pasta", min_score=0.0).routed is None
    monkeypatch.setattr("slim_llm_memory.libraries.ROUTE_AUTO_THRESHOLD", 0)
    assert db.ask("boil pasta", min_score=0.0).routed == ["Cooking at home"]


def test_flat_cache_invalidates_on_change(db: Library):
    fill(db)
    db.topic("nginx").add({"ops.md": "reload nginx"})
    assert db.ask("reload nginx", k=1, min_score=0.0).top.id == "nginx/ops.md#0"   # new doc visible
    db.topic("nginx").forget("ops.md")
    assert db.ask("reload nginx", k=1, min_score=0.0).top.id != "nginx/ops.md#0"      # gone again
    db.archive("nginx")
    assert all(h.meta["topic"] != "nginx" for h in db.ask("reload nginx", k=3, min_score=0.0))


def test_flat_and_per_topic_paths_agree_on_scores(db: Library):
    fill(db)
    a = db.ask("enable tls with certbot", k=4, min_score=0.0, route=False, mode="dense")
    qv = db.embedder.embed(["enable tls with certbot"])[0]
    expected = []
    for name in ("nginx", "Cooking at home"):
        t = db.topic(name)
        expected += [(f"{name}/{h.id}", round(h.score, 5)) for h in t.memory.search_vector(qv, k=4, min_score=0.0)]
    expected.sort(key=lambda x: -x[1])
    assert [(h.id, round(h.score, 5)) for h in a] == expected[:4]


def test_library_hybrid_and_keyword_modes(db: Library):
    fill(db)
    kw = db.ask("certbot", k=2, mode="keyword", min_score=0.0)
    assert kw.top.id == "nginx/setup.md#1" and kw.top.meta["via"] == "keyword" and kw.mode == "keyword"
    hy = db.ask("certbot", k=3, min_score=0.0)                       # hybrid default
    assert "nginx/setup.md#1" in [h.id for h in hy.hits][:2]
    assert set(hy.per_topic_ms) == {"nginx", "Cooking at home"}
    routed = db.ask("certbot", k=2, min_score=0.0, route=True)
    assert routed.routed is not None and routed.mode == "hybrid"
