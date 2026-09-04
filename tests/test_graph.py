from pathlib import Path

import pytest

from slim_llm_memory.graph import Graph, wikilinks


def test_wikilinks():
    assert wikilinks("see [[Alice]] and [[Long note#Section B]] or [[People/Bob|Bob]]; not \\[[x]] or [[Alice]]") == \
        ["Alice", "Long note", "People/Bob"]
    assert wikilinks("nothing") == []


def test_link_unlink_neighbours(tmp_path: Path):
    g = Graph(tmp_path / "graph.json")
    g.link("a", "b", "cites")
    g.link("a", "b", "uses", weight=0.5)
    g.link("b", "c", "cites")
    assert len(g) == 2 and set(g.nodes()) == {"a", "b", "c"}
    assert sorted(g.neighbours("a")) == [("b", "cites", 1.0), ("b", "uses", 0.5)]
    assert g.neighbours("a", relation="cites") == [("b", "cites", 1.0)]
    assert sorted({n for n, _, _ in g.neighbours("a", depth=2)}) == ["b", "c"]
    assert g.neighbours("c", direction="out") == [] and g.neighbours("c", direction="in") == [("b", "cites", 1.0)]
    assert g.neighbours("nope") == []
    assert g.unlink("a", "b", "uses") is True and g.neighbours("a") == [("b", "cites", 1.0)]
    assert g.unlink("a", "b", "uses") is False
    assert g.unlink("a", "b") is True and len(g) == 1
    assert g.drop("c") is True and g.drop("c") is False and len(g) == 0
    with pytest.raises(ValueError):
        g.link("a", "a")


def test_graph_persists(tmp_path: Path):
    p = tmp_path / "graph.json"
    g = Graph(p)
    g.link("a", "b", "cites", weight=2.0, note="why")
    g.save()
    assert p.exists() and "cites" in p.read_text()
    g2 = Graph(p)
    assert g2.edges() == [("a", "b", "cites", 2.0)]
    assert "1 edges" in repr(g2)


# ─── Topic integration ────────────────────────────────────────────────────

from slim_llm_memory import topic


def test_topic_link_related_and_wikilinks(tmp_path: Path):
    with topic("g", path=tmp_path / "g", embedder="noop:64", chunk_words=3, overlap=0) as t:
        t.add({"nginx.md": "install nginx\n\nsee [[certbot]] for tls", "certbot.md": "certbot issues certs",
               "pasta.md": "boil pasta", "other.md": "install nginx"})          # other.md#0 == nginx.md#0 text
        assert t.neighbours("nginx.md") == [("certbot.md", "links", 1.0)]         # wikilink → edge
        t.link("pasta.md", "certbot.md", relation="cites", weight=0.5)
        assert (t.path / "graph.json").exists() and len(t.graph) == 2
        r = t.related("nginx.md", k=3)
        ids = [h.meta["doc"] for h in r]
        assert ids[0] in ("other.md", "certbot.md")                               # identical text (0.6) or graph edge (0.4)
        assert "other.md" in ids and "certbot.md" in ids and "nginx.md" not in ids
        cert = next(h for h in r if h.meta["doc"] == "certbot.md")
        assert cert.meta["relation"] == "links" and cert.meta["via"] == "graph"
        assert r.mode == "related" and r.top.meta["related"] > 0
        with pytest.raises(KeyError):
            t.related("nope.md")
        assert t.unlink("pasta.md", "certbot.md") and len(t.graph) == 1
        t.forget("certbot.md")
        assert "certbot.md" not in t.graph.nodes()
    with topic("g", path=tmp_path / "g", embedder="noop:64", chunk_words=3, overlap=0) as t:
        assert "certbot.md" not in t.graph.nodes() and "nginx.md" in t.graph.nodes()   # persisted
