import logging
import time
from pathlib import Path

import pytest
import yaml

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
    # Embedder.noop is hash-based, not semantic — only an exact text match scores ~1.0.
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


def test_drain_skips_poison_entry_and_continues(brain: Brain, caplog):
    chunks = parse_file(brain.vault, brain.vault / "People/Alice.md")
    brain.spool.write([
        {"op": "file", "path": "x.md", "chunks": [{"text": "t", "meta": {}}]},  # missing "id" -> poison
        file_entry("People/Alice.md", chunks),
    ])
    spool_name = brain.spool.pending()[0].name
    with caplog.at_level(logging.ERROR):
        r = brain.drain()
    assert r["upserted"] == 1 and r["files"] == 1 and r["removed"] == 0 and r["embed_failed"] is False
    assert brain.spool.pending() == []
    assert brain.stats()["items_open"] == 1
    assert any(
        rec.levelno >= logging.ERROR and spool_name in rec.getMessage()
        for rec in caplog.records
    )


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
    # Embedder.noop is hash-based, not semantic — only an exact text match scores ~1.0.
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
    fm = yaml.safe_load(txt.split("---\n")[1])
    assert fm["title"] == "../../etc/passwd"
    assert fm["tags"] == ["claude-session"]
    assert fm["source"] == "claude"
    # no title → slug from first words
    r2 = brain.remember("Remember to buy milk tomorrow morning early")
    assert r2["path"].endswith("-remember-to-buy-milk-tomorrow-morning.md")
    # tags/title with YAML-special characters (comma, bracket, quote, backslash) round-trip safely
    r3 = brain.remember("x", tags=["a,b", "c]d"], title='q"uote\\back')
    txt3 = (brain.vault / r3["path"]).read_text(encoding="utf-8")
    fm3 = yaml.safe_load(txt3.split("---\n")[1])
    assert fm3["tags"] == ["a,b", "c]d"]
    assert fm3["title"] == 'q"uote\\back'
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
