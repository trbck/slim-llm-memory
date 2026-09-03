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
