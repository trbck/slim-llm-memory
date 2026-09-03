from pathlib import Path

from slim_llm_memory.apps.obsidian.parser import (
    Chunk, is_vault_markdown, parse_file, parse_text, rel_path,
)


def ids(chunks: list[Chunk]) -> list[str]:
    return [c.id for c in chunks]


def test_frontmatter_title_tags_and_kind(vault: Path):
    [c] = parse_file(vault, vault / "People" / "Alice.md")
    assert c.id == "People/Alice.md#0"
    assert c.meta["title"] == "Alice Example"
    assert c.meta["kind"] == "People"
    assert c.meta["source"] == "vault"
    assert sorted(c.meta["tags"]) == ["colleague", "person"]   # yaml ∪ inline, deduped
    assert c.meta["section_idx"] == 0
    assert c.meta["heading_path"] == ["Alice Example"]
    assert c.text.startswith("# Alice")          # frontmatter stripped
    assert isinstance(c.meta["mtime"], float)


def test_h1_title_and_links_resolved(vault: Path):
    [c] = parse_file(vault, vault / "Daily" / "2026-05-20.md")
    assert c.meta["title"] == "2026-05-20"       # no fm, no H1 → stem
    assert c.meta["kind"] == "Daily"
    assert c.meta["tags"] == ["obsidian-brain"]
    # [[People/Alice]] resolves to an existing file; [[Long note#Section B]] → anchor dropped,
    # resolved by stem search to Projects/Long note.md
    assert c.meta["links"] == ["People/Alice.md", "Projects/Long note.md"]


def test_long_note_splits_on_h2(vault: Path):
    chunks = parse_file(vault, vault / "Projects" / "Long note.md")
    assert ids(chunks) == [f"Projects/Long note.md#{i}" for i in range(1, 5)]
    assert chunks[0].meta["heading_path"] == ["Long literature note"]
    assert chunks[1].meta["heading_path"] == ["Long literature note", "Section A"]
    assert chunks[3].meta["heading_path"] == ["Long literature note", "Section C"]
    for c in chunks:
        assert c.meta["title"] == "Long literature note"
        assert c.meta["tags"] == ["literature"]
        assert c.meta["links"] == ["People/Alice.md"]   # file-level links on every chunk


def test_quirks_invalid_yaml_code_fence_escaped_links(vault: Path):
    [c] = parse_file(vault, vault / "Projects" / "Quirks.md")
    assert c.meta["title"] == "Quirks"
    assert c.text.startswith("---\nthis is:")          # invalid fm → body unchanged
    assert c.meta["tags"] == ["real-tag"]
    assert c.meta["links"] == ["People/Alice.md"]


def test_no_title_uses_stem_and_root_kind(vault: Path):
    [c] = parse_file(vault, vault / "notitle.md")
    assert c.meta["title"] == "notitle"
    assert c.meta["kind"] == "root"


def test_inbox_kind_and_source(vault: Path):
    [c] = parse_file(vault, vault / "inbox" / "01J-capture.md")
    assert c.meta["kind"] == "inbox" and c.meta["source"] == "inbox"
    assert c.meta["tags"] == ["claude-session"]


def test_is_vault_markdown_filters(vault: Path):
    assert is_vault_markdown(vault, vault / "Daily" / "2026-05-20.md")
    assert not is_vault_markdown(vault, vault / ".obsidian" / "workspace.json")
    assert not is_vault_markdown(vault, vault / ".obsidian" / "x.md")
    assert not is_vault_markdown(vault, vault / "Daily" / "img.png")
    assert not is_vault_markdown(vault, vault.parent / "outside.md")
    assert rel_path(vault, vault / "Daily" / "2026-05-20.md") == "Daily/2026-05-20.md"


def test_non_utf8_is_skipped(vault: Path):
    bad = vault / "bad.md"
    bad.write_bytes(b"\xff\xfe not utf8")
    assert parse_file(vault, bad) == []


def test_parse_text_empty_body_yields_no_chunks(vault: Path):
    assert parse_text(vault, "Daily/empty.md", "---\ntags: [x]\n---\n\n", 0.0) == []
