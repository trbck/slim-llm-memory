from slim_llm_memory.chunking import Chunk, chunk_text, sections


def words(n, p="w"):
    return " ".join(f"{p}{i}" for i in range(n))


def test_sections_split_on_headings_keep_preamble():
    text = "intro\n\n# Title\n\nbody one\n\n## Sub\n\nbody two"
    assert sections(text) == [(None, "intro\n\n"), ("# Title", "\n\nbody one\n\n"), ("## Sub", "\n\nbody two")]
    assert sections("no headings here") == [(None, "no headings here")]


def test_plain_text_packs_paragraphs_with_overlap():
    text = "\n\n".join([words(50, "a"), words(50, "b"), words(50, "c")])
    out = chunk_text(text, max_words=120, overlap=10)
    assert [c.idx for c in out] == [0, 1]
    assert out[0].heading is None and out[0].text.startswith("a0 ")
    assert out[1].text.startswith("… b40 b41")                 # last 10 words of the previous chunk
    assert out[1].text.endswith("c49")


def test_headings_prefix_every_chunk_of_their_section():
    text = "# Persistence model\n\n" + words(100, "p") + "\n\n" + words(100, "q") + "\n\n## Next\n\nshort"
    out = chunk_text(text, max_words=120, overlap=5)
    assert [c.heading for c in out] == ["# Persistence model", "# Persistence model", "## Next"]
    assert all(c.text.startswith(c.heading) for c in out)
    assert "… p95 p96 p97 p98 p99" in out[1].text                 # overlap only within the section
    assert "…" not in out[2].text


def test_no_overlap_and_short_tail_glue():
    text = words(100) + "\n\n# H\n\ntiny"
    out = chunk_text(text, max_words=120, overlap=0)
    assert len(out) == 2 and out[1].text == "# H\n\ntiny"
    text2 = words(100) + "\n\n" + "tail"
    assert len(chunk_text(text2, max_words=120)) == 1              # short tail glued (min_words = 20)


def test_empty_and_heading_only():
    assert chunk_text("") == []
    assert chunk_text("\n\n") == []
    out = chunk_text("# Only a heading")
    assert len(out) == 1 and out[0].text == "# Only a heading"
    assert isinstance(out[0], Chunk)
