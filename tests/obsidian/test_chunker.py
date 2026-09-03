from slim_llm_memory.apps.obsidian.chunker import ChunkSlice, chunk, count_tokens


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_is_one_slice():
    out = chunk("hello world", max_tokens=800)
    assert out == [ChunkSlice(text="hello world", section_idx=0, heading=None)]


def test_count_tokens_is_whitespace():
    assert count_tokens("a  b\nc\t d") == 4
    assert count_tokens("") == 0


def test_splits_on_h2_when_too_long():
    text = "intro line\n\n## Alpha\n" + words(300, "a") + "\n\n## Beta\n" + words(300, "b")
    out = chunk(text, max_tokens=500)
    assert [s.section_idx for s in out] == [1, 2, 3]
    assert out[0].heading is None and out[0].text == "intro line"
    assert out[1].heading == "Alpha" and out[1].text.startswith("## Alpha\n")
    assert out[2].heading == "Beta" and "b299" in out[2].text


def test_h2_split_skips_empty_preamble():
    text = "## Alpha\n" + words(300, "a") + "\n## Beta\n" + words(300, "b")
    out = chunk(text, max_tokens=500)
    assert [s.heading for s in out] == ["Alpha", "Beta"]
    assert [s.section_idx for s in out] == [1, 2]


def test_no_h2_falls_back_to_sliding_window():
    text = words(1000)
    out = chunk(text, max_tokens=800, window=400, overlap=50)
    assert all(s.heading is None for s in out)
    assert [s.section_idx for s in out] == [1, 2, 3]
    # windows step by window-overlap = 350; last window contains the final token.
    assert out[0].text.startswith("w0 ") and "w399" in out[0].text
    assert out[1].text.startswith("w350 ")
    assert out[-1].text.endswith("w999")
    assert all(count_tokens(s.text) <= 400 for s in out)


def test_oversized_h2_section_falls_back_to_window():
    text = "## Big\n" + words(1200) + "\n## Small\nx y z"
    out = chunk(text, max_tokens=800, window=400, overlap=50)
    assert all(s.heading is None for s in out)
    assert len(out) > 2


def test_h3_does_not_split():
    text = "## A\n" + words(300) + "\n### sub\n" + words(300)
    out = chunk(text, max_tokens=500)
    assert len(out) == 1 and out[0].heading == "A"


def test_window_preserves_original_whitespace():
    text = "\n".join(f"line{i}" for i in range(1000))
    out = chunk(text, max_tokens=800, window=400, overlap=50)
    assert "\n" in out[0].text
