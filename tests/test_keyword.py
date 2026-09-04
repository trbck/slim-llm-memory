import time
from pathlib import Path

import numpy as np

from slim_llm_memory.keyword import BM25, rrf, tokenize


def test_tokenize_keeps_useful_tokens_and_drops_stopwords():
    assert tokenize("The manifest.json is the commit-point (v1.2), >20% tombstoned!") == \
        ["manifest.json", "commit-point", "v1.2", "20%", "tombstoned"]
    assert tokenize("") == []


DOCS = [
    "items.jsonl holds one record per item",
    "manifest.json is the atomic commit point of a flush",
    "compaction happens at flush when more than 20% of items are tombstoned",
    "the fcntl lock allows one writer per directory",
    "manifest manifest manifest",                     # term repeated: tf saturation, short doc
]


def test_bm25_ranks_exact_tokens():
    ix = BM25(DOCS)
    assert ix.n == 5
    top = ix.search("which file is the atomic commit point", k=3)
    assert top[0][0] == 1
    assert [d for d, _ in ix.search("manifest", k=5)] == [4]                 # "manifest.json" is a different token
    assert [d for d, _ in ix.search("manifest.json", k=5)] == [1]
    assert ix.search("zebra", k=3) == []                                       # unknown term → nothing
    assert ix.search("tombstoned 20%", k=1)[0][0] == 2
    assert len(ix.search("lock", k=10)) == 1


def test_bm25_scores_shape_and_zero_for_unrelated():
    ix = BM25(DOCS)
    s = ix.scores("fcntl lock")
    assert s.shape == (5,) and s[3] > 0 and s[0] == 0


def test_bm25_roundtrip(tmp_path: Path):
    ix = BM25(DOCS)
    ix.save(tmp_path / "bm25.npz")
    ix2 = BM25.load(tmp_path / "bm25.npz")
    q = "atomic commit point"
    assert ix2.search(q, k=3) == ix.search(q, k=3)
    assert ix2.vocab == ix.vocab and ix2.n == 5


def test_bm25_empty():
    ix = BM25([])
    assert ix.n == 0 and ix.search("x") == [] and ix.scores("x").shape == (0,)


def test_rrf_fuses_rankings():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert fused[1] == 1 / 61 + 1 / 62
    assert fused[3] == 1 / 63 + 1 / 61
    assert sorted(fused, key=fused.get, reverse=True)[0] == 1


def test_bm25_speed_at_50k_chunks():
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(5000)]
    docs = [" ".join(rng.choice(words, 100)) for _ in range(50_000)]
    t0 = time.perf_counter()
    ix = BM25(docs)
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        ix.search("w1 w2 w3 w4 w5 w6", k=10)
    query = (time.perf_counter() - t0) / 20 * 1000
    assert build < 30            # generous CI bound; the demo notebook reports the real number
    assert query < 50            # ms
