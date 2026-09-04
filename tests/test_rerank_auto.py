"""Adaptive reranking — the frozen ruler for the 2026-09-04 goal.

Offline and deterministic: a noop embedder, a counting stub reranker, and a
fixture where two documents are byte-identical so the contested case is exact
rather than statistical.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from slim_llm_memory import Reranker, library, topic
from slim_llm_memory.rerank import RERANK_MARGIN, should_rerank

TWIN = "para two about tls"
SOLO = "something entirely different here"
ROOT = Path(__file__).resolve().parent.parent


class Counting(Reranker):
    """Deterministic stub: longer text wins, and it records every call."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0
        self.pairs = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        self.pairs += len(texts)
        return [float(len(t)) for t in texts]


@pytest.fixture
def t(tmp_path: Path):
    tp = topic("r", path=tmp_path / "r", embedder="noop:64", chunk_words=50, overlap=0)
    tp.add({"twin-a.md": TWIN, "twin-b.md": TWIN, "solo.md": SOLO})
    yield tp
    tp.close()


# ─── M1: the decision is a pure function ──────────────────────────────────

def test_should_rerank_uses_relative_gap():
    # The leader's lead over the runner-up, measured against the pool's spread.
    assert should_rerank([1.0, 0.5, 0.0], margin=0.25) is False   # 50 % clear of the field
    assert should_rerank([1.0, 0.9, 0.0], margin=0.25) is True    # 10 % → contested
    assert should_rerank([1.0, 0.5, 0.0], margin=0.51) is True    # pins the comparison direction
    assert should_rerank([1.0, 0.5, 0.0], margin=0.5) is False    # exactly at the margin → clear
    assert should_rerank([1.0, 1.0, 1.0], margin=0.25) is True    # flat pool → contested
    assert should_rerank([0.5], margin=0.25) is False             # nothing to reorder
    assert should_rerank([], margin=0.25) is False
    assert 0.0 < RERANK_MARGIN < 1.0                              # a usable default exists


# ─── M2: the cross-encoder is configurable and batched ────────────────────

def test_cross_encoder_passes_max_length_and_batch_size(monkeypatch):
    seen: dict = {}

    class FakeCE:
        def __init__(self, model, max_length=None, **kw):
            seen["model"], seen["max_length"] = model, max_length

        def predict(self, pairs, **kw):
            seen["batch_size"] = kw.get("batch_size")
            seen["pairs"] = list(pairs)
            return [float(len(p[1])) for p in pairs]

    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = FakeCE
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    import slim_llm_memory.rerank as R
    R._MODELS.clear()
    try:
        rr = Reranker.cross_encoder("fake/model", max_length=256, batch_size=8)
        assert rr.score("q", ["aa", "b"]) == [2.0, 1.0]
        assert seen["model"] == "fake/model" and seen["max_length"] == 256
        assert seen["batch_size"] == 8
        assert seen["pairs"] == [("q", "aa"), ("q", "b")]
        assert rr.score("q", []) == []
        default = Reranker.cross_encoder()
        assert default.max_length == 256 and default.batch_size == 32
    finally:
        R._MODELS.clear()


# ─── M3: Topic.ask honours the policy and reports it ──────────────────────

def test_topic_auto_skips_when_clear_and_reports(t):
    rr = Counting()
    clear = t.ask(SOLO, k=2, mode="dense", min_score=-1.0, rerank=rr, rerank_margin=0.5)
    assert rr.calls == 0, "a decisive leader must not pay for a reranker call"
    assert clear.rerank_skipped is True and clear.reranked is None
    assert "rerank skipped" in repr(clear)

    contested = t.ask(TWIN, k=2, mode="dense", min_score=-1.0, rerank=rr, rerank_margin=0.5)
    assert rr.calls == 1 and contested.rerank_skipped is False
    assert contested.reranked == "counting" and contested.rerank_ms >= 0

    always = t.ask(SOLO, k=2, mode="dense", min_score=-1.0, rerank=rr)   # no margin → always
    assert rr.calls == 2 and always.rerank_skipped is False and always.reranked == "counting"

    off = t.ask(SOLO, k=2, mode="dense", min_score=-1.0)                  # no reranker at all
    assert rr.calls == 2 and off.rerank_skipped is False and off.reranked is None


def test_topic_auto_keeps_the_same_hits_it_would_have_returned(t):
    rr = Counting()
    skipped = t.ask(SOLO, k=3, mode="dense", min_score=-1.0, rerank=rr, rerank_margin=0.5)
    plain = t.ask(SOLO, k=3, mode="dense", min_score=-1.0)
    assert [h.id for h in skipped] == [h.id for h in plain]
    assert all("rerank" not in h.meta for h in skipped)


# ─── M4: Library.ask does the same across topics ──────────────────────────

def test_library_auto_matches_topic(tmp_path: Path):
    with library(tmp_path / "lib", embedder="noop:64", chunk_words=50, overlap=0) as db:
        db.topic("a").add({"twin-a.md": TWIN, "solo.md": SOLO})
        db.topic("b").add({"twin-b.md": TWIN})
        rr = Counting()
        clear = db.ask(SOLO, k=2, mode="dense", min_score=-1.0, rerank=rr, rerank_margin=0.5)
        assert rr.calls == 0 and clear.rerank_skipped is True and clear.reranked is None
        contested = db.ask(TWIN, k=2, mode="dense", min_score=-1.0, rerank=rr, rerank_margin=0.5)
        assert rr.calls == 1 and contested.rerank_skipped is False
        routed = db.ask(TWIN, k=2, mode="dense", min_score=-1.0, rerank=rr, rerank_margin=0.5, route=True)
        assert routed.rerank_skipped in (True, False) and rr.calls in (1, 2)


# ─── M5: the bench is a real, runnable artifact ───────────────────────────

def test_bench_offline_reports_three_policies(tmp_path: Path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "04_rerank_bench.py"), "--offline", "--json"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    rows = {r["policy"]: r for r in json.loads(proc.stdout)}
    assert set(rows) == {"off", "auto", "always"}
    for r in rows.values():
        assert 0.0 <= r["mrr"] <= 1.0 and r["rerank_calls"] >= 0 and r["ms"] >= 0
    assert rows["always"]["rerank_calls"] > rows["auto"]["rerank_calls"], "auto must save calls"
    assert rows["auto"]["mrr"] >= rows["off"]["mrr"], "auto must not cost quality"
