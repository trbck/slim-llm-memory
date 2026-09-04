from pathlib import Path

from slim_llm_memory import evaluate, topic
from slim_llm_memory.evals import Case, Report


class _Hit:
    def __init__(self, id, text, doc):
        self.id, self.text, self.meta, self.score = id, text, {"doc": doc}, 0.5


class _Fake:
    """ask() returns a fixed ranking so ranks are deterministic."""
    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def ask(self, q, k=5, **kw):
        self.calls.append((q, k, kw))
        return self.ranking[:k]


HITS = [_Hit("a.md#0", "alpha text", "a.md"), _Hit("b.md#1", "beta manifest text", "b.md"), _Hit("c.md#0", "gamma", "c.md")]


def test_case_matching_rules():
    h = HITS[1]
    assert Case("q", "MANIFEST").matches(h)          # substring, case-insensitive
    assert Case("q", "b.md").matches(h)              # doc name
    assert Case("q", "b.md#1").matches(h)            # id prefix
    assert not Case("q", "delta").matches(h)


def test_evaluate_ranks_and_metrics():
    fake = _Fake(HITS)
    r = evaluate(fake, [("q1", "alpha"), ("q2", "manifest"), ("q3", "gamma"), ("q4", "nope")], k=2, mode="dense")
    assert isinstance(r, Report)
    assert [row.rank for row in r.rows] == [1, 2, None, None]      # gamma is rank 3 > k=2
    assert r.hit1 == 0.25 and r.hitk == 0.5
    assert abs(r.mrr - (1 + 0.5) / 4) < 1e-9
    assert r.summary() == {"hit@1": 0.25, "hit@2": 0.5, "mrr": 0.375}
    assert fake.calls[0] == ("q1", 2, {"mode": "dense"})          # ask kwargs pass through
    text = repr(r)
    assert "hit@1 0.25" in text and "—" in text and "expects 'nope'" in text


def test_evaluate_on_a_real_topic(tmp_path: Path):
    with topic("e", path=tmp_path / "e", embedder="noop:64", chunk_words=3) as t:
        t.add({"a.md": "boil pasta\n\nadd sauce", "b.md": "knead dough"})
        r = evaluate(t, [Case("boil pasta", "a.md"), ("knead dough", "knead")], k=3, min_score=0.0, label="noop")
        assert r.hit1 == 1.0 and "noop" in repr(r)
