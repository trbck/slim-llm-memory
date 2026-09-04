"""``evaluate`` — does retrieval find the right chunk? hit@1, hit@k, MRR, in one call.

    from slim_llm_memory import evaluate

    report = evaluate(t, [
        ("which file is the atomic commit point?", "manifest"),   # expected term in the hit text
        ("what do I need from the supermarket?", "einkauf.md"),    # ...or the expected doc name
    ], k=5, mode="hybrid")
    report              # table: rank per question + hit@1 / hit@k / MRR
    report.hit1, report.hitk, report.mrr

A case matches a hit when the expected string is a substring of the hit's text
(case-insensitive), equals its ``meta["doc"]``, or prefixes its id.
Works on a ``Topic``, a ``Library``, or anything with ``.ask(question, k=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass
class Case:
    question: str
    expect: str

    def matches(self, hit: Any) -> bool:
        e = self.expect
        text = getattr(hit, "text", "") or ""
        meta = getattr(hit, "meta", {}) or {}
        hid = getattr(hit, "id", "") or ""
        return (e.lower() in text.lower()) or meta.get("doc") == e or hid.startswith(e)


@dataclass
class Row:
    question: str
    expect: str
    rank: int | None          # 1-based rank of the first matching hit, None if not in top-k


class Report:
    def __init__(self, rows: list[Row], k: int, label: str = "") -> None:
        self.rows = rows
        self.k = k
        self.label = label

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def hit1(self) -> float:
        return sum(r.rank == 1 for r in self.rows) / self.n if self.n else 0.0

    @property
    def hitk(self) -> float:
        return sum(r.rank is not None for r in self.rows) / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return sum(1.0 / r.rank for r in self.rows if r.rank) / self.n if self.n else 0.0

    def summary(self) -> dict[str, float]:
        return {"hit@1": round(self.hit1, 3), f"hit@{self.k}": round(self.hitk, 3), "mrr": round(self.mrr, 3)}

    def __repr__(self) -> str:
        head = f"evaluate({self.label + ', ' if self.label else ''}{self.n} cases, k={self.k}):  " \
               f"hit@1 {self.hit1:.2f} · hit@{self.k} {self.hitk:.2f} · MRR {self.mrr:.2f}"
        w = min(60, max((len(r.question) for r in self.rows), default=8))
        lines = [f"  {str(r.rank) if r.rank else '—':>4}  {r.question[:w]:<{w}}  expects {r.expect!r}" for r in self.rows]
        return "\n".join([head, *lines])


def evaluate(target: Any, cases: Iterable["Case | Sequence[str]"], k: int = 5, *,
             label: str = "", **ask_kwargs: Any) -> Report:
    """Run every case through ``target.ask`` and record the rank of the first correct hit."""
    rows: list[Row] = []
    for c in cases:
        case = c if isinstance(c, Case) else Case(str(c[0]), str(c[1]))
        hits = list(target.ask(case.question, k=k, **ask_kwargs))
        rank = next((i + 1 for i, h in enumerate(hits) if case.matches(h)), None)
        rows.append(Row(case.question, case.expect, rank))
    return Report(rows, k, label)
