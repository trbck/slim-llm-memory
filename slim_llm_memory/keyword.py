"""BM25 keyword index — the exact-token half of hybrid retrieval.

Dense embeddings match meaning and lose rare tokens (a product name, "20%",
``manifest.json``). BM25 is the opposite. ``Topic.ask(mode="hybrid")`` fuses
both by reciprocal rank.

Pure Python tokenising, numpy scoring: postings are stored as flat arrays
(CSR layout) so a query is a handful of gathers and one ``np.add.at``.
Build cost is linear in tokens; the index is cached on disk (``bm25.npz``)
keyed by the store version, so a topic pays it once per change.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'_-][a-z0-9]+)*%?")   # keeps v1.2, don't, items_open, 20%
_STOP = frozenset("""a an and are as at be by for from has have in is it its of on or that the this to was were
will with what which when where who how does do did i you we they not no""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 over a fixed list of documents (rows). ``search`` → [(row, score)]."""

    def __init__(self, docs: Iterable[str] | None = None, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = float(k1), float(b)
        self.n = 0
        self.vocab: dict[str, int] = {}
        self.term_ptr = np.zeros(1, dtype=np.int64)      # CSR row pointer over terms
        self.post_doc = np.zeros(0, dtype=np.int64)      # doc id per posting
        self.post_tf = np.zeros(0, dtype=np.float32)     # term frequency per posting
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.idf = np.zeros(0, dtype=np.float32)
        self.avgdl = 1.0
        if docs is not None:
            self.build(docs)

    # ─── build ────────────────────────────────────────────────────────────
    def build(self, docs: Iterable[str]) -> "BM25":
        vocab: dict[str, int] = {}
        per_term: dict[int, dict[int, int]] = {}
        doc_len: list[int] = []
        for d, text in enumerate(docs):
            toks = tokenize(text)
            doc_len.append(len(toks))
            counts: dict[int, int] = {}
            for t in toks:
                tid = vocab.setdefault(t, len(vocab))
                counts[tid] = counts.get(tid, 0) + 1
            for tid, c in counts.items():
                per_term.setdefault(tid, {})[d] = c
        self.n = len(doc_len)
        self.vocab = vocab
        self.doc_len = np.asarray(doc_len, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 1.0
        ptr = [0]
        docs_flat: list[int] = []
        tf_flat: list[int] = []
        df = np.zeros(len(vocab), dtype=np.float32)
        for tid in range(len(vocab)):
            posting = per_term.get(tid, {})
            df[tid] = len(posting)
            docs_flat.extend(posting.keys())
            tf_flat.extend(posting.values())
            ptr.append(len(docs_flat))
        self.term_ptr = np.asarray(ptr, dtype=np.int64)
        self.post_doc = np.asarray(docs_flat, dtype=np.int64)
        self.post_tf = np.asarray(tf_flat, dtype=np.float32)
        self.idf = np.log(1.0 + (self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        return self

    # ─── query ────────────────────────────────────────────────────────────
    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every row (zeros for rows sharing no term with the query)."""
        s = np.zeros(self.n, dtype=np.float32)
        if self.n == 0:
            return s
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        for t in set(tokenize(query)):
            tid = self.vocab.get(t)
            if tid is None:
                continue
            lo, hi = self.term_ptr[tid], self.term_ptr[tid + 1]
            d, tf = self.post_doc[lo:hi], self.post_tf[lo:hi]
            s[d] += self.idf[tid] * tf * (self.k1 + 1.0) / (tf + norm[d])
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        nz = np.flatnonzero(s)
        if nz.size == 0 or k < 1:
            return []
        kk = min(k, nz.size)
        top = nz[np.argpartition(-s[nz], kk - 1)[:kk]] if kk < nz.size else nz
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top]

    # ─── persistence ──────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        terms = np.array(list(self.vocab.keys()), dtype=object)
        np.savez(path, terms=terms, term_ptr=self.term_ptr, post_doc=self.post_doc, post_tf=self.post_tf,
                 doc_len=self.doc_len, idf=self.idf, params=np.array([self.k1, self.b, self.avgdl, self.n]))

    @classmethod
    def load(cls, path: Path) -> "BM25":
        z = np.load(path, allow_pickle=True)
        o = cls()
        o.k1, o.b, o.avgdl, n = (float(x) for x in z["params"])
        o.n = int(n)
        o.vocab = {str(t): i for i, t in enumerate(z["terms"])}
        o.term_ptr, o.post_doc, o.post_tf = z["term_ptr"], z["post_doc"], z["post_tf"]
        o.doc_len, o.idf = z["doc_len"], z["idf"]
        return o


def rrf(rankings: Iterable[Iterable[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: each ranking contributes 1/(k+rank) per item."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused
