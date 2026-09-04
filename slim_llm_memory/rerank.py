"""Rerankers — a second, slower look at the top candidates.

    Reranker.cross_encoder()          # BAAI/bge-reranker-v2-m3 via sentence-transformers (optional extra)
    Reranker.noop()                   # token-overlap scorer for tests and offline runs

``Topic.ask(rerank=True)`` retrieves a pool of 4·k candidates, scores every
(query, chunk) pair with the reranker and keeps the best k. A cross-encoder reads
both texts together, which is why it beats cosine on exact questions — and why
it is too slow to run over a whole store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .keyword import tokenize

_MODELS: dict[str, object] = {}


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, higher is better. Scale is model-specific."""

    @staticmethod
    def cross_encoder(model: str = "BAAI/bge-reranker-v2-m3", *, max_length: int = 512) -> "Reranker":
        return _CrossEncoder(model, max_length)

    @staticmethod
    def noop() -> "Reranker":
        return _Overlap()


class _Overlap(Reranker):
    """Fraction of query tokens present in the text. Deterministic, dependency-free."""

    name = "noop"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(tokenize(t))) / len(q) for t in texts]


class _CrossEncoder(Reranker):
    def __init__(self, model: str, max_length: int) -> None:
        self.name = f"cross-encoder:{model}"
        self.model_name = model
        self.max_length = max_length

    def _model(self):
        m = _MODELS.get(self.model_name)
        if m is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError("pip install slim-llm-memory[rerank]  (sentence-transformers + torch)") from exc
            m = CrossEncoder(self.model_name, max_length=self.max_length)
            _MODELS[self.model_name] = m
        return m

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(x) for x in self._model().predict([(query, t) for t in texts])]


def resolve(rerank: "bool | Reranker | None") -> "Reranker | None":
    if rerank is None or rerank is False:
        return None
    if rerank is True:
        return Reranker.cross_encoder()
    if isinstance(rerank, Reranker):
        return rerank
    raise TypeError("rerank must be True, False or a Reranker")
