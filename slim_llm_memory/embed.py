"""Embedder factories — ``noop`` for tests, ``ollama`` for local embeddings.

The ``Embedder`` abstract is dead simple: a name, a dim, and an
``embed(texts) -> list[list[float]]`` synchronous method. Async
wrappers are the caller's job (asyncio.to_thread).

Two factories ship in phase 1:

    Embedder.noop(dim=384)
        Deterministic SHA-256-derived float32 vectors. Used by tests
        so they can run with zero network and zero models. Same text
        always produces the same vector, so equality assertions work.

    Embedder.ollama(model="nomic-embed-text", base_url=..., timeout=60)
        Calls the Ollama HTTP API at ``{base_url}/api/embed`` with a
        batched payload. Default base_url is the Ollama default
        (http://localhost:11434). Lazy-imports httpx.

``Embedder.gemini`` lives in phase 2 and isn't shipped here.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod


class EmbedderError(RuntimeError):
    """Raised when an embedder cannot return vectors."""


class Embedder(ABC):
    """The whole interface a callable embedder must satisfy."""

    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order. Raise ``EmbedderError``
        on any failure that left the batch unfulfilled.
        """
        raise NotImplementedError

    # ─── factories ────────────────────────────────────────────────────────

    @staticmethod
    def noop(dim: int = 384) -> "Embedder":
        return _NoopEmbedder(dim=dim)

    @staticmethod
    def ollama(
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        batch_size: int = 16,
    ) -> "Embedder":
        """``batch_size`` texts per HTTP request; ``timeout`` applies per request."""
        return _OllamaEmbedder(model=model, base_url=base_url, timeout=timeout, batch_size=batch_size)


# ─── noop ─────────────────────────────────────────────────────────────────

class _NoopEmbedder(Embedder):
    """Deterministic embedder for tests. Pure stdlib.

    Algorithm: SHA-256 of the text → repeat & truncate to ``dim`` bytes
    → unpack as float32 → normalise to unit length. Same text always
    yields the same vector, so test assertions are stable.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0 or dim > 8192:
            raise ValueError("dim must be in 1..8192")
        self.dim = dim
        self.name = f"noop:{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            seed = hashlib.sha256((t or "").encode("utf-8")).digest()
            # Stretch via repeated SHA-256 to fill ``dim`` bytes.
            buf = bytearray()
            cur = seed
            while len(buf) < self.dim:
                buf.extend(cur)
                cur = hashlib.sha256(cur).digest()
            # Each byte → float in [-1, 1) — bounded, deterministic, never
            # NaN or inf. (Raw-byte reinterpret as float32 produces NaN
            # bit patterns ~1% of the time, which breaks cosine maths.)
            vals = [(b - 127.5) / 127.5 for b in buf[:self.dim]]
            # Normalise to unit length so cosine math stays sane.
            norm = sum(v * v for v in vals) ** 0.5
            if norm == 0:
                vals = [1.0 / (self.dim ** 0.5)] * self.dim
                norm = 1.0
            out.append([v / norm for v in vals])
        return out


# ─── ollama ───────────────────────────────────────────────────────────────

# Models we know about — used only as a hint when probing dim. Ollama
# doesn't return dim in the embed response on older versions, so the first
# successful embed call sets the dim on the instance.
_KNOWN_OLLAMA_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


class _OllamaEmbedder(Embedder):
    def __init__(self, model: str, base_url: str, timeout: float, batch_size: int = 16) -> None:
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.batch_size = int(batch_size)
        self.name = f"ollama:{model}"
        self.dim = _KNOWN_OLLAMA_DIMS.get(model, 0)
        self._http = None  # lazy

    def _client(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(timeout=self.timeout)
        return self._http

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # One HTTP request per ``batch_size`` texts: a CPU-only Ollama embeds
        # ~100-word chunks at roughly 0.5–1 s each, so an unbounded batch of a
        # whole corpus would blow the per-request timeout.
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[start:start + self.batch_size]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama /api/embed accepts {"model": ..., "input": [str|list]}.
        payload = {"model": self.model, "input": texts}
        try:
            resp = self._client().post(f"{self.base_url}/api/embed", json=payload)
        except Exception as exc:
            raise EmbedderError(f"ollama transport error: {exc}") from exc

        if resp.status_code != 200:
            raise EmbedderError(
                f"ollama returned {resp.status_code}: {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except Exception as exc:
            raise EmbedderError(f"ollama returned non-JSON: {exc}") from exc

        # Newer Ollama: {"embeddings": [[...], [...]]}
        # Older: {"embedding": [...]} (single).  Normalise both.
        embeddings = data.get("embeddings")
        if embeddings is None and "embedding" in data:
            embeddings = [data["embedding"]]
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbedderError(
                f"ollama returned {len(embeddings) if isinstance(embeddings, list) else '?'} "
                f"vectors for {len(texts)} inputs"
            )

        # Set dim from the first successful response if we didn't know it.
        if self.dim == 0 and embeddings and isinstance(embeddings[0], list):
            self.dim = len(embeddings[0])

        # Validate shape.
        if self.dim:
            for i, v in enumerate(embeddings):
                if not isinstance(v, list) or len(v) != self.dim:
                    raise EmbedderError(
                        f"ollama returned mismatched vector at index {i} "
                        f"(expected dim {self.dim}, got {len(v) if isinstance(v, list) else type(v).__name__})"
                    )
        return embeddings  # type: ignore[return-value]
