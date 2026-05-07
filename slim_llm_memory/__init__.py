"""slim-llm-memory — phase 1 (Memory core).

See docs/IMPLEMENTATION.md for the full plan. Phase 1 ships:

  * ``Memory`` — persistent vector index with hash-skip incremental upsert,
                 cosine search, duplicate clustering, atomic flush.
  * ``Embedder.noop()`` — deterministic SHA-256 derived vectors for tests.
  * ``Embedder.ollama(model, base_url=...)`` — local embeddings via Ollama.
  * ``Hit`` — search result dataclass.
  * ``Obs`` — per-instance observability (ring buffers + counters).

Phases 2+ (Gemini, Tier router, Graph, ANN swap) land later under the
same public API.
"""

from .embed import Embedder, EmbedderError
from .index import Hit, Memory
from .obs import Obs

__all__ = ["Memory", "Hit", "Embedder", "EmbedderError", "Obs"]
__version__ = "0.1.0"
