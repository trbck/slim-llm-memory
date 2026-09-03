"""Memory — the public surface.

Tiny wrapper around :class:`Store` and :class:`Embedder`:

  * ``upsert``: hash-skip if text unchanged; embed only the deltas.
  * ``search``: numpy linear scan with cosine, optional filters.
  * ``find_duplicates``: pairwise upper-triangle, union-find clusters.
  * ``stats``: composes Store + Obs counters.

That's it. No background threads, no implicit network calls beyond
the embedder, no global state.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .embed import Embedder, EmbedderError
from .obs import Obs
from .store import Item, Store, StoreError

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return v / ||v||₂. Safe for zero vectors (returns zeros)."""
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v.astype(np.float32, copy=False)
    return (v / norm).astype(np.float32, copy=False)


@dataclass
class Hit:
    id: str
    score: float
    text: str
    meta: dict[str, Any]


class Memory:
    """Persistent vector memory backed by ``Store`` + an ``Embedder``."""

    def __init__(self, path: str | Path, embedder: Embedder) -> None:
        if not isinstance(embedder, Embedder):
            raise TypeError("embedder must be an Embedder instance")

        # Probe dim if the embedder doesn't know yet (e.g. unknown ollama model).
        if embedder.dim == 0:
            try:
                probe = embedder.embed(["__probe__"])
            except EmbedderError as exc:
                raise RuntimeError(
                    f"could not probe embedder '{embedder.name}': {exc}"
                ) from exc
            if not probe or not probe[0]:
                raise RuntimeError(f"embedder '{embedder.name}' returned no vector for probe")
            embedder.dim = len(probe[0])
            logger.info("probed embedder dim: %d", embedder.dim)

        self.embedder = embedder
        self.obs = Obs()
        self.store = Store(path, embedder.name, embedder.dim)

    # ─── lifecycle ────────────────────────────────────────────────────────
    def close(self, flush: bool = True) -> None:
        if flush:
            self.flush()
        self.store.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ─── upsert ───────────────────────────────────────────────────────────
    def upsert(self, items: Iterable[dict]) -> dict:
        """Insert or update items.

        Each item dict needs ``id`` and ``text``; ``meta`` optional.
        Items whose text-hash already matches the stored hash are
        meta-updated only — the embedder is not called.

        Returns counts: ``{added, updated, skipped, embed_calls}``.
        """
        items = list(items)
        if not items:
            return {"added": 0, "updated": 0, "skipped": 0, "embed_calls": 0}

        added = updated = skipped = embed_calls = 0
        # Pre-compute hashes and decide who needs embedding.
        to_embed: list[tuple[int, str]] = []   # (input_index, text)
        records: list[Item] = []

        for inp in items:
            iid = (inp or {}).get("id")
            text = (inp or {}).get("text")
            meta = (inp or {}).get("meta", {}) or {}
            if not iid or text is None:
                raise ValueError("each upsert item needs non-empty 'id' and 'text'")
            h = _content_hash(text)
            existing = self.store._id_to_idx.get(iid)
            if existing is not None and self.store.items[existing].hash == h:
                # Same text → meta-only update, skip embed.
                self.store.update_meta(iid, meta)
                self.obs.count("upsert.skipped_text_unchanged")
                skipped += 1
                records.append(self.store.items[existing])
                continue
            records.append(Item(id=iid, text=text, hash=h, meta=meta))
            to_embed.append((len(records) - 1, text))

        if to_embed:
            t0 = time.time()
            try:
                vectors = self.embedder.embed([t for _, t in to_embed])
                embed_calls = 1
            except EmbedderError as exc:
                self.obs.record_error("embed", "embed_batch", exc)
                raise
            except Exception as exc:
                self.obs.record_error("embed", "embed_batch", exc)
                raise
            duration_ms = (time.time() - t0) * 1000
            self.obs.record_slow("embed", duration_ms)
            self.obs.count("embed.calls")
            self.obs.count("embed.items", len(to_embed))

            if len(vectors) != len(to_embed):
                raise EmbedderError(
                    f"embedder returned {len(vectors)} vectors for {len(to_embed)} inputs"
                )

            for (rec_idx, _), vec in zip(to_embed, vectors):
                rec = records[rec_idx]
                v = np.asarray(vec, dtype=np.float32)
                if v.shape != (self.embedder.dim,):
                    raise EmbedderError(
                        f"embedder returned vector of shape {v.shape}, expected ({self.embedder.dim},)"
                    )
                # Normalise upfront so search doesn't have to.
                v = _normalise(v)
                # Was this an update (same id existed) or a new insert?
                pre_existing = self.store._id_to_idx.get(rec.id)
                self.store.add_item(rec, v)
                if pre_existing is not None:
                    updated += 1
                    self.obs.count("upsert.updated")
                else:
                    added += 1
                    self.obs.count("upsert.added")

        return {"added": added, "updated": updated, "skipped": skipped, "embed_calls": embed_calls}

    # ─── update / remove ──────────────────────────────────────────────────
    def update_text(self, item_id: str, text: str) -> bool:
        """Convenience: re-embed and replace text for an existing id."""
        if item_id not in self.store._id_to_idx:
            return False
        result = self.upsert([{"id": item_id, "text": text,
                               "meta": self.store.items[self.store._id_to_idx[item_id]].meta}])
        return (result["added"] + result["updated"]) > 0

    def remove(self, item_id: str) -> bool:
        ok = self.store.mark_deleted(item_id)
        if ok:
            self.obs.count("remove")
        return ok

    # ─── search ───────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        k: int = 10,
        kinds: set[str] | None = None,
        min_score: float = 0.0,
    ) -> list[Hit]:
        """Top-k semantic neighbours of ``query``.

        ``kinds`` (optional) filters by ``meta.kind``. ``min_score``
        is cosine cutoff in [-1, 1]; default 0 returns positive matches
        only.
        """
        if not query or not query.strip():
            return []
        if k < 1:
            return []
        if self.store.vectors.shape[0] == 0:
            return []

        t0 = time.time()
        try:
            qv_raw = self.embedder.embed([query])
        except Exception as exc:
            self.obs.record_error("embed", "search.embed", exc)
            raise
        if not qv_raw:
            return []
        qv = _normalise(np.asarray(qv_raw[0], dtype=np.float32))

        # All store vectors are pre-normalised → dot product == cosine.
        scores = self.store.vectors @ qv  # shape (N,)

        # Build mask for tombstones + kind filter.
        mask = np.ones(scores.shape[0], dtype=bool)
        for i, it in enumerate(self.store.items):
            if it.deleted:
                mask[i] = False
            elif kinds is not None and (it.meta.get("kind") not in kinds):
                mask[i] = False

        if min_score > 0:
            mask &= scores >= min_score

        valid = np.where(mask)[0]
        if valid.size == 0:
            return []

        # argpartition for top-k, then sort the small slice.
        kk = min(k, valid.size)
        candidate_scores = scores[valid]
        if kk < valid.size:
            top_idx = np.argpartition(-candidate_scores, kk - 1)[:kk]
        else:
            top_idx = np.arange(valid.size)
        top_idx = top_idx[np.argsort(-candidate_scores[top_idx])]
        chosen = valid[top_idx]

        hits: list[Hit] = []
        for i in chosen:
            it = self.store.items[int(i)]
            hits.append(Hit(
                id=it.id,
                score=float(scores[i]),
                text=it.text,
                meta=dict(it.meta),
            ))

        duration_ms = (time.time() - t0) * 1000
        self.obs.record_slow("search", duration_ms)
        self.obs.count("search.calls")
        return hits

    # ─── duplicates ───────────────────────────────────────────────────────
    def find_duplicates(self, threshold: float = 0.86) -> list[list[str]]:
        """Cluster open items by cosine similarity ≥ threshold.

        Returns a list of clusters; each cluster is a list of item ids.
        Singletons are omitted. Pairwise upper-triangle scan — fine up
        to ~50k items, slow above.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")

        # float32 dot of identical unit vectors lands at 0.99999988, so a
        # literal 1.0 would never match. Tolerate float32 rounding.
        eff_threshold = min(threshold, 1.0 - 1e-6)

        open_idx = list(self.store.open_indices())
        n = len(open_idx)
        if n < 2:
            return []

        V = self.store.vectors[open_idx]  # already normalised
        # Cosine matrix via M @ M.T. n×n float32 — for 50k that's ~10 GB,
        # so chunk by row to keep memory bounded at ~ ROW_CHUNK × n.
        ROW_CHUNK = 1024
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for start in range(0, n, ROW_CHUNK):
            stop = min(start + ROW_CHUNK, n)
            block = V[start:stop] @ V.T  # shape (chunk, n)
            for r in range(stop - start):
                row_global = start + r
                # Only inspect upper triangle (col > row).
                row = block[r]
                cols = np.where(row >= eff_threshold)[0]
                for c in cols:
                    if c > row_global:
                        union(row_global, int(c))

        clusters: dict[int, list[str]] = {}
        for x in range(n):
            root = find(x)
            clusters.setdefault(root, []).append(self.store.items[open_idx[x]].id)
        return [c for c in clusters.values() if len(c) >= 2]

    # ─── stats / flush ────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        s = self.store.stats()
        s.update(self.obs.stats())
        return s

    def flush(self, force: bool = False) -> bool:
        return self.store.flush(force=force)
