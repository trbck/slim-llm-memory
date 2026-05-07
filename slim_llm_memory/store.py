"""Persistence layer for the slim memory.

Files in ``path/``:

    items.vN.jsonl       one record per item: {"id","text","hash","meta","ts","deleted"}
    vectors.vN.npy       float32 ndarray, shape (N, dim) — row-aligned with items.vN.jsonl
    manifest.json        {"version":N, "embedder":{"name":..., "dim":...}, "n":..., "ts":...}
                         — atomic commit point. Files are loaded only at the version
                         pointed to by the manifest. Writing a new manifest is the
                         single atomic operation that "commits" a flush.
    .lock                advisory exclusive fcntl lock; one writer per directory.

Invariants
----------
- ``vectors.vN.npy`` row ``i`` corresponds to ``items.vN.jsonl`` line ``i``.
- ``items.jsonl`` is append-only during a session; deletes are tombstones
  (``{"id":..., "deleted":true}``). Compaction happens at ``flush()`` when
  >20% tombstoned.
- A crash mid-flush leaves the previous manifest version intact — loading
  always picks the version pointed at by ``manifest.json``.
- Embedder mismatch (different name) refuses to load with a clear error.

Concurrency
-----------
- Single-writer enforced via ``fcntl.flock(LOCK_EX | LOCK_NB)`` on
  ``.lock``. Other readers can still snapshot the index by reading the
  files directly (they're written atomically) — but they must not mutate.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

_COMPACT_THRESHOLD = 0.20  # rewrite when >20% of items are tombstones


# ─── data class ───────────────────────────────────────────────────────────

@dataclass
class Item:
    id: str
    text: str
    hash: str  # sha1(text)[:16]
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    deleted: bool = False

    def to_jsonl(self) -> str:
        return json.dumps({
            "id": self.id, "text": self.text, "hash": self.hash,
            "meta": self.meta, "ts": self.ts,
            **({"deleted": True} if self.deleted else {}),
        }, ensure_ascii=False)


class StoreError(RuntimeError):
    """Raised on any persistence-layer failure."""


# ─── lock ────────────────────────────────────────────────────────────────

class _DirLock:
    """Best-effort exclusive lock on a directory using fcntl.flock.

    On non-posix platforms we silently no-op and trust the user not to
    run concurrent writers.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        try:
            import fcntl
        except ImportError:
            logger.warning("fcntl unavailable — store lock is a no-op on this platform")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._fd)
            self._fd = None
            raise StoreError(f"another writer holds the lock: {self.path}") from exc

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None


# ─── store ───────────────────────────────────────────────────────────────

class Store:
    """Persistent jsonl + npy backing for ``Memory``.

    Caller responsibilities:
      - Construct with a ``path`` (directory) and an embedder *fingerprint*
        (``name`` + ``dim``). The fingerprint is checked against the on-disk
        manifest; mismatch raises ``StoreError`` so callers can prompt for
        a clean rebuild.
      - Mutate via ``add_item / update_item / mark_deleted``. These are
        in-memory only; ``flush()`` writes to disk.
      - Search reads ``self.vectors`` and ``self.items`` directly.
    """

    def __init__(self, path: str | os.PathLike, embedder_name: str, embedder_dim: int) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder_name = embedder_name
        self.embedder_dim = int(embedder_dim)
        if self.embedder_dim <= 0:
            raise ValueError("embedder_dim must be positive (probe the embedder first)")

        self._lock = _DirLock(self.path / ".lock")
        self._lock.acquire()

        self.items: list[Item] = []
        self.vectors: np.ndarray = np.zeros((0, self.embedder_dim), dtype=np.float32)
        self._id_to_idx: dict[str, int] = {}
        self._version: int = 0
        self._dirty: bool = False

        self._load_or_init()

    # ─── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        # No auto-flush — caller decides. Just release the lock.
        self._lock.release()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ─── load / save ──────────────────────────────────────────────────────
    def _manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def _items_path(self, v: int) -> Path:
        return self.path / f"items.v{v}.jsonl"

    def _vectors_path(self, v: int) -> Path:
        return self.path / f"vectors.v{v}.npy"

    def _load_or_init(self) -> None:
        m = self._manifest_path()
        if not m.exists():
            self._version = 0
            return
        try:
            manifest = json.loads(m.read_text(encoding="utf-8"))
        except Exception as exc:
            raise StoreError(f"corrupt manifest at {m}: {exc}") from exc

        on_disk = manifest.get("embedder", {})
        if on_disk.get("name") != self.embedder_name:
            raise StoreError(
                f"embedder mismatch: index built with '{on_disk.get('name')}' "
                f"(dim={on_disk.get('dim')}) but configured embedder is "
                f"'{self.embedder_name}' (dim={self.embedder_dim}). "
                f"Delete the index directory or use the matching embedder."
            )
        if int(on_disk.get("dim", 0)) != self.embedder_dim:
            raise StoreError(
                f"embedder dim changed: on disk {on_disk.get('dim')}, configured {self.embedder_dim}"
            )

        self._version = int(manifest.get("version", 0))

        items_p = self._items_path(self._version)
        vectors_p = self._vectors_path(self._version)
        if not (items_p.exists() and vectors_p.exists()):
            raise StoreError(
                f"manifest points at version {self._version} but files missing: "
                f"items_exists={items_p.exists()} vectors_exists={vectors_p.exists()}"
            )

        # Load items
        loaded_items: list[Item] = []
        with items_p.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StoreError(f"corrupt items.jsonl at line {line_no+1}: {exc}") from exc
                loaded_items.append(Item(
                    id=obj["id"],
                    text=obj.get("text", ""),
                    hash=obj.get("hash", ""),
                    meta=obj.get("meta", {}) or {},
                    ts=float(obj.get("ts", 0.0)),
                    deleted=bool(obj.get("deleted", False)),
                ))

        # Load vectors
        try:
            vectors = np.load(vectors_p)
        except Exception as exc:
            raise StoreError(f"corrupt vectors.npy at {vectors_p}: {exc}") from exc
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[0] != len(loaded_items):
            raise StoreError(
                f"row count mismatch: {len(loaded_items)} items, {vectors.shape[0]} vectors"
            )
        if vectors.ndim != 2 or vectors.shape[1] != self.embedder_dim:
            raise StoreError(
                f"vectors shape mismatch: got {vectors.shape}, expected (N, {self.embedder_dim})"
            )

        self.items = loaded_items
        self.vectors = vectors
        self._id_to_idx = {it.id: i for i, it in enumerate(loaded_items) if not it.deleted}

    # ─── mutation (in-memory) ─────────────────────────────────────────────
    def add_item(self, item: Item, vector: np.ndarray) -> None:
        if vector.shape != (self.embedder_dim,):
            raise ValueError(f"vector shape {vector.shape} != ({self.embedder_dim},)")
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        # If id already exists (and not tombstoned), update in place.
        existing_idx = self._id_to_idx.get(item.id)
        if existing_idx is not None:
            self.items[existing_idx] = item
            self.vectors[existing_idx] = vector
        else:
            # Append
            self.items.append(item)
            self.vectors = np.vstack([self.vectors, vector[None, :]]) if self.vectors.size else vector[None, :].astype(np.float32)
            self._id_to_idx[item.id] = len(self.items) - 1
        self._dirty = True

    def update_meta(self, item_id: str, meta: dict[str, Any]) -> bool:
        idx = self._id_to_idx.get(item_id)
        if idx is None:
            return False
        self.items[idx].meta = dict(meta)
        self._dirty = True
        return True

    def mark_deleted(self, item_id: str) -> bool:
        idx = self._id_to_idx.get(item_id)
        if idx is None:
            return False
        self.items[idx].deleted = True
        del self._id_to_idx[item_id]
        self._dirty = True
        return True

    # ─── flush (atomic commit) ────────────────────────────────────────────
    def needs_compaction(self) -> bool:
        if not self.items:
            return False
        tomb = sum(1 for it in self.items if it.deleted)
        return tomb / len(self.items) > _COMPACT_THRESHOLD

    def flush(self, force: bool = False) -> bool:
        """Persist current state. Returns True if anything was written.

        Atomic: writes new versioned files first, then atomically swaps
        ``manifest.json`` to point at them. A crash before the manifest
        replace leaves the previous version intact. Old version files
        are best-effort cleaned up after the manifest write.
        """
        if not self._dirty and not force:
            return False

        # Compaction: drop tombstones if past threshold.
        if self.needs_compaction():
            keep = [(i, it) for i, it in enumerate(self.items) if not it.deleted]
            new_items = [it for _, it in keep]
            new_vectors = (
                np.vstack([self.vectors[i:i+1] for i, _ in keep])
                if keep else np.zeros((0, self.embedder_dim), dtype=np.float32)
            )
            self.items = new_items
            self.vectors = new_vectors
            self._id_to_idx = {it.id: i for i, it in enumerate(new_items)}

        new_v = self._version + 1
        items_p = self._items_path(new_v)
        vectors_p = self._vectors_path(new_v)

        # Write new versioned files.
        with items_p.open("w", encoding="utf-8") as fh:
            for it in self.items:
                fh.write(it.to_jsonl())
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

        np.save(vectors_p, self.vectors.astype(np.float32, copy=False))
        # np.save buffers internally; force fsync for durability.
        with vectors_p.open("rb") as fh:
            os.fsync(fh.fileno())

        # Atomically replace the manifest. This is the commit point.
        manifest = {
            "version": new_v,
            "embedder": {"name": self.embedder_name, "dim": self.embedder_dim},
            "n": len(self.items),
            "n_open": sum(1 for it in self.items if not it.deleted),
            "ts": time.time(),
        }
        manifest_p = self._manifest_path()
        tmp_p = manifest_p.with_suffix(".json.tmp")
        tmp_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_p, manifest_p)  # atomic on posix

        # Cleanup older versions (best-effort).
        prev_v = self._version
        self._version = new_v
        self._dirty = False
        if prev_v > 0:
            for stale in (self._items_path(prev_v), self._vectors_path(prev_v)):
                try:
                    stale.unlink(missing_ok=True)
                except OSError:
                    pass
        return True

    # ─── stats ────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        manifest_p = self._manifest_path()
        file_age = (
            int(time.time() - manifest_p.stat().st_mtime)
            if manifest_p.exists() else None
        )
        return {
            "items": len(self.items),
            "items_open": sum(1 for it in self.items if not it.deleted),
            "tombstones": sum(1 for it in self.items if it.deleted),
            "embedder": self.embedder_name,
            "embed_dim": self.embedder_dim,
            "version": self._version,
            "dirty": self._dirty,
            "file_age_seconds": file_age,
            "needs_compaction": self.needs_compaction(),
        }

    # ─── iteration ────────────────────────────────────────────────────────
    def open_indices(self) -> Iterable[int]:
        """Yield row indices for non-tombstoned items, in insertion order."""
        for i, it in enumerate(self.items):
            if not it.deleted:
                yield i
