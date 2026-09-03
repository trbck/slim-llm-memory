"""Brain — the MCP-agnostic core of the Obsidian Brain.

Owns the single ``Memory`` writer, drains the spool, applies the flush
policy, and implements the six operations the MCP server exposes. Every
public method takes ``self.lock`` so a background drain thread and the
tool calls never interleave inside ``Memory``.
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slim_llm_memory import EmbedderError, Memory

from .parser import is_vault_markdown
from .spool import Spool

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


class BrainError(ValueError):
    """User-facing failure (bad path, unknown item, bad argument)."""


def _hit_dict(item_id: str, score: float | None, text: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item_id,
        "score": None if score is None else round(float(score), 4),
        "path": meta.get("path"),
        "title": meta.get("title"),
        "kind": meta.get("kind"),
        "tags": list(meta.get("tags") or []),
        "heading_path": list(meta.get("heading_path") or []),
        "text": text,
    }


class Brain:
    def __init__(
        self,
        vault: Path,
        memory: Memory,
        spool: Spool,
        *,
        flush_every_changes: int = 100,
        flush_every_seconds: float = 30.0,
    ) -> None:
        self.vault = Path(vault).expanduser().resolve()
        if not self.vault.is_dir():
            memory.close(flush=False)
            raise BrainError(f"vault directory does not exist: {self.vault}")
        self.memory = memory
        self.spool = spool
        self.flush_every_changes = int(flush_every_changes)
        self.flush_every_seconds = float(flush_every_seconds)
        self.lock = threading.RLock()

        self.inbox = self.vault / "inbox"
        self.inbox.mkdir(exist_ok=True)

        self._changes_since_flush = 0
        self._started_at = time.time()
        self._last_flush_ts: float | None = None
        self._last_drain_ts: float | None = None
        self._embed_failing = False
        # path → set(chunk ids) for stale-chunk removal and path lookups.
        self._by_path: dict[str, set[str]] = {}
        for it in self.memory.store.items:
            if not it.deleted:
                self._by_path.setdefault(str(it.meta.get("path")), set()).add(it.id)

    # ─── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        with self.lock:
            self.memory.close(flush=True)

    # ─── drain + flush ────────────────────────────────────────────────────
    def drain(self) -> dict[str, Any]:
        """Apply every pending spool file in order. Stops at the first
        embedder failure and leaves that file pending for the next tick.
        A structurally-bad individual entry (missing key, bad value) is
        logged and skipped so it can never permanently block the spool;
        an unreadable spool file (e.g. vanished mid-drain) is likewise
        logged and skipped rather than raised."""
        result = {"files": 0, "upserted": 0, "removed": 0, "embed_failed": False}
        with self.lock:
            stopped = False
            for path in self.spool.pending():
                try:
                    entries = self.spool.read(path)
                except OSError as exc:
                    logger.warning("could not read spool file %s, skipping: %s", path.name, exc)
                    continue
                embed_failed_here = False
                for e in entries:
                    try:
                        if e.get("op") == "file":
                            up, rm = self._apply_file(str(e.get("path")), e.get("chunks") or [])
                        elif e.get("op") == "remove":
                            up, rm = 0, self._remove_path(str(e.get("path")))
                        else:
                            logger.warning("unknown spool op %r in %s", e.get("op"), path.name)
                            continue
                        result["upserted"] += up
                        result["removed"] += rm
                    except EmbedderError as exc:
                        if not self._embed_failing:
                            logger.warning("embedder failing, will retry: %s", exc)
                        self._embed_failing = True
                        result["embed_failed"] = True
                        embed_failed_here = True
                        break
                    except Exception:
                        logger.exception("skipping bad spool entry in %s: %r", path.name, e)
                        continue
                if embed_failed_here:
                    stopped = True
                    break
                self.spool.mark_done(path)
                result["files"] += 1
            if not stopped:
                if self._embed_failing:
                    logger.info("embedder recovered")
                self._embed_failing = False
            self._last_drain_ts = time.time()
            self.spool.sweep_done()
            self.maybe_flush()
        return result

    def _apply_file(self, path: str, chunks: list[dict]) -> tuple[int, int]:
        new_ids = {c["id"] for c in chunks}
        stale = self._by_path.get(path, set()) - new_ids
        if chunks:
            self.memory.upsert([{"id": c["id"], "text": c["text"], "meta": c.get("meta") or {}} for c in chunks])
        removed = 0
        for sid in stale:
            if self.memory.remove(sid):
                removed += 1
        if new_ids:
            self._by_path[path] = new_ids
        else:
            self._by_path.pop(path, None)
        self._changes_since_flush += len(chunks) + removed
        return len(chunks), removed

    def _remove_path(self, path: str) -> int:
        removed = 0
        for sid in self._by_path.pop(path, set()):
            if self.memory.remove(sid):
                removed += 1
        self._changes_since_flush += removed
        return removed

    def maybe_flush(self, force: bool = False) -> bool:
        with self.lock:
            due_by_count = self._changes_since_flush >= self.flush_every_changes
            since = time.time() - (self._last_flush_ts or self._started_at)
            due_by_time = self.memory.store._dirty and since >= self.flush_every_seconds
            if not (force or due_by_count or due_by_time):
                return False
            wrote = self.memory.flush(force=force)
            self._changes_since_flush = 0
            self._last_flush_ts = time.time()
            return wrote

    # ─── operations ───────────────────────────────────────────────────────
    def search(self, query: str, k: int = 8, kinds: list[str] | None = None,
               min_score: float = 0.3) -> list[dict]:
        with self.lock:
            hits = self.memory.search(query, k=k, kinds=set(kinds) if kinds else None, min_score=min_score)
            return [_hit_dict(h.id, h.score, h.text, h.meta) for h in hits]

    def _safe_vault_path(self, rel: str) -> Path:
        p = (self.vault / rel).resolve()
        if not p.is_relative_to(self.vault):
            raise BrainError(f"path is outside the vault: {rel}")
        return p

    def _first_chunk_id(self, path: str) -> str | None:
        ids = self._by_path.get(path)
        if not ids:
            return None
        return min(ids, key=lambda i: int(i.rsplit("#", 1)[1]))

    def get(self, path: str) -> dict:
        with self.lock:
            p = self._safe_vault_path(path)
            if not p.is_file() or not is_vault_markdown(self.vault, p):
                raise BrainError(f"note not found: {path}")
            text = p.read_text(encoding="utf-8")
            first = self._first_chunk_id(path)
            meta: dict[str, Any] = {}
            if first is not None:
                meta = dict(self.memory.store.items[self.memory.store._id_to_idx[first]].meta)
            return {"path": path, "title": meta.get("title") or p.stem, "text": text, "meta": meta}

    def related(self, path_or_id: str, k: int = 5) -> list[dict]:
        with self.lock:
            item_id = path_or_id if "#" in path_or_id else self._first_chunk_id(path_or_id)
            if item_id is None or item_id not in self.memory.store._id_to_idx:
                raise BrainError(f"item not found in index: {path_or_id}")
            hits = self.memory.neighbours(item_id, k=k)
            return [_hit_dict(h.id, h.score, h.text, h.meta) for h in hits]

    def _open_items(self):
        return (it for it in self.memory.store.items if not it.deleted)

    def _one_per_file(self, items) -> list:
        """Keep the lowest section_idx chunk per path."""
        best: dict[str, Any] = {}
        for it in items:
            p = it.meta.get("path")
            cur = best.get(p)
            if cur is None or it.meta.get("section_idx", 0) < cur.meta.get("section_idx", 0):
                best[p] = it
        return list(best.values())

    def by_tag(self, tags: list[str], k: int = 20) -> list[dict]:
        wanted = {t.strip().lstrip("#") for t in (tags or []) if t and t.strip()}
        if not wanted:
            return []
        with self.lock:
            items = [it for it in self._open_items() if wanted & set(it.meta.get("tags") or [])]
            items = self._one_per_file(items)
            items.sort(key=lambda it: float(it.meta.get("mtime") or it.ts), reverse=True)
            return [_hit_dict(it.id, None, it.text, it.meta) for it in items[:k]]

    def recent(self, n: int = 10, kind: str | None = None) -> list[dict]:
        with self.lock:
            items = [it for it in self._open_items() if kind is None or it.meta.get("kind") == kind]
            items = self._one_per_file(items)
            items.sort(key=lambda it: float(it.meta.get("mtime") or it.ts), reverse=True)
            out = []
            for it in items[:n]:
                d = _hit_dict(it.id, None, it.text, it.meta)
                d["mtime"] = float(it.meta.get("mtime") or it.ts)
                out.append(d)
            return out

    def remember(self, text: str, tags: list[str] | None = None, title: str | None = None) -> dict:
        if not text or not text.strip():
            raise BrainError("text must not be empty")
        text = text.strip()
        slug_src = title if title and title.strip() else " ".join(text.split()[:6])
        slug = _SLUG.sub("-", slug_src.lower()).strip("-")[:60].strip("-") or "note"
        ts = datetime.now(timezone.utc)
        name = f"{ts.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}-{slug}.md"
        with self.lock:
            target = (self.inbox / name).resolve()
            if not target.is_relative_to(self.inbox.resolve()):
                raise BrainError("refusing to write outside vault/inbox")
            clean_tags = [t.strip().lstrip("#") for t in (tags or []) if t and t.strip()]
            fm: dict[str, Any] = {
                "tags": clean_tags,
                "source": "claude",
                "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            # Filesystem safety comes entirely from the slug regex above — the
            # frontmatter title is metadata, not a path, so it's written as-is
            # (properly YAML-escaped) with no path-shaped guard.
            if title and title.strip():
                fm["title"] = title.strip()
            import yaml  # lazy: part of the optional "obsidian" extra, like parser.py.
            dumped = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=None)
            target.write_text("---\n" + dumped + "---\n" + text + "\n", encoding="utf-8")
            rel = target.relative_to(self.vault).as_posix()
            return {"path": rel, "id": f"{rel}#0", "ingested": False}

    def stats(self) -> dict:
        with self.lock:
            s = self.memory.stats()
            s.update({
                "vault": str(self.vault),
                "spool_depth": self.spool.depth(),
                "last_drain_ts": self._last_drain_ts,
                "last_flush_ts": self._last_flush_ts,
                "changes_since_flush": self._changes_since_flush,
                "embed_failing": self._embed_failing,
            })
            return s
