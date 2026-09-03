"""JSONL spool between the ingest process and the mcp (index writer) process.

One file per watcher flush / sweep batch. Line schema:

    {"op": "file",   "path": "Projects/foo.md", "chunks": [{"id","text","meta"}, ...]}
    {"op": "remove", "path": "Projects/foo.md"}

Drain protocol (see brain.py): read pending files in name order, apply,
rename to ``.done`` on success, sweep ``.done`` older than 24 h.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from .parser import Chunk

logger = logging.getLogger(__name__)


def file_entry(path: str, chunks: list[Chunk]) -> dict:
    return {
        "op": "file",
        "path": path,
        "chunks": [{"id": c.id, "text": c.text, "meta": c.meta} for c in chunks],
    }


def remove_entry(path: str) -> dict:
    return {"op": "remove", "path": path}


class Spool:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _new_name(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        return f"{ts}Z-{secrets.token_hex(4)}.jsonl"

    def write(self, entries: list[dict]) -> Path | None:
        if not entries:
            return None
        final = self.dir / self._new_name()
        tmp = final.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e, ensure_ascii=False))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)   # readers never see a half-written file
        return final

    def pending(self) -> list[Path]:
        return sorted(p for p in self.dir.glob("*.jsonl") if p.is_file())

    def depth(self) -> int:
        return len(self.pending())

    def read(self, path: Path) -> list[dict]:
        out: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d malformed spool line skipped: %s", path.name, line_no, exc)
                    continue
                if isinstance(obj, dict) and "op" in obj:
                    out.append(obj)
                else:
                    logger.warning("%s:%d spool line without op skipped", path.name, line_no)
        return out

    def mark_done(self, path: Path) -> Path:
        done = path.with_suffix(".done")
        os.replace(path, done)
        return done

    def sweep_done(self, max_age_seconds: float = 86400) -> int:
        cutoff = time.time() - max_age_seconds
        n = 0
        for p in self.dir.glob("*.done"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    n += 1
            except OSError:
                pass
        return n
