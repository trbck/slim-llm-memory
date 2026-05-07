"""Per-instance observability — ring buffers + counters.

No global state. Each ``Memory`` / ``Tier`` instance owns its own
``Obs`` and its own counters. Safe to call from any thread (Python's
GIL serialises the relevant ops; no atomics needed).

Surface is intentionally narrow:

    obs.count(key)                         # increment a counter
    obs.record_error(buffer, op, exc)      # push to a ring buffer
    obs.record_slow(op, duration_ms)       # ditto, for slow queries
    obs.stats()                            # JSON-safe dict

Caller wires ``stats()`` into their own ``/health`` endpoint or logs.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

# Ring-buffer caps. Keep tiny — these are debug aids, not metrics stores.
_BUFFER_CAP = 50
_SLOW_THRESHOLD_MS = 100


class Obs:
    """Observability surface attached to a single ``Memory`` or ``Tier``."""

    __slots__ = ("_counters", "_embed_errors", "_llm_errors", "_slow_queries", "_started_at")

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._embed_errors: deque[dict] = deque(maxlen=_BUFFER_CAP)
        self._llm_errors: deque[dict] = deque(maxlen=_BUFFER_CAP)
        self._slow_queries: deque[dict] = deque(maxlen=_BUFFER_CAP)
        self._started_at = time.time()

    # ─── counters ─────────────────────────────────────────────────────────
    def count(self, key: str, delta: int = 1) -> None:
        self._counters[key] = self._counters.get(key, 0) + delta

    # ─── error buffers ────────────────────────────────────────────────────
    def record_error(self, kind: str, op: str, exc: BaseException) -> None:
        """Push to the right ring buffer.

        kind: "embed" | "llm" | anything else (silently ignored).
        """
        entry = {
            "ts": time.time(),
            "op": op,
            "exc": str(exc)[:200] or type(exc).__name__,
        }
        if kind == "embed":
            self._embed_errors.append(entry)
        elif kind == "llm":
            self._llm_errors.append(entry)

    def record_slow(self, op: str, duration_ms: float) -> None:
        if duration_ms < _SLOW_THRESHOLD_MS:
            return
        self._slow_queries.append({
            "ts": time.time(),
            "op": op,
            "ms": round(duration_ms, 2),
        })

    # ─── helpers used by stats() and tests ───────────────────────────────
    def errors_within(self, kind: str, seconds: float) -> list[dict]:
        buf = self._embed_errors if kind == "embed" else self._llm_errors
        cutoff = time.time() - seconds
        return [e for e in buf if e["ts"] >= cutoff]

    def stats(self) -> dict[str, Any]:
        """Return a JSON-safe dict suitable for embedding in a /health endpoint."""
        now = time.time()
        return {
            "counters": dict(self._counters),
            "embed_errors_1h": len(self.errors_within("embed", 3600)),
            "embed_errors_24h": len(self.errors_within("embed", 86400)),
            "llm_errors_1h": len(self.errors_within("llm", 3600)),
            "llm_errors_24h": len(self.errors_within("llm", 86400)),
            "slow_queries": list(self._slow_queries)[-10:],
            "uptime_seconds": int(now - self._started_at),
            "buffer_cap": _BUFFER_CAP,
            "slow_threshold_ms": _SLOW_THRESHOLD_MS,
        }
