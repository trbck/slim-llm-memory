"""TopicStore — one small, fast numpy store per topic, queried for LLM context.

The scenario: an LLM (or a skill it runs) is working on *one topic*. All the
material for that topic lives in a single persistent ``Memory``; every prompt
is embedded once and matched against it in a numpy linear scan. The result is
a context block to prepend to the LLM call.

    store = TopicStore("./.topics/nginx", Embedder.ollama("nomic-embed-text"))
    store.add_docs({"setup.md": open("setup.md").read(), ...})   # chunked + hash-skipped
    ctx = store.context_for("how do I enable TLS?", k=4)
    llm(ctx.prompt + "\\n\\nQuestion: how do I enable TLS?")

Nothing here is clever: chunking is paragraph-based, the retrieval is
``Memory.search``. The point is the shape — a topic is a directory, a
prompt is one embed + one GEMV, and the caller gets a string.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .embed import Embedder
from .index import Hit, Memory

_PARA_SPLIT = re.compile(r"\n\s*\n")
_WS = re.compile(r"\s+")


def chunk_paragraphs(text: str, *, max_words: int = 120, min_words: int = 20) -> list[str]:
    """Greedy paragraph packer: merge consecutive paragraphs until ``max_words``.

    Short trailing paragraphs (headings, one-liners) are glued to their
    neighbour so every chunk carries at least ``min_words`` when possible.
    """
    paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    words = 0
    for p in paras:
        n = len(p.split())
        if buf and words + n > max_words:
            chunks.append("\n\n".join(buf))
            buf, words = [], 0
        buf.append(p)
        words += n
    if buf:
        if chunks and words < min_words:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(buf)
        else:
            chunks.append("\n\n".join(buf))
    return chunks


@dataclass
class Context:
    """What ``context_for`` returns: the hits, a formatted block, and timings."""

    query: str
    hits: list[Hit]
    embed_ms: float
    scan_ms: float
    prompt: str = field(default="")

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.scan_ms


class TopicStore:
    def __init__(self, path: str | Path, embedder: Embedder) -> None:
        self.memory = Memory(path, embedder)

    # ─── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        self.memory.close()

    def __enter__(self) -> "TopicStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ─── ingest ───────────────────────────────────────────────────────────
    def add_docs(self, docs: Mapping[str, str], *, max_words: int = 120,
                 min_words: int = 20) -> dict[str, int]:
        """Chunk every doc and upsert. Unchanged chunks are hash-skipped
        (no embed call); chunks that vanished from a doc are removed."""
        items: list[dict[str, Any]] = []
        new_ids: dict[str, set[str]] = {}
        for name, text in docs.items():
            ids: set[str] = set()
            for i, chunk in enumerate(chunk_paragraphs(text, max_words=max_words, min_words=min_words)):
                cid = f"{name}#{i}"
                ids.add(cid)
                items.append({"id": cid, "text": chunk, "meta": {"kind": "doc", "doc": name, "idx": i}})
            new_ids[name] = ids
        result = self.memory.upsert(items)
        removed = 0
        for it in list(self.memory.store.items):
            if it.deleted:
                continue
            doc = it.meta.get("doc")
            if doc in new_ids and it.id not in new_ids[doc]:
                self.memory.remove(it.id)
                removed += 1
        result["removed"] = removed
        result["chunks"] = len(items)
        return result

    def flush(self) -> bool:
        return self.memory.flush()

    # ─── retrieval ────────────────────────────────────────────────────────
    def context_for(self, prompt: str, *, k: int = 4, min_score: float = 0.3,
                    max_words: int = 600) -> Context:
        """Top-k chunks for ``prompt`` plus a ready-to-prepend context block.

        Timings are split so callers can see what they pay for: ``embed_ms``
        is the embedder round-trip, ``scan_ms`` the numpy ranking.
        """
        t0 = time.perf_counter()
        qv = self.memory.embedder.embed([prompt])
        t1 = time.perf_counter()
        hits = self.memory.search_vector(qv[0], k=k, min_score=min_score)
        t2 = time.perf_counter()

        lines = ["Context (retrieved for this prompt; cite by [n]):"]
        budget = max_words
        for n, h in enumerate(hits, start=1):
            words = h.text.split()
            if budget <= 0:
                break
            body = " ".join(words[:budget]) if len(words) > budget else h.text
            budget -= len(words)
            lines.append(f"[{n}] ({h.meta.get('doc')}, score {h.score:.2f})\n{body}")
        return Context(
            query=prompt,
            hits=hits,
            embed_ms=(t1 - t0) * 1000,
            scan_ms=(t2 - t1) * 1000,
            prompt="\n\n".join(lines) if hits else "",
        )

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()
