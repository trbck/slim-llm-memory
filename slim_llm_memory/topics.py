"""``topic`` — the human-friendly front door. One store per topic, three verbs.

    from slim_llm_memory import topic

    t = topic("nginx")                       # opens ~/.slim-llm-memory/topics/nginx (created if absent)
    t.add("docs/")                           # a file, a directory, raw text, or {name: text}
    r = t.ask("how do I enable TLS?")        # one embed call + one numpy scan
    r                                        # → hits, scores, timings (pretty repr)
    r.context                                # → the block to prepend to an LLM prompt
    t.answer("how do I enable TLS?")         # → retrieval + a local Ollama chat model

Everything is saved to disk after every ``add``/``forget``; nothing else to manage.
Defaults are Ollama ``nomic-embed-text`` for embeddings and ``llama3.2:3b`` for
``answer``; pass ``embedder="noop"`` for offline tests.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .embed import Embedder
from .index import Hit, Memory
from .store import StoreError

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SLUG = re.compile(r"[^a-z0-9._-]+")
_TEXT_SUFFIXES = {".md", ".txt", ".rst", ".markdown"}
DEFAULT_HOME = Path("~/.slim-llm-memory/topics")
DEFAULT_OLLAMA = "http://localhost:11434"


# ─── chunking ─────────────────────────────────────────────────────────────

def chunk_paragraphs(text: str, *, max_words: int = 120, min_words: int = 20) -> list[str]:
    """Greedy paragraph packer: merge consecutive paragraphs until ``max_words``.

    A short trailing paragraph (heading, one-liner) is glued to its
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


# ─── results ──────────────────────────────────────────────────────────────

@dataclass
class Added:
    """What ``Topic.add`` did. Truthy when anything changed."""
    docs: int
    chunks: int
    embedded: int
    skipped: int
    removed: int

    def __bool__(self) -> bool:
        return (self.embedded + self.removed) > 0

    def __repr__(self) -> str:
        return (f"added {self.docs} doc(s), {self.chunks} chunks: "
                f"{self.embedded} embedded, {self.skipped} unchanged, {self.removed} removed")


class Result:
    """What ``Topic.ask`` returns: the hits plus the timings, with a readable repr.

    ``r.hits``      list[Hit] (id, score, text, meta)
    ``r.top``       the best hit or None
    ``r.context``   the block to prepend to an LLM prompt (numbered, word-budgeted)
    ``r.ms``        total retrieval time; ``r.embed_ms`` / ``r.scan_ms`` for the split
    """

    def __init__(self, prompt: str, hits: list[Hit], embed_ms: float, scan_ms: float,
                 max_words: int) -> None:
        self.prompt = prompt
        self.hits = hits
        self.embed_ms = embed_ms
        self.scan_ms = scan_ms
        self._max_words = max_words

    @property
    def ms(self) -> float:
        return self.embed_ms + self.scan_ms

    @property
    def top(self) -> Hit | None:
        return self.hits[0] if self.hits else None

    @property
    def context(self) -> str:
        if not self.hits:
            return ""
        lines = ["Context (retrieved for this prompt; cite by [n]):"]
        budget = self._max_words
        for n, h in enumerate(self.hits, start=1):
            if budget <= 0:
                break
            words = h.text.split()
            body = " ".join(words[:budget]) if len(words) > budget else h.text
            budget -= len(words)
            where = h.meta.get("doc")
            if h.meta.get("topic"):
                where = f"{h.meta['topic']}/{where}"
            lines.append(f"[{n}] ({where}, score {h.score:.2f})\n{body}")
        return "\n\n".join(lines)

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __bool__(self) -> bool:
        return bool(self.hits)

    def __repr__(self) -> str:
        head = (f"ask({self.prompt!r})  {len(self.hits)} hit(s) · "
                f"embed {self.embed_ms:.0f} ms · scan {self.scan_ms:.2f} ms")
        rows = [f"  {n:>2}  {h.score:.2f}  {h.id:<24} {' '.join(h.text.split())[:70]}"
                for n, h in enumerate(self.hits, start=1)]
        return "\n".join([head, *rows])


# ─── topic ────────────────────────────────────────────────────────────────

def _make_embedder(spec: "str | Embedder", ollama_url: str) -> Embedder:
    if isinstance(spec, Embedder):
        return spec
    kind, _, arg = str(spec).partition(":")
    if kind == "ollama":
        return Embedder.ollama(arg or "nomic-embed-text", base_url=ollama_url)
    if kind == "noop":
        return Embedder.noop(int(arg or 384))
    raise ValueError(f"unknown embedder {spec!r}: use 'ollama[:model]', 'noop[:dim]' or an Embedder")


def _slug(name: str) -> str:
    s = _SLUG.sub("-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError("topic name must contain at least one letter or digit")
    return s


class Topic:
    """A named store of text about one subject. Use :func:`topic` to open one."""

    def __init__(self, name: str, *, path: "str | Path | None" = None,
                 embedder: "str | Embedder" = "ollama:nomic-embed-text",
                 ollama_url: str = DEFAULT_OLLAMA, chunk_words: int = 120) -> None:
        self.name = name
        self.path = Path(path) if path else DEFAULT_HOME.expanduser() / _slug(name)
        self.chunk_words = int(chunk_words)
        self.ollama_url = ollama_url.rstrip("/")
        self.memory = Memory(self.path, _make_embedder(embedder, self.ollama_url))
        self.closed = False

    # ─── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        """Save and release the store's lock. The object is not reusable afterwards."""
        if self.closed:
            return
        self.memory.close()
        self.closed = True
        _OPEN.pop(self.path.resolve(), None)

    def __enter__(self) -> "Topic":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __len__(self) -> int:
        return sum(1 for it in self.memory.store.items if not it.deleted)

    def __contains__(self, doc: str) -> bool:
        return doc in self.docs()

    def __repr__(self) -> str:
        return (f"topic({self.name!r}: {len(self.docs())} doc(s), {len(self)} chunks, "
                f"{self.memory.embedder.name}, {self.path})")

    def docs(self) -> list[str]:
        return sorted({it.meta.get("doc") for it in self.memory.store.items if not it.deleted})

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()

    # ─── add / forget ─────────────────────────────────────────────────────
    def add(self, source: "str | Path | Mapping[str, str] | Iterable[str | Path]",
            name: "str | None" = None) -> Added:
        """Add material. ``source`` may be:

        * a file path (``"notes.md"``) — name defaults to the file name
        * a directory — every ``.md/.txt/.rst`` file under it, named by relative path
        * raw text — name defaults to ``note-<hash>``
        * ``{name: text}`` — several docs at once
        * a list of any of the above paths

        Unchanged chunks are hash-skipped (no embedding call); chunks that no
        longer exist in a re-added doc are removed. Saved to disk on return.
        """
        docs = self._collect(source, name)
        items: list[dict[str, Any]] = []
        new_ids: dict[str, set[str]] = {}
        for doc, text in docs.items():
            ids: set[str] = set()
            for i, chunk in enumerate(chunk_paragraphs(text, max_words=self.chunk_words,
                                                       min_words=max(1, self.chunk_words // 6))):
                cid = f"{doc}#{i}"
                ids.add(cid)
                items.append({"id": cid, "text": chunk, "meta": {"kind": "doc", "doc": doc, "idx": i}})
            new_ids[doc] = ids
        r = self.memory.upsert(items)
        removed = 0
        for it in list(self.memory.store.items):
            if not it.deleted and it.meta.get("doc") in new_ids and it.id not in new_ids[it.meta["doc"]]:
                self.memory.remove(it.id)
                removed += 1
        self.memory.flush()
        return Added(docs=len(docs), chunks=len(items), embedded=r["added"] + r["updated"],
                     skipped=r["skipped"], removed=removed)

    def forget(self, doc: str) -> int:
        """Remove one doc by name. Returns the number of chunks removed."""
        n = 0
        for it in list(self.memory.store.items):
            if not it.deleted and it.meta.get("doc") == doc and self.memory.remove(it.id):
                n += 1
        if n:
            self.memory.flush()
        return n

    def _collect(self, source: Any, name: "str | None") -> dict[str, str]:
        if isinstance(source, Mapping):
            return {str(k): str(v) for k, v in source.items()}
        if isinstance(source, (list, tuple, set)):
            out: dict[str, str] = {}
            for s in source:
                out.update(self._collect(s, None))
            return out
        p = Path(source) if isinstance(source, Path) else None
        if p is None and isinstance(source, str) and "\n" not in source and len(source) < 4096:
            cand = Path(source).expanduser()
            if cand.exists():
                p = cand
        if p is not None:
            p = p.expanduser()
            if p.is_dir():
                files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES
                               and not any(part.startswith(".") for part in f.relative_to(p).parts))
                return {f.relative_to(p).as_posix(): f.read_text(encoding="utf-8") for f in files}
            if p.is_file():
                return {name or p.name: p.read_text(encoding="utf-8")}
            raise FileNotFoundError(p)
        text = str(source)
        if not text.strip():
            raise ValueError("nothing to add: empty text")
        return {name or f"note-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:6]}": text}

    # ─── ask / answer ─────────────────────────────────────────────────────
    def ask(self, prompt: str, k: int = 4, *, min_score: float = 0.3, max_words: int = 600) -> Result:
        """Top-``k`` chunks for ``prompt``: one embedding call, one numpy scan."""
        t0 = time.perf_counter()
        qv = self.memory.embedder.embed([prompt])[0]
        t1 = time.perf_counter()
        hits = self.memory.search_vector(qv, k=k, min_score=min_score)
        t2 = time.perf_counter()
        return Result(prompt, hits, (t1 - t0) * 1000, (t2 - t1) * 1000, max_words)

    def answer(self, question: str, model: str = "llama3.2:3b", k: int = 4, *,
               min_score: float = 0.3, timeout: float = 600.0) -> str:
        """``ask`` + a local Ollama chat model grounded on the retrieved chunks."""
        import httpx

        r = self.ask(question, k=k, min_score=min_score)
        messages = [
            {"role": "system", "content": "Answer strictly from the context. Cite chunks as [n]. "
                                          "If the context is insufficient, say so in one sentence."},
            {"role": "user", "content": f"{r.context}\n\nQuestion: {question}"},
        ]
        resp = httpx.post(f"{self.ollama_url}/api/chat",
                          json={"model": model, "messages": messages, "stream": False},
                          timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


_OPEN: dict[Path, Topic] = {}


def topic(name: str, *, path: "str | Path | None" = None,
          embedder: "str | Embedder" = "ollama:nomic-embed-text",
          ollama_url: str = DEFAULT_OLLAMA, chunk_words: int = 120) -> Topic:
    """Open (or create) the store for ``name``. See :class:`Topic`.

    Calling this twice for the same store in one process (re-running a
    notebook cell, say) returns the object that is already open instead of
    fighting over the directory lock. ``close()`` releases it.
    """
    key = (Path(path) if path else DEFAULT_HOME.expanduser() / _slug(name)).expanduser().resolve()
    existing = _OPEN.get(key)
    if existing is not None and not existing.closed:
        return existing
    try:
        t = Topic(name, path=key, embedder=embedder, ollama_url=ollama_url, chunk_words=chunk_words)
    except StoreError as exc:
        if "lock" in str(exc):
            raise StoreError(
                f"{exc}\nThe store is open in another process (another notebook or script?). "
                f"Close it there, or open this topic with a different path=."
            ) from exc
        raise
    _OPEN[key] = t
    return t
