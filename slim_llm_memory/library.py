"""``library`` — a database of topics: one folder per topic, one numpy array each.

    from slim_llm_memory import library

    db = library()                              # ~/.slim-llm-memory/topics
    db.topic("nginx").add("docs/nginx/")        # each topic is its own store
    db.topic("cooking").add({"pasta.md": ...})
    db                                          # → table of topics
    r = db.ask("how do I enable TLS?")          # embed once, scan every topic, merge by score
    r = db.ask("...", topics=["nginx"])         # or just some
    db.archive("cooking")                       # moves the folder to _archive/, out of ask()
    db.ask("pasta", include_archived=True)      # still there when you want it
    db.restore("cooking")
    db.delete("cooking")                        # rm -rf, explicit

The filesystem is the database:

    <home>/
        nginx/        items.vN.jsonl  vectors.vN.npy  manifest.json  topic.json
        cooking/      ...
        _archive/
            old-project/  ...

Every topic shares the library's embedder, so scores are comparable across
topics and a fan-out ``ask`` is one embedding call plus N matrix-vector products.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .embed import Embedder
from .index import Hit
from .topics import DEFAULT_HOME, DEFAULT_OLLAMA, Result, Topic, _make_embedder, _slug, topic as _open_topic

ARCHIVE_DIR = "_archive"
_META = "topic.json"


@dataclass
class TopicInfo:
    name: str
    slug: str
    chunks: int
    docs: int
    archived: bool
    path: Path

    def __repr__(self) -> str:
        flag = "  (archived)" if self.archived else ""
        return f"{self.name}: {self.docs} doc(s), {self.chunks} chunks{flag}"


class Library:
    """A folder of topic stores. Use :func:`library` to open one."""

    def __init__(self, path: "str | Path | None" = None, *,
                 embedder: "str | Embedder" = "ollama:nomic-embed-text",
                 ollama_url: str = DEFAULT_OLLAMA, chunk_words: int = 120) -> None:
        self.path = (Path(path).expanduser() if path else DEFAULT_HOME.expanduser()).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ARCHIVE_DIR).mkdir(exist_ok=True)
        self.embedder = _make_embedder(embedder, ollama_url)
        self.ollama_url = ollama_url
        self.chunk_words = int(chunk_words)
        self._open: dict[str, Topic] = {}          # slug → open Topic (active or archived)

    # ─── paths ────────────────────────────────────────────────────────────
    def _dir(self, slug: str, archived: bool = False) -> Path:
        return self.path / ARCHIVE_DIR / slug if archived else self.path / slug

    def _find(self, name: str) -> tuple[str, bool]:
        """→ (slug, archived). Raises KeyError if the topic does not exist."""
        slug = _slug(name)
        if self._dir(slug).is_dir():
            return slug, False
        if self._dir(slug, archived=True).is_dir():
            return slug, True
        raise KeyError(f"no topic {name!r} in {self.path}")

    @staticmethod
    def _read_meta(d: Path) -> dict:
        try:
            return json.loads((d / _META).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ─── topics ───────────────────────────────────────────────────────────
    def topic(self, name: str) -> Topic:
        """Open (create if absent) an active topic. Archived topics must be restored first."""
        slug = _slug(name)
        if self._dir(slug, archived=True).is_dir() and not self._dir(slug).is_dir():
            raise KeyError(f"topic {name!r} is archived; call restore({name!r}) first")
        return self._open_dir(slug, archived=False, name=name)

    def _open_dir(self, slug: str, archived: bool, name: "str | None" = None) -> Topic:
        t = self._open.get(slug)
        if t is not None and not t.closed and t.path == self._dir(slug, archived):
            return t
        d = self._dir(slug, archived)
        meta = self._read_meta(d) if d.is_dir() else {}
        display = meta.get("name") or name or slug
        t = _open_topic(display, path=d, embedder=self.embedder, ollama_url=self.ollama_url,
                        chunk_words=self.chunk_words)
        if not (d / _META).exists():
            (d / _META).write_text(json.dumps({"name": display, "created": time.time()}), encoding="utf-8")
        self._open[slug] = t
        return t

    def __getitem__(self, name: str) -> Topic:
        return self.topic(name)

    def __contains__(self, name: str) -> bool:
        try:
            self._find(name)
            return True
        except (KeyError, ValueError):
            return False

    def topics(self, *, archived: bool = False) -> list[TopicInfo]:
        """Active topics (or the archived ones), alphabetical. Cheap: reads manifests, opens nothing."""
        base = self.path / ARCHIVE_DIR if archived else self.path
        out: list[TopicInfo] = []
        for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name != ARCHIVE_DIR and not p.name.startswith(".")):
            meta = self._read_meta(d)
            t = self._open.get(d.name)
            if t is not None and not t.closed and t.path == d:
                chunks, docs = len(t), len(t.docs())
            else:
                chunks, docs = self._counts_from_disk(d)
            out.append(TopicInfo(meta.get("name") or d.name, d.name, chunks, docs, archived, d))
        return out

    @staticmethod
    def _counts_from_disk(d: Path) -> tuple[int, int]:
        try:
            m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0, 0
        docs: set[str] = set()
        items_p = d / f"items.v{m.get('version', 0)}.jsonl"
        try:
            with items_p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        obj = json.loads(line)
                        if not obj.get("deleted"):
                            docs.add(str(obj.get("meta", {}).get("doc")))
        except (OSError, ValueError):
            pass
        return int(m.get("n_open", 0)), len(docs)

    def __len__(self) -> int:
        return len(self.topics())

    def __repr__(self) -> str:
        rows = self.topics() + self.topics(archived=True)
        head = f"library({self.path}, {self.embedder.name}): {len(self.topics())} active, {len(self.topics(archived=True))} archived"
        if not rows:
            return head + "\n  (empty — db.topic('name').add(...) to start)"
        w = max(len(r.name) for r in rows)
        lines = [f"  {r.name:<{w}}  {r.docs:>4} doc(s)  {r.chunks:>6} chunks  {'archived' if r.archived else 'active'}"
                 for r in rows]
        return "\n".join([head, *lines])

    # ─── archive / restore / delete ───────────────────────────────────────
    def _close_slug(self, slug: str) -> None:
        t = self._open.pop(slug, None)
        if t is not None:
            t.close()

    def archive(self, name: str) -> Path:
        """Move a topic's folder under ``_archive/``. It leaves ``ask()`` unless asked for."""
        slug, archived = self._find(name)
        if archived:
            return self._dir(slug, archived=True)
        self._close_slug(slug)
        dst = self._dir(slug, archived=True)
        shutil.move(str(self._dir(slug)), str(dst))
        return dst

    def restore(self, name: str) -> Path:
        slug, archived = self._find(name)
        if not archived:
            return self._dir(slug)
        self._close_slug(slug)
        dst = self._dir(slug)
        shutil.move(str(self._dir(slug, archived=True)), str(dst))
        return dst

    def delete(self, name: str) -> None:
        """Remove a topic and its folder for good (active or archived)."""
        slug, archived = self._find(name)
        self._close_slug(slug)
        shutil.rmtree(self._dir(slug, archived))

    # ─── ask ──────────────────────────────────────────────────────────────
    def ask(self, prompt: str, k: int = 5, *, topics: "Iterable[str] | None" = None,
            include_archived: bool = False, min_score: float = 0.3, max_words: int = 600) -> Result:
        """Embed once, scan every selected topic, merge by score.

        Hits come back with ``meta["topic"]`` set and ids prefixed ``topic/``.
        ``r.per_topic_ms`` holds the scan time of each store.
        """
        wanted: list[tuple[str, bool]] = []
        if topics is not None:
            for n in topics:
                wanted.append(self._find(n))
        else:
            wanted += [(i.slug, False) for i in self.topics()]
            if include_archived:
                wanted += [(i.slug, True) for i in self.topics(archived=True)]

        t0 = time.perf_counter()
        qv = self.embedder.embed([prompt])[0]
        t1 = time.perf_counter()

        merged: list[Hit] = []
        per_topic: dict[str, float] = {}
        for slug, archived in wanted:
            t = self._open_dir(slug, archived)
            s0 = time.perf_counter()
            hits = t.memory.search_vector(qv, k=k, min_score=min_score)
            per_topic[t.name] = (time.perf_counter() - s0) * 1000
            for h in hits:
                meta = dict(h.meta, topic=t.name)
                if archived:
                    meta["archived"] = True
                merged.append(Hit(id=f"{t.name}/{h.id}", score=h.score, text=h.text, meta=meta))
        merged.sort(key=lambda h: h.score, reverse=True)
        t2 = time.perf_counter()

        r = Result(prompt, merged[:k], (t1 - t0) * 1000, (t2 - t1) * 1000, max_words)
        r.per_topic_ms = per_topic
        return r

    # ─── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        for slug in list(self._open):
            self._close_slug(slug)

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def library(path: "str | Path | None" = None, *,
            embedder: "str | Embedder" = "ollama:nomic-embed-text",
            ollama_url: str = DEFAULT_OLLAMA, chunk_words: int = 120) -> Library:
    """Open (or create) a library of topics at ``path``. See :class:`Library`."""
    return Library(path, embedder=embedder, ollama_url=ollama_url, chunk_words=chunk_words)
