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

import numpy as np

from .embed import Embedder
from .index import Hit, _normalise
from .llm import grounded_answer
from .rerank import Reranker, resolve as resolve_reranker
from .sessions import Session
from .topics import (DEFAULT_HOME, DEFAULT_OLLAMA, Cand, Result, Topic, _make_embedder, _slug, apply_reranker,
                     fuse, has_entity, topic as _open_topic)

ARCHIVE_DIR = "_archive"
_META = "topic.json"

# Routing defaults — see examples/03_routing_bench.py for the numbers behind them.
ROUTE_AUTO_THRESHOLD = 50_000   # chunks; below this an exact scan is < ~20 ms, routing buys nothing felt
ROUTE_MAX_TOPICS = 5            # stage 2 scans at most this many topics
ROUTE_MARGIN = 0.05             # topics within this of the best centroid score are kept too
ROUTE_MIN_SCORE = 0.2           # best centroid weaker than this → prompt belongs nowhere → exact fallback


@dataclass
class Route:
    """What ``Library.route`` returns: every topic ranked by centroid similarity, and the pick."""
    prompt: str
    ranked: list[tuple[str, float]]
    chosen: list[str]
    embed_ms: float
    route_ms: float

    def __repr__(self) -> str:
        head = f"route({self.prompt!r})  → {self.chosen}  · embed {self.embed_ms:.0f} ms · route {self.route_ms:.2f} ms"
        rows = [f"  {s:.2f}  {n}" for n, s in self.ranked[:10]]
        return "\n".join([head, *rows])


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
                 ollama_url: str = DEFAULT_OLLAMA, chunk_words: int = 120, overlap: int = 20) -> None:
        self.path = (Path(path).expanduser() if path else DEFAULT_HOME.expanduser()).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ARCHIVE_DIR).mkdir(exist_ok=True)
        self.embedder = _make_embedder(embedder, ollama_url)
        self.ollama_url = ollama_url
        self.chunk_words = int(chunk_words)
        self.overlap = int(overlap)
        self._open: dict[str, Topic] = {}          # slug → open Topic (active or archived)
        self._flat_cache: tuple | None = None     # (key, M, owner, row) for exact fan-out
        self._sessions: dict[Path, Session] = {}

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
                        chunk_words=self.chunk_words, overlap=self.overlap)
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
        for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
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

    # ─── sessions ─────────────────────────────────────────────────────────
    def session(self, name: str) -> Session:
        """Conversation memory stored under ``<library>/_sessions/<slug>/`` — see :class:`Session`."""
        s = Session(name, path=self.path / "_sessions" / _slug(name), embedder=self.embedder,
                    ollama_url=self.ollama_url)
        self._sessions[s.path] = s
        return s

    def sessions(self) -> list[str]:
        base = self.path / "_sessions"
        if not base.is_dir():
            return []
        out = []
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            meta = self._read_meta(d)
            out.append((meta.get("name") or d.name).removeprefix("session "))
        return out

    # ─── candidates ───────────────────────────────────────────────────────
    def _candidates(self, topics: "Iterable[str] | None", include_archived: bool) -> list[tuple[Topic, bool]]:
        wanted: list[tuple[str, bool]] = []
        if topics is not None:
            wanted = [self._find(n) for n in topics]
        else:
            wanted = [(i.slug, False) for i in self.topics()]
            if include_archived:
                wanted += [(i.slug, True) for i in self.topics(archived=True)]
        return [(self._open_dir(slug, archived), archived) for slug, archived in wanted]

    # ─── routing (stage 1: one small centroid array) ──────────────────────
    def _route_vec(self, qv: np.ndarray, cands: list[tuple[Topic, bool]], m: int, margin: float,
                   min_score: float) -> tuple[list[tuple[str, float]], list[tuple[Topic, bool]]]:
        """Rank candidate topics by centroid similarity; choose those within ``margin`` of the
        best (at most ``m``). If even the best centroid is weak (< ``min_score``) the prompt
        does not clearly belong anywhere → return every candidate (exact fallback)."""
        C = np.stack([t.centroid() for t, _ in cands])
        s = C @ qv
        order = np.argsort(-s)
        ranked = [(cands[i][0].name, float(s[i])) for i in order]
        best = float(s[order[0]])
        if best < min_score:
            return ranked, list(cands)
        chosen = [cands[i] for i in order if s[i] >= best - margin][: max(1, m)]
        return ranked, chosen

    def route(self, prompt: str, m: "int | None" = None, *, margin: "float | None" = None,
              min_score: "float | None" = None, include_archived: bool = False) -> Route:
        """Stage 1 on its own: which topics does this prompt belong to?"""
        m = ROUTE_MAX_TOPICS if m is None else m                 # resolved at call time (tunable)
        margin = ROUTE_MARGIN if margin is None else margin
        min_score = ROUTE_MIN_SCORE if min_score is None else min_score
        cands = self._candidates(None, include_archived)
        if not cands:
            return Route(prompt, [], [], 0.0, 0.0)
        t0 = time.perf_counter()
        qv = _normalise(np.asarray(self.embedder.embed([prompt])[0], dtype=np.float32))
        t1 = time.perf_counter()
        ranked, chosen = self._route_vec(qv, cands, m, margin, min_score)
        t2 = time.perf_counter()
        return Route(prompt, ranked, [t.name for t, _ in chosen], (t1 - t0) * 1000, (t2 - t1) * 1000)

    # ─── exact fan-out over one concatenated array ────────────────────────
    def _flat(self, cands: list[tuple[Topic, bool]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(M, owner, row): every live vector of every candidate stacked once. Cached until any
        candidate's store changes. Costs a second in-memory copy of the vectors; built lazily."""
        key = tuple((t.path, t._state_key()) for t, _ in cands)
        if self._flat_cache is not None and self._flat_cache[0] == key:
            return self._flat_cache[1:]
        mats, owners, rows = [], [], []
        for ti, (t, _) in enumerate(cands):
            idx = np.fromiter(t.memory.store.open_indices(), dtype=np.int64)
            if idx.size:
                mats.append(t.memory.store.vectors[idx])
                owners.append(np.full(idx.size, ti, dtype=np.int64))
                rows.append(idx)
        if mats:
            M, owner, row = np.vstack(mats), np.concatenate(owners), np.concatenate(rows)
        else:
            M = np.zeros((0, self.embedder.dim), dtype=np.float32)
            owner = row = np.zeros(0, dtype=np.int64)
        self._flat_cache = (key, M, owner, row)
        return M, owner, row

    def _scan_flat(self, qv: np.ndarray, cands: list[tuple[Topic, bool]], k: int, min_score: float) -> list[Hit]:
        M, owner, row = self._flat(cands)
        if M.shape[0] == 0:
            return []
        s = M @ qv
        valid = np.where(s >= min_score)[0] if min_score > 0 else np.arange(s.size)
        if valid.size == 0:
            return []
        kk = min(k, valid.size)
        top = valid[np.argpartition(-s[valid], kk - 1)[:kk]] if kk < valid.size else valid
        top = top[np.argsort(-s[top])]
        out: list[Hit] = []
        for i in top:
            t, archived = cands[int(owner[i])]
            it = t.memory.store.items[int(row[i])]
            out.append(self._label(t, archived, Hit(id=it.id, score=float(s[i]), text=it.text, meta=dict(it.meta))))
        return out

    @staticmethod
    def _label(t: Topic, archived: bool, h: Hit) -> Hit:
        meta = dict(h.meta, topic=t.name)
        if archived:
            meta["archived"] = True
        return Hit(id=f"{t.name}/{h.id}", score=h.score, text=h.text, meta=meta)

    # ─── ask ──────────────────────────────────────────────────────────────
    def ask(self, prompt: str, k: int = 5, *, topics: "Iterable[str] | None" = None,
            include_archived: bool = False, min_score: float = 0.3, max_words: int = 600,
            route: "bool | int | str" = "auto", mode: str = "hybrid",
            rerank: "bool | Reranker | None" = None, entity: "str | None" = None) -> Result:
        """Embed once, then find the best chunks across topics.

        ``route``:
          * ``False``  — exact: scan every candidate topic (one concatenated array).
          * ``True``   — two-stage: centroids pick ≤ ROUTE_MAX_TOPICS topics, scan only those.
          * an int     — two-stage with that many topics at most.
          * ``"auto"`` — exact while the library holds ≤ ROUTE_AUTO_THRESHOLD chunks, else two-stage.

        ``mode`` is ``"hybrid"`` (dense ∪ BM25, default), ``"dense"`` or ``"keyword"`` — see
        ``Topic.ask``. Hits carry ``meta["topic"]`` and ids prefixed ``topic/``. ``r.routed``
        lists the topics scanned (``None`` when exact), ``r.per_topic_ms`` the per-store time.
        """
        cands = self._candidates(topics, include_archived)
        rr = resolve_reranker(rerank)
        t0 = time.perf_counter()
        qv = _normalise(np.asarray(self.embedder.embed([prompt])[0], dtype=np.float32))
        t1 = time.perf_counter()
        if not cands:
            return Result(prompt, [], (t1 - t0) * 1000, 0.0, max_words)

        if route == "auto":
            total = sum(len(t) for t, _ in cands)
            use_route, m = (len(cands) > 1 and total > ROUTE_AUTO_THRESHOLD), ROUTE_MAX_TOPICS
        elif route is False:
            use_route, m = False, 0
        else:
            use_route, m = True, (ROUTE_MAX_TOPICS if route is True else int(route))

        routed = None
        route_ms = 0.0
        per_topic: dict[str, float] = {}
        if use_route:
            r0 = time.perf_counter()
            _, chosen = self._route_vec(qv, cands, m, ROUTE_MARGIN, ROUTE_MIN_SCORE)
            route_ms = (time.perf_counter() - r0) * 1000
            routed = [t.name for t, _ in chosen]
            hits = self._merge(qv, prompt, chosen, k, mode, min_score, per_topic, rr, entity)
        elif mode == "dense" and rr is None and entity is None:
            hits = self._scan_flat(qv, cands, k, min_score)
        else:
            hits = self._merge(qv, prompt, cands, k, mode, min_score, per_topic, rr, entity)
        t2 = time.perf_counter()

        r = Result(prompt, hits, (t1 - t0) * 1000, (t2 - t1) * 1000 - route_ms, max_words)
        r.mode, r.reranked = mode, (rr.name if rr else None)
        r.routed, r.routed_of, r.route_ms, r.per_topic_ms = routed, len(cands), route_ms, per_topic
        return r

    def _merge(self, qv: np.ndarray, prompt: str, cands: list[tuple[Topic, bool]], k: int, mode: str,
               min_score: float, per_topic: dict[str, float], rr: "Reranker | None",
               entity: "str | None" = None) -> list[Hit]:
        """Candidates from every topic, fused ONCE over the union so scores compare across topics."""
        pool = max(20, 4 * k)
        allc: list[Cand] = []
        archived_of: dict[int, bool] = {}
        for t, archived in cands:
            s0 = time.perf_counter()
            got = t._candidates(qv, prompt, pool, mode, min_score)
            per_topic[t.name] = (time.perf_counter() - s0) * 1000
            allc += got
            archived_of[id(t)] = archived
        ranked = fuse(allc, mode)
        if entity:
            ranked = [(c, sc) for c, sc in ranked if has_entity(c.topic.memory.store.items[c.idx].meta, entity)]
        hits = [self._label(c.topic, archived_of[id(c.topic)], c.topic._hit(c)) for c, _ in ranked[: (pool if rr else k)]]
        if rr and hits:
            hits = apply_reranker(rr, prompt, hits, k)
        return hits

    def answer(self, question: str, model: str = "llama3.2:3b", k: int = 4, *, mode: str = "hybrid",
               rerank: "bool | Reranker | None" = None, min_score: float = 0.3, rewrite: bool = False,
               stream: bool = False, refuse_below: "float | None" = None, timeout: float = 600.0):
        """Library-wide ``answer``: retrieve across topics, then one grounded Ollama call. See ``Topic.answer``."""
        return grounded_answer(self, question, model=model, k=k, mode=mode, rerank=rerank, min_score=min_score,
                               rewrite=rewrite, stream=stream, refuse_below=refuse_below,
                               url=self.ollama_url, timeout=timeout)

    # ─── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        for slug in list(self._open):
            self._close_slug(slug)
        for s in list(self._sessions.values()):
            s.close()
        self._sessions.clear()

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def library(path: "str | Path | None" = None, *,
            embedder: "str | Embedder" = "ollama:nomic-embed-text",
            ollama_url: str = DEFAULT_OLLAMA, chunk_words: int = 120, overlap: int = 20) -> Library:
    """Open (or create) a library of topics at ``path``. See :class:`Library`."""
    return Library(path, embedder=embedder, ollama_url=ollama_url, chunk_words=chunk_words, overlap=overlap)
