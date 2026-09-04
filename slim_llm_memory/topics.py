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
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .chunking import chunk_text
from .embed import Embedder
from .index import Hit, Memory, _normalise
from .enrich import extract as _extract
from .graph import Graph, wikilinks
from .keyword import BM25, rrf
from .llm import Answer, grounded_answer
from .rerank import RERANK_MARGIN, Reranker, should_rerank
from .rerank import resolve as resolve_reranker
from .store import StoreError

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SLUG = re.compile(r"[^a-z0-9._-]+")
_TEXT_SUFFIXES = {".md", ".txt", ".rst", ".markdown"}
DEFAULT_HOME = Path("~/.slim-llm-memory/topics")
DEFAULT_OLLAMA = "http://localhost:11434"
FUSION = "linear"          # "linear" (normalised cosine + BM25, weighted) or "rrf" (reciprocal rank)
FUSION_ALPHA = 0.5         # weight of the dense leg in linear fusion


def _minmax(scores: dict[int, float], i: int) -> float:
    lo, hi = min(scores.values()), max(scores.values())
    return 0.5 if hi - lo < 1e-9 else (scores[i] - lo) / (hi - lo)


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
        # Filled in by Library.ask: which topics were scanned, and what routing cost.
        self.mode: str = "dense"
        self.reranked: str | None = None
        self.rerank_ms: float = 0.0
        self.rerank_skipped: bool = False
        self.routed: list[str] | None = None
        self.routed_of: int = 0
        self.route_ms: float = 0.0
        self.per_topic_ms: dict[str, float] = {}

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
        head = (f"ask({self.prompt!r})  {len(self.hits)} hit(s) · {self.mode} · "
                f"embed {self.embed_ms:.0f} ms · scan {self.scan_ms:.2f} ms")
        if self.routed is not None:
            head += f" · routed to {len(self.routed)}/{self.routed_of} topics in {self.route_ms:.2f} ms"
        if self.reranked:
            head += f" · reranked ({self.reranked}) in {self.rerank_ms:.0f} ms"
        elif self.rerank_skipped:
            head += " · rerank skipped"
        rows = [f"  {n:>2}  {h.score:.2f}  {h.id:<24} {' '.join(h.text.split())[:62]}  [{h.meta.get('via', '')}]"
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



@dataclass
class Cand:
    """A retrieval candidate before fusion: which topic and row, its cosine and BM25 score."""
    topic: Any
    idx: int
    cos: float
    bm: float
    in_dense: bool
    in_kw: bool


def fuse(cands: list[Cand], mode: str) -> list[tuple[Cand, float]]:
    """Rank candidates (possibly from several topics) → [(cand, ranking score)] best first.
    dense: cosine; keyword: BM25; hybrid: ``FUSION`` over the whole pool, so scores are
    comparable across topics."""
    if not cands:
        return []
    if mode == "dense":
        return sorted(((c, c.cos) for c in cands), key=lambda x: -x[1])
    if mode == "keyword":
        return sorted(((c, c.bm) for c in cands if c.in_kw), key=lambda x: -x[1])
    if FUSION == "rrf":
        dense_rank = [c for c in sorted(cands, key=lambda c: -c.cos) if c.in_dense]
        kw_rank = [c for c in sorted(cands, key=lambda c: -c.bm) if c.in_kw]
        f = rrf([[id(c) for c in dense_rank], [id(c) for c in kw_rank]])
        return sorted(((c, f.get(id(c), 0.0)) for c in cands), key=lambda x: -x[1])
    cos = {id(c): c.cos for c in cands}
    bm = {id(c): c.bm for c in cands}
    scored = [(c, FUSION_ALPHA * _minmax(cos, id(c)) + (1 - FUSION_ALPHA) * _minmax(bm, id(c))) for c in cands]
    return sorted(scored, key=lambda x: -x[1])


def has_entity(meta: dict, entity: str) -> bool:
    return entity.lower() in {str(e).lower() for e in meta.get("entities") or []}


def apply_reranker(rr: "Reranker", query: str, hits: list[Hit], k: int) -> list[Hit]:
    scores = rr.score(query, [h.text for h in hits])
    order = sorted(range(len(hits)), key=lambda i: -scores[i])[:k]
    out = []
    for i in order:
        h = hits[i]
        h.meta = dict(h.meta, rerank=round(float(scores[i]), 4))
        out.append(h)
    return out


class Topic:
    """A named store of text about one subject. Use :func:`topic` to open one."""

    def __init__(self, name: str, *, path: "str | Path | None" = None,
                 embedder: "str | Embedder" = "ollama:nomic-embed-text",
                 ollama_url: str = DEFAULT_OLLAMA, chunk_words: int = 120, overlap: int = 20) -> None:
        self.name = name
        self.path = Path(path) if path else DEFAULT_HOME.expanduser() / _slug(name)
        self.chunk_words = int(chunk_words)
        self.overlap = int(overlap)
        self._bm25: tuple[tuple, BM25, np.ndarray] | None = None
        self.ollama_url = ollama_url.rstrip("/")
        self.memory = Memory(self.path, _make_embedder(embedder, self.ollama_url))
        self.closed = False
        self._centroid: tuple[tuple, np.ndarray] | None = None
        self._graph: Graph | None = None

    # ─── routing support ──────────────────────────────────────────────────
    def _state_key(self) -> tuple:
        """Changes whenever the store's contents change (add/forget both flush)."""
        s = self.memory.store
        return (s._version, len(s.items), s._dirty)

    def centroid(self) -> np.ndarray:
        """Unit-normalised mean of the topic's live vectors (zeros when empty). Cached per state."""
        key = self._state_key()
        if self._centroid is None or self._centroid[0] != key:
            s = self.memory.store
            idx = list(s.open_indices())
            if idx:
                c = s.vectors[idx].mean(axis=0)
                n = float(np.linalg.norm(c))
                c = (c / n if n else c).astype(np.float32)
            else:
                c = np.zeros(s.embedder_dim, dtype=np.float32)
            self._centroid = (key, c)
        return self._centroid[1]

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
            name: "str | None" = None, *, enrich: "bool | str" = False) -> Added:
        """Add material. ``source`` may be:

        * a file path (``"notes.md"``) — name defaults to the file name
        * a directory — every ``.md/.txt/.rst`` file under it, named by relative path
        * raw text — name defaults to ``note-<hash>``
        * ``{name: text}`` — several docs at once
        * a list of any of the above paths

        Unchanged chunks are hash-skipped (no embedding call); chunks that no
        longer exist in a re-added doc are removed. Saved to disk on return.
        ``enrich=True`` (or a model name) runs a local LLM over every new or changed
        chunk to extract entities and relations into ``meta["entities"]`` and the graph.
        """
        docs = self._collect(source, name)
        items: list[dict[str, Any]] = []
        new_ids: dict[str, set[str]] = {}
        for doc, text in docs.items():
            ids: set[str] = set()
            for c in chunk_text(text, max_words=self.chunk_words, overlap=self.overlap):
                cid = f"{doc}#{c.idx}"
                ids.add(cid)
                meta: dict[str, Any] = {"kind": "doc", "doc": doc, "idx": c.idx}
                if c.heading:
                    meta["heading"] = c.heading
                items.append({"id": cid, "text": c.text, "meta": meta})
            new_ids[doc] = ids
        before = {it.id: it.hash for it in self.memory.store.items if not it.deleted}
        r = self.memory.upsert(items)
        removed = 0
        for it in list(self.memory.store.items):
            if not it.deleted and it.meta.get("doc") in new_ids and it.id not in new_ids[it.meta["doc"]]:
                self.memory.remove(it.id)
                removed += 1
        if enrich:
            model = enrich if isinstance(enrich, str) else "llama3.2:3b"
            after = {it.id: it for it in self.memory.store.items if not it.deleted}
            changed = [it for cid, it in after.items() if cid in {i["id"] for i in items}
                       and before.get(cid) != it.hash]
            self._enrich(model, changed)
        self.memory.flush()
        self._auto_links(docs)
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
            if (self.path / "graph.json").exists() and self.graph.drop(doc):
                self.graph.save()
        return n

    def _auto_links(self, docs: Mapping[str, str]) -> None:
        """``[[Target]]`` in a doc → edge doc → Target when a doc of that name (or stem) exists."""
        known = set(self.docs())
        stems = {Path(d).stem: d for d in known}
        changed = False
        for doc, text in docs.items():
            for target in wikilinks(text):
                dst = target if target in known else stems.get(Path(target).stem)
                if dst and dst != doc:
                    self.graph.link(doc, dst, "links")
                    changed = True
        if changed:
            self.graph.save()

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
    # ─── enrichment ───────────────────────────────────────────────────────
    def _enrich(self, model: str, items: list) -> None:
        """Entities → meta["entities"]; relations → graph edges (doc —mentions→ entity,
        subject —relation→ object). One model call per chunk."""
        store = self.memory.store
        touched = False
        for it in items:
            found = _extract(model, it.text, url=self.ollama_url)
            ents = found["entities"]
            store.update_meta(it.id, dict(it.meta, entities=ents))
            doc = it.meta.get("doc")
            for e in ents:
                if doc and e != doc:
                    self.graph.link(doc, e, "mentions")
                    touched = True
            for sub, rel, obj in found["relations"]:
                if sub != obj:
                    self.graph.link(sub, obj, rel, source=it.id)
                    touched = True
        if touched:
            self.graph.save()

    def entities(self) -> dict[str, int]:
        """Extracted entity → number of chunks mentioning it, most frequent first."""
        counts: dict[str, int] = {}
        for it in self.memory.store.items:
            if not it.deleted:
                for e in it.meta.get("entities") or []:
                    counts[e] = counts.get(e, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower())))

    # ─── graph ────────────────────────────────────────────────────────────
    @property
    def graph(self) -> Graph:
        """Typed edges between this topic's docs (``graph.json``). Needs the ``[graph]`` extra."""
        if self._graph is None:
            self._graph = Graph(self.path / "graph.json")
        return self._graph

    def link(self, a: str, b: str, relation: str = "related", weight: float = 1.0, **meta: Any) -> None:
        """``a`` → ``b`` with a relation name. Nodes are doc names (or chunk ids). Saved at once."""
        self.graph.link(a, b, relation, weight, **meta)
        self.graph.save()

    def unlink(self, a: str, b: str, relation: "str | None" = None) -> bool:
        ok = self.graph.unlink(a, b, relation)
        if ok:
            self.graph.save()
        return ok

    def neighbours(self, node: str, relation: "str | None" = None, depth: int = 1) -> list[tuple[str, str, float]]:
        return self.graph.neighbours(node, relation, depth)

    def related(self, doc_or_id: str, k: int = 5, *, alpha: float = 0.6) -> Result:
        """What goes with this doc: ``alpha``·cosine of its chunks' neighbours + (1-alpha)·graph
        edge weight, one entry per doc, best first. No embedding call."""
        store = self.memory.store
        first = self._first_chunk(doc_or_id)
        if first is None:
            raise KeyError(f"unknown doc or chunk id: {doc_or_id!r}")
        me = store.items[store._id_to_idx[first]].meta.get("doc")
        t0 = time.perf_counter()
        scores: dict[str, float] = {}
        hits_by_doc: dict[str, Hit] = {}
        for h in self.memory.neighbours(first, k=max(20, 4 * k)):
            d = h.meta.get("doc")
            if d == me:
                continue
            if h.score > scores.get(d, -2.0):
                scores[d] = h.score
                hits_by_doc[d] = h
        fused = {d: alpha * max(c, 0.0) for d, c in scores.items()}
        if (self.path / "graph.json").exists() or self._graph is not None:
            # Direct edges to docs count fully; an entity node (no chunks of its own) is a
            # bridge: doc —mentions→ Postgres ←mentions— other doc, at half weight.
            paths: list[tuple[str, str, float]] = []
            for node, rel, w in self.graph.neighbours(me or doc_or_id):
                if self._first_chunk(node) is not None:
                    paths.append((node, rel, w))
                else:
                    for n2, rel2, w2 in self.graph.neighbours(node):
                        if n2 != me and self._first_chunk(n2) is not None:
                            paths.append((n2, f"{rel}→{node}→{rel2}", 0.5 * w * w2))
            for node, rel, w in paths:
                d = node.split("#", 1)[0]
                if d == me:
                    continue
                fused[d] = fused.get(d, 0.0) + (1 - alpha) * min(w, 1.0)
                if d not in hits_by_doc:
                    fid = self._first_chunk(d)
                    if fid is None:
                        continue
                    it = store.items[store._id_to_idx[fid]]
                    hits_by_doc[d] = Hit(id=it.id, score=float(store.vectors[store._id_to_idx[fid]]
                                                                 @ store.vectors[store._id_to_idx[first]]),
                                         text=it.text, meta=dict(it.meta))
                if "relation" not in hits_by_doc[d].meta:
                    hits_by_doc[d].meta = dict(hits_by_doc[d].meta, relation=rel)
        order = sorted(fused, key=lambda d: -fused[d])[:k]
        out = []
        for d in order:
            h = hits_by_doc.get(d)
            if h is None:
                continue
            h.meta = dict(h.meta, related=round(fused[d], 4), via=("graph" if "relation" in h.meta else "vector"))
            out.append(h)
        r = Result(f"related({doc_or_id!r})", out, 0.0, (time.perf_counter() - t0) * 1000, 600)
        r.mode = "related"
        return r

    def _first_chunk(self, doc_or_id: str) -> "str | None":
        store = self.memory.store
        if doc_or_id in store._id_to_idx:
            return doc_or_id
        best = None
        for it in store.items:
            if not it.deleted and it.meta.get("doc") == doc_or_id:
                if best is None or it.meta.get("idx", 0) < best.meta.get("idx", 0):
                    best = it
        return best.id if best else None

    # ─── retrieval core ───────────────────────────────────────────────────
    def _keyword_index(self) -> tuple[BM25, np.ndarray]:
        """BM25 over live chunks (rows → store indices). Cached in memory per store state and
        on disk per store version, so a topic tokenises once per change."""
        key = self._state_key()
        if self._bm25 is not None and self._bm25[0] == key:
            return self._bm25[1], self._bm25[2]
        store = self.memory.store
        rows = np.fromiter(store.open_indices(), dtype=np.int64)
        cache = self.path / f"bm25.v{store._version}.npz"
        ix: BM25 | None = None
        if not store._dirty and cache.exists():
            try:
                ix = BM25.load(cache)
                if ix.n != rows.size:
                    ix = None
            except Exception:
                ix = None
        if ix is None:
            ix = BM25(store.items[int(i)].text for i in rows)
            if not store._dirty:
                try:
                    for stale in self.path.glob("bm25.v*.npz"):
                        stale.unlink()
                    ix.save(cache)
                except OSError:
                    pass
        self._bm25 = (key, ix, rows)
        return ix, rows

    def _candidates(self, qv: np.ndarray, prompt: str, pool: int, mode: str,
                    min_score: float) -> list["Cand"]:
        """Union of the dense top-``pool`` and BM25 top-``pool`` for this topic, each with
        cosine and BM25 score. ``min_score`` applies to the dense leg; an exact keyword
        match is never cut by it."""
        if mode not in ("dense", "keyword", "hybrid"):
            raise ValueError("mode must be 'dense', 'keyword' or 'hybrid'")
        store = self.memory.store
        dense = self.memory.search_vector(qv, k=pool, min_score=min_score) if mode != "keyword" else []
        kw: list[tuple[int, float]] = []
        if mode != "dense":
            ix, rows = self._keyword_index()
            kw = [(int(rows[r]), sc) for r, sc in ix.search(prompt, k=pool)]
        cands: dict[int, Cand] = {}
        for h in dense:
            idx = store._id_to_idx[h.id]
            cands[idx] = Cand(self, idx, cos=h.score, bm=0.0, in_dense=True, in_kw=False)
        for idx, sc in kw:
            c = cands.get(idx)
            if c is None:
                cands[idx] = Cand(self, idx, cos=float(store.vectors[idx] @ qv), bm=sc, in_dense=False, in_kw=True)
            else:
                c.bm, c.in_kw = sc, True
        return list(cands.values())

    def _hit(self, c: "Cand") -> Hit:
        it = self.memory.store.items[c.idx]
        via = "both" if (c.in_dense and c.in_kw) else ("dense" if c.in_dense else "keyword")
        return Hit(id=it.id, score=c.cos, text=it.text, meta=dict(it.meta, via=via))

    def ask(self, prompt: str, k: int = 4, *, mode: str = "hybrid",
            rerank: "bool | str | Reranker | None" = None,
            rerank_margin: "float | None" = None,
            min_score: float = 0.3, max_words: int = 600, entity: "str | None" = None) -> Result:
        """Top-``k`` chunks for ``prompt``: one embedding call, then

        * ``mode="hybrid"`` (default): dense cosine ∪ BM25 keyword, fused by normalised score —
          meaning *and* exact tokens (names, numbers, file names);
        * ``mode="dense"``: cosine only; ``mode="keyword"``: BM25 only;
        * ``rerank=True`` (or a ``Reranker``): a cross-encoder re-scores the top 4·k candidates.
        * ``rerank_margin=<float>``: adaptive — skip the reranker entirely when the fused
          leader is already clear of the field (see ``rerank.should_rerank``).
          ``rerank="auto"`` is sugar for the default cross-encoder + ``RERANK_MARGIN``.

        Each hit's ``meta["via"]`` says which leg(s) found it; ``meta["rerank"]`` the reranker score.
        ``entity="Postgres"`` keeps only chunks whose extracted entities include it (see ``add(enrich=)``).
        """
        t0 = time.perf_counter()
        qv = _normalise(np.asarray(self.memory.embedder.embed([prompt])[0], dtype=np.float32))
        t1 = time.perf_counter()
        rr = resolve_reranker(rerank)
        if rr is not None and rerank_margin is None and rerank == "auto":
            rerank_margin = RERANK_MARGIN
        pool = max(20, 4 * k)
        ranked = fuse(self._candidates(qv, prompt, pool, mode, min_score), mode)
        if entity:
            ranked = [(c, sc) for c, sc in ranked if has_entity(c.topic.memory.store.items[c.idx].meta, entity)]
        rerank_skipped = False
        if rr is not None and rerank_margin is not None:
            if not should_rerank([sc for _, sc in ranked], rerank_margin):
                rerank_skipped = True
                rr = None
        hits = [c.topic._hit(c) for c, _ in ranked[: (pool if rr else k)]]
        rr_ms = 0.0
        if rr and hits:
            r0 = time.perf_counter()
            hits = apply_reranker(rr, prompt, hits, k)
            rr_ms = (time.perf_counter() - r0) * 1000
        t2 = time.perf_counter()
        r = Result(prompt, hits, (t1 - t0) * 1000, (t2 - t1) * 1000 - rr_ms, max_words)
        r.mode, r.reranked, r.rerank_ms = mode, (rr.name if rr else None), rr_ms
        r.rerank_skipped = rerank_skipped
        return r

    def answer(self, question: str, model: str = "llama3.2:3b", k: int = 4, *, mode: str = "hybrid",
               rerank: "bool | Reranker | None" = None, min_score: float = 0.3, rewrite: bool = False,
               stream: bool = False, refuse_below: "float | None" = None,
               timeout: float = 600.0) -> "Answer | Iterator[str]":
        """``ask`` + a local Ollama chat model grounded on the retrieved chunks.

        * ``rewrite=True``: one short LLM call turns the question into a search query first.
        * ``refuse_below``: refuse (no LLM call) when the best hit's cosine is under this.
        * ``stream=True``: returns an iterator of text pieces instead of an ``Answer``.
        The returned ``Answer`` is a ``str`` with ``.hits``, ``.context``, ``.citations`` (validated
        ``[n]`` markers; dangling ones are removed) and ``.refused``.
        """
        return grounded_answer(self, question, model=model, k=k, mode=mode, rerank=rerank, min_score=min_score,
                               rewrite=rewrite, stream=stream, refuse_below=refuse_below,
                               url=self.ollama_url, timeout=timeout)


_OPEN: dict[Path, Topic] = {}


def topic(name: str, *, path: "str | Path | None" = None,
          embedder: "str | Embedder" = "ollama:nomic-embed-text",
          ollama_url: str = DEFAULT_OLLAMA, chunk_words: int = 120, overlap: int = 20) -> Topic:
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
        t = Topic(name, path=key, embedder=embedder, ollama_url=ollama_url, chunk_words=chunk_words,
                  overlap=overlap)
    except StoreError as exc:
        if "lock" in str(exc):
            raise StoreError(
                f"{exc}\nThe store is open in another process (another notebook or script?). "
                f"Close it there, or open this topic with a different path=."
            ) from exc
        raise
    _OPEN[key] = t
    return t
