"""Graph layer — typed edges between docs (or chunks) of a topic, next to the vectors.

    t.link("nginx.md", "certbot.md", relation="uses")     # explicit edge
    t.neighbours("nginx.md")                               # [(node, relation, weight), ...]
    t.related("nginx.md", k=5)                             # 0.6·cosine + 0.4·graph, see Topic.related
    t.add("... see [[certbot]] ...")                        # [[wikilinks]] become "links" edges on add

Backed by ``networkx.DiGraph`` (optional extra ``[graph]``), persisted as
``graph.json`` (node-link format, human-readable) in the topic directory.
Nodes are doc names by default; chunk ids (``doc#i``) are allowed too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

_WIKILINK = re.compile(r"(?<!\\)\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
DEFAULT_RELATION = "related"


def wikilinks(text: str) -> list[str]:
    """``[[Note]]``, ``[[Note#Heading]]``, ``[[Note|alias]]`` → ["Note", ...] in order, deduped."""
    out: list[str] = []
    for m in _WIKILINK.finditer(text):
        t = m.group(1).strip()
        if t and t not in out:
            out.append(t)
    return out


class Graph:
    def __init__(self, path: Path) -> None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("pip install slim-llm-memory[graph]  (networkx)") from exc
        self._nx = nx
        self.path = Path(path)
        self.g = nx.DiGraph()
        if self.path.exists():
            self.load()

    # ─── edit ─────────────────────────────────────────────────────────────
    def link(self, a: str, b: str, relation: str = DEFAULT_RELATION, weight: float = 1.0,
             **meta: Any) -> None:
        """Add (or update) the directed edge a → b. One edge per (a, b, relation)."""
        if a == b:
            raise ValueError("self-links are not allowed")
        key = (a, b)
        rels = self.g.get_edge_data(*key, default={}).get("relations", {})
        rels = dict(rels)
        rels[relation] = {"weight": float(weight), **meta}
        self.g.add_edge(a, b, relations=rels)

    def unlink(self, a: str, b: str, relation: str | None = None) -> bool:
        data = self.g.get_edge_data(a, b)
        if not data:
            return False
        if relation is None:
            self.g.remove_edge(a, b)
            return True
        rels = dict(data.get("relations", {}))
        if relation not in rels:
            return False
        del rels[relation]
        if rels:
            self.g.add_edge(a, b, relations=rels)
        else:
            self.g.remove_edge(a, b)
        return True

    def drop(self, node: str) -> bool:
        if node in self.g:
            self.g.remove_node(node)
            return True
        return False

    # ─── read ─────────────────────────────────────────────────────────────
    def neighbours(self, node: str, relation: str | None = None, depth: int = 1,
                   direction: str = "both") -> list[tuple[str, str, float]]:
        """Nodes reachable within ``depth`` hops → [(node, relation, weight)], nearest first.
        ``direction``: "out" | "in" | "both"."""
        if node not in self.g or depth < 1:
            return []
        seen = {node}
        frontier = [node]
        out: list[tuple[str, str, float]] = []
        for _ in range(depth):
            nxt: list[str] = []
            for n in frontier:
                edges: list[tuple[str, str, dict]] = []
                if direction in ("out", "both"):
                    edges += [(n, m, d) for m, d in self.g[n].items()]
                if direction in ("in", "both"):
                    edges += [(m, n, d) for m, d in self.g.pred[n].items()]
                for a, b, d in edges:
                    other = b if a == n else a
                    if other in seen:
                        continue                      # never walk back; one path per node
                    rels = [(rel, info) for rel, info in d.get("relations", {}).items()
                            if relation is None or rel == relation]
                    if not rels:
                        continue
                    seen.add(other)
                    nxt.append(other)
                    for rel, info in rels:
                        out.append((other, rel, float(info.get("weight", 1.0))))
            frontier = nxt
        # dedupe (node, relation) keeping the max weight
        best: dict[tuple[str, str], float] = {}
        for n, r, w in out:
            best[(n, r)] = max(best.get((n, r), 0.0), w)
        return [(n, r, w) for (n, r), w in best.items()]

    def edges(self) -> list[tuple[str, str, str, float]]:
        return [(a, b, rel, float(info.get("weight", 1.0)))
                for a, b, d in self.g.edges(data=True) for rel, info in d.get("relations", {}).items()]

    def nodes(self) -> list[str]:
        return list(self.g.nodes)

    def __len__(self) -> int:
        return self.g.number_of_edges()

    def __repr__(self) -> str:
        return f"graph({self.g.number_of_nodes()} nodes, {self.g.number_of_edges()} edges, {self.path.name})"

    # ─── persistence ──────────────────────────────────────────────────────
    def save(self) -> None:
        data = self._nx.node_link_data(self.g, edges="edges")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.g = self._nx.node_link_graph(data, directed=True, edges="edges")
