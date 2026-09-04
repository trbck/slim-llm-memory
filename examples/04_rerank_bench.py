"""What does the adaptive reranking policy actually buy?

Reranking with a cross-encoder is slow (~1s/candidate on CPU) but it beats plain
cosine/BM25 fusion on exact questions, because it reads query and text together.
Running it on *every* query pays that cost even when the fused ranking already has
a clear leader. ``rerank_margin`` makes the call adaptive: skip the reranker when
the top of the fused ranking isn't contested (see ``rerank.should_rerank``).

This compares three policies over the same corpus and question set, using
``slim_llm_memory.evaluate``:

  off     ask(rerank=None)                         never reranks
  auto    ask(rerank=<reranker>, rerank_margin=m)   reranks only contested queries
  always  ask(rerank=<reranker>)                    reranks every query

    PYTHONPATH=. python examples/04_rerank_bench.py [--offline] [--json] [--margin FLOAT]

``--offline`` uses ``embedder="noop:64"`` and ``Reranker.noop()`` (token overlap) —
no network, no model download. Without it, ``ollama:nomic-embed-text`` and
``Reranker.cross_encoder()`` are used instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from slim_llm_memory import evaluate, topic
from slim_llm_memory.rerank import RERANK_MARGIN, Reranker

# ─── corpus ──────────────────────────────────────────────────────────────────
# Short, single-chunk docs in confusable clusters (vacuum-*, index-*, backup-*,
# config-*, ...) so a query's top fused candidates are genuinely plausible, plus
# one document ("zebra-fact") with vocabulary unique in the corpus, so its query
# gets a decisive fused leader with nothing else close.

DOCS = {
    "vacuum-full": "VACUUM FULL rewrites the entire table to reclaim disk space but takes an exclusive lock for the duration.",
    "vacuum-analyze": "VACUUM ANALYZE updates the query planner statistics after large bulk inserts, without rewriting the table.",
    "vacuum-auto": "Autovacuum runs VACUUM and ANALYZE automatically in the background based on table activity thresholds.",
    "index-btree": "A B-tree index speeds up equality and range queries but slows down every insert and update.",
    "index-gin": "A GIN index is built for full text search and array containment queries in Postgres.",
    "index-hash": "A hash index only supports equality lookups and is rarely smaller than a B-tree index.",
    "backup-pgdump": "pg_dump produces a logical backup of a single database, portable across Postgres versions.",
    "backup-basebackup": "pg_basebackup takes a physical copy of the entire cluster, used to bootstrap a replica.",
    "replication-streaming": "Streaming replication ships WAL records continuously to a standby server for a near real time replica.",
    "replication-logical": "Logical replication replicates individual tables using publish and subscribe, decoding WAL into row changes.",
    "connection-pooling": "PgBouncer pools client connections so Postgres does not need to spawn a new backend process per connection.",
    "config-shared-buffers": "shared_buffers controls how much memory Postgres dedicates to caching table and index pages.",
    "config-work-mem": "work_mem sets the memory budget for a single sort or hash operation before it spills to disk.",
    "zebra-fact": "The zebra latch protocol appears only in this internal demo document and nowhere else in the corpus.",
}

QUESTIONS = [
    ("what does vacuum full do to table locks", "vacuum-full"),
    ("what does vacuum analyze update after bulk inserts", "vacuum-analyze"),
    ("what runs vacuum and analyze automatically in the background", "vacuum-auto"),
    ("what kind of index is built for full text search", "index-gin"),
    ("what kind of index only supports equality lookups", "index-hash"),
    ("what tool produces a logical backup of a single database", "backup-pgdump"),
    ("what tool takes a physical copy of the entire cluster", "backup-basebackup"),
    ("what pools client connections for postgres", "connection-pooling"),
    ("what is the zebra latch protocol", "zebra-fact"),
    ("what controls memory for a single sort or hash operation", "config-work-mem"),
]


class CountingReranker(Reranker):
    """Wraps a reranker and counts how many times it was actually invoked."""

    def __init__(self, inner: Reranker) -> None:
        self.inner = inner
        self.name = inner.name
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        return self.inner.score(query, texts)


class TimingProxy:
    """Delegates .ask to a Topic and records per-call wall time (ms)."""

    def __init__(self, target) -> None:
        self.target = target
        self.latencies_ms: list[float] = []

    def ask(self, prompt: str, k: int = 4, **kwargs):
        t0 = time.perf_counter()
        r = self.target.ask(prompt, k=k, **kwargs)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return r


def bench(t, base_reranker_factory, margin: float, log=lambda *a: None) -> list[dict]:
    rows = []
    for policy in ("off", "auto", "always"):
        proxy = TimingProxy(t)
        ask_kwargs: dict = {"mode": "hybrid", "min_score": 0.0}
        counter = None
        if policy == "auto":
            counter = CountingReranker(base_reranker_factory())
            ask_kwargs["rerank"] = counter
            ask_kwargs["rerank_margin"] = margin
        elif policy == "always":
            counter = CountingReranker(base_reranker_factory())
            ask_kwargs["rerank"] = counter
        else:
            ask_kwargs["rerank"] = None
        log(f"  running policy={policy!r} ...")
        report = evaluate(proxy, QUESTIONS, k=5, label=policy, **ask_kwargs)
        rows.append(dict(
            policy=policy,
            mrr=round(report.mrr, 4),
            hit1=round(report.hit1, 4),
            rerank_calls=counter.calls if counter else 0,
            ms=round(statistics.median(proxy.latencies_ms), 3) if proxy.latencies_ms else 0.0,
        ))
    return rows


def print_table(rows: list[dict]) -> None:
    head = f"{'policy':<8} {'mrr':>6} {'hit@1':>6} {'rerank_calls':>13} {'ms':>10}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['policy']:<8} {r['mrr']:>6.3f} {r['hit1']:>6.3f} {r['rerank_calls']:>13} {r['ms']:>10.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="noop embedder + token-overlap reranker, no network")
    ap.add_argument("--json", action="store_true", help="print only a JSON list of results to stdout")
    ap.add_argument("--margin", type=float, default=RERANK_MARGIN, help="adaptive rerank margin (default RERANK_MARGIN)")
    args = ap.parse_args()

    log = (lambda *a: print(*a, file=sys.stderr)) if args.json else (lambda *a: None)

    if args.offline:
        embedder = "noop:64"
        reranker_factory = Reranker.noop
    else:
        embedder = "ollama:nomic-embed-text"
        reranker_factory = Reranker.cross_encoder

    log(f"04_rerank_bench: offline={args.offline} margin={args.margin} embedder={embedder}")

    with tempfile.TemporaryDirectory(prefix="slim_llm_memory_rerank_bench_") as tmp:
        t = topic("rerank-bench", path=Path(tmp) / "index", embedder=embedder)
        try:
            log(f"  indexing {len(DOCS)} docs ...")
            t.add(DOCS)
            rows = bench(t, reranker_factory, args.margin, log=log)
        finally:
            t.close()

    if args.json:
        print(json.dumps(rows))
    else:
        print_table(rows)

    off = next(r for r in rows if r["policy"] == "off")
    auto = next(r for r in rows if r["policy"] == "auto")
    always = next(r for r in rows if r["policy"] == "always")
    if not args.json:
        print()
        print(f"always made {always['rerank_calls']} rerank calls, auto made {auto['rerank_calls']} "
              f"(skipped {always['rerank_calls'] - auto['rerank_calls']} of {len(QUESTIONS)} questions)")
        print(f"auto MRR {auto['mrr']:.3f} vs off MRR {off['mrr']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
