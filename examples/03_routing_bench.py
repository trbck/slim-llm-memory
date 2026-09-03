"""Should a library route queries through a topic index first?

Compares three ways to answer a query over T topics of n chunks each (768-dim):

  A. fan-out    scan each topic's array in turn (what library.ask does today)
  B. two-stage  pick top-m topics by centroid similarity, scan only those
  C. flat       one concatenated array + topic-id column, scan once

Synthetic data: each topic is a Gaussian cluster around a random unit centroid;
`spread` controls overlap (0.6 = clean clusters, 1.0 = messy, real-world-ish).
Queries are drawn like chunks. Recall@k is measured against A (exact).

    PYTHONPATH=. python examples/03_routing_bench.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np

DIM = 768
K = 10
QUERIES = 100


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def noise(shape, spread: float, rng) -> np.ndarray:
    """Gaussian noise whose *norm* is ≈ spread (so cosine(chunk, centroid) ≈ 1/√(1+spread²))."""
    return (spread / np.sqrt(DIM)) * rng.standard_normal(shape, dtype=np.float32)


def make_topics(T: int, n: int, spread: float, rng) -> tuple[list[np.ndarray], np.ndarray]:
    centroids = unit(rng.standard_normal((T, DIM), dtype=np.float32))
    topics = [unit(c + noise((n, DIM), spread, rng)) for c in centroids]
    return topics, centroids


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


# ─── strategies: each returns a set of (topic, row) ids ─────────────────────

def fan_out(q, topics):
    cands = []
    for ti, V in enumerate(topics):                      # one call per topic, like library.ask
        s = V @ q
        for r in topk(s, K):
            cands.append((float(s[r]), ti, int(r)))
    cands.sort(reverse=True)
    return {(ti, r) for _, ti, r in cands[:K]}


def two_stage(q, topics, centroids, m):
    sel = topk(centroids @ q, m)                         # stage 1: T×768 GEMV
    cands = []
    for ti in sel:                                       # stage 2: only m topics
        s = topics[ti] @ q
        for r in topk(s, K):
            cands.append((float(s[r]), int(ti), int(r)))
    cands.sort(reverse=True)
    return {(ti, r) for _, ti, r in cands[:K]}


def flat(q, M, owner, offset):
    s = M @ q                                            # one GEMV over everything
    return {(int(owner[i]), int(i - offset[owner[i]])) for i in topk(s, K)}


def bench(T: int, n: int, spread: float, m: int, rng) -> dict:
    topics, centroids = make_topics(T, n, spread, rng)
    M = np.vstack(topics)
    owner = np.repeat(np.arange(T), n)
    offset = np.arange(T) * n
    qs = [unit(centroids[t] + noise(DIM, spread, rng)).astype(np.float32)
          for t in rng.integers(0, T, QUERIES)]

    def timed(fn):
        lat, outs = [], []
        fn(qs[0])                                        # warm up
        for q in qs:
            t0 = time.perf_counter(); outs.append(fn(q)); lat.append((time.perf_counter() - t0) * 1000)
        return statistics.median(lat), outs

    a_ms, a_out = timed(lambda q: fan_out(q, topics))
    b_ms, b_out = timed(lambda q: two_stage(q, topics, centroids, m))
    c_ms, c_out = timed(lambda q: flat(q, M, owner, offset))
    recall_b = statistics.mean(len(a & b) / K for a, b in zip(a_out, b_out))
    recall_c = statistics.mean(len(a & c) / K for a, c in zip(a_out, c_out))
    return dict(T=T, n=n, total=T * n, spread=spread, m=m,
                fan_out_ms=a_ms, two_stage_ms=b_ms, flat_ms=c_ms, recall_two_stage=recall_b, recall_flat=recall_c)


def row(r: dict) -> str:
    return (f"{r['T']:>6} {r['total']:>7} {r['spread']:>6} {r['m']:>3} │ "
            f"{r['fan_out_ms']:>8.2f}ms {r['two_stage_ms']:>8.2f}ms {r['flat_ms']:>6.2f}ms │ "
            f"{r['recall_two_stage']:>8.2f} {r['recall_flat']:>8.2f}")


def main() -> None:
    rng = np.random.default_rng(0)
    head = (f"{'topics':>6} {'chunks':>7} {'spread':>6} {'m':>3} │ {'A fan-out':>10} {'B 2-stage':>10} "
            f"{'C flat':>8} │ {'recall B':>8} {'recall C':>8}")
    print("spread 0.7 ≈ tight topics (chunk·centroid ≈ 0.82); 1.5 ≈ messy, overlapping (≈ 0.55)\n")
    print(head); print("─" * len(head))
    for T, n in [(10, 1000), (100, 500), (500, 200)]:
        for spread in (0.7, 1.5):
            print(row(bench(T, n, spread, max(1, int(np.sqrt(T))), rng)))
    print("\nHow many topics must stage 2 scan? (500 topics × 200 chunks, messy spread 1.5)\n")
    print(head); print("─" * len(head))
    for m in (1, 5, 22, 66, 150):
        print(row(bench(500, 200, 1.5, m, rng)))
    print("\nRecall is top-10 overlap with the exact fan-out result. Times are medians over 100 queries, scan only (no embedding).")


if __name__ == "__main__":
    main()
