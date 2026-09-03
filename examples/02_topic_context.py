"""Topic context demo — one numpy store per topic, queried for LLM context.

Scenario: an LLM is working on ONE topic. Everything about that topic sits in
a small persistent store; each prompt costs one embed + one numpy scan and
yields a context block for the LLM. This script proves the latency and shows
the loop end to end, fully local.

Run (needs Ollama with nomic-embed-text; the LLM step is optional):

    PYTHONPATH=. python examples/02_topic_context.py
    PYTHONPATH=. python examples/02_topic_context.py --llm qwen2.5:7b-instruct
    PYTHONPATH=. python examples/02_topic_context.py --no-scale      # skip the 50k proof

Sections:
  1. Build   — index this repo's own docs as the topic (cold), then re-run (warm, hash-skip).
  2. Query   — live prompts → top-k chunks, latency split embed / scan.
  3. Update  — change one paragraph → only that chunk is re-embedded.
  4. Answer  — (optional) ask a local LLM with the retrieved context, cite chunks.
  5. Scale   — synthetic 1k / 10k / 50k × 768 index: scan latency p50 / p95.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from slim_llm_memory import Embedder, Memory
from slim_llm_memory.topic import TopicStore, chunk_paragraphs

ROOT = Path(__file__).resolve().parent.parent
TOPIC_DOCS = {
    "README.md": ROOT / "README.md",
    "IMPLEMENTATION.md": ROOT / "docs" / "IMPLEMENTATION.md",
}
PROMPTS = [
    "What happens if the process crashes in the middle of a flush?",
    "When should I stop using the linear scan and switch to faiss?",
    "How does the library avoid re-embedding text that did not change?",
    "Which local model do you recommend for a German-language classifier?",
]


def hr(title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


def fmt_ms(ms: float) -> str:
    return f"{ms:7.1f} ms"


# ─── 1. build ─────────────────────────────────────────────────────────────

def load_docs() -> dict[str, str]:
    return {name: p.read_text(encoding="utf-8") for name, p in TOPIC_DOCS.items()}


def build(store_path: Path, embedder: Embedder, docs: dict[str, str]) -> TopicStore:
    t0 = time.perf_counter()
    store = TopicStore(store_path, embedder)
    t_open = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    r = store.add_docs(docs)
    t_add = (time.perf_counter() - t0) * 1000
    store.flush()
    n_words = sum(len(t.split()) for t in docs.values())
    print(f"docs: {len(docs)}  words: {n_words}  chunks: {r['chunks']}")
    print(f"open store: {fmt_ms(t_open)}   add_docs: {fmt_ms(t_add)}   "
          f"added={r['added']} updated={r['updated']} skipped={r['skipped']} "
          f"embed_calls={r['embed_calls']}")
    return store


# ─── 2. query ─────────────────────────────────────────────────────────────

def query(store: TopicStore, prompts: list[str], k: int = 3) -> None:
    print(f"{'prompt':<62} {'embed':>9} {'scan':>9} {'total':>9}")
    for p in prompts:
        ctx = store.context_for(p, k=k, min_score=0.0)
        print(f"{p[:60]:<62} {fmt_ms(ctx.embed_ms)} {fmt_ms(ctx.scan_ms)} {fmt_ms(ctx.total_ms)}")
        for n, h in enumerate(ctx.hits, start=1):
            snippet = " ".join(h.text.split())[:72]
            print(f"    [{n}] {h.score:.2f}  {h.id:<24} {snippet}")


# ─── 3. update ────────────────────────────────────────────────────────────

def update(store: TopicStore, docs: dict[str, str]) -> None:
    docs = dict(docs)
    docs["README.md"] = docs["README.md"] + (
        "\n\n## Demo note\n\nThe topic store demo appended this paragraph at runtime "
        "to show that only the changed chunk is re-embedded. Keyword: zebra-latch.\n"
    )
    t0 = time.perf_counter()
    r = store.add_docs(docs)
    dt = (time.perf_counter() - t0) * 1000
    print(f"re-index after editing one doc: {fmt_ms(dt)}   added={r['added']} "
          f"updated={r['updated']} skipped={r['skipped']} embed_calls={r['embed_calls']}")
    ctx = store.context_for("what is the zebra-latch keyword about?", k=1, min_score=0.0)
    print(f"query for the new paragraph → {ctx.hits[0].id}  score {ctx.hits[0].score:.2f}  "
          f"({fmt_ms(ctx.total_ms)})")


# ─── 4. answer ────────────────────────────────────────────────────────────

def answer(store: TopicStore, model: str, prompt: str) -> None:
    import httpx

    ctx = store.context_for(prompt, k=4, min_score=0.0)
    n_ctx_words = len(ctx.prompt.split())
    n_all_words = sum(len(t.split()) for t in load_docs().values())
    print(f"prompt: {prompt}")
    print(f"context: {len(ctx.hits)} chunks, {n_ctx_words} words "
          f"(vs {n_all_words} words for the whole topic → {n_all_words / max(n_ctx_words, 1):.0f}× fewer tokens)")
    print(f"retrieval: {fmt_ms(ctx.total_ms)}  (embed {ctx.embed_ms:.0f} / scan {ctx.scan_ms:.1f})")
    messages = [
        {"role": "system", "content": "Answer strictly from the context. Cite chunks as [n]. "
                                      "If the context is insufficient, say so in one sentence."},
        {"role": "user", "content": f"{ctx.prompt}\n\nQuestion: {prompt}"},
    ]
    t0 = time.perf_counter()
    resp = httpx.post("http://localhost:11434/api/chat",
                      json={"model": model, "messages": messages, "stream": False},
                      timeout=300)
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    body = resp.json()
    text = body["message"]["content"].strip()
    tok = body.get("eval_count")
    print(f"llm ({model}): {dt:.1f} s" + (f", {tok} tokens" if tok else ""))
    print()
    for line in text.splitlines():
        print("    " + line)


# ─── 5. scale ─────────────────────────────────────────────────────────────

def scale(sizes: list[int], dim: int = 768, queries: int = 200) -> None:
    """Synthetic random unit vectors through the real search path.

    The embedder is noop (µs) so the numbers are the numpy scan + top-k."""
    rng = np.random.default_rng(0)
    print(f"{'items':>8} {'dim':>5} {'build':>10} {'p50':>9} {'p95':>9} {'max':>9}")
    with tempfile.TemporaryDirectory(prefix="slim_scale_") as tmp:
        for n in sizes:
            mem = Memory(Path(tmp) / f"n{n}", Embedder.noop(dim=dim))
            t0 = time.perf_counter()
            vecs = rng.standard_normal((n, dim), dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            from slim_llm_memory.store import Item
            for i in range(n):
                mem.store.add_item(Item(id=str(i), text=f"t{i}", hash=f"h{i}", meta={"kind": "syn"}), vecs[i])
            build_ms = (time.perf_counter() - t0) * 1000
            # warm up BLAS once
            mem.search("warm", k=10)
            lat = []
            for q in range(queries):
                t0 = time.perf_counter()
                mem.search(f"query {q}", k=10)
                lat.append((time.perf_counter() - t0) * 1000)
            lat.sort()
            p50 = statistics.median(lat)
            p95 = lat[int(0.95 * len(lat)) - 1]
            print(f"{n:>8} {dim:>5} {fmt_ms(build_ms):>10} {fmt_ms(p50):>9} {fmt_ms(p95):>9} {fmt_ms(lat[-1]):>9}")
            mem.close(flush=False)


# ─── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=str(ROOT / ".topic_store"), help="persistent store dir")
    ap.add_argument("--embed", default="nomic-embed-text", help="Ollama embedding model")
    ap.add_argument("--llm", default=None, help="Ollama chat model for the answer step (optional)")
    ap.add_argument("--fresh", action="store_true", help="delete the store first (cold build)")
    ap.add_argument("--no-scale", action="store_true", help="skip the synthetic scale proof")
    args = ap.parse_args()

    store_path = Path(args.store)
    if args.fresh and store_path.exists():
        shutil.rmtree(store_path)
    cold = not (store_path / "manifest.json").exists()
    embedder = Embedder.ollama(args.embed)
    docs = load_docs()

    hr(f"1. BUILD  ({'cold — embedding every chunk' if cold else 'warm — unchanged chunks are hash-skipped'})")
    store = build(store_path, embedder, docs)
    try:
        hr("2. QUERY  (live prompts → top-3 chunks; latency = embed + numpy scan)")
        query(store, PROMPTS)

        hr("3. UPDATE (edit one doc → only the changed chunk is re-embedded)")
        update(store, docs)

        if args.llm:
            hr(f"4. ANSWER (local LLM grounded on retrieved context: {args.llm})")
            answer(store, args.llm, PROMPTS[0])
        else:
            hr("4. ANSWER — skipped (pass --llm llama3.2:3b or --llm qwen2.5:7b-instruct)")

        # restore the doc so the store is clean for the next run
        store.add_docs(docs)
        store.flush()
        s = store.stats()
        print(f"\nstore: {s['items_open']} chunks, dim {s['embed_dim']}, "
              f"embed calls this run: {s['counters'].get('embed.calls', 0)}, "
              f"searches: {s['counters'].get('search.calls', 0) + s['counters'].get('search_vector.calls', 0)}")
    finally:
        store.close()

    if not args.no_scale:
        hr("5. SCALE  (synthetic random unit vectors, real search path, 200 queries each)")
        scale([1_000, 10_000, 50_000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
