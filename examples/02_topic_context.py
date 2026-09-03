"""Topic context demo — one store per topic, queried for LLM context.

Same story as notebooks/topic_context_demo.ipynb, as a script:

    PYTHONPATH=. python examples/02_topic_context.py                 # build, ask, update, scale
    PYTHONPATH=. python examples/02_topic_context.py --llm llama3.2:3b
    PYTHONPATH=. python examples/02_topic_context.py --fresh --no-scale

Needs Ollama with nomic-embed-text. The LLM step is optional.
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

from slim_llm_memory import Embedder, Memory, topic
from slim_llm_memory.store import Item

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = [
    "What happens if the process crashes in the middle of a flush?",
    "When should I stop using the linear scan and switch to faiss?",
    "How does the library avoid re-embedding text that did not change?",
]


def hr(title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


def scan_latency(n: int, dim: int = 768, queries: int = 200) -> tuple[float, float]:
    """Synthetic random unit vectors through the real search path (noop embedder → scan only)."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp, Memory(Path(tmp), Embedder.noop(dim)) as mem:
        vecs = rng.standard_normal((n, dim), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        for i in range(n):
            mem.store.add_item(Item(id=str(i), text=f"t{i}", hash=f"h{i}"), vecs[i])
        mem.search("warm-up", k=10)
        lat = []
        for q in range(queries):
            t0 = time.perf_counter()
            mem.search(f"q{q}", k=10)
            lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return statistics.median(lat), lat[int(0.95 * len(lat)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=str(ROOT / ".topic_store"), help="store directory")
    ap.add_argument("--embed", default="ollama:nomic-embed-text", help="embedder spec")
    ap.add_argument("--llm", default=None, help="Ollama chat model for the answer step")
    ap.add_argument("--fresh", action="store_true", help="delete the store first (cold build)")
    ap.add_argument("--no-scale", action="store_true", help="skip the synthetic scale table")
    args = ap.parse_args()

    if args.fresh and Path(args.store).exists():
        shutil.rmtree(args.store)

    hr("1. BUILD   topic('slim-llm-memory').add(...)  — unchanged chunks are never re-embedded")
    t0 = time.perf_counter()
    with topic("slim-llm-memory", path=args.store, embedder=args.embed) as t:
        print(t.add([ROOT / "README.md", ROOT / "docs" / "IMPLEMENTATION.md"]),
              f"  ({(time.perf_counter() - t0) * 1000:,.0f} ms)")
        print(t)

        hr("2. ASK     t.ask(prompt) — one embed call + one numpy scan")
        for p in PROMPTS:
            print(t.ask(p, k=3, min_score=0.0), "\n")

        hr("3. UPDATE  t.add(new text) — only the new chunk is embedded")
        print(t.add("The secret keyword for this demo is zebra-latch.", name="note.md"))
        print(t.ask("what is the secret keyword?", k=1, min_score=0.0))
        t.forget("note.md")

        if args.llm:
            hr(f"4. ANSWER  t.answer(question, model={args.llm!r}) — grounded on the retrieved chunks")
            t0 = time.perf_counter()
            print(t.answer(PROMPTS[0], model=args.llm, min_score=0.0))
            print(f"\n({time.perf_counter() - t0:.1f} s)")
        else:
            hr("4. ANSWER  — skipped (pass --llm llama3.2:3b)")

    if not args.no_scale:
        hr("5. SCALE   synthetic 768-dim stores, 200 searches each (scan only)")
        print(f"{'chunks':>8} {'p50':>9} {'p95':>9}")
        for n in (1_000, 10_000, 50_000):
            p50, p95 = scan_latency(n)
            print(f"{n:>8} {p50:>7.2f} ms {p95:>7.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
