"""Minimal slim-llm-memory example.

Run:

    pip install -e ..      # from this directory
    python 01_minimal.py

Indexes this file's docstring and module body, then runs a search.
Uses ``Embedder.noop()`` so it works offline and without any LLM
service. Swap to ``Embedder.ollama("nomic-embed-text")`` once you
have Ollama running locally.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from slim_llm_memory import Embedder, Memory

# A handful of "documents" — pretend each is a paragraph from a doc.
DOCS = [
    {"id": "nginx-1",  "text": "How to set up nginx on Ubuntu — install, configure server blocks, enable TLS via certbot.",            "meta": {"kind": "note", "tags": ["nginx", "linux"]}},
    {"id": "nginx-2",  "text": "nginx performance tuning: worker_processes, keepalive_timeout, gzip compression.",                    "meta": {"kind": "note", "tags": ["nginx"]}},
    {"id": "shop-1",   "text": "Milch und Brot kaufen.",                                                                              "meta": {"kind": "shopping"}},
    {"id": "shop-2",   "text": "Buy USB-C cable for the laptop.",                                                                     "meta": {"kind": "shopping"}},
    {"id": "idea-1",   "text": "Idee: weekly digest of saved bookmarks summarised by Susi every Sunday morning.",                     "meta": {"kind": "idea"}},
    {"id": "ref-1",    "text": "Kontakt: Steuerberater Mayer, +43 1 234 5678, office@mayer-tax.example.",                             "meta": {"kind": "reference"}},
    {"id": "kn-1",     "text": "PostgreSQL: VACUUM ANALYZE updates the planner's statistics. Run it after big bulk inserts.",         "meta": {"kind": "knowledge"}},
]


def main() -> None:
    # Use a temp dir so the example is self-contained.
    with tempfile.TemporaryDirectory(prefix="slim_llm_memory_") as tmp:
        path = Path(tmp) / "index"

        with Memory(path, Embedder.noop(dim=384)) as mem:
            print(f"→ Indexing into {path}")
            r = mem.upsert(DOCS)
            print(f"  upsert: {r}")

            print()
            for query in [
                "how do i install nginx?",
                "what to buy",
                "tax accountant phone number",
            ]:
                print(f"? {query}")
                hits = mem.search(query, k=3, min_score=0.0)
                for h in hits:
                    print(f"    {h.score:+.3f}  [{h.meta.get('kind','-'):>10}]  {h.id}: {h.text[:80]}")
                print()

            # Idempotent: re-upserting the same docs is a no-op.
            r2 = mem.upsert(DOCS)
            print(f"→ Re-upsert (should all skip): {r2}")

            # Filter by kind: shopping items only.
            print()
            print("? all shopping items (kind filter)")
            hits = mem.search("anything", k=10, kinds={"shopping"})
            for h in hits:
                print(f"    {h.score:+.3f}  {h.id}: {h.text}")

            # Stats — feed these into your own /health endpoint.
            print()
            print("→ stats:", {k: v for k, v in mem.stats().items() if k in ("items_open", "embedder", "embed_dim", "version", "counters")})

            mem.flush()
            print(f"→ Persisted to disk. Files in {path}:")
            for p in sorted(path.iterdir()):
                print(f"    {p.name:30s}  {p.stat().st_size:>8} bytes")


if __name__ == "__main__":
    main()
