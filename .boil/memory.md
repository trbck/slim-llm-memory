# Memory

**Stack:** Python 3.11 (`/home/trbck/miniconda3/envs/trading/bin/python`), numpy + httpx hard deps.
Optional extras: `[graph]` networkx, `[rerank]` sentence-transformers + CPU torch, `[obsidian]`.
Ollama at localhost:11434 with `nomic-embed-text`, `bge-m3`, `llama3.2:3b`, `qwen2.5:7b-instruct`.

**Constraints**
- Requests-style interface is the product: `topic()/library()` + `add/ask/answer`. Additions are
  keyword arguments, never new ceremony.
- CPU only, box often loaded (load avg ~13/8 cores). Cross-encoder ≈ 1 s per candidate.
- No AI attribution trailers in commits (user instruction, 2026-09-03; history was rewritten).
- Tests: offline, no network, `Embedder.noop()`; full suite must stay green (161 before this goal).

**Where things live**
`topics.py` Topic/Result/fuse · `libraries.py` Library/routing · `rerank.py` Reranker · `llm.py`
answer pipeline · `keyword.py` BM25 · `evals.py` evaluate() · `graph.py` · `sessions.py`.
