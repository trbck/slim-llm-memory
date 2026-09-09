---
project: slim-llm-memory
type: life                # personal library first; revisit as `business` when L3 is green
status: candidate         # enters the portfolio without a slot (WIP already 5/3)
stage: L0
score: 0.0
north_star: "MRR@10 margin over plain BM25 on a real corpus, at acceptable latency"
north_star_current: "negative — advisor's own BM25 beats the hybrid on its 798-rule corpus, ~1600x faster"
kill_by: 2026-12-09
created: 2026-09-09
last_audit: 2026-09-09
---

# Charter — slim-llm-memory

## Goal
A slim, persistent memory + retrieval layer for LLM apps: `topic()` / `library()` with a
`requests`-style interface, ~1000 LOC, two hard dependencies (numpy + httpx), local Ollama
embeddings, no vector DB and no framework. The user is the operator's own LLM projects —
a fast per-topic store that hands an LLM the right context. Strangers are a later question
(see the sequencing note below), not the current one.

## Why this wins
Vector DBs and RAG frameworks are overkill below ~50k items: a numpy array, a jsonl file
and a content hash do the job, and the public API stays swap-compatible with faiss/Qdrant
for the day it doesn't. That is the bet. **The bet is not yet won** — see the north star.

## North star
**Metric:** MRR@10 margin over plain BM25 on a *real* corpus, at bounded latency.
**Success:** beats BM25 by >= +0.05 MRR@10 on >= 2 real corpora at <= 2x BM25 wall-clock.
**Now:** negative. Measured 2026-09: on the advisor's 798-rule corpus, advisor's own BM25
beat this library's hybrid retrieval and ran ~1600x faster; embeddings were complementary
only.

Two guardrails, because a quality metric is an internals metric and a coding loop will
chase one forever:
1. **Real corpora only.** A corpus counts only if it exists for its own reasons — the
   advisor's rules, the operator's notes — never one built to be measured on.
2. **Latency is part of the metric.** A win that costs 1600x is not a win.

## Sequencing (life -> business)
The ladder runs the **life** track (L4 = 30 consecutive days of real use). When L3 goes
green, revisit whether strangers want this; only then does the L4/L5 business block
(external users, PyPI, revenue) replace the life block. Until then, packaging work beyond
`pip install -e .` is out of scope.

## Kill criteria (pre-committed)
- Kill/park if by **2026-12-09** the north star is still negative — i.e. no real corpus
  where this beats plain BM25 by >= +0.05 MRR@10 within 2x its latency.
- Kill/park if by **2026-12-09** no other project of the operator's imports it in anger.
  (As of 2026-09-09: zero do. This is the L2 blocker, and the reason the charter exists.)
- Kill immediately if the honest measurement says the premise is wrong — if BM25 plus a
  reranker wins everywhere the operator actually searches, the embedding layer is
  ceremony and the right move is to delete it, not tune it.

## Non-goals (scope fence)
- This will **NOT** become a vector database, a RAG framework, or a service. No web UI,
  no server, no fleet.
- This will **NOT** grow new ceremony: additions arrive as keyword arguments to
  `topic()` / `library()` / `add` / `ask` / `answer`, never as new objects to learn.
- This will **NOT** chase GPU, quantisation, or ANN swaps while the CPU path is unproven.
- This will **NOT** count a retrieval win measured on a corpus built for the benchmark.
- This will **NOT** do distribution work (PyPI, docs site, announcement) before L3.

## Distribution & money (the non-code plan)
- No external distribution on the life track. The "customer" is the operator's own LLM
  projects; the first real integration *is* the distribution milestone.
- Revisit at L3: if a stranger can onboard from the README alone, ask whether to publish.
- "First euro" is not defined and deliberately so — this is a life-tool until L3 says
  otherwise.

## Post-mortem
<!-- only filled when status -> killed: 3 lines — what we believed, what was true, what to reuse -->
