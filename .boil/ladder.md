# Ladder — slim-llm-memory

<!-- Tick only with EVIDENCE lines:
  [x] <criterion> - EVIDENCE: <cmd -> result | URL | number> | YYYY-MM-DD | auto|human
Evidence TTL: auto=14d, human(usage)=30d. `[?]` = plausibly met, unverified.
Track: LIFE (see charter "Sequencing"). The L4/L5 business block is commented out and
replaces the life block only when L3 is green. -->

## L0 — Spark
- [x] Charter written and confirmed by user — EVIDENCE: `.boil/charter.md` (4-question interview answered: life-first sequencing, quality north star, candidate slot, kill_by 2026-12-09) | 2026-09-09 | human
- [x] Portfolio decision made: enters as `candidate`, displaces nothing (WIP already 5/3) — EVIDENCE: `.boil/charter.md` front-matter `status: candidate` | 2026-09-09 | human

## L1 — Skeleton (core loop works once)
- [x] Core loop end-to-end on dev machine: `topic()` -> `add` -> `ask` returns a grounded, cited answer — EVIDENCE: `PYTHONPATH=. python examples/02_topic_context.py --llm llama3.2:3b` -> exit 0; hybrid retrieval over real Ollama `nomic-embed-text`, then a grounded answer carrying citation `[4]` (127.0 s on CPU) | 2026-09-09 | auto
- [x] The demo is repeatable from a clean checkout and documented in the README — EVIDENCE: `git clone` to a temp dir -> fresh venv -> `pip install -e .[test]` -> `pytest` = exit 0, 138 passed / 4 skipped (each skip names its extra); README documents `pip install -e .` and the four hello notebooks | 2026-09-09 | auto
- [ ] `.boil/` governance pointer added to CLAUDE.md

## L2 — Usable by me
- [ ] **>=1 of my own projects imports it in anger** — the L2 blocker; 0 as of 2026-09-09
- [ ] I used it for its REAL purpose >=3 times within one week
- [ ] North star measured on >=1 real corpus, honestly, number written down even if negative
- [x] The retrieval stack is benchmarked, not guessed: MRR, calls and latency per policy — EVIDENCE: `PYTHONPATH=. python examples/04_rerank_bench.py --offline --json` -> off MRR 0.933 / 0 calls, auto MRR 1.000 / 3 calls, always MRR 1.000 / 10 calls | 2026-09-09 | auto
- [x] Full test suite green — EVIDENCE: `/home/trbck/miniconda3/envs/trading/bin/python -m pytest -q` -> exit 0, 169 passed / 0 skipped with every extra present | 2026-09-09 | auto
- [x] No data-loss or corrupting bug open — EVIDENCE: `pytest -q tests/test_store.py tests/test_index.py` -> exit 0, 33 passed (covers the 2026-09-03 lock-leak / O(N^2) upsert / theta=1.0 dedup regressions, fixed on main in 7488914); `.boil/bugs.md` empty | 2026-09-09 | auto
- [x] Setup from scratch documented and re-tested — EVIDENCE: same clean-clone run; `cd /tmp && python examples/01_minimal.py` -> exit 0 with no PYTHONPATH | 2026-09-09 | auto

## L3 — Survives a stranger
- [x] Installable without me: `pip install .` into a clean env, import works — EVIDENCE: `uv build` + `twine check --strict` PASSED (sdist + wheel); wheel[test] into clean py3.10 and py3.13 venvs, tests run from the unpacked sdist -> 138 passed / 4 skipped on both | 2026-09-13 | auto
- [ ] Survives process restart with zero data loss (atomic flush + lock, verified)
- [ ] Errors handled: Ollama down, empty store, bad input — no raw traceback reaches the caller
- [ ] A stranger onboards from the README alone — fresh-eyes run (person or clean-env agent)
- [ ] Sequencing decision recorded in the charter: stay a life-tool, or switch to the business track

## L4 — Valuable (life track)
- [ ] 30 consecutive days of real use — EVIDENCE: usage log | human
- [ ] North star hit: >= +0.05 MRR@10 over BM25 on >=2 real corpora at <=2x latency — EVIDENCE: bench output | auto

<!-- L4 alternative, business track — only if L3 flips the charter:
- [ ] >=1 external user used it without me prompting — EVIDENCE: analytics/log | human
- [ ] Structured feedback from >=3 users captured in docs
- [ ] Distribution channel tested: <channel> produced <N> installs — EVIDENCE | human
-->

## L5 — Load-bearing
- [ ] Removing it for a week would visibly hurt — EVIDENCE: deprivation note | human
- [ ] Maintenance loop defined: what runs weekly, time cost <= 1 h/week
