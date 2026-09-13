# Log

## 2026-09-04 — adaptive reranking (goal 1)

5/5 milestones green, first-attempt pass rate 100%, $0.85 of a $5.00 budget.
M1 `should_rerank` · M2 cross-encoder knobs · M3 `Topic.ask` · M4 `Library.ask` · M5 bench.

Real-model bench (14 docs, 10 questions, `nomic-embed-text` + `bge-reranker-v2-m3`, CPU):
off 447 ms · auto 749 ms · always 3622 ms per query, MRR 1.00 for all three; auto skipped
the reranker on 10 of 10 questions. Offline (harder) corpus: off MRR 0.933, auto 1.000 with
3 of 10 reranked, always 1.000 with 10.

One process correction worth keeping: the ruler was compiled before it was committed, so the
first `score` failed its audit — the test file I had authored looked like a worker write
against `base_sha`. Fixed by committing the ruler first and re-freezing (identical check
hashes, new baseline). Next time: author → compile → **commit** → wire the guard → loop.

Next goal candidate: `auto` costs ~300 ms more than `off` because it still fetches the 4·k
pool before deciding. Worth measuring whether the pool can be sized lazily.

## 2026-09-09 — charter + ladder (governance)

Goal 1 closed: `boil-doctor.py --final` -> FINAL OK, 5/5 boxes with fresh EVIDENCE.

Wrote `.boil/charter.md` and `.boil/ladder.md`, clearing the UNGOVERNED flag. Track is
**life first, revisit business at L3**; north star is retrieval quality vs. plain BM25 on a
*real* corpus at bounded latency (currently negative — advisor's BM25 wins, ~1600x faster);
status `candidate` (workspace WIP already 5/3); kill_by 2026-12-09.

Ladder: 6 criteria ticked from fresh runs — L0 both, L1 core loop (real Ollama + llama3.2:3b,
grounded answer with citation `[4]`, 127 s), L2 suite green (167 passed), bench, no data-loss bug.

Three findings from the verification pass, all pointing the same way:
1. **Nothing in the workspace imports `slim_llm_memory`.** 29 commits/30 days, zero consumers.
2. **The package is not installed** — `pip show slim-llm-memory` finds nothing, and
   `import slim_llm_memory` fails outside the repo root. Every demo needs `PYTHONPATH=.`,
   so the README's own 30-second tour cannot run as written. This is the mechanical reason
   for finding 1.
3. **README is stale**: claims "44 tests", actual count is 167.

Next goal candidate (ahead of the earlier lazy-pool idea): make the thing installable and
integrate it into one real project — that is the L2 blocker and the north star needs a real
corpus to be measured on anyway.

## 2026-09-13 — release-ready package

Added LICENSE, `py.typed`, CHANGELOG, MANIFEST.in, `.github/workflows/ci.yml` (3.10–3.13 +
build/twine) and `release.yml` (tag `v*` -> PyPI trusted publishing). pyproject: SPDX
`license = "MIT"`, version single-sourced from `__version__`, `tomli` for 3.10 tests.

Two bugs found by testing the tarball instead of the checkout:
1. `tests/test_packaging.py` imported `tomllib` — the suite could not run on 3.10, which
   `requires-python` promises.
2. The sdist shipped without `examples/` and `tests/obsidian/`, so a test failed from the tarball.

Proof: `uv build` + `twine check --strict` PASSED x2; wheel into clean venvs, tests from the
unpacked sdist -> py3.10 138 passed/4 skipped (twice), py3.13 138 passed/4 skipped; repo suite
169 passed. One segfault on 3.10 while both venv runs were concurrent; not reproduced solo,
cause unknown. Not published: no PyPI credentials here, and uploading is the user's call.
