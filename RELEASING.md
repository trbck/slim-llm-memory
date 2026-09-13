# Releasing

Pushing a `vX.Y.Z` tag publishes to PyPI via `.github/workflows/release.yml`
(PyPI trusted publishing — no token anywhere). The workflow refuses a tag that
does not match `__version__`.

## Steps

1. Bump `__version__` in `slim_llm_memory/__init__.py` — the only place the version lives.
2. Add a `## X.Y.Z — YYYY-MM-DD` entry at the top of `CHANGELOG.md`.
3. Check the build locally:
   ```bash
   rm -rf dist && uv build && uvx twine check --strict dist/*
   ```
4. Commit and push, wait for `ci` to go green:
   ```bash
   git commit -am "release: vX.Y.Z" && git push origin main
   ```
5. Tag and push the tag:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
6. Verify from a clean env:
   ```bash
   uv run --no-project --no-cache --with slim-llm-memory==X.Y.Z python -c "import slim_llm_memory as m; print(m.__version__)"
   ```

## Rules

- A version on PyPI can never be re-uploaded, even after deleting it. A broken
  release gets fixed by a new patch version.
- Version numbers: patch for fixes, minor for new API, major for breaking changes
  (while at 0.x, breaking changes bump the minor).
- The sdist contains only what `MANIFEST.in` lists. Local tooling state
  (`.boil/`, `.claude/`, notebook stores) stays out of both git and PyPI.

## One-time setup (already done)

PyPI → *Publishing* → trusted publisher: owner `trbck`, repo `slim-llm-memory`,
workflow `release.yml`, environment `pypi`.
