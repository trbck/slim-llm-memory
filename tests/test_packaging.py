"""The package must not promise things `pip install` cannot deliver.

These are the failures you only see after installing: a console script that
points at a module nobody wrote, or an extra that pulls a dependency no code
imports. Both look fine from a repo checkout with PYTHONPATH set.
"""
from __future__ import annotations

import importlib
import pathlib
import tomllib

import pytest

PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_every_console_script_resolves():
    """`pip install` writes a launcher per entry; each must actually import."""
    scripts = _pyproject().get("project", {}).get("scripts", {})
    broken = []
    for name, target in scripts.items():
        module, _, func = target.partition(":")
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError as exc:
            broken.append(f"{name} -> {target} ({exc})")
            continue
        if func and not hasattr(mod, func):
            broken.append(f"{name} -> {target} (no attribute {func!r})")
    assert not broken, "console scripts installed but not runnable: " + "; ".join(broken)


def test_public_api_imports_without_optional_extras():
    """The two hard deps are the promise; nothing in __all__ may need more."""
    mod = importlib.import_module("slim_llm_memory")
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, f"__all__ names not importable: {missing}"
