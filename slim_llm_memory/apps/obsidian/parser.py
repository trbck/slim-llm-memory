"""Obsidian markdown → ``Chunk`` list.

Structured metadata (title, kind, tags, links, heading_path, …) is the
same on every chunk of a file; only ``text``, ``section_idx`` and
``heading_path`` vary per chunk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import chunk as _chunk

logger = logging.getLogger(__name__)

IGNORED_DIRS = {".obsidian", ".git", ".trash"}

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_TAG = re.compile(r"(?<![\w/#&])#([A-Za-z][\w/-]*)")
_WIKILINK = re.compile(r"(?<!\\)\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_H1 = re.compile(r"^# +(.+?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict[str, Any]


# ─── path helpers ─────────────────────────────────────────────────────────

def rel_path(vault_root: Path, abs_path: Path) -> str:
    return abs_path.resolve().relative_to(vault_root.resolve()).as_posix()


def is_vault_markdown(vault_root: Path, abs_path: Path) -> bool:
    try:
        rel = abs_path.resolve().relative_to(vault_root.resolve())
    except ValueError:
        return False
    if abs_path.suffix.lower() != ".md":
        return False
    return not any(part in IGNORED_DIRS or part.startswith(".") for part in rel.parts[:-1])


# ─── frontmatter / tags / links ───────────────────────────────────────────

def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    try:
        import yaml
        data = yaml.safe_load(m.group(1))
    except Exception as exc:  # invalid yaml → body unchanged
        logger.warning("invalid frontmatter ignored: %s", exc)
        return {}, raw
    if not isinstance(data, dict):
        return {}, raw
    return data, raw[m.end():]


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip().lstrip("#") for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(t).strip().lstrip("#") for t in value if str(t).strip()]
    return [str(value)]


def _resolve_link(vault_root: Path, target: str) -> str:
    """``[[People/Alice]]`` → ``People/Alice.md`` if that file exists; else search
    the vault for ``<stem>.md``; else the raw target."""
    target = target.strip()
    direct = vault_root / f"{target}.md"
    if direct.is_file():
        return rel_path(vault_root, direct)
    stem = Path(target).name
    for cand in vault_root.rglob(f"{stem}.md"):
        if is_vault_markdown(vault_root, cand):
            return rel_path(vault_root, cand)
    return target


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ─── main entry points ────────────────────────────────────────────────────

def parse_text(vault_root: Path, rel: str, raw: str, mtime: float) -> list[Chunk]:
    fm, body = _split_frontmatter(raw)
    body = body.strip("\n")
    if not body.strip():
        return []

    stem = Path(rel).stem
    title = str(fm.get("title") or "").strip()
    if not title:
        h1 = _H1.search(body)
        title = h1.group(1).strip() if h1 else stem

    parts = rel.split("/")
    top = parts[0] if len(parts) > 1 else "root"
    source = "inbox" if top == "inbox" else "vault"

    scan = _FENCE.sub("", body)
    tags = _dedupe(_as_str_list(fm.get("tags")) + _INLINE_TAG.findall(scan))
    links = _dedupe([_resolve_link(vault_root, t) for t in _WIKILINK.findall(scan)])

    base_meta = {
        "path": rel, "title": title, "kind": top, "tags": tags, "links": links,
        "mtime": float(mtime), "source": source,
    }
    chunks: list[Chunk] = []
    for s in _chunk(body):
        heading_path = [title] + ([s.heading] if s.heading else [])
        meta = dict(base_meta, heading_path=heading_path, section_idx=s.section_idx)
        chunks.append(Chunk(id=f"{rel}#{s.section_idx}", text=s.text, meta=meta))
    return chunks


def parse_file(vault_root: Path, abs_path: Path) -> list[Chunk]:
    try:
        raw = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("skipping non-UTF-8 file: %s", abs_path)
        return []
    except FileNotFoundError:
        return []
    return parse_text(vault_root, rel_path(vault_root, abs_path), raw, abs_path.stat().st_mtime)
