"""Heading-aware chunking with overlap.

Sections start at markdown headings (any level); paragraphs inside a section
are packed to ``max_words``; consecutive chunks of one section share
``overlap`` words so an answer straddling a boundary is not cut in half.
Every chunk of a section starts with the section's heading line, which keeps
"## Persistence model" attached to the paragraph that explains it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*$", re.MULTILINE)
_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    text: str
    heading: str | None
    idx: int


def _pack(paras: list[str], max_words: int, min_words: int) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    words = 0
    for p in paras:
        n = len(p.split())
        if buf and words + n > max_words:
            chunks.append("\n\n".join(buf))
            buf, words = [], 0
        buf.append(p)
        words += n
    if buf:
        if chunks and words < min_words:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(buf)
        else:
            chunks.append("\n\n".join(buf))
    return chunks


def sections(text: str) -> list[tuple[str | None, str]]:
    """Split on markdown headings → [(heading line or None, body)]; keeps fenced code intact
    only as far as headings inside code fences are rare — good enough for notes and docs."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(None, text)]
    out: list[tuple[str | None, str]] = []
    pre = text[: matches[0].start()]
    if pre.strip():
        out.append((None, pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(0).strip(), text[m.end():end]))
    return out


def chunk_text(text: str, *, max_words: int = 120, overlap: int = 20,
               min_words: int | None = None) -> list[Chunk]:
    if min_words is None:
        min_words = max(1, max_words // 6)
    out: list[Chunk] = []
    for heading, body in sections(text):
        paras = [p.strip() for p in _PARA_SPLIT.split(body) if p.strip()]
        if not paras and not heading:
            continue
        pieces = _pack(paras, max_words, min_words) if paras else []
        if not pieces:
            pieces = [""]
        prev_words: list[str] = []
        for piece in pieces:
            parts: list[str] = []
            if heading:
                parts.append(heading)
            if prev_words and overlap > 0:
                parts.append("… " + " ".join(prev_words[-overlap:]))
            if piece:
                parts.append(piece)
            out.append(Chunk(text="\n\n".join(parts), heading=heading, idx=len(out)))
            prev_words = piece.split()
    return out
