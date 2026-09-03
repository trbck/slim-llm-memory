"""Adaptive chunking: whole file → H2 sections → sliding token windows.

Token counting is whitespace tokenisation (≈15% off vs. BPE, fine for
boundary decisions; keeps tiktoken out of the hot path).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"\S+")
_H2 = re.compile(r"^## +(.*?)\s*$", re.MULTILINE)


@dataclass
class ChunkSlice:
    text: str
    section_idx: int
    heading: str | None


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _h2_sections(text: str) -> list[tuple[str | None, str]]:
    """Split on ``^## `` lines. Returns (heading, section_text) pairs; the
    preamble (heading None) is included only when non-blank."""
    matches = list(_H2.finditer(text))
    if not matches:
        return []
    out: list[tuple[str | None, str]] = []
    pre = text[: matches[0].start()].strip()
    if pre:
        out.append((None, pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), text[m.start():end].strip()))
    return out


def _windows(text: str, window: int, overlap: int) -> list[str]:
    spans = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    if not spans:
        return []
    step = max(1, window - overlap)
    out: list[str] = []
    start = 0
    while True:
        stop = min(start + window, len(spans))
        out.append(text[spans[start][0]: spans[stop - 1][1]])
        if stop >= len(spans):
            break
        start += step
    return out


def chunk(text: str, *, max_tokens: int = 800, window: int = 400, overlap: int = 50) -> list[ChunkSlice]:
    if count_tokens(text) <= max_tokens:
        return [ChunkSlice(text=text, section_idx=0, heading=None)]

    sections = _h2_sections(text)
    if sections and all(count_tokens(body) <= max_tokens for _, body in sections):
        return [
            ChunkSlice(text=body, section_idx=i, heading=heading)
            for i, (heading, body) in enumerate(sections, start=1)
        ]

    return [
        ChunkSlice(text=w, section_idx=i, heading=None)
        for i, w in enumerate(_windows(text, window, overlap), start=1)
    ]
