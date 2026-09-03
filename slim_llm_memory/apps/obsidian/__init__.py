"""Obsidian Brain — vault ingest + MCP server on top of slim-llm-memory.

Optional extra: ``pip install slim-llm-memory[obsidian]``.
See docs/specs/2026-05-20-obsidian-brain-design.md.
"""

from .brain import Brain, BrainError
from .parser import Chunk

__all__ = ["Brain", "BrainError", "Chunk"]
