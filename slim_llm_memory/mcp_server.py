"""An MCP server over :class:`slim_llm_memory.tools.MemoryTools`.

Any MCP client (Claude Code, Claude Desktop, Cursor, ...) gets ``remember``, ``recall``,
``forget``, ``answer`` and ``topics`` as tools with no code written:

    pip install "slim-llm-memory[mcp]"
    slim-memory-mcp --path ~/.slim-llm-memory/topics      # stdio transport

Claude Code: ``claude mcp add memory -- slim-memory-mcp``. Claude Desktop / Cursor, in the
MCP config file::

    {"mcpServers": {"memory": {"command": "slim-memory-mcp", "args": ["--path", "/path/to/topics"]}}}

The ``mcp`` package is imported only inside :func:`build_server`, so this module (and the
console script) import cleanly on an install without the ``[mcp]`` extra.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .tools import DEFAULT_MODEL, MemoryTools
from .topics import DEFAULT_OLLAMA

INSTRUCTIONS = (
    "Persistent memory for this user, searchable by meaning. Call `remember` with a topic and a "
    "stable `name` for facts, decisions and preferences worth keeping across sessions. Call "
    "`recall` before answering questions that may depend on earlier work; keep `k` small, each "
    "call embeds the question locally and takes a second or more on CPU. `answer` also runs a "
    "local model and is slower still."
)


def build_server(path: "str | Path | None" = None, *, embedder: str = "ollama:nomic-embed-text",
                 ollama_url: str = DEFAULT_OLLAMA, model: str = DEFAULT_MODEL,
                 refuse_below: "float | None" = 0.45) -> Any:
    """Return an ``mcp.server.mcpserver.MCPServer`` with the five tools registered."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise ImportError("pip install slim-llm-memory[mcp]  (the Model Context Protocol SDK, mcp>=2)") from exc

    mem = MemoryTools(path, embedder=embedder, ollama_url=ollama_url, model=model, refuse_below=refuse_below)
    server = MCPServer("slim-llm-memory", instructions=INSTRUCTIONS)

    @server.tool(structured_output=True,
                 description="Store text in a named topic so it can be found later by meaning. "
                             "Re-using `name` updates the note instead of adding a second one.")
    def remember(topic: str, text: str, name: str | None = None) -> dict[str, Any]:
        return mem.remember(topic, text, name)

    @server.tool(structured_output=True,
                 description="Find the stored passages most relevant to a question, by meaning and exact words. "
                             "One topic, or all topics when `topic` is omitted. Returns hits and a context block.")
    def recall(question: str, topic: str | None = None, k: int = 4) -> dict[str, Any]:
        return mem.recall(question, topic, k)

    @server.tool(structured_output=True, description="Remove one document from a topic by name.")
    def forget(topic: str, doc: str) -> dict[str, Any]:
        return mem.forget(topic, doc)

    @server.tool(structured_output=True,
                 description="Answer a question from stored passages with a local model that cites them; "
                             "refuses when nothing stored is close enough. Slower than recall.")
    def answer(question: str, topic: str | None = None, k: int = 4) -> dict[str, Any]:
        return mem.answer(question, topic, k)

    @server.tool(structured_output=True, description="List topics with document and passage counts.")
    def topics() -> dict[str, Any]:
        return mem.topics()

    server._slim_memory = mem       # keeps the library open for the server's lifetime
    return server


def main(argv: "list[str] | None" = None) -> None:
    p = argparse.ArgumentParser(prog="slim-memory-mcp", description="slim-llm-memory as an MCP server (stdio).")
    p.add_argument("--path", default=None, help="library folder (default ~/.slim-llm-memory/topics)")
    p.add_argument("--embedder", default="ollama:nomic-embed-text", help="'ollama[:model]' or 'noop[:dim]'")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    p.add_argument("--model", default=DEFAULT_MODEL, help="chat model for `answer`")
    p.add_argument("--refuse-below", type=float, default=0.45, help="`answer` refuses under this cosine")
    a = p.parse_args(argv)
    server = build_server(a.path, embedder=a.embedder, ollama_url=a.ollama_url, model=a.model,
                          refuse_below=a.refuse_below)
    try:
        server.run("stdio")
    finally:
        server._slim_memory.close()


if __name__ == "__main__":
    main()
