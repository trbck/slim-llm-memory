"""The five verbs an agent needs, with JSON in and JSON out.

``MemoryTools`` wraps one :class:`Library` and exposes ``remember``, ``recall``, ``forget``,
``answer`` and ``topics``. Every method takes plain arguments and returns a plain dict, so
the same object serves a hand-written tool loop (``dispatch(name, args)``), the MCP server
(``slim_llm_memory.mcp_server``) or an HTTP wrapper of your own.

``TOOLS`` is the neutral description of those verbs; :func:`anthropic_tools` and
:func:`openai_tools` reshape it for the two common tool-calling APIs.

    from slim_llm_memory.tools import MemoryTools, anthropic_tools
    mem = MemoryTools()                                       # ~/.slim-llm-memory/topics
    client.messages.create(..., tools=anthropic_tools())      # let the model call them
    for call in response.tool_calls:                          # (however your loop names it)
        result = mem.dispatch(call.name, call.input)          # → dict, json.dumps-able

Every ``recall`` and ``answer`` embeds the question through Ollama, which is one to three
seconds on a CPU. An agent that recalls before every turn should keep ``k`` small and not
recall when the turn is not a question.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from .embed import Embedder
from .libraries import Library, library
from .topics import DEFAULT_OLLAMA

DEFAULT_MODEL = "llama3.2:3b"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "remember",
        "description": (
            "Store a piece of text in a named topic so it can be found later by meaning. "
            "Use for facts, decisions, preferences and findings worth keeping across sessions. "
            "Give a stable `name` to make the note updatable: remembering under the same name "
            "replaces the earlier text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic (store) name, e.g. 'project-x' or 'user-prefs'."},
                "text": {"type": "string", "description": "The text to remember. A sentence to a few paragraphs."},
                "name": {"type": "string", "description": "Optional document name. Re-using it updates the note."},
            },
            "required": ["topic", "text"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Find the passages most relevant to a question, by meaning and by exact words. "
            "Searches one topic, or every topic when `topic` is omitted. Returns hits with scores "
            "and a ready-to-paste context block. One embedding call; slow on CPU."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "What you want to know, in plain words."},
                "topic": {"type": "string", "description": "Restrict to one topic. Omit to search all."},
                "k": {"type": "integer", "description": "How many passages to return (default 4).", "minimum": 1},
            },
            "required": ["question"],
        },
    },
    {
        "name": "forget",
        "description": "Remove one document from a topic by its name. Returns how many passages were removed.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic the document lives in."},
                "doc": {"type": "string", "description": "Document name, as returned by remember or recall."},
            },
            "required": ["topic", "doc"],
        },
    },
    {
        "name": "answer",
        "description": (
            "Answer a question from what is stored, with a local model that cites the passages it used. "
            "Refuses instead of guessing when nothing stored is close enough. Slower than recall: "
            "one embedding call plus one chat-model call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer."},
                "topic": {"type": "string", "description": "Restrict to one topic. Omit to use all."},
                "k": {"type": "integer", "description": "How many passages the model may see (default 4).", "minimum": 1},
            },
            "required": ["question"],
        },
    },
    {
        "name": "topics",
        "description": "List the topics that exist, with document and passage counts.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def anthropic_tools() -> list[dict[str, Any]]:
    """``tools=`` for the Anthropic Messages API."""
    return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOLS]


def openai_tools() -> list[dict[str, Any]]:
    """``tools=`` for the OpenAI chat completions API (and the many APIs that copy its shape)."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in TOOLS]


_SCHEMAS = {t["name"]: t["parameters"] for t in TOOLS}


def _k(k: Any) -> int:
    try:
        n = int(k)
    except (TypeError, ValueError):
        raise ValueError(f"k must be a positive integer, got {k!r}") from None
    if n < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return n


def _hit(h: Any) -> dict[str, Any]:
    d = {"id": h.id, "score": round(float(h.score), 4), "text": h.text, "doc": h.meta.get("doc")}
    if h.meta.get("topic"):
        d["topic"] = h.meta["topic"]
    if h.meta.get("via"):
        d["via"] = h.meta["via"]
    return d


class MemoryTools:
    """One :class:`Library` behind five JSON-shaped verbs. See the module docstring."""

    def __init__(self, path: "str | Path | None" = None, *, embedder: "str | Embedder" = "ollama:nomic-embed-text",
                 ollama_url: str = DEFAULT_OLLAMA, model: str = DEFAULT_MODEL,
                 refuse_below: "float | None" = 0.45, lib: "Library | None" = None) -> None:
        self.lib = lib if lib is not None else library(path, embedder=embedder, ollama_url=ollama_url)
        self.model = model
        self.refuse_below = refuse_below
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "remember": self.remember, "recall": self.recall, "forget": self.forget,
            "answer": self.answer, "topics": self.topics,
        }

    # ─── verbs ────────────────────────────────────────────────────────────
    def remember(self, topic: str, text: str, name: "str | None" = None) -> dict[str, Any]:
        text = str(text)
        if not text.strip():
            raise ValueError("nothing to remember: empty text")
        doc = name or f"note-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"
        t = self.lib.topic(topic)
        added = t.add({doc: text})                            # dict form: never mistaken for a file path
        return {"topic": t.name, "doc": doc, "chunks": added.chunks,
                "embedded": added.embedded, "unchanged": added.skipped}

    def recall(self, question: str, topic: "str | None" = None, k: int = 4,
               min_score: "float | None" = None) -> dict[str, Any]:
        k = _k(k)
        t = self._open(topic)
        extra = {} if min_score is None else {"min_score": float(min_score)}
        r = t.ask(question, k=k, **extra) if t is not None else self.lib.ask(question, k=k, **extra)
        return {"question": question, "hits": [self._with_topic(h, t) for h in r.hits],
                "context": r.context, "ms": round(r.ms, 1)}

    def forget(self, topic: str, doc: str) -> dict[str, Any]:
        t = self._open(topic)
        return {"topic": t.name, "doc": doc, "removed": t.forget(doc)}

    def answer(self, question: str, topic: "str | None" = None, k: int = 4,
               refuse_below: "float | None" = None) -> dict[str, Any]:
        k = _k(k)
        t = self._open(topic)
        target = t if t is not None else self.lib
        rb = self.refuse_below if refuse_below is None else refuse_below
        a = target.answer(question, model=self.model, k=k, refuse_below=rb)
        hits = [self._with_topic(h, t) for h in a.hits]
        return {"question": question, "answer": str(a), "refused": a.refused, "citations": list(a.citations),
                "sources": [hits[i - 1] for i in a.citations if 0 < i <= len(hits)], "hits": hits}

    def topics(self) -> dict[str, Any]:
        return {"topics": [{"name": t.name, "docs": t.docs, "chunks": t.chunks} for t in self.lib.topics()]}

    # ─── plumbing ─────────────────────────────────────────────────────────
    def dispatch(self, name: str, args: "dict[str, Any] | None" = None) -> dict[str, Any]:
        """Call a verb by name with a JSON-shaped argument dict, checked against ``TOOLS``:
        ``KeyError`` for an unknown name, ``ValueError`` for unknown or missing arguments."""
        try:
            fn, schema = self._handlers[name], _SCHEMAS[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}; known: {sorted(self._handlers)}") from None
        args = dict(args or {})
        unknown = sorted(set(args) - set(schema["properties"]))
        if unknown:
            raise ValueError(f"{name}: unknown argument(s) {unknown}; accepted: {sorted(schema['properties'])}")
        missing = [r for r in schema.get("required", []) if r not in args]
        if missing:
            raise ValueError(f"{name}: missing required argument(s) {missing}")
        return fn(**args)

    def close(self) -> None:
        self.lib.close()

    def __enter__(self) -> "MemoryTools":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _open(self, topic: "str | None"):
        """The existing Topic for any spelling of its name (case, spaces, hyphens), or None
        when no topic was given. KeyError names the known topics when it does not exist."""
        if topic is None:
            return None
        try:
            _, archived = self.lib._find(topic)
        except (KeyError, ValueError):
            raise KeyError(f"no topic {topic!r}; known: {[t.name for t in self.lib.topics()]}") from None
        if archived:
            raise KeyError(f"topic {topic!r} is archived; restore it first")
        return self.lib.topic(topic)

    @staticmethod
    def _with_topic(h: Any, t: Any) -> dict[str, Any]:
        d = _hit(h)
        if t is not None and "topic" not in d:
            d["topic"] = t.name
        return d
