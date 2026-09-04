"""Conversation memory — a session is a topic whose docs are turns.

    s = db.session("2026-09-04 refactor")     # or: from slim_llm_memory import session; session("name")
    s.turn("user", "the flaky test was the shared tmp dir")
    s.turn("assistant", "fixed by using tmp_path per test")
    s.recall("why were tests flaky?")          # → Result, same as Topic.ask
    s.history(4)                               # last 4 turns, in order
    s.transcript()                             # plain text
    s.summary(model="qwen2.5:7b-instruct")     # one LLM call over the transcript (optional)

Sessions live under ``<library>/_sessions/<slug>/`` (or ``~/.slim-llm-memory/sessions/``)
and are ordinary topic stores: hybrid ``ask``, hash-skip, atomic flush. Long turns are
chunked like any doc; ``history`` always returns whole turns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from .embed import Embedder
from .llm import DEFAULT_OLLAMA, chat
from .topics import DEFAULT_HOME, Result, Topic, _slug, topic as _open_topic

SESSIONS_HOME = Path("~/.slim-llm-memory/sessions")
SYSTEM_SUMMARY = ("Summarise this conversation in at most five bullet points: decisions, findings, open "
                  "questions. Keep names, numbers and file names exact.")


class Session:
    def __init__(self, name: str, *, path: "str | Path | None" = None,
                 embedder: "str | Embedder" = "ollama:nomic-embed-text", ollama_url: str = DEFAULT_OLLAMA) -> None:
        self.name = name
        self.path = Path(path) if path else SESSIONS_HOME.expanduser() / _slug(name)
        self.topic: Topic = _open_topic(f"session {name}", path=self.path, embedder=embedder,
                                        ollama_url=ollama_url, chunk_words=200, overlap=0)
        self.ollama_url = ollama_url
        meta = self.path / "topic.json"
        if not meta.exists():
            meta.write_text(json.dumps({"name": name, "kind": "session", "created": time.time()}), encoding="utf-8")

    # ─── write ────────────────────────────────────────────────────────────
    def turn(self, role: str, text: str, **meta: Any) -> str:
        """Append one turn. Returns its doc name (``00042-user``)."""
        n = len(self._turn_names()) + 1
        doc = f"{n:05d}-{role}"
        self.topic.add({doc: text.strip()})
        store = self.topic.memory.store
        for it in store.items:
            if not it.deleted and it.meta.get("doc") == doc:
                store.update_meta(it.id, dict(it.meta, role=role, turn=n, ts=time.time(), **meta))
        self.topic.memory.flush()
        return doc

    # ─── read ─────────────────────────────────────────────────────────────
    def _turn_names(self) -> list[str]:
        return sorted(self.topic.docs())

    def history(self, n: int = 10) -> list[tuple[str, str]]:
        """Last ``n`` turns as (role, text), oldest first. Chunks of one turn are re-joined."""
        names = self._turn_names()[-n:] if n else self._turn_names()
        by_doc: dict[str, list] = {}
        for it in self.topic.memory.store.items:
            if not it.deleted and it.meta.get("doc") in names:
                by_doc.setdefault(it.meta["doc"], []).append(it)
        out = []
        for doc in names:
            parts = sorted(by_doc.get(doc, []), key=lambda it: it.meta.get("idx", 0))
            role = doc.split("-", 1)[1] if "-" in doc else "note"
            out.append((role, "\n\n".join(p.text for p in parts)))
        return out

    def transcript(self, n: int = 0) -> str:
        return "\n".join(f"{role}: {text}" for role, text in self.history(n))

    def recall(self, prompt: str, k: int = 5, **ask_kwargs: Any) -> Result:
        return self.topic.ask(prompt, k=k, **ask_kwargs)

    def summary(self, model: str = "llama3.2:3b", n: int = 0, *, timeout: float = 600.0) -> str:
        text = self.transcript(n)
        if not text:
            return ""
        return str(chat(model, [{"role": "system", "content": SYSTEM_SUMMARY}, {"role": "user", "content": text}],
                        url=self.ollama_url, timeout=timeout, options={"num_predict": 300}))

    def __len__(self) -> int:
        return len(self._turn_names())

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self.history(0))

    def __repr__(self) -> str:
        return f"session({self.name!r}: {len(self)} turns, {self.path})"

    def close(self) -> None:
        self.topic.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def session(name: str, *, path: "str | Path | None" = None,
            embedder: "str | Embedder" = "ollama:nomic-embed-text", ollama_url: str = DEFAULT_OLLAMA) -> Session:
    """Open (or create) a conversation session store. See :class:`Session`."""
    return Session(name, path=path, embedder=embedder, ollama_url=ollama_url)
