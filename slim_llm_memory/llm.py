"""Small Ollama chat helpers shared by ``Topic.answer`` and ``Library.answer``.

    chat(model, messages)            → str
    chat(model, messages, stream=True) → iterator of text pieces
    rewrite_query(model, question)   → compact search query (one short LLM call)
    Answer                           → str subclass carrying hits, context and validated citations
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

DEFAULT_OLLAMA = "http://localhost:11434"
_CITE = re.compile(r"\[(\d+)\]")

SYSTEM_GROUNDED = ("Answer in one or two sentences, strictly from the context. Cite the chunks you use "
                   "as [n]. If the context does not contain the answer, say so in one sentence.")
SYSTEM_REWRITE = ("Rewrite the user's question as a compact search query: the key nouns, names, numbers and "
                  "verbs, no filler words, no product names that are only scoping. Reply with the query only.")


def chat(model: str, messages: list[dict[str, str]], *, url: str = DEFAULT_OLLAMA, stream: bool = False,
         timeout: float = 600.0, options: dict[str, Any] | None = None,
         fmt: str | None = None) -> "str | Iterator[str]":
    import httpx

    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream,
                               "options": {"temperature": 0, **(options or {})}}
    if fmt:
        payload["format"] = fmt
    if not stream:
        resp = httpx.post(f"{url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _gen() -> Iterator[str]:
        with httpx.stream("POST", f"{url}/api/chat", json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                piece = json.loads(line).get("message", {}).get("content", "")
                if piece:
                    yield piece
    return _gen()


def rewrite_query(model: str, question: str, *, url: str = DEFAULT_OLLAMA, timeout: float = 600.0) -> str:
    out = chat(model, [{"role": "system", "content": SYSTEM_REWRITE}, {"role": "user", "content": question}],
               url=url, timeout=timeout, options={"num_predict": 40})
    out = str(out).strip().strip('"').splitlines()[0] if out else ""
    return out or question


class Answer(str):
    """The answer text, plus what it was built from. Behaves as a plain string."""

    hits: list
    context: str
    citations: list[int]
    refused: bool
    query: str

    def __new__(cls, text: str, *, hits: list, context: str, citations: list[int],
                refused: bool = False, query: str = "") -> "Answer":
        o = super().__new__(cls, text)
        o.hits, o.context, o.citations, o.refused, o.query = hits, context, citations, refused, query
        return o


def validate_citations(text: str, n_hits: int) -> tuple[str, list[int]]:
    """Drop ``[n]`` markers that point past the context; return (clean text, cited indices)."""
    cited: list[int] = []

    def _keep(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_hits:
            if n not in cited:
                cited.append(n)
            return m.group(0)
        return ""

    clean = _CITE.sub(_keep, text)
    return re.sub(r"[ \t]{2,}", " ", clean).strip(), cited


REFUSAL = "I don't have anything about that in this store."


def grounded_answer(asker: Any, question: str, *, model: str, k: int, mode: str, rerank: Any, min_score: float,
                    rewrite: bool, stream: bool, refuse_below: "float | None", url: str,
                    timeout: float) -> "Answer | Iterator[str]":
    """Retrieve (optionally with a rewritten query), refuse when weak, else ask the model."""
    query = question
    if rewrite:
        query = rewrite_query(model, question, url=url, timeout=timeout)
    r = asker.ask(query, k=k, mode=mode, rerank=rerank, min_score=min_score)
    if rewrite and query != question and not r.hits:
        r = asker.ask(question, k=k, mode=mode, rerank=rerank, min_score=min_score)
    if not r.hits or (refuse_below is not None and r.top.score < refuse_below):
        if stream:
            return iter([REFUSAL])
        return Answer(REFUSAL, hits=list(r.hits), context=r.context, citations=[], refused=True, query=query)
    messages = [{"role": "system", "content": SYSTEM_GROUNDED},
                {"role": "user", "content": f"{r.context}\n\nQuestion: {question}"}]
    if stream:
        return chat(model, messages, url=url, stream=True, timeout=timeout)
    text = str(chat(model, messages, url=url, timeout=timeout))
    clean, cited = validate_citations(text, len(r.hits))
    return Answer(clean, hits=list(r.hits), context=r.context, citations=cited, query=query)
