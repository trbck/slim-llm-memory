"""Enrichment at ingest — a local LLM pulls entities and relations out of each new chunk.

    t.add("Postgres pool exhausted; added pgbouncer", enrich=True)   # default model
    t.add(..., enrich="qwen2.5:7b-instruct")                          # pick the model
    t.entities()                     # {"Postgres": 3, "pgbouncer": 1, ...}
    t.ask("connection limits", entity="Postgres")                     # only chunks that mention it
    t.neighbours("Postgres")         # graph edges: doc —mentions→ entity, subject —relation→ object

Slow by design (one model call per new chunk) and opt-in. The graph gets an
``entity`` node per extracted name, so structure questions ("what depends on
Postgres?") become a traversal instead of a hope.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .llm import DEFAULT_OLLAMA, chat

SYSTEM_EXTRACT = (
    "Extract the named entities and the relations between them from the text. Entities are proper "
    "nouns: tools, products, files, people, organisations, places. Relations are short lowercase verbs "
    "such as uses, depends_on, part_of, fixes, causes, replaces, owns. Reply with JSON only, exactly: "
    '{"entities": ["..."], "relations": [["subject", "relation", "object"]]}'
)
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_extraction(text: str) -> dict[str, Any]:
    """Tolerant JSON parse of a model reply → {"entities": [...], "relations": [(s, r, o), ...]}."""
    m = _JSON.search(text or "")
    if not m:
        return {"entities": [], "relations": []}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}
    ents: list[str] = []
    raw_ents = data.get("entities")
    for e in (raw_ents if isinstance(raw_ents, (list, tuple)) else []):
        e = str(e).strip()
        if e and e.lower() not in {x.lower() for x in ents}:
            ents.append(e)
    rels: list[tuple[str, str, str]] = []
    raw_rels = data.get("relations")
    for r in (raw_rels if isinstance(raw_rels, (list, tuple)) else []):
        if isinstance(r, (list, tuple)) and len(r) == 3 and all(str(x).strip() for x in r):
            s, rel, o = (str(x).strip() for x in r)
            rels.append((s, re.sub(r"[^a-z0-9_]+", "_", rel.lower()).strip("_") or "related", o))
    return {"entities": ents, "relations": rels}


def extract(model: str, text: str, *, url: str = DEFAULT_OLLAMA, timeout: float = 600.0) -> dict[str, Any]:
    reply = chat(model, [{"role": "system", "content": SYSTEM_EXTRACT}, {"role": "user", "content": text}],
                 url=url, timeout=timeout, options={"num_predict": 400}, fmt="json")
    return parse_extraction(str(reply))
