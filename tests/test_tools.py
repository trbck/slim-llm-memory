"""Agent-facing surface: tool schemas, JSON in / JSON out dispatch, to_dict(), MCP server — offline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from slim_llm_memory import evaluate, library, topic
from slim_llm_memory.tools import TOOLS, MemoryTools, anthropic_tools, openai_tools


# ─── schemas ──────────────────────────────────────────────────────────────

def test_tool_schemas_are_json_and_cover_the_five_verbs():
    names = [t["name"] for t in TOOLS]
    assert names == ["remember", "recall", "forget", "answer", "topics"]
    json.dumps(TOOLS)                                   # plain data, no objects
    for t in TOOLS:
        assert t["description"] and t["parameters"]["type"] == "object"
        assert set(t["parameters"].get("required", [])) <= set(t["parameters"]["properties"])


def test_vendor_shapes():
    a = anthropic_tools()
    assert a[0].keys() == {"name", "description", "input_schema"}
    o = openai_tools()
    assert o[0]["type"] == "function" and o[0]["function"].keys() == {"name", "description", "parameters"}
    assert a[1]["input_schema"] == o[1]["function"]["parameters"] == TOOLS[1]["parameters"]


# ─── dispatch ─────────────────────────────────────────────────────────────

@pytest.fixture
def tools(tmp_path: Path):
    t = MemoryTools(tmp_path / "lib", embedder="noop:64")
    yield t
    t.close()


def test_remember_recall_forget_topics_round_trip(tools: MemoryTools):
    r = tools.remember("prefs", "The user prefers dark mode and tabs over spaces.", name="editor")
    assert r == {"topic": "prefs", "doc": "editor", "chunks": 1, "embedded": 1, "unchanged": 0}
    again = tools.remember("prefs", "The user prefers dark mode and tabs over spaces.", name="editor")
    assert again["embedded"] == 0 and again["unchanged"] == 1

    auto = tools.remember("prefs", "Deploys happen on Tuesdays.")
    assert auto["doc"].startswith("note-") and len(auto["doc"]) == len("note-") + 6

    rc = tools.recall("The user prefers dark mode and tabs over spaces.", topic="prefs", k=2)
    assert rc["hits"][0]["doc"] == "editor" and rc["hits"][0]["topic"] == "prefs"
    assert rc["context"].startswith("Context") and rc["ms"] >= 0
    json.dumps(rc)

    everywhere = tools.recall("Deploys happen on Tuesdays.", k=5, min_score=-1.0)
    assert {h["topic"] for h in everywhere["hits"]} == {"prefs"}

    assert tools.forget("prefs", "editor") == {"topic": "prefs", "doc": "editor", "removed": 1}
    assert tools.forget("prefs", "editor")["removed"] == 0

    assert tools.topics() == {"topics": [{"name": "prefs", "docs": 1, "chunks": 1}]}


def test_dispatch_by_name_and_unknown_tool(tools: MemoryTools):
    out = tools.dispatch("remember", {"topic": "t", "text": "hello world", "name": "h"})
    assert out["doc"] == "h"
    assert tools.dispatch("topics", {})["topics"][0]["name"] == "t"
    with pytest.raises(KeyError):
        tools.dispatch("nope", {})


def test_answer_refuses_without_a_model_call(tools: MemoryTools):
    tools.remember("t", "nothing relevant here", name="n")
    out = tools.answer("what is the tax deadline?", refuse_below=2.0)     # cosine can never reach 2
    assert out["refused"] is True and out["citations"] == [] and isinstance(out["answer"], str)
    json.dumps(out)


def test_recall_unknown_topic_is_a_clean_error(tools: MemoryTools):
    with pytest.raises(KeyError):
        tools.recall("anything", topic="does-not-exist")


# ─── to_dict on the result types ──────────────────────────────────────────

def test_result_types_serialise(tmp_path: Path):
    with topic("t", path=tmp_path / "t", embedder="noop:64") as t:
        added = t.add({"a.md": "alpha beta", "b.md": "gamma delta"})
        assert added.to_dict() == {"docs": 2, "chunks": 2, "embedded": 2, "unchanged": 0, "removed": 0}
        r = t.ask("alpha beta", k=2, min_score=-1.0)
        d = r.to_dict()
        assert d["prompt"] == "alpha beta" and d["mode"] == "hybrid" and len(d["hits"]) == 2
        assert d["hits"][0].keys() >= {"id", "score", "text", "doc"}
        json.dumps(d)
        rep = evaluate(t, [("alpha beta", "a.md")], k=2, min_score=-1.0)
        assert rep.to_dict()["rows"][0]["rank"] == 1 and "mrr" in rep.to_dict()
        a = t.answer("alpha", refuse_below=2.0)
        assert a.to_dict()["refused"] is True and json.dumps(a.to_dict())
    with library(tmp_path / "lib", embedder="noop:64") as db:
        db.topic("x").add({"a.md": "alpha beta"})
        rt = db.route("alpha beta")
        assert rt.to_dict()["ranked"][0]["topic"] == "x" and json.dumps(rt.to_dict())


# ─── MCP server ───────────────────────────────────────────────────────────

@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="pip install slim-llm-memory[mcp]")
async def test_mcp_server_exposes_the_tools(tmp_path: Path):
    from slim_llm_memory.mcp_server import build_server

    server = build_server(tmp_path / "lib", embedder="noop:64")
    names = sorted(t.name for t in await server.list_tools())
    assert names == ["answer", "forget", "recall", "remember", "topics"]

    res = await server.call_tool("remember", {"topic": "prefs", "text": "likes green", "name": "colour"})
    assert not res.is_error and res.structured_content["doc"] == "colour"
    res = await server.call_tool("recall", {"question": "likes green", "topic": "prefs", "k": 1})
    assert res.structured_content["hits"][0]["doc"] == "colour"
    res = await server.call_tool("topics", {})
    assert res.structured_content["topics"][0]["name"] == "prefs"


def test_mcp_module_imports_without_mcp_installed(monkeypatch):
    """The console script must resolve on a plain install; mcp is imported only when the server is built."""
    import sys
    for mod in ["mcp", "mcp.server", "mcp.server.mcpserver"]:
        monkeypatch.setitem(sys.modules, mod, None)
    import importlib
    import slim_llm_memory.mcp_server as m
    importlib.reload(m)
    with pytest.raises(ImportError, match=r"\[mcp\]"):
        m.build_server("/tmp/never-created", embedder="noop:8")
