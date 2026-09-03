"""Tests for the noop embedder + ollama HTTP wiring (mocked)."""

import pytest

from slim_llm_memory import Embedder, EmbedderError


# ─── noop ─────────────────────────────────────────────────────────────────

def test_noop_dim():
    e = Embedder.noop(dim=384)
    assert e.dim == 384
    assert e.name == "noop:384"


def test_noop_returns_one_vector_per_input():
    e = Embedder.noop(dim=128)
    out = e.embed(["alpha", "beta", "gamma"])
    assert len(out) == 3
    assert all(len(v) == 128 for v in out)


def test_noop_is_deterministic():
    e1 = Embedder.noop(dim=64)
    e2 = Embedder.noop(dim=64)
    assert e1.embed(["x"])[0] == e2.embed(["x"])[0]


def test_noop_different_text_different_vectors():
    e = Embedder.noop(dim=256)
    a, b = e.embed(["one", "two"])
    assert a != b


def test_noop_unit_norm():
    e = Embedder.noop(dim=128)
    v = e.embed(["something"])[0]
    n = sum(x * x for x in v) ** 0.5
    assert abs(n - 1.0) < 1e-5


def test_noop_no_nans_or_infs():
    e = Embedder.noop(dim=384)
    # Try a bunch of pathological inputs.
    for t in ["", " ", "a", "🦀", "long " * 200, "\x00\x01\x02"]:
        v = e.embed([t])[0]
        assert all(x == x for x in v)  # no NaN
        assert all(abs(x) < 1e6 for x in v)  # no inf


def test_noop_handles_empty_batch():
    e = Embedder.noop(dim=64)
    assert e.embed([]) == []


def test_noop_rejects_bad_dim():
    with pytest.raises(ValueError):
        Embedder.noop(dim=0)
    with pytest.raises(ValueError):
        Embedder.noop(dim=99999)


# ─── ollama (mocked transport) ────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)[:200]

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_url = None
        self.last_json = None

    def post(self, url, json=None):
        self.last_url = url
        self.last_json = json
        return self._response


def test_ollama_post_url_and_payload(monkeypatch):
    e = Embedder.ollama(model="nomic-embed-text")
    fake_client = _FakeClient(_FakeResponse(200, {"embeddings": [[0.1] * 768, [0.2] * 768]}))
    monkeypatch.setattr(e, "_client", lambda: fake_client)
    out = e.embed(["a", "b"])
    assert fake_client.last_url == "http://localhost:11434/api/embed"
    assert fake_client.last_json == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert len(out) == 2 and len(out[0]) == 768


def test_ollama_handles_legacy_singular_response(monkeypatch):
    e = Embedder.ollama(model="some-model")
    fake_client = _FakeClient(_FakeResponse(200, {"embedding": [0.5] * 256}))
    monkeypatch.setattr(e, "_client", lambda: fake_client)
    out = e.embed(["only one"])
    assert len(out) == 1 and len(out[0]) == 256
    # dim should now be set from the response
    assert e.dim == 256


def test_ollama_raises_on_non_200(monkeypatch):
    e = Embedder.ollama(model="x")
    monkeypatch.setattr(e, "_client", lambda: _FakeClient(_FakeResponse(500, {"error": "oops"})))
    with pytest.raises(EmbedderError):
        e.embed(["x"])


def test_ollama_raises_on_count_mismatch(monkeypatch):
    e = Embedder.ollama(model="nomic-embed-text")  # known dim 768
    fake = _FakeClient(_FakeResponse(200, {"embeddings": [[0.1] * 768]}))  # 1 vec for 2 inputs
    monkeypatch.setattr(e, "_client", lambda: fake)
    with pytest.raises(EmbedderError, match="vectors"):
        e.embed(["a", "b"])


def test_ollama_raises_on_dim_mismatch(monkeypatch):
    e = Embedder.ollama(model="nomic-embed-text")  # known dim 768
    fake = _FakeClient(_FakeResponse(200, {"embeddings": [[0.1] * 100]}))  # wrong size
    monkeypatch.setattr(e, "_client", lambda: fake)
    with pytest.raises(EmbedderError, match="mismatched"):
        e.embed(["a"])


def test_ollama_raises_on_transport_error(monkeypatch):
    e = Embedder.ollama(model="x")

    class _Boom:
        def post(self, *a, **k):
            raise RuntimeError("connection refused")
    monkeypatch.setattr(e, "_client", lambda: _Boom())
    with pytest.raises(EmbedderError, match="transport"):
        e.embed(["x"])


def test_ollama_empty_batch_is_noop():
    e = Embedder.ollama(model="x")
    assert e.embed([]) == []


def test_ollama_batches_requests_in_order(monkeypatch):
    e = Embedder.ollama(model="nomic-embed-text", batch_size=4)

    class _BatchClient:
        def __init__(self):
            self.calls: list[list[str]] = []

        def post(self, url, json=None):
            self.calls.append(list(json["input"]))
            # Each text → a vector whose first coordinate encodes its index.
            vecs = [[float(t[1:])] + [0.0] * 767 for t in json["input"]]
            return _FakeResponse(200, {"embeddings": vecs})

    client = _BatchClient()
    monkeypatch.setattr(e, "_client", lambda: client)
    out = e.embed([f"t{i}" for i in range(10)])
    assert [len(c) for c in client.calls] == [4, 4, 2]          # 10 texts → 3 requests
    assert [v[0] for v in out] == [float(i) for i in range(10)]  # order preserved across batches


def test_ollama_rejects_bad_batch_size():
    with pytest.raises(ValueError):
        Embedder.ollama(model="x", batch_size=0)
