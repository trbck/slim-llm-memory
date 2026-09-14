# slim-llm-memory

Local memory and retrieval for LLM apps. Put your notes or docs into a store, ask a
question, and get back the passages to paste into a prompt, or a cited answer from a
local model.

It needs only numpy and httpx. Embeddings come from [Ollama](https://ollama.com) on your
own machine. It is meant for personal projects and research code, up to about 50,000
passages per store, where one numpy array is fast enough and a vector database is
extra work.

## Install

```bash
pip install slim-llm-memory
ollama pull nomic-embed-text      # embeddings
ollama pull llama3.2:3b           # only needed for answer()
```

Python 3.10 or newer. Optional extras:

| Extra | Adds |
|---|---|
| `slim-llm-memory[rerank]` | cross-encoder reranking (sentence-transformers) |
| `slim-llm-memory[graph]` | links between documents (NetworkX) |
| `slim-llm-memory[obsidian]` | experimental Obsidian vault ingest; Python API only, no command yet |

The `[gemini]` and `[anthropic]` extras are placeholders. No code uses them yet.

## Quickstart

```python
from slim_llm_memory import topic

t = topic("nginx")                  # a store in ~/.slim-llm-memory/topics/nginx
t.add("docs/nginx/")                # every .md, .txt and .rst file in the folder
r = t.ask("how do I enable TLS?")   # the best matching passages
print(r)                            # hits, scores and timings
print(r.context)                    # numbered passages, ready to paste into a prompt

print(t.answer("how do I enable TLS?"))   # a local model answers from those passages and cites them
```

`add` also takes a single file, raw text, or a `{name: text}` dict. Running it again only
re-embeds what changed, and `t.forget("old.md")` removes a document. To try the API
without Ollama, pass `embedder="noop"`: every call works, but the similarity scores mean
nothing.

## Many topics

A library is a folder of topics that you can search together.

```python
from slim_llm_memory import library

db = library()                          # ~/.slim-llm-memory/topics
db.topic("nginx").add("docs/nginx/")
db.topic("cooking").add({"pasta.md": "Boil 100 g pasta per person in salted water."})

print(db)                               # a table of topics
db.ask("how do I enable TLS?")          # searches every topic; each hit names its topic
db.ask("...", topics=["nginx"])         # search only some topics
db.archive("cooking")                   # hide from ask(); db.restore("cooking") brings it back
```

Once a library holds more than 50,000 passages, `ask` first picks the topics closest to
the question and searches only those. `db.route(question)` shows that choice on its own.

## Better results

By default `ask` combines embedding search with keyword search (BM25), so exact names,
numbers and file names still match. You can change that per call:

```python
t.ask(q, mode="dense")          # embeddings only
t.ask(q, mode="keyword")        # keywords only
t.ask(q, rerank=True)           # a cross-encoder re-reads the top candidates (needs [rerank])
t.ask(q, rerank="auto")         # the same, but only when the top results are close together
t.answer(q, refuse_below=0.4)   # refuse instead of answering when no passage is close enough
t.answer(q, rewrite=True)       # the model rewrites the question as a search query first
```

Measure on your own questions before you tune anything:

```python
from slim_llm_memory import evaluate

# each case: a question, and a word from the right passage (or its document name)
evaluate(t, [("which file is the commit point?", "manifest")], k=5)   # hit@1, hit@5, MRR
```

On eight questions about this repo's docs, the default search had the right passage in
its top 5 for 7 of them, and adding the reranker found all 8. `rerank="auto"` gave the same
answers as always reranking and was 4.8× faster. Tables and setup:
[docs/BENCHMARKS.md](https://github.com/trbck/slim-llm-memory/blob/main/docs/BENCHMARKS.md).

## Links, entities and chat history

```python
t.link("nginx.md", "certbot.md", relation="uses")   # needs [graph]
t.related("nginx.md")                               # similar and linked documents
t.add("docs/", enrich=True)                         # a local model extracts names and relations (slow, needs [graph])
t.ask(q, entity="Postgres")                         # only passages that mention Postgres

s = db.session("2026-09-13 refactor")               # a conversation you can search later
s.turn("user", "the flaky test was the shared tmp dir")
s.recall("why were tests flaky?")
s.history(5)                                        # the last 5 turns, in order
```

With `[graph]` installed, `[[wikilinks]]` in your documents become links when you add them.

## Low-level API

`topic()` is built on `Memory`, a vector index for when you want to manage ids and
splitting yourself.

```python
from slim_llm_memory import Embedder, Memory

with Memory("./mymemory", Embedder.ollama("nomic-embed-text")) as mem:
    mem.upsert([
        {"id": "doc1", "text": "how to set up nginx", "meta": {"kind": "note"}},
        {"id": "doc2", "text": "buy milk",            "meta": {"kind": "shopping"}},
    ])                                          # embeds only new or changed text
    hits = mem.search("nginx tutorial", k=5, kinds={"note"}, min_score=0.55)
    groups = mem.find_duplicates(threshold=0.86)
```

`Memory` also has `neighbours(id)`, `search_vector(vec)`, `update_text(id, text)`,
`remove(id)`, `stats()` and `flush()`. `Embedder.noop()` gives offline vectors for tests.

## How your data is stored

Each store is a plain folder:

```
items.vN.jsonl    one line per item: id, text, content hash, metadata
vectors.vN.npy    the embeddings, one row per item
manifest.json     points at the current version
.lock             only one process writes to a folder at a time
```

A save writes new versioned files first and switches `manifest.json` over only once they
are complete. If the process crashes mid-save, the previous version still loads.

## Speed and limits

Search is one numpy scan: 0.3 ms over 1,000 passages, 2.4 ms over 10,000 and 12 ms over
50,000 (median, 8-core CPU). The slow step is Ollama embedding the question, about 1.2 to
1.5 s on that CPU without a GPU.

It is built for one process on one machine. When that stops being enough:

| When | Replace |
|---|---|
| search takes over 100 ms at your size | `index.py` with faiss (HNSW) |
| a second process needs to write | `store.py` with SQLite + sqlite-vss |
| more than 1M items | both, with a vector database such as Qdrant or Weaviate |

## Notebooks and examples

Start with the four hello notebooks. Each is about ten lines and needs Ollama running.

| Notebook | Shows |
|---|---|
| [00_hello_topic](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/00_hello_topic.ipynb) | `topic()`, `.add()`, `.ask()` |
| [01_hello_library](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/01_hello_library.ipynb) | many topics behind one handle, and `route()` |
| [02_hello_memory](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/02_hello_memory.ipynb) | the low-level `Memory` API |
| [03_hello_answer](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/03_hello_answer.ipynb) | a cited answer, and refusal |

Then the five use-case notebooks. Each builds one small application end to end and measures it.

| Notebook | Use case |
|---|---|
| [10_usecase_codebase_qa](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/10_usecase_codebase_qa.ipynb) | ask a repo questions; dense vs keyword vs hybrid on identifiers, cited answer |
| [11_usecase_ticket_triage](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/11_usecase_ticket_triage.ipynb) | route support tickets to an area, draft a reply, escalate with `refuse_below` |
| [12_usecase_agent_memory](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/12_usecase_agent_memory.ipynb) | an agent that recalls facts and earlier turns, and survives a restart |
| [13_usecase_semantic_cache](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/13_usecase_semantic_cache.ipynb) | cache expensive calls by meaning; choose the threshold from measured scores |
| [14_usecase_notes_housekeeping](https://github.com/trbck/slim-llm-memory/blob/main/notebooks/14_usecase_notes_housekeeping.ipynb) | re-index a notes folder cheaply, find duplicates, `related()` over `[[wikilinks]]`, `forget()` |

The longer notebooks in [notebooks/](https://github.com/trbck/slim-llm-memory/tree/main/notebooks)
measure things. `use_cases_demo` tests cited answers, paraphrased questions, other
languages and chat memory, and ends with a list of what is missing compared with a full
RAG stack.

Scripts in [examples/](https://github.com/trbck/slim-llm-memory/tree/main/examples):
`01_minimal.py` runs offline, `02_topic_context.py` is the full question-to-answer flow,
and `03_routing_bench.py` and `04_rerank_bench.py` produce the benchmark numbers.

## Development

```bash
git clone https://github.com/trbck/slim-llm-memory && cd slim-llm-memory
pip install -e ".[test]"
pytest          # no network needed; tests for extras you have not installed are skipped
```

Install the package instead of running with `PYTHONPATH=.`, which hides packaging bugs.

- [docs/IMPLEMENTATION.md](https://github.com/trbck/slim-llm-memory/blob/main/docs/IMPLEMENTATION.md): the original design plan
- [RELEASING.md](https://github.com/trbck/slim-llm-memory/blob/main/RELEASING.md): how to publish a new version
- [CHANGELOG.md](https://github.com/trbck/slim-llm-memory/blob/main/CHANGELOG.md): what changed

## License

MIT
