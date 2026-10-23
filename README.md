# MapNet

**A small private model that knows *where* knowledge lives, not everything.**

Instead of a giant model that memorizes the world, or one flat search index over
everything, MapNet keeps a live **map** of where each piece of knowledge is, the
way DNS knows which server holds a domain. A small local model only has to read
the few passages the map points to and write the answer.

- **Private.** Runs entirely on your machine with a small model through
  [Ollama](https://ollama.com). Nothing leaves your computer.
- **Live.** Add, edit or delete a file and the next question already sees it:
  the map updates in milliseconds, with no re-indexing and no retraining.
- **Permission-aware.** Put a `.access` file in a folder and only those roles can
  reach it. Restricted branches are never even opened for other callers.
- **Agents leave trail markers.** When an agent finds something, it leaves a
  verified pointer so the next agent goes straight there. Pointers are checked on
  every use; false ones are deleted.

> Status: research prototype. It works end to end on the included sample company
> and a synthetic benchmark of up to 92,000 documents. It has not been tested on
> large real-world corpora yet.

## Quick start

```bash
git clone https://github.com/<you>/mapnet.git
cd mapnet
pip install -r requirements.txt

# a small local model (optional: without it you get the best passage instead of an answer)
ollama pull qwen2.5:3b

python ask.py examples/company
```

Example session (the answer's wording depends on the model):

```
MapNet mapped 18 files (18 passages) in 0.1s. Model: qwen2.5:3b (local). Watching for changes.

> How many weeks of paid parental leave do birth parents get?
Birth parents get 20 weeks of fully paid parental leave. (source: hr/parental-leave-policy.md)
via map | hr, chat | opened 3 of 18 files | lookup 13 ms, answer 1.4s | note #1 left for next time
```

Then, while it's running, drop a new file into the folder (try
`cp examples/drop-in/2026-10-23-checkout-outage.md examples/company/engineering/`)
and ask about it right away.

Point it at your own folder: `python ask.py ~/Documents/work`. Supported: `.md`,
`.txt`, `.csv`, `.pdf`, `.docx` and other plain-text formats. Other models:
`--model llama3.2:3b`, `--model phi4-mini`, `--model gemma3:4b`. Roles:
`--role hr,finance`. Commands inside: `/map`, `/notes`, `/role <names>`, `/quit`.

## How it works

```
question ──► notes: has an agent already found this? ──yes──► open that one file
                │ no                                              │
                ▼                                                 │
           walk the map (root → department → topic → file)        │
                │  keep the best few branches at each level        │
                ▼                                                 ▼
          a handful of passages ─────────────► small local model ─► answer + source
                │
                └──► solid answer? leave a note for the next agent
```

Every node in the map advertises, about everything beneath it:

| | Built from | Used for |
|---|---|---|
| Summary | running centroid of its documents | routing by topic |
| Names held | every term that exists below it (like a DNS zone) | following a specific project, vendor or ticket name to the one place that has it |
| Access list | union of its children's permissions | never contacting branches the caller can't see |
| Learned questions | questions answered through it (`feedback()`) | learning how people actually phrase things |

The map doesn't stop at the file. Like DNS, resolution goes from general to
specific: the folder map gets you to the file, and the file's own map gets you to
the passage. Each file is mapped **the first time a question arrives at it**: the
local model writes a short card per section, what it covers and the questions
people would ask in their own words, so "why is my namespace stuck deleting?"
lands on a page about *finalizers*. The map is cached like a DNS answer: revisits
are instant, a changed file is remapped on its next visit, and the cards flow up
into the folder map so the file is easier to find from then on. Files the lookup
touched but didn't open are mapped in the background, and every file gets a
one-line description at the folder level while the computer is idle, so the
router can find files no one has visited yet. Cards guide routing only;
they are never shown to the answering model. (To map a whole folder in advance:
`python -m mapnet.cards FOLDER`.)

### Agent notes, and why they're safe

The idea came from a real failure mode: agents that discover a shared space and
start leaving each other instructions. MapNet's notes are deliberately limited:

1. **Pointers, never instructions.** A note has only "what it's about" and "which
   file". Its text is never put in a model's prompt. Following a note only changes
   which file gets opened.
2. **Verified on every use.** If the file doesn't answer, the note weakens; at zero
   it's deleted. A planted false note dies the first time someone follows it.
3. **Permission-bound and scoped.** A note can only point to a file that exists,
   is only shown to callers allowed to open that file, and is only used by callers
   with the same roles as its writer.
4. **Traceable and self-cleaning.** Each note records its author, task and last
   confirmation. Notes whose file is deleted disappear.

## Try the demos

| Command | What it shows |
|---|---|
| `python ask.py examples/company` | Interactive, private Q&A over a folder, live updates |
| `python agents_demo.py` | Later agents reach the right file in 1 step instead of ~10; five safety checks on notes |
| `python compare.py` | Accuracy, tokens and time on 27 questions, including 2 about files added seconds earlier |
| `ANTHROPIC_API_KEY=... python compare.py --cloud --price-in 3 --price-out 15` | Same, side by side with a large cloud model (reading the same passages, and reading every file) |
| `python research/benchmark.py` | Map vs flat search on a synthetic company of up to 2,304 systems |
| `bash realtest/setup.sh && python -m realtest.run` | **The real test:** 1,700 real Kubernetes docs, 40 hand-written questions, a local 3B model, against keyword and embedding search over everything. See [realtest/README.md](realtest/README.md) |

See [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) for a 2-minute recorded demo plan.

## Results so far

**Sample company (`agents_demo.py`)**, 18 files, 8 tasks:

| | Files opened per task | Right file found |
|---|---|---|
| Agent with no map (opens files until one looks relevant) | 10.0 | 7 / 8 |
| Agent using the map | 3.5 | 8 / 8 |
| Later agent following notes | **1.0** | 8 / 8 |

**Synthetic benchmark (`research/`)**, ordinary questions:

| Documents | Flat search right first | Map right first | Items scored: flat | Items scored: map |
|---|---|---|---|---|
| 1,440 | 97.7% | 97.7% | 1,440 | 217 |
| 23,040 | 98.7% | 99.7% | 23,040 | 297 |
| 92,160 | 92.7% | **98.7%** | 92,160 | **575** |

Honest caveats, all detailed in [research/README.md](research/README.md):
the data is synthetic; when routing takes a wrong turn the answer is never seen
(flat search at least ranks it lower), so low-confidence queries should fall back
to a wider search; and on one machine a flat search is still faster in clock time,
since it's one matrix multiplication. The map's savings in work become savings in
time once sources live on different machines, or when the "searcher" is an agent
paying tokens for every file it opens.

## Project layout

```
mapnet/
  core.py        the map: nodes, routing, permissions, live add/remove, feedback
  folder.py      keeps the map in sync with a folder (polling watcher, .access files)
  notes.py       agent notes: pointer-only, verified, scoped, self-cleaning
  llm.py         readers: Ollama (local), Anthropic (cloud, for comparison), no-model fallback
  assistant.py   the full loop: notes -> map -> reader -> leave a note
  synthetic.py   synthetic company generator for the benchmark
ask.py           interactive CLI
agents_demo.py   agents leaving trail markers + safety checks
compare.py       local vs cloud comparison, including live updates
examples/        a fictional company's documents, files to drop in, question set
research/        scale benchmark, results and the synthetic walkthrough
tests/           pytest suite (runs without any model or network)
```

## Swapping parts

- **Router encoder:** the map uses a dependency-free hashed bag-of-words encoder
  with live IDF, so new files never require re-fitting. Pass any object with
  `n_features` and `encode(texts) -> scipy.sparse rows` as `MapNet(embedder=...)`
  to try a learned embedding model.
- **Reader:** any class with `answer(question, hits, stream=None) -> Answer`.
- **Sources:** today a source is a file. The same `register_source` /
  `remove_source` / `search_in` interface fits a database table, an API or an MCP
  server.

## Roadmap

- [x] Real-data test harness (`realtest/`) with embedding-search baseline and freshness test
- [ ] Run it with real models and publish the results
- [ ] Small embedding model as the router encoder
- [ ] Fallback to a wider search when routing confidence is low
- [ ] Remote sources (MCP servers) so the map spans machines, DNS-style
- [ ] Signed note authorship and per-agent reputation
- [ ] Route *tools*, not only documents: give an agent only the MCP tools the map points to
- [ ] Mobile build for an app that owns its own data

## Contributing

Issues and pull requests are welcome. Run `pytest` before submitting; the suite
needs no model, GPU or network.

## License

MIT, see [LICENSE](LICENSE).
