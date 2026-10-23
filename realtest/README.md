# The real test

Does MapNet hold up on real, messy documents with a real small model, against
strong baselines? This folder runs that test on your own machine.

## What it compares

Every method sees exactly the same passages of the same files.

| Method | What it is |
|---|---|
| `keyword-all` | Keyword search over every passage (classic search index) |
| `embed-all` | Embedding search over every passage with a local embedding model: the standard modern RAG setup, and the baseline to beat |
| `map` | MapNet: walk the map, search only the files it routes to |
| `map+embed` | MapNet routes to a wider set of files, embeddings rank the passages inside them |

For each it measures: whether the right file is among the passages handed to the
model, how high it ranks, how much work the search did, how long it took, whether
the local model's final answer is right, and how fast newly added files become
answerable.

## The data

- **Corpus:** the Kubernetes documentation (about 1,700 real files in deep
  folders, with front matter, templates, cross-references and overlapping pages),
  downloaded and cleaned automatically.
- **40 hand-written questions** phrased the way people ask ("my app takes minutes
  to start and the liveness check kills it first"), each with the file(s) that
  answer it and keywords to check the answer. Five of them are about files that
  are held back and added after the indexes are built: the freshness test.
- **Optional generated questions** (`--generate 60`): your local model writes
  questions from random passages. More coverage, but they tend to copy the
  passage's wording, which favours keyword search.

## Run it

Needs: Python 3.10+, git, [Ollama](https://ollama.com/download), about 4 GB free
RAM and 3 GB of disk.

```bash
bash realtest/setup.sh            # macOS / Linux  (Windows: realtest\setup.ps1)
source .venv/bin/activate
python -m realtest.run            # full test
```

Useful variations:

```bash
python -m realtest.run --retrieval-only          # minutes instead of tens of minutes
python -m realtest.run --generate 60             # add generated questions
python -m realtest.run --model llama3.2:3b       # try another small reader
python -m realtest.run --model qwen2.5:7b        # how much does a bigger reader help?
```

Rough timing on a recent laptop: building embeddings for ~25,000 passages takes
several minutes the first time (cached after that); answering 40 questions × 4
methods with a 3B model takes roughly 10–30 minutes on CPU, less with a GPU or
Apple silicon. These are estimates; your report records the real times.

The report lands in `realtest/results/<timestamp>/report.md`, with every
question, retrieved file and answer in `results.json` next to it.

## Your own documents

```bash
python -m realtest.run --corpus ~/Documents/work --questions my-questions.json
```

`my-questions.json` is a list like:

```json
[
  {"q": "What is our notice period for the Tessaro contract?",
   "files": ["legal/contracts/tessaro.md"],
   "expect": ["60 days"]}
]
```

`files` are paths relative to the folder (any of them counts as correct);
`expect` are case-insensitive patterns that must all appear in a right answer
(use `a|b` for alternatives). Twenty to fifty real questions from real users
beat hundreds of generated ones. Your documents never leave your machine.

## What would count as success

Decide this before looking at the results:

1. **Retrieval:** `map` or `map+embed` finds the right file at least as often as
   `embed-all`, within 5 percentage points, while scanning under 5% of the passages.
2. **Answers:** the local model's answers from the map's passages are right at
   least as often as from `embed-all`'s, within 5 points.
3. **Freshness:** new files are found right after they're added, with ingest
   taking well under a second.

If 1 and 2 hold, the "small model that knows where to look" idea survives
contact with real data. If the map falls clearly behind `embed-all`, look at the
"Misses" section of the report: it shows where routing took a wrong turn, which
is the thing to fix next (for example, a fallback to wider search on low
confidence, or an embedding-based router).

## What we already saw without any models

Run in a sandbox with no Ollama, so `embed-all` used a weak no-model stand-in
(LSA) and answers weren't generated. Treat as a smoke test, not a result.

| Method | Right file found (35 questions) | Passages scanned per question |
|---|---|---|
| keyword-all | 69% | 25,412 |
| embed-all (stand-in) | 69% | 25,412 |
| map | **74%** | **284** |
| map+embed (stand-in) | 66% | 635 |

On the 5 freshness questions, the map absorbed the new files in about 0.1 s and
found 3 of 5; the stand-in embedding methods found 4 of 5. The real embedding
model on your machine is the test that matters.
