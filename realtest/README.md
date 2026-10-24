# The real test

Does MapNet hold up on real, messy documents with a real small model, against
strong baselines? This folder runs that test on your own machine.

## What it compares

Every method sees exactly the same passages of the same files.

| Method | What it is |
|---|---|
| `keyword-all` | Keyword search over every passage (classic search index) |
| `embed-all` | Embedding search over every passage with a local embedding model: the standard modern RAG setup, and the baseline to beat |
| `map` | MapNet: walk the folder map, search only the files it routes to |
| `map+sections` | ...then walk each file's own map (its headings) to the right section |
| `map+lazy-cards` | ...where each file is mapped by your local model **the first time a question arrives at it**, starting from no cards at all. DNS style: the folder map resolves the file, the file's own map resolves the passage, and the answer is cached. First-visit time is counted in its lookup time |
| `map+file-cards` | **The full chain.** Every file also has a one-line description at the folder level, written in advance, so the router can find files nobody has visited yet; sections are mapped on arrival as above, and the file-level decision carries down into passage ranking. Only if you've built file descriptions, see below |
| `map+meaning` | The chain above, walked by **meaning** as well as words. It embeds the *map*, not the passages: one vector per file up front (its path, description and outline), and a file's sections only when a question first arrives there. At each level the word router's best branches are kept and meaning's best are added, and one of the k passages goes to meaning's first choice, so a weak meaning signal costs little and a good one fixes wrong turns |
| `map+cards` | ...with every file's sections mapped in advance (only if you've built them, see below) |
| `ref-map` | **A map of references, DNS style.** Zones hold references, not data: each file's label and description in its own folder's zone, and **each question the model wrote for it registered wherever that question resolves**, next to the closest existing reference to another file. So one file is reachable from several zones (unlike a tree, where each item has one place). Adding a file is resolving its questions, with no rebuild. Restricted files are only referenced inside their own branch. Contents are fetched and embedded only when a question arrives. Needs the file descriptions (`--files-only`) |
| `embed-tree` | **The map walked by meaning alone, embedded in real time.** It starts with only one short "direction" vector per file (path, headings, description). Folders keep a few representative vectors of what's below them. When the walk arrives at a file it hasn't embedded yet, it embeds that file's passages right then and keeps them, so the map sharpens wherever questions go. It also checks a branch no one has looked inside unless the known branches are clearly better, and looks wider when the match stops rising on the way down. Its embedding cache starts empty each run, so every first visit is real, counted work |
| `map+embed` | MapNet routes to a wider set of files, embeddings rank the passages inside them |

For each it measures: whether the right file is among the passages handed to the
model, how high it ranks, **whether those passages actually contain the answer**
(checked by keyword, no model needed), how much work the search did, how long it
took, whether the local model's final answer is right, and how fast newly added
files become answerable.

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

## Testing ref-map and embed-tree

Both need `nomic-embed-text`; `ref-map` also needs the file descriptions from
`python -m mapnet.cards ... --files-only`. Building ref-map embeds about seven
short texts per file (label, description, five questions) and places each
question by walking the map; a few minutes on a laptop, counted in the report.

```bash
python -m realtest.run --only concepts,tasks,tutorials,setup --retrieval-only --skip keyword-all,map,map+sections,map+lazy-cards,map+file-cards,map+meaning
python -m realtest.run --only concepts,tasks,tutorials,setup --reader-model qwen2.5:7b --skip keyword-all,map,map+sections,map+lazy-cards,map+file-cards,map+meaning --answer-with embed-all,embed-tree,ref-map
```

The report says how many references were placed outside their file's own
folder (the multi-homing a tree can't do) and how much was embedded on arrival.

### embed-tree details

Fastest first look, no answers (minutes):

```bash
python -m realtest.run --only concepts,tasks,tutorials,setup --retrieval-only --skip keyword-all,map,map+sections,map+lazy-cards,map+file-cards,map+meaning
```

Then with the 7B reader on the three methods that matter:

```bash
python -m realtest.run --only concepts,tasks,tutorials,setup --reader-model qwen2.5:7b --skip keyword-all,map,map+sections,map+lazy-cards,map+file-cards,map+meaning --answer-with embed-all,map+embed,embed-tree
```

The report says how many files embed-tree embedded on the way, out of how many,
and how long that took. Checked without a real embedding model (stand-in vectors,
35 questions): flat search over every passage found the right file for 25; the
fully embedded tree walk for 22 (17 before folders kept several representative
vectors instead of one average); the real-time walk for 17 on its first pass
while embedding 128 of 469 files. Before a file is visited, the walk relies on
its short label, so its accuracy depends on how good those directions are. That
is what the run with `nomic-embed-text` and your file descriptions measures.

## Three models, three jobs

Because the map does the finding, the work splits into roles that need very
different models:

| Role | Job | Runs | Model |
|---|---|---|---|
| **Navigator** | walks the map to the right passages; never writes text | every question | words + a small embedding model (`--embed-model`, default `nomic-embed-text`, ~137M) |
| **Cartographer** | writes the map: file descriptions, section cards | once per file (or on change) | `--map-model` |
| **Reader** | reads the k passages the navigator found and answers | every question | `--reader-model` |

The reader only ever sees k passages, so it can be much bigger than a model that
has to search. Give several readers to compare them **on exactly the same
passages** in one run:

```bash
ollama pull qwen2.5:7b
python -m realtest.run --only concepts,tasks,tutorials,setup --map-model qwen2.5:3b --reader-model qwen2.5:3b,qwen2.5:7b --skip keyword-all,map,map+embed
```

The report then has one "Answer right" column per reader. Retrieval runs once;
then each reader answers every method's passages in turn (so Ollama isn't
swapping models on every question). A 7B reader needs about 6 GB of free memory
and roughly twice the time per answer of a 3B one.

## Section cards

`map+lazy-cards` needs no preparation: it maps files as questions reach them, so
it runs as part of the normal test (expect a few extra minutes for the first
visits; skip it with `--no-lazy`). The report says how many files were mapped and
how long a first visit took.

### File descriptions (for `map+file-cards`, about an hour)

One short model call per file, from its headings and opening only: what the file
is for and five questions people would ask, in everyday words. They fill in the
folder level of the map, which mapping on arrival can't do (a file that is never
visited never gets mapped). In everyday use (`ask.py`) they are written in the
background while you work.

```bash
# time a sample first: it prints seconds per file and an estimate
python -m mapnet.cards realtest/data/k8s-docs --store realtest/data/cards --files-only --only concepts,tasks,tutorials,setup --limit 10
# then the rest (Ctrl+C any time; rerun to continue)
python -m mapnet.cards realtest/data/k8s-docs --store realtest/data/cards --files-only --only concepts,tasks,tutorials,setup
# then the test
python -m realtest.run --only concepts,tasks,tutorials,setup
```

Files in the freshness test are treated as undescribed until they arrive; their
description is written on arrival and the time is reported.

### Mapping every section in advance (optional, overnight)

A file has its own map in its headings, and `map+sections` walks it. But people
ask in words the docs never use ("why is my namespace stuck deleting?" is answered
by a page about *finalizers*). Cards close that gap: your local model reads each
section once and writes a short card, what it covers and the questions a person
would ask, in plain words. The map routes by the cards; the cards are never shown
to the answering model.

Writing cards takes one model call per few sections. For the whole Kubernetes
docs that's about 2,950 calls (most of them for the generated API reference), so
the test supports running on the hand-written folders only: about 950 calls,
roughly an overnight job on a laptop CPU. Every method then runs on the same
smaller set, so the comparison stays fair.

```bash
# 1. time a sample (prints seconds per file and an estimate for the rest)
python -m mapnet.cards realtest/data/k8s-docs --store realtest/data/cards --only concepts,tasks,tutorials,setup --limit 10

# 2. the rest, overnight. Ctrl+C any time; rerunning continues where it stopped
python -m mapnet.cards realtest/data/k8s-docs --store realtest/data/cards --only concepts,tasks,tutorials,setup

# 3. the test, on the same folders
python -m realtest.run --only concepts,tasks,tutorials,setup
```

Files used in the freshness test are treated as never carded: when they "arrive"
in phase B, the model writes their cards on the spot, and that time is reported.

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

## How the meaning router behaves (sanity check, no real embedding model)

The router was checked on the 35 hand-written questions with "meaning" signals of
known quality, since no real embedding model was available where it was built:

| Meaning signal | Right file found | Answer in passages |
|---|---|---|
| none (map+sections) | 25/35 | 24/35 |
| pure noise | 25/35 | 23/35 |
| weak stand-in (LSA) | 25/35 | 23/35 |
| perfect (knows the answer files) | **35/35** | **28/35** |

So the design costs at most about one question when meaning is useless and
fixes every wrong turn when it's perfect. A real model (`nomic-embed-text`) will
land in between; the real test measures where.

## Second real run: hand-written folders (469 files), sections and mapping on arrival

| Method | Right file found | Answer in passages | Answer right | Scanned |
|---|---|---|---|---|
| embed-all | 80% | 89% | **71%** | 7,460 |
| map | 74% | 57% | 57% | 253 |
| map+sections | 71% | 69% | 66% | 313 |
| map+lazy-cards | 71% | 71% | **71%** | 412 |

The small model answered as accurately from MapNet's passages as from embedding
search, on 35 questions (each question is about 3 points, so this is a tie, not a
win). Remaining gaps: finding the file (71% vs 80%), which is what file
descriptions target, and a 32 s first visit per file on a laptop CPU.

## First real run (qwen2.5:3b + nomic-embed-text, Windows laptop)

| Method | Right file found | Answer right | Scanned per question |
|---|---|---|---|
| keyword-all | 69% | 46% | 25,412 |
| embed-all | **77%** | **77%** | 25,412 |
| map | 74% | 51% | 284 |
| map+embed | 74% | 66% | 635 |

Building: map 17.6 s, embeddings 2,076 s (35 min). The map found the right file
almost as often while reading 1% of the text, but the model answered much less
often from its passages: they came from the right file but the wrong part of it.
That is what `map+sections` and `map+cards` were built to fix. Without any model,
the share of questions whose answer is actually in the passages went from 63%
(`map`) to 71% (`map+sections`), and from 57% to 69% on the hand-written folders.
The answer-quality numbers for the new methods need a run on your machine.

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

## Where the time goes

The test runs **one method at a time** (all its questions, then the next method), so Ollama
keeps the same model loaded instead of swapping on every question. The report's header lists
the total wall-clock time (setup, phase A, phase B) and the time spent retrieving and
answering. Each table is followed by a **time table** per method: lookup median, mean,
slowest and total, then for every reader the answer time, the time per question (lookup +
answer) and the total. The median hides first visits (a file mapped or embedded for the
first time), so look at the mean, the slowest lookup and the total. Reader answers are
usually the largest part: every method's passages are answered by every reader. To make a
run shorter, use `--skip`, `--answer-with` (which methods get answered) or a single
`--reader-model`.
