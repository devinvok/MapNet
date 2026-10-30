# MapNet: briefing for the next agent

You are taking over a research prototype from a previous Claude session. Read this whole file first.
The owner is Gal (GitHub `galmaron26`). The repo is https://github.com/galmaron26/walk--and-talk-ai
(package name `mapnet`, public, MIT). Everything below was learned over many working sessions with him.

## 1. The idea

A DNS-style, live, permission-aware **map of knowledge** that a small private model (and agents)
navigate from general to specific, instead of one giant model that memorises the world or one flat
index over everything. "Know where to look, not everything."

- **Live:** add/edit/delete a file and the next question sees it. No retraining, no re-index.
- **Permission-aware:** `.access` files give folders role lists; a caller never even *contacts* a
  branch it cannot open. Restricted references stay in their own branch/graph.
- **Agents leave trail markers** (`mapnet/notes.py`): pointers only (never put in a prompt),
  verified on every use, strength decays, deleted at zero, scoped to the writer's roles, dropped if
  the target is deleted. This came from a real failure mode (agents leaving each other
  instructions), so keep it pointer-only.
- **Three model roles, deliberately separate:** navigator (words + a small embedding model, never
  writes text), cartographer (writes the map: file descriptions, section cards; `--map-model`),
  reader (reads the few passages found and answers; `--reader-model`, can be bigger).

Gal's principles. Do not re-litigate them without evidence:
1. **General to specific, like DNS.** Folder map gets you to the file, the file's own map gets you
   to the passage.
2. **Keep it real time.** No bulk precomputation of the whole corpus; things are mapped/embedded
   when a question first arrives (cost of a first visit is counted). Cheap one-off steps at
   registration time are fine; embedding every passage up front is not.
3. **Compare against the other directions and choose the best**, not just follow one signal.
4. **The map should hold references, not data** (`ref-map`): a reference says "this is answered at
   file X"; content is fetched on arrival. References may be multi-homed (one item, several places).
5. **The long-term goal is cold-start quality.** Learning from use (question logs) comes after.

## 2. Repo map

```
mapnet/            core.py (the map: nodes, routing, ACLs, live add/remove, feedback)
                   folder.py (FolderMap: polling watcher, .access, chunk_with_sections)
                   notes.py, assistant.py, llm.py (Ollama/Anthropic/Extractive readers), embed.py
                   cards.py (CardStore, file descriptions, section cards, LazyCarder)
                   meaning.py (MeaningRouter, add-only), embedtree.py (EmbedTree)
                   refmap.py (RefMap, TermTranslator), refgraph.py (RefGraph = HNSW over references)
realtest/          run.py (the experiment), retrievers.py (all methods), corpus.py, judge.py,
                   qgen.py, setup.sh/.ps1, questions/k8s.json (40 hand-written Qs), README.md
tests/             49 tests, no model/network needed (fake_ollama.py is a stand-in server)
research/          synthetic scale benchmark; examples/company, examples/drop-in; docs/DEMO_SCRIPT.md
.github/workflows  tests on ubuntu/windows/macos, Python 3.10 and 3.12
```

Run the tests with `python -m pytest -q` (about 25 s). **Run them before every commit.**
`realtest/README.md` documents every method and flag; keep it updated when you add one.

## 3. Working with Gal

- He works on **Windows 11 + PowerShell**, with a **local Ollama** (`qwen2.5:3b`, `qwen2.5:7b`,
  `nomic-embed-text`) on a CPU laptop. **You will not have his Ollama** in a cloud sandbox. Smoke-test
  the harness against `tests/fake_ollama.py` (see section 8); its embeddings are junk, so its accuracy
  numbers mean nothing. Real numbers only come from his runs; he pastes the console output or
  `report.md`/`results.json` back to you.
- Give him commands **one per line**, PowerShell syntax, and say which folder to `cd` into. Past
  mistakes: several commands on one line, wrong folder, stale code after copying a zip. The repo is now on GitHub; check whether he has git set up on the laptop before telling him to `git pull`
  (until then he copies files over his folder with Replace). If Ollama says port 11434 is already in use, it is already running.
- He thinks in terms of the network picture (DNS, hubs, directions). Explain results in plain
  language with that picture, say what you measured vs what you inferred, and **do not oversell**.
  He asks "why" a lot; answer the actual question first.
- He was right when he corrected the previous agent: bulk precomputation (rejected), "compare to
  other directions", "we are still working with ref-map". When he proposes an idea, say honestly
  whether it will move the numbers (see the question-log discussion in section 7) before building.
- Ask before expensive or ambiguous work; otherwise make the call and say which reading you took.
  Full test runs cost him 40-70 minutes, so make each run count and print progress.
- Commit messages end with the attribution lines your harness gives you. Do not push unless he asks.

## 4. The experiment (`python -m realtest.run`)

Corpus: Kubernetes docs (downloaded by `realtest/setup.sh` / `setup.ps1`, or on first run), 40 hand-written questions, 5 of them "fresh" (their files
are held out and added after indexing). The run he uses:

```powershell
python -m realtest.run --only concepts,tasks,tutorials,setup --map-model qwen2.5:3b --reader-model qwen2.5:3b --explain --skip map+meaning,map+file-cards,embed-tree,map+lazy-cards --answer-with embed-all,map+sections,ref-map,ref-map+terms,ref-graph,ref-graph+terms
```

`--only` gives 469 files / 7,460 passages / 35 normal + 5 fresh questions. File descriptions for
exactly those 469 files exist on his machine in `realtest/data/cards` (31 min of 3B model time,
keyed by content hash). Without descriptions the reference methods only have path + headings.

Metrics: **Right file in top-k** (a right file is among the files of the 4 returned passages),
**MRR**, **Answer in passages** (keyword check, no model), **Answer right** (reader's answer),
**Scanned**, **Lookup** (median hides first visits: read mean/slowest/total in the time tables).
The run is one method at a time; reader answers come after all retrieval. Every ref-map variant has its
**own** file cache and embedding cache (`RefMap.fork`), so first-visit costs are not shared; the two
`+terms` variants share one `TermTranslator`, so translation time lands in whichever ran first.
`--skip` matches whole method names. 35 questions means **one question = 2.9 points**.

### Methods
`keyword-all`, `embed-all` (embeds every passage: the strong baseline; cached on disk), `map`
(word router over the folder tree), `map+sections` (+ section/heading walk), `map+lazy-cards`,
`map+file-cards`, `map+meaning` (add-only embedding router), `map+embed` (uses embed-all's matrix, so
it is *not* real time; reference only), `embed-tree`, `ref-map` (references walked through the folder
zones), `ref-map+terms` (question first translated into technical terms by the small model),
`ref-graph` / `ref-graph+terms` (same references walked as an HNSW graph).
`token-net`, `token-net+assoc`, `token-net+sections` (a model-free token network; see section 5a).

## 5. Results so far (35 hand-written questions, 3B reader, 469 files)

| Method | Right file | Answer in passages | Answer right | Lookup |
|---|---|---|---|---|
| keyword-all | 74% | 57% | 49% | ms |
| embed-all | 80% | 89% | 71% | ms (but 7,460 passages embedded up front) |
| map | 74% | 57% | 57% | 20 ms |
| **map+sections** | 71% | 69% | **69%** | 65 ms, no model |
| map+lazy-cards | 71% | 74% | 71% | 5-10 s/question (card writing) |
| map+file-cards | 71% | 74% | 69% | 6-60 s |
| map+meaning | 71% | 63% | 63% | 6 s |
| embed-tree | 49% | 69% | 60% | 11 s |
| ref-map | 51% | 66% | 54% | 12 s |
| ref-map+terms | 57% | 77% | 71% | 17 s |
| ref-graph | 57% | 69% | 46% | 12 s |
| ref-graph+terms | 63% | 80% | 69% | 14 s |
| token-net (model-free) | 74% | 74% | not run with a reader | 2 ms |
| token-net+assoc | 71% | 74% | not run | 5 ms |
| **token-net+sections** (RRF merge) | **80%** | 77% | not run | 15 ms |

What this means:
- **The plain word map with sections (`map+sections`) matches embed-all's answers with a 3B reader**
  (69% vs 71%), scans ~300 passages instead of 7,460, needs no model, and answers in ~60 ms. Model-written
  cards did not clearly add anything on top.
- **embed-all's real edge is evidence**: 89% of questions have the answer text in the retrieved passages vs
  69%. The 3B reader hides this; with a 7B reader embed-all reached 86% in an earlier run. Not yet re-measured
  on the current set.
- **Reference methods: the diagnosis is solid.** `--explain` scores each question against every reference
  directly with no walk ("control"). Right file among the 6 picked: **folder walk 20, graph walk 24,
  control 25** (with translated terms: folder 23, graph 25). So the folder walk lost 5 questions by never
  entering the right folder (summaries of a whole folder hide outliers); the HNSW graph recovers them and
  reaches the references' ceiling. **10 of 35 questions cannot be reached from the references at all**
  (right file ranks ~26th in the control). The translator often writes the right words (`maxSurge`,
  `ipvs`, `GracePeriodSeconds`) but the references (path + headings + a 3B-written sentence and 5 questions)
  contain no such content, so they cannot match. That is the "get points from the file into the map" gap.
- The word map and the reference map **fail on different questions** (comparing the miss lists of two runs, 6 of the 8 unreachable questions shown
  in the report are found by `map+sections`; word map misses are everyday-vocabulary questions: finalizers,
  nodeSelector, headless service, ReadWriteOncePod, port-forward, downward API, init containers, LimitRange,
  PriorityClass, NetworkPolicy). Complementary, but not yet merged or measured exactly.
- A loss after the walk: graph+terms picks the right file for 25 questions but only 22 reach the 4 returned
  passages (other files' passages outrank them; `ref_weight=0.5`, `max_new=4`).
- **Cost:** the reference variants spend ~12 s per question embedding the files they arrive at (about 4 files
  x 18 passages at ~0.16 s each on his CPU) + ~3 s translating. `ref-map` build is 250 s for 469 files; the graph
  builds in 3.5 s. Reader answers take ~3 s each with the 3B model.
- **Unexplained, reproducible:** `map+sections` answers take ~14.7 s each vs ~3.3 s for every other method
  (seen in three runs; in the first run also `lazy-cards` and `file-cards`). Not the laptop. Hypothesis
  (**unverified**): the bag-of-words router returns long passages and the prompts are longer.
  `results.json` rows have `tokens_in`; compare it per method before trusting any speed claim.
- Fresh-file results (5 questions) are too small to read. `embed-all`'s "ingest 9 ms" is only the cache.

## 5a. The token network (newest; measured in the cloud sandbox, not on Gal's laptop)

Gal's idea: "a token is relative to a word, so build a network of token references; find the tokens closest
to the query, and the token references connect to the pages that contain them." Implemented as
`mapnet/tokennet.py`: tokens are whole words/identifiers (maxSurge -> maxsurge, max, surge; crude stemming);
each token points at the passages containing it with BM25 weights (rare token = specific name = deep in the DNS
sense; headings and path count double); associations between tokens are computed on demand from co-occurrence
(PMI with a support factor), and the strongest associates of each question word light up at weight 0.3. Builds in
about 1.3 s (16,283 tokens, 7,460 passages), 2 ms per question, live (`add`, replace, `remove_missing`,
`TokenNetRoute.sync()`), no model, no embeddings.

- **Offline testing works.** The Kubernetes docs download in ~3 s in the sandbox, and the model-free baselines
  reproduce Gal's numbers exactly (keyword-all 74%/57%, map 74%/57%, map+sections 71%/69%). So any idea that
  needs no model can be measured without his laptop: `python -m realtest.run --data <dir> --only
  concepts,tasks,tutorials,setup --retrieval-only --skip ...` (embed-all is then a weak stand-in; ignore it).
- **token-net alone: 74% right file, 74% answer-in-passages** (map+sections 71% / 69%).
- **Merged with map+sections** (reciprocal rank, c=60): **80% / 77%**; a 2+2 seat merge gave 74% / 80%. The two
  rules were fixed before looking. The oracle union of evidence is 31/35 (89%, same as embed-all). They fail on
  different questions. Right file 80% equals embed-all's, with no model and no up-front embedding.
- **Associations did not help** (71% vs 74% right file; lost the Job-retry question). The links look right
  (ssd -> disktype, nodeaffinity; helper -> sidecar; surge -> maxsurge, rollingupdate; gracefully -> grace,
  sigkill; autoscaler -> horizontalpodautoscaler); the way neighbours are weighted into the score is what needs
  work. Deliberately **not tuned**: the 35 questions have been looked at many times, so tuning on them overfits.
- A test found a real bug (identifier built after dropping stopword parts: ReadWriteOncePod became
  readwritepod); fixed, numbers unchanged.
- Not yet measured: answers right with a reader (needs Gal's Ollama), the 7B reader, anything on a second
  question set. **No access control yet** (see 7a.5).

## 6. Known pitfalls

- Windows: always read/write UTF-8; `results.json` is written before `report.md`.
- The embedder used to rewrite its whole cache after every call (quadratic) and `meaning`/`ref-map` embedded
  one file per request. Fixed: saves at most every 30 s and at exit, batched builds with progress lines. Keep
  progress output on anything that can take more than a minute; the user cannot tell "slow" from "stuck".
- Chunk text must stay identical to the old `chunk()`; section labels are added beside it. All sources
  must go through the same `_score_docs` path (a heading bonus once made CSV/plain files unfairly lose).
- Restricted references: never register or link them outside their own branch/graph; never return what the
  caller cannot open. There are tests; keep them passing.
- Do not state a number you did not measure. The previous agent underestimated run times twice.

## 7. What to do next (agreed order)

1. **Check the slow `map+sections` answers.** Ask him for the latest `results.json`, compare mean `tokens_in`
   per method. If passages are long, consider a length-normalised score or a passage length cap; check whether
   it also biases accuracy.
2. **Read passages with the word router, not by embedding files on arrival.** Use the references/graph only to
   choose files, then `search_in` (section-aware word scoring) for passages. Should cut lookups from ~12 s to
   roughly the translation time (~3 s). Add as new methods (e.g. `ref-graph+terms (words read)`) and compare;
   keep the embedding-on-arrival version as it is.
3. **Extractive points as references** for the 10 unreachable questions: sentences with numbers, defaults and
   identifiers (`terminationGracePeriodSeconds`), copied verbatim from each section (no model), registered as
   `point` references and embedded in batches at registration. About 4 extra minutes; measure it. An alternative
   that costs nothing: a lexical "names held" signal in the walk.
4. **Union with `map+sections`** (add-only, one reserved passage seat, like `MeaningRouter`). First compute
   the exact oracle union from his `results.json`; build it only if the union beats both.
5. **Passage selection after the walk** (25 chosen -> 22 returned): tune `ref_weight`, take 1-2 passages per
   file before filling up.
6. **Run the 7B reader** on the cheap methods (`--reader-model qwen2.5:7b`, skip the slow ones) to see whether
   embed-all's evidence advantage turns into an accuracy gap.
7. **Association map built from the corpus itself** (word co-occurrence, updated live on file add), used to
   expand questions before routing. Classic query expansion, no model.
8. **Then the question log ("stamps")**: at every node on the route, store the question, the file that
   answered it (only after a verified good answer), and 2-3 copied points from the passage; strength decays and
   dies on failed use; roles-scoped; dropped when the file changes. Plus a model-written *general* description
   per folder (about 60 folders, a few minutes). Used only for routing, never shown to a reader. **Agreed test
   before building:** hand-write a second phrasing for each of the 35 questions; agents answer set A, leaving
   stamps, then ask set B. Pass criteria: warm right-file +15 points, cold results change by at most one
   question, planted false stamps die after their first failed use. Warn him in the report that the warm test
   measures memory, not better routing.

## 7a. Update after the token-net work (takes priority over the order above where they conflict)

1. Gal runs `token-net`, `token-net+assoc`, `token-net+sections` with the 3B and 7B readers (the `--answer-with`
   list must include them). This is the first reading of answers right for them.
2. Write a **second, independent question set** (40+, varied phrasing: everyday wording, symptoms, exact
   settings, "what happens if"), with gold files and keyword checks. The current 35 are burned for tuning.
3. Improve the association weighting on that set, e.g. expand only question words that match weakly, scale
   neighbours by their idf, cap the total expansion weight, use the translator's terms as seeds.
4. Use the token network to give the reference methods the content they lack (the 10 questions their references
   cannot reach), or make it the file chooser inside `ref-graph`; keep the comparison fair (own caches).
5. **Before any real use: access control.** `TokenNet` has none. Build one network per access list (like
   `RefGraph`), and never compute or show associations from passages the caller cannot open (an association
   list can leak restricted vocabulary). Make BM25 refresh incremental (it rebuilds in ~1 s per change now).
6. Then the remaining items of section 7 that still apply (7B run, extractive points, question log with its
   agreed test).

Also open: `general_up` (labels/descriptions get a higher HNSW layer) is untested against plain random
levels; the 3B translator's knowledge limits it (it wrote Docker Compose terms for "helper container"); a
7B translator may help.

## 8. Smoke-testing the harness without Ollama

```python
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "tests")
import fake_ollama
srv, url = fake_ollama.start()
os.environ["OLLAMA_HOST"] = url
sys.argv = ["run", "--corpus", "examples/company", "--questions", "<json with files: [...] per item>",
            "--data", "<scratch dir>", "--reader-model", "qwen2.5:3b", "--explain"]
from realtest import run; run.main()
```
`examples/questions.json` uses `file`; the harness wants `files` (a list). Delete any `realtest/results/*`
the smoke run creates.
