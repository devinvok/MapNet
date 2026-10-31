# MapNet: where we stand and the road ahead (31 Oct 2026)

## The decision

After testing more than twenty ways to find the passage that answers a question, one is chosen as
MapNet's retrieval core:

**Words find the neighbourhood, meaning picks the page** (`mapnet/shortlist.py`,
`token-net+rerank` in the real test). The token network (no model, live, about 2 ms) returns its 50
best passages; only those are embedded, the first time a question reaches them, and they are ordered
by meaning.

| Questions | token-net alone | **token-net+rerank** | embed-all (every passage embedded up front) |
|---|---|---|---|
| hand-written (40) | 75% | **80%** | 78% |
| everyday wording, dev half (112) | 38% | **45%** | 41% |
| everyday wording, test half (108, scored once) | 38% | **51%** | 56% |
| everyday wording, all 220 | 38% | **48%** | 49% |

Right file among the 4 returned passages. Embedding cost: at most 50 passages per question, about
5 s on Gal's CPU the first time, nothing after; embed-all needs all 7,703 passages embedded first
(about 12 minutes on the same machine).

**What this means, plainly:** it finds the right file about as often as embedding search, while
staying live and real time. It does **not** beat it: on the test half it is 5 points behind; across
all 220 questions the gap is 2 questions. And once a 7B model answers from the passages (step 1
below, measured the same day), embedding search is ahead: 86% of answers right against 77% (35
questions). So the claim we can make is "finds files as well as embedding everything, without
embedding everything, with live updates and permissions", not "better than embeddings"; and where
a corpus can be embedded up front, embed-all is the better ranking layer under the map.

## What is closed, and why

Each line was measured; none moved the everyday-language questions by more than noise (±2 questions
of 112).

| Idea | Result | Why it failed |
|---|---|---|
| Word associations (`token-net+assoc`) | 71% vs 74% (hand) | neighbours look right, weighting dilutes the question |
| Words beside words, proximity (`tokenwin.py`) | 34–38% (dev) | finds how text is written, not what it is about |
| Words linked by context vectors (`tokengraph.py`), convergence | 38% (dev) | same words as above, just found another way |
| Recursive spreading, 1–30 hops (Personalised PageRank) | 36–45/112 | settles after ~3 hops; common words are hubs and drag towards generic pages |
| Embedding every word, question → nearest words | 38–41% (dev) | nearest "words" are page names repeating the question's topic |
| Gateways: a model picks 3 of 45 folders | 7B: right folder 39%; 24–26% right file | the docs are filed by kind (concepts/tasks/tutorials), not by topic |
| Model-written file cards, reference maps (`ref-map`, `ref-graph`) | 51–63% (hand) | a summary leaves out the detail a question asks for |

The common lesson: the token network reaches the right neighbourhood (right file in its top 50 for
70% of everyday questions) and then ranks by shared words, which this kind of question was written to
defeat. The misses are not missing synonyms; they are passages that share the question's filler words
(want, make, need, ensure) outranking the page that answers. Ranking needs something that reads
meaning, and it only needs to read the shortlist.

## What MapNet is for

Retrieval accuracy is now on par with the standard approach. What MapNet has that the standard
approach does not:

1. **Live**: a new or edited file is found by the next question; nothing to re-index.
2. **Permission-aware by construction**: a caller never reaches a branch it cannot open.
3. **Real time**: no up-front embedding of the corpus; cost is paid only where questions go.
4. **A map agents can walk and mark**: folders, files, sections, and trail markers (`notes.py`).

So the product is **the live, permission-aware map that agents navigate**, with
`token-net+rerank` as its search. The road below builds that, in order. Each step has a test that
says whether it is done.

## The road

### Phase 1: make the core real (needs Gal's laptop for steps 1 and 5)

1. **Answers, not just files.** Run the reader on the chosen method against embed-all, 3B and 7B:
   ```powershell
   cd E:\Projectd\MapNet\MapNet
   python -m realtest.run --only concepts,tasks,tutorials,setup --reader-model qwen2.5:7b --skip map+meaning,map+file-cards,embed-tree,map+lazy-cards,ref-map,ref-map+terms,ref-graph,ref-graph+terms,page-mean,page-text --answer-with embed-all,token-net,token-net+rerank,map+sections
   ```
   Done when: answers right for `token-net+rerank` is within 3 points of embed-all with the 7B reader.
   If it is more than 5 points behind, stop tuning ranking and use embed-all as the ranking layer
   under the map (keep everything else).

   *Measured 31 Oct 2026 (35 hand-written questions, `qwen2.5:7b`):* answer right embed-all 86%,
   token-net+rerank 77%, token-net 74%, map+sections 63%; answer in passages 89% / 83% / 74% / 69%;
   right file 80% / 80% / 74% / 71%. 9 points behind (3 questions), so the rule applies:
   **where the corpus can be embedded up front, rank with embed-all under the map**;
   `token-net+rerank` stays the option for live corpora and for not embedding everything (it
   costs ~7 s per question when nothing is cached). Both must go through the same permission
   filter (embed-all does not have one yet: add it before it is used for real).
2. **Access control in the token network and the shortlist.** *Done 31 Oct 2026*
   (`tests/test_tokennet_access.py`). With `roles`, only passages the caller may open are scored, and
   word rarity, passage length and associations are counted from those passages alone: scores equal
   those of a network where the restricted passages never existed. A restricted passage is never
   returned and never embedded for that caller. Also fixed: editing or deleting an `.access` file
   now re-indexes the files beneath it on the next sync (before, a revoked role kept access until the
   documents themselves changed). Not covered: the research variants `tokenwin.py`/`tokengraph.py`
   take no roles (they are not part of the core).
3. **Incremental index updates.** The token network rebuilds BM25 weights (~1 s) on every change. Done
   when: adding one file costs milliseconds and results are identical to a full rebuild.
4. **Question filler words.** A fixed list (want, make, need, ensure, sure, certain, specific...)
   chosen from the language, not from the test questions. Done when: measured on a new question
   set, not on the 220 already looked at.
5. **A new question set** (v3, 100+, hand-checked, several right files allowed per question). Both
   current halves have now been seen; the test half was scored once and must not be used for tuning.
   About half of the generated questions were judged unanswerable (53 of 112 kept on dev).

### Phase 2: the agent interface (the gateway idea, done where it works)

6. **An MCP server over MapNet**: `search(question)`, `open(file, section)`, `list(folder)`,
   `leave_note` / `read_notes`, all filtered by the caller's roles. Gateways failed as a fixed folder
   menu; an agent that can search, read, and search again is the chooser instead.
7. **One extra round**: the reader may search again with words it learned from the first passages
   (agentic search). Done when: measured against the single round on v3, with time per question.

### Phase 3: better ranking (only if Phase 1 step 1 asks for it)

8. **A real reranker** (a small cross-encoder, ~22M parameters, or a local decision model) instead of
   embedding similarity on the shortlist. Needs PyTorch and a model download, so ask first.
9. **Widen the shortlist** where words find nothing: merge in `map+sections`' candidates.

### Phase 4: learning from use

10. **The question log ("stamps")** as agreed in CLAUDE.md section 7.8, with its A/B test (second
    phrasing of each question, warm +15 points, cold unchanged, planted false stamps die on first
    failed use).

## Reproducing the decision

```powershell
cd E:\Projectd\MapNet\MapNet
python research/gateway_experiment.py --corpus <copy of concepts,tasks,tutorials,setup> --cache <dir>
```
Add `--test` to score the sealed half (done once on 31 Oct 2026). Needs Ollama with
`nomic-embed-text` and `qwen2.5:7b`; model answers are cached in `--cache`.
