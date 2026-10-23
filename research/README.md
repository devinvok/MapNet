# Research: does routing through a map beat searching everything?

`benchmark.py` builds a synthetic company (6 departments × 4 topics, N systems per
topic, plus chat channels full of chatter that name-drops the same projects) and
compares, at growing scale:

- **flat**: score every document (the plain-RAG baseline)
- **map-summaries**: route by automatic node summaries only
- **map**: summaries + the names each node holds (the default)
- **map+feedback**: `map` after learning from 3,000 answered questions

on two question styles: **normal** (question words overlap the documents) and
**slang** (everyday words that appear in no document). hit@1 = right document
ranked first; comparisons = nodes + documents scored per query. Full numbers in
`results.json` (300 test questions per cell). Run time: about 10 minutes.

```bash
python research/benchmark.py          # full
python research/benchmark.py --quick  # skips the largest size
python research/synthetic_demo.py     # feature walkthrough on synthetic data
```

## Normal questions

| Systems per topic | Docs | Flat hit@1 | Map hit@1 | Flat comparisons | Map comparisons |
|---|---|---|---|---|---|
| 1 | 1,440 | 0.977 | 0.977 | 1,440 | 217 |
| 4 | 5,760 | 0.987 | 0.997 | 5,760 | 212 |
| 16 | 23,040 | 0.987 | 0.997 | 23,040 | 297 |
| 64 | 92,160 | 0.927 | **0.987** | 92,160 | **575** |

## Slang questions

| Systems per topic | Flat hit@1 / hit@5 | Map hit@1 / hit@5 | Map + feedback hit@1 / hit@5 |
|---|---|---|---|
| 1 | 0.957 / 1.000 | 0.957 / 0.993 | 0.957 / 1.000 |
| 4 | 0.953 / 1.000 | 0.960 / 1.000 | 0.960 / 1.000 |
| 16 | 0.850 / 1.000 | 0.843 / 0.963 | **1.000 / 1.000** |
| 64 | 0.600 / 1.000 | 0.617 / 0.870 | **0.833** / 0.840 |

## Findings

1. **The map matched or beat flat search on every normal-question run while
   scoring 7–160× fewer items.** Its cost grows slowly with size; flat grows
   linearly. At the largest size it was also more accurate, because routing skips
   chat chatter that mentions the same names.
2. **Summaries alone don't scale.** Topic centroids only: 0.98 → 0.54 hit@1 as
   sources multiply, because sibling systems look identical on average. Advertising
   the names each node holds is what keeps routing precise.
3. **Feedback teaches new vocabulary.** On slang questions, 3,000 answered
   questions raised hit@1 to 1.000 (16/topic) and 0.833 (64/topic), versus flat's
   0.850 and 0.600.
4. **The main risk is a wrong turn.** At the largest size the map's slang hit@5
   (0.84) is below flat's (1.00): a wrongly routed answer is never seen. Needed: a
   fallback to wider search on low confidence.
5. **Feedback without names can hurt at scale** (summaries + feedback: 0.31 at
   64/topic): learned vectors overfit to individual siblings.
6. **In one process flat search is faster in wall-clock time** (~2–9 ms vs
   ~10–30 ms): one sparse matrix multiply versus Python walking a tree. The map's
   advantage is work done and nodes contacted, which becomes time once sources are
   remote or the searcher is an agent paying per file.

## Limits

Synthetic data with generated questions; a lexical (hashed bag-of-words) encoder;
feedback uses the known right answer, where a real system would use a user's
confirmation or an agent's verification.
