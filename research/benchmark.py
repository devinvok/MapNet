"""
Does routing through a map beat searching everything?

For growing company sizes, compares:
  flat            score every document (the plain-RAG baseline)
  map-summaries   route by automatic node summaries only
  map             route by summaries + names each node holds (default)
  map+feedback    `map` after learning from 3,000 answered questions

on two question styles:
  normal   question words overlap the documents' wording
  slang    question uses everyday words found in no document

Metrics: hit@1 (right document ranked first), comparisons per query (nodes +
documents scored, a proxy for compute and network calls), and latency.

    python research/benchmark.py            # full run, ~5-10 minutes
    python research/benchmark.py --quick    # smaller sizes
"""
import json
import sys
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapnet import MapNet
from mapnet.synthetic import build_company, make_questions

SIZES = [1, 4, 16] if "--quick" in sys.argv else [1, 4, 16, 64]
N_TEST, N_TRAIN = 300, 3000


def evaluate(fn, questions):
    hit1 = hit5 = cost = 0
    t0 = time.perf_counter()
    for q, gold in questions:
        r = fn(q)
        ids = [h.doc_id for h in r.hits]
        hit1 += ids[:1] == [gold]
        hit5 += gold in ids
        cost += r.cost
    n = len(questions)
    return dict(hit1=round(hit1 / n, 3), hit5=round(hit5 / n, 3), cost=round(cost / n),
                latency_ms=round((time.perf_counter() - t0) / n * 1000, 2))


def build(size, **kw):
    sources, facts = build_company(size, noise_ratio=0.5)
    m = MapNet(**kw)
    t0 = time.perf_counter()
    for s in sources:
        m.register_source(**s)
    return m, sources, facts, time.perf_counter() - t0


def main():
    rows = []
    for size in SIZES:
        m, sources, facts, build_s = build(size)
        n_docs = sum(len(s["docs"]) for s in sources)
        print(f"\n== {size} systems per topic: {len(sources)} sources, {n_docs:,} docs "
              f"(built in {build_s:.1f}s) ==", flush=True)
        for style in ["normal", "slang"]:
            test = make_questions(facts, N_TEST, seed=1, slang=style == "slang")
            train = make_questions(facts, N_TRAIN, seed=2, slang=style == "slang",
                                   exclude=[g for _, g in test])
            res = {"flat": evaluate(m.flat_search, test)}
            m.name_weight = 0.0
            res["map-summaries"] = evaluate(m.search, test)
            m.name_weight = 0.5
            res["map"] = evaluate(m.search, test)
            # feedback on a fresh copy so it doesn't leak into the other style
            fb, *_ = build(size)
            for q, gold in train:
                fb.feedback(q, gold)
            res["map+feedback"] = evaluate(fb.search, test)
            fb.name_weight = 0.0
            res["map-summaries+feedback"] = evaluate(fb.search, test)
            for name, r in res.items():
                print(f"  {style:6} {name:23} hit@1 {r['hit1']:.3f}  hit@5 {r['hit5']:.3f}  "
                      f"comparisons {r['cost']:>7,}  {r['latency_ms']:>6} ms", flush=True)
                rows.append(dict(systems_per_topic=size, sources=len(sources), docs=n_docs,
                                 style=style, method=name, **r))
    with open(Path(__file__).parent / "results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
