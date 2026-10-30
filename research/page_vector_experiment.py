"""
Can a page-level vector replace (or sit beside) the model-written file descriptions?

Uses only vectors the real test already cached, so it needs no model calls beyond a few
query embeddings. File-level scores compared:

  descriptions   best reference (label/description/guessed questions) per file   [ref-map's control]
  page-mean      mean of the file's passage vectors (one vector per page)
  best-passage   best passage per file (embed-all, grouped by file)
  mix a          a * page-mean + (1 - a) * descriptions

Then the realistic pipeline: take the top 6 files by that score, rank their passages
against the question, and ask whether a passage of the right file is in the top 4 (the
same measure as the main table).

    python research/page_vector_experiment.py [--cards realtest/data/cards7b]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mapnet.cards import CardStore  # noqa: E402
from mapnet.embed import OllamaEmbedder  # noqa: E402
from mapnet.refmap import RefMap  # noqa: E402
from realtest.retrievers import Shared  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "realtest/data"))
    ap.add_argument("--cards", default=str(ROOT / "realtest/data/cards7b"))
    ap.add_argument("--only", default="concepts,tasks,tutorials,setup")
    a = ap.parse_args()
    data = Path(a.data)
    work = data / "work" / "k8s-docs"

    shared = Shared(work, cards=CardStore(a.cards))
    srcs = shared.map.sources()
    print(f"{len(srcs)} files, {sum(len(s.texts) for s in srcs):,} passages")

    p_emb = OllamaEmbedder(cache_dir=str(data / ".cache"))
    r_emb = OllamaEmbedder(cache_dir=str(data / "work" / "refmap-cache"))

    # passage vectors (cached by the real test) and page-mean vectors
    paths = [s.path for s in srcs]
    P = {}
    for s in srcs:
        V = np.asarray(p_emb.embed(list(s.texts), "document"), dtype=np.float32)
        P[s.path] = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    page = {p: (lambda m: m / max(np.linalg.norm(m), 1e-9))(P[p].mean(0)) for p in paths}

    # descriptions: the references ref-map registers (home ones only; same texts the test embedded)
    rm = RefMap(shared.map, r_emb)
    for s in srcs:
        rm.register(s, place=False)
    file_of = {s.path: s.doc_ids[0].split("#")[0] for s in srcs}

    items = [x for x in json.load(open(ROOT / "realtest/questions/k8s.json"))
             if all(f.split("/")[0] in a.only.split(",") for f in x["files"])]
    have = set(file_of.values())
    items = [x for x in items if set(x["files"]) <= have]  # drop files held out for the freshness test
    print(f"{len(items)} questions")

    mixes = [0.0, 0.3, 0.5, 0.7, 1.0]
    names = ["descriptions" if m == 0 else "page-mean" if m == 1 else f"mix {m:.1f}" for m in mixes]
    names.append("best-passage")
    stats = {n: dict(r1=0, r6=0, pipe=0) for n in names}

    for it in items:
        gold = set(it["files"])
        q = np.asarray(r_emb.embed([it["q"]], "query")[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        desc = {p: -2.0 for p in paths}
        for refs in rm.refs.values():
            for r in refs:
                desc[r.target] = max(desc[r.target], float(r.vec @ q))
        pg = {p: float(page[p] @ q) for p in paths}
        bp = {p: float((P[p] @ q).max()) for p in paths}
        scores = {n: {p: m * pg[p] + (1 - m) * desc[p] for p in paths} for n, m in zip(names, mixes)}
        scores["best-passage"] = bp
        for n in names:
            order = sorted(paths, key=lambda p: -scores[n][p])
            ranks = [file_of[p] in gold for p in order]
            stats[n]["r1"] += ranks[0]
            top6 = order[:6]
            stats[n]["r6"] += any(ranks[:6])
            pool = [(float(P[p][i] @ q), p) for p in top6 for i in range(len(P[p]))]
            pool.sort(key=lambda x: -x[0])
            stats[n]["pipe"] += any(file_of[p] in gold for _, p in pool[:4])

    N = len(items)
    print(f"\n{'file score':<14}{'top-1':>8}{'top-6':>8}{'top-4 passages after routing to 6 files':>44}")
    for n in names:
        s = stats[n]
        print(f"{n:<14}{s['r1']/N:>8.0%}{s['r6']/N:>8.0%}{s['pipe']/N:>44.0%}")


if __name__ == "__main__":
    main()
