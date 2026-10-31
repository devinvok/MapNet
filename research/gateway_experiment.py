"""
Deciding experiment: gateways chosen by a small local model + a shortlist reranked by meaning.

Candidates, all scored on the same questions (right file among the 4 returned passages, and the
answer's key word among them):

  token-net                 words only (BM25 over tokens), no model
  embed-all                 every passage embedded up front (the strong baseline; not real time)
  token-net+embed-all       rank merge of the two (reciprocal rank, c=60)
  token-net@50 > embed      token-net's top 50 passages re-ranked by meaning: only those 50 are
                            embedded, on arrival (real time)
  token-net@50 > rrf        same 50, final order = rank merge of word rank and meaning rank
  gate:<model> > ...        the model first picks 3 of the 41 folders (the gateways) from a menu of
                            folder names and the page names inside; the rest runs only inside them

Question sets: the plain-language dev half, the dev half filtered by a judge that never sees any
retrieval result ("is this a sensible question with a specific answer?"), and the hand-written 35.
The test half is only scored with --test, once, after the method is chosen.

Every model answer (judge, gateway picks) is cached in --cache, so a rerun costs nothing.

    python research/gateway_experiment.py --corpus <dir with concepts,tasks,tutorials,setup> --cache <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mapnet.embed import OllamaEmbedder  # noqa: E402
from mapnet.tokennet import tokens  # noqa: E402
from realtest.retrievers import Shared, TokenNetRoute  # noqa: E402

HOST = "http://localhost:11434"


def ollama(model, system, user, timeout=600):
    body = json.dumps({"model": model, "stream": False, "format": "json", "keep_alive": "30m",
                       "options": {"temperature": 0, "num_ctx": 8192},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["message"]["content"]


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text(encoding="utf8")) if self.path.exists() else {}

    def get(self, key, make):
        if key not in self.d:
            self.d[key] = make()
            self.path.write_text(json.dumps(self.d, indent=1), encoding="utf8")
        return self.d[key]


JUDGE = """You check questions for a test of a documentation search engine. You get a question a user
might ask about Kubernetes and the answer the documentation gives.
Say ok=true only if BOTH hold:
- the question makes sense on its own and a real user could plausibly ask it;
- the answer is a specific fact, setting, name or behaviour from the documentation (not an opinion,
  not an attitude, not a random example value such as a hash or a sample name, not a vague word).
Reply with JSON only: {"ok": true} or {"ok": false}"""


def gateways(corpus: Path):
    """The 41 folders two levels down (plus one bucket per top folder for its loose pages)."""
    out = {}
    for top in ("concepts", "tasks", "tutorials", "setup"):
        for f in sorted((corpus / top).rglob("*.md")):
            rel = f.relative_to(corpus).as_posix()
            parts = rel.split("/")
            key = "/".join(parts[:2]) if len(parts) > 2 else top
            out.setdefault(key, []).append(rel)
    return out


def menu(gates):
    lines = []
    for i, (g, files) in enumerate(gates.items(), 1):
        names = sorted({Path(f).stem if Path(f).stem != "_index" else Path(f).parent.name for f in files})
        lines.append(f"[{i}] {g}: {', '.join(names)}")
    return ("Documentation folders, each with the names of the pages inside it:\n\n" + "\n".join(lines) +
            "\n\nThe user will ask a question. Pick the 3 folders most likely to contain the page that "
            "answers it, most likely first. Reply with JSON only: {\"folders\": [n1, n2, n3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--models", default="qwen2.5:7b,qwen2.5:3b")
    ap.add_argument("--test", action="store_true", help="also score the sealed test half")
    ap.add_argument("--short", type=int, default=50)
    args = ap.parse_args()
    corpus, cache_dir = Path(args.corpus), Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    say = lambda *a: print(*a, flush=True)  # noqa: E731

    sh = Shared(corpus)
    net = TokenNetRoute(sh).net
    W = net.W.tocsr()
    ids, texts = net.ids, net.texts
    files = np.array([d.split("#")[0] for d in ids])
    say(f"{len(ids)} passages, {len(set(files))} files")

    emb = OllamaEmbedder(cache_dir=str(ROOT / "realtest" / "data" / ".cache"))
    t0 = time.time()
    P = np.asarray(emb.embed(list(texts), "document"), np.float32)
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    emb.flush()
    say(f"passage vectors ready in {time.time() - t0:.0f}s (from the embed-all cache where present)")

    # how long one passage takes to embed on this machine, for the on-arrival cost
    probe = [f"probe {i} {t[:600]}" for i, t in enumerate(texts[:32])]
    t0 = time.time()
    emb._call(probe, "search_document: ")
    per_passage = (time.time() - t0) / len(probe)
    say(f"embedding cost on this machine: {per_passage * 1000:.0f} ms per passage (batch of 32)")

    gates = gateways(corpus)
    names = list(gates)
    gate_of = {f: g for g, fs in gates.items() for f in fs}
    file_gate = np.array([names.index(gate_of.get(f, f.split("/")[0])) for f in files])
    system = menu(gates)
    say(f"{len(gates)} gateways; menu is {len(system):,} characters")

    judge_cache = Cache(cache_dir / "judge.json")
    gate_cache = Cache(cache_dir / "gates.json")

    def qtok(q):
        v = np.zeros(len(net.vocab), np.float32)
        for t in tokens(q):
            if t in net.vocab:
                v[net.vocab[t]] = 1.0
        return v

    def rrf(*orders, c=60):
        s = {}
        for o in orders:
            for r, i in enumerate(o):
                s[i] = s.get(i, 0.0) + 1.0 / (c + r + 1)
        return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])]

    def pick(model, q):
        def make():
            t = time.time()
            raw = ollama(model, system, f"Question: {q}")
            try:
                said = json.loads(raw).get("folders", [])
            except Exception:
                said = re.findall(r"[\w/-]+", raw)
            got = []
            for x in said:  # the 3B model answers with numbers, the 7B with folder names
                x = str(x).strip().strip("[]")
                if x.isdigit() and 1 <= int(x) <= len(names):
                    g = int(x) - 1
                elif x in names:
                    g = names.index(x)
                else:
                    continue
                if g not in got:
                    got.append(g)
            return {"gates": got[:3], "raw": raw, "s": round(time.time() - t, 2)}
        return gate_cache.get(f"{model}|{q}", make)

    def methods(q, qv, models):
        """Each method returns (top-4 passage indexes, passages embedded on arrival, model seconds)."""
        tv = qtok(q)
        ts = W @ tv
        es = P @ qv
        t_order = np.argsort(-ts)
        e_order = np.argsort(-es)
        out = {}
        out["token-net"] = (t_order[:4], 0, 0.0)
        out["embed-all"] = (e_order[:4], 0, 0.0)
        out["token-net+embed-all"] = (rrf(t_order[:100], e_order[:100])[:4], 0, 0.0)
        sl = [i for i in t_order[:args.short] if ts[i] > 0]
        out[f"token-net@{args.short} > embed"] = (sorted(sl, key=lambda i: -es[i])[:4], len(sl), 0.0)
        out[f"token-net@{args.short} > rrf"] = (rrf(sl, sorted(sl, key=lambda i: -es[i]))[:4], len(sl), 0.0)
        for m in models:
            g = pick(m, q)
            inside = np.isin(file_gate, g["gates"])
            tm = np.where(inside, ts, -1.0)
            to = [i for i in np.argsort(-tm)[:args.short] if tm[i] > 0]
            out[f"gate:{m} > token-net"] = (to[:4], 0, g["s"])
            out[f"gate:{m} > token-net@{args.short} > rrf"] = (
                rrf(to, sorted(to, key=lambda i: -es[i]))[:4], len(to), g["s"])
            em = np.where(inside, es, -9.0)
            out[f"gate:{m} > embed inside"] = (np.argsort(-em)[:4], int(inside.sum()), g["s"])
        return out

    def run(label, items, models):
        say(f"\n=== {label}: {len(items)} questions ===")
        Q = np.asarray(emb.embed([it["q"] for it in items], "query"), np.float32)
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        emb.flush()
        score, cost, msec, gate_hit = {}, {}, {}, {m: 0 for m in models}
        for m in models:  # one model at a time, so Ollama keeps it loaded and reuses the menu
            t0 = time.time()
            for n, it in enumerate(items, 1):
                pick(m, it["q"])
                if n % 20 == 0 or n == len(items):
                    say(f"  {m} picked gateways for {n}/{len(items)}, {time.time() - t0:.0f}s")
        t0 = time.time()
        for n, (it, qv) in enumerate(zip(items, Q), 1):
            res = methods(it["q"], qv, models)
            gold = set(it["files"])
            expect = it.get("expect", [])  # regexes, all must match (as realtest.run.evidence)
            for m in models:
                gate_hit[m] += any(names.index(gate_of.get(f, f.split("/")[0])) in pick(m, it["q"])["gates"]
                                   for f in gold)
            for name, r in res.items():
                top, emb_n, ms = r
                f_ok = bool(gold & set(files[list(top)]))
                joined = "\n".join(texts[i] for i in top).lower()
                a_ok = bool(expect) and all(re.search(p, joined) for p in expect) and \
                    (not it.get("expect_any") or any(re.search(p, joined) for p in it["expect_any"]))
                s = score.setdefault(name, [0, 0])
                s[0] += f_ok
                s[1] += a_ok
                cost.setdefault(name, []).append(emb_n)
                msec.setdefault(name, []).append(ms)
            if n % 10 == 0 or n == len(items):
                say(f"  {n}/{len(items)} questions, {time.time() - t0:.0f}s")
        N = len(items)
        say(f"\n| Method | Right file in top 4 | Answer in passages | Embedded on arrival (mean) | Model s/question |")
        say("|---|---|---|---|---|")
        for name, (f, a) in score.items():
            c, s = np.mean(cost[name]), np.mean(msec[name])
            say(f"| {name} | {f}/{N} ({100 * f / N:.0f}%) | {a}/{N} ({100 * a / N:.0f}%) | "
                f"{c:.0f} (~{c * per_passage:.1f}s) | {s:.1f} |")
        for m in models:
            say(f"gateway {m}: right folder among its 3 picks for {gate_hit[m]}/{N} ({100 * gate_hit[m] / N:.0f}%)")
        return score

    qdir = ROOT / "realtest" / "questions"
    dev = json.load(open(qdir / "k8s-plain-dev.json", encoding="utf8"))
    hand = [x for x in json.load(open(qdir / "k8s.json", encoding="utf8"))
            if any(f.split("/")[0] in ("concepts", "tasks", "tutorials", "setup") for f in x["files"])]
    hand = [dict(x, files=[f for f in x["files"] if f.split("/")[0] in ("concepts", "tasks", "tutorials", "setup")])
            for x in hand]
    for x in hand:
        x.setdefault("q", x.get("question"))
    models = [m for m in args.models.split(",") if m]

    say("judging the dev questions (the judge sees only question + answer, never a retrieval result)")
    t0 = time.time()
    for n, it in enumerate(dev, 1):
        judge_cache.get(f"qwen2.5:7b|{it['q']}", lambda it=it: json.loads(
            ollama("qwen2.5:7b", JUDGE, f"Question: {it['q']}\nAnswer: {it['answer']}")).get("ok") is True)
        if n % 20 == 0:
            say(f"  judged {n}/{len(dev)}, {time.time() - t0:.0f}s")
    clean = [it for it in dev if judge_cache.d[f"qwen2.5:7b|{it['q']}"]]
    say(f"judge kept {len(clean)} of {len(dev)} dev questions")

    run("hand-written (burned for tuning; for reference)", hand, models)
    run("plain dev, all", dev, models)
    run("plain dev, judged sensible", clean, models)
    if args.test:
        test = json.load(open(qdir / "k8s-plain-test.json", encoding="utf8"))
        for it in test:
            judge_cache.get(f"qwen2.5:7b|{it['q']}", lambda it=it: json.loads(
                ollama("qwen2.5:7b", JUDGE, f"Question: {it['q']}\nAnswer: {it['answer']}")).get("ok") is True)
        run("plain TEST, all (sealed until now)", test, models)
        run("plain TEST, judged sensible", [it for it in test if judge_cache.d[f"qwen2.5:7b|{it['q']}"]], models)


if __name__ == "__main__":
    main()
