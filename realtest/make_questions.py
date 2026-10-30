"""
A larger question set written the way a newcomer would ask, for testing vocabulary gaps.

The 40 hand-written questions are mostly single lookups that share words with their page. These
are the opposite on purpose: the local model reads one passage and writes a question in everyday
words, without the passage's technical terms, usually combining two ideas. Then the questions are
filtered so the test is honest:

  - the question shares at most one rare word with its passage (so keyword search can't just match)
  - the answer's key word (copied from the passage) must be rare in the corpus (so "the retrieved
    passages contain the answer" means something) and must not appear in the question
  - one question per file, sampled across the folders

The file a question came from is its right answer, but other files may answer it too, so
"right file" understates every method equally. "Answer in passages" (the key word) is the fairer number.

Files are split into a dev half (tune on it) and a test half (look once, at the end), by a hash of the
path, so no file's questions are in both.

    python -m realtest.make_questions --n 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mapnet.tokennet import TokenNet, tokens  # noqa: E402
from realtest.retrievers import Shared  # noqa: E402

PROMPT = """Here is a passage from the Kubernetes documentation:

\"\"\"
{text}
\"\"\"

Write ONE question that a newcomer who does not know Kubernetes terminology would ask, and whose
correct answer is the term "{key}" (the answer may explain it). Rules:
- Use everyday words. Do NOT use any of these words: {avoid}.
- If the passage allows it, make the question involve two related ideas (a situation and what happens,
  or a goal and a limit).
- It must make sense on its own. Do not mention "the passage" or "the document".
- "answer": one sentence that contains "{key}".

Reply with JSON only: {{"question": "...", "answer": "..."}}"""

WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")
STOPISH = {"this", "that", "with", "from", "have", "will", "then", "than", "also", "such", "when", "which",
           "kubernetes", "cluster", "should", "would", "could", "more", "most", "into", "your", "each"}


def corpus_words(shared):
    df = {}
    for _, t in shared.chunks():
        for w in set(x.lower() for x in WORD.findall(t)):
            df[w] = df.get(w, 0) + 1
    return df


def avoid_list(text, df, n=12):
    ws = {w.lower() for w in WORD.findall(text)}
    return sorted(ws, key=lambda w: (df.get(w, 1), w))[:n]


def candidate_keys(text, net, rng, lo=1, hi=6):
    """Rare words of the passage that could be the answer: identifiers first (camelCase, digits,
    dashes), then other rare words. Rare = found in only a few passages of the whole corpus."""
    out = []
    for w in dict.fromkeys(WORD.findall(text)):
        n = sum(1 for _, t in net._corpus if w.lower() in t.lower())
        if lo <= n <= hi and w.lower() not in STOPISH and len(re.findall(r"[-_]", w)) < 2:
            ident = bool(re.search(r"[a-z][A-Z]|[0-9_\-]", w)) or w[0].isupper()
            out.append((0 if ident else 1, rng.random(), w))
    out.sort()
    return [w for _, _, w in out[:3]]


def ask(host, model, prompt, timeout=240):
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "options": {"temperature": 0.4}, "prompt": prompt}).encode()
    req = urllib.request.Request(f"{host}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(json.load(r)["response"])


def accept(qa, key, text, net, n_pass):
    q, ans = str(qa.get("question", "")).strip(), str(qa.get("answer", "")).strip()
    if len(q) < 25 or not ans or re.search(r"(passage|document|text above)", q, re.I):
        return None, "unusable"
    if key.lower() in q.lower():
        return None, "key in question"
    if key.lower() not in ans.lower():
        return None, "answer lacks the key"
    ptoks = set(tokens(text))
    rare = [t for t in set(tokens(q)) if t in net.vocab and net.df[net.vocab[t]] <= 0.01 * n_pass]
    if len([t for t in rare if t in ptoks]) > 1:
        return None, "shares rare words with the passage"
    return dict(q=q, answer=ans, key=key), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="accepted questions to collect")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--work", default=str(HERE / "data" / "work" / "k8s-docs"))
    ap.add_argument("--only", default="concepts,tasks,tutorials,setup")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(HERE / "questions"))
    ap.add_argument("--max-tries", type=int, default=0)
    a = ap.parse_args()

    shared = Shared(Path(a.work))
    keep = set(a.only.split(","))
    df = corpus_words(shared)
    net = TokenNet()
    net._corpus = [(d, t) for d, t in shared.chunks()]
    for d, t in net._corpus:
        net.add(d, t)
    net._build()
    n_pass = len(net.ids)

    by_file = {}
    for d, t in net._corpus:
        f = d.split("#")[0]
        if f.split("/")[0] in keep and 450 <= len(t) <= 1500 and "\n" in t:
            by_file.setdefault(f, []).append((d, t))
    files = sorted(by_file)
    random.Random(a.seed).shuffle(files)
    print(f"{len(files)} files with a usable passage; collecting {a.n} questions with {a.model}", flush=True)

    ckpt = Path(a.out) / "k8s-plain.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            done[x["files"][0]] = x
    rejects = {}
    tries = 0
    limit = a.max_tries or a.n * 3
    for f in files:
        if len(done) >= a.n or tries >= limit:
            break
        if f in done:
            continue
        rng = random.Random(f)
        d, text = rng.choice(by_file[f])
        keys = candidate_keys(text, net, rng)
        if not keys:
            rejects["no rare word"] = rejects.get("no rare word", 0) + 1
            continue
        item = None
        for key in keys:
            tries += 1
            avoid = ", ".join(dict.fromkeys(avoid_list(text, df) + [key.lower()]))
            try:
                qa = ask(a.host, a.model, PROMPT.format(text=text[:1500], key=key, avoid=avoid))
            except Exception:
                rejects["model error"] = rejects.get("model error", 0) + 1
                continue
            item, why = accept(qa, key, text, net, n_pass)
            if item is not None:
                break
            rejects[why] = rejects.get(why, 0) + 1
        if item is None:
            continue
        item.update(files=[f], passage=d, expect=[re.escape(item["key"].lower())], generated=True, plain=True)
        done[f] = item
        with open(ckpt, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item) + chr(10))
        print(f"[{len(done)}/{a.n}] {item['q']}  (key: {item['key']})", flush=True)
    print("rejected:", rejects, flush=True)

    items = list(done.values())
    split = lambda x: int(hashlib.md5(x["files"][0].encode()).hexdigest(), 16) % 2
    for name, part in (("dev", [x for x in items if split(x) == 0]), ("test", [x for x in items if split(x) == 1])):
        p = Path(a.out) / f"k8s-plain-{name}.json"
        p.write_text(json.dumps(part, indent=1), encoding="utf-8")
        print(f"{name}: {len(part)} questions -> {p}", flush=True)


if __name__ == "__main__":
    main()
