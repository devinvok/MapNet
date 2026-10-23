"""
Generate extra test questions from random passages with your local model.

Caveat built into the report: generated questions tend to reuse the passage's
wording, which favours keyword search. The hand-written set is the fairer test;
generated questions add volume and coverage of files nobody thought to ask about.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request

PROMPT = """Here is a passage from a documentation file:

\"\"\"
{text}
\"\"\"

Write ONE question that a user of this software might genuinely ask, which this
passage answers with a specific fact. Rules:
- Phrase it the way a person would ask, in your own words. Do not copy phrases
  from the passage, and do not mention "the passage" or "the document".
- It must make sense on its own, without seeing the passage.
- Give the short factual answer.

Reply with JSON only: {{"question": "...", "answer": "..."}}"""


def generate(shared, reader, n, seed=0, log=print):
    rng = random.Random(seed)
    pool = [(d, t) for d, t in shared.chunks() if len(t) > 400]
    rng.shuffle(pool)
    out, tried = [], 0
    for doc_id, text in pool:
        if len(out) >= n or tried >= n * 3:
            break
        tried += 1
        body = json.dumps({"model": reader.model, "stream": False, "format": "json",
                           "options": {"temperature": 0.3},
                           "prompt": PROMPT.format(text=text[:1500])}).encode()
        req = urllib.request.Request(f"{reader.host}/api/generate", body,
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                qa = json.loads(json.load(r)["response"])
            q, a = qa["question"].strip(), str(qa["answer"]).strip()
        except Exception:
            continue
        if len(q) < 15 or not a or re.search(r"\b(passage|document|text above)\b", q, re.I):
            continue
        out.append({"q": q, "answer": a, "files": [doc_id.split("#")[0]], "generated": True})
        log(f"    [{len(out)}/{n}] {q}")
    return out
