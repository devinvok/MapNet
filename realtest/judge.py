"""Was the answer right?

- keyword: deterministic. Each entry in `expect` must match (regex alternatives
  with |, case-insensitive); at least one entry of `expect_any` must match.
  Used for the hand-written question set.
- model: asks a model whether the candidate states the reference answer's key
  fact. Used for generated questions, which have a reference answer, not keywords.
"""
from __future__ import annotations

import json
import re
import urllib.request


def keyword_judge(answer: str, item: dict) -> bool:
    a = answer.lower()
    if re.search(r"could not find|couldn't find|not (?:in|found in) the (?:files|excerpts)", a):
        return False
    ok = all(re.search(p, a) for p in item.get("expect", []))
    if item.get("expect_any"):
        ok = ok and any(re.search(p, a) for p in item["expect_any"])
    return ok


JUDGE_PROMPT = """Question: {q}
Reference answer: {ref}
Candidate answer: {cand}

Does the candidate answer state the key fact of the reference answer (it may add
more detail)? Reply with exactly one word: yes or no."""


class ModelJudge:
    def __init__(self, reader):
        self.reader = reader  # an OllamaReader or AnthropicReader

    def __call__(self, answer, item):
        prompt = JUDGE_PROMPT.format(q=item["q"], ref=item["answer"], cand=answer)
        if hasattr(self.reader, "host"):  # Ollama: plain generate, no system prompt
            body = json.dumps({"model": self.reader.model, "prompt": prompt, "stream": False,
                               "options": {"temperature": 0}}).encode()
            req = urllib.request.Request(f"{self.reader.host}/api/generate", body,
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)["response"]
        else:
            out = self.reader.answer(item["q"], prompt=prompt).text
        return out.strip().lower().startswith("yes")


def judge(answer, item, model_judge=None):
    if item.get("expect"):
        return keyword_judge(answer, item)
    if model_judge and item.get("answer"):
        return model_judge(answer, item)
    return None  # can't judge
