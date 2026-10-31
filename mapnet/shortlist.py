"""
Words find the neighbourhood, meaning picks the page.

The token network (tokennet.py) is fast and live but ranks by shared words, so a chatty passage that
repeats the question's filler words ("want", "make", "need") beats the page that answers it. Its top
50 usually hold the right page (70% of everyday-language questions, against 38% in its top 4).

So: take the token network's shortlist, embed only those passages (on arrival, the first time a
question reaches them; cached by their text, so an edited passage is embedded again), and order the
shortlist by meaning. Nothing is embedded up front, and a passage no question reaches is never
embedded. With `roles`, the shortlist holds only passages those roles may open (the token network
scores nothing else), so a restricted passage is never embedded for, or shown to, a caller.
mode="embed" orders by meaning alone; mode="rrf" merges the word rank and the meaning rank.

Measured on 220 everyday-language Kubernetes questions (research/gateway_experiment.py): right file
in the top 4 for 48%, against 49% for embedding every passage up front and 38% for the token network
alone; 50 passages embedded per question at most (about 5 s on a CPU laptop the first time).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class ShortlistRerank:
    def __init__(self, net, embedder, short=50, mode="embed", c=60):
        self.net, self.emb, self.short, self.mode, self.c = net, embedder, short, mode, c
        self._vec: dict[str, np.ndarray] = {}   # text hash -> unit vector
        self.embedded = 0                        # passages embedded on arrival, in total
        self.embed_s = 0.0

    def _vectors(self, texts):
        todo = [t for t in dict.fromkeys(texts) if _key(t) not in self._vec]
        if todo:
            t0 = time.perf_counter()
            V = np.asarray(self.emb.embed(todo, "document"), dtype=np.float32)
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
            for t, v in zip(todo, V):
                self._vec[_key(t)] = v
            self.embedded += len(todo)
            self.embed_s += time.perf_counter() - t0
        return np.stack([self._vec[_key(t)] for t in texts])

    def search(self, query: str, k: int = 4, roles=None):
        """Top-k (doc_id, text, score), the number of passages the words reached, and how many
        passages had to be embedded for this question."""
        before = self.embedded
        hits, reached = self.net.search(query, k=self.short, roles=roles)
        if not hits:
            return [], reached, 0
        q = np.asarray(self.emb.embed([query], "query"), dtype=np.float32)[0]
        q /= max(float(np.linalg.norm(q)), 1e-9)
        sims = self._vectors([t for _, t, _ in hits]) @ q
        by_meaning = list(np.argsort(-sims))
        if self.mode == "rrf":
            score = {i: 1.0 / (self.c + r + 1) for r, i in enumerate(range(len(hits)))}
            for r, i in enumerate(by_meaning):
                score[i] += 1.0 / (self.c + r + 1)
            order = sorted(score, key=lambda i: -score[i])
            out = [(hits[i][0], hits[i][1], score[i]) for i in order[:k]]
        else:
            out = [(hits[i][0], hits[i][1], float(sims[i])) for i in by_meaning[:k]]
        return out, reached, self.embedded - before
