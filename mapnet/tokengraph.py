"""
Tokens linked by meaning, not by position.

tokenwin.py lit up the words that sit beside a question word. That finds how text is written
(collocations: "cap" beside "sys"), not what it is about. This links words by the CONTEXTS they
are used in: a word's vector is the average of the vectors of the passages that contain it
(weighted by how much the word matters in each: the BM25 weights the token network already has).
Two words used in the same kinds of passages end up close, even if they never occur in the same
sentence, so "kill" can reach "terminat" and "grace" without a model reading the question.

The question's own words light up at 1.0; each lights its nearest words by meaning, dimmer; the
light flows along the postings to the passages, exactly as in the token network.

No new embedding work: the passage vectors are the ones the embedding index already holds, passed
in as `vectors(doc_ids) -> matrix`. Links are looked up on demand (a matrix-vector product over the
vocabulary), so adding a file only changes the vectors of the words it contains.
"""
from __future__ import annotations

import numpy as np

from .tokenwin import PosTokenNet


class TokenGraph(PosTokenNet):
    def __init__(self, vectors, n_neighbours=5, min_sim=0.0, expand_weight=0.3, specific=0.3, **kw):
        super().__init__(n_neighbours=n_neighbours, expand_weight=expand_weight, **kw)
        self.vectors, self.min_sim = vectors, min_sim
        self.specific = specific  # words in the vaguest share of contexts neither expand nor are expanded into
        self._T = None
        self._ok = None

    def _build(self):
        super()._build()
        self._T = self._ok = None

    def _token_vectors(self):
        """One unit vector per token: the weighted mean of its passages' vectors, with the common
        direction (what every passage shares) removed so that similarity means something."""
        if self._T is None:
            P = np.asarray(self.vectors(self.ids), dtype=np.float32)
            P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
            T = np.asarray(self.W.T.dot(P), dtype=np.float32)          # tokens x dim
            wsum = np.asarray(self.W.sum(axis=0)).ravel()
            spec = np.linalg.norm(T, axis=1) / np.maximum(wsum, 1e-9)   # 1 = one consistent context, low = used everywhere
            seen = self.df >= 2
            cut = np.quantile(spec[seen], self.specific) if seen.any() else 0.0
            self._ok = seen & (spec >= cut - 1e-6)
            T[~seen] = 0.0
            T -= T[seen].mean(axis=0, keepdims=True) * seen[:, None]
            self._T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        return self._T

    def _neighbours(self, token, n=None):
        if self._dirty:
            self._build()
        j = self.vocab.get(token)
        if j is None or self.df[j] < 2:
            return []
        T = self._token_vectors()
        if not self._ok[j]:
            return []
        sims = T @ T[j]
        N = self.B.shape[0]
        sims[~self._ok] = -1.0
        sims[self.df > self.max_common * N] = -1.0   # words used everywhere link to everything
        sims[j] = -1.0
        out = []
        n = n or self.n_neighbours
        for o in np.argsort(-sims)[: n * 2]:
            if sims[o] < self.min_sim:
                break
            other = self._inv[int(o)]
            if len(other) >= 3 and len(token) >= 3 and (other in token or token in other):
                continue  # parts of the same identifier are not news
            out.append((other, float(sims[o])))
            if len(out) >= n:
                break
        return out
