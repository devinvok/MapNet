"""
The token network with positions: where a token sits, and which tokens sit next to it.

tokennet.py stores, for every passage, which tokens it contains. This keeps the order too, which
adds two things, both model-free:

  proximity   a passage where the question's words sit close together beats one where they are
              scattered through it. The top candidates from the word score are re-ranked by how
              tightly the question's tokens cluster (smallest window holding all of them).

  neighbours  for each question token, the tokens that sit right beside it across the corpus
              (within `window` tokens, scored by pointwise mutual information: how much more
              often than chance). They light up too, dimmer, so a question about "kill" also
              reaches pages that talk about "grace" and "period" next to it. Narrower than the
              passage-level associations in tokennet.py, which count anything in the same chunk.

The neighbour table has one row per distinct token and keeps only each token's strongest
neighbours, so it grows with the vocabulary (slowly) and not with the amount of text. Postings
(token -> passages) are what grow with the corpus.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse

from .tokennet import TokenNet, tokens


class PosTokenNet(TokenNet):
    def __init__(self, window=5, n_neighbours=8, expand_weight=0.3, min_pair=2, prox_weight=0.5,
                 prox_scale=5.0, pool=50, **kw):
        super().__init__(**kw)
        self.window, self.n_neighbours, self.expand_weight = window, n_neighbours, expand_weight
        self.min_pair, self.prox_weight, self.prox_scale, self.pool = min_pair, prox_weight, prox_scale, pool
        self._seqs: list[list[str]] = []
        self._ids: list[np.ndarray] = []
        self._C = None
        self._uni = None
        self._T = 1.0
        self._inv: list[str] = []
        self.window_seconds = 0.0
        self.last_expansion: dict = {}

    # --------------------------------------------------------------- building
    def add(self, doc_id, text, heading="", path=""):
        seq = tokens(text)
        new = doc_id not in self._index
        super().add(doc_id, text, heading=heading, path=path)
        i = self._index[doc_id]
        if new:
            self._seqs.append(seq)
        else:
            self._seqs[i] = seq

    def remove_missing(self, keep):
        if all(d in keep for d in self.ids):
            return
        seqs = [s for d, s in zip(self.ids, self._seqs) if d in keep]
        super().remove_missing(keep)
        self._seqs = seqs

    def _build(self):
        super()._build()
        t0 = time.perf_counter()
        v = self.vocab
        self._inv = [None] * len(v)
        for t, i in v.items():
            self._inv[i] = t
        self._ids = [np.fromiter((v[t] for t in s if t in v), dtype=np.int32) for s in self._seqs]
        rows, cols = [], []
        for a in self._ids:
            for d in range(1, self.window + 1):
                if len(a) > d:
                    rows.append(a[:-d])
                    cols.append(a[d:])
        V = len(v)
        if rows:
            r, c = np.concatenate(rows), np.concatenate(cols)
            keep = r != c
            C = sparse.coo_matrix((np.ones(int(keep.sum()), dtype=np.float32), (r[keep], c[keep])),
                                  shape=(V, V)).tocsr()
            C = (C + C.T).tocsr()
        else:
            C = sparse.csr_matrix((V, V), dtype=np.float32)
        uni = np.bincount(np.concatenate(self._ids), minlength=V).astype(np.float32) if self._ids else np.zeros(V)
        self._C, self._uni, self._T = C, uni, max(float(C.sum()), 1.0)
        self.window_seconds += time.perf_counter() - t0
        self.seconds += time.perf_counter() - t0

    # -------------------------------------------------------------- neighbours
    def window_neighbours(self, token, n=None):
        """Tokens that sit beside `token` more often than chance: [(token, strength 0..1)]."""
        if self._dirty:
            self._build()
        j = self.vocab.get(token)
        if j is None:
            return []
        row = self._C.getrow(j)
        cols, counts = row.indices, row.data
        if len(cols) == 0:
            return []
        N = self.B.shape[0]
        ok = (counts >= self.min_pair) & (self.df[cols] >= 2) & (self.df[cols] <= self.max_common * N)
        cols, counts = cols[ok], counts[ok]
        if len(cols) == 0:
            return []
        pmi = np.log(counts * self._T / (self._uni[j] * self._uni[cols] + 1e-9))
        score = pmi * np.log1p(counts)
        order = np.argsort(-score)[: (n or self.n_neighbours)]
        out = []
        top = float(score[order[0]]) if len(order) else 0.0
        for o in order:
            if score[o] <= 0 or top <= 0:
                continue
            other = self._inv[int(cols[o])]
            if len(other) >= 3 and len(token) >= 3 and (other in token or token in other):
                continue  # parts of the same identifier (maxsurge / surge) are not news
            out.append((other, float(score[o] / top)))
        return out

    def _neighbours(self, token, n=None):
        """Which words light up beside a question word. Here: the words that sit next to it."""
        return self.window_neighbours(token, n)

    def _expand(self, base, converge=1, hops=1, link_n=None):
        """Words that light up beyond the question's own, as {word: brightness}.

        Every question word reaches its neighbours with a strength. A word's brightness is the SUM
        of what reaches it, so a word that two question words both point to is brighter than a
        word one of them points to. With converge=2 only words reached by at least two different
        question words light up at all: a neighbour of just one word is as likely noise as topic,
        while a word several of the question's words agree on is the topic they share.
        With hops=2 the network is walked one step further: a neighbour's neighbours are reached
        too, dimmer, so two question words can meet through a word neither of them is linked to
        directly."""
        N = self.B.shape[0]
        sources = [t for t in base if self.df[self.vocab[t]] <= self.max_common * N]
        reach: dict[str, dict[str, float]] = {}      # word -> {question word: strength}
        first: dict[str, dict[str, float]] = {}
        for t in sources:
            for nb, s in self._neighbours(t, link_n):
                if nb in base:
                    continue
                first.setdefault(nb, {})[t] = max(s, first.get(nb, {}).get(t, 0.0))
        for w, d in first.items():
            reach[w] = dict(d)
        if hops >= 2:
            for mid, d in list(first.items()):
                for nb, s in self._neighbours(mid, link_n):
                    if nb in base or nb in first:
                        continue
                    for t, s0 in d.items():
                        v = 0.5 * s0 * s          # a second step is dimmer than a first
                        slot = reach.setdefault(nb, {})
                        if v > slot.get(t, 0.0):
                            slot[t] = v
        out = {}
        for w, d in reach.items():
            if len(d) < converge:
                continue
            out[w] = min(1.0, self.expand_weight * sum(d.values()))
            self.last_expansion.setdefault(w, []).append(sorted(d))
        return out

    # --------------------------------------------------------------- proximity
    def _proximity(self, i, base_ids):
        """How tightly the question's tokens cluster in passage i: 0 (scattered or fewer than two
        of them present) up to the summed rarity of the tokens that are present."""
        arr = self._ids[i]
        events = []
        for k, tid in enumerate(base_ids):
            for p in np.nonzero(arr == tid)[0]:
                events.append((int(p), k))
        present = {k for _, k in events}
        m = len(present)
        if m < 2:
            return 0.0
        events.sort()
        need, have, left, best = m, {}, 0, None
        for right, (pos, k) in enumerate(events):
            have[k] = have.get(k, 0) + 1
            if have[k] == 1:
                need -= 1
            while need == 0:
                span = pos - events[left][0] + 1
                best = span if best is None else min(best, span)
                lk = events[left][1]
                have[lk] -= 1
                if have[lk] == 0:
                    need += 1
                left += 1
        gap = max(best - m, 0)
        rare = sum(float(self.idf[base_ids[k]]) for k in present)
        return rare * (m / len(base_ids)) / (1.0 + gap / self.prox_scale)

    # --------------------------------------------------------------- searching
    def search(self, query, k=4, prox=True, expand=True, assoc=False, extra_terms="", converge=1, hops=1,
               link_n=None):
        if self._dirty:
            self._build()
        base = list(dict.fromkeys(t for t in tokens(query) if t in self.vocab))
        if not base:
            return [], 0
        act = {t: 1.0 for t in base}
        self.last_expansion = {}
        if expand:
            for t, w in self._expand(base, converge, hops, link_n).items():
                if w > act.get(t, 0.0):
                    act[t] = w
        q = sparse.csr_matrix((list(act.values()), ([0] * len(act), [self.vocab[t] for t in act])),
                              shape=(1, len(self.vocab)), dtype=np.float32)
        scores = np.asarray((self.W @ q.T).todense()).ravel()
        reached = int((scores > 0).sum())
        cand = [int(i) for i in np.argsort(-scores)[: max(self.pool, k)] if scores[i] > 0]
        if prox and len(base) > 1:
            base_ids = [self.vocab[t] for t in base]
            for i in cand:
                scores[i] += self.prox_weight * self._proximity(i, base_ids)
            cand.sort(key=lambda i: -scores[i])
        return [(self.ids[i], self.texts[i], float(scores[i])) for i in cand[:k]], reached
