"""
A network of tokens that point at the passages containing them.

Two kinds of edges, no model and no embedding of passages:

  token -> passage   a posting: this word is in that passage, with a weight (BM25: rare words weigh
                     more, which is the DNS idea again: a rare token is a specific name, a common
                     token is a generic one).
  token <-> token    an association: two words that keep turning up in the same passages, so
                     "terminating" is linked to "finalizer" because the pages about one talk about
                     the other. Worked out from the documents themselves when a word is looked up,
                     so nothing is precomputed and a new file changes the links the moment it is added.

A question is split into tokens. Each token lights up, and so do its strongest associates (less
brightly). The light flows along the postings to the passages, and the brightest passages win.
This is spreading activation over a bipartite graph, the classic inverted index with query
expansion done by the corpus itself.

Tokens here are whole words and identifiers (maxSurge also gives max, surge), not model sub-word
pieces. Words are crudely stemmed so "terminating" and "terminated" meet.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter

import numpy as np
from scipy import sparse

STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been being
it its this that these those there here which who whom whose what when where why how do does did done doing
can could should would will shall may might must not no nor so than too very just also into onto over under
about above below between through during before after again further once only own same such some any each
both few more most other i you he she we they me him her us them my your his our their am have has had having
use used using uses see example note etc via per""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def stem(w: str) -> str:
    n = len(w)
    if n <= 3:
        return w
    if w.endswith("ies") and n > 4:
        return w[:-3] + "y"
    if w.endswith("ation") and n > 7:
        return w[:-5]
    if w.endswith("ing") and n > 5:
        return w[:-3]
    if w.endswith("ed") and n > 4:
        return w[:-2]
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("es") and n > 4 and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and n > 3:
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    """Words and identifiers as stemmed lower-case tokens. maxSurge -> maxsurge, max, surge."""
    out = []
    for raw in _WORD.findall(text):
        every = [p.lower() for p in _PART.findall(raw.replace("-", " ").replace("_", " "))]
        if len(every) > 1:  # the identifier as written (ReadWriteOncePod), stopword parts included
            out.append(stem("".join(every)))
        out += [stem(p) for p in every if len(p) > 1 and p not in STOP]
    return out


class TokenNet:
    def __init__(self, k1=1.2, b=0.75, heading_weight=2, assoc_n=5, assoc_weight=0.3,
                 max_common=0.05, min_support=2):
        self.k1, self.b = k1, b
        self.heading_weight = heading_weight   # headings and the path count this many times
        self.assoc_n, self.assoc_weight = assoc_n, assoc_weight
        self.max_common = max_common           # a token in more than this share of passages is too generic to associate with
        self.min_support = min_support
        self.ids: list[str] = []
        self.texts: list[str] = []
        self._bags: list[Counter] = []
        self._index: dict[str, int] = {}
        self.vocab: dict[str, int] = {}
        self._dirty = True
        self.W = self.B = None
        self.seconds = 0.0

    # --------------------------------------------------------------- building
    def add(self, doc_id: str, text: str, heading: str = "", path: str = ""):
        """Add (or replace) one passage. The posting weights are refreshed on the next search."""
        bag = Counter(tokens(text))
        for t in tokens(heading) + tokens(path):
            bag[t] += self.heading_weight
        if doc_id in self._index:
            i = self._index[doc_id]
            self._bags[i], self.texts[i] = bag, text
        else:
            self._index[doc_id] = len(self.ids)
            self.ids.append(doc_id)
            self.texts.append(text)
            self._bags.append(bag)
        self._dirty = True

    def remove_missing(self, keep: set):
        if all(d in keep for d in self.ids):
            return
        pairs = [(d, t, b) for d, t, b in zip(self.ids, self.texts, self._bags) if d in keep]
        self.ids, self.texts, self._bags = [list(x) for x in zip(*pairs)] if pairs else ([], [], [])
        self._index = {d: i for i, d in enumerate(self.ids)}
        self._dirty = True

    def _build(self):
        t0 = time.perf_counter()
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, bag in enumerate(self._bags):
            for t, c in bag.items():
                rows.append(i)
                cols.append(vocab.setdefault(t, len(vocab)))
                vals.append(c)
        n, v = len(self._bags), len(vocab)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(n, v), dtype=np.float32)
        length = np.asarray(tf.sum(axis=1)).ravel()
        avg = max(float(length.mean()), 1.0) if n else 1.0
        self.B = (tf > 0).astype(np.float32).tocsr()           # passage x token, present or not
        df = np.asarray(self.B.sum(axis=0)).ravel()
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        tf = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * length[tf.row] / avg)
        w = idf[tf.col] * tf.data * (self.k1 + 1) / (tf.data + norm)
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=(n, v), dtype=np.float32)
        self.Bc = self.B.tocsc()
        self.vocab, self.df, self.idf = vocab, df, idf
        self._dirty = False
        self.seconds += time.perf_counter() - t0

    # ------------------------------------------------------------ associations
    def neighbours(self, token: str, n: int | None = None):
        """The tokens that turn up in the same passages as `token` far more than chance would
        explain: [(token, strength 0..1)]. Worked out on demand from the current documents."""
        if self._dirty:
            self._build()
        j = self.vocab.get(token)
        if j is None:
            return []
        rows = self.Bc[:, j].nonzero()[0]
        if len(rows) < self.min_support:
            return []
        counts = np.asarray(self.B[rows].sum(axis=0)).ravel()
        N = self.B.shape[0]
        ok = (counts >= self.min_support) & (self.df >= self.min_support) & (self.df <= self.max_common * N)
        ok[j] = False
        if not ok.any():
            return []
        idx = np.nonzero(ok)[0]
        lift = (counts[idx] / len(rows)) / (self.df[idx] / N)
        score = np.log(lift) * np.log1p(counts[idx])      # surprising AND seen more than once
        order = np.argsort(-score)[: (n or self.assoc_n)]
        top = score[order]
        keep = top > 0
        if not keep.any():
            return []
        inv = {i: t for t, i in self.vocab.items()} if not hasattr(self, "_inv") or len(self._inv) != len(self.vocab) else self._inv
        self._inv = inv
        top = top / top.max()
        return [(inv[int(idx[o])], float(s)) for o, s, k in zip(order, top, keep) if k]

    # --------------------------------------------------------------- searching
    def search(self, query: str, k: int = 4, assoc: bool = False, extra_terms: str = ""):
        """Top-k passages as (doc_id, text, score), and how many passages the light reached."""
        if self._dirty:
            self._build()
        act: dict[str, float] = {}
        for t in tokens(query):
            if t in self.vocab:
                act[t] = 1.0
        for t in tokens(extra_terms):
            if t in self.vocab:
                act.setdefault(t, 0.8)
        if assoc:
            for t in list(act):
                if act[t] < 1.0:
                    continue
                for nb, s in self.neighbours(t):
                    act[nb] = max(act.get(nb, 0.0), self.assoc_weight * s)
        if not act:
            return [], 0
        q = sparse.csr_matrix((list(act.values()), ([0] * len(act), [self.vocab[t] for t in act])),
                              shape=(1, len(self.vocab)), dtype=np.float32)
        scores = np.asarray((self.W @ q.T).todense()).ravel()
        reached = int((scores > 0).sum())
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], self.texts[i], float(scores[i])) for i in top if scores[i] > 0], reached
