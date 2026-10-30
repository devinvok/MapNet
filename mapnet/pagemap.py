"""
Page vectors: each file is represented by what it actually says, not by a description.

The file-description map (refmap.py) asks a model to write what each file is about and
searches those short texts. They miss the specific fact a question is about. This map
represents a file by ONE vector made from its own text, finds the best few files for a
question by that vector, and only then ranks the passages inside them.

Two ways to make the page vector:
  "mean"  the mean of the vectors of the file's passages. Every passage is embedded when the
          file is registered, so build cost equals embedding search; a question then scores
          one vector per file and reads the passage vectors of the few files it picked.
  "text"  one embedding of the page's text (its start, up to `max_chars`). Passages are
          embedded only when a question first reaches the file, as in refmap and embed-tree.
          Cheaper to build, but a long page is represented by its beginning.

Adding a file, changing it and removing it touch only that file's vector. A caller only ever
scores the pages of files it may open.
"""
from __future__ import annotations

import time

import numpy as np

from .core import Hit, Result


def _unit_rows(V):
    V = np.asarray(V, dtype=np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


class PageMap:
    def __init__(self, mapnet, embedder, mode="mean", file_beam=6, max_chars=6000, batch_files=40):
        if mode not in ("mean", "text"):
            raise ValueError("mode must be 'mean' or 'text'")
        self.m, self.emb, self.mode = mapnet, embedder, mode
        self.file_beam, self.max_chars, self.batch_files = file_beam, max_chars, batch_files
        self.page: dict[str, np.ndarray] = {}      # path -> page vector
        self._sig: dict[str, tuple] = {}           # path -> signature of the text it was built from
        self._passages: dict[str, tuple] = {}      # path -> (signature, passage vectors)
        self.files_embedded = 0
        self.passages_embedded = 0
        self.embed_seconds = 0.0
        self._rows = None                          # cached matrix of page vectors, in `_paths` order
        self._paths: list[str] = []

    # ------------------------------------------------------------ embedding
    def _embed(self, texts, kind="document"):
        t0 = time.perf_counter()
        V = _unit_rows(self.emb.embed(list(texts), kind))
        self.embed_seconds += time.perf_counter() - t0
        return V

    @staticmethod
    def _signature(src):
        return (len(src.doc_ids), hash(tuple(src.texts)))

    def _page_text(self, src):
        rel = src.path[len("_top/"):] if src.path.startswith("_top/") else src.path
        return (rel + "\n" + "\n".join(src.texts))[:self.max_chars]

    # ------------------------------------------------------------- building
    def sync(self):
        """Register new and changed files, forget removed ones. Returns how many were (re)built."""
        srcs = {s.path: s for s in self.m.sources()}
        gone = [p for p in self.page if p not in srcs]
        for p in gone:
            self.page.pop(p), self._sig.pop(p, None), self._passages.pop(p, None)
        todo = [s for s in srcs.values() if self._sig.get(s.path) != self._signature(s)]
        for a in range(0, len(todo), self.batch_files):
            part = todo[a:a + self.batch_files]
            if self.mode == "mean":
                V = self._embed([t for s in part for t in s.texts])
                i = 0
                for s in part:
                    n = len(s.texts)
                    P = V[i:i + n]
                    i += n
                    self._passages[s.path] = (self._signature(s), P)
                    self.page[s.path] = _unit_rows(P.mean(axis=0, keepdims=True))[0]
                    self.files_embedded += 1
                    self.passages_embedded += n
            else:
                V = self._embed([self._page_text(s) for s in part])
                for s, v in zip(part, V):
                    self.page[s.path] = v
                    self._passages.pop(s.path, None)
            for s in part:
                self._sig[s.path] = self._signature(s)
        if todo or gone:
            self._rows = None
        return len(todo)

    def _matrix(self):
        if self._rows is None:
            self._paths = list(self.page)
            self._rows = np.vstack([self.page[p] for p in self._paths]) if self._paths else np.zeros((0, 1))
        return self._paths, self._rows

    def _arrive(self, src):
        cur = self._passages.get(src.path)
        if cur is None or cur[0] != self._signature(src):
            cur = (self._signature(src), self._embed(src.texts))
            self._passages[src.path] = cur
            self.files_embedded += 1
            self.passages_embedded += len(src.texts)
        return cur[1]

    # -------------------------------------------------------------- reading
    def search(self, query, k=5, roles=None, **_):
        t0 = time.perf_counter()
        self.sync()
        roles = set(roles) if roles is not None else None
        qv = self._embed([query], "query")[0]
        paths, M = self._matrix()
        scores = M @ qv if len(paths) else np.zeros(0)
        cost = len(paths)
        chosen = []
        for i in np.argsort(-scores):
            try:
                src = self.m.find(paths[i])
            except KeyError:
                continue
            if self.m._allowed(src, roles):  # never read what the caller can't open
                chosen.append((float(scores[i]), src))
            if len(chosen) >= self.file_beam:
                break
        hits, searched = [], []
        for _, src in chosen:
            V = self._arrive(src)
            s = V @ qv
            cost += len(s)
            searched.append(src.path)
            for i in np.argsort(-s)[:k]:
                hits.append(Hit(src.doc_ids[i], float(s[i]), src.path, src.texts[i]))
        hits.sort(key=lambda h: -h.score)
        res = Result(hits[:k], cost, searched, [[s.path for _, s in chosen]])
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res
