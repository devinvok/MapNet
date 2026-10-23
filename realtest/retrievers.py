"""
The contenders. All of them see exactly the same chunks of the same files.

  keyword-all   keyword search over every chunk (classic flat index)
  embed-all     embedding search over every chunk (the standard modern RAG baseline)
  map           MapNet routing: walk the map, search only the chosen files
  map+embed     MapNet routing to a wider set of files, then embedding ranking inside them
"""
from __future__ import annotations

import time

import numpy as np

from mapnet.folder import FolderMap


class Shared:
    """One folder, chunked once; every retriever reads from this."""

    def __init__(self, folder):
        t0 = time.perf_counter()
        self.fm = FolderMap(folder)
        self.fm.sync()
        self.map = self.fm.map
        self.build_s = time.perf_counter() - t0

    def chunks(self):
        for src in self.map.sources():
            for doc_id, text in zip(src.doc_ids, src.texts):
                yield doc_id, text


def _file(doc_id):
    return doc_id.split("#")[0]


class KeywordAll:
    name = "keyword-all"

    def __init__(self, shared):
        self.s = shared

    def search(self, q, k):
        r = self.s.map.flat_search(q, k=k)
        return [(_file(h.doc_id), h.text, h.score) for h in r.hits], r.cost


class MapRoute:
    name = "map"

    def __init__(self, shared):
        self.s = shared

    def search(self, q, k):
        r = self.s.map.search(q, k=k, min_score=0.12)
        return [(_file(h.doc_id), h.text, h.score) for h in r.hits], r.cost


class EmbedAll:
    name = "embed-all"

    def __init__(self, shared, embedder, log=print):
        self.s, self.emb = shared, embedder
        self.ids, self.texts = [], []
        self.M = None
        t0 = time.perf_counter()
        self._add(list(shared.chunks()), log)
        self.build_s = time.perf_counter() - t0

    def _add(self, items, log=print):
        if not items:
            return
        if hasattr(self.emb, "fit") and self.M is None:
            self.emb.fit([t for _, t in items])
        ids, texts = zip(*items)

        def progress(done, total):
            if total > 200 and (done % 2000 < self.emb.batch if hasattr(self.emb, "batch") else False):
                log(f"    embedded {done:,}/{total:,} chunks")
        V = self.emb.embed(list(texts), "document", progress=progress)
        self.ids += ids
        self.texts += texts
        self.index = {d: i for i, d in enumerate(self.ids)}
        self.M = V if self.M is None else np.vstack([self.M, V])

    def add(self, items):
        t0 = time.perf_counter()
        self._add(items)
        return time.perf_counter() - t0

    def vector(self, q):
        return self.emb.embed([q], "query")[0]

    def search(self, q, k):
        s = self.M @ self.vector(q)
        top = np.argsort(-s)[:k]
        return [(_file(self.ids[i]), self.texts[i], float(s[i])) for i in top], len(self.ids)


class MapEmbed:
    """Cheap lexical routing picks a generous set of files; embeddings rank the
    chunks inside them. Tests whether routing loses answers that embedding
    search would find, at a fraction of the comparisons."""

    name = "map+embed"

    def __init__(self, shared, embed_all: EmbedAll, beams=(3, 4), source_beam=12):
        self.s, self.e = shared, embed_all
        self.beams, self.source_beam = beams, source_beam

    def search(self, q, k):
        r = self.s.map.search(q, k=1, beams=self.beams, source_beam=self.source_beam, escalate=False)
        rows = [self.e.index[d] for p in r.sources_searched for d in self.s.map.find(p).doc_ids
                if d in self.e.index]
        if not rows:
            return [], r.cost
        s = self.e.M[rows] @ self.e.vector(q)
        order = np.argsort(-s)[:k]
        cost = r.cost - sum(len(self.s.map.find(p).doc_ids) for p in r.sources_searched) + len(rows)
        return [(_file(self.e.ids[rows[i]]), self.e.texts[rows[i]], float(s[i])) for i in order], cost
