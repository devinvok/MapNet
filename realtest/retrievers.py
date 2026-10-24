"""
The contenders. All of them see exactly the same chunks of the same files.

  keyword-all   keyword search over every chunk (classic flat index)
  embed-all     embedding search over every chunk (the standard modern RAG baseline)
  map           MapNet routing: walk the map, search only the chosen files
  map+sections  ...and inside each file, walk its headings to the right section
  map+lazy-cards  ...where each file is mapped by the model the first time a question reaches it
  map+file-cards  ...plus a one-line description of every file at the folder level: the full chain
  map+meaning     the full chain, walked by meaning (an embedding of the map, not of the
                  passages) as well as by words
  map+cards     ...with agent-written section cards for every file, made in advance
  map+embed     MapNet routing to a wider set of files, then embedding ranking inside them
  embed-tree    the map walked by meaning alone; each file embedded when the walk first arrives
  ref-map       a namespace of references (descriptions + questions), each question registered
                where it resolves; contents fetched and embedded on arrival
"""
from __future__ import annotations

import time

import numpy as np

from mapnet.folder import FolderMap


class Shared:
    """One folder, chunked once; every retriever reads from this."""

    def __init__(self, folder, cards=None):
        t0 = time.perf_counter()
        self.fm = FolderMap(folder, cards=cards)
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
    def __init__(self, shared, name="map", sections=False):
        self.s, self.name, self.sections = shared, name, sections

    def search(self, q, k):
        m = self.s.map
        old, m.section_weight = m.section_weight, (1.0 if self.sections else 0.0)
        try:
            r = m.search(q, k=k, min_score=0.12)
        finally:
            m.section_weight = old
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


class MapLazy:
    """map+sections, where each file is mapped (carded by the local model) the
    first time a question arrives at it. Starts with no cards at all: the map
    gets better as questions come in, and the cost of every first visit is
    counted in this method's lookup time."""

    def __init__(self, shared, store, reader, name="map+lazy-cards", route_weight=0.0, router=None):
        from mapnet.cards import LazyCarder
        self.s, self.name = shared, name
        shared.map.route_weight = route_weight  # file-level decisions carry down to passages
        self.router = router or shared.map  # MapNet itself, or a MeaningRouter over it
        self.lazy = LazyCarder(shared.fm, store, reader, sync_top=1, background=False) if reader else None

    def search(self, q, k):
        m = self.s.map
        old, m.section_weight = m.section_weight, 1.0
        try:
            r = self.router.search(q, k=k, min_score=0.12)
            if self.lazy and self.lazy.visit(r.sources_searched):
                again = self.router.search_in(q, r.sources_searched, k=k)
                again.cost += r.cost
                r = again
        finally:
            m.section_weight = old
        return [(_file(h.doc_id), h.text, h.score) for h in r.hits], r.cost


class CombinedCards:
    """File descriptions from one store (built in advance), section cards from
    another (written on arrival during the run)."""

    def __init__(self, file_cards, section_cards):
        self.files, self.sections = file_cards, section_cards

    def get(self, text):
        return self.sections.get(text)

    def put(self, text, cards, meta):
        self.sections.put(text, cards, meta)

    def get_file(self, text):
        return self.files.get_file(text) or self.sections.get_file(text)

    def put_file(self, text, card, meta):
        self.sections.put_file(text, card, meta)


class EmbedTreeRoute:
    """embed-tree: the map walked by meaning, each file embedded the first time
    the walk arrives there (see mapnet/embedtree.py). Starts with only one short
    label vector per file; its own embedding cache starts empty, so every
    first-visit embedding is real work and counted in its lookup time."""

    name = "embed-tree"

    def __init__(self, shared, embedder):
        from mapnet.embedtree import EmbedTree
        self.s = shared
        self.tree = EmbedTree(shared.map, embedder)
        self.tree.build()

    def search(self, q, k):
        r = self.tree.search(q, k=k)
        return [(_file(h.doc_id), h.text, h.score) for h in r.hits], r.cost


class RefMapRoute:
    """ref-map: a namespace of references, DNS style (see mapnet/refmap.py).
    Entries are the file descriptions and the questions the local model wrote
    for each file, each question registered wherever it resolves; file
    contents are embedded only when a question arrives. Its embedding cache
    starts empty, so all of that work is counted."""

    name = "ref-map"

    def __init__(self, shared, embedder, log=print):
        from mapnet.refmap import RefMap
        self.s = shared
        self.map = RefMap(shared.map, embedder)
        t0 = time.perf_counter()
        self.map.build(log)
        self.build_s = time.perf_counter() - t0

    def search(self, q, k):
        r = self.map.search(q, k=k)
        return [(_file(h.doc_id), h.text, h.score) for h in r.hits], r.cost
