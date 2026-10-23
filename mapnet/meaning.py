"""
Meaning-based routing: walk the same map, but compare the question with each
node by meaning (an embedding model), not only by shared words.

What gets embedded is the MAP, not the documents, and DNS-style, level by level:
  - each file, up front: its path, its one-line description (if any) and its outline
  - each folder: the average of what's below it (no model call)
  - a file's sections (heading path + card), only when a question first arrives
    at that file
So the up-front index is one vector per file, and vectors change only when the
map changes (a new file, a new description or card).

Routing asks two resolvers and follows both. Word matching keeps what it's good
at: following exact names like "PodDisruptionBudget" to the one file that has
them. Meaning adds what words miss ("my app keeps getting killed while
starting" ~ startup probes). At every level the word router's best branches are
kept, and the meaning router's best branches are ADDED (never swapped in), so a
weak or wrong meaning signal costs a little extra work but can't push out a good
word match. Inside a file, meaning picks out sections the same way.

Defaults (gamma, delta) are untuned starting values.
"""
from __future__ import annotations

import time

import numpy as np

from .core import Hit, Result


class MeaningRouter:
    def __init__(self, mapnet, embedder, extra=1, extra_sources=2, extra_sections=1):
        self.m, self.emb = mapnet, embedder
        self.extra = extra                    # branches meaning may add at each folder level
        self.extra_sources = extra_sources    # files meaning may add
        self.extra_sections = extra_sections  # sections meaning may add inside each file
        self._file: dict[str, tuple] = {}  # path -> (signature, file_vec)
        self._secs: dict[str, tuple] = {}  # path -> (signature, {section: vec}), on arrival
        self._version = 0
        self._gcache: dict[int, np.ndarray] = {}
        self._gversion = -1
        self.vectors = 0
        self.seconds = 0.0

    # ------------------------------------------------------------ the index
    @staticmethod
    def _rel(path):
        return path[len("_top/"):] if path.startswith("_top/") else path

    def _file_text(self, src):
        rel = self._rel(src.path)
        outline = "\n".join(s for s in dict.fromkeys(src.sec) if s)[:1500]
        return "\n".join(t for t in (rel, src.file_card, outline) if t)

    def _embed(self, texts):
        t0 = time.perf_counter()
        V = np.asarray(self.emb.embed(texts, "document"), dtype=np.float32)
        self.vectors += len(texts)
        self.seconds += time.perf_counter() - t0
        return V

    def _ensure(self, src):
        """The file's vector (folder level of the map)."""
        text = self._file_text(src)
        cur = self._file.get(src.path)
        if cur is None or cur[0] != text:
            cur = (text, self._embed([text])[0])
            self._file[src.path] = cur
            self._version += 1
        return cur

    def _sections(self, src):
        """The file's section vectors, made the first time a question arrives here."""
        rel = self._rel(src.path)
        secs = list(dict.fromkeys(src.sec)) or [""]
        texts = {s: "\n".join(t for t in (f"{rel}: {s or '(top)'}", src.cards.get(s, "")) if t) for s in secs}
        sig = tuple(texts.values())
        cur = self._secs.get(src.path)
        if cur is None or cur[0] != sig:
            cur = (sig, dict(zip(texts, self._embed(list(texts.values())))))
            self._secs[src.path] = cur
        return cur[1]

    def build(self, log=None):
        """Embed the map for every file that isn't embedded yet (or changed)."""
        srcs = self.m.sources()
        for i, src in enumerate(srcs, 1):
            self._ensure(src)
            if log and i % 100 == 0:
                log(f"    meaning index: {i:,}/{len(srcs):,} files")
        return self.vectors, self.seconds

    def _vec(self, node):
        if self._gversion != self._version:
            self._gcache, self._gversion = {}, self._version
        v = self._gcache.get(id(node))
        if v is None:
            if node.kind == "source":
                v = self._ensure(node)[1]
            else:
                kids = [self._vec(c) for c in node.children.values()]
                v = np.mean(kids, axis=0) if kids else np.zeros(1, dtype=np.float32)
                n = np.linalg.norm(v)
                v = v / n if n else v
            self._gcache[id(node)] = v
        return v

    @staticmethod
    def _z(x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) < 2:
            return np.zeros(len(x))
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-9 else np.zeros(len(x))

    # ------------------------------------------------------------ searching
    def _query(self, query):
        return self.m._encode_query(query), np.asarray(self.emb.embed([query], "query")[0], dtype=np.float32)

    def _score_in(self, src, q, qv, k):
        """Word scores for every passage, plus: the best passage of each section
        meaning ranks highest gets a place among the file's top k."""
        scores, c = self.m._score_docs(src, q)
        secvecs = self._sections(src)
        order = list(secvecs)
        picks = list(np.argsort(-scores)[:k])
        if len(order) > 1 and self.extra_sections:
            sims = np.array([secvecs[s] @ qv for s in order])
            for si in np.argsort(-sims)[: self.extra_sections]:
                rows = [i for i, s in enumerate(src.sec) if s == order[si]]
                best = max(rows, key=lambda i: scores[i])
                if best not in picks:
                    picks.append(best)
        return scores, picks, c + len(order)

    def _route(self, q, qv, k, roles, beams, source_beam):
        m = self.m
        frontier, pool, cost, trace, level = [m.root], [], 0, [], 0
        while frontier:
            candidates = []
            for node in frontier:
                if not node.children:
                    continue
                kids, lex = m._child_scores(node, q)
                ok = [(kid, s) for kid, s in zip(kids, lex) if m._allowed(kid, roles)]
                if not ok:
                    continue
                sims = [float(self._vec(kid) @ qv) for kid, _ in ok]
                for (kid, s), sim in zip(ok, sims):
                    cost += 1
                    (pool if kid.kind == "source" else candidates).append((s, sim, kid))
            b = beams[min(level, len(beams) - 1)]
            by_words = sorted(candidates, key=lambda x: -x[0])[:b]
            by_meaning = sorted(candidates, key=lambda x: -x[1])[: self.extra]
            frontier = list(dict.fromkeys(n for _, _, n in by_words + by_meaning))
            if frontier:
                trace.append([n.path for n in frontier])
            level += 1
        by_words = sorted(pool, key=lambda x: -x[0])[:source_beam]
        by_meaning = sorted(pool, key=lambda x: -x[1])[: self.extra_sources]
        chosen, seen = [], set()
        for s, sim, src in by_words + by_meaning:
            if id(src) not in seen:
                seen.add(id(src))
                chosen.append((s, src))
        return self._collect(chosen, q, qv, k, cost, trace)

    def _collect(self, chosen, q, qv, k, cost, trace):
        """Rank the passages of the chosen files. The word ranking fills k-1
        seats; the last seat goes to meaning's first choice: the best passage of
        the section meaning ranks highest, in the file meaning ranks highest."""
        m = self.m
        word_hits, meaning_pick = [], None
        for route_score, src in chosen:
            scores, picks, c = self._score_in(src, q, qv, k)
            cost += c
            if m.route_weight:
                scores = scores + m.route_weight * route_score
            hs = [Hit(src.doc_ids[i], float(scores[i]), src.path, src.texts[i]) for i in picks]
            word_hits += hs[:k]
            sim = float(self._vec(src) @ qv)
            cand = hs[k] if len(hs) > k else hs[0] if hs else None
            if cand is not None and (meaning_pick is None or sim > meaning_pick[0]):
                meaning_pick = (sim, cand)
        word_hits.sort(key=lambda h: -h.score)
        top = word_hits[: max(k - 1, 1)] if meaning_pick else word_hits[:k]
        ids = {h.doc_id for h in top}
        for h in ([meaning_pick[1]] if meaning_pick else []) + word_hits[len(top):]:
            if len(top) >= k:
                break
            if h.doc_id not in ids:
                top.append(h)
                ids.add(h.doc_id)
        return Result(top, cost, [s.path for _, s in chosen], trace)

    def search(self, query, k=5, roles=None, beams=(2, 3), source_beam=4, escalate=True, min_score=0.3):
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        q, qv = self._query(query)
        res = self._route(q, qv, k, roles, beams, source_beam)
        if escalate and (not res.hits or res.hits[0].score < min_score):
            wide = self._route(q, qv, k, roles, tuple(b * 3 for b in beams), source_beam * 3)
            wide.cost += res.cost
            wide.escalated = True
            res = wide
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res

    def search_in(self, query, source_paths, k=5, roles=None):
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        q, qv = self._query(query)
        chosen = []
        for p in source_paths:
            try:
                src = self.m.find(p)
            except KeyError:
                continue
            if src.kind != "source" or src.X is None or not self.m._allowed(src, roles):
                continue
            route = 0.0
            if self.m.route_weight and src.parent is not None:
                kids, lex = self.m._child_scores(src.parent, q)
                route = float(lex[kids.index(src)])
            chosen.append((route, src))
        res = self._collect(chosen, q, qv, k, 0, [])
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res
