"""
The references as a navigable graph, not a folder tree.

The folder walk (refmap.py) decides each turn from a summary of a whole folder, so a page that is
unusual for its folder makes the folder look far away, and a wrong turn can never be undone. This
walk uses no summaries. Every reference is linked to the references closest to it, wherever they
live, so related items are neighbours even when their files sit in different folders. The graph
has layers, like zooming out on a map: a few general hub references at the top (the file
labels and descriptions), more and more detail below. A question starts at the top and moves,
step by step, to whichever neighbour is closest to it, keeping a few candidates in parallel
(this is the HNSW index of Malkov and Yashunin, built from scratch).

Adding is still resolving: a new reference is inserted by walking to where it belongs and linking
to its neighbours. Nothing is rebuilt. Removed or changed references are marked dead (they can
still be walked through, but are never returned).

Permissions: there is one graph per access list, and a caller only ever enters the graphs of
files they may open. A restricted reference is never linked to a public one.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .core import EVERYONE  # noqa: F401  (documented in the module text above)
from .embedtree import _unit


class Hnsw:
    """A small HNSW index over unit vectors (similarity = dot product)."""

    def __init__(self, dim, M=12, ef_construction=64, seed=0):
        self.M, self.M0, self.efc = M, 2 * M, ef_construction
        self.X = np.zeros((256, dim), dtype=np.float32)
        self.n = 0
        self.levels: list[int] = []
        self.links: list[list[list[int]]] = []  # node -> layer -> neighbour ids
        self.entry, self.top = None, -1
        self.rng = np.random.default_rng(seed)
        self.mL = 1.0 / math.log(max(M, 2))
        self.compared = 0  # similarity computations, the cost of a search

    def _search_layer(self, q, entries, ef, layer):
        seen = set(entries)
        sims = self.X[entries] @ q
        self.compared += len(entries)
        cand = [(-float(s), e) for s, e in zip(sims, entries)]
        heapq.heapify(cand)
        best = [(float(s), e) for s, e in zip(sims, entries)]
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)
        while cand:
            neg, c = heapq.heappop(cand)
            if len(best) >= ef and -neg < best[0][0]:
                break
            nbrs = [x for x in self.links[c][layer] if x not in seen]
            if not nbrs:
                continue
            seen.update(nbrs)
            sims = self.X[nbrs] @ q
            self.compared += len(nbrs)
            for s, x in zip(sims, nbrs):
                s = float(s)
                if len(best) < ef or s > best[0][0]:
                    heapq.heappush(cand, (-s, x))
                    heapq.heappush(best, (s, x))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted(best, reverse=True)

    def add(self, vec, boost=0):
        """Insert a vector. `boost` raises its layer (a general item sits nearer the top)."""
        v = _unit(np.asarray(vec, dtype=np.float32))
        if self.n == len(self.X):
            self.X = np.vstack([self.X, np.zeros_like(self.X)])
        i = self.n
        self.X[i] = v
        self.n += 1
        level = int(-math.log(max(self.rng.random(), 1e-12)) * self.mL) + boost
        self.levels.append(level)
        self.links.append([[] for _ in range(level + 1)])
        if self.entry is None:
            self.entry, self.top = i, level
            return i
        ep = [self.entry]
        for layer in range(self.top, level, -1):
            ep = [self._search_layer(v, ep, 1, layer)[0][1]]
        for layer in range(min(self.top, level), -1, -1):
            found = self._search_layer(v, ep, self.efc, layer)
            cap = self.M0 if layer == 0 else self.M
            chosen = [j for _, j in found[:self.M]]
            self.links[i][layer] = list(chosen)
            for j in chosen:
                self.links[j][layer].append(i)
                if len(self.links[j][layer]) > cap:  # keep a node's closest neighbours only
                    nb = self.links[j][layer]
                    keep = np.argsort(-(self.X[nb] @ self.X[j]))[:cap]
                    self.links[j][layer] = [nb[x] for x in keep]
            ep = [j for _, j in found]
        if level > self.top:
            self.entry, self.top = i, level
        return i

    def search(self, q, ef=64):
        """The best `ef` nodes for q as (similarity, id), and how many comparisons it took."""
        if self.entry is None:
            return [], 0
        q = _unit(np.asarray(q, dtype=np.float32))
        before = self.compared
        ep = [self.entry]
        for layer in range(self.top, 0, -1):
            ep = [self._search_layer(q, ep, 1, layer)[0][1]]
        out = self._search_layer(q, ep, ef, 0)
        return out, self.compared - before


class _Group:
    """The graph for one access list."""

    def __init__(self, dim, probe, **kw):
        self.h = Hnsw(dim, **kw)
        self.probe = probe          # a source in this group, to ask "may this caller open it?"
        self.nodes: list = []       # id -> Ref
        self.index: dict = {}       # (target, text) -> id
        self.dead: set = set()


class RefGraph:
    def __init__(self, refmap, M=12, ef_construction=64, ef=64, general_up=True, seed=0):
        """`refmap` supplies the references (and reads files on arrival). With `general_up`,
        file labels and descriptions get a higher layer than the questions written for them."""
        self.base = refmap
        self.M, self.efc, self.ef, self.general_up, self.seed = M, ef_construction, ef, general_up, seed
        self.groups: dict = {}
        self._v = -1
        self.build_seconds = 0.0

    # --------------------------------------------------------------- building
    def sync(self):
        """Bring the graph up to date with the references: insert new ones, mark gone ones dead."""
        t0 = time.perf_counter()
        m = self.base.m
        seen = set()
        for refs in list(self.base.refs.values()):
            for r in refs:
                if not r.home:
                    continue  # copies placed in other folders are what the graph replaces
                try:
                    src = m.find(r.target)
                except KeyError:
                    continue
                key = (r.target, r.text)
                seen.add(key)
                acl = tuple(sorted(src.acl))
                g = self.groups.get(acl)
                if g is None:
                    g = self.groups[acl] = _Group(len(r.vec), src, M=self.M, ef_construction=self.efc,
                                                  seed=self.seed)
                if key in g.index:
                    g.dead.discard(g.index[key])  # back again (a re-registered claim)
                    continue
                boost = 1 if (self.general_up and r.kind in ("label", "about")) else 0
                g.index[key] = g.h.add(r.vec, boost=boost)
                g.nodes.append(r)
        for g in self.groups.values():
            g.dead |= {i for k, i in g.index.items() if k not in seen}
        self._v = self.base._version
        self.build_seconds += time.perf_counter() - t0

    def stats(self):
        return dict(groups=len(self.groups), nodes=sum(g.h.n for g in self.groups.values()),
                    dead=sum(len(g.dead) for g in self.groups.values()),
                    layers=max((g.h.top + 1 for g in self.groups.values()), default=0))

    # --------------------------------------------------------------- walking
    def _rank(self, qv, roles):
        if self._v != self.base._version:
            self.sync()
        m = self.base.m
        best, cost = {}, 0
        for g in self.groups.values():
            if not m._allowed(g.probe, roles):
                continue  # this group's files can't be opened by the caller: never contacted
            found, c = g.h.search(qv, ef=self.ef)
            cost += c
            for s, i in found:
                if i in g.dead:
                    continue
                t = g.nodes[i].target
                if s > best.get(t, -2.0):
                    best[t] = s
        ranked = sorted(best.items(), key=lambda x: -x[1])
        chosen = []
        for path, s in ranked:
            try:
                src = m.find(path)
            except KeyError:
                continue
            if m._allowed(src, roles):
                chosen.append((s, src))
            if len(chosen) >= self.base.file_beam:
                break
        return ranked, chosen, cost, [[f"graph: {self.stats()['layers']} layers"]]

    def search(self, query, k=5, roles=None, translator=None, **_):
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        qv, _terms = self.base._query_vector(query, translator)
        _ranked, chosen, cost, trace = self._rank(qv, roles)
        return self.base._read(chosen, qv, k, cost, trace, t0)

    def search_in(self, query, source_paths, k=5, roles=None, translator=None):
        return self.base.search_in(query, source_paths, k=k, roles=roles, translator=translator)

    def explain(self, query, gold, roles=None, translator=None):
        """Was a right file among the files this walk picked to read? And at what rank?"""
        roles = set(roles) if roles is not None else None
        gold = set(gold)
        qv, terms = self.base._query_vector(query, translator)
        ranked, chosen, _, _ = self._rank(qv, roles)
        rank = next((i + 1 for i, (p, _) in enumerate(ranked) if self.base._file_of(p) in gold), None)
        return dict(rank=rank, in_chosen=any(self.base._file_of(s.path) in gold for _, s in chosen),
                    terms=terms)
