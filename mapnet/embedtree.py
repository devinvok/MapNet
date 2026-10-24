"""
The map, walked by meaning, embedded on the way (DNS style, real time).

Nothing is embedded in bulk. The map starts with light "directions" only: one
small vector per file, from its path, headings and one-line description (if
there is one). Folders get the average of what's below them.

A question walks down from the root toward the closest children. When the walk
arrives at a file whose content hasn't been embedded yet, it embeds that file's
passages right then, uses them, and keeps them: from then on that file is
represented by its real content, and the folders above it sharpen too. The map
gets more precise wherever questions actually go.

Checking undirected data: at every level the walk takes its best branches, and
ALSO checks a branch nobody has looked inside yet, unless the branches it does
know about are clearly better (by more than `margin`). So a branch with only a
vague label still gets visited when there's no strong reason to go elsewhere.

When the match stops rising on the way down (the best child matches worse than
its parent), the walk is probably off course, so it looks wider at that level.

Folders don't publish one blurry average. Like a DNS zone listing the names it
answers for, each folder keeps a few representative vectors (clusters of what's
below it), and its match is the best of them. A specific question about one file
inside a broad folder then still finds its way in.
"""
from __future__ import annotations

import time

import numpy as np

from .core import Hit, Result


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _cluster(X, r, iters=8):
    """A few cluster centres of the rows of X (deterministic k-means)."""
    if len(X) <= r:
        return X
    C = X[np.linspace(0, len(X) - 1, r).astype(int)]
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)
        C = np.stack([_unit(X[a == j].mean(axis=0)) if (a == j).any() else C[j] for j in range(r)])
    return C


class EmbedTree:
    def __init__(self, mapnet, embedder, beam=4, file_beam=6, margin=0.03, explore=1,
                 max_new=4, widen=2, reps=4):
        self.m, self.emb = mapnet, embedder
        self.beam = beam            # branches kept at each folder level
        self.file_beam = file_beam  # files read per question
        self.margin = margin        # how much better known branches must be to skip an unchecked one
        self.explore = explore      # unchecked branches that may be added per level
        self.max_new = max_new      # files embedded on arrival per question (bounds latency)
        self.widen = widen
        self.reps = reps            # representative vectors per folder (its distinct sub-topics)
        self._label: dict[str, tuple] = {}    # path -> (text, vec)   the "direction"
        self._content: dict[str, tuple] = {}  # path -> (signature, unit mean vec, passage vecs)
        self._version, self._memo_v, self._memo = 0, -1, {}
        self.label_vectors = 0
        self.files_embedded = 0
        self.passages_embedded = 0
        self.embed_seconds = 0.0

    # ----------------------------------------------------------- directions
    @staticmethod
    def _rel(path):
        return path[len("_top/"):] if path.startswith("_top/") else path

    def _label_text(self, src):
        outline = "\n".join(s for s in dict.fromkeys(src.sec) if s)[:1200]
        return "\n".join(t for t in (self._rel(src.path), getattr(src, "file_card", ""), outline) if t)

    def _embed(self, texts, kind="document"):
        t0 = time.perf_counter()
        V = np.asarray(self.emb.embed(list(texts), kind), dtype=np.float32)
        self.embed_seconds += time.perf_counter() - t0
        return V

    def build(self):
        """Embed the directions (one short label per file) for files that lack one
        or whose label changed. Batched; cheap compared with embedding content."""
        todo = []
        for src in self.m.sources():
            text = self._label_text(src)
            cur = self._label.get(src.path)
            if cur is None or cur[0] != text:
                todo.append((src.path, text))
        if todo:
            V = self._embed([t for _, t in todo])
            for (path, text), v in zip(todo, V):
                self._label[path] = (text, _unit(v))
            self.label_vectors += len(todo)
            self._version += 1
        return len(todo)

    # ------------------------------------------------------------- arriving
    def _signature(self, src):
        return (len(src.doc_ids), src.doc_ids[0] if src.doc_ids else "", hash(tuple(src.texts)))

    def _arrive(self, src):
        """Embed a file's passages the first time the walk gets here."""
        sig = self._signature(src)
        cur = self._content.get(src.path)
        if cur is not None and cur[0] == sig:
            return cur
        V = self._embed(src.texts)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        cur = (sig, _unit(V.mean(axis=0)), V)
        self._content[src.path] = cur
        self.files_embedded += 1
        self.passages_embedded += len(src.texts)
        self._version += 1
        return cur

    def embedded(self, src):
        cur = self._content.get(src.path)
        return cur is not None and cur[0] == self._signature(src)

    # -------------------------------------------------------------- vectors
    def _info(self, node):
        """(vector, checked) for a node; checked = some file below it has real content."""
        if self._memo_v != self._version:
            self._memo, self._memo_v = {}, self._version
        hit = self._memo.get(id(node))
        if hit is not None:
            return hit
        if node.kind == "source":
            if self.embedded(node):
                out = (self._content[node.path][1], True)
            else:
                lab = self._label.get(node.path)
                out = (lab[1], False) if lab else (None, False)
        else:
            parts = [self._info(c) for c in node.children.values()]
            vecs = [v for v, _ in parts if v is not None]
            out = (_unit(np.mean(vecs, axis=0)) if vecs else None, any(c for _, c in parts))
        self._memo[id(node)] = out
        return out

    def _reps(self, node):
        """(representative vectors, checked). Files: their one vector. Folders:
        up to `reps` cluster centres of the vectors below them."""
        self._info(node)  # keeps the memo current
        key = ("reps", id(node))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        if node.kind == "source":
            v, c = self._info(node)
            out = (v[None, :] if v is not None else None, c)
        else:
            parts = [self._reps(ch) for ch in node.children.values()]
            vecs = [r for r, _ in parts if r is not None]
            out = (_cluster(np.vstack(vecs), self.reps) if vecs else None, any(c for _, c in parts))
        self._memo[key] = out
        return out

    # ------------------------------------------------------------- walking
    def _pick(self, cands, width):
        """Best `width` branches, plus up to `explore` unchecked ones that the
        known branches don't clearly beat."""
        chosen = cands[:width]
        known = [s for s, n, _ in cands if self._info(n)[1]]
        best_known = max(known) if known else None
        extra = [x for x in cands[width:] if not self._info(x[1])[1]
                 and (best_known is None or x[0] >= best_known - self.margin)][: self.explore]
        return chosen + extra

    def search(self, query, k=5, roles=None, min_score=None, **_):
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        qv = _unit(self._embed([query], "query")[0])
        m = self.m
        cost, files, trace = 0, [], []
        frontier = [(m.root, -1.0)]
        while frontier:
            cands = []
            for node, parent_score in frontier:
                for child in node.children.values():
                    if not m._allowed(child, roles):
                        continue  # never even contacted
                    reps, _ = self._reps(child)
                    if reps is None:
                        continue
                    s = float((reps @ qv).max())
                    cost += len(reps)
                    (files if child.kind == "source" else cands).append((s, child, parent_score))
            if not cands:
                break
            cands.sort(key=lambda x: -x[0])
            width = self.beam
            if cands[0][0] < max(p for _, _, p in cands):
                width *= self.widen  # the match stopped rising: look wider here
            picked = self._pick(cands, width)
            trace.append([n.path for _, n, _ in picked])
            frontier = [(n, s) for s, n, _ in picked]
        files.sort(key=lambda x: -x[0])
        chosen = [src for _, src, _ in self._pick(files, self.file_beam)]
        return self._read(chosen, qv, k, cost, trace, t0)

    def search_in(self, query, source_paths, k=5, roles=None):
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        qv = _unit(self._embed([query], "query")[0])
        chosen = []
        for p in source_paths:
            try:
                src = self.m.find(p)
            except KeyError:
                continue
            if src.kind == "source" and src.X is not None and self.m._allowed(src, roles):
                chosen.append(src)
        return self._read(chosen, qv, k, 0, [], t0)

    def _read(self, chosen, qv, k, cost, trace, t0):
        new, hits, searched = 0, [], []
        for src in chosen:
            if not self.embedded(src):
                if new >= self.max_new:
                    continue  # bounded latency: the rest wait for the next visit
                self._arrive(src)
                new += 1
            V = self._content[src.path][2]
            scores = V @ qv
            cost += len(scores)
            searched.append(src.path)
            for i in np.argsort(-scores)[:k]:
                hits.append(Hit(src.doc_ids[i], float(scores[i]), src.path, src.texts[i]))
        hits.sort(key=lambda h: -h.score)
        res = Result(hits[:k], cost, searched, trace)
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res
