"""
MapNet core: a live, hierarchical "map" of data sources that a router walks
like DNS, instead of searching one big flat index.

Tree shape (any depth works):
    root -> group nodes (e.g. domain -> topic) -> source nodes -> documents

Every node carries an automatic summary (the centroid of everything beneath
it), an access list (union of its children's), and a "learned" vector built
from feedback: the questions that were successfully answered through it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

EVERYONE = "*"


# --------------------------------------------------------------------------- #
# Text encoding
# --------------------------------------------------------------------------- #
class HashEmbedder:
    """Fit-free sparse encoder. Because nothing is fitted, a new source can join
    the map at any moment without re-indexing anything else.

    Swap in any object with `.n_features` and `.encode(texts) -> csr_matrix`
    (e.g. a wrapper around a sentence-transformer that returns sparse rows,
    or a small LLM's embeddings) to upgrade the "router brain"."""

    def __init__(self, n_features: int = 2 ** 18):
        self.n_features = n_features
        self._v = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm="l2", stop_words="english"
        )

    def encode(self, texts) -> sp.csr_matrix:
        return self._v.transform(list(texts)).tocsr()


def _norm_rows(m: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / n) @ m)


class LiveIdf:
    """Document-frequency counts updated incrementally as sources register."""

    def __init__(self, n_features: int):
        self.df = np.zeros(n_features)
        self.n = 0

    def add(self, X: sp.csr_matrix):
        self.df += np.bincount(X.indices, minlength=self.df.shape[0])
        self.n += X.shape[0]

    def weigh(self, q: sp.csr_matrix) -> sp.csr_matrix:
        q = q.tocsr().copy()
        q.data = q.data * (np.log((self.n + 1) / (self.df[q.indices] + 1)) + 1)
        return _norm_rows(q)


# --------------------------------------------------------------------------- #
# Map nodes
# --------------------------------------------------------------------------- #
class Node:
    def __init__(self, name: str, n_features: int, parent: "Node | None" = None, kind="group"):
        self.name, self.parent, self.kind = name, parent, kind
        self.children: dict[str, Node] = {}
        self.acl: set[str] = set()
        self.n_docs = 0
        self.summary = sp.csr_matrix((1, n_features))  # running sum of doc vectors
        self.learned = sp.csr_matrix((1, n_features))  # running sum of answered queries
        self.n_feedback = 0
        # Which terms exist anywhere below this node, like a DNS zone listing
        # its names. Lets a router follow a specific name (a project, vendor or
        # ticket id) even when the node's average summary is about something else.
        self.vocab: set[int] = set()
        self._cache = None  # (children list, centroid matrix, learned matrix)
        # source-only
        self.doc_ids: list[str] = []
        self.texts: list[str] = []
        self.X: sp.csr_matrix | None = None

    @property
    def path(self) -> str:
        parts, n = [], self
        while n.parent is not None:
            parts.append(n.name)
            n = n.parent
        return "/".join(reversed(parts)) or "/"

    def __repr__(self):
        return f"<{self.kind} {self.path} docs={self.n_docs}>"


@dataclass
class Hit:
    doc_id: str
    score: float
    source: str
    text: str


@dataclass
class Result:
    hits: list[Hit]
    cost: int  # vector comparisons: nodes scored + documents scored
    sources_searched: list[str] = field(default_factory=list)
    trace: list[list[str]] = field(default_factory=list)  # chosen nodes per level
    escalated: bool = False
    restricted_docs_scanned: int = 0
    latency_ms: float = 0.0


# --------------------------------------------------------------------------- #
# The map
# --------------------------------------------------------------------------- #
class MapNet:
    def __init__(self, embedder=None, feedback_weight: float = 0.5, name_weight: float = 0.5):
        self.emb = embedder or HashEmbedder()
        self.F = self.emb.n_features
        self.root = Node("root", self.F)
        self.idf = LiveIdf(self.F)
        self.doc_source: dict[str, Node] = {}
        self.feedback_weight = feedback_weight
        self.name_weight = name_weight  # 0 disables name-following
        self._flat = None  # lazily rebuilt global matrix for the flat baseline

    # ----------------------------- building -------------------------------- #
    def register_source(self, name, docs, path=None, acl=(EVERYONE,)):
        """Add a data source (a system, an MCP server, a folder...) to the map.

        docs: list of (doc_id, text). path: list of group names, e.g.
        ["finance", "invoices"]. If path is None the source places itself
        under the group whose summary it most resembles."""
        if path is None:
            path = self.auto_place(self.emb.encode([t for _, t in docs]))
        parent = self._ensure_path(path)
        if name in parent.children:
            raise ValueError(f"{parent.path}/{name} already exists")
        node = Node(name, self.F, parent, kind="source")
        node.acl = set(acl)
        parent.children[name] = node
        self._propagate_acl(node)
        self.add_documents(node, docs)
        return node

    def add_documents(self, source, docs):
        """Live update: push new documents into an existing source."""
        node = source if isinstance(source, Node) else self.find(source)
        if not docs:
            return node
        ids, texts = zip(*docs)
        X = self.emb.encode(texts)
        node.X = X if node.X is None else sp.vstack([node.X, X]).tocsr()
        node.doc_ids += ids
        node.texts += texts
        for i in ids:
            self.doc_source[i] = node
        delta = sp.csr_matrix(X.sum(axis=0))
        terms = set(np.unique(X.indices).tolist())
        n = node
        while n is not None:  # summaries of every ancestor change
            n.summary = n.summary + delta
            n.vocab |= terms
            n.n_docs += len(ids)
            n._cache = None
            n = n.parent
        self.idf.add(X)
        self._flat = None
        return node

    def remove_source(self, source):
        """Live removal: a file was deleted or is about to be re-registered.
        Ancestor summaries are reduced; names and permissions are rebuilt from
        the remaining children; empty groups are pruned."""
        node = source if isinstance(source, Node) else self.find(source)
        if node.X is not None:
            self.idf.df -= np.bincount(node.X.indices, minlength=self.F)
            self.idf.n -= node.X.shape[0]
        for i in node.doc_ids:
            self.doc_source.pop(i, None)
        parent = node.parent
        del parent.children[node.name]
        n = parent
        while n is not None:
            n.summary = n.summary - node.summary
            n.summary.eliminate_zeros()
            n.learned = n.learned - node.learned
            n.learned.eliminate_zeros()
            n.n_docs -= node.n_docs
            n.vocab = set().union(*(c.vocab for c in n.children.values())) if n.children else set()
            n.acl = set().union(*(c.acl for c in n.children.values())) if n.children else set()
            n._cache = None
            n = n.parent
        while parent.parent is not None and not parent.children:  # prune empty groups
            gp = parent.parent
            del gp.children[parent.name]
            gp._cache = None
            parent = gp
        self._flat = None

    def search_in(self, query, source_paths, k=5, roles=None):
        """Search only the given sources (used when a note points straight there)."""
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        q = self._encode_query(query)
        hits, cost, searched = [], 0, []
        for p in source_paths:
            try:
                src = self.find(p)
            except KeyError:
                continue
            if src.kind != "source" or src.X is None or not self._allowed(src, roles):
                continue
            scores = (src.X @ q.T).toarray().ravel()
            cost += len(scores)
            searched.append(src.path)
            for i in np.argsort(-scores)[:k]:
                hits.append(Hit(src.doc_ids[i], float(scores[i]), src.path, src.texts[i]))
        hits.sort(key=lambda h: -h.score)
        return Result(hits[:k], cost, searched, latency_ms=(time.perf_counter() - t0) * 1000)

    def sources(self):
        return list(self._sources())

    def auto_place(self, X: sp.csr_matrix) -> list[str]:
        """Descend by similarity until reaching a group that holds sources."""
        c = _norm_rows(sp.csr_matrix(X.sum(axis=0)))
        node, path = self.root, []
        while True:
            groups = [ch for ch in node.children.values() if ch.kind == "group"]
            if not groups:
                return path or ["unsorted"]
            C = _norm_rows(sp.vstack([g.summary for g in groups]).tocsr())
            best = groups[int(np.argmax((C @ c.T).toarray().ravel()))]
            node = best
            path.append(best.name)

    def _ensure_path(self, path):
        node = self.root
        for part in path:
            if part not in node.children:
                node.children[part] = Node(part, self.F, node, kind="group")
                node._cache = None
            node = node.children[part]
            if node.kind != "group":
                raise ValueError(f"{node.path} is a source, not a group")
        return node

    def _propagate_acl(self, node):
        n = node.parent
        while n is not None:
            n.acl |= node.acl
            n = n.parent

    def find(self, path: str) -> Node:
        node = self.root
        for part in path.strip("/").split("/"):
            node = node.children[part]
        return node

    # ----------------------------- feedback -------------------------------- #
    def feedback(self, query: str, doc_id: str):
        """Tell the map that `doc_id` answered `query`. Every node on the path to
        that document learns what people actually ask it, so its route gets
        easier to find next time, even if its own summary is poor."""
        q = self._encode_query(query)
        n = self.doc_source[doc_id]
        while n.parent is not None:
            n.learned = n.learned + q
            n.n_feedback += 1
            n.parent._cache = None
            n = n.parent

    # ----------------------------- searching ------------------------------- #
    def _encode_query(self, query):
        return self.idf.weigh(self.emb.encode([query]))

    @staticmethod
    def _allowed(node, roles):
        return roles is None or EVERYONE in node.acl or bool(node.acl & roles)

    def _child_scores(self, node, q):
        if node._cache is None:
            kids = list(node.children.values())
            C = _norm_rows(sp.vstack([k.summary for k in kids]).tocsr())
            L = _norm_rows(sp.vstack([k.learned for k in kids]).tocsr())
            node._cache = (kids, C, L)
        kids, C, L = node._cache
        s = (C @ q.T).toarray().ravel() + self.feedback_weight * (L @ q.T).toarray().ravel()
        if self.name_weight and q.nnz:
            # share of the query's (idf-weighted) mass whose terms exist below each child
            w = q.data / q.data.sum()
            s += self.name_weight * np.array(
                [sum(wi for t, wi in zip(q.indices, w) if t in k.vocab) for k in kids])
        return kids, s

    def search(self, query, k=5, roles=None, beams=(2, 3), source_beam=4,
               escalate=True, min_score=0.3):
        """Route the query down the map, then search only the chosen sources.
        If the best answer looks weak, retry once with a 3x wider beam
        (the 'backtracking' step)."""
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        q = self._encode_query(query)
        res = self._route(q, k, roles, beams, source_beam)
        if escalate and (not res.hits or res.hits[0].score < min_score):
            wide = self._route(q, k, roles, tuple(b * 3 for b in beams), source_beam * 3)
            wide.cost += res.cost
            wide.escalated = True
            res = wide
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res

    def _route(self, q, k, roles, beams, source_beam):
        frontier, pool, cost, trace, level = [self.root], [], 0, [], 0
        while frontier:
            candidates = []
            for node in frontier:
                if not node.children:
                    continue
                kids, scores = self._child_scores(node, q)
                for kid, s in zip(kids, scores):
                    if not self._allowed(kid, roles):
                        continue  # never even contacted
                    cost += 1
                    (pool if kid.kind == "source" else candidates).append((s, kid))
            candidates.sort(key=lambda x: -x[0])
            frontier = [n for _, n in candidates[: beams[min(level, len(beams) - 1)]]]
            if frontier:
                trace.append([n.path for n in frontier])
            level += 1
        pool.sort(key=lambda x: -x[0])
        chosen = [n for _, n in pool[:source_beam]]
        hits = []
        for src in chosen:
            scores = (src.X @ q.T).toarray().ravel()
            cost += len(scores)
            for i in np.argsort(-scores)[:k]:
                hits.append(Hit(src.doc_ids[i], float(scores[i]), src.path, src.texts[i]))
        hits.sort(key=lambda h: -h.score)
        return Result(hits[:k], cost, [s.path for s in chosen], trace)

    # ----------------------------- baseline -------------------------------- #
    def _sources(self, node=None):
        node = node or self.root
        for ch in node.children.values():
            if ch.kind == "source":
                yield ch
            else:
                yield from self._sources(ch)

    def flat_search(self, query, k=5, roles=None):
        """Baseline: score every document, then filter by permission."""
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        if self._flat is None:
            srcs = [s for s in self._sources() if s.X is not None]
            rows = [(s, i) for s in srcs for i in range(len(s.doc_ids))]
            self._flat = (sp.vstack([s.X for s in srcs]).tocsr(), rows)
        G, rows = self._flat
        q = self._encode_query(query)
        scores = (G @ q.T).toarray().ravel()
        restricted = 0
        if roles is not None:
            for j, (s, _) in enumerate(rows):
                if not self._allowed(s, roles):
                    scores[j] = -np.inf
                    restricted += 1
        top = np.argsort(-scores)[:k]
        hits = [Hit(rows[j][0].doc_ids[rows[j][1]], float(scores[j]), rows[j][0].path,
                    rows[j][0].texts[rows[j][1]]) for j in top if np.isfinite(scores[j])]
        return Result(hits, G.shape[0], restricted_docs_scanned=restricted,
                      latency_ms=(time.perf_counter() - t0) * 1000)

    # ----------------------------- display --------------------------------- #
    def show(self, node=None, depth=0, max_sources=3):
        node = node or self.root
        lines = []
        kids = list(node.children.values())
        srcs = [c for c in kids if c.kind == "source"]
        for c in kids:
            if c.kind == "group":
                acl = "" if EVERYONE in c.acl else f"  [acl: {','.join(sorted(c.acl))}]"
                lines.append("  " * depth + f"{c.name}/  ({c.n_docs} docs){acl}")
                lines += self.show(c, depth + 1, max_sources)
        for c in srcs[:max_sources]:
            acl = "" if EVERYONE in c.acl else f"  [acl: {','.join(sorted(c.acl))}]"
            lines.append("  " * depth + f"- {c.name}  ({c.n_docs} docs){acl}")
        if len(srcs) > max_sources:
            lines.append("  " * depth + f"- ... {len(srcs) - max_sources} more sources")
        return lines if depth else print("\n".join(lines))
