"""
A map of references, DNS style.

The map is a namespace of zones. A zone doesn't contain data; it holds
REFERENCES: entries that say "this question is answered at <file>". The data
stays where it is and is only fetched at the end.

How it differs from a tree embedding (hierarchical clustering):
  - One item can be referenced from many zones. A file is registered in its own
    folder's zone, and each question it answers is registered wherever that
    question naturally belongs, so a page about finalizers can also be found
    from the zone where "namespace stuck terminating" questions live.
  - Zones are represented by the specific references registered in them (and a
    few exemplars of what's below), not by an average of their contents.
  - Adding is resolving: to register a new reference, the agent walks the map
    with it exactly as a question would, and puts it next to the closest
    existing reference. No global rebuild.
  - References carry where the data is and who may see it. A reference to a
    restricted file is only ever registered inside that file's own branch, and
    is only returned to callers allowed to open the file.

Where the references come from: the one-line file descriptions written by the
local model (what the file is for + five questions people would ask), plus the
file's path and headings. Content is embedded only when a question arrives at
a file (as in embed-tree).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from .core import EVERYONE, Hit, Result
from .embedtree import _cluster, _unit


@dataclass(eq=False)  # identity, not field-by-field (fields hold arrays)
class Ref:
    target: str       # path of the source in the map (where the data is)
    text: str         # the claim that registered it (a question, a description)
    vec: np.ndarray
    kind: str         # "label" | "about" | "ask"
    home: bool        # registered in the target's own folder zone
    zone: object = None  # the zone it is registered in


TERMS_PROMPT = """Someone asks a question about a collection of documents, in everyday words.
Write the exact technical terms, names, settings and concepts the documents would use
when they answer it (the words that would appear on the right page, not the words in the
question).

Question: {q}

Reply with JSON only: {{"terms": ["...", "...", "...", "...", "..."]}} with 5 to 8 entries."""


class TermTranslator:
    """Turns an everyday question into the technical terms the right page would use
    ("my namespace is stuck deleting" -> finalizers, deletionTimestamp ...). One short
    call to the small model per question; answers are remembered. The terms are only
    used to route; they are never shown to the reader."""

    def __init__(self, reader):
        self.reader = reader
        self._cache: dict[str, str] = {}
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, question: str) -> str:
        if question in self._cache:
            return self._cache[question]
        from .cards import _generate
        t0 = time.perf_counter()
        try:
            out = _generate(self.reader, TERMS_PROMPT.format(q=question))
            terms = out.get("terms", []) if isinstance(out, dict) else []
            text = ", ".join(str(t).strip() for t in terms[:8] if str(t).strip())
        except Exception:
            text = ""
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        self._cache[question] = text
        return text


def split_card(card: str):
    """A stored file description is 'about. ask? ask? ...' -> (about, [asks])."""
    card = " ".join((card or "").split())
    if not card:
        return "", []
    sents = [x.strip() for x in re.split(r"(?<=[.?!])\s+", card) if x.strip()]
    asks = [x for x in sents if x.endswith("?") and len(x) > 8]
    about = " ".join(x for x in sents if not x.endswith("?"))
    return about, asks


class RefMap:
    def __init__(self, mapnet, embedder, reps=4, beam=4, file_beam=6, explore=1, margin=0.03,
                 widen=2, max_new=4, ref_weight=0.5, expand_weight=0.5):
        self.m, self.emb = mapnet, embedder
        self.expand_weight = expand_weight  # how much the translated terms count in the question vector
        self.reps, self.beam, self.file_beam = reps, beam, file_beam
        self.explore, self.margin, self.widen, self.max_new = explore, margin, widen, max_new
        self.ref_weight = ref_weight  # how much the best reference's match counts in passage ranking
        self.refs: dict[int, list[Ref]] = {}       # zone (group node id) -> references
        self._registered: dict[str, tuple] = {}     # target path -> (signature, [(zone id, Ref)])
        self._content: dict[str, tuple] = {}        # target path -> (signature, passage vecs)
        self._version, self._ex_v, self._ex = 0, -1, {}
        self.vectors = 0
        self.files_embedded = 0
        self.passages_embedded = 0
        self.embed_seconds = 0.0
        self.placed_away = 0  # references registered outside their file's own folder

    # ------------------------------------------------------------ embedding
    def _embed(self, texts, kind="document"):
        t0 = time.perf_counter()
        V = np.asarray(self.emb.embed(list(texts), kind), dtype=np.float32)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        self.embed_seconds += time.perf_counter() - t0
        return V

    @staticmethod
    def _rel(path):
        return path[len("_top/"):] if path.startswith("_top/") else path

    @staticmethod
    def _public(src):
        return EVERYONE in src.acl

    # ------------------------------------------------------------ the zones
    def _exemplars(self, zone):
        """A few vectors standing for everything registered in and under a zone."""
        if self._ex_v != self._version:
            self._ex, self._ex_v = {}, self._version
        hit = self._ex.get(id(zone))
        if hit is not None:
            return hit
        parts = [r.vec[None, :] for r in self.refs.get(id(zone), [])]
        for ch in zone.children.values():
            if ch.kind == "group":
                e = self._exemplars(ch)
                if e is not None:
                    parts.append(e)
        out = _cluster(np.vstack(parts), self.reps) if parts else None
        self._ex[id(zone)] = out
        return out

    # ------------------------------------------------------------- walking
    def _walk(self, qv, roles=None, public_only=False):
        """Walk the zones toward qv. Returns (references met, comparisons, route)."""
        m = self.m
        met, cost, trace = [], 0, []
        frontier = [(m.root, -1.0)]
        while frontier:
            cands = []
            for zone, parent_score in frontier:
                for r in self.refs.get(id(zone), []):
                    met.append((float(r.vec @ qv), r))
                    cost += 1
                for ch in zone.children.values():
                    if ch.kind != "group" or not m._allowed(ch, roles):
                        continue
                    if public_only and EVERYONE not in ch.acl:
                        continue
                    ex = self._exemplars(ch)
                    if ex is None:
                        continue
                    cost += len(ex)
                    cands.append((float((ex @ qv).max()), ch, parent_score))
            if not cands:
                break
            cands.sort(key=lambda x: -x[0])
            width = self.beam
            if cands[0][0] < max(p for _, _, p in cands):
                width *= self.widen  # the match stopped rising: look wider here
            picked = cands[:width] + cands[width:width + self.explore]
            trace.append([z.path for _, z, _ in picked])
            frontier = [(z, s) for s, z, _ in picked]
        return met, cost, trace

    # ------------------------------------------------------------- building
    def _signature(self, src):
        return (len(src.doc_ids), hash(tuple(src.texts)), getattr(src, "file_card", ""))

    def _label(self, src):
        outline = "; ".join(s for s in dict.fromkeys(src.sec) if s)[:600]
        return f"{self._rel(src.path)}\n{outline}"

    def _claims(self, src):
        about, asks = split_card(getattr(src, "file_card", ""))
        out = [("label", self._label(src))]
        if about:
            out.append(("about", about))
        out += [("ask", a) for a in asks]
        return out

    def _add(self, zone, ref):
        ref.zone = zone
        self.refs.setdefault(id(zone), []).append(ref)
        return (id(zone), ref)

    def _unregister(self, path):
        for zid, ref in self._registered.pop(path, (None, []))[1]:
            self.refs[zid] = [r for r in self.refs.get(zid, []) if r is not ref]
        self._version += 1

    def register(self, src, place=True):
        """Add a source to the map: its label and description in its own folder's
        zone; each question it answers wherever that question resolves to."""
        sig = self._signature(src)
        cur = self._registered.get(src.path)
        if cur is not None and cur[0] == sig:
            return 0
        if cur is not None:
            self._unregister(src.path)
        claims = self._claims(src)
        V = self._embed([t for _, t in claims])
        self.vectors += len(claims)
        home = src.parent
        placed = []
        for (kind, text), v in zip(claims, V):
            # every claim is kept at home, so the file is always findable from its own folder
            placed.append(self._add(home, Ref(src.path, text, v, kind, True)))
            if place and kind == "ask" and self._public(src):
                zone = self._resolve(v, src.path)
                if zone is not None and zone is not home:
                    placed.append(self._add(zone, Ref(src.path, text, v, kind, False)))
                    self.placed_away += 1
        self._registered[src.path] = (sig, placed)
        self._version += 1
        return len(claims)

    def _resolve(self, v, target):
        """Where does this claim belong? Next to the closest existing reference
        to a DIFFERENT, public file, found by walking the map like a question."""
        met, _, _ = self._walk(v, public_only=True)
        met = [(s, r) for s, r in met if r.target != target]
        if not met:
            return None
        return max(met, key=lambda x: x[0])[1].zone

    def build(self, log=None):
        """Register every source not registered yet (or changed). Home references
        first, so later questions have something to resolve against."""
        srcs = [s for s in self.m.sources() if self._registered.get(s.path, (None,))[0] != self._signature(s)]
        if not srcs:
            return 0
        step = 40  # files per embedding request (many texts at once; a request per file is slow)
        for a in range(0, len(srcs), step):  # pass 1: everything at home
            part = srcs[a:a + step]
            self._embed([t for s in part for _, t in self._claims(s)])  # warms the embedder's cache
            for s in part:
                self.register(s, place=False)
            if log and (a // step) % 5 == 4:
                log(f"    ref-map: registered {min(a + step, len(srcs)):,}/{len(srcs):,} files")
        # pass 2: move each question to where it resolves (walking against pass 1)
        moved = 0
        for i, s in enumerate(srcs, 1):
            for zid, ref in list(self._registered[s.path][1]):
                if ref.kind != "ask" or not self._public(s):
                    continue
                zone = self._resolve(ref.vec, s.path)
                if zone is not None and id(zone) != zid:
                    away = Ref(ref.target, ref.text, ref.vec, ref.kind, False)
                    self._registered[s.path][1].append(self._add(zone, away))
                    self.placed_away += 1
                    moved += 1
            if log and i % 100 == 0:
                log(f"    ref-map: placed the questions of {i:,}/{len(srcs):,} files")
        self._version += 1
        return len(srcs)

    # ------------------------------------------------------------- reading
    def _arrive(self, src):
        sig = (len(src.doc_ids), hash(tuple(src.texts)))
        cur = self._content.get(src.path)
        if cur is None or cur[0] != sig:
            cur = (sig, self._embed(src.texts))
            self._content[src.path] = cur
            self.files_embedded += 1
            self.passages_embedded += len(src.texts)
        return cur[1]

    def _arrived(self, src):
        cur = self._content.get(src.path)
        return cur is not None and cur[0] == (len(src.doc_ids), hash(tuple(src.texts)))

    def _query_vector(self, query, translator=None):
        """The question as a vector. With a translator, the vector also carries the
        technical terms the right page would use (everyday words -> the page's words)."""
        qv = self._embed([query], "query")[0]
        terms = ""
        if translator is not None:
            terms = translator(query)
            if terms:
                q2 = self._embed([f"{query}\nRelated terms: {terms}"], "query")[0]
                qv = _unit((1 - self.expand_weight) * qv + self.expand_weight * q2)
        return qv, terms

    def _rank(self, qv, roles):
        """Walk the zones; the files whose references were met, best first, and the
        first `file_beam` of them that the caller may open."""
        met, cost, trace = self._walk(qv, roles)
        best: dict[str, float] = {}
        for s, r in met:
            if s > best.get(r.target, -2.0):
                best[r.target] = s
        ranked = sorted(best.items(), key=lambda x: -x[1])
        chosen = []
        for path, s in ranked:
            try:
                src = self.m.find(path)
            except KeyError:
                continue
            if self.m._allowed(src, roles):  # never return what the caller can't open
                chosen.append((s, src))
            if len(chosen) >= self.file_beam:
                break
        return ranked, chosen, cost, trace

    def search(self, query, k=5, roles=None, translator=None, **_):
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        qv, _terms = self._query_vector(query, translator)
        _ranked, chosen, cost, trace = self._rank(qv, roles)
        return self._read(chosen, qv, k, cost, trace, t0)

    # ---------------------------------------------------------- diagnostics
    def _file_of(self, path):
        try:
            return self.m.find(path).doc_ids[0].split("#")[0]
        except (KeyError, IndexError):
            return self._rel(path)

    def flat_ranking(self, qv, roles=None):
        """CONTROL: score the question against every reference with no walk at all.
        Files best first as (score, path). Shows what the references can find at best."""
        best: dict[str, float] = {}
        for refs in self.refs.values():
            for r in refs:
                s = float(r.vec @ qv)
                if s > best.get(r.target, -2.0):
                    best[r.target] = s
        out = []
        for path, s in sorted(best.items(), key=lambda x: -x[1]):
            try:
                if self.m._allowed(self.m.find(path), roles):
                    out.append((s, path))
            except KeyError:
                continue
        return out

    def explain(self, query, gold, roles=None, translator=None):
        """Why was (or wasn't) a right file found? `gold` = the expected file names.
        Compares the real walk with the no-walk control, so a miss can be blamed on
        the references themselves or on the walk."""
        roles = set(roles) if roles is not None else None
        gold = set(gold)
        qv, terms = self._query_vector(query, translator)
        flat = self.flat_ranking(qv, roles)
        ranked, chosen, _, trace = self._rank(qv, roles)
        files = {p: self._file_of(p) for _, p in flat}
        flat_rank = next((i + 1 for i, (_, p) in enumerate(flat) if files[p] in gold), None)
        gold_score = next((s for s, p in flat if files[p] in gold), None)
        walk_rank = next((i + 1 for i, (p, _) in enumerate(ranked) if self._file_of(p) in gold), None)
        in_chosen = any(self._file_of(src.path) in gold for _, src in chosen)
        visited = {self.m.root.path} | {z for level in trace for z in level}
        gold_srcs = [s for s in self.m.sources() if self._file_of(s.path) in gold]
        home_visited = any(s.parent.path in visited or s.parent is self.m.root for s in gold_srcs)
        cut, new = False, 0  # chosen, but not read: only max_new unread files are opened per question
        for _, src in chosen:
            if not self._arrived(src):
                if new >= self.max_new:
                    cut = cut or self._file_of(src.path) in gold
                    continue
                new += 1
        top = flat[0] if flat else None
        return dict(q=query, terms=terms, flat_rank=flat_rank,
                    flat_top=files[top[1]] if top else None, flat_top_score=top[0] if top else None,
                    gold_score=gold_score, walk_rank=walk_rank, in_chosen=in_chosen,
                    home_visited=home_visited, cut=cut)

    def search_in(self, query, source_paths, k=5, roles=None, translator=None):
        t0 = time.perf_counter()
        roles = set(roles) if roles is not None else None
        qv, _terms = self._query_vector(query, translator)
        chosen = []
        for p in source_paths:
            try:
                src = self.m.find(p)
            except KeyError:
                continue
            if src.kind == "source" and self.m._allowed(src, roles):
                chosen.append((0.0, src))
        return self._read(chosen, qv, k, 0, [], t0)

    def _read(self, chosen, qv, k, cost, trace, t0):
        new, hits, searched = 0, [], []
        for ref_score, src in chosen:
            if not self._arrived(src):
                if new >= self.max_new:
                    continue
                new += 1
            V = self._arrive(src)
            scores = V @ qv + self.ref_weight * ref_score
            cost += len(scores)
            searched.append(src.path)
            for i in np.argsort(-scores)[:k]:
                hits.append(Hit(src.doc_ids[i], float(scores[i]), src.path, src.texts[i]))
        hits.sort(key=lambda h: -h.score)
        res = Result(hits[:k], cost, searched, trace)
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res

    def describe(self, path):
        """Where is this file referenced from? (for explaining routes)"""
        return [(ref.zone.path if ref.zone is not None else "?", ref.kind, ref.text[:80])
                for _, ref in self._registered.get(path, (None, []))[1]]
