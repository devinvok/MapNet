"""
Agent notes: trail markers that agents leave for the agents after them.

Design rules (the lessons of agents using a shared message board as a
command channel):

1. A note is a POINTER, never an instruction. It has exactly two content
   fields: what it is about, and which source in the map holds it. There is no
   field for "what to do", and note text is never put into a model's prompt.
   Following a note only changes which file gets opened.
2. A note can only point at a source that exists in the map, and is only shown
   to callers allowed to open that source.
3. Every use is a test. If the target really answers the question the note gets
   stronger; if not it gets weaker, and at zero it is deleted.
4. Notes record who wrote them, for what task, and when they were last confirmed.
   Notes whose target disappears are dropped.
5. A note reflects what its writer could see, so it is only used by callers with
   the same roles. (An employee's best guess about payroll is not useful to HR,
   who can see the real payroll file.)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

MAX_ABOUT = 200


@dataclass
class Note:
    id: int
    about: str
    target: str
    author: str
    task: str = ""
    roles: list | None = None  # roles of the writer; None = unrestricted caller
    created: float = field(default_factory=time.time)
    last_confirmed: float | None = None
    strength: float = 1.0
    uses: int = 0
    failures: int = 0


def _scope(roles):
    return sorted(set(roles)) if roles is not None else None


class NoteBoard:
    def __init__(self, mapnet, min_similarity=0.35, max_strength=3.0):
        self.map = mapnet
        self.notes: dict[int, Note] = {}
        self._vec = {}
        self._next = 1
        self.min_similarity = min_similarity
        self.max_strength = max_strength
        self.log: list[str] = []

    # ----------------------------------------------------------------- write
    def add(self, about: str, target: str, author: str, task: str = "", roles=None) -> Note | None:
        about = " ".join(about.split())[:MAX_ABOUT]
        try:
            node = self.map.find(target)
        except KeyError:
            self.log.append(f"rejected note from {author}: target {target!r} is not in the map")
            return None
        if node.kind != "source":
            self.log.append(f"rejected note from {author}: {target!r} is a folder, not a source")
            return None
        # same pointer already exists for a similar question: reinforce instead of duplicating
        for n, sim in self.lookup(about, roles=roles, include_weak=True):
            if n.target == target and sim > 0.8:
                return n
        n = Note(self._next, about, target, author, task, _scope(roles))
        self._next += 1
        self.notes[n.id] = n
        self._vec[n.id] = self.map._encode_query(about)
        self.log.append(f"note #{n.id} by {author}: '{about}' -> {target}")
        return n

    # ------------------------------------------------------------------ read
    def lookup(self, query, roles=None, include_weak=False):
        if not self.notes:
            return []
        q = self.map._encode_query(query)
        scope = _scope(roles)
        roles = set(roles) if roles is not None else None
        out = []
        for n in self.notes.values():
            if n.roles != scope:
                continue
            sim = float((self._vec[n.id] @ q.T).toarray()[0, 0])
            if sim < self.min_similarity and not include_weak:
                continue
            try:
                node = self.map.find(n.target)
            except KeyError:
                continue
            if not self.map._allowed(node, roles):
                continue
            out.append((n, sim))
        out.sort(key=lambda x: -(x[1] * x[0].strength))
        return out

    # ----------------------------------------------------------- verification
    def confirm(self, note: Note):
        note.uses += 1
        note.strength = min(self.max_strength, note.strength + 0.5)
        note.last_confirmed = time.time()

    def reject(self, note: Note, reason="target did not answer"):
        note.uses += 1
        note.failures += 1
        note.strength -= 1.0
        if note.strength <= 0:
            self.notes.pop(note.id, None)
            self._vec.pop(note.id, None)
            self.log.append(f"note #{note.id} by {note.author} deleted ({reason}, "
                            f"{note.failures} failed use{'s' if note.failures > 1 else ''})")

    def prune_missing(self):
        for n in list(self.notes.values()):
            try:
                self.map.find(n.target)
            except KeyError:
                self.notes.pop(n.id)
                self._vec.pop(n.id, None)
                self.log.append(f"note #{n.id} dropped: {n.target} no longer exists")

    # ----------------------------------------------------------- persistence
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(n) for n in self.notes.values()], f, indent=1)

    def load(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
        except FileNotFoundError:
            return
        for d in items:
            n = Note(**d)
            self.notes[n.id] = n
            self._vec[n.id] = self.map._encode_query(n.about)
            self._next = max(self._next, n.id + 1)
        self.prune_missing()
