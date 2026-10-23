"""
The full loop: check agent notes -> otherwise route through the map -> hand
the few best excerpts to a reader model -> leave a note if the answer was solid.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .folder import FolderMap
from .llm import pick_reader
from .notes import NoteBoard


@dataclass
class Turn:
    question: str
    answer: str
    via: str  # "note" or "map" or "none"
    sources: list[str]
    files_opened: int
    comparisons: int
    route: list[list[str]] = field(default_factory=list)
    note_used: int | None = None
    note_written: int | None = None
    retrieve_ms: float = 0.0
    answer_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


class Assistant:
    def __init__(self, folder, reader=None, notes_path=None, confident=0.12, k=4):
        self.fm = FolderMap(folder)
        self.fm.sync()
        self.map = self.fm.map
        self.notes = NoteBoard(self.map)
        self.notes_path = notes_path
        if notes_path:
            self.notes.load(notes_path)
        self.reader = reader or pick_reader()
        self.confident = confident  # top excerpt score that counts as "found it"
        self.k = k

    def retrieve(self, question, roles=None, agent="user", use_notes=True, write_notes=True):
        t0 = time.perf_counter()
        with self.fm.lock:
            self.notes.prune_missing()
            via, res, used, written = "map", None, None, None
            if use_notes:
                for note, _ in self.notes.lookup(question, roles)[:2]:
                    r = self.map.search_in(question, [note.target], k=self.k, roles=roles)
                    if r.hits and r.hits[0].score >= self.confident:
                        self.notes.confirm(note)
                        via, res, used = "note", r, note.id
                        break
                    self.notes.reject(note)
            if res is None:
                res = self.map.search(question, k=self.k, roles=roles, min_score=self.confident)
                if not res.hits or res.hits[0].score < self.confident:
                    via = "none" if not res.hits else "map"
                elif write_notes:
                    n = self.notes.add(question, res.hits[0].source, author=agent, task=question,
                                       roles=roles)
                    written = n.id if n else None
            if self.notes_path:
                self.notes.save(self.notes_path)
        return res, via, used, written, (time.perf_counter() - t0) * 1000

    def ask(self, question, roles=None, agent="user", stream=None, use_notes=True):
        res, via, used, written, ms = self.retrieve(question, roles, agent, use_notes)
        hits = res.hits[: self.k]
        a = self.reader.answer(question, hits, stream=stream)
        return Turn(question, a.text, via, [h.doc_id.split("#")[0] for h in hits],
                    len(res.sources_searched), res.cost, res.trace, used, written, ms,
                    a.seconds, a.input_tokens, a.output_tokens, a.model)
