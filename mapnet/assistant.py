"""
The full loop: check agent notes -> otherwise route through the map -> hand
the few best excerpts to a reader model -> leave a note if the answer was solid.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .core import MapNet
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
    mapped: list = field(default_factory=list)  # files mapped on arrival: [(path, seconds)]
    retrieve_ms: float = 0.0
    answer_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


class Assistant:
    def __init__(self, folder, reader=None, notes_path=None, confident=0.12, k=4, cards=None,
                 map_files_with=None, on_mapped=None, background=True):
        """cards: a CardStore. map_files_with: a model (OllamaReader) that maps each
        file the first time a lookup arrives there; its cards go into `cards`."""
        # walk folders, then the headings (and agent cards, if any) inside each file
        self.fm = FolderMap(folder, mapnet=MapNet(section_weight=1.0), cards=cards)
        self.fm.sync()
        self.map = self.fm.map
        self.notes = NoteBoard(self.map)
        self.notes_path = notes_path
        if notes_path:
            self.notes.load(notes_path)
        self.reader = reader or pick_reader()
        self.lazy = None
        if map_files_with is not None:
            from .cards import CardStore, LazyCarder
            store = cards if cards is not None else CardStore(".mapnet-cards")
            self.fm.cards = store
            self.lazy = LazyCarder(self.fm, store, map_files_with, on_mapped=on_mapped,
                                   background=background)
        self.confident = confident  # top excerpt score that counts as "found it"
        self.k = k

    def retrieve(self, question, roles=None, agent="user", use_notes=True, write_notes=True):
        t0 = time.perf_counter()
        with self.fm.lock:
            self.notes.prune_missing()
            via, res, used, written, mapped = "map", None, None, None, []
            if use_notes:
                for note, _ in self.notes.lookup(question, roles)[:2]:
                    if self.lazy:
                        mapped += self.lazy.visit([note.target])
                    r = self.map.search_in(question, [note.target], k=self.k, roles=roles)
                    if r.hits and r.hits[0].score >= self.confident:
                        self.notes.confirm(note)
                        via, res, used = "note", r, note.id
                        break
                    self.notes.reject(note)
            if res is None:
                res = self.map.search(question, k=self.k, roles=roles, min_score=self.confident)
                if self.lazy and res.sources_searched:
                    # arrived at the files: map the top one now, then walk its map
                    new = self.lazy.visit(res.sources_searched)
                    if new:
                        mapped += new
                        again = self.map.search_in(question, res.sources_searched, k=self.k, roles=roles)
                        again.trace, again.cost = res.trace, res.cost + again.cost
                        again.escalated = res.escalated
                        res = again
                if not res.hits or res.hits[0].score < self.confident:
                    via = "none" if not res.hits else "map"
                elif write_notes:
                    n = self.notes.add(question, res.hits[0].source, author=agent, task=question,
                                       roles=roles)
                    written = n.id if n else None
            if self.notes_path:
                self.notes.save(self.notes_path)
        self._last_mapped = mapped
        return res, via, used, written, (time.perf_counter() - t0) * 1000

    def ask(self, question, roles=None, agent="user", stream=None, use_notes=True):
        res, via, used, written, ms = self.retrieve(question, roles, agent, use_notes)
        mapped = getattr(self, "_last_mapped", [])
        hits = res.hits[: self.k]
        a = self.reader.answer(question, hits, stream=stream)
        return Turn(question, a.text, via, [h.doc_id.split("#")[0] for h in hits],
                    len(res.sources_searched), res.cost, route=res.trace, note_used=used,
                    note_written=written, mapped=mapped, retrieve_ms=ms, answer_s=a.seconds,
                    tokens_in=a.input_tokens, tokens_out=a.output_tokens, model=a.model)
