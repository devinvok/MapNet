"""
Keep a MapNet in sync with a real folder on disk.

- Each sub-folder becomes a group node, each file a source, each chunk a document.
- New, changed and deleted files are picked up by polling (no extra dependencies).
- A file named `.access` in a folder (e.g. containing `hr, finance`) restricts that
  folder and everything below it to those roles.
"""
from __future__ import annotations

import csv
import io
import os
import re
import threading
import time
from pathlib import Path

from .core import EVERYONE, MapNet

TEXT_EXT = {".md", ".txt", ".rst", ".log", ".json", ".yaml", ".yml", ".html", ".xml"}
SUPPORTED = TEXT_EXT | {".csv", ".pdf", ".docx"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}


# --------------------------------------------------------------------------- #
# Reading files
# --------------------------------------------------------------------------- #
def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".csv":
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8", errors="replace"))))
        if not rows:
            return ""
        head = rows[0]
        return "\n".join("; ".join(f"{h}: {v}" for h, v in zip(head, r)) for r in rows[1:])
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            return ""
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return ""


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def chunk_with_sections(text: str, title: str, max_chars=700):
    """Split on blank lines and pack paragraphs up to max_chars, like before,
    and also record where in the file each chunk sits: its heading path
    ("Pod Lifecycle > Container restarts") plus any headings inside it.
    The file's own map, one level below the folder map.

    Returns [(chunk_text, section_path, headings_text)]. Chunk text is exactly
    what `chunk` returns, so embeddings computed earlier stay valid."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    stack: list[tuple[int, str]] = []  # (level, heading)
    in_code = False
    out, cur, cur_heads = [], "", []
    cur_size: dict[str, int] = {}  # characters of the current chunk per section

    def path():
        return " > ".join(h for _, h in stack)

    def flush():
        nonlocal cur, cur_heads, cur_size
        if cur:  # a chunk spanning two sections belongs to the one holding most of it
            sec = max(cur_size, key=cur_size.get) if cur_size else ""
            out.append((cur, sec, " > ".join(dict.fromkeys(cur_heads))))
        cur, cur_heads, cur_size = "", [], {}

    for p in paras:
        sec_before = path()
        heads = []
        for line in p.split("\n"):
            if _FENCE.match(line):
                in_code = not in_code
                continue
            m = None if in_code else _HEADING.match(line)
            if m:
                level, h = len(m.group(1)), re.sub(r"\{#[^}]*\}|[`*_]", "", m.group(2)).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, h))
                heads.append(path())
        # a paragraph that opens with a heading belongs to that heading's section
        sec = heads[0] if heads and _HEADING.match(p.split("\n", 1)[0]) else sec_before
        while len(p) > max_chars:  # very long paragraph
            out.append((p[:max_chars], sec, " > ".join(dict.fromkeys([sec] + heads))))
            p = p[max_chars:]
        if cur and len(cur) + len(p) > max_chars:
            flush()
        cur_size[sec] = cur_size.get(sec, 0) + len(p)
        cur_heads += [sec] + heads
        cur = f"{cur}\n\n{p}" if cur else p
    flush()
    if not out:
        return [(f"[{title}]", "", "")]
    return [(f"[{title}]\n{c}", s, h) for c, s, h in out]


def chunk(text: str, title: str, max_chars=700) -> list[str]:
    """Split on blank lines, pack paragraphs up to max_chars. Every chunk is
    prefixed with the file title so it keeps its context."""
    return [c for c, _, _ in chunk_with_sections(text, title, max_chars)]


# --------------------------------------------------------------------------- #
# Folder <-> map
# --------------------------------------------------------------------------- #
class FolderMap:
    def __init__(self, root, mapnet: MapNet | None = None, on_change=None, cards=None):
        self.root = Path(root).resolve()
        self.map = mapnet or MapNet()
        self.cards = cards  # optional CardStore: agent-written section cards, by file content
        self.lock = threading.RLock()
        self.seen: dict[str, tuple[float, int]] = {}  # rel path -> (mtime, size)
        self.on_change = on_change  # callback(kind, rel_path, seconds)
        self._stop = threading.Event()

    # map paths use "/" and must not collide with real names, so files keep
    # their extension and folders are groups
    @staticmethod
    def _map_path(rel: str) -> str:
        return rel.replace(os.sep, "/")

    def _acl_for(self, rel_dir: Path):
        acl = {EVERYONE}
        cur = self.root
        for part in (Path(),) + tuple(Path(*rel_dir.parts[: i + 1]) for i in range(len(rel_dir.parts))):
            f = self.root / part / ".access"
            if f.exists():
                roles = {r.strip() for r in f.read_text(encoding="utf-8").replace("\n", ",").split(",") if r.strip()}
                acl = roles or acl
        return acl

    def _scan(self):
        found = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                p = Path(dirpath) / fn
                if fn.startswith(".") or p.suffix.lower() not in SUPPORTED:
                    continue
                st = p.stat()
                found[str(p.relative_to(self.root))] = (st.st_mtime, st.st_size)
        return found

    def _index(self, rel: str):
        p = self.root / rel
        text = read_file(p)
        rel_path = Path(rel)
        group = list(rel_path.parent.parts) or ["_top"]
        title = rel_path.as_posix()
        parts = chunk_with_sections(text, title)
        docs = [(f"{title}#{i}", c) for i, (c, _, _) in enumerate(parts)]
        node = self.map.register_source(rel_path.name, docs, path=group,
                                        acl=tuple(self._acl_for(rel_path.parent)),
                                        sections=[(s, h) for _, s, h in parts])
        if self.cards is not None:
            cards = self.cards.get(text)
            if cards:
                self.map.set_cards(node, cards)

    def _unindex(self, rel: str):
        rel_path = Path(rel)
        group = list(rel_path.parent.parts) or ["_top"]
        try:
            self.map.remove_source("/".join(group + [rel_path.name]))
        except KeyError:
            pass

    def sync(self):
        """One pass: apply every added, changed and deleted file. Returns changes."""
        found = self._scan()
        changes = []
        with self.lock:
            for rel in sorted(set(self.seen) - set(found)):
                t0 = time.perf_counter()
                self._unindex(rel)
                changes.append(("deleted", rel, time.perf_counter() - t0))
            for rel, sig in sorted(found.items()):
                old = self.seen.get(rel)
                if old == sig:
                    continue
                t0 = time.perf_counter()
                if old is not None:
                    self._unindex(rel)
                try:
                    self._index(rel)
                except Exception as e:  # unreadable file: skip, keep going
                    changes.append(("error", f"{rel}: {e}", 0.0))
                    continue
                changes.append(("changed" if old else "added", rel, time.perf_counter() - t0))
            self.seen = found
        if self.on_change:
            for c in changes:
                self.on_change(*c)
        return changes

    def watch(self, interval=1.0):
        """Start a background thread that keeps the map in sync."""
        def loop():
            while not self._stop.wait(interval):
                try:
                    self.sync()
                except Exception:
                    pass
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop.set()
