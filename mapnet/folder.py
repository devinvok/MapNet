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


def chunk(text: str, title: str, max_chars=700) -> list[str]:
    """Split on blank lines, pack paragraphs up to max_chars. Every chunk is
    prefixed with the file title so it keeps its context."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > max_chars:  # very long paragraph
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return [f"[{title}]\n{c}" for c in chunks] or [f"[{title}]"]


# --------------------------------------------------------------------------- #
# Folder <-> map
# --------------------------------------------------------------------------- #
class FolderMap:
    def __init__(self, root, mapnet: MapNet | None = None, on_change=None):
        self.root = Path(root).resolve()
        self.map = mapnet or MapNet()
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
        chunks = chunk(text, title)
        docs = [(f"{title}#{i}", c) for i, c in enumerate(chunks)]
        self.map.register_source(rel_path.name, docs, path=group, acl=tuple(self._acl_for(rel_path.parent)))

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
