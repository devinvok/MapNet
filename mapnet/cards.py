"""
Agent-written section cards: a navigation layer an agent writes once per file.

For every section of a file, a local model writes a short card:
  - what the section is about, in plain words
  - the questions a person would ask that this section answers, phrased the
    way people talk ("why is my namespace stuck deleting?"), not the way the
    docs are written ("finalizers")

The map uses cards only for routing: to find the file, and to find the right
section inside it. Card text is never handed to the answering model, so a
document that tries to smuggle instructions into its own card gains nothing.

Cards are stored by the hash of the file's content, so a file is only re-carded
when it changes, and a long build can be stopped and resumed at any time.

    python -m mapnet.cards FOLDER                 # card every file (resumable)
    python -m mapnet.cards FOLDER --limit 50      # just the first 50 files
    python -m mapnet.cards FOLDER --only concepts,tasks   # only these top-level folders
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from .folder import SKIP_DIRS, SUPPORTED, chunk_with_sections, read_file

PROMPT = """You are building a navigation index for a documentation file, so that people
can find the right section by asking questions in their own words.

File: {title}

Sections:
{sections}

For EACH section number, write:
- "about": one plain-language sentence saying what the section covers, including
  any key facts, defaults, numbers or names it states.
- "asks": 3 short questions a user might type when this section has their answer.
  Use everyday words and describe symptoms and goals ("my app keeps restarting",
  "how do I stop...") instead of copying the section's technical terms.

Reply with JSON only: {{"cards": [{{"n": 1, "about": "...", "asks": ["...", "...", "..."]}}, ...]}}"""

SECTION_CHARS = 900   # text shown per section
WINDOW_CHARS = 4000   # sections batched into one model call


def content_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


class CardStore:
    def __init__(self, directory):
        self.dir = Path(directory)

    def _path(self, text):
        k = content_key(text)
        return self.dir / k[:2] / f"{k}.json"

    def get(self, text):
        p = self._path(text)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))["cards"]
        except Exception:
            return None

    def put(self, text, cards, meta):
        p = self._path(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(cards=cards, **meta), ensure_ascii=False, indent=1), encoding="utf-8")

    def meta(self, text):
        p = self._path(text)
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except Exception:
            return None

    def count(self):
        return sum(1 for _ in self.dir.rglob("*.json")) if self.dir.exists() else 0


def file_sections(text, title):
    """[(section_path, section_text)] in file order."""
    secs: dict[str, list[str]] = {}
    for chunk, sec, _ in chunk_with_sections(text, title):
        body = chunk.split("\n", 1)[1] if "\n" in chunk else ""
        secs.setdefault(sec, []).append(body)
    return [(s, "\n\n".join(parts)) for s, parts in secs.items()]


def _generate(reader, prompt):
    body = json.dumps({"model": reader.model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(f"{reader.host}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.load(r)["response"])


def make_cards(text, title, reader):
    """Ask the model for one card per section. Returns {section_path: card_text}."""
    secs = [(s, t) for s, t in file_sections(text, title) if t.strip()]
    windows, cur, size = [], [], 0
    for s, t in secs:
        piece = t[:SECTION_CHARS]
        if cur and size + len(piece) > WINDOW_CHARS:
            windows.append(cur)
            cur, size = [], 0
        cur.append((s, piece))
        size += len(piece)
    if cur:
        windows.append(cur)
    cards = {}
    for win in windows:
        listing = "\n\n".join(f"[{i + 1}] {s or '(top of file)'}\n{t}" for i, (s, t) in enumerate(win))
        try:
            out = _generate(reader, PROMPT.format(title=title, sections=listing))
            items = out.get("cards", []) if isinstance(out, dict) else []
        except Exception:
            items = []
        for it in items:
            try:
                n = int(it.get("n", 0)) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= n < len(win):
                about = str(it.get("about", "")).strip()
                asks = " ".join(str(a).strip() for a in (it.get("asks") or [])[:5])
                if about or asks:
                    cards[win[n][0]] = f"{about} {asks}".strip()[:800]
    return cards


def iter_files(root):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if not fn.startswith(".") and p.suffix.lower() in SUPPORTED:
                yield p


def build(folder, store, reader, limit=None, only=None, log=print):
    """Card every file that has no cards yet. Safe to stop with Ctrl+C and rerun."""
    files = list(iter_files(folder))
    if only:
        files = [f for f in files if f.relative_to(folder).parts[0] in only]
    todo = [f for f in files if store.get(read_file(f)) is None]
    log(f"{len(files) - len(todo):,} of {len(files):,} files already have cards.")
    if limit:
        todo = todo[:limit]
    log(f"Carding {len(todo):,} files with {reader.model}. Stop any time with Ctrl+C; "
        f"rerun to continue.")
    t_start, done = time.perf_counter(), 0
    for f in todo:
        text = read_file(f)
        title = f.relative_to(folder).as_posix()
        t0 = time.perf_counter()
        cards = make_cards(text, title, reader)
        secs = time.perf_counter() - t0
        store.put(text, cards, dict(file=title, model=reader.model, seconds=round(secs, 1)))
        done += 1
        rate = (time.perf_counter() - t_start) / done
        eta = rate * (len(todo) - done)
        log(f"  [{done}/{len(todo)}] {title}: {len(cards)} cards in {secs:.0f}s "
            f"(about {eta / 60:.0f} min left)")
    return done, time.perf_counter() - t_start


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    from .llm import OllamaReader
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--store", default=".mapnet-cards", help="where cards are kept")
    ap.add_argument("--model", help="Ollama model (default $MAPNET_MODEL or qwen2.5:3b)")
    ap.add_argument("--limit", type=int, help="card at most N files this run")
    ap.add_argument("--only", help="only these top-level folders, comma-separated")
    args = ap.parse_args()
    reader = OllamaReader(args.model)
    if not reader.available():
        sys.exit(f"Ollama model '{reader.model}' not available. Run: ollama pull {reader.model}")
    try:
        only = [x.strip() for x in args.only.split(",")] if args.only else None
        n, secs = build(Path(args.folder).resolve(), CardStore(args.store), reader, args.limit, only)
        print(f"Done: {n} files in {secs / 60:.1f} min.")
    except KeyboardInterrupt:
        print("\nStopped. Cards written so far are saved; run the same command to continue.")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Map a file when an agent first arrives at it (DNS style: resolve the
# general level first, map the specific level on demand, cache the answer)
# --------------------------------------------------------------------------- #
class LazyCarder:
    """Writes a file's section cards the first time the router lands on it.

    - The top file of a lookup is mapped right away, so the visit that
      arrives there already uses the file's map to reach the exact passage.
    - Other files the lookup touched are queued and mapped in the background.
    - Cards are stored by file content: a revisit costs nothing, a changed file
      is mapped again on its next visit, and the cards also flow up into the
      folder map, so the file gets easier to find from then on.
    """

    def __init__(self, folder_map, store, reader, sync_top=1, background=True, on_mapped=None):
        import queue
        import threading
        self.fm, self.store, self.reader = folder_map, store, reader
        self.sync_top = sync_top
        self.on_mapped = on_mapped  # callback(path, n_cards, seconds, where)
        self.tried: dict[str, object] = {}  # path -> node already handled
        self.files = 0
        self.seconds = 0.0
        self._q = queue.Queue()
        if background:
            threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _rel(path):
        return path[len("_top/"):] if path.startswith("_top/") else path

    def _map_file(self, path, where):
        rel = self._rel(path)
        try:
            text = read_file(self.fm.root / rel)
        except OSError:
            return 0.0
        t0 = time.perf_counter()
        cards = self.store.get(text)
        if cards is None:
            cards = make_cards(text, rel, self.reader)
            self.store.put(text, cards, dict(file=rel, model=self.reader.model,
                                             seconds=round(time.perf_counter() - t0, 1)))
        secs = time.perf_counter() - t0
        with self.fm.lock:
            try:
                node = self.fm.map.find(path)
            except KeyError:
                return secs
            if cards and not node.cards:
                self.fm.map.set_cards(node, cards)
        self.files += 1
        self.seconds += secs
        if self.on_mapped:
            self.on_mapped(rel, len(cards), secs, where)
        return secs

    def _worker(self):
        while True:
            path = self._q.get()
            try:
                self._map_file(path, "background")
            except Exception:
                pass

    def visit(self, paths):
        """Call with the files a lookup routed to, best first. Returns the files
        mapped right now as [(path, seconds)]."""
        mapped = []
        for i, path in enumerate(paths):
            try:
                node = self.fm.map.find(path)
            except KeyError:
                continue
            if node.kind != "source" or node.cards or self.tried.get(path) is node:
                continue
            self.tried[path] = node
            if i < self.sync_top:
                mapped.append((path, self._map_file(path, "on arrival")))
            else:
                self._q.put(path)
        return mapped
