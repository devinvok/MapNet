"""
Ask questions about a folder, privately, with a small local model.
The folder is watched: add, edit or delete files and the next question sees it.

    python ask.py examples/company
    python ask.py ~/Documents/work --role hr --model llama3.2:3b

Commands inside: /map  /notes  /role <names>  /quit
"""
import argparse
import sys
import time

from mapnet.assistant import Assistant
from mapnet.llm import ExtractiveReader, OllamaReader

DIM, BOLD, GREEN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    DIM = BOLD = GREEN = YELLOW = RESET = ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", help="Ollama model (default: $MAPNET_MODEL or qwen2.5:3b)")
    ap.add_argument("--role", default="employee", help="comma-separated roles for permissions")
    ap.add_argument("--no-model", action="store_true", help="skip the LLM, show best passage")
    ap.add_argument("--notes", default=".mapnet-notes.json", help="where agent notes are kept")
    args = ap.parse_args()

    if args.no_model:
        reader = ExtractiveReader()
    else:
        reader = OllamaReader(args.model)
        if not reader.available():
            print(f"{YELLOW}Ollama model '{reader.model}' not found at {reader.host}.{RESET}\n"
                  f"Install Ollama (ollama.com), then: ollama pull {reader.model}\n"
                  f"Continuing without a model (showing the best passage instead).\n")
            reader = ExtractiveReader()

    t0 = time.perf_counter()
    a = Assistant(args.folder, reader=reader, notes_path=args.notes)
    n_src = len(a.map.sources())
    print(f"{BOLD}MapNet{RESET} mapped {n_src} files ({a.map.root.n_docs} passages) "
          f"in {time.perf_counter() - t0:.1f}s. Model: {reader.model}. Watching for changes.\n")

    def on_change(kind, rel, secs):
        print(f"\n{GREEN}[map] {kind}: {rel} ({secs * 1000:.0f} ms){RESET}\n> ", end="", flush=True)

    a.fm.on_change = on_change
    a.fm.watch(interval=1.0)
    roles = [r.strip() for r in args.role.split(",")]

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/map":
            with a.fm.lock:
                a.map.show(max_sources=10)
            continue
        if q == "/notes":
            for n in a.notes.notes.values():
                print(f"  #{n.id} [{n.author}, strength {n.strength}] {n.about!r} -> {n.target}")
            print(f"  ({len(a.notes.notes)} notes)")
            continue
        if q.startswith("/role"):
            roles = [r.strip() for r in q[5:].split(",") if r.strip()] or ["employee"]
            print(f"  roles: {roles}")
            continue

        print(GREEN, end="")
        t = a.ask(q, roles=roles, stream=lambda s: print(s, end="", flush=True))
        print(RESET)
        how = f"note #{t.note_used}" if t.via == "note" else "map"
        route = " -> ".join(", ".join(level) for level in t.route) if t.via != "note" else "straight to file"
        print(f"{DIM}via {how} | {route} | opened {t.files_opened} of {n_src} files | "
              f"lookup {t.retrieve_ms:.0f} ms, answer {t.answer_s:.1f}s"
              f"{f' | note #{t.note_written} left for next time' if t.note_written else ''}{RESET}\n")
        n_src = len(a.map.sources())
    a.fm.stop()


if __name__ == "__main__":
    main()
