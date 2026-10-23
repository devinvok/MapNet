"""
Side-by-side: small local model + map  vs  large cloud model.

Runs the question set in examples/questions.json, then drops the new files from
examples/drop-in/ into the folder and asks about them immediately (the live
update test). For each setup it reports answers correct, right file retrieved,
tokens per question, time per question and, if you give prices, cost.

    python compare.py                                  # local model only
    ANTHROPIC_API_KEY=... python compare.py --cloud    # add the cloud model
    python compare.py --cloud --price-in 3 --price-out 15   # USD per million tokens

Setups:
  local+map    small model on your machine reads the excerpts the map picked
  cloud+map    cloud model reads the same excerpts (same retrieval, bigger reader)
  cloud+all    cloud model gets every file in the prompt, no map at all

Note: the cloud setups send your excerpts / files to the cloud provider.
"""
import argparse
import sys
import json
import shutil
import tempfile
import time
from pathlib import Path

from mapnet.assistant import Assistant
from mapnet.folder import read_file
from mapnet.llm import AnthropicReader, ExtractiveReader, OllamaReader, build_prompt

for _stream in (sys.stdout, sys.stderr):  # Windows consoles can't show every character
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

HERE = Path(__file__).parent
LIVE_QUESTIONS = [
    {"q": "What will the Pro plan cost per robot from January 2027?", "expect": ["95"],
     "file": "sales/pricing.md"},
    {"q": "Is there an ongoing checkout incident, and who is the incident commander?",
     "expect": ["avi"], "file": "engineering/2026-10-23-checkout-outage.md"},
]


def all_files_prompt(folder, question, roles):
    """The no-map baseline: paste every readable file the caller may see."""
    a_tmp = Assistant(folder, reader=ExtractiveReader())
    parts = []
    for src in a_tmp.map.sources():
        if a_tmp.map._allowed(src, set(roles)):
            rel = src.path[len("_top/"):] if src.path.startswith("_top/") else src.path
            parts.append(f"--- file {rel} ---\n{read_file(Path(folder) / rel)}")
    return "Files:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"


def run(name, a, reader, questions, folder, all_files=False):
    rows = []
    for x in questions:
        roles = x.get("roles", ["employee"])
        res, *_ = a.retrieve(x["q"], roles, use_notes=False, write_notes=False)
        hits = res.hits[: a.k]
        t0 = time.perf_counter()
        if all_files:
            ans = reader.answer(x["q"], prompt=all_files_prompt(folder, x["q"], roles))
        else:
            ans = reader.answer(x["q"], hits)
        secs = time.perf_counter() - t0
        text = ans.text.lower()
        rows.append(dict(
            q=x["q"], answer=ans.text,
            correct=all(e.lower() in text for e in x["expect"]),
            retrieved=x["file"] in [h.doc_id.split("#")[0] for h in hits],
            tokens_in=ans.input_tokens, tokens_out=ans.output_tokens, seconds=secs))
        mark = "ok " if rows[-1]["correct"] else "NO "
        print(f"  [{name}] {mark} {x['q'][:60]}")
    return rows


def summarize(name, rows, price_in, price_out):
    n = len(rows)
    tin = sum(r["tokens_in"] for r in rows) / n
    tout = sum(r["tokens_out"] for r in rows) / n
    cost = ""
    if price_in is not None and "cloud" in name:
        cost = f"{(tin * price_in + tout * price_out) / 1e6 * 1000:.2f} USD per 1,000 questions"
    elif "local" in name:
        cost = "no per-question fee (your hardware)"
    return (f"{name:11} correct {sum(r['correct'] for r in rows):>2}/{n}  "
            f"right file {sum(r['retrieved'] for r in rows):>2}/{n}  "
            f"tokens in/out {tin:>6.0f}/{tout:<4.0f} "
            f"{sum(r['seconds'] for r in rows) / n:>5.1f}s/q  {cost}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="local Ollama model")
    ap.add_argument("--cloud", action="store_true", help="also run the cloud model")
    ap.add_argument("--cloud-model", help="cloud model id (default $MAPNET_CLOUD_MODEL)")
    ap.add_argument("--price-in", type=float, help="cloud USD per million input tokens")
    ap.add_argument("--price-out", type=float, help="cloud USD per million output tokens")
    ap.add_argument("--out", default="compare-results.json")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp()) / "company"
    shutil.copytree(HERE / "examples" / "company", tmp)
    questions = json.loads((HERE / "examples" / "questions.json").read_text(encoding="utf-8"))

    setups = []
    local = OllamaReader(args.model)
    if local.available():
        setups.append(("local+map", local, False))
    else:
        print(f"Ollama model '{local.model}' not available; using the no-model extractive reader "
              f"as the local setup. (Install Ollama and run: ollama pull {local.model})\n")
        setups.append(("extractive", ExtractiveReader(), False))
    if args.cloud:
        cloud = AnthropicReader(args.cloud_model)
        if cloud.available():
            setups += [("cloud+map", cloud, False), ("cloud+all", cloud, True)]
        else:
            print("ANTHROPIC_API_KEY not set; skipping the cloud setups.\n")

    a = Assistant(tmp, reader=ExtractiveReader())
    results = {}
    print("Phase 1: questions about the existing files")
    for name, reader, all_files in setups:
        results[name] = run(name, a, reader, questions, tmp, all_files)

    print("\nPhase 2: drop new files in, ask immediately")
    t0 = time.perf_counter()
    shutil.copy(HERE / "examples/drop-in/pricing.md", tmp / "sales/pricing.md")
    shutil.copy(HERE / "examples/drop-in/2026-10-23-checkout-outage.md", tmp / "engineering")
    changes = a.fm.sync()
    print(f"  map updated in {(time.perf_counter() - t0) * 1000:.0f} ms: "
          + ", ".join(f"{k} {p}" for k, p, _ in changes))
    for name, reader, all_files in setups:
        results[name] += run(name, a, reader, LIVE_QUESTIONS, tmp, all_files)

    print(f"\nResults ({len(questions) + len(LIVE_QUESTIONS)} questions, "
          f"{len(LIVE_QUESTIONS)} of them about files added seconds earlier)")
    for name, rows in results.items():
        print("  " + summarize(name, rows, args.price_in, args.price_out or 0))
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nPer-question answers saved to {args.out}")
    shutil.rmtree(tmp.parent)


if __name__ == "__main__":
    main()
