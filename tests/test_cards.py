import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from mapnet import MapNet
from mapnet.cards import CardStore, build, file_sections, make_cards
from mapnet.folder import FolderMap, chunk, chunk_with_sections

ROOT = Path(__file__).resolve().parent.parent

DOC = """# Finalizers

Finalizers are namespaced keys that tell Kubernetes to wait until specific
conditions are met before it fully deletes resources.

## How finalizers work

When you delete an object that has finalizers, the API server marks it for
deletion and it stays in a Terminating state until the finalizers are removed.

```bash
# this is a comment inside code, not a heading
kubectl get ns
```

## Owner references

Owner references describe how groups of objects are related.
"""


def test_sections_follow_headings_and_ignore_code():
    parts = chunk_with_sections(DOC, "f.md", max_chars=120)
    secs = [s for _, s, _ in parts]
    assert "Finalizers > How finalizers work" in secs
    assert "Finalizers > Owner references" in secs
    # a chunk spanning two sections is labelled by the section holding most of it,
    # and still lists every heading it contains
    mixed = chunk_with_sections(DOC, "f.md", max_chars=400)
    owner = next((s, h) for c, s, h in mixed if "Owner references describe" in c)
    assert "Owner references" in owner[1]
    assert not any("comment inside code" in s for s in secs)
    assert [c for c, _, _ in parts] == chunk(DOC, "f.md", max_chars=120)  # same chunk text


def test_section_walk_picks_the_right_part_of_a_file():
    m = MapNet(section_weight=1.0)
    parts = chunk_with_sections(DOC, "f.md", max_chars=200)
    m.register_source("f.md", [(f"f#{i}", c) for i, (c, _, _) in enumerate(parts)], path=["docs"],
                      sections=[(s, h) for _, s, h in parts])
    hit = m.search("owner references between objects").hits[0]
    assert "Owner references" in hit.text


class FakeOllama(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.calls += 1
        n = body["prompt"].count("\n[")  # sections listed in this window
        if "inalizer" in body["prompt"]:
            about, asks = ("explains why a namespace is stuck deleting forever",
                           ["why is my namespace stuck", "namespace won't delete"])
        else:
            about, asks = "how to create and organise namespaces", ["how do I make a new namespace"]
        cards = [{"n": i + 1, "about": about, "asks": asks} for i in range(n)]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"response": json.dumps({"cards": cards})}).encode())


class Reader:
    model = "fake"

    def __init__(self, host):
        self.host = host


def test_cards_route_by_everyday_words_and_stay_out_of_passages(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        reader = Reader(f"http://127.0.0.1:{srv.server_port}")
        cards = make_cards(DOC, "f.md", reader)
        assert set(cards) == {s for s, t in file_sections(DOC, "f.md") if t.strip()}

        # a folder with the finalizers page and a distractor
        folder = tmp_path / "docs"
        (folder / "a").mkdir(parents=True)
        (folder / "a" / "finalizers.md").write_text(DOC, encoding="utf-8")
        (folder / "a" / "other.md").write_text("# Namespaces\n\nNamespaces group objects. Create a namespace "
                                               "with kubectl create namespace.\n", encoding="utf-8")
        store = CardStore(tmp_path / "cards")
        n, _ = build(folder, store, reader, log=lambda *a: None)
        assert n == 2 and store.count() == 2
        n_again, _ = build(folder, store, reader, log=lambda *a: None)
        assert n_again == 0  # resumable: nothing left to do

        plain = FolderMap(folder, mapnet=MapNet(section_weight=1.0))
        plain.sync()
        carded = FolderMap(folder, mapnet=MapNet(section_weight=1.0), cards=store)
        carded.sync()
        q = "why is my namespace stuck deleting forever"
        src = carded.map.find("a/finalizers.md")
        assert src.cards
        top_plain = plain.map.search(q).hits[0].source
        top_carded = carded.map.search(q).hits[0]
        assert top_plain == "a/other.md"           # word overlap alone picks the wrong page
        assert top_carded.source == "a/finalizers.md"  # the cards bridge the vocabulary gap
        assert "stuck deleting" not in top_carded.text  # card text is never a passage

        # cards are removed cleanly with the file
        carded.map.set_cards(src, {})
        assert carded.map.search(q).hits[0].source == "a/other.md"
        carded.map.remove_source(src)
        assert carded.map.root.n_docs == plain.map.find("a/other.md").n_docs
    finally:
        srv.shutdown()


def test_harness_with_cards_and_subset(tmp_path):
    """End to end: the real-test harness picks up a card store and the --only subset."""
    import subprocess
    import sys

    data = tmp_path / "data"
    store = CardStore(data / "cards")
    store.put((ROOT / "examples/company/legal/data-retention.md").read_text(encoding="utf-8"),
              {"Data Retention Rules": "how long we keep old applicant resumes"}, dict(seconds=1.0))
    out = subprocess.run(
        [sys.executable, "-m", "realtest.run", "--corpus", str(ROOT / "examples/company"),
         "--questions", str(ROOT / "realtest/questions/sample-company.json"), "--retrieval-only",
         "--only", "hr,legal,finance", "--data", str(data)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert out.returncode == 0, out.stderr
    assert "map+cards" in out.stdout and "map+sections" in out.stdout
    assert "1 of" in out.stdout  # one file carded
    shutil.rmtree(ROOT / "realtest" / "results", ignore_errors=True)
