import json
import os
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
         "--only", "hr,legal,finance", "--data", str(data), "--results", str(tmp_path / "results")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=300,
        # Windows pipes default to the console codepage; and no Ollama, even if one is running
        # here, so the stand-in embedder is used and the test does not depend on a real model
        env=dict(os.environ, PYTHONIOENCODING="utf-8", OLLAMA_HOST="http://127.0.0.1:9"))
    assert out.returncode == 0, out.stderr
    assert "map+cards" in out.stdout and "map+sections" in out.stdout
    assert "1 of" in out.stdout  # one file carded


# ---- mapping files on arrival (DNS style), and the full pipeline with fake models
import time  # noqa: E402

from fake_ollama import Handler, start  # noqa: E402


def test_files_are_mapped_on_arrival(tmp_path):
    from mapnet.assistant import Assistant
    from mapnet.llm import ExtractiveReader, OllamaReader

    srv, host = start()
    try:
        folder = tmp_path / "company"
        shutil.copytree(ROOT / "examples" / "company", folder)
        mapped_bg = []
        a = Assistant(folder, reader=ExtractiveReader(), cards=CardStore(tmp_path / "cards"),
                      map_files_with=OllamaReader("qwen2.5:3b", host=host),
                      on_mapped=lambda rel, n, s, where: mapped_bg.append((rel, where)))
        q = "What are the payment terms with Kalmar Metals?"
        t1 = a.ask(q, use_notes=False)
        assert [p for p, _ in t1.mapped] == ["finance/vendors/kalmar-metals.md"]  # top file, right away
        assert a.map.find("finance/vendors/kalmar-metals.md").cards
        assert t1.sources[0] == "finance/vendors/kalmar-metals.md"
        t2 = a.ask(q, use_notes=False)
        assert t2.mapped == []  # cached: a revisit costs nothing
        for _ in range(50):  # the other files that lookup touched get mapped in the background
            if any(w == "background" for _, w in mapped_bg):
                break
            time.sleep(0.1)
        assert any(w == "background" for _, w in mapped_bg)
        # a changed file is mapped again on its next visit
        f = folder / "finance" / "vendors" / "kalmar-metals.md"
        f.write_text(f.read_text(encoding="utf-8") + "\n\nNew: payments go to a new IBAN.\n", encoding="utf-8")
        a.fm.sync()
        t3 = a.ask(q, use_notes=False)
        assert [p for p, _ in t3.mapped] == ["finance/vendors/kalmar-metals.md"]
        # a fresh assistant reuses the stored file maps without calling the model (first let the
        # first assistant's background worker finish, or its calls are counted against the second)
        a.lazy.wait_idle()
        before = Handler.calls.get("/api/generate", 0)
        b = Assistant(folder, reader=ExtractiveReader(), cards=CardStore(tmp_path / "cards"),
                      map_files_with=OllamaReader("qwen2.5:3b", host=host), background=False)
        assert b.ask(q, use_notes=False).mapped == []
        assert Handler.calls.get("/api/generate", 0) == before
        assert b.map.find("finance/vendors/kalmar-metals.md").cards
    finally:
        srv.shutdown()


def test_full_real_test_with_fake_models(tmp_path):
    import subprocess
    import sys

    srv, host = start()
    try:
        env = dict(os.environ, OLLAMA_HOST=host, PYTHONIOENCODING="utf-8")
        out = subprocess.run(
            [sys.executable, "-m", "realtest.run", "--corpus", str(ROOT / "examples/company"),
             "--questions", str(ROOT / "realtest/questions/sample-company.json"),
             "--data", str(tmp_path / "data"), "--results", str(tmp_path / "results")],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600, env=env)
        assert out.returncode == 0, out.stderr[-2000:]
        for name in ["keyword-all", "embed-all", "map+sections", "map+lazy-cards", "map+meaning", "map+embed"]:
            assert f"| {name} |" in out.stdout, name
        assert "files mapped on first visit" in out.stdout
        report = next((tmp_path / "results").iterdir()) / "results.json"
        rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
        assert all("answer" in r for r in rows)  # every method was answered by the (fake) model
    finally:
        srv.shutdown()


def test_file_descriptions_guide_the_router_to_the_file(tmp_path):
    """A file whose words don't match the question is found through its description."""
    from mapnet.llm import OllamaReader

    folder = tmp_path / "docs"
    (folder / "a").mkdir(parents=True)
    (folder / "a" / "finalizers.md").write_text(DOC, encoding="utf-8")
    (folder / "a" / "other.md").write_text("# Namespaces\n\nNamespaces group objects. Create a namespace "
                                           "with kubectl create namespace.\n", encoding="utf-8")
    store = CardStore(tmp_path / "cards")
    store.put_file(DOC, "why a namespace is stuck deleting and won't go away", dict(seconds=2.0))
    store.put_file((folder / "a" / "other.md").read_text(encoding="utf-8"),
                   "how to create and organise namespaces", dict(seconds=2.0))
    fm = FolderMap(folder, mapnet=MapNet(section_weight=1.0, route_weight=0.3), cards=store)
    fm.sync()
    assert fm.map.find("a/finalizers.md").file_card
    q = "why is my namespace stuck deleting"
    assert fm.map.search(q).hits[0].source == "a/finalizers.md"
    assert all("won't go away" not in h.text for h in fm.map.search(q).hits)  # never a passage
    fm.map.set_file_card("a/finalizers.md", "")  # removable
    assert fm.map.find("a/finalizers.md").file_card == ""

    # the CLI-equivalent builder writes descriptions, resumably, with a (fake) model
    srv, host = start()
    try:
        s2 = CardStore(tmp_path / "cards2")
        n, _ = build(folder, s2, OllamaReader("qwen2.5:3b", host=host), files_only=True, log=lambda *a: None)
        assert n == 2 and s2.count_files() == 2 and s2.count() == 0
        assert build(folder, s2, OllamaReader("qwen2.5:3b", host=host), files_only=True,
                     log=lambda *a: None)[0] == 0
    finally:
        srv.shutdown()


def test_full_chain_in_real_test(tmp_path):
    import subprocess
    import sys

    srv, host = start()
    try:
        data = tmp_path / "data"
        env = dict(os.environ, OLLAMA_HOST=host, PYTHONIOENCODING="utf-8")
        build_out = subprocess.run([sys.executable, "-m", "mapnet.cards", str(ROOT / "examples/company"),
                                    "--store", str(data / "cards"), "--files-only"],
                                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        assert build_out.returncode == 0, build_out.stderr[-2000:]
        out = subprocess.run(
            [sys.executable, "-m", "realtest.run", "--corpus", str(ROOT / "examples/company"),
             "--questions", str(ROOT / "realtest/questions/sample-company.json"), "--data", str(data),
             "--results", str(tmp_path / "results")],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600, env=env)
        assert out.returncode == 0, out.stderr[-2000:]
        assert "| map+file-cards |" in out.stdout
        assert "File descriptions:" in out.stdout
        assert "file descriptions (written on arrival)" in out.stdout
    finally:
        srv.shutdown()


def test_three_roles_and_two_readers(tmp_path):
    """Navigator, cartographer and reader are separate; two readers answer the same passages."""
    import subprocess
    import sys

    srv, host = start()
    try:
        out = subprocess.run(
            [sys.executable, "-m", "realtest.run", "--corpus", str(ROOT / "examples/company"),
             "--questions", str(ROOT / "realtest/questions/sample-company.json"), "--data", str(tmp_path / "d"),
             "--results", str(tmp_path / "results"),
             "--map-model", "qwen2.5:3b", "--reader-model", "qwen2.5:3b,qwen2.5:7b",
             "--skip", "map+meaning,map+embed"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600,
            env=dict(os.environ, OLLAMA_HOST=host, PYTHONIOENCODING="utf-8"))
        assert out.returncode == 0, out.stderr[-2000:]
        assert "Answer right (qwen2.5:3b) | Answer right (qwen2.5:7b)" in out.stdout
        assert "cartographer (writes the map): qwen2.5:3b" in out.stdout
        assert "navigator: words + nomic-embed-text" in out.stdout
        res = json.loads(next((tmp_path / "results").iterdir()).joinpath("results.json")
                         .read_text(encoding="utf-8"))
        answered = [r for r in res["rows"] if "by_reader" in r]
        assert answered and all(set(r["by_reader"]) == {"qwen2.5:3b", "qwen2.5:7b"} for r in answered)
        # same passages, different reader: the bigger fake reader answers more often
        s7 = sum(r["by_reader"]["qwen2.5:7b"]["correct"] for r in answered)
        s3 = sum(r["by_reader"]["qwen2.5:3b"]["correct"] for r in answered)
        assert s7 >= s3
    finally:
        srv.shutdown()
