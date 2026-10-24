import json
import shutil
from pathlib import Path

from mapnet.embed import LsaEmbedder
from realtest.corpus import clean_hugo
from realtest.judge import keyword_judge
from realtest.retrievers import EmbedAll, KeywordAll, MapEmbed, MapRoute, Shared

ROOT = Path(__file__).resolve().parent.parent


def test_clean_hugo():
    page = ('---\ntitle: Pod Lifecycle\nweight: 30\n---\n\nA {{< glossary_tooltip text="Pod" '
            'term_id="pod" >}} has phases.\n\n{{< note >}}\nNote text.\n{{< /note >}}\n')
    out = clean_hugo(page)
    assert out.startswith("# Pod Lifecycle")
    assert "A Pod has phases." in out and "{{" not in out and "weight" not in out


def test_keyword_judge():
    item = {"expect": ["limit"], "expect_any": ["equal|same"]}
    assert keyword_judge("Requests must equal limits for every container.", item)
    assert not keyword_judge("Set limits.", item)
    assert not keyword_judge("I could not find it in the files, but limits equal requests", item)


def test_all_retrievers_run(tmp_path):
    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    shared = Shared(folder)
    e = EmbedAll(shared, LsaEmbedder(dims=16), log=lambda *a: None)
    qs = json.loads((ROOT / "examples" / "questions.json").read_text(encoding="utf-8"))
    for r in [KeywordAll(shared), e, MapRoute(shared), MapEmbed(shared, e)]:
        found = sum(q["file"] in [f for f, _, _ in r.search(q["q"], 4)[0]] for q in qs[:10])
        assert found >= 7, r.name
    # freshness: a new file is searchable by every method after ingest
    (folder / "engineering" / "new.md").write_text("Zebra project launch is on May 5.")
    shared.fm.sync()
    before = len(e.ids)
    e.add([(d, t) for d, t in shared.chunks() if d.startswith("engineering/new.md")])
    assert MapRoute(shared).search("zebra project launch", 4)[0][0][0] == "engineering/new.md"
    # the no-model stand-in can't know a word it never saw ("zebra"), so only check
    # that the passage was indexed; a real embedding model is tested by realtest.run
    assert len(e.ids) == before + 1 and e.M.shape[0] == len(e.ids)


# ---- the Ollama code paths, against a fake server speaking the same protocol
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_GET(self):
        self._send({"models": [{"name": "tiny:1b"}, {"name": "emb:latest"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":
            self._send({"embeddings": [[len(t) % 7 + 1.0, t.count("e") + 1.0, 1.0] for t in body["input"]]})
        elif self.path == "/api/generate" and body.get("format") == "json":
            self._send({"response": json.dumps({"question": "How long is parental leave for birth parents?",
                                                "answer": "20 weeks"})})
        else:  # judge
            self._send({"response": "yes" if "20" in body["prompt"].split("Candidate answer:")[1] else "no"})


def test_ollama_paths(tmp_path):
    from mapnet.embed import OllamaEmbedder
    from mapnet.llm import OllamaReader
    from realtest.judge import ModelJudge
    from realtest.qgen import generate

    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{srv.server_port}"
    try:
        folder = tmp_path / "company"
        shutil.copytree(ROOT / "examples" / "company", folder)
        shared = Shared(folder)
        emb = OllamaEmbedder("emb", host=host, cache_dir=str(tmp_path / "cache"))
        assert emb.available()
        e = EmbedAll(shared, emb, log=lambda *a: None)
        assert e.M.shape == (shared.map.root.n_docs, 3)
        assert (tmp_path / "cache" / "emb.npz").exists()  # cached for the next run
        reader = OllamaReader("tiny:1b", host=host)
        qs = generate(shared, reader, 2, log=lambda *a: None)
        assert len(qs) == 2 and qs[0]["answer"] == "20 weeks"
        j = ModelJudge(reader)
        assert j("Birth parents get 20 weeks.", qs[0]) and not j("Twelve weeks.", qs[0])
    finally:
        srv.shutdown()


def test_evaluate_runs_one_method_at_a_time_and_reports_time():
    from realtest.run import evaluate, summarize, time_table

    order = []

    class R:
        def __init__(self, name):
            self.name = name

        def search(self, q, k):
            order.append((self.name, q))
            return [("a.md", "text", 1.0)], 1

    class Reader:
        model = "m"

        def answer(self, q, hits):
            from types import SimpleNamespace
            return SimpleNamespace(text="ok", seconds=2.0, input_tokens=5)

    items = [{"q": "q1", "files": ["a.md"]}, {"q": "q2", "files": ["a.md"]}]
    timing = {}
    rows = evaluate(items, [R("x"), R("y")], 4, [Reader()], ["x", "y"], None, timing)
    # all of x's questions come before any of y's: models stay loaded
    assert [n for n, _ in order] == ["x", "x", "y", "y"]
    assert set(timing["retrieve"]) == {"x", "y"} and "m" in timing["answer"]
    s = summarize(rows, ["x", "y"])
    assert s["x"]["answer_mean_s"]["m"] == 2.0 and s["x"]["answer_total_s"]["m"] == 4.0
    t = time_table(s, "T")
    assert "Per question (m)" in t and "2." in t


def test_embedder_cache_is_not_rewritten_on_every_call(tmp_path):
    """Saving the whole cache after each call made many small calls quadratic."""
    import numpy as np
    from mapnet.embed import OllamaEmbedder

    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{srv.server_port}"
    try:
        emb = OllamaEmbedder("emb", host=host, cache_dir=str(tmp_path / "c"))
        saves = []
        real = emb.save
        emb.save = lambda: (saves.append(1), real())
        for i in range(5):
            emb.embed([f"text number {i}"])
        assert len(saves) == 1  # the first call saves; the rest wait
        emb.flush()
        assert len(saves) == 2
        again = OllamaEmbedder("emb", host=host, cache_dir=str(tmp_path / "c"))
        assert len(again._cache) == 5
    finally:
        srv.shutdown()
