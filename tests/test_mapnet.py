import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from mapnet import MapNet
from mapnet.assistant import Assistant
from mapnet.llm import ExtractiveReader, OllamaReader

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def company(tmp_path):
    dst = tmp_path / "company"
    shutil.copytree(EXAMPLES / "company", dst)
    return dst


@pytest.fixture
def assistant(company):
    return Assistant(company, reader=ExtractiveReader())


# --------------------------------------------------------------------- core
def test_register_search_and_remove():
    m = MapNet()
    m.register_source("a", [("a1", "kestrel invoice payment overdue")], path=["finance", "invoices"])
    m.register_source("b", [("b1", "outage pager rollback")], path=["eng", "incidents"])
    assert m.search("kestrel invoice").hits[0].doc_id == "a1"
    m.remove_source("finance/invoices/a")
    assert "finance" not in m.root.children  # empty groups are pruned
    assert m.search("kestrel invoice").hits[0].doc_id == "b1"  # only b is left
    assert m.root.n_docs == 1
    assert "a1" not in m.doc_source


def test_permissions_hide_branches():
    m = MapNet()
    m.register_source("pay", [("p1", "salary payday tenth")], path=["hr", "payroll"], acl=["hr"])
    m.register_source("faq", [("f1", "general questions")], path=["support"])
    assert all(h.doc_id != "p1" for h in m.search("salary payday", roles=["employee"]).hits)
    assert m.search("salary payday", roles=["hr"]).hits[0].doc_id == "p1"
    assert m.flat_search("salary payday", roles=["employee"]).restricted_docs_scanned == 1


def test_auto_place():
    m = MapNet()
    m.register_source("inv", [("i1", "invoice vendor billing remittance")], path=["finance", "invoices"])
    m.register_source("inc", [("o1", "outage pager severity rollback")], path=["eng", "incidents"])
    node = m.register_source("portal", [("v1", "vendor invoice billing overdue")])
    assert node.path == "finance/invoices/portal"


# ------------------------------------------------------------------- folder
def test_folder_sync_add_change_delete(assistant, company):
    a = assistant
    assert "89 USD" in a.ask("How much does the Pro plan cost per robot?").answer
    shutil.copy(EXAMPLES / "drop-in" / "pricing.md", company / "sales" / "pricing.md")
    (company / "engineering" / "new.md").write_text("Zebra project launch is on May 5.")
    kinds = {k for k, *_ in a.fm.sync()}
    assert kinds == {"added", "changed"}
    assert "95 USD" in a.ask("Pro plan price from January 2027").answer
    assert "new.md" in a.ask("When is the zebra project launch?").sources[0]
    (company / "engineering" / "new.md").unlink()
    assert [k for k, *_ in a.fm.sync()] == ["deleted"]
    assert all("new.md" not in s for s in a.ask("zebra project launch").sources)


def test_access_file_restricts_folder(assistant):
    a = assistant
    assert "payroll" not in " ".join(a.ask("On what day are salaries paid?", roles=["employee"]).sources)
    assert a.ask("On what day are salaries paid?", roles=["hr"]).sources[0].startswith("hr/payroll")


def test_question_set_retrieval(assistant):
    qs = json.loads((EXAMPLES / "questions.json").read_text())
    found = sum(q["file"] in assistant.ask(q["q"], roles=q.get("roles", ["employee"])).sources
                for q in qs)
    assert found == len(qs)


# -------------------------------------------------------------------- notes
def test_notes_shortcut_and_verification(assistant):
    a = assistant
    first = a.ask("What are the payment terms with Kalmar Metals?", roles=["employee"], agent="A")
    assert first.via == "map" and first.note_written
    second = a.ask("Kalmar Metals invoice payment terms", roles=["employee"], agent="B")
    assert a.ask("Kalmar Metals invoice payment terms", roles=["hr"]).via == "map"  # other scope
    assert second.via == "note" and second.files_opened == 1

    bad = a.notes.add("Q3 budget numbers by department", "support/it-help.md", author="X",
                      roles=["employee"])
    t = a.ask("What was the Q3 budget for Sales?", roles=["employee"], agent="C")
    assert bad.id not in a.notes.notes  # failed verification -> deleted
    assert t.sources[0] == "finance/budget-q3-2026.csv"


def test_notes_rules(assistant, company):
    a = assistant
    assert a.notes.add("anything", "does/not/exist.md", author="X") is None
    assert a.notes.add("anything", "finance", author="X") is None  # folders are not targets
    n = a.notes.add("salary payment day", "hr/payroll/payroll-calendar-2026.md", author="hr",
                    roles=["employee"])  # even a mis-scoped note can't reveal the folder
    assert n.id not in [x.id for x, _ in a.notes.lookup("salary payment day", ["employee"])]
    m = a.notes.add("halden renewal", "sales/accounts/halden-foods.md", author="A")
    (company / "sales" / "accounts" / "halden-foods.md").unlink()
    a.fm.sync()
    a.notes.prune_missing()
    assert m.id not in a.notes.notes


def test_notes_persist(company, tmp_path):
    path = tmp_path / "notes.json"
    a = Assistant(company, reader=ExtractiveReader(), notes_path=str(path))
    a.ask("When does Halden Foods renew?")
    b = Assistant(company, reader=ExtractiveReader(), notes_path=str(path))
    assert b.ask("Halden Foods renewal date").via == "note"


# ---------------------------------------------------------------- readers
class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"models": [{"name": "tiny:1b"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["role"] == "system"
        assert "Kalmar" in body["messages"][1]["content"]
        self.send_response(200)
        self.end_headers()
        for piece in ["Net 45 ", "days."]:
            self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
        self.wfile.write((json.dumps({"message": {"content": ""}, "done": True,
                                      "prompt_eval_count": 120, "eval_count": 4}) + "\n").encode())


def test_ollama_reader_streams(assistant):
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    r = OllamaReader("tiny:1b", host=f"http://127.0.0.1:{srv.server_port}")
    assert r.available()
    assistant.reader = r
    pieces = []
    t = assistant.ask("What are the payment terms with Kalmar Metals?", stream=pieces.append)
    srv.shutdown()
    assert t.answer == "Net 45 days." and pieces == ["Net 45 ", "days."]
    assert (t.tokens_in, t.tokens_out) == (120, 4)
