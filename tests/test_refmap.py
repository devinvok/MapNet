import shutil
from pathlib import Path

from mapnet import EVERYONE, MapNet
from mapnet.folder import FolderMap
from mapnet.refmap import RefMap, split_card

from test_embedtree import TinyEmbedder

ROOT = Path(__file__).resolve().parent.parent


def company(tmp_path):
    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    fm = FolderMap(folder, mapnet=MapNet())
    fm.sync()
    return fm


def test_split_card():
    about, asks = split_card("Explains how long data is kept. How long do we keep resumes? "
                             "When is telemetry deleted? Do tickets expire?")
    assert about == "Explains how long data is kept."
    assert asks == ["How long do we keep resumes?", "When is telemetry deleted?", "Do tickets expire?"]


def test_questions_are_registered_where_they_belong(tmp_path):
    fm = company(tmp_path)
    m = fm.map
    # the retention policy lives under legal/, but one of its questions is about hiring
    m.set_file_card("legal/data-retention.md",
                    "Rules for how long records are kept. How long do we keep a job candidate resume "
                    "after the hiring interview?")
    r = RefMap(m, TinyEmbedder())
    assert r.build() == len(m.sources())
    zones = {z for z, kind, _ in r.describe("legal/data-retention.md")}
    assert "legal" in zones                 # home
    assert len(zones) >= 2 and r.placed_away >= 1   # and somewhere else too: multi-homed
    away = [z for z, kind, _ in r.describe("legal/data-retention.md") if z != "legal"]
    assert away and all(not z.startswith("legal") for z in away)


def test_search_fetches_from_the_referenced_file(tmp_path):
    fm = company(tmp_path)
    r = RefMap(fm.map, TinyEmbedder())
    r.build()
    res = r.search("What are the payment terms with Kalmar Metals?", k=4)
    assert res.hits[0].source == "finance/vendors/kalmar-metals.md"
    assert r.files_embedded <= r.max_new and r.passages_embedded < fm.map.root.n_docs


def test_restricted_references_stay_home_and_hidden(tmp_path):
    fm = company(tmp_path)
    m = fm.map
    m.set_file_card("hr/payroll/payroll-calendar-2026.md",
                    "When salaries are paid. On what day of the month are salaries paid?")
    r = RefMap(m, TinyEmbedder())
    r.build()
    for zone, kind, _ in r.describe("hr/payroll/payroll-calendar-2026.md"):
        assert zone.startswith("hr/payroll")  # never registered in a public zone
    q = "On what day of the month are salaries paid?"
    assert all("payroll" not in h.source for h in r.search(q, k=4, roles=["employee"]).hits)
    assert r.search(q, k=4, roles=["hr"]).hits[0].source.startswith("hr/payroll")


def test_adding_and_changing_files(tmp_path):
    fm = company(tmp_path)
    r = RefMap(fm.map, TinyEmbedder())
    r.build()
    (fm.root / "engineering" / "zebra.md").write_text("# Zebra launch\n\nThe zebra project launches on May 5.",
                                                      encoding="utf-8")
    fm.sync()
    assert r.build() == 1  # only the new file is registered; nothing is rebuilt
    assert r.search("when does the zebra project launch", k=4).hits[0].source == "engineering/zebra.md"
    fm.map.set_file_card("engineering/zebra.md", "Launch plan. When does the zebra go live?")
    assert r.build() == 1  # a changed description re-registers that file only
    kinds = {k for _, k, _ in r.describe("engineering/zebra.md")}
    assert {"label", "about", "ask"} <= kinds
