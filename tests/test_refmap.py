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


def test_translator_blends_terms_into_the_question_vector(tmp_path):
    import numpy as np
    r = RefMap(company(tmp_path).map, TinyEmbedder())
    r.build()
    plain, none = r._query_vector("when does the thing start")
    assert none == ""
    blended, terms = r._query_vector("when does the thing start", lambda q: "zebra launch")
    assert terms == "zebra launch"
    assert abs(float(np.linalg.norm(blended)) - 1.0) < 1e-5 and not np.allclose(plain, blended)
    # a translator that fails or says nothing leaves the question as it was
    same, _ = r._query_vector("when does the thing start", lambda q: "")
    assert np.allclose(plain, same)


def test_term_translator_remembers_answers(monkeypatch):
    import mapnet.cards as cards
    from mapnet.refmap import TermTranslator

    seen = []

    def fake(reader, prompt):
        seen.append(prompt)
        return {"terms": ["finalizer", " deletionTimestamp ", "", "namespace"]}

    monkeypatch.setattr(cards, "_generate", fake)
    t = TermTranslator(reader=object())
    assert t("why is my namespace stuck") == "finalizer, deletionTimestamp, namespace"
    assert t("why is my namespace stuck") == "finalizer, deletionTimestamp, namespace"
    assert t.calls == 1 and len(seen) == 1 and "why is my namespace stuck" in seen[0]

    def boom(reader, prompt):
        raise RuntimeError("model is down")

    monkeypatch.setattr(cards, "_generate", boom)
    assert TermTranslator(reader=object())("anything") == ""  # routing falls back to the plain question


def test_control_scores_every_reference_and_respects_roles(tmp_path):
    fm = company(tmp_path)
    fm.map.set_file_card("hr/payroll/payroll-calendar-2026.md",
                         "When salaries are paid. On what day of the month are salaries paid?")
    r = RefMap(fm.map, TinyEmbedder())
    r.build()
    qv, _ = r._query_vector("On what day of the month are salaries paid?")
    open_ = [p for _, p in r.flat_ranking(qv, roles={"employee"})]
    assert open_ and not any("payroll" in p for p in open_)
    assert r.flat_ranking(qv, roles={"hr"})[0][1].startswith("hr/payroll")


def test_explain_says_where_a_file_was_found_or_lost(tmp_path):
    r = RefMap(company(tmp_path).map, TinyEmbedder())
    r.build()
    ex = r.explain("What are the payment terms with Kalmar Metals?", ["finance/vendors/kalmar-metals.md"])
    assert ex["in_chosen"] and ex["flat_rank"] == 1 and ex["walk_rank"] is not None
    assert ex["home_visited"] and not ex["cut"] and ex["flat_top"] == "finance/vendors/kalmar-metals.md"
    wrong = r.explain("What are the payment terms with Kalmar Metals?", ["hr/parental-leave-policy.md"])
    assert wrong["flat_rank"] != 1 and wrong["flat_top"] == "finance/vendors/kalmar-metals.md"


def test_explain_report_sorts_misses_into_walk_or_references():
    from realtest.run import explain_report

    def row(in_chosen, flat_rank, home=True, cut=False, top="a.md"):
        return dict(q="q", in_chosen=in_chosen, flat_rank=flat_rank, home_visited=home, cut=cut,
                    flat_top=top, walk_rank=1 if in_chosen else None)

    rows = [row(True, 1), row(True, 20), row(False, 2, home=False), row(False, 3, cut=True),
            row(False, None, top="m.md"), row(False, 30, top="m.md")]
    text = explain_report(rows, beam=6)
    assert "| Found by the walk and by the control | 1 |" in text
    assert "| Found by the walk only | 1 |" in text
    assert "the walk lost it** | 2 |" in text and "can't)** | 2 |" in text
    assert "never entered the right folder: 1" in text and "past the limit of new files read per question: 1" in text
    assert "`m.md` ×2" in text
