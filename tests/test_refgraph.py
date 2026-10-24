import shutil
from pathlib import Path

import numpy as np

from mapnet import MapNet
from mapnet.folder import FolderMap
from mapnet.refgraph import Hnsw, RefGraph
from mapnet.refmap import RefMap

from test_embedtree import TinyEmbedder

ROOT = Path(__file__).resolve().parent.parent


def company(tmp_path):
    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    fm = FolderMap(folder, mapnet=MapNet())
    fm.sync()
    return fm


def unit_rows(n, dim, seed):
    X = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_graph_search_matches_exact_search():
    X = unit_rows(600, 24, 1)
    h = Hnsw(24, M=12, ef_construction=64)
    for v in X:
        h.add(v)
    Q = unit_rows(60, 24, 2)
    hit = 0
    for q in Q:
        best = int(np.argmax(X @ q))
        found, cost = h.search(q, ef=48)
        hit += found[0][1] == best
        assert cost < len(X)  # looked at far fewer vectors than an exact scan
    assert hit >= 55  # finds the true nearest neighbour for (nearly) every query


def test_adding_needs_no_rebuild():
    X = unit_rows(300, 16, 3)
    h = Hnsw(16)
    for v in X:
        h.add(v)
    new = unit_rows(1, 16, 99)[0]
    i = h.add(new)
    found, _ = h.search(new, ef=32)
    assert found[0][1] == i and abs(found[0][0] - 1.0) < 1e-5


def test_general_items_sit_in_the_upper_layers():
    h = Hnsw(8)
    X = unit_rows(200, 8, 4)
    ids = [h.add(v, boost=1 if i % 10 == 0 else 0) for i, v in enumerate(X)]
    boosted = [h.levels[i] for k, i in enumerate(ids) if k % 10 == 0]
    ordinary = [h.levels[i] for k, i in enumerate(ids) if k % 10 != 0]
    assert min(boosted) >= 1 and np.mean(boosted) > np.mean(ordinary)


def test_graph_finds_the_referenced_file(tmp_path):
    fm = company(tmp_path)
    base = RefMap(fm.map, TinyEmbedder())
    base.build()
    g = RefGraph(base.fork(TinyEmbedder()))
    res = g.search("What are the payment terms with Kalmar Metals?", k=4)
    assert res.hits[0].source == "finance/vendors/kalmar-metals.md"
    st = g.stats()
    assert st["nodes"] >= len(fm.map.sources()) and st["layers"] >= 1


def test_restricted_references_have_their_own_graph(tmp_path):
    fm = company(tmp_path)
    fm.map.set_file_card("hr/payroll/payroll-calendar-2026.md",
                         "When salaries are paid. On what day of the month are salaries paid?")
    base = RefMap(fm.map, TinyEmbedder())
    base.build()
    g = RefGraph(base.fork())
    g.sync()
    assert g.stats()["groups"] >= 2  # the public graph and at least one restricted one
    q = "On what day of the month are salaries paid?"
    assert all("payroll" not in h.source for h in g.search(q, k=4, roles=["employee"]).hits)
    assert g.search(q, k=4, roles=["hr"]).hits[0].source.startswith("hr/payroll")
    # a caller who may not open a group never touches it: the cost only counts the groups they may enter
    assert g.search(q, k=4, roles=["employee"]).cost < g.search(q, k=4, roles=["hr"]).cost


def test_new_and_changed_files_without_a_rebuild(tmp_path):
    fm = company(tmp_path)
    base = RefMap(fm.map, TinyEmbedder())
    base.build()
    g = RefGraph(base.fork())
    g.sync()
    (fm.root / "engineering" / "zebra.md").write_text("# Zebra launch\n\nThe zebra project launches on May 5.",
                                                      encoding="utf-8")
    fm.sync()
    base.build()  # the original registers the new file; the graph notices by itself
    assert g.search("when does the zebra project launch", k=4).hits[0].source == "engineering/zebra.md"
    fm.map.set_file_card("engineering/zebra.md", "Launch plan. When does the zebra go live?")
    base.build()
    assert g.search("when does the zebra go live", k=4).hits[0].source == "engineering/zebra.md"
    before = g.stats()
    assert before["dead"] == 0
    fm.map.set_file_card("engineering/zebra.md", "Release schedule. Which day is the zebra release?")
    base.build()  # the description changed: its old claims die, the new ones are inserted, no rebuild
    assert g.search("which day is the zebra release", k=4).hits[0].source == "engineering/zebra.md"
    after = g.stats()
    assert after["dead"] >= 2 and after["nodes"] > before["nodes"]
    # dead claims are never returned: only live references can bring a file up
    g2 = g.groups[next(iter(g.groups))]
    assert all(g2.nodes[i].text != "Launch plan." for i in range(len(g2.nodes)) if i not in g2.dead)


def test_each_variant_reads_files_on_its_own(tmp_path):
    fm = company(tmp_path)
    base = RefMap(fm.map, TinyEmbedder())
    base.build()
    a, b = base.fork(TinyEmbedder()), base.fork(TinyEmbedder())
    a.search("What are the payment terms with Kalmar Metals?", k=4)
    assert a.files_embedded > 0 and b.files_embedded == 0 and base.files_embedded == 0
    assert a.refs is b.refs is base.refs  # same references
    (fm.root / "engineering" / "zebra.md").write_text("# Zebra\n\nThe zebra project launches on May 5.",
                                                      encoding="utf-8")
    fm.sync()
    base.build()
    assert b.search("when does the zebra project launch", k=4).hits[0].source == "engineering/zebra.md"
