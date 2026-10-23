import json
import shutil
from pathlib import Path

import numpy as np

from mapnet import MapNet
from mapnet.folder import FolderMap
from mapnet.meaning import MeaningRouter

ROOT = Path(__file__).resolve().parent.parent
QS = json.loads((ROOT / "examples" / "questions.json").read_text(encoding="utf-8"))
QS = [q for q in QS if not q.get("roles")]


def company(tmp_path):
    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    fm = FolderMap(folder, mapnet=MapNet(section_weight=1.0, route_weight=0.3))
    fm.sync()
    return fm.map


class Noise:
    rng = np.random.default_rng(0)

    def embed(self, texts, kind="document", progress=None):
        v = self.rng.normal(size=(len(texts), 32))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class Oracle:
    """Knows which file answers which question: an upper bound for meaning."""

    def embed(self, texts, kind="document", progress=None):
        out = []
        for t in texts:
            v = np.zeros(len(QS) + 1)
            if kind == "query":
                v[[i for i, q in enumerate(QS) if q["q"] == t][0]] = 1
            else:
                v[[i for i, q in enumerate(QS) if t.startswith(q["file"])]] = 1
                v[-1] = 0.01
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def right_files(search):
    return sum(q["file"] in [h.doc_id.split("#")[0] for h in search(q["q"]).hits] for q in QS)


def test_meaning_is_safe_when_wrong_and_helps_when_right(tmp_path):
    m = company(tmp_path)
    base = right_files(lambda q: m.search(q, k=4, min_score=0.12))
    noisy = right_files(lambda q: MeaningRouter(m, Noise()).search(q, k=4, min_score=0.12))
    oracle = right_files(lambda q: MeaningRouter(m, Oracle()).search(q, k=4, min_score=0.12))
    assert noisy >= base - 1  # a useless meaning signal costs at most one question
    assert oracle == len(QS)  # a perfect one fixes every wrong turn


def test_meaning_index_is_the_map_not_the_passages(tmp_path):
    m = company(tmp_path)
    r = MeaningRouter(m, Noise())
    n, _ = r.build()
    assert n == len(m.sources())  # one vector per file up front
    r.search("What are the payment terms with Kalmar Metals?", k=4)
    assert r.vectors > n  # sections of visited files get vectors on arrival...
    assert r.vectors < m.root.n_docs + n  # ...but nowhere near one per passage


def test_meaning_respects_permissions(tmp_path):
    m = company(tmp_path)
    r = MeaningRouter(m, Oracle())
    q = next(x for x in json.loads((ROOT / "examples" / "questions.json").read_text(encoding="utf-8"))
             if x.get("roles"))
    QS.append({"q": q["q"], "file": q["file"]})
    try:
        hits = r.search(q["q"], k=4, roles=["employee"]).hits
        assert all("payroll" not in h.source for h in hits)
        assert r.search(q["q"], k=4, roles=["hr"]).hits[0].source == q["file"]
    finally:
        QS.pop()
