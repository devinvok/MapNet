import json
import shutil
import zlib
from pathlib import Path

import numpy as np

from mapnet import MapNet
from mapnet.embedtree import EmbedTree
from mapnet.folder import FolderMap

ROOT = Path(__file__).resolve().parent.parent
QS = [q for q in json.loads((ROOT / "examples" / "questions.json").read_text(encoding="utf-8"))
      if not q.get("roles")]


class TinyEmbedder:
    """Dense hashed bag of words: a small, deterministic stand-in for an embedding model."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind="document", progress=None):
        self.calls += 1
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in t.lower().replace("?", " ").split():
                out[i, zlib.crc32(w.encode()) % 256] += 1
        return out


def company(tmp_path):
    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    fm = FolderMap(folder, mapnet=MapNet())
    fm.sync()
    return fm


def test_files_are_embedded_only_when_the_walk_arrives(tmp_path):
    fm = company(tmp_path)
    t = EmbedTree(fm.map, TinyEmbedder())
    assert t.build() == len(fm.map.sources())  # one direction vector per file
    assert t.files_embedded == 0
    r = t.search("What are the payment terms with Kalmar Metals?", k=4)
    assert 0 < t.files_embedded <= t.max_new
    assert t.passages_embedded < fm.map.root.n_docs
    assert r.hits and r.hits[0].source == "finance/vendors/kalmar-metals.md"
    # files skipped because of the per-question limit are embedded on the next visit...
    t.search("What are the payment terms with Kalmar Metals?", k=4)
    before = t.files_embedded
    # ...and once everything on the route is embedded, a revisit embeds nothing
    t.search("What are the payment terms with Kalmar Metals?", k=4)
    assert t.files_embedded == before


def test_it_finds_the_sample_answers_and_improves_with_use(tmp_path):
    fm = company(tmp_path)
    t = EmbedTree(fm.map, TinyEmbedder())
    t.build()

    def found():
        return sum(q["file"] in [h.doc_id.split("#")[0] for h in t.search(q["q"], k=4).hits] for q in QS)

    first = found()
    second = found()
    assert first >= len(QS) // 2
    assert second >= first  # the map sharpens where questions went


def test_changed_and_new_files(tmp_path):
    fm = company(tmp_path)
    t = EmbedTree(fm.map, TinyEmbedder())
    t.build()
    t.search("Kalmar Metals payment terms", k=4)
    (fm.root / "engineering" / "zebra.md").write_text("# Zebra launch\n\nThe zebra project launches on May 5.",
                                                      encoding="utf-8")
    fm.sync()
    assert t.build() == 1  # only the new file needs a direction vector
    assert t.search("when does the zebra project launch", k=4).hits[0].source == "engineering/zebra.md"
    f = fm.root / "finance" / "vendors" / "kalmar-metals.md"
    f.write_text(f.read_text(encoding="utf-8") + "\n\nNew bank: Nordbank.", encoding="utf-8")
    fm.sync()
    t.build()
    n = t.files_embedded
    t.search("Kalmar Metals bank", k=4)
    assert t.files_embedded > n  # a changed file is embedded again on its next visit


def test_permissions(tmp_path):
    fm = company(tmp_path)
    t = EmbedTree(fm.map, TinyEmbedder())
    t.build()
    q = "payroll calendar: on what day are salaries paid?"
    assert all("payroll" not in h.source for h in t.search(q, k=4, roles=["employee"]).hits)
    assert t.search(q, k=4, roles=["hr"]).hits[0].source.startswith("hr/payroll")
