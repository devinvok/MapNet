import shutil
from pathlib import Path

import numpy as np
import pytest

from mapnet import MapNet
from mapnet.folder import FolderMap
from mapnet.pagemap import PageMap

from test_embedtree import TinyEmbedder

ROOT = Path(__file__).resolve().parent.parent


def company(tmp_path):
    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    fm = FolderMap(folder, mapnet=MapNet())
    fm.sync()
    return fm


@pytest.mark.parametrize("mode", ["mean", "text"])
def test_finds_the_file_by_its_own_text(tmp_path, mode):
    fm = company(tmp_path)
    p = PageMap(fm.map, TinyEmbedder(), mode=mode)
    assert p.sync() == len(fm.map.sources())
    r = p.search("What are the payment terms with Kalmar Metals?", k=4)
    assert r.hits[0].source == "finance/vendors/kalmar-metals.md"


def test_text_mode_embeds_passages_only_on_arrival(tmp_path):
    fm = company(tmp_path)
    n = len(fm.map.sources())
    mean, text = PageMap(fm.map, TinyEmbedder(), "mean"), PageMap(fm.map, TinyEmbedder(), "text")
    mean.sync()
    text.sync()
    assert mean.files_embedded == n and text.files_embedded == 0
    text.search("What are the payment terms with Kalmar Metals?", k=4)
    assert 0 < text.files_embedded <= text.file_beam


def test_adding_changing_and_removing_files(tmp_path):
    fm = company(tmp_path)
    p = PageMap(fm.map, TinyEmbedder(), "mean")
    p.sync()
    f = fm.root / "engineering" / "zebra.md"
    f.write_text("# Zebra launch\n\nThe zebra project launches on May 5.", encoding="utf-8")
    fm.sync()
    assert p.sync() == 1  # only the new file is embedded
    assert p.search("when does the zebra project launch", k=4).hits[0].source == "engineering/zebra.md"
    f.write_text("# Zebra launch\n\nThe zebra project is cancelled.", encoding="utf-8")
    fm.sync()
    assert p.sync() == 1
    assert "cancelled" in p.search("is the zebra project cancelled", k=4).hits[0].text
    f.unlink()
    fm.sync()
    p.sync()
    assert "engineering/zebra.md" not in p.page
    assert all(h.source != "engineering/zebra.md" for h in p.search("zebra project", k=4).hits)


@pytest.mark.parametrize("mode", ["mean", "text"])
def test_never_reads_files_the_caller_cannot_open(tmp_path, mode):
    p = PageMap(company(tmp_path).map, TinyEmbedder(), mode)
    p.sync()
    q = "On what day of the month are salaries paid?"
    seen = p.search(q, k=4, roles=["employee"])
    assert all("payroll" not in s for s in seen.sources_searched)
    assert p.search(q, k=4, roles=["hr"]).hits[0].source.startswith("hr/payroll")


def test_page_vectors_are_unit_length(tmp_path):
    p = PageMap(company(tmp_path).map, TinyEmbedder(), "mean")
    p.sync()
    assert all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-4 for v in p.page.values())
