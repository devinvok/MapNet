"""The token network and its shortlist must never let a caller see, or be steered by, what they
cannot open: not the passage, not its words in the scores, not its words among the associations."""
import shutil
from pathlib import Path

import numpy as np

from mapnet.shortlist import ShortlistRerank
from mapnet.tokennet import TokenNet

from test_embedtree import TinyEmbedder

ROOT = Path(__file__).resolve().parent.parent

OPEN = [
    ("handbook.md#0", "Salary is paid on the last working day of the month."),
    ("handbook.md#1", "Holiday requests go to your manager two weeks ahead."),
    ("handbook.md#2", "Expenses are paid back with the next salary."),
    ("it.md#0", "Reset your laptop password from the self service portal."),
]
SECRET = [
    ("payroll.md#0", "Salary bands: grade seven earns 91000, bonus pool zyxquart quarterly."),
    ("payroll.md#1", "The zyxquart bonus pool is paid with the salary in March."),
    ("payroll.md#2", "Salary review for grade seven happens in zyxquart meetings."),
]


def build(with_secret=True):
    net = TokenNet(min_support=1, max_common=1.0)  # tiny corpus: no word is 'too common'
    for d, t in OPEN:
        net.add(d, t)
    if with_secret:
        for d, t in SECRET:
            net.add(d, t, acl=("hr",))
    return net


def test_restricted_passages_are_never_returned():
    net = build()
    hits, _ = net.search("salary bonus pool grade", k=10, roles=["employee"])
    assert hits and all(not d.startswith("payroll") for d, _, _ in hits)
    hits, _ = net.search("salary bonus pool grade", k=10, roles=["hr"])
    assert hits[0][0].startswith("payroll")
    assert net.search("zyxquart", k=10, roles=["employee"]) == ([], 0)  # a word only they hold


def test_restricted_words_do_not_change_the_scores():
    """For a caller without the role, scores must equal those of a network where the restricted
    passages never existed (how rare 'salary' is must not count the payroll pages)."""
    full, bare = build(), build(with_secret=False)
    q = "when is salary paid"
    a = {d: s for d, _, s in full.search(q, k=10, roles=["employee"])[0]}
    b = {d: s for d, _, s in bare.search(q, k=10)[0]}
    assert a.keys() == b.keys()
    assert all(np.isclose(a[d], b[d]) for d in a)


def test_associations_come_only_from_open_passages():
    full, bare = build(), build(with_secret=False)
    seen = [t for t, _ in full.neighbours("salary", n=20, roles=["employee"])]
    assert "zyxquart" not in seen and "bonu" not in seen
    assert seen == [t for t, _ in bare.neighbours("salary", n=20)]
    assert "zyxquart" in [t for t, _ in full.neighbours("salary", n=20, roles=["hr"])]


def test_no_roles_means_everything_as_before():
    net = build()
    assert net.search("zyxquart", k=4)[0][0][0].startswith("payroll")


def test_shortlist_never_embeds_what_the_caller_cannot_open():
    seen = []

    class Recording(TinyEmbedder):
        def embed(self, texts, kind="document", progress=None):
            seen.extend(texts)
            return super().embed(texts, kind, progress)

    rr = ShortlistRerank(build(), Recording(), short=10)
    hits, _, _ = rr.search("salary bonus pool grade seven", k=4, roles=["employee"])
    assert all(not d.startswith("payroll") for d, _, _ in hits)
    assert not any(t in seen for _, t in SECRET)


def test_folder_access_files_reach_the_token_network(tmp_path):
    """The company example: hr/payroll has an .access file. The real-test route reads it, and a
    change to the .access file alone (no document touched) takes effect on the next sync, both
    granting and revoking."""
    from realtest.retrievers import Shared, TokenNetRoute

    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    route = TokenNetRoute(Shared(folder))
    net = route.net
    q = "payroll salary bands"
    assert all("payroll" not in d for d, _, _ in net.search(q, k=20, roles=["employee"])[0])
    assert any("payroll" in d for d, _, _ in net.search(q, k=20, roles=["hr"])[0])

    (folder / "hr" / "payroll" / ".access").write_text("hr, employee", encoding="utf-8")
    route.s.fm.sync()
    route.sync()
    assert any("payroll" in d for d, _, _ in net.search(q, k=20, roles=["employee"])[0])

    (folder / "hr" / "payroll" / ".access").write_text("hr", encoding="utf-8")
    route.s.fm.sync()
    route.sync()
    assert all("payroll" not in d for d, _, _ in net.search(q, k=20, roles=["employee"])[0])


def test_revoking_access_reaches_the_map_itself(tmp_path):
    """The same for the folder map's own search: deleting or editing .access re-indexes the
    files beneath it."""
    from mapnet import MapNet
    from mapnet.folder import FolderMap

    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    fm = FolderMap(folder, mapnet=MapNet())
    fm.sync()
    q = "payroll salary bands"
    hidden = lambda: all("payroll" not in h.source for h in fm.map.flat_search(q, k=20, roles=["employee"]).hits)  # noqa: E731
    assert hidden()
    (folder / "hr" / "payroll" / ".access").unlink()           # now open to everyone
    changes = fm.sync()
    assert changes and not hidden()
    (folder / "hr" / "payroll" / ".access").write_text("hr", encoding="utf-8")
    fm.sync()
    assert hidden()
    assert fm.sync() == []                                       # nothing changed: nothing re-indexed
