import shutil
from pathlib import Path

from mapnet import MapNet
from mapnet.folder import FolderMap
from mapnet.tokennet import TokenNet, stem, tokens

ROOT = Path(__file__).resolve().parent.parent


def test_tokens_split_identifiers_and_meet_across_endings():
    t = tokens("Set maxSurge for the ReadWriteOncePod volume")
    assert {"maxsurge", "max", "surge", "readwriteoncepod", "read", "write", "pod"} <= set(t)
    assert "the" not in t and "for" not in t
    assert stem("terminating") == stem("terminated") == "terminat"
    assert stem("replicas") == "replica" and stem("policies") == "policy" and stem("class") == "class"


def corpus():
    net = TokenNet(max_common=0.5)  # a ten-passage toy corpus: 'common' must be relative
    docs = {
        "a1": "A namespace stays in terminating state until its finalizer is removed by the controller",
        "a2": "When a namespace is terminating the finalizer list blocks deletion of the namespace",
        "a3": "Stuck terminating objects usually wait on a finalizer that nothing clears",
        "b1": "To remove metadata finalizer entries patch the object and clear the list",
        "c1": "Pods use persistent volumes and claims for storage of data",
        "c2": "Services expose pods through a stable virtual IP address",
        "c3": "Nodes join the cluster by presenting a bootstrap token",
        "c4": "Ingress routes external traffic to services by host and path",
        "c5": "Configuration comes from config maps mounted into containers",
        "c6": "Secrets hold passwords and keys in encoded form",
    }
    for d, text in docs.items():
        net.add(d, text)
    return net


def test_search_finds_by_rare_word_and_headings_count():
    net = corpus()
    hits, reached = net.search("how do I clear a finalizer", k=3)
    assert hits[0][0] in {"a1", "a2", "a3", "b1"} and reached < len(net.ids)
    net.add("h1", "Some text about nothing in particular", heading="Autoscaling walkthrough")
    assert net.search("autoscaling", k=1)[0][0][0] == "h1"


def test_associations_come_from_the_documents_and_are_used_only_when_asked():
    net = corpus()
    nb = dict(net.neighbours("terminat", 5))
    assert "finalizer" in nb and 0 < nb["finalizer"] <= 1
    plain = [h[0] for h in net.search("terminating", k=5)[0]]
    with_assoc = [h[0] for h in net.search("terminating", k=5, assoc=True)[0]]
    assert "b1" not in plain            # b1 never says "terminating"
    assert "b1" in with_assoc           # but it shares "finalizer" with the pages that do
    assert net.neighbours("nonexistentword") == []


def test_live_add_replace_and_remove():
    net = corpus()
    net.search("finalizer")             # builds
    net.add("z1", "The zebra project launches on May fifth")
    assert net.search("zebra launch", k=1)[0][0][0] == "z1"
    net.add("z1", "The zebra project was cancelled")
    assert "cancel" in net.texts[net._index["z1"]] or "cancelled" in net.texts[net._index["z1"]]
    assert net.search("cancelled zebra", k=1)[0][0][0] == "z1"
    net.remove_missing({d for d in net.ids if d != "z1"})
    assert all(h[0] != "z1" for h in net.search("zebra", k=5)[0])


def test_route_follows_the_folder_and_new_files(tmp_path):
    from realtest.retrievers import Shared, TokenFusion, TokenNetRoute, MapRoute
    folder = tmp_path / "company"
    shutil.copytree(ROOT / "examples" / "company", folder)
    shared = Shared(folder)
    route = TokenNetRoute(shared)
    hits, cost = route.search("What are the payment terms with Kalmar Metals?", 4)
    assert hits[0][0] == "finance/vendors/kalmar-metals.md" and cost > 0
    (folder / "engineering" / "zebra.md").write_text("# Zebra launch\n\nThe zebra project launches on May 5.",
                                                     encoding="utf-8")
    shared.fm.sync()
    route.sync()
    assert route.search("when does the zebra project launch", 4)[0][0][0] == "engineering/zebra.md"
    fused = TokenFusion(route, MapRoute(shared, "map+sections", sections=True))
    out, _ = fused.search("What are the payment terms with Kalmar Metals?", 4)
    assert out[0][0] == "finance/vendors/kalmar-metals.md"
    assert len(out) <= 4 and len({(f, t[:80]) for f, t, _ in out}) == len(out)  # no duplicates
