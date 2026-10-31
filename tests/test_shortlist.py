import numpy as np

from mapnet.shortlist import ShortlistRerank
from mapnet.tokennet import TokenNet

from test_embedtree import TinyEmbedder


def net_with(passages):
    net = TokenNet()
    for doc_id, text in passages:
        net.add(doc_id, text)
    return net


PASSAGES = [
    ("chatty.md#0", "If you want to make sure your app does what you want, make a plan and make it work."),
    ("answer.md#0", "The downward API exposes the pod name and namespace to the container as files."),
    ("other.md#0", "Volumes keep data when a container restarts."),
]


class Meaning:
    """Stand-in for an embedding model that knows a passage about the downward API answers a
    question about a pod finding its own name, though they share few words."""

    def embed(self, texts, kind="document", progress=None):
        return np.array([[1.0, 0.1] if ("downward" in t or "own" in t) else [0.1, 1.0] for t in texts])


def test_meaning_reorders_the_word_shortlist():
    net = net_with(PASSAGES)
    q = "how does my app make sure it knows its own pod name"
    words_first = net.search(q, k=1)[0][0][0]
    rr = ShortlistRerank(net, Meaning(), short=3)
    hits, reached, embedded = rr.search(q, k=1)
    assert words_first == "chatty.md#0"         # filler words win on word overlap alone
    assert hits[0][0] == "answer.md#0"          # meaning picks the page that answers
    assert embedded == reached == 2              # only what the words reached is embedded


def test_embeds_only_the_shortlist_and_only_once():
    net = net_with(PASSAGES)
    emb = TinyEmbedder()
    rr = ShortlistRerank(net, emb, short=2)
    _, _, first = rr.search("pod name namespace container", k=2)
    _, _, again = rr.search("pod name namespace container", k=2)
    assert first == 2 and again == 0            # never the whole corpus; a second visit is free
    assert rr.embedded == 2


def test_an_edited_passage_is_embedded_again():
    net = net_with(PASSAGES)
    rr = ShortlistRerank(net, TinyEmbedder(), short=3, mode="rrf")
    rr.search("pod name namespace", k=2)
    net.add("answer.md#0", "The downward API exposes the pod name, namespace and labels to the container.")
    _, _, embedded = rr.search("pod name namespace", k=2)
    assert embedded == 1                        # live: only the changed passage


def test_no_words_no_work():
    rr = ShortlistRerank(net_with(PASSAGES), TinyEmbedder())
    assert rr.search("zzzz qqqq", k=4) == ([], 0, 0)
