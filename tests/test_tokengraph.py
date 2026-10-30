import numpy as np

from mapnet.tokengraph import TokenGraph


def build():
    """Three topics (axes a, b, c). 'alpha' and 'beta' never share a passage, but both are used in
    topic-a passages, so by context they are neighbours; 'gamma' lives in topic b."""
    vec = {}
    texts = {
        "a1#0": "alpha one", "a2#0": "alpha two", "a3#0": "beta three", "a4#0": "beta four",
        "b1#0": "gamma five", "b2#0": "gamma six", "c1#0": "delta seven", "c2#0": "delta eight",
    }
    axis = {"a": 0, "b": 1, "c": 2}
    rng = np.random.default_rng(0)
    for d in texts:
        v = np.zeros(3, dtype=np.float32)
        v[axis[d[0]]] = 1.0
        vec[d] = v + 0.02 * rng.normal(size=3).astype(np.float32)
    net = TokenGraph(lambda ids: np.vstack([vec[i] for i in ids]), max_common=1.0, min_sim=0.5)
    for d, t in texts.items():
        net.add(d, t)
    return net


def test_words_used_in_the_same_contexts_are_neighbours():
    net = build()
    near = dict(net._neighbours("alpha"))
    assert "beta" in near and "gamma" not in near and "delta" not in near


def test_a_question_word_reaches_pages_it_never_appears_in():
    net = build()
    plain = [d for d, _, _ in net.search("alpha", k=6, prox=False, expand=False)[0]]
    wide = [d for d, _, _ in net.search("alpha", k=6, prox=False, expand=True)[0]]
    assert set(plain) == {"a1#0", "a2#0"}
    assert {"a3#0", "a4#0"} <= set(wide) and not ({"b1#0", "c1#0"} & set(wide))
    assert net.last_expansion["beta"] == [["alpha"]]  # inspectable: beta was lit by the question word alpha


def test_new_files_update_the_links():
    net = build()
    net._neighbours("alpha")
    net.add("a5#0", "epsilon nine")
    net.add("a6#0", "epsilon ten")
    net.vectors = lambda ids: np.vstack([
        np.array([1, 0, 0], dtype=np.float32) if i[0] == "a" else
        np.array([0, 1, 0], dtype=np.float32) if i[0] == "b" else np.array([0, 0, 1], dtype=np.float32)
        for i in ids])
    assert "epsilon" in dict(net._neighbours("alpha"))


def converging():
    """Topic a has the question words alpha and omega plus 'target'; 'solo' sits in a different,
    nearby context and is reached by alpha only."""
    vec, texts = {}, {}
    rng = np.random.default_rng(1)
    def put(d, text, v):
        texts[d] = text
        vec[d] = np.asarray(v, dtype=np.float32) + 0.01 * rng.normal(size=3).astype(np.float32)
    put("a1#0", "alpha one", [1, 0, 0]); put("a2#0", "alpha two", [1, 0, 0])
    put("a3#0", "omega three", [1, 0, 0]); put("a4#0", "omega four", [1, 0, 0])
    put("a5#0", "target five", [1, 0, 0]); put("a6#0", "target six", [1, 0, 0])
    put("s1#0", "solo seven", [0.9, 0.45, 0]); put("s2#0", "solo eight alpha", [0.9, 0.45, 0])
    put("b1#0", "gamma nine", [0, 1, 0]); put("b2#0", "gamma ten", [0, 1, 0])
    put("c1#0", "delta eleven", [0, 0, 1]); put("c2#0", "delta twelve", [0, 0, 1])
    net = TokenGraph(lambda ids: np.vstack([vec[i] for i in ids]), max_common=1.0, specific=0.0,
                     expand_weight=0.5, min_sim=0.5)
    for d, t in texts.items():
        net.add(d, t)
    return net


def test_a_word_two_question_words_agree_on_lights_up_and_a_lone_one_does_not():
    net = converging()
    net.search("alpha omega", k=5, prox=False, expand=True, converge=2, link_n=6)
    lit = set(net.last_expansion)
    assert "target" in lit and {"alpha", "omega"} <= set(sum(net.last_expansion["target"], []))
    assert "solo" not in lit  # only alpha reaches it
    net.search("alpha omega", k=5, prox=False, expand=True, converge=1, link_n=6)
    assert "solo" in set(net.last_expansion)  # without the gate it lights too


def test_a_single_question_word_cannot_converge():
    net = converging()
    hits, _ = net.search("alpha", k=6, prox=False, expand=True, converge=2, link_n=6)
    assert net.last_expansion == {}
    assert {d for d, _, _ in hits} == {"a1#0", "a2#0", "s2#0"}


def test_second_hop_reaches_words_through_a_middle_word():
    net = converging()
    net.search("alpha", k=3, prox=False, expand=True, converge=1, hops=1, link_n=1)
    one = set(net.last_expansion)
    net.search("alpha", k=3, prox=False, expand=True, converge=1, hops=2, link_n=1)
    assert one == {"omega"} and "target" in set(net.last_expansion)  # alpha -> omega -> target
