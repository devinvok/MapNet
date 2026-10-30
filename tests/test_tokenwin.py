from mapnet.tokenwin import PosTokenNet


def corpus(net):
    net.add("a#0", "The pod is killed after the grace period ends and the pod stops.")
    net.add("b#0", "The grace period is set per pod. A killed process cannot clean up.")
    net.add("c#0", "Rolling updates replace pods gradually with a surge of extra pods.")
    net.add("d#0", "Grace period seconds control how long to wait before the pod is stopped by force.")
    net.add("e#0", "Volumes are mounted into the container at the given path.")


def test_adjacent_words_beat_scattered_words():
    net = PosTokenNet()
    net.add("near#0", "Set the rolling update surge value for the deployment strategy now.")
    net.add("far#0", "Rolling is a word. " + "Filler text goes here. " * 12 + "The surge value is separate.")
    plain = [d for d, _, _ in net.search("rolling surge", k=2, prox=False, expand=False)[0]]
    close = [d for d, _, _ in net.search("rolling surge", k=2, prox=True, expand=False)[0]]
    assert close[0] == "near#0" and set(plain) == set(close)


def test_neighbours_are_the_words_that_sit_beside_a_word():
    net = PosTokenNet(window=3, min_pair=2, max_common=1.0)
    corpus(net)
    near = {t for t, _ in net.window_neighbours("kill")}
    assert "kill" not in near and ("grace" in near or "period" in near or "pod" in near)
    assert net.window_neighbours("nonexistent") == []


def test_a_question_word_reaches_pages_through_its_neighbours():
    net = PosTokenNet(window=4, min_pair=2, expand_weight=0.6, max_common=1.0)
    for i in range(3):
        net.add(f"p{i}#0", "Stop the process killed after the grace period ends. Grace period seconds apply.")
    net.add("target#0", "Grace period seconds control how long to wait. The period is configurable.")
    net.add("noise#0", "Volumes are mounted into the container at the given path.")
    plain = [d for d, _, _ in net.search("killed", k=5, prox=False, expand=False)[0]]
    wide = [d for d, _, _ in net.search("killed", k=5, prox=False, expand=True)[0]]
    assert "target#0" not in plain and "target#0" in wide
    assert any("kill" in srcs for lst in net.last_expansion.values() for srcs in lst)  # inspectable: which question word lit it


def test_adding_and_removing_keeps_positions_in_step():
    net = PosTokenNet()
    corpus(net)
    net.search("pod", k=2)
    net.add("f#0", "Zebra launch happens in May with the new pod.")
    assert net.search("zebra launch", k=1)[0][0][0] == "f#0"
    net.remove_missing({"a#0", "f#0"})
    assert len(net._seqs) == len(net.ids) == 2
    assert net.search("zebra launch", k=1)[0][0][0] == "f#0"
