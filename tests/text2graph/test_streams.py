from vlm_floorplan.text2graph.streams import Streams


def test_the_same_keys_give_the_same_sequence():
    first = [Streams(7).rng("template_test", "resplan", "pool").random() for _ in range(2)]
    assert first[0] == first[1]


def test_a_stream_value_is_fixed_across_processes():
    # Pinned so a switch to Python's salted hash(), which changes per process, fails here.
    assert Streams(20261122).rng("template_test", "resplan", "pool").random() == 0.4199613292189578


def test_other_keys_or_another_seed_give_another_sequence():
    base = Streams(7).rng("template_test", "resplan", "pool").random()
    assert Streams(7).rng("template_val", "resplan", "pool").random() != base
    assert Streams(7).rng("template_test", "swiss_dwellings", "pool").random() != base
    assert Streams(8).rng("template_test", "resplan", "pool").random() != base


def test_keys_are_not_joined_ambiguously():
    assert Streams(7).sort_key("a", "bc") != Streams(7).sort_key("ab", "c")
    assert Streams(7).sort_key("a", "b") != Streams(7).sort_key("a:b")
