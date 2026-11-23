import json
import random
from pathlib import Path

import pytest
from t2g_fixtures import make_record, write_split

from vlm_floorplan.schema import LinkKind, RoomType
from vlm_floorplan.text2graph.build import BuildConfig, build_sets, verify_no_leak
from vlm_floorplan.text2graph.frame import canonicalize
from vlm_floorplan.text2graph.leak import LeakError, LeakIndex, structure_hash
from vlm_floorplan.text2graph.pools import SourceFiles, build_leak_index, held_out_candidates
from vlm_floorplan.text2graph.rows import read_references, read_rows
from vlm_floorplan.text2graph.sets import level_sequence, make_prompt
from vlm_floorplan.text2graph.splits import swiss_split


def test_swiss_split_is_by_site_and_matches_the_stage_one_training_set():
    # Sites drawn into the 1,000-sample stage-1 set are all train.
    assert {swiss_split(site) for site in ("100", "1005", "3406", "4250")} == {"train"}
    assert swiss_split("210") == "test"
    buckets = [swiss_split(str(site)) for site in range(2000)]
    assert 0.03 < buckets.count("val") / 2000 < 0.08
    assert 0.03 < buckets.count("test") / 2000 < 0.08


def test_structure_hash_ignores_positions_but_not_types_or_links():
    a = make_record(1).graph
    moved = a.model_copy(deep=True)
    moved.floors[0].rooms[0].xy = (1, 1)
    assert structure_hash(a) == structure_hash(moved)
    retyped = a.model_copy(deep=True)
    retyped.floors[0].rooms[0].type = RoomType.GARAGE
    assert structure_hash(a) != structure_hash(retyped)
    relinked = a.model_copy(deep=True)
    relinked.floors[0].links[0].kind = (
        LinkKind.WALL if a.floors[0].links[0].kind != LinkKind.WALL else LinkKind.DOOR
    )
    assert structure_hash(a) != structure_hash(relinked)


def test_leak_index_flags_ids_sites_structures_and_texts():
    train = make_record(1, "swiss_dwellings")
    train.id = "swiss_dwellings/55/a"
    index = LeakIndex()
    index.add_train_plan(train.id, train.graph)
    index.add_train_texts(["Draw a  small flat."])
    other = make_record(2).graph
    assert index.violations("resplan/9", other, "something else") == []
    assert "site holds a train apartment" in index.violations("swiss_dwellings/55/b", other)
    assert "record id is a train record" in index.violations(train.id, other)
    assert "structure seen in train" in index.violations("resplan/9", train.graph)
    assert "text equals a train prompt" in index.violations(
        "resplan/9", other, "draw a small flat."
    )


def test_verify_no_leak_raises_on_an_overlap():
    record = make_record(3)
    canon = canonicalize(record).record
    row = make_prompt(
        type(
            "C",
            (),
            {
                "canon": type("K", (), {"record": canon})(),
                "source": "resplan",
                "record_id": canon.id,
            },
        )(),
        1,
        "template_test",
        0,
        random.Random(0),
    )
    index = LeakIndex()
    index.add_train_plan(canon.id, canon.graph)
    with pytest.raises(LeakError):
        verify_no_leak([row], {canon.id: canon}, index)


def test_level_sequence_follows_the_mix():
    levels = level_sequence(200, random.Random(0))
    assert len(levels) == 200
    assert [levels.count(i) for i in range(5)] == [10, 40, 60, 60, 30]


def _files(tmp_path: Path) -> SourceFiles:
    train = [make_record(seed, "resplan", "train") for seed in range(60)]
    val = [make_record(seed, "resplan", "val") for seed in range(1000, 1060)]
    test = [make_record(seed, "resplan", "test") for seed in range(2000, 2060)]
    directory = tmp_path / "resplan"
    directory.mkdir()
    write_split(directory / "train.jsonl", train)
    write_split(directory / "val.jsonl", val)
    write_split(directory / "test.jsonl", test)
    return SourceFiles(resplan_dir=directory)


def test_held_out_candidates_never_share_a_structure_with_train(tmp_path):
    files = _files(tmp_path)
    from vlm_floorplan.text2graph.pools import read_records

    index = build_leak_index(read_records(files.paths()))
    pools = held_out_candidates(read_records(files.paths()), "test", index, random.Random(0))
    assert pools["resplan"]
    for candidate in pools["resplan"]:
        assert candidate.record_id not in index.ids
        assert structure_hash(candidate.canon.record.graph) not in index.structures
    ids = [c.record_id for c in pools["resplan"]]
    assert all(i.startswith("resplan/2") for i in ids)


def test_build_sets_writes_every_set_from_held_out_plans_only(tmp_path):
    out = tmp_path / "sets"
    config = BuildConfig(
        out_dir=out,
        files=_files(tmp_path),
        quotas={"resplan": 12},
        behavior_quotas={"edit": 3, "restart": 2, "contradiction": 2, "unsupported": 2},
        log=lambda _message: None,
    )
    manifest = build_sets(config)
    assert manifest["leak_check"]["violations"] == 0
    for name in ("template_test", "template_val", "paraphrase_test", "behavior_test"):
        assert (out / f"{name}.jsonl").exists()
    assert (out / "handwritten_test.jsonl").read_text() == ""
    test_rows = read_rows(out / "template_test.jsonl")
    val_rows = read_rows(out / "template_val.jsonl")
    assert {r.record_id[:9] for r in test_rows} == {"resplan/2"}
    assert {r.record_id[:9] for r in val_rows} == {"resplan/1"}
    assert set(read_references(out)) >= {r.record_id for r in test_rows}
    assert json.loads((out / "manifest.json").read_text())["seed"] == config.seed


def test_the_same_seed_builds_the_same_sets(tmp_path):
    files = _files(tmp_path)
    texts = []
    for name in ("one", "two"):
        config = BuildConfig(
            out_dir=tmp_path / name,
            files=files,
            quotas={"resplan": 10},
            behavior_quotas={"edit": 3, "restart": 1, "contradiction": 1, "unsupported": 1},
            log=lambda _message: None,
        )
        build_sets(config)
        texts.append((tmp_path / name / "template_test.jsonl").read_text())
    assert texts[0] == texts[1]
