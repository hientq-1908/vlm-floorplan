"""Fetching, building and the CLI for MSD and Swiss Dwellings, against a local range server."""

from __future__ import annotations

import io
import os
import zipfile
from functools import partial
from pathlib import Path

import pytest
from msd_fixture import apartment_pickle
from range_server import serve
from swiss_fixture import apartment_rows, row, write_csv

from vlm_floorplan import cli
from vlm_floorplan.cli import main
from vlm_floorplan.converters.swiss_rows import iter_site_rows
from vlm_floorplan.data import http_zip
from vlm_floorplan.data.build import BuildOptions
from vlm_floorplan.data.build_msd import build_msd
from vlm_floorplan.data.build_swiss import build_swiss
from vlm_floorplan.data.fetch_msd import MsdSelection, fetch_msd
from vlm_floorplan.data.fetch_swiss import SWISS_MEMBER, fetch_swiss
from vlm_floorplan.data.jsonl import read_records

BIG = os.urandom(300_000)  # incompressible, so unwanted members dominate the zip's size


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_zip.time, "sleep", lambda _seconds: None)


def _msd_zip(plan_ids: tuple[str, ...] = ("0", "1", "2", "10")) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for plan_id in plan_ids:
            archive.writestr(f"graph_in/{plan_id}.pickle", b"zoning graph")
            archive.writestr(f"struct_in/{plan_id}.npy", BIG)
            archive.writestr(f"full_out/{plan_id}.npy", BIG)
            archive.writestr(f"graph_out/{plan_id}.pickle", apartment_pickle())
    return buffer.getvalue()


def _two_site_csv(tmp_path: Path) -> Path:
    rows = (
        apartment_rows("a1", site="s1")
        + apartment_rows("a2", site="s1")
        + [row("area", "STAIRCASE", (-3.2, 0, -0.2, 2), apartment="", site="s1")]
        + apartment_rows("a3", site="s2")
        + apartment_rows("a4", site="s3")
    )
    return write_csv(tmp_path / "source.csv", rows)


def _swiss_zip(csv_path: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("swiss-dwellings-v3.0.0/simulations.csv", BIG)
        archive.write(csv_path, SWISS_MEMBER)
    return buffer.getvalue()


def test_fetch_msd_downloads_only_graph_out_of_the_chosen_plans(tmp_path: Path) -> None:
    payload = _msd_zip()
    out = tmp_path / "msd"
    with serve(payload) as (url, state):
        count = fetch_msd(out, MsdSelection(plan_ids=("2",)), url=url)

    assert count == 1
    assert sorted(p.name for p in (out / "graph_out").iterdir()) == ["2.pickle"]
    assert (out / "ATTRIBUTION.txt").read_text().startswith("Modified Swiss Dwellings")
    assert state["bytes_sent"] < len(payload) / 2, "must not download the whole archive"


def test_fetch_msd_samples_reproducibly_and_can_take_everything(tmp_path: Path) -> None:
    payload = _msd_zip()
    with serve(payload) as (url, _state):
        fetch_msd(tmp_path / "a", MsdSelection(count=2, seed=5), url=url)
        fetch_msd(tmp_path / "b", MsdSelection(count=2, seed=5), url=url)
        fetch_msd(tmp_path / "all", MsdSelection(count=None), url=url)

    names = lambda d: sorted(p.name for p in (tmp_path / d / "graph_out").iterdir())  # noqa: E731
    assert names("a") == names("b") and len(names("a")) == 2
    assert names("all") == ["0.pickle", "1.pickle", "10.pickle", "2.pickle"]


@pytest.mark.parametrize("bad", ["../x", "1/2", "abc", "99"])
def test_fetch_msd_rejects_bad_or_unknown_plan_ids(tmp_path: Path, bad: str) -> None:
    with serve(_msd_zip()) as (url, _state), pytest.raises(ValueError):
        fetch_msd(tmp_path / "msd", MsdSelection(plan_ids=(bad,)), url=url)


def test_fetch_swiss_subset_stops_at_a_site_boundary(tmp_path: Path) -> None:
    payload = _swiss_zip(_two_site_csv(tmp_path))
    with serve(payload) as (url, state):
        path = fetch_swiss(tmp_path / "swiss", apartments=3, url=url)

    # 3 apartments are reached inside site s2 (a1, a2, a3): s2 is kept whole, s3 is not.
    sites = [rows[0]["site_id"] for rows in iter_site_rows(path)]
    assert sites == ["s1", "s2"]
    assert (tmp_path / "swiss" / "ATTRIBUTION.txt").exists()
    assert state["bytes_sent"] < len(payload), "simulations.csv must not be downloaded"


def test_fetch_swiss_all_copies_the_whole_file_and_keeps_an_existing_one(tmp_path: Path) -> None:
    source = _two_site_csv(tmp_path)
    with serve(_swiss_zip(source)) as (url, _state):
        path = fetch_swiss(tmp_path / "swiss", apartments=None, url=url)
        assert path.read_bytes() == source.read_bytes()
        path.write_text("kept")
        assert fetch_swiss(tmp_path / "swiss", apartments=None, url=url).read_text() == "kept"


def test_build_msd_writes_graph_only_records_and_reports_bad_plans(tmp_path: Path) -> None:
    graphs = tmp_path / "data" / "msd" / "graph_out"
    graphs.mkdir(parents=True)
    for plan_id in ("2", "10"):
        (graphs / f"{plan_id}.pickle").write_bytes(apartment_pickle())
    (graphs / "3.pickle").write_bytes(b"corrupt")

    report = build_msd(BuildOptions(data_dir=tmp_path / "data", out_dir=tmp_path / "out"))

    assert report.written == 2
    assert [plan_id for plan_id, _ in report.skipped] == ["3"]
    records = list(read_records(report.output))
    assert [r.id for r in records] == ["msd/2", "msd/10"]  # numeric order, not "10" < "2"
    assert all(r.image == "" and r.split is None for r in records)
    assert (tmp_path / "out" / "msd.ATTRIBUTION.txt").exists()


def test_build_msd_limit_is_a_seeded_sample(tmp_path: Path) -> None:
    graphs = tmp_path / "data" / "msd" / "graph_out"
    graphs.mkdir(parents=True)
    for plan_id in range(6):
        (graphs / f"{plan_id}.pickle").write_bytes(apartment_pickle())

    options = BuildOptions(data_dir=tmp_path / "data", out_dir=tmp_path / "out", limit=3, seed=1)
    first = [r.id for r in read_records(build_msd(options).output)]
    second = [r.id for r in read_records(build_msd(options).output)]

    assert len(first) == 3 and first == second


def test_build_swiss_writes_one_record_per_apartment_with_site_in_the_id(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_csv(data / "swiss_dwellings" / "geometries.csv", _csv_rows(tmp_path))

    report = build_swiss(BuildOptions(data_dir=data, out_dir=tmp_path / "out"))

    assert report.written == 4 and not report.skipped
    ids = [r.id for r in read_records(report.output)]
    assert ids == [
        f"swiss_dwellings/{s}/{a}"
        for s, a in [("s1", "a1"), ("s1", "a2"), ("s2", "a3"), ("s3", "a4")]
    ]
    assert (tmp_path / "out" / "swiss_dwellings.ATTRIBUTION.txt").exists()


def _csv_rows(tmp_path: Path) -> list[dict[str, str]]:
    import csv

    with _two_site_csv(tmp_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_swiss_limit_and_unreadable_apartments(tmp_path: Path) -> None:
    data = tmp_path / "data"
    rows = _csv_rows(tmp_path) + [
        row("area", "BEDROOM", (0, 0, 1, 1), apartment="bad", site="s4", wkt="POLYGON ((x")
    ]
    write_csv(data / "swiss_dwellings" / "geometries.csv", rows)

    everything = build_swiss(BuildOptions(data_dir=data, out_dir=tmp_path / "all"))
    sampled = build_swiss(BuildOptions(data_dir=data, out_dir=tmp_path / "some", limit=2, seed=3))

    assert everything.written == 4
    assert [key for key, _ in everything.skipped] == ["s4/bad"]
    # 2 of the 5 apartments (4 good, 1 unreadable) were chosen, whichever they were.
    assert sampled.written + len(sampled.skipped) == 2


def test_build_swiss_reports_a_site_whose_rows_are_not_consecutive(tmp_path: Path) -> None:
    data = tmp_path / "data"
    rows = apartment_rows("a1", site="s1") + apartment_rows("a2", site="s2")
    rows += apartment_rows("a3", site="s1")
    write_csv(data / "swiss_dwellings" / "geometries.csv", rows)

    report = build_swiss(BuildOptions(data_dir=data, out_dir=tmp_path / "out"))

    assert report.written == 2
    assert [key for key, _ in report.skipped] == ["site s1"]


def test_cli_fetch_and_build_msd_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    with serve(_msd_zip()) as (url, _state):
        monkeypatch.setattr(cli, "fetch_msd", partial(fetch_msd, url=url))
        assert main(["fetch", "msd", "--all", "--data-dir", str(data)]) == 0
    assert main(["build", "msd", "--data-dir", str(data), "--out-dir", str(tmp_path / "out")]) == 0

    assert "wrote 4 records" in capsys.readouterr().out
    assert len(list(read_records(tmp_path / "out" / "msd.jsonl"))) == 4


def test_cli_rejects_split_for_datasets_without_one(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no train/val/test split"):
        main(["build", "msd", "--split", "train", "--data-dir", str(tmp_path)])
