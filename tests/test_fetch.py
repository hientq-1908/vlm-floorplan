"""Fetching CubiCasa5K plans by range request, against a local server (no network needed)."""

from __future__ import annotations

import io
import os
import urllib.error
import zipfile
from pathlib import Path

import pytest
from range_server import serve

from vlm_floorplan.data import http_zip
from vlm_floorplan.data.fetch import CubiCasaSelection, fetch_cubicasa
from vlm_floorplan.data.http_zip import HttpRangeFile, RangeNotSupportedError

PLAN_IDS = ["colorful/1", "colorful/2", "high_quality/3"]
BIG = os.urandom(300_000)  # incompressible, so it dominates the zip's size


def _zip_bytes(tmp_path: Path) -> bytes:
    from cubicasa_fixture import write_cubicasa_plan

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for plan_id in PLAN_IDS:
            plan = write_cubicasa_plan(tmp_path / "src" / plan_id)
            for name in ("model.svg", "F1_scaled.png"):
                archive.write(plan / name, f"cubicasa5k/{plan_id}/{name}")
            archive.writestr(f"cubicasa5k/{plan_id}/F1_original.png", BIG)
        archive.writestr("cubicasa5k/train.txt", "/colorful/1/\n/colorful/2/\n")
        archive.writestr("cubicasa5k/val.txt", "/high_quality/3/\n")
        archive.writestr("cubicasa5k/test.txt", "")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_zip.time, "sleep", lambda _seconds: None)


def test_range_file_reads_arbitrary_slices() -> None:
    data = bytes(range(256)) * 4000
    with serve(data) as (url, _state):
        remote = HttpRangeFile(url)
        remote.seek(1000)
        assert remote.read(10) == data[1000:1010]
        remote.seek(-5, io.SEEK_END)
        assert remote.read() == data[-5:]
        assert remote.tell() == len(data)


def test_fetch_downloads_only_selected_plans_and_only_needed_files(tmp_path: Path) -> None:
    payload = _zip_bytes(tmp_path)
    out = tmp_path / "data"
    with serve(payload) as (url, state):
        count = fetch_cubicasa(out, CubiCasaSelection(plan_ids=("colorful/2",)), url=url)
    assert count == 1
    assert (out / "colorful/2/model.svg").exists()
    assert (out / "colorful/2/F1_scaled.png").exists()
    assert not (out / "colorful/2/F1_original.png").exists()
    assert not (out / "colorful/1").exists()
    assert (out / "train.txt").read_text().startswith("/colorful/1/")
    assert state["bytes_sent"] < len(payload) / 2, "range requests must not download the whole zip"


def test_fetch_is_resumable_and_skips_existing_files(tmp_path: Path) -> None:
    payload = _zip_bytes(tmp_path)
    out = tmp_path / "data"
    selection = CubiCasaSelection(plan_ids=("colorful/1",))
    with serve(payload) as (url, _state):
        fetch_cubicasa(out, selection, url=url)
        marker = out / "colorful/1/model.svg"
        marker.write_text("local edit")
        fetch_cubicasa(out, selection, url=url)
    assert marker.read_text() == "local edit"


def test_random_selection_is_reproducible_and_respects_the_split(tmp_path: Path) -> None:
    payload = _zip_bytes(tmp_path)
    with serve(payload) as (url, _state), zipfile.ZipFile(HttpRangeFile(url)) as archive:
        from vlm_floorplan.data.fetch import choose_cubicasa_ids

        first = choose_cubicasa_ids(archive, CubiCasaSelection(count=2, seed=7))
        again = choose_cubicasa_ids(archive, CubiCasaSelection(count=2, seed=7))
        val = choose_cubicasa_ids(archive, CubiCasaSelection(count=5, split="val"))
    assert first == again and len(first) == 2
    assert val == ["high_quality/3"]


@pytest.mark.parametrize("plan_id", ["../evil", "colorful/1/../../x", "colorful", "/abs/1"])
def test_plan_ids_cannot_escape_the_data_directory(tmp_path: Path, plan_id: str) -> None:
    payload = _zip_bytes(tmp_path)
    with serve(payload) as (url, _state), pytest.raises(ValueError, match="plan id"):
        fetch_cubicasa(tmp_path / "data", CubiCasaSelection(plan_ids=(plan_id,)), url=url)


def test_transient_server_errors_are_retried() -> None:
    with serve(b"x" * 100, fail_first=2) as (url, state):
        assert HttpRangeFile(url).read(3) == b"xxx"
    assert state["requests"] >= 3


def test_persistent_server_errors_are_raised() -> None:
    with serve(b"x", fail_first=100) as (url, _state), pytest.raises(urllib.error.HTTPError):
        HttpRangeFile(url)


def test_truncated_responses_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body cut short (http.client.IncompleteRead) is transient, like a 503."""
    import http.client

    real = http_zip.urllib.request.urlopen
    calls = {"n": 0}

    def flaky(request: object, timeout: float) -> object:
        response = real(request, timeout=timeout)
        calls["n"] += 1
        if calls["n"] == 1:
            response.read = lambda: (_ for _ in ()).throw(http.client.IncompleteRead(b"x"))  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(http_zip.urllib.request, "urlopen", flaky)
    with serve(b"abcdef") as (url, _state):
        assert HttpRangeFile(url).read(3) == b"abc"
    assert calls["n"] >= 2


def test_server_without_range_support_is_reported() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Plain(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", "1")
            self.end_headers()
            self.wfile.write(b"x")

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Plain)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RangeNotSupportedError):
            HttpRangeFile(f"http://127.0.0.1:{server.server_port}/")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.network
def test_real_zenodo_plan_can_be_fetched(tmp_path: Path) -> None:
    """Hits Zenodo for one real plan. Skip offline with `-m "not network"`."""
    try:
        count = fetch_cubicasa(tmp_path, CubiCasaSelection(plan_ids=("colorful/8759",)))
    except OSError as err:
        pytest.skip(f"Zenodo unreachable: {err}")
    assert count == 1
    assert (tmp_path / "colorful/8759/model.svg").stat().st_size > 1000
