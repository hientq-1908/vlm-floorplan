from __future__ import annotations

import hashlib
import io
import urllib.error
import zipfile
from pathlib import Path

import pytest

from vlm_floorplan.resplan import fetch

PICKLE_BYTES = b"pretend this is a pickle"


@pytest.fixture
def fake_zip(tmp_path: Path) -> Path:
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(fetch.PICKLE_NAME, PICKLE_BYTES)
    return path


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, fake_zip: Path) -> list[str]:
    """Replace the network: the zip URL serves `fake_zip`, REMOVED.md is a 404."""
    requested: list[str] = []

    def fake_download(url: str, target: Path) -> None:
        requested.append(url)
        if url == fetch.REMOVED_URL:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]
        target.write_bytes(fake_zip.read_bytes() if url == fetch.ZIP_URL else b"small file")

    monkeypatch.setattr(fetch, "_download", fake_download)
    return requested


def test_fetch_downloads_verifies_and_unpacks(
    tmp_path: Path, offline: list[str], fake_zip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "ZIP_SHA256", hashlib.sha256(fake_zip.read_bytes()).hexdigest())
    dest = tmp_path / "data"
    pickle_path = fetch.fetch_resplan(dest)
    assert pickle_path.read_bytes() == PICKLE_BYTES
    assert not (dest / fetch.ZIP_NAME).exists()
    for name in fetch.SMALL_FILES:
        assert (dest / name).exists()


def test_fetch_pins_every_download_to_the_commit(
    tmp_path: Path, offline: list[str], fake_zip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "ZIP_SHA256", hashlib.sha256(fake_zip.read_bytes()).hexdigest())
    fetch.fetch_resplan(tmp_path / "data")
    pinned = [url for url in offline if url != fetch.REMOVED_URL]
    assert pinned
    assert all(fetch.COMMIT in url for url in pinned)


def test_fetch_rejects_a_zip_with_the_wrong_checksum(tmp_path: Path, offline: list[str]) -> None:
    dest = tmp_path / "data"
    with pytest.raises(RuntimeError, match="SHA-256"):
        fetch.fetch_resplan(dest)
    assert not (dest / fetch.ZIP_NAME).exists()
    assert not (dest / fetch.PICKLE_NAME).exists()


def test_fetch_skips_a_download_that_is_already_there(tmp_path: Path, offline: list[str]) -> None:
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / fetch.PICKLE_NAME).write_bytes(b"existing")
    fetch.fetch_resplan(dest)
    assert offline == [fetch.REMOVED_URL]
    assert (dest / fetch.PICKLE_NAME).read_bytes() == b"existing"


def test_an_interrupted_extract_leaves_no_pickle_behind(
    tmp_path: Path, fake_zip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_midway(source: object, target: io.BufferedWriter, length: int) -> None:
        target.write(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(fetch.shutil, "copyfileobj", fail_midway)
    with pytest.raises(OSError, match="disk full"):
        fetch.extract_pickle(fake_zip, tmp_path)
    assert not (tmp_path / fetch.PICKLE_NAME).exists()


def test_download_retries_a_connection_reset_midstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    class FlakyResponse(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if len(attempts) == 1:
                raise ConnectionResetError("reset by peer")
            return super().read(size)

    def fake_urlopen(url: str, timeout: int) -> FlakyResponse:
        attempts.append(1)
        return FlakyResponse(b"payload")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fetch, "BACKOFF_SECONDS", 0)
    target = tmp_path / "file"
    fetch._download("https://example.invalid/file", target)
    assert len(attempts) == 2
    assert target.read_bytes() == b"payload"


def test_download_does_not_retry_a_local_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[str] = []

    def fake_urlopen(url: str, timeout: int) -> io.BytesIO:
        attempts.append(url)
        return io.BytesIO(b"payload")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fetch, "BACKOFF_SECONDS", 0)
    with pytest.raises(FileNotFoundError):
        fetch._download("https://example.invalid/file", tmp_path / "missing" / "file")
    assert len(attempts) == 1


def test_a_failed_recheck_keeps_the_saved_removed_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / fetch.PICKLE_NAME).write_bytes(b"existing")
    (dest / "REMOVED.md").write_text("123\n")

    def rate_limited(url: str, target: Path) -> None:
        target.write_bytes(b"partial")
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(fetch, "_download", rate_limited)
    fetch.fetch_resplan(dest)
    assert (dest / "REMOVED.md").read_text() == "123\n"
    assert not (dest / "REMOVED.md.partial").exists()
    assert "warning" in capsys.readouterr().out


def test_a_404_recheck_removes_a_stale_removed_list(tmp_path: Path, offline: list[str]) -> None:
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / fetch.PICKLE_NAME).write_bytes(b"existing")
    (dest / "REMOVED.md").write_text("123\n")
    fetch.fetch_resplan(dest)
    assert not (dest / "REMOVED.md").exists()


def test_a_failed_removed_check_does_not_fail_the_first_download(
    tmp_path: Path,
    fake_zip: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(fetch, "ZIP_SHA256", hashlib.sha256(fake_zip.read_bytes()).hexdigest())

    def fake_download(url: str, target: Path) -> None:
        if url == fetch.REMOVED_URL:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)  # type: ignore[arg-type]
        target.write_bytes(fake_zip.read_bytes() if url == fetch.ZIP_URL else b"small file")

    monkeypatch.setattr(fetch, "_download", fake_download)
    dest = tmp_path / "data"
    assert fetch.fetch_resplan(dest).read_bytes() == PICKLE_BYTES
    out = capsys.readouterr().out
    assert "warning" in out
    assert "ResPlan ready" in out
