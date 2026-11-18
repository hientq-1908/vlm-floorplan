from __future__ import annotations

import hashlib
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
    assert offline == []
    assert (dest / fetch.PICKLE_NAME).read_bytes() == b"existing"
