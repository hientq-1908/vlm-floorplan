"""Download ResPlan into the gitignored `data/` directory.

    uv run python -m vlm_floorplan.resplan.fetch [--dest data/resplan]

The download is pinned to one commit of the upstream repository and the zip is
checked against a SHA-256, so every run gets the same bytes and a force-pushed
or tampered file is rejected. To move to a newer release, update `COMMIT` and
`ZIP_SHA256` together (see "Takedowns" in docs/resplan.md).

ResPlan: https://github.com/m-agour/ResPlan (data CC BY 4.0, code MIT).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "m-agour/ResPlan"
COMMIT = "e2b78fe069aee1ab1e1828a612743f308e3c32a7"
"""Upstream commit of 2026-07-28 that `ZIP_SHA256` was computed from."""
ZIP_SHA256 = "f718de8865e51bbe93b49b584798e3c536ed6b4b8a5d32f01b56812f389aeb46"

ZIP_URL = f"https://github.com/{REPO}/raw/{COMMIT}/ResPlan.zip"
ZIP_NAME = "ResPlan.zip"
PICKLE_NAME = "ResPlan.pkl"
SMALL_FILES = ("LICENSE", "TAKEDOWN.md", "split.json")
"""Kept next to the data: the licence and takedown terms travel with the plans."""
REMOVED_URL = f"https://raw.githubusercontent.com/{REPO}/main/REMOVED.md"
"""ResPlan lists removed plan ids here once it has honored a takedown request."""

DEFAULT_DEST = Path("data/resplan")
TIMEOUT_SECONDS = 120
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 5
CHUNK_BYTES = 1 << 20


def _download(url: str, target: Path) -> None:
    """Stream `url` to `target`, retrying on transient errors (GitHub sometimes returns 503)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with (
                urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, CHUNK_BYTES)
            return
        except (OSError, http.client.HTTPException) as err:
            if isinstance(err, urllib.error.HTTPError) and err.code < 500:
                raise
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"download of {url} failed after {attempt} attempts") from err
            time.sleep(BACKOFF_SECONDS * attempt)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pickle(zip_path: Path, dest: Path) -> Path:
    """Copy `ResPlan.pkl` out of the zip. Only that one member is read, never its other paths."""
    target = dest / PICKLE_NAME
    partial = dest / f"{PICKLE_NAME}.partial"
    with (
        zipfile.ZipFile(zip_path) as archive,
        archive.open(PICKLE_NAME) as member,
        partial.open("wb") as out,
    ):
        shutil.copyfileobj(member, out, CHUNK_BYTES)
    os.replace(partial, target)
    return target


def _note_removed_plans(dest: Path) -> None:
    """Fetch upstream's current removal list, if it has one, so takedowns can be honored."""
    target = dest / "REMOVED.md"
    try:
        _download(REMOVED_URL, target)
    except urllib.error.HTTPError as err:
        target.unlink(missing_ok=True)
        if err.code != 404:
            raise
        print("upstream lists no removed plans (REMOVED.md not found)")
        return
    print(
        f"upstream lists removed plans: see {target}.\n"
        "Put their ids in a file (one per line) and pass it to the builder with --exclude-ids."
    )


def fetch_resplan(dest: Path = DEFAULT_DEST, force: bool = False) -> Path:
    """Download and unpack ResPlan into `dest`. Returns the path of `ResPlan.pkl`."""
    dest.mkdir(parents=True, exist_ok=True)
    pickle_path = dest / PICKLE_NAME
    if pickle_path.exists() and not force:
        print(f"{pickle_path} already exists (use --force to download again)")
        return pickle_path

    for name in SMALL_FILES:
        _download(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}", dest / name)
    zip_path = dest / ZIP_NAME
    _download(ZIP_URL, zip_path)
    actual = sha256_of(zip_path)
    if actual != ZIP_SHA256:
        zip_path.unlink()
        raise RuntimeError(f"{ZIP_NAME} has SHA-256 {actual}, expected {ZIP_SHA256}")
    extract_pickle(zip_path, dest)
    zip_path.unlink()
    _note_removed_plans(dest)
    print(f"ResPlan ready in {dest}. Data is CC BY 4.0: keep the attribution (see README.md).")
    return pickle_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the ResPlan dataset.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="download again even if present")
    args = parser.parse_args()
    fetch_resplan(args.dest, args.force)


if __name__ == "__main__":
    main()
