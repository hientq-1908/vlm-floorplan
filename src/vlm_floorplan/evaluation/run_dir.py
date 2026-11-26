"""Run-directory helpers: stable file names and a checksum manifest.

The manifest lists the SHA-256 of every file in a directory. The job writes it before
anything leaves the GPU machine, and the launcher re-checks it after downloading, so a
partial or corrupted download is caught before the remote copy is deleted.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 1 << 20


def safe_name(plan_id: str) -> str:
    """A plan id such as "cubicasa5k/colorful/8759" as a flat, filesystem-safe name."""
    return "".join(c if c.isalnum() or c in "-_." else "__" for c in plan_id)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path) -> dict[str, str]:
    """Write `MANIFEST.json` (relative path -> sha256) for every file under `root`."""
    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    }
    (root / MANIFEST_NAME).write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
    return files


def bundle_results(out_dir: Path, archive: Path) -> Path:
    """Write the manifest, then zip the directory (the manifest is inside the zip)."""
    write_manifest(out_dir)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))
    return archive


def verify_manifest(root: Path) -> list[str]:
    """Problems found when comparing `root` with its manifest; empty means verified."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"{MANIFEST_NAME} is missing"]
    expected: dict[str, str] = json.loads(manifest_path.read_text())
    if not expected:
        return [f"{MANIFEST_NAME} lists no files"]
    problems = []
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256_file(path) != digest:
            problems.append(f"checksum mismatch: {name}")
    actual = {
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME
    }
    problems += [f"unlisted: {name}" for name in sorted(actual - set(expected))]
    return problems
