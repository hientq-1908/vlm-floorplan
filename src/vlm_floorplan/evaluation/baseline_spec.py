"""What the zero-shot baseline job is, and how the eval split travels to the GPU machine.

Design choice: ClearML is compute only. The split (JSONL plus images) is packed into
one archive and attached to the task as an artifact, instead of living in a ClearML
dataset. When the task is deleted at the end, the split goes with it, and nothing of
ours stays on the ClearML server.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vlm_floorplan.evaluation.harness import load_records

RUNS_DIR_ENV = "VLM_FLOORPLAN_RUNS_DIR"
DEFAULT_RUNS_DIR = Path("~/Documents/pets/vlm-floorplan-runs")
SPLIT_ARTIFACT = "eval_split"
RESULT_ARTIFACT = "baseline_run"
SPLIT_FILE_IN_ARCHIVE = "split.jsonl"


def default_runs_dir() -> Path:
    """Where finished runs are kept locally: `$VLM_FLOORPLAN_RUNS_DIR` or the default."""
    return Path(os.environ.get(RUNS_DIR_ENV) or DEFAULT_RUNS_DIR).expanduser()


class BaselineSpec(BaseModel):
    """Everything that defines one baseline launch. Passed to the job as task parameters."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    project: str = "vlm-floorplan"
    queue: str = "3090"
    model: str = "Qwen/Qwen3.5-4B"
    served_name: str = "qwen3.5-4b"
    resolutions: list[int] = Field(default_factory=lambda: [768, 1280], min_length=1)
    """Longest image side in pixels, one evaluation pass per value (the M2 comparison)."""
    max_tokens: int = 12288
    limit: int | None = None
    concurrency: int = 8
    docker_image: str | None = None

    def job_parameters(self) -> dict[str, str]:
        """Flat string parameters, the form ClearML hyperparameters take."""
        return {
            "model": self.model,
            "served_name": self.served_name,
            "resolutions": ",".join(str(r) for r in self.resolutions),
            "max_tokens": str(self.max_tokens),
            "limit": "" if self.limit is None else str(self.limit),
            "concurrency": str(self.concurrency),
            "run_name": self.run_name,
        }


def pack_split(split: Path, archive: Path, limit: int | None = None) -> int:
    """Pack a split JSONL and the images it refers to into a `.tar.gz`; returns plan count.

    Paths inside the archive mirror the originals relative to the JSONL's directory, so
    `PlanRecord.image` stays valid after unpacking.
    """
    records = load_records(split, limit)
    root = split.resolve().parent
    with tarfile.open(archive, "w:gz") as tar:
        lines = [record.model_dump_json() for record in records]
        data = ("\n".join(lines) + "\n").encode()
        info = tarfile.TarInfo(SPLIT_FILE_IN_ARCHIVE)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for record in records:
            image = (root / record.image).resolve()
            if not image.is_relative_to(root):
                raise ValueError(
                    f"{record.id}: image path {record.image!r} leaves the split directory"
                )
            tar.add(image, arcname=str(image.relative_to(root)))
    return len(records)


def unpack_split(archive: Path, dest: Path) -> Path:
    """Unpack a split archive; returns the path of its JSONL."""
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")  # "data" refuses absolute paths and `..`
    return dest / SPLIT_FILE_IN_ARCHIVE
