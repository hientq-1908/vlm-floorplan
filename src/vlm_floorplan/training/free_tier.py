"""The input files for the free-tier notebook (`notebooks/kaggle_sft.ipynb`).

The notebook needs the repository source and the training file, and a free notebook cannot clone
the private repository. This builds one directory with both, ready to upload as a private Kaggle
Dataset (`kaggle datasets create`) or to drop into a Google Drive folder for Colab:

- `vlm-floorplan-source.tar.gz`: `git archive HEAD`, filtered like the ClearML source
  (`source_pack.py`): no data, outputs or secret-looking files,
- `train.jsonl`: the training file, checked and copied as is,
- `dataset-metadata.json`: what `kaggle datasets create` reads (private, no license claimed).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from vlm_floorplan.evaluation.launcher import REPO_ROOT
from vlm_floorplan.training.source_pack import pack_repo_source

SOURCE_NAME = "vlm-floorplan-source.tar.gz"
TRAIN_NAME = "train.jsonl"
METADATA_NAME = "dataset-metadata.json"


def build_inputs(repo_root: Path, train_file: Path, out_dir: Path, dataset_id: str | None) -> Path:
    """Write the input directory and return it. `out_dir` must not hold files already."""
    if not train_file.is_file():
        raise FileNotFoundError(f"training file not found: {train_file}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already has files; pick an empty directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_repo_source(repo_root, out_dir / SOURCE_NAME)
    shutil.copyfile(train_file, out_dir / TRAIN_NAME)
    if dataset_id:
        owner, _, slug = dataset_id.partition("/")
        if not owner or not slug:
            raise ValueError(f"dataset id {dataset_id!r} must look like owner/slug")
        metadata = {
            "id": dataset_id,
            "title": slug.replace("-", " "),
            "licenses": [{"name": "other"}],
        }
        (out_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="a new or empty directory")
    parser.add_argument(
        "--dataset-id",
        help="OWNER/SLUG of the Kaggle Dataset; writes dataset-metadata.json (omit for Colab)",
    )
    args = parser.parse_args(argv)
    out = build_inputs(REPO_ROOT, args.train_file, args.out_dir, args.dataset_id)
    print(f"wrote {SOURCE_NAME} and {TRAIN_NAME} to {out}")
    return 0
