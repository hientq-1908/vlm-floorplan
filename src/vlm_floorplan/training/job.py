"""The SFT job that runs on the GPU agent, without the ClearML glue.

`scripts/sft_job.py` does the ClearML part (read parameters, fetch the training file, upload the
result). This module does the rest, so it can be tested without ClearML:

1. build the config from the committed TOML file, the launch overrides and the agent's paths,
2. run `run_training` (full or smoke) and never let an exception skip the bundle,
3. write `job_summary.json` and zip the whole run directory with a SHA-256 manifest.

`check_sft_result` is the collector's side: it decides whether the downloaded result is complete
enough to delete the ClearML task.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vlm_floorplan.evaluation.collect import DownloadVerificationError, check_manifest
from vlm_floorplan.evaluation.run_dir import bundle_results
from vlm_floorplan.training.config import TrainConfig, load_config
from vlm_floorplan.training.train import (
    ADAPTER_DIR,
    DATA_REPORT_FILE,
    SMOKE_FILE,
    SUMMARY_FILE,
    run_training,
)

log = logging.getLogger("sft_job")

TRAIN_ARTIFACT = "sft_train"
RESULT_ARTIFACT = "sft_run"
SUMMARY = "job_summary.json"
RUN_SUBDIR = "run"
"""Where the training run directory sits inside the result archive."""


class SftSpec(BaseModel):
    """Everything that defines one SFT launch. Passed to the job as task parameters."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    project: str = "vlm-floorplan"
    queue: str = "3090"
    config: str = "configs/sft_stage1.toml"
    """Config file, relative to the repository root (the agent reads it from the commit)."""
    overrides: list[str] = Field(default_factory=list)
    """`section.key=value` strings applied on top of the config file."""
    smoke: bool = False
    docker_image: str | None = None

    def job_parameters(self) -> dict[str, str]:
        """Flat string parameters, the form ClearML hyperparameters take."""
        return {
            "run_name": self.run_name,
            "config": self.config,
            "overrides": json.dumps(self.overrides),
            "smoke": "true" if self.smoke else "false",
        }


def spec_from_parameters(params: dict[str, str]) -> SftSpec:
    """Rebuild the spec from the flat string parameters the launcher attached."""
    return SftSpec(
        run_name=params["run_name"],
        config=params["config"],
        overrides=json.loads(params["overrides"]),
        smoke=params["smoke"] == "true",
    )


def agent_config(spec: SftSpec, repo_root: Path, train_file: Path, run_dir: Path) -> TrainConfig:
    """The config the agent runs: the launch overrides, then the agent's own paths."""
    overrides = [
        *spec.overrides,
        f"data.train_file={json.dumps(str(train_file))}",
        f"train.output_dir={json.dumps(str(run_dir))}",
    ]
    if not spec.smoke:
        overrides.append('train.report_to="clearml"')
    return load_config(repo_root / spec.config, overrides)


def run_job(
    spec: SftSpec, repo_root: Path, train_file: Path, out_dir: Path, archive: Path
) -> dict[str, Any]:
    """Train (or smoke-test), then bundle `out_dir` into `archive`. The bundle is always written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(out_dir / "job.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    result: dict[str, Any] = {
        "mode": "smoke" if spec.smoke else "train",
        "ok": False,
        "error": None,
    }
    try:
        config = agent_config(spec, repo_root, train_file, out_dir / RUN_SUBDIR)
        run_training(config, smoke=spec.smoke)
        result["ok"] = True
    except Exception as err:  # the bundle below must still be written
        log.exception("training failed")
        result["error"] = f"{type(err).__name__}: {err}"
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
    (out_dir / SUMMARY).write_text(
        json.dumps({**result, "spec": spec.model_dump()}, indent=2) + "\n"
    )
    bundle_results(out_dir, archive)
    return result


def check_sft_result(result_dir: Path) -> list[str]:
    """Collector check: the manifest matches and a successful job left its outputs.

    A job that failed (`ok` false) is only checked for its manifest; its logs are the point.
    """
    checks = check_manifest(result_dir)
    summary_path = result_dir / SUMMARY
    if not summary_path.is_file():
        raise DownloadVerificationError(f"{SUMMARY} is missing from the result")
    summary = json.loads(summary_path.read_text())
    if not summary.get("ok"):
        return [*checks, f"job did not succeed ({summary.get('error')}); logs kept"]
    run = result_dir / RUN_SUBDIR
    if summary.get("mode") == "smoke":
        required = [run / "smoke" / SMOKE_FILE]
    else:
        adapter = run / ADAPTER_DIR
        required = [
            adapter / "adapter_config.json",
            adapter / "adapter_model.safetensors",
            run / SUMMARY_FILE,
            run / DATA_REPORT_FILE,
        ]
    for path in required:
        if not path.is_file():
            raise DownloadVerificationError(f"{path.relative_to(result_dir)} is missing")
        checks.append(f"{path.relative_to(result_dir)} present")
    return checks
