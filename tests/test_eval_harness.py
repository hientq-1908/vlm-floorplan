from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from helpers_eval import gt_record
from PIL import Image

from vlm_floorplan.compact import to_compact
from vlm_floorplan.evaluation.cli import main as cli_main
from vlm_floorplan.evaluation.client import ApiError, ChatResponse, parse_chat_response
from vlm_floorplan.evaluation.config import EvalConfig
from vlm_floorplan.evaluation.harness import collect_outputs
from vlm_floorplan.evaluation.run_dir import verify_manifest
from vlm_floorplan.evaluation.scoring import score_run


class FakeClient:
    """Answers with the ground-truth graph, or fails for chosen call numbers."""

    def __init__(self, replies: list[ChatResponse | Exception]):
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    def complete(self, request: dict[str, Any]) -> ChatResponse:
        self.requests.append(request)
        reply = self.replies[len(self.requests) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


def reply(content: str, finish: str = "stop") -> ChatResponse:
    return ChatResponse(
        content=content,
        reasoning="some thinking",
        finish_reason=finish,
        prompt_tokens=500,
        completion_tokens=300,
        latency_s=1.5,
    )


@pytest.fixture
def split(tmp_path: Path) -> Path:
    """A one-plan split: JSONL plus a 2000x1000 image beside it."""
    Image.new("RGB", (2000, 1000), "white").save(tmp_path / "plan0.png")
    (tmp_path / "split.jsonl").write_text(gt_record().model_dump_json() + "\n")
    return tmp_path / "split.jsonl"


def config_for(split: Path, **overrides: Any) -> EvalConfig:
    values: dict[str, Any] = {
        "run_name": "test-run",
        "split": str(split),
        "model": "fake-model",
        "concurrency": 1,
    }
    return EvalConfig(**{**values, **overrides})


def test_run_records_inputs_outputs_and_metrics(split: Path, tmp_path: Path) -> None:
    config = config_for(split, max_long_side=768)
    client = FakeClient([reply(to_compact(gt_record().graph))])
    run_dir = tmp_path / "runs" / "test-run"

    records = collect_outputs(config, client, run_dir)
    summary = score_run(run_dir, records)

    request = client.requests[0]
    assert request["chat_template_kwargs"] == {"enable_thinking": True}
    assert request["model"] == "fake-model"
    sent_image = Image.open(run_dir / "inputs" / "images" / "synthetic__plan0.png")
    assert sent_image.size == (768, 384)  # long side shrunk to 768, aspect kept

    inputs = [json.loads(line) for line in (run_dir / "inputs.jsonl").read_text().splitlines()]
    assert inputs[0]["original_size"] == [2000, 1000]
    assert inputs[0]["sent_size"] == [768, 384]
    assert "base64" not in json.dumps(inputs[0])  # the image is stored as a file, not inline
    outputs = [json.loads(line) for line in (run_dir / "outputs.jsonl").read_text().splitlines()]
    assert outputs[0]["content"] == to_compact(gt_record().graph)
    assert outputs[0]["reasoning"] == "some thinking"
    assert (run_dir / "prompt.txt").read_text().startswith("You are given a floorplan image")

    assert summary["overall"]["validity_rate"] == 1.0
    assert summary["overall"]["room_recall"] == 1.0
    assert summary["parse"]["strict_validity_rate"] == 1.0
    assert json.loads((run_dir / "metrics.json").read_text())["n_plans"] == 1
    assert verify_manifest(run_dir) == []


def test_images_are_never_enlarged(split: Path, tmp_path: Path) -> None:
    config = config_for(split, max_long_side=4000)
    run_dir = tmp_path / "run"
    collect_outputs(config, FakeClient([reply("{}")]), run_dir)
    assert Image.open(run_dir / "inputs" / "images" / "synthetic__plan0.png").size == (2000, 1000)


def test_thinking_default_sends_nothing_and_top_k_can_be_dropped(
    split: Path, tmp_path: Path
) -> None:
    config = config_for(split, enable_thinking=None, top_k=None, seed=None)
    client = FakeClient([reply("{}")])
    collect_outputs(config, client, tmp_path / "run")
    assert "chat_template_kwargs" not in client.requests[0]
    assert "top_k" not in client.requests[0]
    assert "seed" not in client.requests[0]


def test_api_failure_is_recorded_and_counts_as_invalid(split: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    records = collect_outputs(config_for(split), FakeClient([ApiError("HTTP 500")]), run_dir)
    summary = score_run(run_dir, records)
    assert summary["overall"]["validity_rate"] == 0.0
    assert summary["parse"]["error_counts"] == {"api_error": 1}


def test_output_cut_off_by_max_tokens_is_reported_as_truncated(split: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cut = reply("I am still thinking about the kitchen", finish="length")
    records = collect_outputs(config_for(split), FakeClient([cut]), run_dir)
    assert score_run(run_dir, records)["parse"]["error_counts"] == {"truncated": 1}


def test_rerun_resumes_without_calling_the_model_again(split: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = config_for(split)
    collect_outputs(config, FakeClient([reply("{}")]), run_dir)
    second = FakeClient([])
    collect_outputs(config, second, run_dir)
    assert second.requests == []
    assert len((run_dir / "outputs.jsonl").read_text().splitlines()) == 1


def test_resume_retries_failed_plans_and_scores_only_the_retry(split: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = config_for(split)
    collect_outputs(config, FakeClient([ApiError("server restarting")]), run_dir)
    retry = FakeClient([reply(to_compact(gt_record().graph))])
    records = collect_outputs(config, retry, run_dir)
    assert len(retry.requests) == 1
    assert len((run_dir / "inputs.jsonl").read_text().splitlines()) == 1
    summary = score_run(run_dir, records)
    assert summary["n_plans"] == 1
    assert summary["overall"]["validity_rate"] == 1.0
    assert summary["parse"]["error_counts"] == {}


def test_unreadable_image_is_recorded_per_plan_not_fatal(split: Path, tmp_path: Path) -> None:
    second = gt_record().model_copy(update={"id": "synthetic/broken", "image": "broken.png"})
    (split.parent / "broken.png").write_bytes(b"not a png")
    with split.open("a") as handle:
        handle.write(second.model_dump_json() + "\n")
    run_dir = tmp_path / "run"
    client = FakeClient([reply(to_compact(gt_record().graph))])
    records = collect_outputs(config_for(split), client, run_dir)
    summary = score_run(run_dir, records)
    assert len(client.requests) == 1
    assert summary["n_plans"] == 2
    assert summary["overall"]["validity_rate"] == 0.5
    assert summary["parse"]["error_counts"] == {"input_error": 1}


def test_existing_run_dir_with_other_settings_is_refused(split: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    collect_outputs(config_for(split, max_long_side=768), FakeClient([reply("{}")]), run_dir)
    with pytest.raises(ValueError, match="different settings"):
        collect_outputs(config_for(split, max_long_side=1280), FakeClient([]), run_dir)


def test_resume_allows_a_new_server_url_and_concurrency(split: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    collect_outputs(config_for(split), FakeClient([reply("{}")]), run_dir)
    moved = config_for(split, base_url="http://other:9000/v1", concurrency=8)
    collect_outputs(moved, FakeClient([]), run_dir)


def test_resume_without_limit_lets_rescore_cover_every_plan(split: Path, tmp_path: Path) -> None:
    second = gt_record().model_copy(update={"id": "synthetic/second"})
    with split.open("a") as handle:
        handle.write(second.model_dump_json() + "\n")
    out_dir = tmp_path / "runs"
    answer = reply(to_compact(gt_record().graph))
    collect_outputs(config_for(split, limit=1), FakeClient([answer]), out_dir / "test-run")
    collect_outputs(config_for(split), FakeClient([answer]), out_dir / "test-run")
    assert cli_main(["rescore", str(out_dir / "test-run")]) == 0
    metrics = json.loads((out_dir / "test-run" / "metrics.json").read_text())
    assert metrics["n_plans"] == 2


def test_rescoring_uses_saved_outputs_only(split: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = config_for(split)
    records = collect_outputs(config, FakeClient([reply(to_compact(gt_record().graph))]), run_dir)
    first = score_run(run_dir, records)
    (run_dir / "metrics.json").unlink()
    assert score_run(run_dir, records)["overall"] == first["overall"]


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_both_vllm_reasoning_field_names_are_read(field: str) -> None:
    body = {
        "choices": [{"message": {"content": "answer", field: "thoughts"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    parsed = parse_chat_response(body, 0.5)
    assert (parsed.content, parsed.reasoning, parsed.completion_tokens) == (
        "answer",
        "thoughts",
        20,
    )


def test_malformed_response_raises_api_error() -> None:
    with pytest.raises(ApiError):
        parse_chat_response({"error": "nope"}, 0.1)
