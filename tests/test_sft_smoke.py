"""The smoke meter and the smoke slice, without a GPU."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from vlm_floorplan.training.data import Rendered  # noqa: E402
from vlm_floorplan.training.smoke import SmokeMeter, select_smoke_slice  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def run_steps(meter: SmokeMeter, clock: FakeClock, steps: list[tuple[float, int]]) -> None:
    """Each step is (seconds it took, total tokens seen after it)."""
    meter.on_train_begin(None, None, None)
    for seconds, tokens_seen in steps:
        clock.now += seconds
        meter.on_step_end(None, SimpleNamespace(num_input_tokens_seen=tokens_seen), None)
    meter.on_train_end(None, None, None)


def make_meter(tmp_path: Path, clock: FakeClock, gpu: dict | None = None) -> SmokeMeter:
    return SmokeMeter(
        tmp_path / "smoke.json", full_run_tokens=3_600_000, clock=clock, memory=lambda: gpu
    )


def test_tokens_per_second_skips_the_warm_up_step(tmp_path: Path) -> None:
    clock = FakeClock()
    meter = make_meter(tmp_path, clock)
    # step 1 is slow (warm-up); steps 2 and 3 move 2,000 tokens each in 2 s.
    run_steps(meter, clock, [(10.0, 1000), (2.0, 3000), (2.0, 5000)])
    result = json.loads((tmp_path / "smoke.json").read_text())
    assert result["tokens_per_second"] == 1000.0
    assert result["optimizer_steps"] == 3
    assert result["step_seconds"] == [10.0, 2.0, 2.0]
    assert result["tokens_total"] == 5000


def test_projection_uses_the_measured_speed(tmp_path: Path) -> None:
    clock = FakeClock()
    meter = make_meter(tmp_path, clock)
    run_steps(meter, clock, [(5.0, 500), (1.0, 1500)])  # 1,000 tokens per second
    result = json.loads((tmp_path / "smoke.json").read_text())
    assert result["projected_hours_per_epoch"] == 1.0  # 3.6M tokens at 1,000 per second


def test_single_step_falls_back_to_that_step(tmp_path: Path) -> None:
    clock = FakeClock()
    run_steps(make_meter(tmp_path, clock), clock, [(4.0, 2000)])
    assert json.loads((tmp_path / "smoke.json").read_text())["tokens_per_second"] == 500.0


def test_peak_gpu_memory_is_written_when_there_is_a_gpu(tmp_path: Path) -> None:
    clock = FakeClock()
    gpu = {
        "device": "RTX 3090",
        "peak_memory_allocated_gib": 17.5,
        "peak_memory_reserved_gib": 19.0,
    }
    run_steps(make_meter(tmp_path, clock, gpu), clock, [(1.0, 10), (1.0, 20)])
    assert json.loads((tmp_path / "smoke.json").read_text())["gpu"] == gpu


def test_no_gpu_is_recorded_as_null(tmp_path: Path) -> None:
    clock = FakeClock()
    run_steps(make_meter(tmp_path, clock), clock, [(1.0, 10), (1.0, 20)])
    assert json.loads((tmp_path / "smoke.json").read_text())["gpu"] is None


def test_a_run_without_steps_is_an_error(tmp_path: Path) -> None:
    clock = FakeClock()
    meter = make_meter(tmp_path, clock)
    meter.on_train_begin(None, None, None)
    with pytest.raises(RuntimeError, match="without a single optimizer step"):
        meter.on_train_end(None, None, None)


def item(name: str, length: int) -> Rendered:
    return Rendered(name, [0] * length, [0] * length)


def test_smoke_slice_always_contains_the_longest_sample() -> None:
    items = [item("a", 10), item("b", 20), item("c", 90), item("d", 30), item("e", 40)]
    chosen = select_smoke_slice(items, 3)
    assert [i.id for i in chosen] == ["c", "a", "b"]


def test_smoke_slice_of_a_small_set_is_the_whole_set() -> None:
    items = [item("a", 10), item("b", 20)]
    assert select_smoke_slice(items, 5) == items
