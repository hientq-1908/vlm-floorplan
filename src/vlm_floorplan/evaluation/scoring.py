"""Phase 2: parse saved outputs, score them, and write `per_plan.jsonl` and `metrics.json`.

This reads only files, so it can be re-run (`rescore`) after improving the parser or
the metrics, without calling the model again.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from vlm_floorplan.evaluation.aggregate import aggregate, aggregate_by_size
from vlm_floorplan.evaluation.floors_json import to_floors_json
from vlm_floorplan.evaluation.harness import FAILURE_KEYS
from vlm_floorplan.evaluation.metrics import PlanMetrics, evaluate_plan
from vlm_floorplan.evaluation.parser import ParseError, ParseResult, parse_model_output
from vlm_floorplan.evaluation.run_dir import read_jsonl, write_manifest
from vlm_floorplan.record import PlanRecord


def _failure_key(output: dict[str, Any]) -> str | None:
    """`api_error` or `input_error` when the plan never got a reply; it is also the category."""
    return next((key for key in FAILURE_KEYS if key in output), None)


def _parse_row(output: dict[str, Any], record: PlanRecord) -> ParseResult:
    failure = _failure_key(output)
    if failure is not None:
        return ParseResult(graph=None, error=None, detail=output[failure])
    result = parse_model_output(output.get("content"), record.graph.image, output.get("reasoning"))
    # A reply cut off by max_tokens can still hold a parseable graph, but if it did not,
    # "truncated" is the true cause even when the text happens to lack an open brace.
    if (
        not result.ok
        and output.get("finish_reason") == "length"
        and result.error != ParseError.TRUNCATED
    ):
        return ParseResult(
            graph=None, error=ParseError.TRUNCATED, detail=f"hit max_tokens; {result.detail}"
        )
    return result


def _error_name(result: ParseResult, output: dict[str, Any]) -> str | None:
    if result.ok:
        return None
    return _failure_key(output) or str(result.error)


def _token_stats(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [row[key] for row in outputs if row.get(key) is not None]
        return round(sum(values) / len(values), 2) if values else None

    return {
        "mean_prompt_tokens": mean("prompt_tokens"),
        "mean_completion_tokens": mean("completion_tokens"),
        "mean_latency_s": mean("latency_s"),
        "finish_reasons": dict(Counter(row.get("finish_reason") for row in outputs)),
    }


def score_run(run_dir: Path, records: list[PlanRecord]) -> dict[str, Any]:
    """Score every plan in `outputs.jsonl` and write the result files."""
    by_id = {record.id: record for record in records}
    # A resumed run appends a retry after the failed row, so the last row per plan wins.
    latest = {row["plan_id"]: row for row in read_jsonl(run_dir / "outputs.jsonl")}
    outputs = [row for plan_id, row in latest.items() if plan_id in by_id]
    per_plan: list[dict[str, Any]] = []
    metrics: list[PlanMetrics] = []
    parse_results: list[ParseResult] = []
    for output in outputs:
        record = by_id[output["plan_id"]]
        result = _parse_row(output, record)
        plan_metrics = evaluate_plan(result.graph, record)
        metrics.append(plan_metrics)
        parse_results.append(result)
        per_plan.append(
            {
                "plan_id": record.id,
                "valid": result.ok,
                "error": _error_name(result, output),
                "detail": result.detail,
                "source": result.source,
                "lenient": result.lenient,
                "prediction": to_floors_json(result.graph) if result.graph else None,
                "metrics": plan_metrics.model_dump(),
            }
        )

    errors = Counter(row["error"] for row in per_plan if row["error"])
    summary = {
        "n_plans": len(per_plan),
        "n_requested": len(records),
        "overall": aggregate(metrics).model_dump(),
        "by_size": {k: v.model_dump() for k, v in aggregate_by_size(metrics).items()},
        "parse": {
            "error_counts": dict(errors),
            "strict_validity_rate": (
                sum(r.ok and not r.lenient for r in parse_results) / len(parse_results)
                if parse_results
                else None
            ),
            "sources": dict(Counter(str(r.source) for r in parse_results if r.ok)),
        },
        "generation": _token_stats(outputs),
    }
    (run_dir / "per_plan.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in per_plan)
    )
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_manifest(run_dir)
    return summary
