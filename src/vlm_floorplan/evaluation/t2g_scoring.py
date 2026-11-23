"""Phase 2 of a text-to-graph run: score saved outputs, aggregate, write `metrics.json`.

Aggregation sums counts over prompts and divides once, as `aggregate.py` does; a ratio with an
empty denominator is `None`. Requirement satisfaction is micro-averaged over every stated
requirement of every draw prompt, and an output with no plan meets none of them.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vlm_floorplan.evaluation.plausibility import (
    PlanPlausibility,
    ReferenceTables,
    aggregate_plausibility,
    plan_plausibility,
)
from vlm_floorplan.evaluation.run_dir import read_jsonl, write_manifest
from vlm_floorplan.evaluation.t2g_score_one import score_row
from vlm_floorplan.text2graph.rows import (
    REFERENCES_FILE,
    TABLES_FILE,
    PromptRow,
    read_references,
)

_DRAW = ("draw", "restart")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _requirement_scores(scored: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    """Satisfaction and all-met for one reading (`met_strict` or `met_any`)."""
    stated = met = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_level: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    prompts = all_met = valid_prompts = valid_all_met = 0
    for row in scored:
        kinds = row["requirement_kinds"]
        results = row[key]
        count = len(kinds)
        got = sum(results) if results else 0
        stated, met = stated + count, met + got
        for kind, ok in zip(kinds, results or [False] * count, strict=True):
            by_type[kind][0] += 1
            by_type[kind][1] += ok
        level = by_level[str(row["level"])]
        level[0] += count
        level[1] += got
        if not count:
            continue
        done = bool(results) and all(results)
        prompts, all_met = prompts + 1, all_met + done
        valid_prompts, valid_all_met = valid_prompts + bool(results), valid_all_met + done
        level[2] += 1
        level[3] += done
        for table, name in ((by_source, row["source"]), (by_kind, row["kind"])):
            table[name][0] += 1
            table[name][1] += done
    return {
        "requirement_satisfaction": ratio(met, stated),
        "requirements_stated": stated,
        "all_met_rate": ratio(all_met, prompts),
        "all_met_prompts": prompts,
        "all_met_given_plan": ratio(valid_all_met, valid_prompts),
        "satisfaction_by_type": {
            k: {"met": v[1], "stated": v[0], "rate": ratio(v[1], v[0])}
            for k, v in sorted(by_type.items())
        },
        "by_level": {
            f"L{k}": {
                "requirement_satisfaction": ratio(v[1], v[0]),
                "all_met_rate": ratio(v[3], v[2]),
                "all_met_prompts": v[2],
            }
            for k, v in sorted(by_level.items())
        },
        "all_met_by_source": {
            k: {"rate": ratio(v[1], v[0]), "prompts": v[0]} for k, v in sorted(by_source.items())
        },
        "all_met_by_kind": {
            k: {"rate": ratio(v[1], v[0]), "prompts": v[0]} for k, v in sorted(by_kind.items())
        },
    }


def _behavior_scores(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[row["kind"]].append(row)
    result = {}
    for kind, rows in sorted(groups.items()):
        n = len(rows)
        entry: dict[str, Any] = {
            "n": n,
            "correct_rate": ratio(sum(bool(r["correct"]) for r in rows), n),
            "new_plan_rate": ratio(sum(r["called_new_plan"] for r in rows), n),
            "no_tool_call_rate": ratio(sum(r["no_tool_call"] for r in rows), n),
        }
        for flag in ("edit_tools_only", "edit_effect_ok", "asked_question", "names_issue"):
            if flag in rows[0]:
                entry[f"{flag}_rate"] = ratio(sum(bool(r.get(flag)) for r in rows), n)
        result[kind] = entry
    return result


def aggregate_run(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Dataset-level scores from per-prompt records."""
    draws = [row for row in scored if row["behavior"] in _DRAW]
    behaviors = [row for row in scored if row["behavior"] not in _DRAW]
    n = len(draws)
    strict_plans = [
        PlanPlausibility(**r["plausibility"])
        for r in draws
        if r["plausibility"] and r["plausibility_strict"]
    ]
    any_plans = [PlanPlausibility(**r["plausibility"]) for r in draws if r["plausibility"]]
    l4 = [r for r in draws if r["level"] == 4]
    scores = [r["plan_score"] or 0.0 for r in l4]
    got = [r["plan_score"] for r in l4 if r["plan_score"] is not None]
    summary: dict[str, Any] = {"n_prompts": len(scored), "n_draw_prompts": n}
    if draws:
        summary |= {
            "tool_call_rate": ratio(sum(r["called_new_plan"] for r in draws), n),
            "validity_rate": ratio(sum(r["strict_valid"] for r in draws), n),
            "lenient_plan_rate": ratio(sum(r["lenient_found"] for r in draws), n),
            "invalid_reasons": dict(
                Counter(r["reject_reason"] for r in draws if r["reject_reason"])
            ),
            "other_tool_rate": ratio(
                sum(any(t != "new_plan" for t in r["tools"]) for r in draws), n
            ),
            "api_errors": sum(bool(r.get("api_error")) for r in draws),
            "strict": _requirement_scores(draws, "met_strict"),
            "lenient": _requirement_scores(draws, "met_any"),
            "l4_plan_score_incl_invalid": ratio(sum(scores), len(scores)) if scores else None,
            "l4_plan_score_given_plan": ratio(sum(got), len(got)) if got else None,
            "plausibility": aggregate_plausibility(strict_plans),
            "plausibility_incl_lenient": aggregate_plausibility(any_plans),
        }
    if behaviors:
        summary["behavior"] = _behavior_scores(behaviors)
    return summary


def score_run(run_dir: Path, rows: list[PromptRow]) -> dict[str, Any]:
    """Score every prompt in `outputs.jsonl`, write `per_prompt.jsonl` and `metrics.json`."""
    config = json.loads((run_dir / "config.json").read_text())
    directory = Path(config["prompts"]).parent
    references = read_references(directory)
    tables_path = directory / TABLES_FILE
    tables = (
        ReferenceTables.model_validate_json(tables_path.read_text())
        if tables_path.exists()
        else ReferenceTables()
    )
    latest = {row["prompt_id"]: row for row in read_jsonl(run_dir / "outputs.jsonl")}
    by_id = {row.id: row for row in rows}
    outputs = {pid: out for pid, out in latest.items() if pid in by_id}
    scored = [
        score_row(by_id[pid], out, references.get(by_id[pid].record_id or ""), tables)
        for pid, out in outputs.items()
    ]
    real = {
        "n_plans": len({r.record_id for r in rows if r.record_id in references}),
        "plausibility": aggregate_plausibility(
            [
                plan_plausibility(references[rid].graph, tables)
                for rid in sorted({r.record_id for r in rows if r.record_id in references})
            ]
        ),
    }
    summary = aggregate_run(scored) | {
        "n_requested": len(rows),
        "real_reference": real,
        "generation": _token_stats(list(outputs.values())),
    }
    (run_dir / "per_prompt.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in scored)
    )
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_manifest(run_dir)
    return summary


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


__all__ = ["REFERENCES_FILE", "aggregate_run", "score_run"]
