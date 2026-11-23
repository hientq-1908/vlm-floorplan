"""Build every held-out set into one directory, with its leak check and reference tables.

    uv run python -m vlm_floorplan.text2graph build-sets --out DIR \
        --resplan-dir ... --cubicasa-file ... --swiss-file ... [--paraphrase]

The directory holds the five set files, `references.jsonl`, `reference_tables.json` and
`manifest.json`. It is generated, never committed, and the same seed gives the same sets
(the paraphrases excepted: they come from a language model, so they are saved, not regenerated).
"""

from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vlm_floorplan.evaluation.plausibility import (
    aggregate_plausibility,
    build_reference_tables,
    plan_plausibility,
)
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.text2graph.behavior import BEHAVIOR_QUOTAS, behavior_counts, build_behavior_rows
from vlm_floorplan.text2graph.frame import unusable_reason
from vlm_floorplan.text2graph.leak import LeakError, LeakIndex
from vlm_floorplan.text2graph.paraphrase import Teacher
from vlm_floorplan.text2graph.pools import (
    SourceFiles,
    build_leak_index,
    held_out_candidates,
    read_records,
)
from vlm_floorplan.text2graph.rows import (
    MANIFEST_FILE,
    REFERENCES_FILE,
    TABLES_FILE,
    PromptRow,
    write_jsonl,
)
from vlm_floorplan.text2graph.sets import (
    PARAPHRASE_TARGET,
    TEMPLATE_QUOTAS,
    build_paraphrase_set,
    build_template_set,
)
from vlm_floorplan.text2graph.splits import split_of

DEFAULT_SEED = 20261122
HANDWRITTEN_NOTE = "empty slot: the captain writes 60 prompts (see docs/text2graph-evaluation.md)"


@dataclass
class BuildConfig:
    out_dir: Path
    files: SourceFiles
    seed: int = DEFAULT_SEED
    quotas: dict[str, int] = field(default_factory=lambda: dict(TEMPLATE_QUOTAS))
    behavior_quotas: dict[str, int] = field(default_factory=lambda: dict(BEHAVIOR_QUOTAS))
    paraphrase_target: int = PARAPHRASE_TARGET
    train_texts: list[str] = field(default_factory=list)
    teacher: Teacher | None = None
    log: Callable[[str], None] = print


def verify_no_leak(rows: Iterable[PromptRow], references: dict[str, PlanRecord], index: LeakIndex):
    """Raise `LeakError` if any prompt overlaps the train data; returns the number checked."""
    checked, problems = 0, []
    for row in rows:
        if row.record_id is None:
            continue
        checked += 1
        found = index.violations(row.record_id, references[row.record_id].graph, row.text)
        problems += [f"{row.id} ({row.record_id}): {item}" for item in found]
    if problems:
        raise LeakError("; ".join(problems[:5]))
    return checked


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _summary(rows: list[PromptRow]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "by_source": dict(Counter(r.source for r in rows)),
        "by_level": dict(sorted(Counter(str(r.level) for r in rows).items())),
        "by_kind": dict(sorted(Counter(r.kind for r in rows).items())),
    }


def real_reference(references: dict[str, PlanRecord], rows: list[PromptRow], tables) -> dict:
    """Plausibility of the source plans of the template test prompts, overall and per source."""
    ids = sorted({r.record_id for r in rows if r.record_id})
    scored = {i: plan_plausibility(references[i].graph, tables) for i in ids}
    result = {"overall": aggregate_plausibility(list(scored.values()))}
    for source in sorted({references[i].source for i in ids}):
        subset = [scored[i] for i in ids if references[i].source == source]
        result[source] = aggregate_plausibility(subset)
    return result


def build_sets(config: BuildConfig) -> dict[str, Any]:
    """Write all sets to `config.out_dir` and return the manifest."""
    log, out = config.log, config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = config.files.paths()
    log(f"indexing train plans from {len(paths)} files")
    index = build_leak_index(read_records(paths), config.train_texts)
    log(f"train index: {len(index.ids)} plans, {len(index.structures)} structures")
    rng = random.Random(config.seed)
    rows: dict[str, list[PromptRow]] = {}
    references: dict[str, PlanRecord] = {}
    behavior_plans = []
    for split, name in (("test", "template_test"), ("val", "template_val")):
        pools = held_out_candidates(read_records(paths), split, index, rng)
        log(f"{split}: candidates " + ", ".join(f"{s} {len(p)}" for s, p in pools.items()))
        rows[name] = build_template_set(name, pools, rng, config.quotas, index.texts)
        for pool in pools.values():
            for candidate in pool:
                references[candidate.record_id] = candidate.canon.record
        if split == "test":
            used = {r.record_id for r in rows[name]}
            behavior_plans = [
                (c.record_id, c.source, c.canon.record.graph)
                for pool in pools.values()
                for c in pool
                if c.record_id not in used
            ] or [(c.record_id, c.source, c.canon.record.graph) for p in pools.values() for c in p]
        log(f"{name}: {len(rows[name])} prompts")
    rows["behavior_test"] = build_behavior_rows(
        behavior_plans, rng, config.behavior_quotas, index.texts
    )
    rows["paraphrase_test"], teacher_stats = [], {"skipped": "no teacher given"}
    if config.teacher is not None:
        rows["paraphrase_test"], teacher_stats = build_paraphrase_set(
            rows["template_test"],
            config.teacher,
            rng,
            config.paraphrase_target,
            lambda ok, sent: log(f"paraphrase: {ok} accepted of {sent} sent"),
        )
    rows["handwritten_test"] = []
    all_rows = [row for set_rows in rows.values() for row in set_rows]
    checked = verify_no_leak(all_rows, references, index)
    log(f"leak check passed on {checked} prompts")

    used_ids = {row.record_id for row in all_rows if row.record_id}
    for name, set_rows in rows.items():
        write_jsonl(out / f"{name}.jsonl", set_rows)
    write_jsonl(out / REFERENCES_FILE, [references[i] for i in sorted(used_ids)])
    train_graphs = [
        r.graph
        for r in read_records(paths)
        if split_of(r) == "train" and unusable_reason(r) is None
    ]
    tables = build_reference_tables(train_graphs)
    (out / TABLES_FILE).write_text(tables.model_dump_json(indent=1) + "\n")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": config.seed,
        "git_commit": _git_commit(),
        "sets": {name: _summary(set_rows) for name, set_rows in rows.items()},
        "behavior_kinds": behavior_counts(rows["behavior_test"]),
        "leak_check": {"prompts_checked": checked, "violations": 0, "train_plans": len(index.ids)},
        "paraphrase": teacher_stats,
        "handwritten_test": HANDWRITTEN_NOTE,
        "real_reference": real_reference(references, rows["template_test"], tables),
        "reference_tables": {"train_plans": tables.n_plans, "train_links": tables.n_links},
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest
