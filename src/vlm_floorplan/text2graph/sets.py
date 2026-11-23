"""Build the held-out sets: template test and val, paraphrased test, behavior test.

Every set is generated from held-out plans only (test or val splits, passing the leak index)
with a fixed seed; nothing generated is committed. The recipe follows the plan, section 5.1:

- template test and val: 600 prompts each, 200 ResPlan, 250 Swiss Dwellings, 150 CubiCasa5K
  (every qualifying plan, several levels each), levels in the 5 / 20 / 30 / 30 / 15 % mix;
- paraphrased test: 200 of the template test prompts rewritten by a language model, each
  passing the slot check and the blind round trip;
- behavior test: 100 single requests (edit, restart, contradiction, unsupported ask);
- hand-written: an empty file the captain fills (60 prompts), same row format.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor

from vlm_floorplan.requirements import apply_labels, derive_requirements, requirements_to_json
from vlm_floorplan.text2graph.leak import normalize_text
from vlm_floorplan.text2graph.paraphrase import Teacher
from vlm_floorplan.text2graph.pools import Candidate
from vlm_floorplan.text2graph.rows import PromptRow, empty_start
from vlm_floorplan.text2graph.templates import choose_kind, render_text
from vlm_floorplan.text2graph.words import slot_check

TEMPLATE_QUOTAS = {"resplan": 200, "swiss_dwellings": 250, "cubicasa5k": 150}
LEVEL_SHARES = {0: 0.05, 1: 0.20, 2: 0.30, 3: 0.30, 4: 0.15}
MIN_REQUIREMENTS = {0: 0, 1: 1, 2: 3, 3: 5, 4: 1}
"""Fewest requirements a prompt of each level may have; a plan that cannot reach it is retried."""
PARAPHRASE_TARGET = 200
PARAPHRASE_BATCH = 16
DERIVE_TRIES = 12
TEXT_TRIES = 10
_FULL_KINDS = ("full", "full_natural")


def level_sequence(n: int, rng: random.Random) -> list[int]:
    """`n` levels in the target mix, shuffled; the rounding remainder goes to L2."""
    counts = {level: round(n * share) for level, share in LEVEL_SHARES.items()}
    counts[2] += n - sum(counts.values())
    levels = [level for level, count in counts.items() for _ in range(count)]
    rng.shuffle(levels)
    return levels


def make_prompt(
    candidate: Candidate,
    level: int,
    set_name: str,
    number: int,
    rng: random.Random,
    forbidden: Collection[str] = (),
) -> PromptRow | None:
    """One template prompt for a plan at a level, or None if the plan cannot make one.

    `forbidden` holds normalized train prompt texts; a text equal to one is drawn again, so
    short generic prompts ("Draw a two-bedroom flat.") never repeat a train prompt.
    """
    graph = candidate.canon.record.graph
    for _ in range(DERIVE_TRIES):
        reqs = derive_requirements(graph, level, rng, candidate.source)
        if len(reqs) >= MIN_REQUIREMENTS[level]:
            break
    else:
        return None
    for _ in range(TEXT_TRIES):
        kind = choose_kind(level, reqs, rng)
        text = render_text(graph, reqs, kind, candidate.source, rng)
        if normalize_text(text) in forbidden:
            continue
        if kind in _FULL_KINDS or not slot_check(text, reqs):
            return PromptRow(
                id=f"{'T' if set_name.endswith('test') else 'V'}{number:04d}",
                set=set_name,
                source=candidate.source,
                record_id=candidate.record_id,
                level=level,
                kind=kind,
                text=text,
                requirements=requirements_to_json(reqs),
                start=empty_start(apply_labels(graph, reqs)),
            )
    return None


def _take(pool: Sequence[Candidate], quota: int, per_site_cap: int) -> list[Candidate]:
    """Up to `quota` plans, at most `per_site_cap` per Swiss site."""
    chosen, per_site = [], Counter()
    for candidate in pool:
        site = candidate.record_id.split("/")[1] if candidate.source == "swiss_dwellings" else None
        if site is not None and per_site[site] >= per_site_cap:
            continue
        per_site[site] += 1
        chosen.append(candidate)
        if len(chosen) >= quota:
            break
    return chosen


def build_template_set(
    set_name: str,
    pools: dict[str, list[Candidate]],
    rng: random.Random,
    quotas: dict[str, int] | None = None,
    forbidden: Collection[str] = (),
) -> list[PromptRow]:
    """The template prompts of one split. A source short of plans simply gives fewer prompts."""
    quotas = quotas or TEMPLATE_QUOTAS
    rows: list[PromptRow] = []
    for source, quota in quotas.items():
        pool = pools.get(source, [])
        if source == "cubicasa5k":
            rows += _multi_level(set_name, pool, quota, len(rows), rng, forbidden)
            continue
        cap = 1
        while len(_take(pool, quota, cap)) < min(quota, len(pool)):
            cap += 1
        plans = _take(pool, quota, cap)
        for plan, level in zip(plans, level_sequence(len(plans), rng), strict=True):
            row = make_prompt(plan, level, set_name, len(rows), rng, forbidden)
            if row:
                rows.append(row)
    return rows


def _multi_level(
    set_name: str,
    pool: Sequence[Candidate],
    quota: int,
    offset: int,
    rng: random.Random,
    forbidden: Collection[str] = (),
) -> list[PromptRow]:
    """Few plans, several prompts each at different levels, until the quota is met."""
    if not pool:
        return []
    base, extra = divmod(quota, len(pool))
    rows: list[PromptRow] = []
    for index, plan in enumerate(pool):
        wanted = min(4, base + (1 if index < extra else 0))
        for level in rng.sample([1, 2, 3, 4], wanted):
            row = make_prompt(plan, level, set_name, offset + len(rows), rng, forbidden)
            if row:
                rows.append(row)
    return rows


def _stratified(rows: list[PromptRow], rng: random.Random) -> list[PromptRow]:
    """Rows interleaved across (source, level) buckets, so any prefix is balanced."""
    buckets: dict[tuple[str, int | None], list[PromptRow]] = {}
    for row in rows:
        buckets.setdefault((row.source, row.level), []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered, keys = [], sorted(buckets, key=str)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop())
    return ordered


def build_paraphrase_set(
    template_rows: list[PromptRow],
    teacher: Teacher,
    rng: random.Random,
    target: int = PARAPHRASE_TARGET,
    forbidden: Collection[str] = (),
    on_batch: Callable[[int, int], None] | None = None,
) -> tuple[list[PromptRow], dict[str, object]]:
    """Paraphrase template prompts until `target` pass both checks.

    Prompts that carry a whole plan (levels 4) are not paraphrased. Returns the rows and the
    acceptance statistics (sent, accepted, personas, failure reasons).
    """
    candidates = _stratified([r for r in template_rows if r.kind not in _FULL_KINDS], rng)
    accepted: list[PromptRow] = []
    stats: dict[str, object] = {"sent": 0, "accepted": 0, "persona_accepted": 0}
    reasons: Counter[str] = Counter()
    seeds = {row.id: rng.randrange(2**32) for row in candidates}

    def work(row: PromptRow):
        local = random.Random(seeds[row.id])
        return row, teacher.checked_paraphrase(
            row.text, row.parsed_requirements(), local, forbidden=forbidden
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(candidates), PARAPHRASE_BATCH):
            if len(accepted) >= target:
                break
            for row, result in pool.map(work, candidates[start : start + PARAPHRASE_BATCH]):
                stats["sent"] += 1  # type: ignore[operator]
                reasons.update(reason.split(";")[0][:60] for reason in result.reasons)
                if result.text is None or len(accepted) >= target:
                    continue
                accepted.append(
                    row.model_copy(
                        update={
                            "id": f"P{len(accepted):04d}",
                            "set": "paraphrase_test",
                            "text": result.text,
                            "text_source": "paraphrase",
                            "paraphrase_of": row.id,
                        }
                    )
                )
                stats["persona_accepted"] += int(result.persona)  # type: ignore[operator]
            if on_batch:
                on_batch(len(accepted), int(stats["sent"]))  # type: ignore[call-overload]
    stats["accepted"] = len(accepted)
    stats["failure_reasons"] = dict(reasons.most_common(8))
    return accepted, stats
