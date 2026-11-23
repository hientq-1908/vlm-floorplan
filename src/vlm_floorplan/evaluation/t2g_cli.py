"""Command line of the text-to-graph evaluation: `python -m vlm_floorplan.evaluation t2g-run`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vlm_floorplan.evaluation.client import HttpChatClient
from vlm_floorplan.evaluation.t2g_harness import T2gConfig, collect_outputs
from vlm_floorplan.evaluation.t2g_prompt import VARIANTS
from vlm_floorplan.evaluation.t2g_scoring import score_run
from vlm_floorplan.text2graph.rows import read_rows


def add_commands(sub: Any) -> None:
    run = sub.add_parser("t2g-run", help="text-to-graph: call the endpoint, then score")
    run.add_argument("--prompts", required=True, help="a set JSONL from `text2graph build-sets`")
    run.add_argument("--run-name", required=True)
    run.add_argument("--out-dir", type=Path, default=Path("runs"))
    run.add_argument("--model", required=True, help="served name: the base model or a LoRA module")
    run.add_argument("--variant", choices=VARIANTS, default="as-is")
    run.add_argument("--base-url", default="http://localhost:8000/v1")
    run.add_argument("--api-key-env")
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("--temperature", type=float, default=0.6)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--top-k", type=int, default=20, help="0 leaves it out of the request")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--thinking", choices=["on", "off", "server-default"], default="on")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--limit", type=int)
    rescore = sub.add_parser("t2g-rescore", help="score saved text-to-graph outputs again")
    rescore.add_argument("run_dir", type=Path)
    rescore.add_argument("--prompts", type=Path, help="set JSONL, if config.json's path is gone")


def headline(summary: dict[str, Any]) -> str:
    """The numbers worth reading first; the full breakdown is in metrics.json."""
    lines = [f"prompts: {summary['n_prompts']}"]
    if "tool_call_rate" in summary:
        strict = summary["strict"]
        lines += [
            f"tool-call rate (new_plan): {summary['tool_call_rate']:.3f}",
            f"validity (accepted by the tool): {summary['validity_rate']:.3f}",
            f"all-met (strict): {_fmt(strict['all_met_rate'])}"
            f"  (lenient: {_fmt(summary['lenient']['all_met_rate'])})",
        ]
        lines += [
            f"  {level}: {_fmt(v['all_met_rate'])} over {v['all_met_prompts']}"
            for level, v in strict["by_level"].items()
        ]
        plausibility = summary["plausibility"]
        real = summary["real_reference"]["plausibility"]
        for key in ("circulation_rate", "crossing_rate", "link_length_ratio_p95"):
            lines.append(f"{key}: {_fmt(plausibility[key])} (real {_fmt(real[key])})")
    for kind, entry in summary.get("behavior", {}).items():
        lines.append(f"behavior {kind}: correct {_fmt(entry['correct_rate'])} of {entry['n']}")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run_command(args: argparse.Namespace) -> int:
    if args.command == "t2g-rescore":
        config = T2gConfig.model_validate_json((args.run_dir / "config.json").read_text())
        prompts = args.prompts or Path(config.prompts)
        rows = read_rows(prompts)[: config.limit]
        run_dir, sets_dir = args.run_dir, prompts.parent
    else:
        config = T2gConfig(
            run_name=args.run_name,
            prompts=str(Path(args.prompts).resolve()),
            variant=args.variant,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k or None,
            seed=args.seed,
            enable_thinking={"on": True, "off": False, "server-default": None}[args.thinking],
            concurrency=args.concurrency,
            limit=args.limit,
        )
        run_dir = args.out_dir / config.run_name
        client = HttpChatClient(config.base_url, config.api_key_env, config.timeout_s)
        rows = collect_outputs(config, client, run_dir)
        sets_dir = None
    print(headline(score_run(run_dir, rows, sets_dir)))
    print(f"run directory: {run_dir}")
    return 0
