"""Command line: `uv run python -m vlm_floorplan.evaluation run|rescore ...`."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_floorplan.evaluation.baseline_spec import SPLIT_FILE_IN_ARCHIVE
from vlm_floorplan.evaluation.client import HttpChatClient
from vlm_floorplan.evaluation.config import EvalConfig
from vlm_floorplan.evaluation.harness import collect_outputs, load_records
from vlm_floorplan.evaluation.scoring import score_run
from vlm_floorplan.evaluation.t2g_cli import add_commands, run_command


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", required=True, help="JSONL of PlanRecords")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--model", required=True, help="model name the endpoint serves")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", help="environment variable that holds the API key")
    parser.add_argument("--max-long-side", type=int, default=1024, help="768, 1280, ...")
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20, help="0 leaves it out of the request")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thinking",
        choices=["on", "off", "server-default"],
        default="on",
        help="chat_template_kwargs.enable_thinking; not every endpoint accepts it",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-s", type=float, default=600.0)


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    thinking = {"on": True, "off": False, "server-default": None}[args.thinking]
    return EvalConfig(
        run_name=args.run_name,
        split=str(Path(args.split).resolve()),
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        max_long_side=args.max_long_side,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k or None,
        seed=args.seed,
        enable_thinking=thinking,
        concurrency=args.concurrency,
        limit=args.limit,
        timeout_s=args.timeout_s,
    )


def headline(summary: dict) -> str:
    """The few numbers worth reading first; the full breakdown is in metrics.json."""
    overall = summary["overall"]
    any_links = overall["links"]["any"]["end_to_end"]
    rows = {
        "plans (valid / total)": f"{overall['n_valid']} / {overall['n_plans']}",
        "room recall": overall["room_recall"],
        "room precision": overall["room_precision"],
        "room-type accuracy": overall["type_accuracy"],
        "link F1 (any kind, end to end)": any_links["f1"],
        "entrance accuracy": overall["entrance_accuracy"],
        "parse errors": summary["parse"]["error_counts"] or "none",
    }
    return "\n".join(
        f"{name:34} {value:.3f}" if isinstance(value, float) else f"{name:34} {value}"
        for name, value in rows.items()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vlm_floorplan.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_arguments(sub.add_parser("run", help="call the endpoint, then score"))
    rescore = sub.add_parser("rescore", help="score saved outputs again, no model calls")
    rescore.add_argument("run_dir", type=Path)
    rescore.add_argument(
        "--split",
        type=Path,
        help="split JSONL, or the unpacked split directory, when config.json's path is gone "
        "(a collected ClearML run records the agent's temporary copy); config.json is unchanged",
    )
    add_commands(sub)
    args = parser.parse_args(argv)

    if args.command in ("t2g-run", "t2g-rescore"):
        return run_command(args)
    if args.command == "rescore":
        config = EvalConfig.model_validate_json((args.run_dir / "config.json").read_text())
        split = args.split or Path(config.split)
        if split.is_dir():
            split = split / SPLIT_FILE_IN_ARCHIVE
        records = load_records(split, config.limit)
        run_dir = args.run_dir
    else:
        config = config_from_args(args)
        run_dir = args.out_dir / config.run_name
        client = HttpChatClient(config.base_url, config.api_key_env, config.timeout_s)
        records = collect_outputs(config, client, run_dir)
    summary = score_run(run_dir, records)
    print(headline(summary))
    print(f"run directory: {run_dir}")
    return 0
