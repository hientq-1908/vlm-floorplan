"""Command line: `python -m vlm_floorplan.text2graph build-sets ...`."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_floorplan.text2graph.build import DEFAULT_SEED, BuildConfig, build_sets
from vlm_floorplan.text2graph.paraphrase import teacher_from_env
from vlm_floorplan.text2graph.pools import SourceFiles
from vlm_floorplan.text2graph.sets import PARAPHRASE_TARGET


def _train_texts(path: Path | None) -> list[str]:
    """Train prompts to check held-out text against: a JSONL whose `text` or messages hold them."""
    import json

    if path is None:
        return []
    texts = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "text" in row:
            texts.append(row["text"])
        for message in row.get("messages", []):
            if message.get("role") == "user":
                texts.append(message["content"].split("Request: ", 1)[-1])
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vlm_floorplan.text2graph")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-sets", help="build the held-out text-to-graph sets")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--resplan-dir", type=Path, help="dir with train/val/test.jsonl")
    build.add_argument("--cubicasa-file", type=Path)
    build.add_argument("--swiss-file", type=Path)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--paraphrase", action="store_true", help="use DEEPSEEK_* from the env")
    build.add_argument("--paraphrase-target", type=int, default=PARAPHRASE_TARGET)
    build.add_argument(
        "--train-prompts", type=Path, help="JSONL of train prompts for the text check"
    )
    args = parser.parse_args(argv)
    config = BuildConfig(
        out_dir=args.out,
        files=SourceFiles(args.resplan_dir, args.cubicasa_file, args.swiss_file),
        seed=args.seed,
        paraphrase_target=args.paraphrase_target,
        train_texts=_train_texts(args.train_prompts),
        teacher=teacher_from_env() if args.paraphrase else None,
    )
    manifest = build_sets(config)
    for name, summary in manifest["sets"].items():
        print(f"{name}: {summary['n']} prompts {summary['by_source']}")
    return 0
