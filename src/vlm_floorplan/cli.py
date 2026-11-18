"""Command line: `uv run vlm-floorplan <fetch|build|view|sheet> ...`.

fetch cubicasa --count 20          download 20 random CubiCasa5K plans
build cubicasa --limit 50  convert fetched plans to JSONL
view  data/processed/x.jsonl --id cubicasa5k/colorful/8759 --out plan.png
sheet data/processed/x.jsonl --limit 24 --out-dir data/sheets/x
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from itertools import islice
from pathlib import Path

from PIL import Image

from vlm_floorplan.data.build import BuildOptions, build_cubicasa
from vlm_floorplan.data.fetch import CubiCasaSelection, fetch_cubicasa
from vlm_floorplan.data.jsonl import image_path, read_records
from vlm_floorplan.viewer.contact_sheet import write_contact_sheet
from vlm_floorplan.viewer.overlay import draw_overlay

DEFAULT_DATA_DIR = Path("data")


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-floorplan", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download a dataset into data/")
    fetch_sources = fetch.add_subparsers(dest="source", required=True)
    cubicasa = fetch_sources.add_parser(
        "cubicasa", help="CubiCasa5K, single plans by range request"
    )
    _add_data_dir(cubicasa)
    cubicasa.add_argument("--count", type=int, default=20)
    cubicasa.add_argument("--split", choices=["train", "val", "test"])
    cubicasa.add_argument("--seed", type=int, default=0)
    cubicasa.add_argument(
        "--plan",
        action="append",
        default=[],
        metavar="STYLE/ID",
        help="fetch this plan, e.g. colorful/8759 (repeatable)",
    )

    build = commands.add_parser("build", help="convert fetched plans to JSONL")
    build.add_argument("source", choices=["cubicasa"])
    _add_data_dir(build)
    build.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR / "processed")
    build.add_argument("--limit", type=int)
    build.add_argument("--split", choices=["train", "val", "test"])
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--drop-mostly-unlabeled", action="store_true")

    view = commands.add_parser("view", help="draw one plan's graph over its image (PNG)")
    view.add_argument("jsonl", type=Path)
    view.add_argument("--id", help="plan id; the first record if omitted")
    view.add_argument("--out", type=Path, default=Path("overlay.png"))

    sheet = commands.add_parser("sheet", help="HTML contact sheet of many plans")
    sheet.add_argument("jsonl", type=Path)
    sheet.add_argument("--limit", type=int, default=24)
    sheet.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR / "sheets" / "latest")
    return parser


def _run_fetch(args: argparse.Namespace) -> None:
    selection = CubiCasaSelection(
        plan_ids=tuple(args.plan), count=args.count, split=args.split, seed=args.seed
    )
    count = fetch_cubicasa(args.data_dir / "cubicasa5k", selection)
    print(f"CubiCasa5K: {count} plans in {args.data_dir / 'cubicasa5k'} (CC BY-NC 4.0)")


def _run_build(args: argparse.Namespace) -> None:
    options = BuildOptions(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        split=args.split,
        seed=args.seed,
        drop_mostly_unlabeled=args.drop_mostly_unlabeled,
    )
    print(build_cubicasa(options).summary())


def _run_view(args: argparse.Namespace) -> None:
    for record in read_records(args.jsonl):
        if args.id is None or record.id == args.id:
            with Image.open(image_path(args.jsonl, record)) as image:
                draw_overlay(image, record.graph).save(args.out)
            print(f"{record.id}: wrote {args.out}")
            return
    raise SystemExit(f"no record with id {args.id!r} in {args.jsonl}")


def _run_sheet(args: argparse.Namespace) -> None:
    records = list(islice(read_records(args.jsonl), args.limit))
    index = write_contact_sheet(args.jsonl, records, args.out_dir)
    print(f"{len(records)} plans: open {index}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runners = {"fetch": _run_fetch, "build": _run_build, "view": _run_view, "sheet": _run_sheet}
    runners[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
