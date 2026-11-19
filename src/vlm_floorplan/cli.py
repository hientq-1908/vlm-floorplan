"""Command line: `uv run vlm-floorplan <fetch|build|view|sheet> ...`.

fetch cubicasa --count 20          download 20 random CubiCasa5K plans
fetch msd --count 50               download 50 random MSD plans (--all for all 4167)
fetch swiss --apartments 100       download the first ~100 Swiss Dwellings apartments (--all)
build cubicasa|msd|swiss --limit 50  convert fetched plans to JSONL
view  data/processed/x.jsonl --id cubicasa5k/colorful/8759 --out plan.png
sheet data/processed/x.jsonl --limit 24 --out-dir data/sheets/x
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from itertools import islice
from pathlib import Path

from PIL import Image

from vlm_floorplan.data.build import BuildOptions, BuildReport, build_cubicasa
from vlm_floorplan.data.build_msd import build_msd
from vlm_floorplan.data.build_swiss import build_swiss
from vlm_floorplan.data.fetch import CubiCasaSelection, fetch_cubicasa
from vlm_floorplan.data.fetch_msd import MsdSelection, fetch_msd
from vlm_floorplan.data.fetch_swiss import fetch_swiss
from vlm_floorplan.data.jsonl import image_path, read_records
from vlm_floorplan.viewer.contact_sheet import write_contact_sheet
from vlm_floorplan.viewer.overlay import draw_overlay

DEFAULT_DATA_DIR = Path("data")

_BUILDERS: dict[str, Callable[[BuildOptions], BuildReport]] = {
    "cubicasa": build_cubicasa,
    "msd": build_msd,
    "swiss": build_swiss,
}


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

    msd = fetch_sources.add_parser("msd", help="MSD access graphs (CC BY 4.0), by range request")
    _add_data_dir(msd)
    msd.add_argument("--count", type=int, default=20)
    msd.add_argument("--all", action="store_true", help="every plan (about 40 MB)")
    msd.add_argument("--seed", type=int, default=0)
    msd.add_argument(
        "--plan", action="append", default=[], metavar="ID", help="e.g. 0 (repeatable)"
    )

    swiss = fetch_sources.add_parser(
        "swiss", help="Swiss Dwellings geometries.csv (CC BY 4.0), first sites or all"
    )
    _add_data_dir(swiss)
    swiss.add_argument("--apartments", type=int, default=50)
    swiss.add_argument("--all", action="store_true", help="the whole file (about 290 MB download)")

    build = commands.add_parser("build", help="convert fetched plans to JSONL")
    build.add_argument("source", choices=sorted(_BUILDERS))
    _add_data_dir(build)
    build.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR / "processed")
    build.add_argument("--limit", type=int)
    build.add_argument("--split", choices=["train", "val", "test"])
    build.add_argument("--seed", type=int, default=0)

    view = commands.add_parser("view", help="draw one plan's graph over its image (PNG)")
    view.add_argument("jsonl", type=Path)
    view.add_argument("--id", help="plan id; the first record if omitted")
    view.add_argument("--out", type=Path, default=Path("overlay.png"))

    sheet = commands.add_parser("sheet", help="HTML contact sheet of many plans")
    sheet.add_argument("jsonl", type=Path)
    sheet.add_argument("--limit", type=int, default=24)
    sheet.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR / "sheets" / "latest")
    return parser


def _fetch_cubicasa(args: argparse.Namespace) -> None:
    selection = CubiCasaSelection(
        plan_ids=tuple(args.plan), count=args.count, split=args.split, seed=args.seed
    )
    count = fetch_cubicasa(args.data_dir / "cubicasa5k", selection)
    print(f"CubiCasa5K: {count} plans in {args.data_dir / 'cubicasa5k'} (CC BY-NC 4.0)")


def _fetch_msd(args: argparse.Namespace) -> None:
    selection = MsdSelection(
        plan_ids=tuple(args.plan), count=None if args.all else args.count, seed=args.seed
    )
    count = fetch_msd(args.data_dir / "msd", selection)
    print(f"MSD: {count} plans in {args.data_dir / 'msd'} (CC BY 4.0)")


def _fetch_swiss(args: argparse.Namespace) -> None:
    path = fetch_swiss(
        args.data_dir / "swiss_dwellings", apartments=None if args.all else args.apartments
    )
    print(f"Swiss Dwellings: {path} (CC BY 4.0)")


_FETCHERS = {"cubicasa": _fetch_cubicasa, "msd": _fetch_msd, "swiss": _fetch_swiss}


def _run_fetch(args: argparse.Namespace) -> None:
    _FETCHERS[args.source](args)


def _run_build(args: argparse.Namespace) -> None:
    if args.split is not None and args.source != "cubicasa":
        raise SystemExit(f"{args.source} has no train/val/test split to filter by")
    options = BuildOptions(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        split=args.split,
        seed=args.seed,
    )
    print(_BUILDERS[args.source](options).summary())


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
