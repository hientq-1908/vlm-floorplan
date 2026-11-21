"""Build a few example tool-use episodes and write them for review.

    uv run python scripts/build_tool_episodes.py --records data/processed/train.jsonl \\
        --out-dir data/tool_episodes --per-kind 2
    uv run python scripts/build_tool_episodes.py          # no data: uses the built-in sample plan

Writes two files into `--out-dir`:

- `episodes.jsonl`: one TRL conversational record per line (`messages` + `tools`).
- `episodes.txt`: the same episodes as text, rendered through the Qwen3.5 chat template.
  The text view needs `transformers` (`uv run --group ml ...`); without it only the JSONL
  is written.

This is the prototype's data path, not the session generator: each episode is one repair
task on one real plan, solved by the scripted inverse edit.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_floorplan.agent.episode import sample_episodes  # noqa: E402
from vlm_floorplan.agent.render import MODEL_ID, load_tokenizer, render_episode  # noqa: E402
from vlm_floorplan.agent.sample_plan import sample_record  # noqa: E402
from vlm_floorplan.data.jsonl import read_records  # noqa: E402
from vlm_floorplan.record import PlanRecord  # noqa: E402

PLANS_TO_READ = 200
"""Only the first plans of a JSONL file are read: examples do not need the whole split."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", type=Path, help="processed JSONL; default: the sample plan")
    parser.add_argument("--out-dir", type=Path, default=Path("data/tool_episodes"))
    parser.add_argument("--per-kind", type=int, default=2, help="episodes per task kind")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=MODEL_ID, help="tokenizer used for the text view")
    return parser.parse_args()


def _load_records(path: Path | None) -> list[PlanRecord]:
    if path is None or not path.exists():
        print("no records file: using the built-in sample plan", file=sys.stderr)
        return [sample_record()]
    return list(islice(read_records(path), PLANS_TO_READ))


def main() -> None:
    args = _parse_args()
    episodes = sample_episodes(_load_records(args.records), args.per_kind, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_dir / "episodes.jsonl"
    jsonl.write_text("".join(json.dumps(ep, ensure_ascii=False) + "\n" for ep in episodes))
    print(f"wrote {len(episodes)} episodes to {jsonl}")

    try:
        tokenizer = load_tokenizer(args.model)
    except ImportError:
        print("transformers is not installed: skipped the text view (use --group ml)")
        return
    rendered = [
        f"##### {ep['id']}  (reward {ep['reward']:.2f})\n{render_episode(ep, tokenizer)}"
        for ep in episodes
    ]
    text = args.out_dir / "episodes.txt"
    text.write_text("\n\n".join(rendered) + "\n")
    print(f"wrote the rendered view to {text}")


if __name__ == "__main__":
    main()
