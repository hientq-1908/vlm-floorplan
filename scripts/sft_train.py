"""Run SFT stage 1 locally. See `vlm_floorplan/training/train.py` and `docs/sft-training.md`."""

import sys

from vlm_floorplan.training.train import main

if __name__ == "__main__":
    sys.exit(main())
