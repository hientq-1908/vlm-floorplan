"""Launch SFT on ClearML and collect its results. See `vlm_floorplan/training/launcher.py`."""

import sys

from vlm_floorplan.training.launcher import main

if __name__ == "__main__":
    sys.exit(main())
