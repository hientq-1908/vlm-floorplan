"""Pack the source and a training file for the free-tier notebook. See `training/free_tier.py`."""

import sys

from vlm_floorplan.training.free_tier import main

if __name__ == "__main__":
    sys.exit(main())
