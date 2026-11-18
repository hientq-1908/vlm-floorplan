"""Launch the zero-shot baseline on ClearML and collect its results. See the module doc."""

import sys

from vlm_floorplan.evaluation.launcher import main

if __name__ == "__main__":
    sys.exit(main())
