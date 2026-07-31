#!/usr/bin/env python3
"""Root-level entry point for the LiDAR effect diagnosis script."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMPLEMENTATION = ROOT / "scripts" / "diagnose_lidar_effect.py"
if not IMPLEMENTATION.is_file():
    raise FileNotFoundError(
        "The full diagnosis implementation was not found at "
        f"{IMPLEMENTATION}. On a checkout where all scripts live in the "
        "repository root, copy scripts/diagnose_lidar_effect.py itself to "
        f"{ROOT / 'diagnose_lidar_effect.py'} instead of copying this small entry point."
    )

from scripts.diagnose_lidar_effect import main


if __name__ == "__main__":
    main()
