#!/usr/bin/env python3
"""Root-level entry point for depth-only LiDAR calibration."""

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMPLEMENTATION = ROOT / "scripts" / "calibrate_lidar_depth_extrinsics.py"
if not IMPLEMENTATION.is_file():
    raise FileNotFoundError(
        "The full calibration implementation was not found at "
        f"{IMPLEMENTATION}. On a checkout where all scripts live in the "
        "repository root, copy scripts/calibrate_lidar_depth_extrinsics.py "
        f"itself to {ROOT / 'calibrate_lidar_depth_extrinsics.py'} instead "
        "of copying this small entry point."
    )

spec = importlib.util.spec_from_file_location("calibrate_lidar_depth_extrinsics_impl", IMPLEMENTATION)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not create an import specification for {IMPLEMENTATION}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
main = module.main


if __name__ == "__main__":
    main()
