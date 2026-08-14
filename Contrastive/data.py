# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_reference_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "train_LiDAR+LoRA.py"
    if not script_path.exists():
        raise ImportError(f"Cannot locate reference dataset implementation: {script_path}")

    module_name = "_mapanything_train_lidar_lora_ref"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reference module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ref = _load_reference_module()

Seq1LidarDataset = _ref.Seq1LidarDataset
collate_fn = _ref.collate_fn


def __getattr__(name: str):
    return getattr(_ref, name)
