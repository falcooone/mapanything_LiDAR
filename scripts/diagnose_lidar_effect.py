#!/usr/bin/env python3
# coding: gbk
"""
Measure the real forward contribution of LiDAR for one checkpoint.

The important comparison is made with the same model weights and the same
RGB batch:

  1. RGB-only: ``use_lidar=False``
  2. RGB+LiDAR: ``use_lidar=True``

This is a functional ablation, not only a parameter-norm check.  The script
also records the tensors entering and leaving the LiDAR fusion modules, so it
can distinguish these cases:

  * LiDAR weights were not loaded or did not change.
  * LiDAR features are computed but the fusion contribution is nearly zero.
  * LiDAR changes the predictions, but the change is not beneficial.

Run this in the same environment used for evaluation because it imports the
project's existing test.py data/model pipeline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import importlib


def find_repo_root() -> Path:
    """Find the repository for both root-level and scripts/ execution."""
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        has_project_package = (candidate / "mapanything").is_dir()
        has_eval_script = (
            (candidate / "test.py").is_file()
            or (candidate / "scripts" / "test.py").is_file()
        )
        if has_project_package and has_eval_script:
            return candidate
    raise RuntimeError(
        "Could not locate repository root. Expected mapanything/ and "
        "test.py or scripts/test.py."
    )


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse exactly the model construction, 9-channel LiDAR preprocessing,
# intrinsics scaling, timestamp matching, and inference path used by the
# repository's evaluation script. Different server revisions place test.py
# either at the repository root or under scripts/. Select the first candidate
# that actually exposes the required API; a same-named but older test.py may
# exist and may only contain a different evaluation entry point.
TEST_API = ("build_model_for_eval", "load_eval_data", "run_inference_batch")
TEST_CANDIDATES = [ROOT / "test.py", ROOT / "scripts" / "test.py"]
PROJECT_TEST = None
TEST_SCRIPT = None
candidate_errors = []

for candidate in TEST_CANDIDATES:
    if not candidate.is_file():
        continue
    module_name = f"mapanything_project_eval_test_{len(candidate_errors)}"
    try:
        test_spec = importlib.util.spec_from_file_location(module_name, candidate)
        if test_spec is None or test_spec.loader is None:
            raise ImportError("could not create import specification")
        candidate_module = importlib.util.module_from_spec(test_spec)
        sys.modules[module_name] = candidate_module
        test_spec.loader.exec_module(candidate_module)
        missing_api = [name for name in TEST_API if not hasattr(candidate_module, name)]
        if missing_api:
            candidate_errors.append(f"{candidate}: missing {', '.join(missing_api)}")
            continue
        PROJECT_TEST = candidate_module
        TEST_SCRIPT = candidate
        break
    except Exception as exc:
        candidate_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

if PROJECT_TEST is None or TEST_SCRIPT is None:
    details = "\n".join(f"  - {item}" for item in candidate_errors)
    raise ImportError(
        "No project evaluation script exposes the required API "
        f"{TEST_API}. Candidates checked:\n{details}"
    )

print(f"[Eval API] loaded: {TEST_SCRIPT}")
build_model_for_eval = PROJECT_TEST.build_model_for_eval
load_eval_data = PROJECT_TEST.load_eval_data
run_inference_batch = PROJECT_TEST.run_inference_batch

# Older copies of test.py may not expose this private helper. Keep a local
# compatibility implementation so checkpoint inspection does not depend on
# that implementation detail.
_LEGACY_KEY_ALIASES = {
    "fusion_gate_mlp.0.weight": "fusion_module.gate_mlp.0.weight",
    "fusion_gate_mlp.0.bias": "fusion_module.gate_mlp.0.bias",
    "fusion_gate_mlp.2.weight": "fusion_module.gate_mlp.2.weight",
    "fusion_gate_mlp.2.bias": "fusion_module.gate_mlp.2.bias",
    "fusion_refine.weight": "fusion_module.refine.weight",
}


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]):
    helper = getattr(PROJECT_TEST, "_normalize_state_dict_keys", None)
    if helper is not None:
        return helper(state_dict)
    normalized = {}
    aliases = {}
    for key, value in state_dict.items():
        new_key = _LEGACY_KEY_ALIASES.get(key, key)
        normalized[new_key] = value
        if new_key != key:
            aliases[key] = new_key
    return normalized, aliases


OUTPUT_KEYS = (
    "pts3d",
    "pts3d_cam",
    "depth_along_ray",
    "depth_z",
    "ray_directions",
    "cam_trans",
    "cam_quats",
    "camera_poses",
    "metric_scaling_factor",
    "conf",
)

PARAM_GROUPS = OrderedDict(
    (
        ("lidars_encoder", ("lidars_encoder.",)),
        ("lidar_film", ("lidar_film.",)),
        ("fusion_module", ("fusion_module.",)),
        ("fusion_conv", ("fusion_conv.",)),
        ("pose_scale_proj", ("pose_scale_proj.",)),
    )
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_checkpoint_prefix(key: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_state_dict_file(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load raw/safetensors/full training checkpoints into one state dict."""
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        raw = load_file(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)

    meta: Dict[str, Any] = {
        "path": os.path.abspath(path),
        "container_keys": list(raw.keys())[:50] if isinstance(raw, dict) else [],
        "selected_state_key": None,
    }
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(raw)!r}")

    state = None
    for candidate in ("model_state_dict", "state_dict", "ema_shadow", "lora_state_dict"):
        if candidate in raw and isinstance(raw[candidate], dict):
            state = raw[candidate]
            meta["selected_state_key"] = candidate
            break
    if state is None and raw and all(torch.is_tensor(v) for v in raw.values()):
        state = raw
        meta["selected_state_key"] = "<raw_state_dict>"
    if state is None:
        raise KeyError(
            f"No tensor state dict found in {path}. Available keys: {list(raw)[:20]}"
        )

    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            normalized[strip_checkpoint_prefix(key)] = value.detach().cpu()
    normalized, aliases = _normalize_state_dict_keys(normalized)
    meta["num_tensors"] = len(normalized)
    meta["num_lora_tensors"] = sum("lora_" in key for key in normalized)
    meta["num_lidar_tensors"] = sum(
        key.startswith(tuple(prefix for prefixes in PARAM_GROUPS.values() for prefix in prefixes))
        for key in normalized
    )
    meta["legacy_aliases"] = aliases
    return normalized, meta


def tensor_group(key: str) -> Optional[str]:
    for group, prefixes in PARAM_GROUPS.items():
        if key.startswith(prefixes):
            return group
    return None


def tensor_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for value in tensors:
        x = value.detach().float()
        total += float((x * x).sum().item())
    return math.sqrt(total)


def parameter_diagnostics(
    model: torch.nn.Module,
    base_state: Optional[Dict[str, torch.Tensor]],
    checkpoint_state: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    current = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    rows: Dict[str, Any] = {}
    for group in PARAM_GROUPS:
        keys = [key for key in current if tensor_group(key) == group and current[key].is_floating_point()]
        values = [current[key] for key in keys]
        row: Dict[str, Any] = {
            "model_tensors": len(keys),
            "model_numel": int(sum(value.numel() for value in values)),
            "model_l2": tensor_norm(values),
            "model_mean_abs": float(
                sum(value.abs().sum().item() for value in values)
                / max(sum(value.numel() for value in values), 1)
            ),
            "trainable_tensors": sum(
                1
                for key, value in model.named_parameters()
                if tensor_group(key) == group and value.requires_grad
            ),
        }

        for label, reference in (("base", base_state), ("checkpoint", checkpoint_state)):
            if reference is None:
                continue
            matched = []
            delta_values = []
            missing = 0
            for key in keys:
                ref = reference.get(key)
                if ref is None or tuple(ref.shape) != tuple(current[key].shape):
                    missing += 1
                    continue
                matched.append(key)
                delta_values.append(current[key].float() - ref.float())
            delta_numel = sum(value.numel() for value in delta_values)
            row[f"{label}_matched_tensors"] = len(matched)
            row[f"{label}_missing_or_shape_mismatch"] = missing
            row[f"{label}_delta_l2"] = tensor_norm(delta_values)
            row[f"{label}_delta_mean_abs"] = float(
                sum(value.abs().sum().item() for value in delta_values)
                / max(delta_numel, 1)
            )
            row[f"{label}_changed_tensors"] = sum(
                1 for value in delta_values if value.numel() and value.abs().max().item() > 1e-8
            )
        rows[group] = row
    return rows


class RunningDiff:
    def __init__(self) -> None:
        self.numel = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0
        self.base_sum_abs = 0.0
        self.base_sum_sq = 0.0
        self.base_max_abs = 0.0

    def add(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if tuple(reference.shape) != tuple(candidate.shape):
            raise ValueError(
                f"Output shape mismatch: RGB={tuple(reference.shape)}, LiDAR={tuple(candidate.shape)}"
            )
        ref = reference.detach().float()
        diff = candidate.detach().float() - ref
        self.numel += int(diff.numel())
        self.sum_abs += float(diff.abs().sum().item())
        self.sum_sq += float((diff * diff).sum().item())
        self.max_abs = max(self.max_abs, float(diff.abs().max().item()) if diff.numel() else 0.0)
        self.base_sum_abs += float(ref.abs().sum().item())
        self.base_sum_sq += float((ref * ref).sum().item())
        self.base_max_abs = max(self.base_max_abs, float(ref.abs().max().item()) if ref.numel() else 0.0)

    def report(self) -> Dict[str, Any]:
        if self.numel == 0:
            return {"numel": 0}
        rms = math.sqrt(self.sum_sq / self.numel)
        base_rms = math.sqrt(self.base_sum_sq / self.numel)
        return {
            "numel": self.numel,
            "mean_abs_delta": self.sum_abs / self.numel,
            "rms_delta": rms,
            "max_abs_delta": self.max_abs,
            "relative_rms_delta": rms / max(base_rms, 1e-8),
            "relative_mean_abs_delta": self.sum_abs / max(self.base_sum_abs, 1e-8),
            "rgb_only_rms": base_rms,
            "rgb_only_max_abs": self.base_max_abs,
        }


class InputStats:
    def __init__(self) -> None:
        self.frames = 0
        self.valid_sum = 0.0
        self.valid_min = 1.0
        self.valid_max = 0.0
        self.scales: List[float] = []

    def add_views(self, views: Optional[List[Dict[str, Any]]]) -> None:
        if not views:
            return
        for view in views:
            pcd = view.get("pcd")
            if not torch.is_tensor(pcd) or pcd.ndim != 4 or pcd.shape[-1] < 9:
                continue
            valid = pcd[..., 7].detach().float().clamp(0.0, 1.0)
            coverage = float(valid.mean().item())
            self.frames += int(valid.shape[0])
            self.valid_sum += coverage
            self.valid_min = min(self.valid_min, coverage)
            self.valid_max = max(self.valid_max, coverage)
            scale = view.get("lidar_depth_scale")
            if torch.is_tensor(scale):
                self.scales.extend(float(x) for x in scale.detach().float().flatten().cpu())

    def report(self) -> Dict[str, Any]:
        if not self.frames:
            return {"frames": 0}
        scales = np.asarray(self.scales, dtype=np.float64)
        return {
            "frames": self.frames,
            "valid_coverage_mean": self.valid_sum / self.frames,
            "valid_coverage_min": self.valid_min,
            "valid_coverage_max": self.valid_max,
            "lidar_depth_scale_mean": float(scales.mean()) if scales.size else None,
            "lidar_depth_scale_std": float(scales.std()) if scales.size else None,
            "lidar_depth_scale_min": float(scales.min()) if scales.size else None,
            "lidar_depth_scale_max": float(scales.max()) if scales.size else None,
        }


def first_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if hasattr(value, "features") and torch.is_tensor(value.features):
        return value.features
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


class ForwardRecorder:
    """Collect scalar activation statistics without retaining feature maps."""

    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, float]]] = OrderedDict()
        self.handles: List[Any] = []

    @staticmethod
    def rms(value: torch.Tensor) -> float:
        return float(value.detach().float().pow(2).mean().sqrt().item())

    def add(self, name: str, row: Dict[str, float]) -> None:
        self.rows.setdefault(name, []).append(row)

    def fusion_hook(self, _module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if len(inputs) < 2:
            return
        rgb = first_tensor(inputs[0])
        lidar = first_tensor(inputs[1])
        fused = first_tensor(output)
        if rgb is None or lidar is None or fused is None:
            return
        delta = fused.float() - rgb.float()
        reliability = first_tensor(inputs[2]) if len(inputs) >= 3 else None
        row = {
            "rgb_rms": self.rms(rgb),
            "lidar_rms": self.rms(lidar),
            "fused_rms": self.rms(fused),
            "fusion_delta_rms": self.rms(delta),
            "relative_delta_rms": self.rms(delta) / max(self.rms(rgb), 1e-8),
        }
        if reliability is not None:
            rel = reliability.detach().float()
            row.update(
                {
                    "reliability_mean": float(rel.mean().item()),
                    "reliability_max": float(rel.max().item()),
                    "reliability_nonzero_fraction": float((rel > 1e-6).float().mean().item()),
                }
            )
        self.add("fusion_module", row)

    def unary_hook(self, name: str):
        def hook(_module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            before = first_tensor(inputs[0]) if inputs else None
            after = first_tensor(output)
            if after is None:
                return
            row = {"output_rms": self.rms(after)}
            if before is not None:
                row["input_rms"] = self.rms(before)
                row["delta_rms"] = self.rms(after.float() - before.float()) if before.shape == after.shape else 0.0
            self.add(name, row)
        return hook

    def attach(self, model: torch.nn.Module) -> None:
        module = getattr(model, "fusion_module", None)
        if module is not None:
            self.handles.append(module.register_forward_hook(self.fusion_hook))
        for name in ("lidar_film", "lidars_encoder", "fusion_conv", "pose_scale_proj"):
            module = getattr(model, name, None)
            if module is not None:
                self.handles.append(module.register_forward_hook(self.unary_hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def report(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, rows in self.rows.items():
            keys = sorted({key for row in rows for key in row})
            result[name] = {
                "calls": len(rows),
                "mean": {
                    key: float(np.mean([row[key] for row in rows if key in row]))
                    for key in keys
                    if any(key in row for row in rows)
                },
                "min": {
                    key: float(np.min([row[key] for row in rows if key in row]))
                    for key in keys
                    if any(key in row for row in rows)
                },
                "max": {
                    key: float(np.max([row[key] for row in rows if key in row]))
                    for key in keys
                    if any(key in row for row in rows)
                },
            }
        return result


def image_pcd_indices(batch_paths: List[str], pcd_timestamps: List[int]) -> List[int]:
    if not pcd_timestamps:
        raise ValueError("LiDAR mode requested, but no PCD timestamps were found")
    timestamps = np.asarray(pcd_timestamps)
    indices = []
    for image_path in batch_paths:
        match = re.search(r"color_(\d+)", image_path)
        image_ts = int(match.group(1)) if match else int(os.path.getmtime(image_path) * 1e9)
        indices.append(int(np.argmin(np.abs(timestamps - image_ts))))
    return indices


def add_prediction_diffs(
    rgb_predictions: List[Dict[str, Any]],
    lidar_predictions: List[Dict[str, Any]],
    accumulators: Dict[str, RunningDiff],
) -> None:
    if len(rgb_predictions) != len(lidar_predictions):
        raise ValueError("RGB-only and RGB+LiDAR returned different frame counts")
    for rgb_pred, lidar_pred in zip(rgb_predictions, lidar_predictions):
        for key in OUTPUT_KEYS:
            reference = rgb_pred.get(key)
            candidate = lidar_pred.get(key)
            if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
                continue
            accumulators.setdefault(key, RunningDiff()).add(reference, candidate)


def print_summary(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("LiDAR functional-effect diagnosis")
    print("=" * 78)
    print("[Input]", json.dumps(result["input"], ensure_ascii=False, indent=2))
    print("\n[Forward hooks]")
    hooks = result["forward_hooks"]
    for name, value in hooks.items():
        mean = value.get("mean", {})
        print(
            f"  {name:16s}: calls={value['calls']:4d}, "
            f"input/output RMS={mean.get('input_rms', float('nan')):.6g}/"
            f"{mean.get('output_rms', float('nan')):.6g}, "
            f"delta RMS={mean.get('fusion_delta_rms', mean.get('delta_rms', float('nan'))):.6g}"
        )
        if name == "fusion_module":
            print(
                f"    reliability mean/max/nonzero = "
                f"{mean.get('reliability_mean', float('nan')):.6g}/"
                f"{mean.get('reliability_max', float('nan')):.6g}/"
                f"{mean.get('reliability_nonzero_fraction', float('nan')):.6g}"
            )
            print(f"    relative fusion delta RMS = {mean.get('relative_delta_rms', float('nan')):.6g}")

    print("\n[RGB-only -> RGB+LiDAR output differences]")
    for key, value in result["output_differences"].items():
        print(
            f"  {key:22s}: mean_abs={value.get('mean_abs_delta', 0.0):.6g}, "
            f"relative_rms={value.get('relative_rms_delta', 0.0):.6g}, "
            f"max_abs={value.get('max_abs_delta', 0.0):.6g}"
        )

    print("\n[Interpretation]")
    fusion = hooks.get("fusion_module", {}).get("mean", {})
    output = result["output_differences"]
    fusion_relative = fusion.get("relative_delta_rms", 0.0)
    pts_relative = output.get("pts3d", {}).get("relative_rms_delta", 0.0)
    pose_relative = output.get("camera_poses", {}).get("relative_rms_delta", 0.0)
    if not hooks.get("fusion_module"):
        print("  ERROR: fusion_module hook was never called. The LiDAR branch was not executed.")
    elif fusion_relative < 1e-4:
        print("  WARNING: LiDAR reached fusion_module, but its effective feature contribution is near zero.")
    elif pts_relative < 1e-4 and pose_relative < 1e-4:
        print("  WARNING: fusion changes features, but the downstream predictions are almost unchanged.")
    else:
        print("  LiDAR has a measurable forward contribution. Compare this with validation metrics to judge whether it helps.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="/home/xuyh/mapanything/")
    parser.add_argument("--checkpoint", default="/add02/users/xuyh/checkpoints/32_lora_lidar/checkpoints/epoch_002.pt")
    parser.add_argument("--seq_root", default="/add02/users/xuyh/seq2/",help="One sequence containing rgb/ and lidar/")
    parser.add_argument("--output_json", default="lidar_effect_diagnosis.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--max_images", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    checkpoint_state, checkpoint_meta = load_state_dict_file(args.checkpoint)
    base_path = os.path.join(args.model_dir, "model.safetensors")
    base_state, base_meta = load_state_dict_file(base_path) if os.path.exists(base_path) else ({}, {})

    print("[Checkpoint]", json.dumps(checkpoint_meta, ensure_ascii=False, indent=2))
    if checkpoint_meta.get("num_lidar_tensors", 0) == 0:
        print("[Checkpoint] WARNING: checkpoint contains no LiDAR/Fusion/Pose-scale tensors.")
    elif checkpoint_meta.get("selected_state_key") == "lora_state_dict":
        print("[Checkpoint] NOTE: this is LoRA-only; LiDAR weights come from model.safetensors, not this checkpoint.")

    print("[Model] building exactly through scripts/test.py ...")
    model = build_model_for_eval(
        args.model_dir,
        device,
        trained_ckpt_path=args.checkpoint,
        use_lora=bool(args.use_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model.eval()
    parameter_stats = parameter_diagnostics(model, base_state, checkpoint_state)

    (
        image_paths,
        _intrinsics,
        _gt_poses,
        _gt_depths,
        pcd_file_list,
        pcd_timestamps,
        intrinsics_raw,
        intrinsics_size,
    ) = load_eval_data(
        args.seq_root,
        img_size=args.img_size,
        use_lidar=True,
        max_images=args.max_images,
    )
    if not pcd_file_list:
        raise RuntimeError("No LiDAR PCD files found. Check --seq_root/lidar.")

    recorder = ForwardRecorder()
    recorder.attach(model)
    output_accumulators: Dict[str, RunningDiff] = {}
    input_stats = InputStats()
    failed_batches = 0
    pcd_cache: Dict[int, Dict[str, Any]] = {}
    t0 = time.time()

    for batch_start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[batch_start : batch_start + args.batch_size]
        batch_number = batch_start // args.batch_size + 1
        total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
        indices = image_pcd_indices(batch_paths, pcd_timestamps)

        # Resetting the RNG makes optional geometric-input dropout identical
        # in both forwards. Evaluation mode removes dropout from the network.
        set_seed(args.seed + batch_number)
        rgb_predictions, _ = run_inference_batch(
            model,
            batch_paths,
            device,
            use_lidar=False,
            pcd_file_list=[],
            pcd_timestamps=[],
            intrinsics_raw=intrinsics_raw,
            intrinsics_size=intrinsics_size,
            img_size=args.img_size,
            view_pcd_indices=[0] * len(batch_paths),
            pcd_cache=pcd_cache,
        )

        set_seed(args.seed + batch_number)
        lidar_predictions, lidar_views = run_inference_batch(
            model,
            batch_paths,
            device,
            use_lidar=True,
            pcd_file_list=pcd_file_list,
            pcd_timestamps=pcd_timestamps,
            intrinsics_raw=intrinsics_raw,
            intrinsics_size=intrinsics_size,
            img_size=args.img_size,
            view_pcd_indices=indices,
            pcd_cache=pcd_cache,
        )

        if rgb_predictions is None or lidar_predictions is None:
            failed_batches += 1
            print(f"[Batch {batch_number}/{total_batches}] failed")
            continue
        add_prediction_diffs(rgb_predictions, lidar_predictions, output_accumulators)
        input_stats.add_views(lidar_views)
        print(f"[Batch {batch_number}/{total_batches}] compared {len(batch_paths)} frames")
        del rgb_predictions, lidar_predictions, lidar_views
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    recorder.close()
    result = {
        "checkpoint": checkpoint_meta,
        "base_model": base_meta,
        "args": vars(args),
        "input": input_stats.report(),
        "parameter_diagnostics": parameter_stats,
        "forward_hooks": recorder.report(),
        "output_differences": {key: value.report() for key, value in output_accumulators.items()},
        "failed_batches": failed_batches,
        "elapsed_seconds": time.time() - t0,
    }
    output_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print_summary(result)
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    main()
