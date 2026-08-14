#!/usr/bin/env python3
"""
Compare a warmup checkpoint against the *training-script-initialized* model.

This script mirrors the initialization logic in scripts/train_LiDAR+LoRA.py:
  - load config.json + model.safetensors
  - skip LiDAR encoder weights from the pretrained state dict
  - zero fusion_conv
  - zero fusion_module gate/refine
  - apply warmup gate bias when lidar_warmup_epochs > 0

It is meant for warmup checkpoints only, where the interesting changes should be
in LiDAR / fusion / selected heads, not in the base RGB encoder weights.

Example:
  python scripts/compare_warmup_checkpoint_norms.py ^
    --model-dir E:\\path\\to\\mapanything_model_dir ^
    --checkpoint E:\\path\\to\\epoch_003.pt ^
    --lidar-warmup-epochs 4
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from mapanything.models import MapAnything
from safetensors.torch import load_file

LIDAR_NUM_CHANNELS = 9


def _gate_bias_from_alpha(alpha: float) -> float:
    alpha = float(max(1e-4, min(1.0 - 1e-4, alpha)))
    return math.log(alpha / (1.0 - alpha))


def normalize_key(key: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_checkpoint_state(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")

    meta = {
        "path": os.path.abspath(path),
        "epoch": obj.get("epoch"),
        "best_loss": obj.get("best_loss"),
        "selected_state_key": None,
        "checkpoint_keys": list(obj.keys())[:50],
    }

    for candidate in ("model_state_dict", "state_dict", "ema_shadow", "model"):
        if candidate in obj and isinstance(obj[candidate], dict):
            meta["selected_state_key"] = candidate
            state = obj[candidate]
            break
    else:
        if all(torch.is_tensor(v) for v in obj.values()):
            meta["selected_state_key"] = "<raw_state_dict>"
            state = obj
        else:
            raise KeyError(f"No tensor state dict found in {path}")

    normalized = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            normalized[normalize_key(key)] = value.detach().cpu()
    return normalized, meta


def build_training_initialized_model(
    model_dir: str,
    lidar_warmup_epochs: int,
    lidar_warmup_gate_alpha: float,
) -> Tuple[MapAnything, Dict[str, torch.Tensor]]:
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    encoder_config = config.get("encoder_config", {}).copy()
    encoder_config.pop("pretrained", None)
    encoder_config.pop("weights", None)
    encoder_config["uses_torch_hub"] = False

    geometric_input_config = copy.deepcopy(config.get("geometric_input_config", {}))
    lidar_encoder_config = geometric_input_config.get("lidars_encoder_config", {}).copy()
    for key in ("pretrained", "weights", "pretrained_checkpoint_path", "checkpoint_path", "custom_ckpt_path", "load_pretrained_weights"):
        lidar_encoder_config.pop(key, None)
    lidar_encoder_config["pretrained"] = False
    lidar_encoder_config["weights"] = None
    lidar_encoder_config["uses_torch_hub"] = False
    lidar_encoder_config["in_chans"] = LIDAR_NUM_CHANNELS
    lidar_encoder_config["input_size"] = 512
    geometric_input_config["lidars_encoder_config"] = lidar_encoder_config

    model = MapAnything(
        name=config.get("name", "mapanything"),
        encoder_config=encoder_config,
        info_sharing_config=config.get("info_sharing_config", {}),
        pred_head_config=config.get("pred_head_config", {}),
        geometric_input_config=geometric_input_config,
        pretrained_checkpoint_path=None,
        torch_hub_force_reload=False,
        info_sharing_mlp_layer_str="swiglufused",
    )

    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
        lidar_encoder_keys = [
            key for key in state_dict.keys()
            if key.replace("_orig_mod.", "").startswith("lidars_encoder.")
        ]
        for key in lidar_encoder_keys:
            state_dict.pop(key, None)
        model.load_state_dict(state_dict, strict=False)

    if lidar_warmup_epochs > 0:
        fusion_conv = getattr(model, "fusion_conv", None)
        if fusion_conv is not None and hasattr(fusion_conv, "weight"):
            with torch.no_grad():
                fusion_conv.weight.zero_()
                if fusion_conv.bias is not None:
                    fusion_conv.bias.zero_()

        fusion_module = getattr(model, "fusion_module", None)
        if fusion_module is not None:
            with torch.no_grad():
                gate_mlp = getattr(fusion_module, "gate_mlp", None)
                if gate_mlp is not None and len(gate_mlp) >= 3 and hasattr(gate_mlp[-1], "weight"):
                    gate_mlp[-1].weight.zero_()
                    gate_mlp[-1].bias.zero_()
                    gate_mlp[-1].bias.fill_(_gate_bias_from_alpha(lidar_warmup_gate_alpha))
                refine = getattr(fusion_module, "refine", None)
                if refine is not None and hasattr(refine, "weight"):
                    refine.weight.zero_()

    model_state = {k.replace("module.", "", 1): v.detach().cpu() for k, v in model.state_dict().items()}
    return model, model_state


def is_matrix_like(key: str, value: torch.Tensor) -> bool:
    return key.endswith(".weight") and torch.is_tensor(value) and value.is_floating_point() and value.ndim >= 2


def group_name(key: str) -> str:
    for prefix in (
        "encoder.",
        "info_sharing.",
        "lidars_encoder.",
        "lidar_film.",
        "fusion_module.",
        "fusion_conv.",
        "pose_head.",
        "dense_head.",
        "scale_head.",
        "pose_scale_proj.",
        "shared_linear.",
        "shared_decoder.",
        "output_proj.",
        "ray_dirs_encoder.",
        "depth_encoder.",
        "depth_scale_encoder.",
        "cam_rot_encoder.",
        "cam_trans_encoder.",
        "cam_trans_scale_encoder.",
    ):
        if key.startswith(prefix):
            return prefix[:-1]
    return "other"


def tensor_l2(values: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for value in values:
        x = value.detach().float().cpu()
        total += float(torch.sum(x * x).item())
    return math.sqrt(total)


def summarize_groups(base_state: Dict[str, torch.Tensor], ckpt_state: Dict[str, torch.Tensor]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for key, value in ckpt_state.items():
        if is_matrix_like(key, value):
            grouped[group_name(key)].append(key)

    rows = []
    for group in sorted(grouped):
        keys = grouped[group]
        matched = 0
        missing = 0
        base_sq = 0.0
        ckpt_sq = 0.0
        delta_sq = 0.0
        max_abs_delta = 0.0

        for key in keys:
            ckpt_val = ckpt_state[key].detach().float().cpu()
            base_val = base_state.get(key)
            if base_val is None or tuple(base_val.shape) != tuple(ckpt_val.shape):
                missing += 1
                continue
            base_val = base_val.detach().float().cpu()
            delta = ckpt_val - base_val
            matched += 1
            base_sq += float(torch.sum(base_val * base_val).item())
            ckpt_sq += float(torch.sum(ckpt_val * ckpt_val).item())
            delta_sq += float(torch.sum(delta * delta).item())
            if delta.numel():
                max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))

        if matched == 0:
            continue

        base_norm = math.sqrt(base_sq)
        ckpt_norm = math.sqrt(ckpt_sq)
        delta_norm = math.sqrt(delta_sq)
        rows.append(
            {
                "group": group,
                "weight_tensors": len(keys),
                "matched_tensors": matched,
                "missing_tensors": missing,
                "base_norm": base_norm,
                "ckpt_norm": ckpt_norm,
                "delta_norm": delta_norm,
                "delta_over_base": delta_norm / (base_norm + 1e-12),
                "norm_change_ratio": (ckpt_norm - base_norm) / (base_norm + 1e-12),
                "max_abs_delta": max_abs_delta,
            }
        )

    rows.sort(key=lambda row: row["delta_norm"], reverse=True)
    return rows


def summarize_tensors(base_state: Dict[str, torch.Tensor], ckpt_state: Dict[str, torch.Tensor], top_k: int) -> List[Dict[str, object]]:
    rows = []
    for key, value in ckpt_state.items():
        if not is_matrix_like(key, value):
            continue
        base_val = base_state.get(key)
        if base_val is None or tuple(base_val.shape) != tuple(value.shape):
            continue
        ckpt_val = value.detach().float().cpu()
        base_val = base_val.detach().float().cpu()
        delta = ckpt_val - base_val
        rows.append(
            {
                "key": key,
                "shape": list(value.shape),
                "group": group_name(key),
                "base_norm": float(base_val.norm().item()),
                "ckpt_norm": float(ckpt_val.norm().item()),
                "delta_norm": float(delta.norm().item()),
                "delta_over_base": float(delta.norm().item()) / (float(base_val.norm().item()) + 1e-12),
                "norm_change_ratio": (float(ckpt_val.norm().item()) - float(base_val.norm().item())) / (float(base_val.norm().item()) + 1e-12),
                "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
            }
        )

    rows.sort(key=lambda row: row["delta_norm"], reverse=True)
    return rows[:top_k]


def print_report(checkpoint_meta, base_path, base_state, ckpt_state, group_rows, top_rows):
    print("=" * 96)
    print("Warmup checkpoint vs training-initialized model")
    print("=" * 96)
    print(f"checkpoint : {checkpoint_meta['path']}")
    print(f"state key  : {checkpoint_meta['selected_state_key']}")
    if checkpoint_meta.get("epoch") is not None:
        print(f"epoch      : {checkpoint_meta['epoch']}")
    if checkpoint_meta.get("best_loss") is not None:
        print(f"best_loss  : {checkpoint_meta['best_loss']}")
    print(f"base model : {base_path}")
    print()

    header = (
        f"{'group':18s} {'tensors':>7s} {'matched':>7s} {'missing':>7s} "
        f"{'base_norm':>13s} {'ckpt_norm':>13s} {'delta_norm':>13s} "
        f"{'delta/base':>11s} {'ckpt/base-1':>13s} {'max_abs':>11s}"
    )
    print("Module summary")
    print(header)
    for row in group_rows:
        print(
            f"{row['group']:18s} {row['weight_tensors']:7d} {row['matched_tensors']:7d} {row['missing_tensors']:7d} "
            f"{row['base_norm']:13.6e} {row['ckpt_norm']:13.6e} {row['delta_norm']:13.6e} "
            f"{row['delta_over_base']:11.4e} {row['norm_change_ratio']:13.4e} {row['max_abs_delta']:11.4e}"
        )

    print()
    print("Top changed tensors")
    for row in top_rows:
        print(
            f"  {row['key']} shape={row['shape']} group={row['group']} "
            f"base={row['base_norm']:.6e} ckpt={row['ckpt_norm']:.6e} "
            f"delta={row['delta_norm']:.6e} delta/base={row['delta_over_base']:.4e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare warmup checkpoint weights against training-script initial state.")
    parser.add_argument("--model-dir", required=True, help="Directory containing config.json and model.safetensors")
    parser.add_argument("--checkpoint", required=True, help="Warmup checkpoint path (.pt/.pth)")
    parser.add_argument("--top-k", type=int, default=24, help="Number of changed tensors to print")
    parser.add_argument("--lidar-warmup-epochs", type=int, default=0, help="Must match the training run's warmup epoch setting")
    parser.add_argument("--lidar-warmup-gate-alpha", type=float, default=0.60, help="Must match the training run's warmup gate alpha")
    parser.add_argument("--json", default="", help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, base_state = build_training_initialized_model(
        model_dir=args.model_dir,
        lidar_warmup_epochs=args.lidar_warmup_epochs,
        lidar_warmup_gate_alpha=args.lidar_warmup_gate_alpha,
    )
    ckpt_state, ckpt_meta = load_checkpoint_state(args.checkpoint)

    group_rows = summarize_groups(base_state, ckpt_state)
    top_rows = summarize_tensors(base_state, ckpt_state, args.top_k)

    print_report(
        checkpoint_meta=ckpt_meta,
        base_path=os.path.join(os.path.abspath(args.model_dir), "model.safetensors"),
        base_state=base_state,
        ckpt_state=ckpt_state,
        group_rows=group_rows,
        top_rows=top_rows,
    )

    if args.json:
        payload = {
            "checkpoint": ckpt_meta,
            "base_model": os.path.join(os.path.abspath(args.model_dir), "model.safetensors"),
            "groups": group_rows,
            "top_tensors": top_rows,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved to: {args.json}")


if __name__ == "__main__":
    main()
