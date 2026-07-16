#!/usr/bin/env python3
"""
Inspect LiDAR/fusion parameter norms in a MapAnything checkpoint.

This script is intentionally lightweight: it does not instantiate the model.
It only reads the checkpoint state dict and reports whether LiDAR/fusion
parameters are present, non-zero, and optionally changed relative to a base
checkpoint or base model.safetensors file.

Examples:
  python scripts/check_lidar_norms.py --checkpoint /path/to/epoch_009.pt

  python scripts/check_lidar_norms.py \
    --checkpoint /path/to/epoch_009.pt \
    --base /path/to/base/model.safetensors \
    --top-k 8

  python scripts/check_lidar_norms.py \
    --checkpoint /path/to/epoch_009.pt \
    --state-key ema_shadow
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

torch = None


def require_torch():
    global torch
    if torch is None:
        try:
            import torch as torch_module
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "This script needs torch to inspect checkpoint tensors. "
                "Run it inside the same environment used for training/evaluation."
            ) from exc
        torch = torch_module
    return torch


def load_safetensors(path: str):
    require_torch()
    try:
        from safetensors.torch import load_file
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError("safetensors is not installed, cannot read .safetensors") from exc
    return load_file(path)


LIDAR_PREFIXES = (
    "lidars_encoder.",
    "lidar_film.",
    "fusion_module.",
    "fusion_conv.",
)

LEGACY_KEY_ALIASES = {
    "fusion_gate_mlp.0.weight": "fusion_module.gate_mlp.0.weight",
    "fusion_gate_mlp.0.bias": "fusion_module.gate_mlp.0.bias",
    "fusion_gate_mlp.2.weight": "fusion_module.gate_mlp.2.weight",
    "fusion_gate_mlp.2.bias": "fusion_module.gate_mlp.2.bias",
    "fusion_refine.weight": "fusion_module.refine.weight",
}


def load_checkpoint_object(path: str):
    torch_module = require_torch()
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        return load_safetensors(path)
    return torch_module.load(path, map_location="cpu", weights_only=False)


def unwrap_state_dict(obj, state_key: str = "auto") -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")

    meta = {
        "checkpoint_keys": list(obj.keys())[:50],
        "epoch": obj.get("epoch"),
        "best_loss": obj.get("best_loss"),
        "selected_state_key": None,
    }

    if state_key != "auto":
        if state_key not in obj:
            raise KeyError(f"Requested state key {state_key!r} not found in checkpoint")
        state = obj[state_key]
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint key {state_key!r} is not a state dict")
        meta["selected_state_key"] = state_key
        return state, meta

    for candidate in ("model_state_dict", "ema_shadow", "state_dict", "lora_state_dict"):
        if candidate in obj and isinstance(obj[candidate], dict):
            meta["selected_state_key"] = candidate
            return obj[candidate], meta

    if all(torch.is_tensor(v) for v in obj.values()):
        meta["selected_state_key"] = "<raw_state_dict>"
        return obj, meta

    raise KeyError(
        "Could not find a tensor state dict. Try --state-key model_state_dict or --state-key ema_shadow."
    )


def normalize_key(key: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return LEGACY_KEY_ALIASES.get(key, key)


def normalize_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[normalize_key(key)] = value.detach().cpu()
    return out


def module_group(key: str) -> str:
    if key.startswith("lidars_encoder."):
        return "lidars_encoder"
    if key.startswith("lidar_film."):
        return "lidar_film"
    if key.startswith("fusion_module."):
        return "fusion_module"
    if key.startswith("fusion_conv."):
        return "fusion_conv"
    return "other"


def is_lidar_key(key: str) -> bool:
    return key.startswith(LIDAR_PREFIXES)


BUFFER_SUFFIXES = (
    ".num_batches_tracked",
    ".running_mean",
    ".running_var",
)


def is_buffer_like_key(key: str) -> bool:
    return key.endswith(BUFFER_SUFFIXES)


def keep_key_for_report(key: str, value: torch.Tensor, include_buffers: bool) -> bool:
    if not is_lidar_key(key):
        return False
    if include_buffers:
        return True
    if is_buffer_like_key(key):
        return False
    return value.is_floating_point()


def tensor_l2_sum(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        x = tensor.float()
        total += float(torch.sum(x * x).item())
    return math.sqrt(total)


def group_stats(
    state: Dict[str, torch.Tensor],
    base: Optional[Dict[str, torch.Tensor]] = None,
    include_buffers: bool = False,
) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
    for key, value in state.items():
        if keep_key_for_report(key, value, include_buffers):
            grouped[module_group(key)].append((key, value))

    rows: Dict[str, Dict[str, object]] = {}
    for group in ("lidars_encoder", "lidar_film", "fusion_module", "fusion_conv"):
        items = grouped.get(group, [])
        numel = sum(int(v.numel()) for _, v in items)
        nonzero = sum(int(torch.count_nonzero(v).item()) for _, v in items)
        l2_norm = tensor_l2_sum(v for _, v in items)
        abs_sum = sum(float(v.float().abs().sum().item()) for _, v in items)
        abs_max = max((float(v.float().abs().max().item()) for _, v in items if v.numel()), default=0.0)

        row: Dict[str, object] = {
            "tensors": len(items),
            "numel": numel,
            "l2_norm": l2_norm,
            "mean_abs": abs_sum / max(numel, 1),
            "max_abs": abs_max,
            "zero_frac": 1.0 - (nonzero / max(numel, 1)),
        }

        if base is not None:
            matched = []
            missing_in_base = 0
            delta_abs_sum = 0.0
            delta_sq_sum = 0.0
            delta_abs_max = 0.0
            changed_tensors = 0
            delta_numel = 0
            for key, value in items:
                base_value = base.get(key)
                if base_value is None or tuple(base_value.shape) != tuple(value.shape):
                    missing_in_base += 1
                    continue
                delta = value.float() - base_value.float()
                matched.append(key)
                delta_numel += int(delta.numel())
                delta_abs_sum += float(delta.abs().sum().item())
                delta_sq_sum += float(torch.sum(delta * delta).item())
                current_max = float(delta.abs().max().item()) if delta.numel() else 0.0
                delta_abs_max = max(delta_abs_max, current_max)
                if current_max > 1e-8:
                    changed_tensors += 1
            row.update(
                {
                    "base_matched_tensors": len(matched),
                    "base_missing_tensors": missing_in_base,
                    "delta_l2_norm": math.sqrt(delta_sq_sum),
                    "delta_mean_abs": delta_abs_sum / max(delta_numel, 1),
                    "delta_max_abs": delta_abs_max,
                    "changed_tensors": changed_tensors,
                }
            )

        rows[group] = row
    return rows


def top_tensor_rows(
    state: Dict[str, torch.Tensor],
    base: Optional[Dict[str, torch.Tensor]],
    top_k: int,
    include_buffers: bool = False,
) -> List[Dict[str, object]]:
    rows = []
    for key, value in state.items():
        if not keep_key_for_report(key, value, include_buffers):
            continue
        x = value.float()
        row = {
            "key": key,
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "l2_norm": float(x.norm().item()),
            "mean_abs": float(x.abs().mean().item()) if x.numel() else 0.0,
            "max_abs": float(x.abs().max().item()) if x.numel() else 0.0,
            "zero_frac": float((x == 0).float().mean().item()) if x.numel() else 0.0,
        }
        if base is not None:
            base_value = base.get(key)
            if base_value is not None and tuple(base_value.shape) == tuple(value.shape):
                delta = x - base_value.float()
                row.update(
                    {
                        "delta_l2_norm": float(delta.norm().item()),
                        "delta_mean_abs": float(delta.abs().mean().item()) if delta.numel() else 0.0,
                        "delta_max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
                    }
                )
            else:
                row.update(
                    {
                        "delta_l2_norm": None,
                        "delta_mean_abs": None,
                        "delta_max_abs": None,
                    }
                )
        rows.append(row)

    sort_key = "delta_l2_norm" if base is not None else "l2_norm"
    rows.sort(key=lambda row: row[sort_key] if row[sort_key] is not None else -1.0, reverse=True)
    return rows[:top_k]


def scalar_diagnostics(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    out: Dict[str, object] = {}

    gate_bias = state.get("fusion_module.gate_mlp.2.bias")
    if gate_bias is not None:
        x = gate_bias.float()
        out["fusion_gate_bias_mean"] = float(x.mean().item())
        out["fusion_gate_sigmoid_mean"] = float(torch.sigmoid(x).mean().item())

    gate_weight = state.get("fusion_module.gate_mlp.2.weight")
    if gate_weight is not None:
        x = gate_weight.float()
        out["fusion_gate_last_weight_l2"] = float(x.norm().item())
        out["fusion_gate_last_weight_mean_abs"] = float(x.abs().mean().item())

    refine_weight = state.get("fusion_module.refine.weight")
    if refine_weight is not None:
        x = refine_weight.float()
        out["fusion_refine_weight_l2"] = float(x.norm().item())
        out["fusion_refine_weight_mean_abs"] = float(x.abs().mean().item())

    conv_weight = state.get("fusion_conv.weight")
    if conv_weight is not None:
        x = conv_weight.float()
        out["fusion_conv_weight_l2"] = float(x.norm().item())
        out["fusion_conv_weight_mean_abs"] = float(x.abs().mean().item())

    film_last_weight = state.get("lidar_film.mlp.2.weight")
    film_last_bias = state.get("lidar_film.mlp.2.bias")
    if film_last_weight is not None:
        out["lidar_film_last_weight_l2"] = float(film_last_weight.float().norm().item())
    if film_last_bias is not None:
        out["lidar_film_last_bias_l2"] = float(film_last_bias.float().norm().item())

    return out


def print_report(report: Dict[str, object], top_rows: List[Dict[str, object]]) -> None:
    print("=" * 88)
    print("LiDAR / Fusion checkpoint norm report")
    print("=" * 88)
    print(f"checkpoint: {report['checkpoint']}")
    print(f"state_key : {report['state_key']}")
    if report.get("epoch") is not None:
        print(f"epoch     : {report['epoch']}")
    if report.get("best_loss") is not None:
        print(f"best_loss : {report['best_loss']}")
    if report.get("base"):
        print(f"base      : {report['base']}")
    print()

    print("Module summary")
    header = (
        f"{'module':16s} {'tensors':>7s} {'numel':>12s} {'l2_norm':>13s} "
        f"{'mean_abs':>11s} {'max_abs':>11s} {'zero%':>8s}"
    )
    if report.get("base"):
        header += f" {'delta_l2':>13s} {'delta_mean':>11s} {'delta_max':>11s} {'changed':>9s}"
    print(header)
    for group, row in report["groups"].items():
        line = (
            f"{group:16s} {row['tensors']:7d} {row['numel']:12d} "
            f"{row['l2_norm']:13.6e} {row['mean_abs']:11.4e} "
            f"{row['max_abs']:11.4e} {100.0 * row['zero_frac']:7.2f}%"
        )
        if report.get("base"):
            line += (
                f" {row['delta_l2_norm']:13.6e} {row['delta_mean_abs']:11.4e} "
                f"{row['delta_max_abs']:11.4e} {row['changed_tensors']:4d}/"
                f"{row['base_matched_tensors']:<4d}"
            )
        print(line)

    print()
    print("Key diagnostics")
    diagnostics = report["diagnostics"]
    if diagnostics:
        for key in sorted(diagnostics):
            print(f"  {key:34s}: {diagnostics[key]:.6e}")
    else:
        print("  No known scalar diagnostics found.")

    print()
    sort_label = "delta_l2_norm" if report.get("base") else "l2_norm"
    print(f"Top tensors by {sort_label}")
    for row in top_rows:
        line = (
            f"  {row['key']} shape={row['shape']} "
            f"l2={row['l2_norm']:.6e} mean_abs={row['mean_abs']:.4e} "
            f"max_abs={row['max_abs']:.4e} zero={100.0 * row['zero_frac']:.2f}%"
        )
        if report.get("base"):
            line += (
                f" delta_l2={row['delta_l2_norm'] if row['delta_l2_norm'] is not None else 'NA'}"
                f" delta_max={row['delta_max_abs'] if row['delta_max_abs'] is not None else 'NA'}"
            )
        print(line)

    print()
    print("Interpretation hints")
    print("  - Non-zero LiDAR/fusion norms only prove parameters exist in the checkpoint.")
    print("  - A non-zero delta versus --base is stronger evidence that training changed them.")
    print("  - fusion_conv/refine/gate last-layer norms near zero mean the learned fusion path may still be inactive.")
    print("  - To prove LiDAR is used functionally, also compare inference with use_lidar=1 vs use_lidar=0.")


def build_report(args: argparse.Namespace) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    ckpt_obj = load_checkpoint_object(args.checkpoint)
    state, meta = unwrap_state_dict(ckpt_obj, args.state_key)
    state = normalize_state_dict(state)

    base_state = None
    if args.base:
        base_obj = load_checkpoint_object(args.base)
        base_state, _ = unwrap_state_dict(base_obj, args.base_state_key)
        base_state = normalize_state_dict(base_state)

    lidar_keys = [
        key
        for key, value in state.items()
        if keep_key_for_report(key, value, args.include_buffers)
    ]
    if not lidar_keys:
        print("WARNING: no LiDAR/fusion keys found in the selected state dict.")
        print("Expected prefixes:", ", ".join(LIDAR_PREFIXES))

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "base": os.path.abspath(args.base) if args.base else None,
        "state_key": meta["selected_state_key"],
        "epoch": meta.get("epoch"),
        "best_loss": meta.get("best_loss"),
        "total_state_tensors": len(state),
        "lidar_state_tensors": len(lidar_keys),
        "include_buffers": args.include_buffers,
        "groups": group_stats(state, base_state, include_buffers=args.include_buffers),
        "diagnostics": scalar_diagnostics(state),
    }
    return report, top_tensor_rows(
        state,
        base_state,
        args.top_k,
        include_buffers=args.include_buffers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print LiDAR/fusion module norms from a MapAnything checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt/.pth/.safetensors checkpoint")
    parser.add_argument(
        "--base",
        default="",
        help="Optional base checkpoint or model.safetensors for delta comparison",
    )
    parser.add_argument(
        "--state-key",
        default="auto",
        help="Checkpoint state key to inspect: auto, model_state_dict, ema_shadow, state_dict, etc.",
    )
    parser.add_argument(
        "--base-state-key",
        default="auto",
        help="State key for --base. Defaults to auto.",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Number of largest tensors to print")
    parser.add_argument("--json", default="", help="Optional path to save the full report as JSON")
    parser.add_argument(
        "--include-buffers",
        action="store_true",
        help="Include BatchNorm running stats and num_batches_tracked buffers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, top_rows = build_report(args)
    print_report(report, top_rows)

    if args.json:
        payload = dict(report)
        payload["top_tensors"] = top_rows
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON report saved to: {args.json}")


if __name__ == "__main__":
    main()
