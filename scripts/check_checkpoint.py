#!/usr/bin/env python3
"""
Checkpoint inspection utility for MapAnything LiDAR + LoRA runs.

What it checks:
  - Whether the checkpoint can be loaded into the current model definition.
  - Which parameter keys are present / missing / unexpected.
  - Per-module parameter deltas against the base model weights.
  - LoRA-specific norms and fusion-module gate/refine diagnostics.

Typical usage:
  python scripts/check_checkpoint.py ^
    --model_dir E:\\path\\to\\base_model ^
    --checkpoint E:\\path\\to\\checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from mapanything.models import MapAnything

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:  # pragma: no cover - optional dependency fallback
    load_safetensors = None


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, lora_alpha: float = 32.0):
        super().__init__()
        self.linear = linear
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear.out_features))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        lora = x.float() @ self.lora_A.float()
        lora = lora @ self.lora_B.float()
        lora = lora.to(out.dtype)
        return out.add_(lora, alpha=self.scaling)


def inject_lora_to_module(module: nn.Module, r: int = 8, lora_alpha: float = 32.0) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(child, r, lora_alpha))
        else:
            inject_lora_to_module(child, r, lora_alpha)


def load_any_state_dict(path: str) -> Dict[str, torch.Tensor]:
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        if load_safetensors is None:
            raise RuntimeError("safetensors is not available in this environment")
        return load_safetensors(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "lora_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    return ckpt


def unwrap_state_dict(ckpt_obj) -> Tuple[Dict[str, torch.Tensor], Dict]:
    meta = {}
    if isinstance(ckpt_obj, dict):
        meta = {
            "epoch": ckpt_obj.get("epoch"),
            "best_loss": ckpt_obj.get("best_loss"),
            "keys": list(ckpt_obj.keys()),
        }
        for key in ("model_state_dict", "lora_state_dict", "state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key], meta
    if not isinstance(ckpt_obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(ckpt_obj)}")
    return ckpt_obj, meta


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        out[key.replace("module.", "", 1)] = value
    return out


LEGACY_KEY_ALIASES = {
    "fusion_gate_mlp.0.weight": "fusion_module.gate_mlp.0.weight",
    "fusion_gate_mlp.0.bias": "fusion_module.gate_mlp.0.bias",
    "fusion_gate_mlp.2.weight": "fusion_module.gate_mlp.2.weight",
    "fusion_gate_mlp.2.bias": "fusion_module.gate_mlp.2.bias",
    "fusion_refine.weight": "fusion_module.refine.weight",
}


def normalize_legacy_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    normalized = {}
    alias_hits = {}
    for key, value in state_dict.items():
        new_key = LEGACY_KEY_ALIASES.get(key, key)
        normalized[new_key] = value
        if new_key != key:
            alias_hits[key] = new_key
    return normalized, alias_hits


def infer_use_lora(ckpt_state: Dict[str, torch.Tensor]) -> bool:
    return any("lora_A" in k or "lora_B" in k for k in ckpt_state.keys())


def build_model(model_dir: str, device: str, use_lora: bool, lora_r: int, lora_alpha: float):
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    encoder_config = config.get("encoder_config", {}).copy()
    encoder_config.pop("pretrained", None)
    encoder_config.pop("weights", None)
    encoder_config["uses_torch_hub"] = False

    model = MapAnything(
        name=config.get("name", "mapanything"),
        encoder_config=encoder_config,
        info_sharing_config=config.get("info_sharing_config", {}),
        pred_head_config=config.get("pred_head_config", {}),
        geometric_input_config=config.get("geometric_input_config", {}),
        pretrained_checkpoint_path=None,
        torch_hub_force_reload=False,
        info_sharing_mlp_layer_str="swiglufused",
    )

    if os.path.exists(weights_path):
        base_state = load_any_state_dict(weights_path)
        model.load_state_dict(base_state, strict=False)
    else:
        base_state = {}

    if use_lora:
        lora_targets = []
        if hasattr(model, "encoder") and model.encoder is not None:
            lora_targets.append(model.encoder)
        if hasattr(model, "info_sharing") and model.info_sharing is not None:
            lora_targets.append(model.info_sharing)
        if hasattr(model, "dense_head") and model.dense_head is not None:
            lora_targets.append(model.dense_head)
        if hasattr(model, "pose_head") and model.pose_head is not None:
            lora_targets.append(model.pose_head)
        if hasattr(model, "scale_head") and model.scale_head is not None:
            lora_targets.append(model.scale_head)
        for target in lora_targets:
            inject_lora_to_module(target, r=lora_r, lora_alpha=lora_alpha)

    model = model.to(device)
    model.eval()
    return model, base_state


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "norm": x.norm().item(),
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item() if x.numel() > 1 else 0.0,
        "abs_mean": x.abs().mean().item(),
        "abs_max": x.abs().max().item() if x.numel() > 0 else 0.0,
        "zero_frac": (x == 0).float().mean().item() if x.numel() > 0 else 0.0,
        "numel": float(x.numel()),
    }


def top_level_group(key: str) -> str:
    return key.split(".", 1)[0]


def group_summary(
    base_state: Dict[str, torch.Tensor],
    ckpt_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    delta_threshold: float,
) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for key, ckpt_val in ckpt_state.items():
        if key in model_state:
            grouped[top_level_group(key)].append(key)

    rows = []
    for group, keys in sorted(grouped.items()):
        total_numel = 0
        total_abs_delta = 0.0
        max_abs_delta = 0.0
        changed_keys = 0
        group_base_norm = 0.0
        group_ckpt_norm = 0.0
        samples = []

        for key in keys:
            ckpt_val = ckpt_state[key].detach().float().cpu()
            base_val = base_state.get(key)
            if base_val is None:
                continue
            base_val = base_val.detach().float().cpu()
            delta = ckpt_val - base_val
            abs_delta = delta.abs()
            key_max = abs_delta.max().item() if abs_delta.numel() else 0.0
            key_abs_mean = abs_delta.mean().item() if abs_delta.numel() else 0.0

            total_numel += int(delta.numel())
            total_abs_delta += abs_delta.sum().item()
            max_abs_delta = max(max_abs_delta, key_max)
            if key_max > delta_threshold:
                changed_keys += 1
            group_base_norm += base_val.norm().item()
            group_ckpt_norm += ckpt_val.norm().item()
            samples.append((key, key_abs_mean, key_max))

        if not keys:
            continue

        samples.sort(key=lambda x: x[2], reverse=True)
        rows.append(
            {
                "group": group,
                "keys": len(keys),
                "changed_keys": changed_keys,
                "change_ratio": changed_keys / max(len(keys), 1),
                "mean_abs_delta": total_abs_delta / max(total_numel, 1),
                "max_abs_delta": max_abs_delta,
                "base_norm_sum": group_base_norm,
                "ckpt_norm_sum": group_ckpt_norm,
                "top_keys": samples[:3],
            }
        )
    return rows


def print_load_report(
    model,
    base_state,
    ckpt_raw_state,
    ckpt_state,
    alias_hits,
    delta_threshold: float,
):
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())
    raw_ckpt_keys = set(ckpt_raw_state.keys())
    common_keys = model_keys & ckpt_keys
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    print("\n[Checkpoint]")
    print(f"  model keys     : {len(model_keys)}")
    print(f"  checkpoint keys: {len(raw_ckpt_keys)} raw / {len(ckpt_keys)} normalized")
    print(f"  common keys    : {len(common_keys)}")
    print(f"  missing keys   : {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    if alias_hits:
        print("  legacy aliases detected:")
        for old_key, new_key in alias_hits.items():
            print(f"    - {old_key} -> {new_key}")

    if missing:
        print("  missing sample :")
        for key in missing[:20]:
            print(f"    - {key}")
    if unexpected:
        print("  unexpected sample:")
        for key in unexpected[:20]:
            print(f"    - {key}")

    if base_state:
        print("\n[Module Delta Summary]")
        rows = group_summary(base_state, ckpt_state, model_state, delta_threshold)
        for row in rows:
            print(
                f"  {row['group']:<16s} "
                f"keys={row['keys']:4d} "
                f"changed={row['changed_keys']:4d} "
                f"delta_mean={row['mean_abs_delta']:.6e} "
                f"delta_max={row['max_abs_delta']:.6e} "
                f"change_ratio={row['change_ratio']:.2%}"
            )
            for key, mean_abs, max_abs in row["top_keys"]:
                print(f"    - {key}: mean|d|={mean_abs:.6e}, max|d|={max_abs:.6e}")

    print("\n[Key Sanity]")
    for key in [
        "fusion_module.gate_mlp.2.bias",
        "fusion_module.gate_mlp.2.weight",
        "fusion_module.refine.weight",
    ]:
        if key in ckpt_state and key in model_state:
            ckpt_val = ckpt_state[key].detach().float().cpu()
            base_val = base_state.get(key)
            base_norm = base_val.detach().float().cpu().norm().item() if base_val is not None else float("nan")
            delta_norm = (ckpt_val - base_val.detach().float().cpu()).norm().item() if base_val is not None else float("nan")
            print(
                f"  {key}: "
                f"ckpt_norm={ckpt_val.norm().item():.6e}, "
                f"base_norm={base_norm:.6e}, "
                f"delta_norm={delta_norm:.6e}, "
                f"zero_frac={tensor_stats(ckpt_val)['zero_frac']:.2%}"
            )
        else:
            print(f"  {key}: not present")


def print_fusion_report(model: nn.Module):
    fusion_module = getattr(model, "fusion_module", None)
    if fusion_module is None:
        print("\n[Fusion Module] not found")
        return

    gate_mlp = getattr(fusion_module, "gate_mlp", None)
    refine = getattr(fusion_module, "refine", None)

    print("\n[Fusion Module]")
    if gate_mlp is not None and len(gate_mlp) >= 3:
        last = gate_mlp[-1]
        bias = last.bias.detach().float().cpu()
        weight = last.weight.detach().float().cpu()
        gate_prob = torch.sigmoid(bias.mean()).item()
        print(
            f"  gate_bias_mean={bias.mean().item():.6f}, "
            f"gate_bias_std={bias.std(unbiased=False).item() if bias.numel() > 1 else 0.0:.6f}, "
            f"gate_weight_norm={weight.norm().item():.6e}, "
            f"gate_weight_abs_mean={weight.abs().mean().item():.6e}, "
            f"gate_sigmoid(mean_bias)={gate_prob:.6f}"
        )
    else:
        print("  gate_mlp not found or too short")

    if refine is not None and hasattr(refine, "weight"):
        refine_w = refine.weight.detach().float().cpu()
        print(
            f"  refine_weight_norm={refine_w.norm().item():.6e}, "
            f"refine_weight_abs_mean={refine_w.abs().mean().item():.6e}, "
            f"refine_zero_frac={tensor_stats(refine_w)['zero_frac']:.2%}"
        )
    else:
        print("  refine not found")


def load_checkpoint_for_model(model: nn.Module, checkpoint_path: str):
    if Path(checkpoint_path).suffix.lower() == ".safetensors":
        ckpt_obj = load_any_state_dict(checkpoint_path)
    else:
        ckpt_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, meta = unwrap_state_dict(ckpt_obj)
    state_dict = strip_module_prefix(state_dict)
    state_dict, alias_hits = normalize_legacy_keys(state_dict)
    model_state = model.state_dict()

    filtered = {}
    shape_mismatch = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            filtered[key] = value
        elif key in model_state:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return state_dict, meta, filtered, missing, unexpected, shape_mismatch, alias_hits


def main():
    parser = argparse.ArgumentParser(description="Inspect MapAnything checkpoints")
    parser.add_argument("--model_dir", type=str, required=True, help="Base model directory containing config.json and model.safetensors")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to inspect")
    parser.add_argument("--device", type=str, default="cpu", help="Device used to build the model")
    parser.add_argument("--use_lora", type=int, default=None, help="Force LoRA injection on/off. Default: auto-detect from checkpoint")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--delta_threshold", type=float, default=1e-6, help="Per-key threshold used to mark a key as changed")
    args = parser.parse_args()

    ckpt_obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_state_raw, meta = unwrap_state_dict(ckpt_obj)
    ckpt_state_raw = strip_module_prefix(ckpt_state_raw)
    ckpt_state, alias_hits = normalize_legacy_keys(ckpt_state_raw)
    use_lora = infer_use_lora(ckpt_state) if args.use_lora is None else bool(args.use_lora)

    print(f"[Input] model_dir   = {args.model_dir}")
    print(f"[Input] checkpoint  = {args.checkpoint}")
    print(f"[Input] use_lora    = {use_lora}")
    if meta:
        print(f"[Input] meta        = epoch={meta.get('epoch')}, best_loss={meta.get('best_loss')}, root_keys={meta.get('keys')}")

    model, base_state = build_model(args.model_dir, args.device, use_lora, args.lora_r, args.lora_alpha)
    _, _, _, missing, unexpected, shape_mismatch, load_alias_hits = load_checkpoint_for_model(model, args.checkpoint)

    print_load_report(model, base_state, ckpt_state_raw, ckpt_state, alias_hits or load_alias_hits, args.delta_threshold)

    if shape_mismatch:
        print("\n[Shape Mismatch]")
        for key, ckpt_shape, model_shape in shape_mismatch[:30]:
            print(f"  - {key}: ckpt={ckpt_shape} model={model_shape}")

    print("\n[Load Result]")
    print(f"  missing keys after load   : {len(missing)}")
    print(f"  unexpected keys after load: {len(unexpected)}")
    if missing:
        for key in list(missing)[:20]:
            print(f"    - {key}")
    if unexpected:
        for key in list(unexpected)[:20]:
            print(f"    - {key}")

    print_fusion_report(model)

    # Global LoRA sanity summary.
    lora_a = [v for k, v in model.state_dict().items() if "lora_A" in k]
    lora_b = [v for k, v in model.state_dict().items() if "lora_B" in k]
    if lora_a:
        a_norm = sum(v.detach().float().cpu().norm().item() for v in lora_a) / len(lora_a)
        b_norm = sum(v.detach().float().cpu().norm().item() for v in lora_b) / len(lora_b) if lora_b else 0.0
        print("\n[LoRA]")
        print(f"  LoRA_A mean norm = {a_norm:.6e}")
        print(f"  LoRA_B mean norm = {b_norm:.6e}")
        print("  Note: LoRA_B should usually move away from 0 if training actually updated it.")


if __name__ == "__main__":
    main()
