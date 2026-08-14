#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pretrained DINO versus projected-LiDAR ResNet contrastive training.

Dataset logic:
    The Seq1LidarDataset, timestamp matching, RGB quality filtering, calibrated
    projection and cache generation are loaded directly from
    scripts/train_LiDAR+LoRA.py.

Model logic:
    RGB uses the pretrained MapAnything DINO encoder.
    Projected LiDAR uses an independent 9-channel 2D ResNet.
    The two spatial feature maps are projected into a shared contrastive
    space and optimized with pixel-level symmetric InfoNCE.
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys

    _repo_root_for_import = __import__("pathlib").Path(__file__).resolve().parents[1]
    if str(_repo_root_for_import) not in sys.path:
        sys.path.insert(0, str(_repo_root_for_import))

import argparse
import copy
import gc
import importlib.util
import inspect
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

from mapanything.models import MapAnything
from uniception.models.encoders import ViTEncoderInput, ViTEncoderNonImageInput


LIDAR_NUM_CHANNELS = 9


# ---------------------------------------------------------------------------
# Dataset: reuse scripts/train_LiDAR+LoRA.py
# ---------------------------------------------------------------------------

def load_reference_dataset_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "train.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot locate reference script: {script_path}")

    module_name = "_contrastive_train_lidar_lora_reference"
    loaded = __import__("sys").modules.get(module_name)
    if loaded is not None:
        return loaded

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_REFERENCE = load_reference_dataset_module()
ReferenceSeq1LidarDataset = _REFERENCE.Seq1LidarDataset


class ContrastiveSeq1LidarDataset(ReferenceSeq1LidarDataset):
    """Reference dataset with a short-window fix for contrastive batches.

    The reference script defaults to seq_len=4 and stride=3. Some current
    sequences have only two valid RGB/LiDAR matches after filtering. The
    reference __getitem__ can pad short windows, but __len__ returns zero
    before it can be called. For contrastive training, retain the available
    views and shorten only the effective window. No views are duplicated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requested_seq_len = int(getattr(self, "seq_len", 0))
        available_views = len(getattr(self, "all_views_meta", []))
        if available_views > 0 and available_views < requested_seq_len:
            self.requested_seq_len = requested_seq_len
            self.seq_len = available_views
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[ContrastiveDataset] short sequence: "
                    f"available_views={available_views}, "
                    f"requested_seq_len={requested_seq_len}, "
                    f"effective_seq_len={self.seq_len}"
                )


def build_reference_dataset_kwargs(
    dataset_cls,
    raw_kwargs: dict,
    rank: int = 0,
) -> dict:
    """Support both current and older remote copies of the reference script."""

    try:
        parameters = inspect.signature(dataset_cls.__init__).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return dict(raw_kwargs)

    valid = set(parameters)
    filtered = {
        key: value for key, value in raw_kwargs.items() if key in valid
    }
    dropped = sorted(set(raw_kwargs) - set(filtered))
    if dropped and rank == 0:
        print(
            "[Dataset][WARN] Remote reference dataset does not support "
            f"{dropped}; those options are ignored."
        )
    return filtered


def make_dataset(
    *,
    seq_root: str,
    seq_len: int,
    stride: int,
    img_size: int,
    cache_dir: Optional[str],
    tolerance: float,
    use_lidar: bool,
    max_samples: Optional[int],
    max_rgb_brightness: Optional[float],
    max_rgb_contrast: Optional[float],
    lidar_extrinsics_path: str,
):
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    raw_kwargs = {
        "seq_len": seq_len,
        "stride": stride,
        "img_size": img_size,
        "cache_dir": cache_dir,
        "tolerance": tolerance,
        "use_lidar": use_lidar,
        "max_samples": max_samples,
        "max_rgb_brightness": max_rgb_brightness,
        "max_rgb_contrast": max_rgb_contrast,
        "lidar_extrinsics_path": lidar_extrinsics_path,
    }
    kwargs = build_reference_dataset_kwargs(
        ReferenceSeq1LidarDataset,
        raw_kwargs,
    )
    return ContrastiveSeq1LidarDataset(seq_root=seq_root, **kwargs)


def flatten_collate(batch):
    views = []
    for sequence in batch:
        views.extend(sequence)
    return views


# ---------------------------------------------------------------------------
# Project encoders and pixel-level contrastive model
# ---------------------------------------------------------------------------

def load_safetensors(path: Path):
    try:
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint {path}: {exc}") from exc
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def compatible_state_dict(state_dict, model):
    model_state = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key in model_state
        and hasattr(value, "shape")
        and value.shape == model_state[key].shape
    }


def extract_features(output):
    features = output.features if hasattr(output, "features") else output
    if not torch.is_tensor(features):
        raise TypeError(
            f"Encoder output does not contain a tensor feature map: {type(features)}"
        )
    return features


def to_feature_map(features: torch.Tensor) -> torch.Tensor:
    """Convert common DINO/ResNet outputs to [B,C,H,W] without pooling."""

    if features.ndim == 4:
        return features

    if features.ndim != 3:
        raise ValueError(
            f"Expected a spatial feature tensor [B,C,H,W] or token tensor [B,N,C], "
            f"got {tuple(features.shape)}"
        )

    batch, dim_a, dim_b = features.shape
    # [B,N,C] token layout. Remove a CLS token when N-1 is a square.
    if dim_b >= dim_a:
        token_count = dim_a
        spatial_count = token_count - 1
        side = int(round(spatial_count ** 0.5))
        if side * side == spatial_count:
            tokens = features[:, 1:, :]
        else:
            side = int(round(token_count ** 0.5))
            if side * side != token_count:
                raise ValueError(
                    f"Cannot reshape token feature tensor {tuple(features.shape)} "
                    "to a spatial map"
                )
            tokens = features
        return tokens.transpose(1, 2).contiguous().view(
            batch, dim_b, side, side
        )

    # [B,C,N] token layout.
    side = int(round(dim_b ** 0.5))
    if side * side != dim_b:
        raise ValueError(
            f"Cannot reshape feature tensor {tuple(features.shape)} "
            "to a spatial map"
        )
    return features.view(batch, dim_a, side, side)


def build_project_encoders(model_dir: str | Path, device: torch.device):
    """Construct the exact encoders used by train_LiDAR+LoRA.py.

    The RGB encoder is loaded from the MapAnything checkpoint. The LiDAR
    encoder is the project's configured encoder, including its own stride and
    dilation settings. Only its input channel count and input size are
    overridden to match the current 9-channel projection, exactly as in the
    reference training script.
    """

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing MapAnything config: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    encoder_config = copy.deepcopy(config.get("encoder_config", {}))
    encoder_config.pop("pretrained", None)
    encoder_config.pop("weights", None)
    encoder_config["uses_torch_hub"] = False

    geometric_input_config = copy.deepcopy(
        config.get("geometric_input_config", {})
    )
    lidar_encoder_config = geometric_input_config.get(
        "lidars_encoder_config",
        {},
    )
    for key in (
        "pretrained",
        "weights",
        "pretrained_checkpoint_path",
        "checkpoint_path",
        "custom_ckpt_path",
        "load_pretrained_weights",
    ):
        lidar_encoder_config.pop(key, None)
    lidar_encoder_config["pretrained"] = False
    lidar_encoder_config["weights"] = None
    lidar_encoder_config["uses_torch_hub"] = False
    lidar_encoder_config["in_chans"] = LIDAR_NUM_CHANNELS
    lidar_encoder_config["input_size"] = 512
    geometric_input_config["lidars_encoder_config"] = lidar_encoder_config

    container = MapAnything(
        name=config.get("name", "mapanything"),
        encoder_config=encoder_config,
        info_sharing_config=config.get("info_sharing_config", {}),
        pred_head_config=config.get("pred_head_config", {}),
        geometric_input_config=geometric_input_config,
        pretrained_checkpoint_path=None,
        torch_hub_force_reload=False,
        info_sharing_mlp_layer_str="swiglufused",
    )

    if weights_path.exists():
        state_dict = load_safetensors(weights_path)
        # train_LiDAR+LoRA.py explicitly skips pretrained LiDAR weights and
        # trains that encoder from scratch after changing the input layout.
        lidar_keys = [
            key
            for key in state_dict
            if key.replace("_orig_mod.", "").startswith("lidars_encoder.")
        ]
        for key in lidar_keys:
            state_dict.pop(key, None)
        compatible = compatible_state_dict(state_dict, container)
        container.load_state_dict(compatible, strict=False)
        print(
            f"[Encoders] loaded RGB-compatible tensors={len(compatible)} "
            f"from={weights_path}; skipped_lidar_tensors={len(lidar_keys)}"
        )
    else:
        warnings.warn(
            f"Checkpoint not found at {weights_path}; RGB DINO is not pretrained.",
            RuntimeWarning,
        )

    rgb_encoder = container.encoder
    lidar_encoder = container.lidars_encoder
    rgb_dim = int(getattr(rgb_encoder, "enc_embed_dim", 0))
    lidar_dim = int(getattr(lidar_encoder, "enc_embed_dim", rgb_dim))
    if rgb_dim <= 0 or lidar_dim <= 0:
        raise RuntimeError(
            f"Invalid encoder dimensions: rgb_dim={rgb_dim}, lidar_dim={lidar_dim}"
        )

    for parameter in rgb_encoder.parameters():
        parameter.requires_grad = False
    for parameter in lidar_encoder.parameters():
        parameter.requires_grad = True

    rgb_encoder.to(device)
    lidar_encoder.to(device)
    del container
    gc.collect()
    print(
        f"[Encoders] RGB={type(rgb_encoder).__name__} dim={rgb_dim}; "
        f"LiDAR={type(lidar_encoder).__name__} dim={lidar_dim}; "
        "using project-defined stride/dilation"
    )
    return rgb_encoder, lidar_encoder, rgb_dim, lidar_dim


class PixelProjectionHead(nn.Module):
    """1x1 projection head that preserves H x W pixel correspondence."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor):
        return F.normalize(self.net(feature_map), dim=1)


class CrossModalPixelModel(nn.Module):
    def __init__(
        self,
        rgb_encoder: nn.Module,
        lidar_encoder: nn.Module,
        rgb_dim: int,
        lidar_dim: int,
        *,
        proj_dim: int,
        proj_hidden_dim: int,
        lidar_input_size: int,
        projector_dropout: float,
        train_dino: bool,
    ):
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.lidar_encoder = lidar_encoder
        self.train_dino = bool(train_dino)
        self.lidar_input_size = int(lidar_input_size)
        self.rgb_projector = PixelProjectionHead(
            rgb_dim,
            proj_hidden_dim,
            proj_dim,
            projector_dropout,
        )
        self.lidar_projector = PixelProjectionHead(
            lidar_dim,
            proj_hidden_dim,
            proj_dim,
            projector_dropout,
        )

        for parameter in self.rgb_encoder.parameters():
            parameter.requires_grad = self.train_dino
        for parameter in self.lidar_encoder.parameters():
            parameter.requires_grad = True

    @staticmethod
    def stack_view_field(views, key: str):
        values = []
        for view in views:
            value = view[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            values.append(value)
        return torch.cat(values, dim=0)

    @staticmethod
    def data_norm_type(views):
        value = views[0].get("data_norm_type", "dinov2")
        if isinstance(value, (list, tuple)):
            return str(value[0])
        return str(value)

    def forward(self, views):
        if not views:
            raise ValueError("The batch contains no views")

        images = self.stack_view_field(views, "img")
        lidar_nhwc = self.stack_view_field(views, "pcd")
        if lidar_nhwc.ndim != 4:
            raise ValueError(
                f"Expected pcd tensors shaped [B,H,W,C], got {tuple(lidar_nhwc.shape)}"
            )
        if lidar_nhwc.shape[-1] != LIDAR_NUM_CHANNELS:
            raise ValueError(
                f"Projected LiDAR must have {LIDAR_NUM_CHANNELS} channels, "
                f"got {lidar_nhwc.shape[-1]}"
            )

        # Channel 7 is the valid projected-pixel mask in train.py's 9-channel
        # layout. Keep it separate and resize it with nearest-neighbour.
        valid_mask = lidar_nhwc[..., 7:8].permute(0, 3, 1, 2).contiguous()
        lidar = lidar_nhwc.permute(0, 3, 1, 2).contiguous()
        if tuple(lidar.shape[-2:]) != (
            self.lidar_input_size,
            self.lidar_input_size,
        ):
            lidar = F.interpolate(
                lidar,
                size=(self.lidar_input_size, self.lidar_input_size),
                mode="bilinear",
                align_corners=False,
            )
            valid_mask = F.interpolate(
                valid_mask,
                size=(self.lidar_input_size, self.lidar_input_size),
                mode="nearest",
            )

        rgb_input = ViTEncoderInput(
            image=images,
            data_norm_type=self.data_norm_type(views),
        )
        rgb_output = self.rgb_encoder(rgb_input)
        lidar_output = self.lidar_encoder(
            ViTEncoderNonImageInput(data=lidar)
        )
        rgb_map = to_feature_map(extract_features(rgb_output))
        lidar_map = to_feature_map(extract_features(lidar_output))

        if lidar_map.shape[-2:] != rgb_map.shape[-2:]:
            lidar_map = F.interpolate(
                lidar_map,
                size=rgb_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        valid_mask = F.interpolate(
            valid_mask,
            size=rgb_map.shape[-2:],
            mode="nearest",
        ) > 0.5

        return (
            self.rgb_projector(rgb_map),
            self.lidar_projector(lidar_map),
            valid_mask,
        )


def build_cross_modal_model(args, device):
    rgb_encoder, lidar_encoder, rgb_dim, lidar_dim = build_project_encoders(
        args.model_dir,
        device,
    )
    return CrossModalPixelModel(
        rgb_encoder=rgb_encoder,
        lidar_encoder=lidar_encoder,
        rgb_dim=rgb_dim,
        lidar_dim=lidar_dim,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        lidar_input_size=args.lidar_input_size,
        projector_dropout=args.projector_dropout,
        train_dino=args.train_dino,
    ).to(device)


class PixelSymmetricInfoNCE(nn.Module):
    """Pixel-level symmetric InfoNCE with diagonal positive matches."""

    def __init__(
        self,
        temperature: float = 0.07,
        rgb_to_lidar_weight: float = 0.3,
        lidar_to_rgb_weight: float = 0.7,
        num_matches: int = 4096,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.rgb_to_lidar_weight = float(rgb_to_lidar_weight)
        self.lidar_to_rgb_weight = float(lidar_to_rgb_weight)
        self.num_matches = int(num_matches)

    def forward(
        self,
        rgb_map: torch.Tensor,
        lidar_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        if rgb_map.shape != lidar_map.shape:
            raise ValueError(
                f"RGB/LiDAR map shape mismatch: {tuple(rgb_map.shape)} vs "
                f"{tuple(lidar_map.shape)}"
            )
        if valid_mask.shape[0] != rgb_map.shape[0]:
            raise ValueError("valid_mask batch dimension does not match features")

        rgb = F.normalize(rgb_map, dim=1)
        lidar = F.normalize(lidar_map, dim=1)
        rgb = rgb.permute(0, 2, 3, 1).reshape(-1, rgb.shape[1])
        lidar = lidar.permute(0, 2, 3, 1).reshape(-1, lidar.shape[1])
        mask = valid_mask[:, 0].reshape(-1).bool()
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if self.num_matches > 0 and indices.numel() > self.num_matches:
            indices = indices[
                torch.randperm(indices.numel(), device=indices.device)[
                    : self.num_matches
                ]
            ]

        if indices.numel() < 2:
            zero = (rgb_map.sum() + lidar_map.sum()) * 0.0
            return zero, int(indices.numel())

        rgb = rgb[indices]
        lidar = lidar[indices]
        logits = (rgb @ lidar.t()) / self.temperature
        target = torch.arange(logits.shape[0], device=logits.device)
        rgb_to_lidar = F.cross_entropy(logits, target)
        lidar_to_rgb = F.cross_entropy(logits.t(), target)
        total_weight = (
            self.rgb_to_lidar_weight + self.lidar_to_rgb_weight
        )
        if total_weight <= 0:
            loss = 0.5 * (rgb_to_lidar + lidar_to_rgb)
        else:
            loss = (
                self.rgb_to_lidar_weight * rgb_to_lidar
                + self.lidar_to_rgb_weight * lidar_to_rgb
            ) / total_weight
        return loss, int(indices.numel())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_root(raw_root: str) -> str:
    path = Path(raw_root)
    if path.exists():
        return str(path)
    if not path.is_absolute():
        candidates = [
            Path.cwd() / path,
            Path(__file__).resolve().parent.parent / path,
        ]
    else:
        candidates = [
            Path.cwd() / path.name,
            Path(__file__).resolve().parent.parent / path.name,
        ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return raw_root


def move_views_to_device(views, device: torch.device):
    for view in views:
        for key, value in list(view.items()):
            if torch.is_tensor(value):
                view[key] = value.to(device, non_blocking=True)
    return views


def make_loader(dataset, args, device):
    workers = int(args.num_workers)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=flatten_collate,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return GradScaler(enabled=enabled)


def make_autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    args,
    epoch: int,
):
    model.train()
    if not args.train_dino:
        model.rgb_encoder.eval()

    total_loss = 0.0
    steps = 0
    skipped_small = 0
    started = time.time()

    for step, batch in enumerate(loader):
        views = move_views_to_device(batch, device)
        if len(views) < 2:
            skipped_small += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        with make_autocast(device, args.amp):
            rgb_embed, lidar_embed, valid_mask = model(views)
            loss, pixel_count = criterion(rgb_embed, lidar_embed, valid_mask)

        if step == 0:
            print(
                f"[PixelDiag] rgb_map={tuple(rgb_embed.shape)} "
                f"lidar_map={tuple(lidar_embed.shape)} "
                f"valid_mask={tuple(valid_mask.shape)} "
                f"valid_pixels={pixel_count}"
            )

        if not torch.isfinite(loss):
            print(f"[WARN] epoch={epoch} step={step}: non-finite loss; skipped")
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            optimizer.step()

        value = float(loss.detach().item())
        total_loss += value
        steps += 1
        if step % args.log_every == 0:
            print(
                f"[Epoch {epoch:03d}] step={step:05d}/{len(loader)} "
                f"views={len(views)} pixels={pixel_count} loss={value:.6f}"
            )

        if device.type == "cuda" and args.empty_cache_each_step:
            torch.cuda.empty_cache()
        del views, rgb_embed, lidar_embed, valid_mask, loss

    average = total_loss / max(steps, 1)
    print(
        f"[Epoch {epoch:03d}] avg_loss={average:.6f} "
        f"steps={steps} skipped_small_batches={skipped_small} "
        f"time={time.time() - started:.1f}s"
    )
    return average


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    best_loss: float,
    args,
):
    payload = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def load_checkpoint(path: Path, model, optimizer, scaler):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Contrastive learning between pretrained DINO and "
            "projected-LiDAR ResNet"
        )
    )

    # These dataset defaults match scripts/train_LiDAR+LoRA.py.
    parser.add_argument(
        "--seq_roots",
        "--seq_root",
        type=str,
        nargs="+",
        default=[
            "/add02/users/xuyh/seq1/",
            "/add02/users/xuyh/seq3/",
        ],
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/xuyh/mapanything/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/add02/users/xuyh/checkpoints/"
            "contrastive_projected_lidar_resnet"
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/add02/users/xuyh/cache/lidar_9ch_calib",
    )
    parser.add_argument(
        "--lidar_extrinsics_path",
        type=str,
        default=(
            "/home/xuyh/mapanything/"
            "output/calibration/calibration_summary.json"
        ),
    )
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--max_samples_per_dataset", type=int, default=0)
    parser.add_argument("--max_rgb_brightness", type=float, default=1.0)
    parser.add_argument("--max_rgb_contrast", type=float, default=1.0)
    # Open3D projection/cache access is safer in the main process.
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--num_matches", type=int, default=4096)
    parser.add_argument("--lidar_input_size", type=int, default=512)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--proj_hidden_dim", type=int, default=1024)
    parser.add_argument("--projector_dropout", type=float, default=0.0)
    parser.add_argument("--train_dino", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lidar_lr_scale", type=float, default=0.5)
    parser.add_argument("--projector_lr_scale", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--rgb_to_lidar_weight", type=float, default=0.3)
    parser.add_argument("--lidar_to_rgb_weight", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument(
        "--empty_cache_each_step",
        action="store_true",
        default=False,
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "contrastive_args.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    roots = [resolve_root(root) for root in args.seq_roots]
    datasets = []
    for index, root in enumerate(roots):
        cache_subdir = (
            str(Path(args.cache_dir) / f"seq{index}")
            if args.cache_dir
            else None
        )
        dataset = make_dataset(
            seq_root=root,
            seq_len=args.seq_len,
            stride=args.stride,
            img_size=args.img_size,
            cache_dir=cache_subdir,
            tolerance=args.tolerance,
            use_lidar=True,
            max_samples=(
                args.max_samples_per_dataset
                if args.max_samples_per_dataset > 0
                else None
            ),
            max_rgb_brightness=args.max_rgb_brightness,
            max_rgb_contrast=args.max_rgb_contrast,
            lidar_extrinsics_path=args.lidar_extrinsics_path,
        )
        print(
            f"[Dataset] root={root} "
            f"valid_views={len(dataset.all_views_meta)} "
            f"effective_seq_len={dataset.seq_len} "
            f"stride={dataset.stride} samples={len(dataset)}"
        )
        datasets.append(dataset)

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if len(dataset) == 0:
        raise RuntimeError(
            "The dataset is empty. Check RGB/GT/LiDAR timestamp matching "
            "and the diagnostics printed by the reference dataset."
        )
    loader = make_loader(dataset, args, device)
    print(
        f"[Dataset] total_samples={len(dataset)} "
        f"batches={len(loader)} num_workers={args.num_workers}"
    )

    model = build_cross_modal_model(args, device)
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer_groups = [
        {
            "params": model.lidar_encoder.parameters(),
            "lr": args.lr * args.lidar_lr_scale,
            "weight_decay": args.weight_decay,
        },
        {
            "params": (
                list(model.lidar_projector.parameters())
                + list(model.rgb_projector.parameters())
            ),
            "lr": args.lr * args.projector_lr_scale,
            "weight_decay": args.weight_decay,
        },
    ]
    if args.train_dino:
        optimizer_groups.append(
            {
                "params": model.rgb_encoder.parameters(),
                "lr": args.lr * 0.05,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = AdamW(optimizer_groups)
    if not trainable:
        raise RuntimeError("No trainable parameters found")

    criterion = PixelSymmetricInfoNCE(
        temperature=args.temperature,
        rgb_to_lidar_weight=args.rgb_to_lidar_weight,
        lidar_to_rgb_weight=args.lidar_to_rgb_weight,
        num_matches=args.num_matches,
    )
    scaler = make_scaler(args.amp and device.type == "cuda")

    start_epoch = 1
    best_loss = math.inf
    if args.resume:
        state = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scaler,
        )
        start_epoch = int(state.get("epoch", 0)) + 1
        best_loss = float(state.get("best_loss", math.inf))

    print(
        f"[Model] device={device} "
        f"pixel_matches={args.num_matches} "
        f"train_dino={args.train_dino} trainable_params="
        f"{sum(parameter.numel() for parameter in trainable) / 1e6:.2f}M"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        average_loss = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            scaler,
            device,
            args,
            epoch,
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            best_loss,
            args,
        )
        if average_loss < best_loss:
            best_loss = average_loss
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_loss,
                args,
            )
            print(
                f"[Checkpoint] new best: {output_dir / 'best.pt'} "
                f"loss={best_loss:.6f}"
            )


if __name__ == "__main__":
    main()
