#!/usr/bin/env python3
#coding=gbk
from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

import argparse
import gc
import inspect
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed(enabled: bool):
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_local_module(module_filename: str, module_name: str):
    import importlib.util

    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / module_filename

    if not module_path.exists():
        module_path = Path.cwd() / module_filename

    if not module_path.exists():
        raise ImportError(
            f"Missing local module: {module_filename} "
            f"(tried {script_dir} and {Path.cwd()})"
        )

    if is_main_process(int(os.environ.get("RANK", 0))):
        print(f"[load_local_module] Loading '{module_name}' from: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_sequence_root(root: str) -> tuple[str, bool]:
    """Resolve a dataset root against the current working directory and repo layout."""

    raw = Path(root)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates = [raw]

    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, script_dir / raw, repo_root / raw])
    else:
        # If the absolute path does not exist in this environment, also try the
        # same leaf directory name under the common local layouts.
        leaf = raw.name
        candidates.extend([Path.cwd() / leaf, script_dir / leaf, repo_root / leaf])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate), str(candidate) != str(raw)

    return root, False


def build_dataset_kwargs(ds_cls, raw_kwargs: dict) -> dict:
    sig = inspect.signature(ds_cls.__init__)
    valid = set(sig.parameters.keys())
    filtered = {k: v for k, v in raw_kwargs.items() if k in valid}
    dropped = sorted(k for k in raw_kwargs.keys() if k not in valid)
    if dropped and is_main_process(0):
        print(f"[Dataset][WARN] Dropped unsupported dataset kwargs: {dropped}")
    return filtered


def _describe_path(path: str, max_entries: int = 12) -> list[str]:
    p = Path(path)
    if not p.exists():
        return [f"exists=False path={path}"]
    if p.is_file():
        return [f"exists=True file={path} size={p.stat().st_size}"]
    entries = []
    try:
        for child in sorted(p.iterdir()):
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.name}{suffix}")
            if len(entries) >= max_entries:
                break
    except Exception as exc:
        entries.append(f"<error listing {path}: {exc}>")
    return [f"exists=True dir={path} entries={entries}"]


def print_dataset_diagnostics(dataset, seq_roots, cache_dir: str):
    print("[Dataset][Diag] root summary:")
    for idx, root in enumerate(seq_roots):
        print(f"  root[{idx}]={root}")
        for line in _describe_path(root):
            print(f"    {line}")

    if hasattr(dataset, "datasets"):
        subdatasets = list(getattr(dataset, "datasets"))
        print(f"[Dataset][Diag] ConcatDataset with {len(subdatasets)} subdataset(s)")
        for idx, ds in enumerate(subdatasets):
            try:
                ds_len = len(ds)
            except Exception as exc:
                ds_len = f"<error {exc}>"
            print(f"  subdataset[{idx}] type={type(ds).__name__} len={ds_len}")
            for attr in ("seq_root", "part_folders", "gt_timestamps", "all_views_meta", "cache_dir"):
                if not hasattr(ds, attr):
                    continue
                value = getattr(ds, attr)
                if attr == "gt_timestamps" and value is not None:
                    try:
                        value = f"len={len(value)} first={value[0] if len(value) else 'n/a'} last={value[-1] if len(value) else 'n/a'}"
                    except Exception as exc:
                        value = f"<error {exc}>"
                elif attr == "all_views_meta" and value is not None:
                    try:
                        preview = value[:3]
                        value = f"len={len(value)} preview={preview}"
                    except Exception as exc:
                        value = f"<error {exc}>"
                print(f"    {attr}={value}")
            if hasattr(ds, "all_views_meta") and hasattr(ds, "seq_len"):
                try:
                    views_len = len(getattr(ds, "all_views_meta"))
                    seq_len = int(getattr(ds, "seq_len"))
                    stride = int(getattr(ds, "stride")) if hasattr(ds, "stride") else "n/a"
                    computed_len = max(0, (views_len - seq_len) // stride + 1) if isinstance(stride, int) else "n/a"
                    print(
                        f"    windowing: all_views_meta={views_len} seq_len={seq_len} stride={stride} "
                        f"computed_len={computed_len}"
                    )
                    if views_len < seq_len:
                        print(
                            f"    [WARN] all_views_meta has only {views_len} view(s) but seq_len={seq_len}; "
                            f"__len__() will be 0 with stride={stride}."
                        )
                except Exception as exc:
                    print(f"    [WARN] unable to evaluate dataset windowing: {exc}")
    else:
        print(f"[Dataset][Diag] dataset type={type(dataset).__name__}")
        for attr in ("seq_root", "part_folders", "gt_timestamps", "all_views_meta", "cache_dir"):
            if not hasattr(dataset, attr):
                continue
            value = getattr(dataset, attr)
            if attr == "gt_timestamps" and value is not None:
                try:
                    value = f"len={len(value)} first={value[0] if len(value) else 'n/a'} last={value[-1] if len(value) else 'n/a'}"
                except Exception as exc:
                    value = f"<error {exc}>"
            elif attr == "all_views_meta" and value is not None:
                try:
                    preview = value[:3]
                    value = f"len={len(value)} preview={preview}"
                except Exception as exc:
                    value = f"<error {exc}>"
            print(f"  {attr}={value}")
        if hasattr(dataset, "all_views_meta") and hasattr(dataset, "seq_len"):
            try:
                views_len = len(getattr(dataset, "all_views_meta"))
                seq_len = int(getattr(dataset, "seq_len"))
                stride = int(getattr(dataset, "stride")) if hasattr(dataset, "stride") else "n/a"
                computed_len = max(0, (views_len - seq_len) // stride + 1) if isinstance(stride, int) else "n/a"
                print(
                    f"  windowing: all_views_meta={views_len} seq_len={seq_len} stride={stride} "
                    f"computed_len={computed_len}"
                )
                if views_len < seq_len:
                    print(
                        f"  [WARN] all_views_meta has only {views_len} view(s) but seq_len={seq_len}; "
                        f"__len__() will be 0 with stride={stride}."
                    )
            except Exception as exc:
                print(f"  [WARN] unable to evaluate dataset windowing: {exc}")

    if hasattr(dataset, "seq_root") and hasattr(dataset, "part_folders") and hasattr(dataset, "gt_timestamps"):
        try:
            seq_root = Path(getattr(dataset, "seq_root"))
            part_folders = list(getattr(dataset, "part_folders"))
            gt_timestamps = np.asarray(getattr(dataset, "gt_timestamps"))
            tolerance = float(getattr(dataset, "tolerance", 0.05))
            use_lidar = bool(getattr(dataset, "use_lidar", True))
            print("[Dataset][Diag] per-part scan:")
            for pidx, part in enumerate(part_folders):
                rgb_dir = seq_root / part / "rgb"
                raw = 0
                ts_ok = 0
                time_ok = 0
                lidar_ok = 0
                examples = []
                if rgb_dir.exists():
                    for child in sorted(rgb_dir.iterdir()):
                        if not child.is_file():
                            continue
                        if child.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                            continue
                        raw += 1
                        match = re.search(r"color_(\d+)", child.name) or re.search(r"(\d+)", child.name)
                        if not match:
                            if len(examples) < 3:
                                examples.append(f"{part}/{child.name}:no_ts")
                            continue
                        ts_ok += 1
                        img_ts_ns = int(match.group(1))
                        img_ts_sec = img_ts_ns / 1e9
                        if gt_timestamps.size == 0:
                            if len(examples) < 3:
                                examples.append(f"{part}/{child.name}:no_gt_pose")
                            continue
                        pos = np.searchsorted(gt_timestamps, img_ts_sec)
                        if pos == 0:
                            nearest_idx = 0
                        elif pos == len(gt_timestamps):
                            nearest_idx = len(gt_timestamps) - 1
                        else:
                            left_diff = abs(gt_timestamps[pos - 1] - img_ts_sec)
                            right_diff = abs(gt_timestamps[pos] - img_ts_sec)
                            nearest_idx = pos - 1 if left_diff < right_diff else pos
                        diff = abs(gt_timestamps[nearest_idx] - img_ts_sec)
                        if diff > tolerance:
                            if len(examples) < 3:
                                examples.append(f"{part}/{child.name}:time_diff={diff:.6f}")
                            continue
                        time_ok += 1
                        if use_lidar and hasattr(dataset, "_find_closest_pcd"):
                            try:
                                if dataset._find_closest_pcd(img_ts_sec, pidx) is None:
                                    if len(examples) < 3:
                                        examples.append(f"{part}/{child.name}:no_lidar")
                                    continue
                            except Exception as exc:
                                if len(examples) < 3:
                                    examples.append(f"{part}/{child.name}:lidar_err={exc}")
                                continue
                        lidar_ok += 1
                print(
                    f"  part[{pidx}]={part}: raw={raw} ts_ok={ts_ok} time_ok={time_ok} lidar_ok={lidar_ok}"
                )
                if examples:
                    print(f"    examples={examples}")
        except Exception as exc:
            print(f"[Dataset][Diag][WARN] per-part scan failed: {exc}")
    print(f"[Dataset][Diag] cache_dir={cache_dir}")


def move_views_to_device(views, device: torch.device):
    moved = []
    for view in views:
        new_view = {}
        for key, value in view.items():
            if torch.is_tensor(value):
                new_view[key] = value.to(device, non_blocking=True)
            else:
                new_view[key] = value
        moved.append(new_view)
    return moved


def build_dataloader(dataset, args, sampler, device: torch.device, collate_fn, num_workers_override=None):
    num_workers = args.num_workers if num_workers_override is None else int(num_workers_override)
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 4 if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def make_optimizer(model, args):
    base = unwrap_model(model)
    param_groups = []

    seen_param_ids = set()

    def _unique_params(params):
        unique = []
        for p in params:
            pid = id(p)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            unique.append(p)
        return unique

    rgb_encoder_params = _unique_params([p for p in base.model.encoder.parameters() if p.requires_grad])
    lidar_encoder_params = _unique_params([p for p in base.model.lidars_encoder.parameters() if p.requires_grad])
    rgb_proj_params = _unique_params([p for p in base.rgb_proj.parameters() if p.requires_grad])
    lidar_proj_params = _unique_params([p for p in base.lidar_proj.parameters() if p.requires_grad])

    if rgb_encoder_params:
        param_groups.append(
            {
                "params": rgb_encoder_params,
                "lr": args.lr * args.rgb_lr_scale,
                "weight_decay": args.weight_decay,
                "name": "rgb_encoder",
            }
        )
    if lidar_encoder_params:
        param_groups.append(
            {
                "params": lidar_encoder_params,
                "lr": args.lr * args.lidar_lr_scale,
                "weight_decay": args.weight_decay,
                "name": "lidar_encoder",
            }
        )
    if rgb_proj_params:
        param_groups.append(
            {
                "params": rgb_proj_params,
                "lr": args.proj_lr,
                "weight_decay": args.weight_decay,
                "name": "rgb_proj",
            }
        )
    if lidar_proj_params and not args.share_projector:
        param_groups.append(
            {
                "params": lidar_proj_params,
                "lr": args.proj_lr,
                "weight_decay": args.weight_decay,
                "name": "lidar_proj",
            }
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")
    return AdamW(param_groups)


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, args, best_loss: float):
    base = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "best_loss": best_loss,
        "args": vars(args),
        "rgb_encoder": base.model.encoder.state_dict(),
        "lidar_encoder": base.model.lidars_encoder.state_dict(),
        "rgb_proj": base.rgb_proj.state_dict(),
        "lidar_proj": base.lidar_proj.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    base = unwrap_model(model)
    base.model.encoder.load_state_dict(ckpt["rgb_encoder"], strict=True)
    base.model.lidars_encoder.load_state_dict(ckpt["lidar_encoder"], strict=True)
    base.rgb_proj.load_state_dict(ckpt["rgb_proj"], strict=True)
    if "lidar_proj" in ckpt:
        base.lidar_proj.load_state_dict(ckpt["lidar_proj"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def reduce_metric(value: float, count: float, device: torch.device):
    tensor = torch.tensor([value, count], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total = tensor[0].item()
    denom = max(tensor[1].item(), 1.0)
    return total / denom


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args, rank: int, epoch: int):
    model.train()
    running_loss = 0.0
    running_count = 0.0
    start_time = time.time()

    if is_main_process(rank):
        print(f"[Epoch {epoch}] dataloader length: {len(loader)}")

    for step, batch in enumerate(loader):
        views = batch
        views = move_views_to_device(views, device)
        n_views = len(views)

        if n_views == 0:
            if is_main_process(rank) and step < 3:
                print(f"[WARN] step {step}: empty views")
            continue

        # Diagnostic: print first batch info
        if is_main_process(rank) and step == 0:
            v0 = views[0]
            print(f"[Diag] Batch0 has {n_views} views")
            for k in ["img", "pcd", "lidar_depth_scale", "confidence"]:
                if k in v0 and torch.is_tensor(v0[k]):
                    print(f"  {k}: shape={v0[k].shape} dtype={v0[k].dtype} device={v0[k].device} "
                          f"min={v0[k].min().item():.4f} max={v0[k].max().item():.4f} "
                          f"mean={v0[k].float().mean().item():.4f}")

        optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda")
        with amp_ctx:
            rgb_embed, lidar_embed = model(views, use_amp=args.amp and device.type == "cuda")
            loss = criterion(rgb_embed, lidar_embed)

        if not torch.isfinite(loss):
            if is_main_process(rank):
                print(f"[Epoch {epoch}] skip non-finite loss at step {step}: {loss.item()}")
            continue

        loss_value = float(loss.item())

        if is_main_process(rank) and step < 3:
            print(f"[Diag] step {step} loss={loss_value:.4f} rgb_embed={rgb_embed.shape if hasattr(rgb_embed, 'shape') else type(rgb_embed)} "
                  f"lidar_embed={lidar_embed.shape if hasattr(lidar_embed, 'shape') else type(lidar_embed)}")

        if scaler is not None and args.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        del loss
        del rgb_embed
        del lidar_embed
        del views
        del batch
        if device.type == "cuda" and args.empty_cache_each_step:
            torch.cuda.empty_cache()
        gc.collect()

        running_loss += loss_value * n_views
        running_count += float(n_views)

        if is_main_process(rank) and (step % args.log_every == 0):
            print(
                f"Epoch {epoch:03d} | step {step:05d}/{len(loader)} | "
                f"loss={loss_value:.4f} | views={n_views}"
            )

    avg_loss = reduce_metric(running_loss, running_count, device)
    epoch_time = time.time() - start_time
    if is_main_process(rank):
        print(f"Epoch {epoch} finished | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s | processed={int(running_count)} views")
    return avg_loss


def main():
    rank = 0
    distributed = False
    try:
        data_module = load_local_module("data.py", "contrastive_data_local")
        losses_module = load_local_module("losses.py", "contrastive_losses_local")
        model_module = load_local_module("model.py", "contrastive_model_local")

        missing = []
        if not hasattr(data_module, "Seq1LidarDataset"):
            missing.append("Seq1LidarDataset")
        if not hasattr(data_module, "collate_fn"):
            missing.append("collate_fn")
        if missing:
            available = [k for k in dir(data_module) if not k.startswith("_")]
            raise ImportError(
                f"{Path(__file__).resolve().with_name('data.py')} does not expose: {missing}\n"
                f"Available names in loaded module: {available[:50]}..."
            )

        if not hasattr(losses_module, "PairInfoNCELoss"):
            raise ImportError(
                f"{Path(__file__).resolve().with_name('losses.py')} does not expose PairInfoNCELoss"
            )
        if not hasattr(model_module, "build_trainable_contrastive_model"):
            raise ImportError(
                f"{Path(__file__).resolve().with_name('model.py')} does not expose build_trainable_contrastive_model"
            )

        parser = argparse.ArgumentParser(description="Contrastive training for RGB DINO vs LiDAR ResNet")
        parser.add_argument("--seq_root", "--seq_roots", type=str, nargs="+", default=["/add02/users/xuyh/seq1/", "/add02/users/xuyh/seq3/"], help="One or more dataset roots")
        parser.add_argument("--model_dir", type=str, default="/home/xuyh/mapanything/", help="Directory with config.json and model.safetensors")
        parser.add_argument("--output_dir", type=str, default="/add02/users/xuyh/checkpoints/contrastive_lidar_dino")
        parser.add_argument("--cache_dir", type=str, default="/add02/users/xuyh/cache/lidar_9ch_calib")
        parser.add_argument("--resume", type=str, default="")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--rgb_lr_scale", type=float, default=0.05)
        parser.add_argument("--lidar_lr_scale", type=float, default=0.5)
        parser.add_argument("--proj_lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--temperature", type=float, default=0.07)
        parser.add_argument("--rgb_to_lidar_weight", type=float, default=0.3)
        parser.add_argument("--lidar_to_rgb_weight", type=float, default=0.7)
        parser.add_argument("--proj_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=1024)
        parser.add_argument("--share_projector", action="store_true")
        parser.add_argument("--train_rgb_encoder", action="store_true", default=False)
        parser.add_argument("--no-train_rgb_encoder", action="store_false", dest="train_rgb_encoder")
        parser.add_argument("--train_lidar_encoder", action="store_true", default=True)
        parser.add_argument("--no-train_lidar_encoder", action="store_false", dest="train_lidar_encoder")
        parser.add_argument("--seq_len", type=int, default=2)
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument("--img_size", type=int, default=448)
        parser.add_argument("--tolerance", type=float, default=0.05)
        parser.add_argument("--max_samples_per_dataset", "--max_samples", type=int, default=0, help="Max samples per dataset; 0 or negative means no limit")
        parser.add_argument("--max_rgb_brightness", type=float, default=1)
        parser.add_argument("--max_rgb_contrast", type=float, default=1)
        parser.add_argument(
            "--lidar_extrinsics_path",
            type=str,
            default="/home/xuyh/mapanything/output/calibration/calibration_summary.json",
            help="LiDAR-to-camera extrinsics path",
        )
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--amp", action="store_true")
        parser.add_argument("--no-amp", action="store_false", dest="amp")
        parser.set_defaults(amp=True)
        parser.add_argument("--grad_checkpointing", action="store_true")
        parser.add_argument("--no-grad_checkpointing", action="store_false", dest="grad_checkpointing")
        parser.add_argument("--empty_cache_each_step", action="store_true")
        parser.add_argument("--no-empty_cache_each_step", action="store_false", dest="empty_cache_each_step")
        parser.set_defaults(grad_checkpointing=True, empty_cache_each_step=True)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--log_every", type=int, default=10)
        parser.add_argument("--save_every", type=int, default=1)
        args = parser.parse_args()

        rank, local_rank, world_size, distributed = setup_distributed()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        seed_everything(args.seed + rank)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process(rank):
            with (output_dir / "contrastive_args.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, ensure_ascii=False)

        ds_cls = data_module.Seq1LidarDataset
        if len(args.seq_root) == 1:
            dataset = ds_cls(
                seq_root=args.seq_root[0],
                seq_len=args.seq_len,
                stride=args.stride,
                img_size=args.img_size,
                cache_dir=args.cache_dir,
                tolerance=args.tolerance,
                use_lidar=True,
                max_samples=args.max_samples_per_dataset if args.max_samples_per_dataset > 0 else None,
                max_rgb_brightness=args.max_rgb_brightness,
                max_rgb_contrast=args.max_rgb_contrast,
                lidar_extrinsics_path=args.lidar_extrinsics_path,
            )
        else:
            datasets = []
            for idx, root in enumerate(args.seq_root):
                cache_subdir = os.path.join(args.cache_dir, f"seq{idx}") if args.cache_dir else None
                ds = ds_cls(
                    seq_root=root,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    img_size=args.img_size,
                    cache_dir=cache_subdir,
                    tolerance=args.tolerance,
                    use_lidar=True,
                    max_samples=args.max_samples_per_dataset if args.max_samples_per_dataset > 0 else None,
                    max_rgb_brightness=args.max_rgb_brightness,
                    max_rgb_contrast=args.max_rgb_contrast,
                    lidar_extrinsics_path=args.lidar_extrinsics_path,
                )
                datasets.append(ds)
            dataset = ConcatDataset(datasets)

        if is_main_process(rank):
            print(f"[Dataset] total samples: {len(dataset)} from {len(args.seq_root)} root(s)")
            print_dataset_diagnostics(dataset, args.seq_root, args.cache_dir)
        if len(dataset) == 0:
            msg = (
                "[Dataset][WARN] Final dataset is empty after loading. "
                f"seq_root={args.seq_root} cache_dir={args.cache_dir} "
                "Check root paths, folder layout, timestamps, and matching LiDAR/RGB files."
            )
            print(msg)
            raise RuntimeError("Loaded dataset is empty; aborting training.")

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=True)

        loader = build_dataloader(dataset, args, sampler, device, data_module.collate_fn)

        if is_main_process(rank):
            print(f"[DataLoader] batches per rank: {len(loader)}")

        model = model_module.build_trainable_contrastive_model(
            model_dir=args.model_dir,
            device=device,
            proj_dim=args.proj_dim,
            proj_hidden_dim=args.proj_hidden_dim,
            share_projector=args.share_projector,
            train_rgb_encoder=args.train_rgb_encoder,
            train_lidar_encoder=args.train_lidar_encoder,
        )

        model = model.to(device)
        if is_main_process(rank):
            devices = {p.device.type for p in model.parameters()}
            print(f"[Device check] model parameters are on: {devices}")

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        criterion = losses_module.PairInfoNCELoss(
            temperature=args.temperature,
            symmetric=True,
            rgb_to_lidar_weight=args.rgb_to_lidar_weight,
            lidar_to_rgb_weight=args.lidar_to_rgb_weight,
        )
        optimizer = make_optimizer(model, args)
        scaler = GradScaler(enabled=args.amp and device.type == "cuda")

        start_epoch = 1
        best_loss = math.inf
        if args.resume:
            ckpt = load_checkpoint(Path(args.resume), model, optimizer=optimizer, scaler=scaler)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_loss = float(ckpt.get("best_loss", math.inf))
            if is_main_process(rank):
                print(f"Resumed from {args.resume} at epoch {start_epoch}")

        for epoch in range(start_epoch, args.epochs + 1):
            if distributed and sampler is not None:
                sampler.set_epoch(epoch)
            try:
                avg_loss = train_one_epoch(
                    model=model,
                    loader=loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    args=args,
                    rank=rank,
                    epoch=epoch,
                )
            except RuntimeError as exc:
                err_text = str(exc)
                worker_crash = (
                    "DataLoader worker" in err_text
                    or "segmentation fault" in err_text.lower()
                    or "exited unexpectedly" in err_text.lower()
                )
                if not worker_crash or args.num_workers <= 0:
                    raise
                if is_main_process(rank):
                    print(
                        "[Dataset][WARN] DataLoader worker crashed; "
                        "retrying this epoch with num_workers=0 and persistent_workers=False."
                    )
                loader = build_dataloader(dataset, args, sampler, device, data_module.collate_fn, num_workers_override=0)
                if is_main_process(rank):
                    print(f"[DataLoader] fallback batches per rank: {len(loader)}")
                avg_loss = train_one_epoch(
                    model=model,
                    loader=loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    args=args,
                    rank=rank,
                    epoch=epoch,
                )

            if is_main_process(rank):
                last_path = output_dir / "last.pt"
                save_checkpoint(last_path, model, optimizer, scaler, epoch, args, best_loss)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = output_dir / "best.pt"
                    save_checkpoint(best_path, model, optimizer, scaler, epoch, args, best_loss)
                    print(f"New best checkpoint: {best_path} | loss={best_loss:.4f}")
                if epoch % args.save_every == 0:
                    epoch_path = output_dir / f"epoch_{epoch:03d}.pt"
                    save_checkpoint(epoch_path, model, optimizer, scaler, epoch, args, best_loss)

            if distributed:
                dist.barrier()
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
