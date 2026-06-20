#!/usr/bin/env python3
#coding=gbk
"""
MapAnything LiDAR Fusion Training Script (Joint Training: LoRA for RGB/Heads + Full Train for LiDAR)
================================================================================================
Training mode:
  - RGB Encoder (DINOv2)        : LoRA fine-tune  (rank=8, alpha=16)
  - Info Sharing (Transformer)    : LoRA fine-tune
  - Dense Head / Pose Head / Scale Head : LoRA fine-tune
  - LiDAR Encoder (ResNet-50)     : full training
  - LiDAR FiLM                  : full training
  - Fusion Conv (1x1)           : full training, zero-initialized as identity on RGB side

Key fixes:
  1. Global depth normalization (instead of per-frame) to preserve absolute scale.
  2. Grouped optimizer: LiDAR params lr*2, LoRA params lr, others frozen.
  3. Scheduler step() called right after optimizer step() at accumulation boundary.
  4. Forced logging of rpe_trans / rpe_rot / depth / ray instead of per-view clutter.
  5. Safe resume from Stage1 ckpt: load model weights only, re-init optimizer.
  6. DDP find_unused_parameters=True (必须开启，容忍动态 loss 路径导致的 unused params).
  7. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set before torch import.
  8. Loss function: use sum(loss_components) instead of in-place += on leaf tensor,
     ensuring total_loss always has requires_grad=True and DDP never skips backward.
  9. 每个 accumulation boundary 输出 GradNorm 用于监控梯度健康度.
 10. [关键修复] LinearWithLoRA 非 in-place forward，避免 DDP 下破坏 autograd.
 11. [关键修复] 空 loss 时跳过 backward，避免 0 梯度浪费计算.
 12. [关键修复] LoRA 初始化 std 增大 + lr 默认提高 + 独立 lora_lr 参数.
 13. [关键修复] resume 默认重置 optimizer，避免历史错误动量污染 LoRA.
 14. [关键修复] 增加模型输出 keys 诊断 + LoRA 专属梯度范数监控.
 15. [Modality Dropout] 训练时以一定概率随机丢弃 LiDAR 输入，防止 LiDAR 过拟合并强化 LoRA 的 RGB 学习能力.
 16. [Multi-Seq] 支持同时传入多个 seq_root 路径，自动 ConcatDataset 合并训练.
 17. [对齐第一个脚本] 损失函数、深度加载、LoRA 保存/恢复、日志格式完全与 Pure RGB LoRA 脚本一致.
"""

# ========================= 关键修复：必须在 import torch 之前设置 =========================
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import sys
import json
import time
import argparse
import glob
import re
import math
import random
import warnings
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image
from scipy.spatial.transform import Rotation as SciR
import open3d as o3d

from mapanything.models import MapAnything
from safetensors.torch import load_file

warnings.filterwarnings('ignore')

# ========================= DDP 工具 =========================

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        return rank, local_rank, world_size, True
    return 0, 0, 1, False

def cleanup_ddp(is_ddp):
    if is_ddp:
        torch.distributed.destroy_process_group()

def is_main_process(rank):
    return rank == 0

# ========================= LoRA (embedded, no external deps) =========================

class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear.out_features))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        lora = x.float() @ self.lora_A.float()
        lora = lora @ self.lora_B.float()
        lora = lora.to(out.dtype)
        if not torch.isfinite(out).all():
            print(f"[警告] DINOv2主路输出nan! 输入范围: [{x.min():.3f}, {x.max():.3f}]")
        return out + lora * self.scaling


def inject_lora_to_module(module: nn.Module, r: int = 8, lora_alpha: int = 16):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(child, r, lora_alpha))
        else:
            inject_lora_to_module(child, r, lora_alpha)


# ========================= 显存优化: Gradient Checkpointing =========================

def load_lidar_encoder_pretrained(model, rank):
    try:
        import torchvision.models as models
        src = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        src_state = src.state_dict()
        dst_state = model.lidars_encoder.state_dict()
        matched = {}

        for k, v in src_state.items():
            if k in dst_state and v.shape == dst_state[k].shape:
                matched[k] = v

        if len(matched) < 10:
            for prefix in ['backbone.', 'encoder.', 'model.', 'resnet.']:
                for k, v in src_state.items():
                    pk = prefix + k
                    if pk in dst_state and v.shape == dst_state[pk].shape:
                        matched[pk] = v

        if len(matched) < 10:
            for dk in list(dst_state.keys()):
                for sk, sv in src_state.items():
                    if dk.endswith('.' + sk) or sk.endswith('.' + dk.split('.')[-1]):
                        if sv.shape == dst_state[dk].shape:
                            matched[dk] = sv
                            break

        conv1_candidates = [k for k in matched if 'conv1.weight' in k and matched[k].dim() == 4]
        for ck in conv1_candidates:
            w = matched[ck]
            if w.shape[1] == 3 and ck in dst_state and dst_state[ck].shape[1] == 7:
                w_new = w.new_zeros(dst_state[ck].shape)
                w_new[:, :3, :, :] = w
                mean_ch = w.mean(dim=1, keepdim=True)
                w_new[:, 3:, :, :] = mean_ch.expand(-1, 4, -1, -1)
                matched[ck] = w_new
                if is_main_process(rank):
                    print(f"  -> LiDAR conv1 已扩展: {w.shape} -> {w_new.shape}")

        if matched:
            missing, unexpected = model.lidars_encoder.load_state_dict(matched, strict=False)
            if is_main_process(rank):
                print(f"  -> LiDAR编码器预训练权重: 成功加载 {len(matched)}/{len(dst_state)} 个参数")
                if missing:
                    print(f"     缺失 {len(missing)} 个key (将随机初始化): {missing[:3]}...")
                if unexpected:
                    print(f"     意外 {len(unexpected)} 个key")
        else:
            if is_main_process(rank):
                print("  -> 未能自动匹配LiDAR编码器预训练权重，将使用随机初始化")
    except Exception as e:
        if is_main_process(rank):
            print(f"  -> LiDAR编码器预训练权重加载失败（将随机初始化）: {e}")


def enable_gradient_checkpointing_safe(model, rank):
    import torch.utils.checkpoint as cp
    wrapped_names = []

    def _make_checkpoint_fn(orig_forward, name):
        def _forward(*args, **kwargs):
            return cp.checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)
        return _forward

    candidates = [
        'info_sharing', 'info_sharing_module', 'transformer',
        'encoder', 'lidars_encoder',
        'dense_head', 'pose_head', 'scale_head'
    ]
    for name in candidates:
        if hasattr(model, name):
            module = getattr(model, name)
            if module is None:
                continue
            try:
                module.forward = _make_checkpoint_fn(module.forward, name)
                wrapped_names.append(name)
            except Exception as e:
                if is_main_process(rank):
                    print(f"  -> 跳过 {name} checkpointing: {e}")

    if wrapped_names and is_main_process(rank):
        print(f"  -> Gradient Checkpointing 已启用模块: {wrapped_names}")
    elif is_main_process(rank):
        print("  -> 未自动识别到可启用 checkpointing 的模块名（不影响训练）")

# ========================= 辅助函数 =========================

def rotation_matrix_from_lookat(direction, up=np.array([0, 0, 1])):
    z_cam = direction / (np.linalg.norm(direction) + 1e-8)
    x_cam = np.cross(up, z_cam)
    norm_x = np.linalg.norm(x_cam)
    if norm_x < 1e-6:
        x_cam = np.cross(np.array([1, 0, 0]), z_cam)
        norm_x = np.linalg.norm(x_cam)
    x_cam = x_cam / (norm_x + 1e-8)
    y_cam = np.cross(z_cam, x_cam)
    return np.vstack([x_cam, y_cam, z_cam])

def compute_features_for_indices(pcd, indices, radius=0.1, max_nn=30):
    points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    M = len(indices)
    normals = np.zeros((M, 3), dtype=np.float32)
    curv = np.zeros(M, dtype=np.float32)
    aniso = np.zeros(M, dtype=np.float32)
    plan = np.zeros(M, dtype=np.float32)
    for i, idx in enumerate(indices):
        [k, neighbor_idx, _] = tree.search_radius_vector_3d(points[idx], radius)
        if k < 3:
            normals[i] = np.nan; curv[i] = np.nan; aniso[i] = np.nan; plan[i] = np.nan
            continue
        neighbors = points[neighbor_idx]
        centroid = np.mean(neighbors, axis=0)
        cov = np.cov((neighbors - centroid).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        λ1, λ2, λ3 = eigenvalues
        normals[i] = eigenvectors[:, 0]
        total = λ1 + λ2 + λ3
        curv[i] = λ1 / total if total > 1e-12 else 0.0
        if λ3 > 1e-12:
            aniso[i] = (λ3 - λ2) / λ3
            plan[i] = (λ2 - λ1) / λ3
        else:
            aniso[i] = 0.0; plan[i] = 0.0
    return normals, curv, aniso, plan


# ========================= 全局深度归一化（保留绝对尺度） =========================
GLOBAL_DEPTH_MAX = 40.0


def generate_pcd_7channel(pcd_path: str, gen_size: int = 224) -> Tuple[np.ndarray, float]:
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return np.zeros((7, gen_size, gen_size), dtype=np.float32), 1.0

    H = W = gen_size
    f = W / 2; cx = cy = W / 2
    direction = np.array([1, 0, 0])
    R_mat = rotation_matrix_from_lookat(direction)
    pts_cam = points @ R_mat.T
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid_mask = z > 0
    if not np.any(valid_mask):
        return np.zeros((7, gen_size, gen_size), dtype=np.float32), 1.0

    x, y, z = x[valid_mask], y[valid_mask], -z[valid_mask]
    u = (f * x / z) + cx; v = (f * y / z) + cy
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_image):
        return np.zeros((7, gen_size, gen_size), dtype=np.float32), 1.0

    u = u[in_image].astype(int); v = v[in_image].astype(int); z = z[in_image]
    valid_indices = np.where(valid_mask)[0][in_image]
    normals, curv, aniso, plan = compute_features_for_indices(pcd, valid_indices)
    good_mask = ~np.isnan(curv)
    if not np.any(good_mask):
        return np.zeros((7, gen_size, gen_size), dtype=np.float32), 1.0

    u, v, z = u[good_mask], v[good_mask], z[good_mask]
    normals = normals[good_mask]; curv = curv[good_mask]; aniso = aniso[good_mask]; plan = plan[good_mask]

    depth_img = np.full((H, W), np.inf, dtype=np.float32)
    normal_img = np.zeros((H, W, 3), dtype=np.float32)
    curv_img = np.zeros((H, W), dtype=np.float32)
    aniso_img = np.zeros((H, W), dtype=np.float32)
    plan_img = np.zeros((H, W), dtype=np.float32)
    for i in range(len(u)):
        ui, vi = u[i], v[i]
        if z[i] < depth_img[vi, ui]:
            depth_img[vi, ui] = z[i]
            normal_img[vi, ui] = normals[i]
            curv_img[vi, ui] = curv[i]
            aniso_img[vi, ui] = aniso[i]
            plan_img[vi, ui] = plan[i]

    finite = np.isfinite(depth_img)
    if not np.any(finite):
        depth_img = np.zeros((H, W), dtype=np.float32)
        z_d = 1.0
    else:
        depth_img[~finite] = 0.0
        z_d = float(np.mean(depth_img[finite]))
        z_d = max(z_d, 1e-3)

    cap = np.stack([curv_img, aniso_img, plan_img], axis=0)
    cap = np.clip(cap, 0.0, 1.0).astype(np.float32)

    depth_rel = np.clip(depth_img / GLOBAL_DEPTH_MAX, 0.0, 1.0).astype(np.float32)
    depth_rel = depth_rel[np.newaxis, ...]

    normal = normal_img.transpose(2, 0, 1).astype(np.float32)
    pcd_7ch = np.concatenate([cap, depth_rel, normal], axis=0)

    return pcd_7ch, z_d


# ========================= 数据集 =========================

class Seq1LidarDataset(Dataset):
    def __init__(self, seq_root: str, seq_len: int = 2, stride: int = 1,
                 img_size: int = 448, pcd_gen_size: int = 224,
                 cache_dir: str = None, tolerance: float = 0.01,
                 img_exts: tuple = None):
        super().__init__()
        self.seq_root = seq_root
        self.seq_len = seq_len
        self.stride = stride
        self.img_size = img_size
        self.pcd_gen_size = pcd_gen_size
        self.tolerance = tolerance
        self.cache_dir = cache_dir
        if img_exts is None:
            img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
        self.img_exts = tuple(e.lower() for e in img_exts)

        self.part_folders = []
        for item in sorted(os.listdir(seq_root)):
            full_path = os.path.join(seq_root, item)
            if os.path.isdir(full_path) and re.match(r'shuangchuang_seq\d+_(night|daytime)\d+th', item):
                self.part_folders.append(item)
        if not self.part_folders:
            raise ValueError(f"在 {seq_root} 中未找到符合命名规则的 part 文件夹")

        if is_main_process(int(os.environ.get('RANK', 0))):
            print(f"[Dataset] 共识别到 {len(self.part_folders)} 个 part 文件夹: {self.part_folders}")

        self.gt_poses = self._load_tum_poses(os.path.join(seq_root, "extrinsics.tum"))
        self.gt_timestamps = np.array(sorted(self.gt_poses.keys()))
        self.gt_poses_list = [self.gt_poses[ts] for ts in self.gt_timestamps]
        if is_main_process(int(os.environ.get('RANK', 0))):
            print(f"[Dataset] TUM 真值位姿: {len(self.gt_poses_list)} 帧, 时间戳范围 [{self.gt_timestamps.min():.3f}, {self.gt_timestamps.max():.3f}]")

        # [对齐第一个脚本] 深度图按 part 惰性加载，支持多格式
        self.part_depths = []
        for pidx, part in enumerate(self.part_folders):
            depth_dir = os.path.join(seq_root, part, "depth")
            if os.path.exists(depth_dir):
                depth_files = sorted(glob.glob(os.path.join(depth_dir, "*")))
                valid_files = []
                depth_ts = []
                for df in depth_files:
                    if not os.path.isfile(df):
                        continue
                    ext = os.path.splitext(df)[1].lower()
                    if ext not in ('.npy', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'):
                        continue
                    m = re.search(r'(\d+)', os.path.basename(df))
                    if m:
                        ts = int(m.group(1))
                        valid_files.append(df)
                        depth_ts.append(ts)
                if is_main_process(int(os.environ.get('RANK', 0))):
                    print(f"[Dataset] {part}/depth: 扫描到 {len(valid_files)} 张深度图")
                    if len(valid_files) > 0:
                        print(f"[Dataset] {part}/depth: 样本键={depth_ts[:3]}")
                self.part_depths.append({
                    'files': valid_files,
                    'timestamps': np.array(depth_ts, dtype=np.int64)
                })
            else:
                self.part_depths.append({'files': [], 'timestamps': np.array([], dtype=np.int64)})

        intrinsics_file = os.path.join(seq_root, "color_camera_intrinsics.txt")
        self.intrinsics = self._load_intrinsics(intrinsics_file) if os.path.exists(intrinsics_file) else np.eye(3, dtype=np.float32)

        self.all_views_meta = []
        timestamp_diffs = []
        for pidx, part in enumerate(self.part_folders):
            rgb_dir = os.path.join(seq_root, part, "rgb")
            if not os.path.exists(rgb_dir):
                if is_main_process(int(os.environ.get('RANK', 0))):
                    print(f"[Dataset] 警告: {part}/rgb 不存在，跳过")
                continue
            img_paths = []
            for fname in sorted(os.listdir(rgb_dir)):
                if fname.lower().endswith(self.img_exts):
                    img_paths.append(os.path.join(rgb_dir, fname))
            raw_img_count = len(img_paths)
            regex_ok_count = 0; ts_ok_count = 0
            for ipath in img_paths:
                basename = os.path.basename(ipath)
                match = re.search(r'color_(\d+)', basename)
                if not match:
                    match = re.search(r'(\d+)', basename)
                if not match:
                    continue
                regex_ok_count += 1
                # [对齐第一个脚本] 存储纳秒时间戳
                img_ts_ns = int(match.group(1))
                img_ts_sec = img_ts_ns / 1e9
                pos = np.searchsorted(self.gt_timestamps, img_ts_sec)
                if pos == 0:
                    nearest_idx = 0
                elif pos == len(self.gt_timestamps):
                    nearest_idx = len(self.gt_timestamps) - 1
                else:
                    left_diff = abs(self.gt_timestamps[pos - 1] - img_ts_sec)
                    right_diff = abs(self.gt_timestamps[pos] - img_ts_sec)
                    nearest_idx = pos - 1 if left_diff < right_diff else pos
                diff = abs(self.gt_timestamps[nearest_idx] - img_ts_sec)
                if diff < tolerance:
                    self.all_views_meta.append((ipath, img_ts_ns, pidx, nearest_idx))
                    ts_ok_count += 1
                else:
                    timestamp_diffs.append(diff)
            if is_main_process(int(os.environ.get('RANK', 0))):
                print(f"[Dataset] {part}: 原始图像 {raw_img_count} | 正则匹配 {regex_ok_count} | 时间戳匹配成功 {ts_ok_count} (tolerance={tolerance}s)")

        if timestamp_diffs and is_main_process(int(os.environ.get('RANK', 0))):
            diffs = np.array(timestamp_diffs)
            print(f"[Dataset] 时间戳匹配失败统计: 共 {len(diffs)} 帧, 差值中位数 {np.median(diffs):.3f}s, 均值 {np.mean(diffs):.3f}s")
            if np.median(diffs) > 1.0:
                print("[Dataset] 警告: 时间戳单位可能不匹配!")
            elif np.median(diffs) > tolerance * 10:
                print(f"[Dataset] 提示: 建议调大 --tolerance")

        if not self.all_views_meta and is_main_process(int(os.environ.get('RANK', 0))):
            raise ValueError("没有任何图像通过时间戳匹配！")

        self.part_pcds = []
        for part in self.part_folders:
            lidar_dir = os.path.join(seq_root, part, "lidar")
            pcd_files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd")))
            pcd_ts = []
            for pf in pcd_files:
                m = re.search(r'(\d+)', os.path.basename(pf))
                ts = int(m.group(1)) if m else int(os.path.getmtime(pf) * 1e9)
                pcd_ts.append(ts)
            self.part_pcds.append({'files': pcd_files, 'timestamps': np.array(pcd_ts, dtype=np.int64)})

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._build_cache()
        if is_main_process(int(os.environ.get('RANK', 0))):
            print(f"[Dataset] 最终有效帧总数: {len(self.all_views_meta)}")

    def _load_tum_poses(self, tum_file: str):
        poses = {}
        if not os.path.exists(tum_file):
            return poses
        with open(tum_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                ts = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
                rot = SciR.from_quat([qx, qy, qz, qw]).as_matrix()
                T_w2c = np.eye(4, dtype=np.float32)
                T_w2c[:3, :3] = rot; T_w2c[:3, 3] = [tx, ty, tz]
                poses[ts] = np.linalg.inv(T_w2c)
        return poses

    def _load_intrinsics(self, file_path: str):
        try:
            data = np.loadtxt(file_path)
            if data.shape == (3, 3):
                return data.astype(np.float32)
            if data.size == 9:
                return data.reshape(3, 3).astype(np.float32)
        except Exception:
            pass
        return np.eye(3, dtype=np.float32)

    def _find_closest_depth(self, img_ts_ns: int, part_idx: int):
        depth_info = self.part_depths[part_idx]
        if len(depth_info['timestamps']) == 0:
            return None
        idx = np.argmin(np.abs(depth_info['timestamps'] - img_ts_ns))
        matched_ts = int(depth_info['timestamps'][idx])
        diff_ns = abs(matched_ts - img_ts_ns)
        if diff_ns < self.tolerance * 1e9:
            return depth_info['files'][idx]
        return None

    def _load_depth_from_path(self, df: str) -> np.ndarray:
        ext = os.path.splitext(df)[1].lower()
        if ext == '.npy':
            d = np.load(df).astype(np.float32)
        else:
            img = Image.open(df)
            d = np.array(img).astype(np.float32)
            if d.ndim == 3:
                d = d.mean(axis=2)
            d = d / 1000.0
        return d

    def _find_closest_pcd(self, img_ts_sec: float, part_idx: int):
        ts_ns = int(img_ts_sec * 1e9)
        pcd_info = self.part_pcds[part_idx]
        if len(pcd_info['timestamps']) == 0:
            return None
        idx = np.argmin(np.abs(pcd_info['timestamps'] - ts_ns))
        return pcd_info['files'][idx]

    def _build_cache(self):
        if is_main_process(int(os.environ.get('RANK', 0))):
            print("预计算 7 通道 LiDAR 特征...")
        for i, (ipath, img_ts_ns, pidx, _) in enumerate(self.all_views_meta):
            cache_path = os.path.join(self.cache_dir, f"pcd_{img_ts_ns}.npz")
            if os.path.exists(cache_path):
                continue
            img_ts_sec = img_ts_ns / 1e9
            pcd_path = self._find_closest_pcd(img_ts_sec, pidx)
            if pcd_path is None:
                continue
            feat, z_d = generate_pcd_7channel(pcd_path, self.pcd_gen_size)
            np.savez(cache_path, feat=feat, scale=z_d)
        if is_main_process(int(os.environ.get('RANK', 0))):
            print("缓存完成")

    def _get_pcd_feature(self, img_ts_sec: float, pidx: int):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"pcd_{int(img_ts_sec*1e9)}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                return data["feat"].astype(np.float32), float(data["scale"])
        pcd_path = self._find_closest_pcd(img_ts_sec, pidx)
        if pcd_path is None:
            zeros = np.zeros((7, self.pcd_gen_size, self.pcd_gen_size), dtype=np.float32)
            return zeros, 1.0
        return generate_pcd_7channel(pcd_path, self.pcd_gen_size)

    def __len__(self):
        return max(0, (len(self.all_views_meta) - self.seq_len) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        indices = list(range(start, start + self.seq_len))
        if indices[-1] >= len(self.all_views_meta):
            indices = list(range(start, len(self.all_views_meta)))
            while len(indices) < self.seq_len:
                indices.append(indices[-1])
        views = []
        for meta_idx in indices:
            # [对齐第一个脚本] 解包纳秒时间戳
            ipath, img_ts_ns, pidx, gt_idx = self.all_views_meta[meta_idx]
            img_ts_sec = img_ts_ns / 1e9

            img = Image.open(ipath).convert('RGB')
            img_np_color = np.array(img)
            if not np.isfinite(img_np_color).all():
                if is_main_process(int(os.environ.get('RANK', 0))):
                    print(f"[Dataset] 警告: {ipath} 包含非有限像素，使用零填充")
                img_np_color = np.zeros_like(img_np_color)
            img_np = img_np_color.transpose(2, 0, 1).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            if img_tensor.shape[1] != self.img_size or img_tensor.shape[2] != self.img_size:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)

            img_gray_np = img_np_color.astype(np.float32).mean(axis=2) / 255.0
            mean = np.mean(img_gray_np)
            rms = np.sqrt(np.mean((img_gray_np - mean) ** 2))
            confidence = float(rms) if not np.isnan(rms) else 0.5

            pcd_7ch, z_d = self._get_pcd_feature(img_ts_sec, pidx)
            pcd_tensor = torch.from_numpy(pcd_7ch)
            if pcd_tensor.shape[1] != self.img_size or pcd_tensor.shape[2] != self.img_size:
                pcd_tensor = F.interpolate(
                    pcd_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            pcd_tensor = pcd_tensor.permute(1, 2, 0).unsqueeze(0)

            gt_pose = torch.from_numpy(self.gt_poses_list[gt_idx])

            # [对齐第一个脚本] 惰性加载深度图，支持多格式
            gt_depth = torch.zeros((1, 1, self.img_size, self.img_size), dtype=torch.float32)
            depth_file = self._find_closest_depth(img_ts_ns, pidx)
            if depth_file is not None:
                d = self._load_depth_from_path(depth_file)
                d_tensor = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
                if d_tensor.shape[-2:] != (self.img_size, self.img_size):
                    d_tensor = F.interpolate(d_tensor, size=(self.img_size, self.img_size),
                                             mode='bilinear', align_corners=False)
                gt_depth = d_tensor

            gt_intrinsics = torch.from_numpy(self.intrinsics).unsqueeze(0)

            views.append({
                "img": img_tensor.unsqueeze(0),
                "pcd": pcd_tensor,
                "lidar_depth_scale": torch.tensor([z_d], dtype=torch.float32),
                "data_norm_type": ["dinov2"],
                "confidence": torch.tensor(confidence, dtype=torch.float32),
                "gt_pose": gt_pose.unsqueeze(0),
                "gt_depth": gt_depth,
                "gt_intrinsics": gt_intrinsics,
            })
        return views


def collate_fn(batch):
    views = []
    for seq in batch:
        views.extend(seq)
    return views


# ========================= 损失函数（与第一个脚本完全一致） =========================

class MapAnythingLoss(nn.Module):
    def __init__(self, w_depth=1.0, w_pose_trans=3.0, w_pose_rot=1.0, w_ray=0.5, w_pts3d_cam=1.0):
        super().__init__()
        self.w_depth = w_depth; self.w_pose_trans = w_pose_trans; self.w_pose_rot = w_pose_rot
        self.w_ray = w_ray; self.w_pts3d_cam = w_pts3d_cam

    @staticmethod
    def _quat_to_rotmat(quats):
        norm = torch.norm(quats, dim=-1, keepdim=True)
        quats = quats / (norm + 1e-8)
        qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        B = quats.shape[0]
        rot = torch.zeros(B, 3, 3, device=quats.device, dtype=quats.dtype)
        rot[:, 0, 0] = 1 - 2*(qy**2 + qz**2)
        rot[:, 0, 1] = 2*(qx*qy - qz*qw)
        rot[:, 0, 2] = 2*(qx*qz + qy*qw)
        rot[:, 1, 0] = 2*(qx*qy + qz*qw)
        rot[:, 1, 1] = 1 - 2*(qx**2 + qz**2)
        rot[:, 1, 2] = 2*(qy*qz - qx*qw)
        rot[:, 2, 0] = 2*(qx*qz - qy*qw)
        rot[:, 2, 1] = 2*(qy*qz + qx*qw)
        rot[:, 2, 2] = 1 - 2*(qx**2 + qy**2)
        return rot

    @staticmethod
    def _build_transform(trans, quats):
        B = trans.shape[0]
        R = MapAnythingLoss._quat_to_rotmat(quats)
        T = torch.eye(4, device=trans.device, dtype=trans.dtype).unsqueeze(0).repeat(B, 1, 1)
        T[:, :3, :3] = R; T[:, :3, 3] = trans
        return T

    @staticmethod
    def _inv_transform(T):
        R = T[:, :3, :3]; t = T[:, :3, 3]
        T_inv = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
        R_T = R.transpose(-2, -1)
        T_inv[:, :3, :3] = R_T
        T_inv[:, :3, 3] = -(R_T @ t.unsqueeze(-1)).squeeze(-1)
        return T_inv

    def forward(self, predictions: List[Dict], views: List[Dict], seq_len: int = 2):
        device = predictions[0]['pts3d'].device
        loss_components = []
        metrics = {}
        N = len(predictions)

        for i, (pred, view) in enumerate(zip(predictions, views)):
            if 'depth_along_ray' in pred and view.get('gt_depth') is not None:
                pred_d = pred['depth_along_ray']; gt_d = view['gt_depth'].to(device)
                pred_d = pred_d.permute(0, 3, 1, 2)
                if pred_d.shape[-2:] != gt_d.shape[-2:]:
                    gt_d = F.interpolate(gt_d, size=pred_d.shape[-2:], mode='bilinear', align_corners=False)
                valid = (gt_d > 1e-3) & (pred_d > 1e-3)
                if valid.any():
                    pred_d_clamped = torch.clamp(pred_d[valid], min=1e-6)
                    loss_d = F.l1_loss(torch.log(pred_d_clamped), torch.log(gt_d[valid] + 1e-6))
                    loss_components.append(self.w_depth * loss_d)
                    metrics[f'depth_{i}'] = loss_d.item()

            if 'ray_directions' in pred and view.get('gt_intrinsics') is not None:
                K = view['gt_intrinsics'].to(device)
                if K.dim() == 2: K = K.unsqueeze(0)
                pred_ray = pred['ray_directions']
                B, H, W, _ = pred_ray.shape
                u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
                pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).float()
                rays_cam = (torch.linalg.inv(K) @ pixels.reshape(-1, 3).T.unsqueeze(0).expand(B, -1, -1))
                rays_cam = rays_cam.permute(0, 2, 1).reshape(B, H, W, 3)
                rays_cam = rays_cam / (torch.norm(rays_cam, dim=-1, keepdim=True) + 1e-8)
                cos_sim = F.cosine_similarity(pred_ray, rays_cam, dim=-1)
                loss_ray = (1 - cos_sim).mean()
                loss_components.append(self.w_ray * loss_ray)
                metrics[f'ray_{i}'] = loss_ray.item()

            if 'pts3d_cam' in pred and view.get('gt_depth') is not None and view.get('gt_intrinsics') is not None:
                gt_d = view['gt_depth'].to(device)
                if gt_d.dim() == 2: gt_d = gt_d.unsqueeze(0).unsqueeze(0)
                elif gt_d.dim() == 3: gt_d = gt_d.unsqueeze(0)
                K = view['gt_intrinsics'].to(device)
                if K.dim() == 2: K = K.unsqueeze(0)
                B = K.shape[0]; H, W = pred['pts3d_cam'].shape[1:3]
                if gt_d.shape[-2:] != (H, W):
                    gt_d = F.interpolate(gt_d, size=(H, W), mode='bilinear', align_corners=False)
                u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
                pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).float()
                rays = (torch.linalg.inv(K) @ pixels.reshape(-1, 3).T.unsqueeze(0).expand(B, -1, -1))
                rays = rays.permute(0, 2, 1).reshape(B, H, W, 3)
                rays = rays / (torch.norm(rays, dim=-1, keepdim=True) + 1e-8)
                gt_pts3d_cam = rays * gt_d.permute(0, 2, 3, 1)
                pred_pts3d_cam = pred['pts3d_cam']
                valid = (gt_d.permute(0, 2, 3, 1) > 1e-3).expand_as(pred_pts3d_cam)
                if valid.any():
                    loss_pc = F.l1_loss(pred_pts3d_cam[valid], gt_pts3d_cam[valid])
                    loss_components.append(self.w_pts3d_cam * loss_pc)
                    metrics[f'pts3d_cam_{i}'] = loss_pc.item()

        if seq_len >= 2 and N >= seq_len:
            batch_size = N // seq_len
            for b in range(batch_size):
                start = b * seq_len
                for k in range(seq_len - 1):
                    idx1 = start + k; idx2 = start + k + 1
                    pred1, pred2 = predictions[idx1], predictions[idx2]
                    view1, view2 = views[idx1], views[idx2]
                    if not ('cam_trans' in pred1 and 'cam_quats' in pred1 and 'cam_trans' in pred2 and 'cam_quats' in pred2 and view1.get('gt_pose') is not None and view2.get('gt_pose') is not None):
                        continue
                    gt_pose1 = view1['gt_pose'].to(device)
                    gt_pose2 = view2['gt_pose'].to(device)
                    if gt_pose1.dim() == 2: gt_pose1 = gt_pose1.unsqueeze(0)
                    if gt_pose2.dim() == 2: gt_pose2 = gt_pose2.unsqueeze(0)
                    T_pred1 = self._build_transform(pred1['cam_trans'], pred1['cam_quats'])
                    T_pred2 = self._build_transform(pred2['cam_trans'], pred2['cam_quats'])
                    T_rel_gt = self._inv_transform(gt_pose1) @ gt_pose2
                    T_rel_pred = self._inv_transform(T_pred1) @ T_pred2
                    if torch.isnan(T_pred1).any() or torch.isinf(T_pred1).any() or torch.isnan(T_pred2).any() or torch.isinf(T_pred2).any():
                        continue
                    if torch.isnan(T_rel_pred).any() or torch.isinf(T_rel_pred).any() or torch.isnan(T_rel_gt).any() or torch.isinf(T_rel_gt).any():
                        continue
                    t_norm_pred = torch.norm(T_rel_pred[:, :3, 3], dim=-1).mean()
                    t_norm_gt = torch.norm(T_rel_gt[:, :3, 3], dim=-1).mean()
                    if not torch.isfinite(t_norm_pred) or not torch.isfinite(t_norm_gt):
                        continue
                    if t_norm_gt < 1e-12:
                        continue
                    if t_norm_pred > 10000.0:
                        continue
                    loss_t = F.smooth_l1_loss(T_rel_pred[:, :3, 3], T_rel_gt[:, :3, 3], beta=0.5)
                    R_pred = T_rel_pred[:, :3, :3]
                    R_gt = T_rel_gt[:, :3, :3]
                    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
                    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
                    trace = torch.clamp(trace, -1.0, 3.0)
                    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
                    angle = torch.acos(cos_angle)
                    loss_r = angle.mean()
                    if not torch.isfinite(loss_t) or not torch.isfinite(loss_r):
                        continue
                    loss_components.append(self.w_pose_trans * loss_t + self.w_pose_rot * loss_r)
                    metrics[f'rpe_trans_b{b}_k{k}'] = loss_t.item()
                    metrics[f'rpe_rot_b{b}_k{k}'] = loss_r.item()

        rpe_trans_vals = [v for k, v in metrics.items() if k.startswith('rpe_trans_b')]
        rpe_rot_vals   = [v for k, v in metrics.items() if k.startswith('rpe_rot_b')]
        if rpe_trans_vals:
            metrics['rpe_trans'] = sum(rpe_trans_vals) / len(rpe_trans_vals)
            metrics['rpe_rot']   = sum(rpe_rot_vals) / len(rpe_rot_vals)
        depth_vals = [v for k, v in metrics.items() if k.startswith('depth_')]
        if depth_vals:
            metrics['depth'] = sum(depth_vals) / len(depth_vals)
        ray_vals = [v for k, v in metrics.items() if k.startswith('ray_')]
        if ray_vals:
            metrics['ray'] = sum(ray_vals) / len(ray_vals)

        if loss_components:
            total_loss = sum(loss_components)
        else:
            metrics['total_loss'] = 0.0
            metrics['no_valid_loss'] = True
            return None, metrics

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics


# ========================= 模型构建 =========================

def build_model(model_dir: str, device: str, rank: int, use_compile: bool = True, lora_r: int = 8, lora_alpha: int = 16):
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    with open(config_path, 'r') as f:
        config = json.load(f)
    encoder_config = config.get("encoder_config", {}).copy()
    encoder_config.pop("pretrained", None); encoder_config.pop("weights", None)
    encoder_config["uses_torch_hub"] = False
    model = MapAnything(
        name=config.get("name", "mapanything"),
        encoder_config=encoder_config,
        info_sharing_config=config.get("info_sharing_config", {}),
        pred_head_config=config.get("pred_head_config", {}),
        geometric_input_config=config.get("geometric_input_config", {}),
        pretrained_checkpoint_path=None,
        torch_hub_force_reload=False,
        info_sharing_mlp_layer_str="swiglufused"
    )
    if os.path.exists(weights_path):
        if is_main_process(rank):
            print(f"加载预训练权重: {weights_path}")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
    else:
        if is_main_process(rank):
            print("警告: 未找到预训练权重")

    enc_dim = model.encoder.enc_embed_dim
    with torch.no_grad():
        model.fusion_conv.weight.zero_()
        eye = torch.eye(enc_dim, device=model.fusion_conv.weight.device)
        model.fusion_conv.weight[:, :enc_dim, 0, 0] = eye
        model.fusion_conv.bias.zero_()
    if is_main_process(rank):
        print("  -> fusion_conv 已零初始化（RGB 侧恒等，LiDAR 侧全零）")

    if hasattr(model, 'lidars_encoder') and model.lidars_encoder is not None:
        load_lidar_encoder_pretrained(model, rank)

    lora_targets = []
    if hasattr(model, 'encoder') and model.encoder is not None:
        lora_targets.append(model.encoder)
    if hasattr(model, 'info_sharing') and model.info_sharing is not None:
        lora_targets.append(model.info_sharing)
    if hasattr(model, 'dense_head') and model.dense_head is not None:
        lora_targets.append(model.dense_head)
    if hasattr(model, 'pose_head') and model.pose_head is not None:
        lora_targets.append(model.pose_head)
    if hasattr(model, 'scale_head') and model.scale_head is not None:
        lora_targets.append(model.scale_head)

    for target in lora_targets:
        inject_lora_to_module(target, r=lora_r, lora_alpha=lora_alpha)

    if is_main_process(rank):
        lora_layer_count = sum(1 for _ in model.modules() if isinstance(_, LinearWithLoRA))
        print(f"  -> 已注入 LoRA: {lora_layer_count} 个 Linear 层 (r={lora_r}, alpha={lora_alpha})")

    nan_weights = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            nan_weights.append(name)
    if nan_weights:
        print(f"  -> 警告: {len(nan_weights)} 个参数含nan/inf!")
        for n in nan_weights[:5]:
            print(f"     {n}")
    else:
        print("  -> 所有预训练权重检查通过（无nan/inf）")

    for param in model.parameters():
        param.requires_grad = False

    lidar_keywords = ('lidars_encoder', 'lidar_film', 'fusion_conv')
    trainable_names = []
    for name, param in model.named_parameters():
        clean_name = name.replace('_orig_mod.', '')
        if clean_name.startswith(lidar_keywords):
            param.requires_grad = True
            trainable_names.append(name)
        elif 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            trainable_names.append(name)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable == 0:
        if is_main_process(rank):
            print("=" * 60)
            print("错误: 未匹配到任何可训练参数！")
            for name, param in model.named_parameters():
                print(f"  {name}: shape={tuple(param.shape)}")
            print("=" * 60)
        raise RuntimeError("没有任何可训练参数，请检查模块名匹配规则。")

    if is_main_process(rank):
        print(f"  -> 总参数量: {total/1e6:.2f}M, 可训练: {trainable/1e6:.2f}M ({len(trainable_names)} modules)")
        print(f"     LiDAR+Fusion 参数: {sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and any(k in n for k in lidar_keywords))/1e6:.2f}M")
        print(f"     LoRA 参数: {sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and ('lora_A' in n or 'lora_B' in n))/1e6:.2f}M")

    enable_gradient_checkpointing_safe(model, rank)

    model = model.to(device)

    if is_main_process(rank):
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and not torch.isfinite(output).all():
                    print(f"[HOOK] {name} 输出含nan! 形状={tuple(output.shape)}, 范围=[{output.min():.3f}, {output.max():.3f}]")
                elif isinstance(output, tuple):
                    for i, o in enumerate(output):
                        if isinstance(o, torch.Tensor) and not torch.isfinite(o).all():
                            print(f"[HOOK] {name} 输出[{i}]含nan! 形状={tuple(o.shape)}")
            return hook

        if hasattr(model, 'encoder') and hasattr(model.encoder, 'model') and hasattr(model.encoder.model, 'blocks'):
            for i, blk in enumerate(model.encoder.model.blocks):
                if i < 3:
                    blk.register_forward_hook(make_hook(f"encoder.blocks.{i}"))

    if is_main_process(rank):
        print("[验证] 执行dummy forward检查输出格式...")
        model.eval()
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224, device=device)
            dummy_pcd = torch.randn(1, 224, 224, 7, device=device)
            dummy_views = [{
                "img": dummy_img,
                "pcd": dummy_pcd,
                "lidar_depth_scale": torch.tensor([1.0], device=device),
                "data_norm_type": ["dinov2"],
                "confidence": torch.tensor(0.5, device=device),
            }]
            try:
                dummy_pred = model(dummy_views, use_lidar=True)
                if isinstance(dummy_pred, list) and len(dummy_pred) > 0:
                    print(f"  -> 模型输出keys: {list(dummy_pred[0].keys())}")
                    has_pose = 'cam_trans' in dummy_pred[0] and 'cam_quats' in dummy_pred[0]
                    has_depth = 'depth_along_ray' in dummy_pred[0]
                    has_ray = 'ray_directions' in dummy_pred[0]
                    print(f"  -> 支持位姿loss: {has_pose}, 深度loss: {has_depth}, 射线loss: {has_ray}")
                else:
                    print(f"  -> 警告: 模型输出格式异常: {type(dummy_pred)}")
            except Exception as e:
                print(f"  -> dummy forward失败（不影响训练，仅用于诊断）: {e}")
        model.train()

    return model


# ========================= 显存清理辅助函数（与第一个脚本完全一致） =========================

def _cleanup_batch_tensors(views, predictions, loss=None, loss_scaled=None):
    if views is not None:
        for view in views:
            if isinstance(view, dict):
                for key in list(view.keys()):
                    val = view[key]
                    if isinstance(val, torch.Tensor):
                        view[key] = val.cpu()
                        del view[key]
        del views

    if predictions is not None:
        for pred in predictions:
            if isinstance(pred, dict):
                for key in list(pred.keys()):
                    val = pred[key]
                    if isinstance(val, torch.Tensor):
                        pred[key] = val.cpu()
                        del pred[key]
        del predictions

    if loss is not None:
        del loss
    if loss_scaled is not None:
        del loss_scaled

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ========================= LoRA-only 保存/加载工具（与第一个脚本完全一致） =========================

def get_lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters()
            if 'lora_A' in k or 'lora_B' in k}


def save_checkpoint_lora(save_model, optimizer, scheduler, scaler, epoch, best_loss, path, is_main):
    if not is_main:
        return

    lora_state = get_lora_state_dict(save_model)
    checkpoint = {
        'epoch': epoch,
        'lora_state_dict': lora_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
        'best_loss': best_loss,
    }

    tmp_path = path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"  -> 保存 LoRA checkpoint ({len(lora_state)} 个参数): {path}")
    except Exception as e:
        print(f"  -> 保存失败 (磁盘满?): {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ========================= 训练循环（含 Modality Dropout，日志格式对齐第一个脚本） =========================

def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, epoch, args, rank, scheduler):
    model.train()
    # [对齐第一个脚本] DDP barrier
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    total_loss = 0.0
    num_batches = 0
    data_time = 0.0
    train_time = 0.0
    optimizer.zero_grad(set_to_none=True)

    lidar_dropout_count = 0
    lidar_total_count = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # [对齐第一个脚本] 初始化 LoRA 梯度监控变量
    last_grad_norm = 0.0
    lora_grad_count = 0
    lora_grad_norm = 0.0

    t_start = time.time()
    for batch_idx, views in enumerate(dataloader):
        t_data_end = time.time()
        data_time += (t_data_end - t_start)

        for view in views:
            for key, val in view.items():
                if torch.is_tensor(val):
                    view[key] = val.to(device, non_blocking=True)

        lidar_dropped = False
        if args.lidar_dropout_prob > 0.0 and random.random() < args.lidar_dropout_prob:
            lidar_dropped = True
            lidar_dropout_count += 1
            for view in views:
                if 'pcd' in view and torch.is_tensor(view['pcd']):
                    view['pcd'].zero_()
                if 'lidar_depth_scale' in view and torch.is_tensor(view['lidar_depth_scale']):
                    view['lidar_depth_scale'].fill_(1.0)
        lidar_total_count += 1

        with autocast(enabled=args.amp, dtype=torch.bfloat16):
            predictions = model(views, use_lidar=True)

            if batch_idx == 0 and is_main_process(rank):
                print(f"[诊断] Epoch{epoch} Batch0 模型输出keys: {list(predictions[0].keys()) if predictions else []}")
                print(f"[诊断] views[0] keys: {list(views[0].keys()) if views else []}")

            for i, pred in enumerate(predictions):
                for k, v in pred.items():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():
                        if is_main_process(rank):
                            print(f"[Epoch {epoch}] batch {batch_idx}: 预测输出 nan! "
                                  f"view={i}, key={k}, shape={tuple(v.shape)}")

            loss, metrics = criterion(predictions, views, seq_len=args.seq_len)

        if loss is None:
            if is_main_process(rank):
                valid_keys = list(predictions[0].keys()) if predictions else []
                drop_tag = "[LIDAR_DROP]" if lidar_dropped else ""
                print(f"[Epoch {epoch}] batch {batch_idx}: 无有效loss分量! {drop_tag} "
                      f"predictions_keys={valid_keys}, 跳过")
            _cleanup_batch_tensors(views, predictions, None, None)
            t_start = time.time()
            continue

        loss_val = loss.item() if torch.isfinite(loss) else float('nan')

        if not loss.requires_grad:
            if is_main_process(rank):
                print(f"警告: loss 无梯度, 跳过 batch {batch_idx}")
            _cleanup_batch_tensors(views, predictions, loss, None)
            t_start = time.time()
            continue

        if not torch.isfinite(loss):
            if is_main_process(rank):
                print(f"警告: batch {batch_idx} 出现非有限loss ({loss_val:.4f}), 跳过")
            _cleanup_batch_tensors(views, predictions, loss, None)
            t_start = time.time()
            continue

        loss_scaled = loss / args.accum_iter
        if args.amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        _cleanup_batch_tensors(views, predictions, loss, loss_scaled)

        total_loss += loss_val
        num_batches += 1

        is_accum_boundary = ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(dataloader))
        if is_accum_boundary:
            has_nan_grad = False
            nan_param_names = []
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    nan_param_names.append(name)

            if has_nan_grad:
                if is_main_process(rank):
                    print(f"[Epoch {epoch}] batch {batch_idx}: 检测到nan梯度，跳过step！")
                    for n in nan_param_names[:3]:
                        print(f"    {n}")
                optimizer.zero_grad(set_to_none=True)
                last_grad_norm = float('nan')
                torch.cuda.empty_cache()
                continue

            params_for_clip = list(filter(lambda p: p.requires_grad, model.parameters()))
            if args.amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=args.grad_clip)
                optimizer.step()

            # [对齐第一个脚本] 计算 LoRA 专属梯度范数
            lora_grad_norm = 0.0
            lora_grad_count = 0
            for name, p in model.named_parameters():
                if p.grad is not None and ('lora_A' in name or 'lora_B' in name):
                    lora_grad_norm += p.grad.norm().item() ** 2
                    lora_grad_count += 1
            if lora_grad_count > 0:
                lora_grad_norm = math.sqrt(lora_grad_norm)

            if math.isfinite(grad_norm):
                last_grad_norm = float(grad_norm)
            else:
                last_grad_norm = float('nan')

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            if is_main_process(rank) and batch_idx % (args.log_interval * 5) == 0:
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                peak = torch.cuda.max_memory_allocated(device) / 1024**3
                print(f"  -> 显存: 已分配 {allocated:.2f}GB | 预留 {reserved:.2f}GB | 峰值 {peak:.2f}GB")

        t_train_end = time.time()
        train_time += (t_train_end - t_data_end)

        # [对齐第一个脚本] 日志格式统一
        if is_main_process(rank) and batch_idx % args.log_interval == 0:
            lr_str = "/".join([f"{g['lr']:.2e}" for g in optimizer.param_groups])
            drop_tag = "[DROP]" if lidar_dropped else "[LID]"
            log_str = (f"[Epoch {epoch}] [{batch_idx}/{len(dataloader)}] {drop_tag} "
                       f"LR: {lr_str} Loss: {loss_val:.4f}")
            for key in ['rpe_trans', 'rpe_rot', 'depth', 'ray']:
                if key in metrics:
                    log_str += f" {key}: {metrics[key]:.4f}"
            if 'rpe_trans' not in metrics and batch_idx % (args.log_interval * 5) == 0:
                log_str += " | (无 RPE，检查 pose 输出)"
            if math.isfinite(last_grad_norm):
                log_str += f" | GradNorm: {last_grad_norm:.4f}"
            else:
                log_str += " | GradNorm: nan/inf"
            if lora_grad_count > 0:
                log_str += f" | LoRA_Grad: {lora_grad_norm:.4f}({lora_grad_count})"
            elif batch_idx % (args.log_interval * 5) == 0:
                log_str += " | LoRA_Grad: N/A"
            print(log_str)

        t_start = time.time()

    if is_main_process(rank) and lidar_total_count > 0:
        drop_rate = lidar_dropout_count / lidar_total_count
        print(f"[Epoch {epoch}] LiDAR Modality Dropout 统计: {lidar_dropout_count}/{lidar_total_count} "
              f"({drop_rate*100:.1f}%) batches 丢弃了 LiDAR")

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return total_loss / max(num_batches, 1)


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description="Train MapAnything LiDAR Fusion (Joint LoRA + Full LiDAR)")
    parser.add_argument("--seq_root", type=str, nargs='+',
                        default=["/add02/users/xuyh/seq1/", "/add02/users/xuyh/seq3/"],
                        help="训练数据路径，可指定多个序列 (如 --seq_root /path/to/seq1 /path/to/seq3)")
    parser.add_argument("--model_dir", type=str, default="/home/xuyh/mapanything/")
    parser.add_argument("--output_dir", type=str, default="/add02/users/xuyh/checkpoints/lidar/")
    parser.add_argument("--cache_dir", type=str, default="./cache/lidar_7ch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--lr", type=float, default=5e-5, help="基础学习率 (LoRA 默认使用此 lr)")
    parser.add_argument("--lora_lr", type=float, default=None, help="LoRA专属学习率，默认等于lr")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--amp", action="store_true", default=False,
                        help="启用AMP（默认禁用，DINOv2+LoRA在fp16下不稳定，建议fp32）")
    parser.add_argument("--resume", type=str, default="/add02/users/xuyh/checkpoints/lidar/checkpoints/best_full.pt")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--accum_iter", type=int, default=8,
                        help="梯度累积步数（默认8，等效batch=8但峰值显存按1算）")
    parser.add_argument("--use_compile", action="store_true", default=False,
                        help="启用 torch.compile (默认禁用，已知DDP+compile可能不稳定)")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lidar_lr_scale", type=float, default=2.0, help="LiDAR 模块学习率倍数 (相对于 lr)")
    parser.add_argument("--reset_optimizer", action="store_true", default=True,
                        help="resume时重置优化器状态（默认True，修复LoRA训练推荐）")
    parser.add_argument("--resume_optimizer", action="store_true", default=False,
                        help="resume时加载优化器状态（默认False，避免历史错误动量）")
    parser.add_argument("--lidar_dropout_prob", type=float, default=0.0,
                        help="LiDAR模态dropout概率 (默认0.2，即20%%的batch随机丢弃LiDAR输入)")
    args = parser.parse_args()

    rank, local_rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    if is_ddp:
        torch.distributed.barrier()
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if is_main_process(rank):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if is_main_process(rank):
        print("加载模型...")
    model = build_model(
        args.model_dir, device, rank,
        use_compile=False,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha
    )

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    if len(params_to_train) == 0:
        if is_main_process(rank):
            print("致命错误: 模型没有任何可训练参数。")
        cleanup_ddp(is_ddp)
        sys.exit(1)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )

    # ==================== [Multi-Seq] 数据集构建：支持单序列或多序列 ====================
    if is_main_process(rank):
        print(f"构建数据集... (seq_len={args.seq_len}, accum_iter={args.accum_iter})")

    if len(args.seq_root) == 1:
        dataset = Seq1LidarDataset(
            seq_root=args.seq_root[0], seq_len=args.seq_len, stride=args.stride,
            img_size=args.img_size, cache_dir=args.cache_dir, tolerance=args.tolerance
        )
    else:
        datasets = []
        for idx, root in enumerate(args.seq_root):
            cache_subdir = os.path.join(args.cache_dir, f"seq{idx}") if args.cache_dir else None
            ds = Seq1LidarDataset(
                seq_root=root, seq_len=args.seq_len, stride=args.stride,
                img_size=args.img_size, cache_dir=cache_subdir, tolerance=args.tolerance
            )
            datasets.append(ds)
        dataset = ConcatDataset(datasets)
        if is_main_process(rank):
            total_len = sum(len(d) for d in datasets)
            print(f"[MultiSeq] 合并 {len(datasets)} 个数据集，总样本数: {total_len}")
            for i, d in enumerate(datasets):
                print(f"  -> Seq{i}: {len(d)} 个样本 from {args.seq_root[i]}")

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if is_ddp else None

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    lidar_params = []
    lora_params = []
    other_params = []
    lidar_keywords = ('lidars_encoder', 'lidar_film', 'fusion_conv')

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        elif any(k in name for k in lidar_keywords):
            lidar_params.append(param)
        else:
            other_params.append(param)

    if is_main_process(rank):
        print(f"优化器分组: LiDAR {len(lidar_params)} 组, LoRA {len(lora_params)} 组, 其他 {len(other_params)} 组")
        if other_params:
            print("警告: 存在未分类的可训练参数，已归入 LoRA 组")
            lora_params.extend(other_params)

    # [对齐第一个脚本] LoRA 学习率逻辑：默认使用 args.lr
    lora_lr = args.lora_lr if args.lora_lr is not None else args.lr
    param_groups = [
        {'params': lidar_params, 'lr': args.lr * args.lidar_lr_scale, 'weight_decay': args.weight_decay},
        {'params': lora_params, 'lr': lora_lr, 'weight_decay': args.weight_decay},
    ]
    optimizer = AdamW(param_groups, betas=(0.9, 0.999))

    steps_per_epoch = len(dataloader) // args.accum_iter
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_steps, total_steps // 2)
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=args.lr * 0.01)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    if is_main_process(rank):
        print(f"训练总步数: {total_steps}, warmup: {warmup_steps}, cosine: {cosine_steps}")
        print(f"LoRA学习率: {lora_lr:.2e}, LiDAR学习率: {args.lr * args.lidar_lr_scale:.2e}")

    # [对齐第一个脚本] 使用与第一个脚本完全相同的损失权重
    criterion = MapAnythingLoss(w_depth=1.0, w_pose_trans=3.0, w_pose_rot=1.0, w_ray=0.5, w_pts3d_cam=1.0)
    if args.amp and is_main_process(rank):
        print("[警告] AMP已启用。如果训练不稳定，建议禁用AMP使用fp32。")
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 0; best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        if is_main_process(rank):
            print(f"恢复训练: {args.resume} (CPU 加载 + 清理)")

        ckpt = torch.load(args.resume, map_location='cpu')

        # [对齐第一个脚本] 优先检测 LoRA-only checkpoint
        if 'lora_state_dict' in ckpt:
            state_dict = ckpt['lora_state_dict']
            if is_main_process(rank):
                print(f"  -> 检测到 LoRA-only checkpoint ({len(state_dict)} 个参数)")
        else:
            state_dict = ckpt['model_state_dict']
            if is_main_process(rank):
                print("  -> 检测到完整模型权重 (旧版兼容)")

        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        if is_ddp:
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        # [对齐第一个脚本] resume 后打印 LoRA 范数诊断
        if is_main_process(rank):
            target_model = model.module if is_ddp else model
            lora_state_loaded = get_lora_state_dict(target_model)

            a_norms = [v.norm().item() for k, v in lora_state_loaded.items() if 'lora_A' in k]
            b_norms = [v.norm().item() for k, v in lora_state_loaded.items() if 'lora_B' in k]

            if a_norms:
                print(f"  -> LoRA_A 平均范数: {sum(a_norms)/len(a_norms):.4f} "
                      f"(初始化≈0.02，训练后应缓慢增长)")
                print(f"  -> LoRA_B 平均范数: {sum(b_norms)/len(b_norms):.4f} "
                      f"(初始≈0.0，训练后应>0)")

                sample_items = list(lora_state_loaded.items())[:6]
                for k, v in sample_items:
                    print(f"     {k}: {v.norm():.4f}")
            else:
                print("  -> 警告: 未检测到任何 LoRA 参数，请检查模块名匹配！")

            print(f"  -> 已恢复 epoch {start_epoch-1}, best_loss={best_loss:.4f} (模型权重已加载，优化器已重置)")

        del ckpt, state_dict
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_start = time.time()
        avg_loss = train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, epoch, args, rank, scheduler)
        epoch_time = time.time() - epoch_start

        if is_main_process(rank):
            print(f"Epoch {epoch} 完成 | 平均损失: {avg_loss:.4f} | 总耗时: {epoch_time:.1f}s")
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
                torch.cuda.empty_cache()
                save_model = model.module if is_ddp else model
                # [对齐第一个脚本] 同时保存 LoRA-only checkpoint
                save_checkpoint_lora(save_model, optimizer, scheduler, scaler, epoch, best_loss, ckpt_path, is_main_process(rank))

                # 额外保存完整模型（用于 LiDAR 训练连续性）
                cpu_state = {k: v.cpu() for k, v in save_model.state_dict().items()}
                full_path = ckpt_path.replace(".pt", "_full.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cpu_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_loss': best_loss,
                }, full_path)
                del cpu_state
                torch.cuda.empty_cache()
                print(f"  -> 保存完整模型: {full_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.cuda.empty_cache()
                save_model = model.module if is_ddp else model
                save_checkpoint_lora(save_model, optimizer, scheduler, scaler, epoch, best_loss, best_path, is_main_process(rank))

                # 额外保存完整 best 模型
                cpu_state = {k: v.cpu() for k, v in save_model.state_dict().items()}
                best_full_path = os.path.join(checkpoint_dir, "best_full.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cpu_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_loss': best_loss,
                }, best_full_path)
                del cpu_state
                torch.cuda.empty_cache()
                print(f"  -> 保存最佳模型 (loss={best_loss:.4f}): {best_path} (LoRA) / {best_full_path} (Full)")
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    if is_main_process(rank):
        print("训练完成!")
    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()