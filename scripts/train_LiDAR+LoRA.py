#!/usr/bin/env python3
# coding: utf-8
"""
MapAnything Training Script (LiDAR Warmup + Unified Training)
================================================================
  --lora  : 训练 LoRA 参数 (RGB Encoder / Info Sharing / Heads)
  --lidar : 训练 LiDAR 模块 (LiDAR Encoder / FiLM / Fusion)
  --lidar_warmup_epochs N : 前 N 个 epoch 冻结 LoRA，强制 LiDAR 主导

  冷启动阶段 (epoch <= warmup):
    - 冻结所有 LoRA 参数 (lora_A / lora_B)
    - fusion_module : LiDAR/RGB gated fusion (LiDAR warmup favors the LiDAR branch)
    - 可训练: lidars_encoder, fusion_module, pose_head*, dense_head*
    - Pose/Dense heads 中原本冻结的基参数也会解冻，让 head 适配 LiDAR 特征

  联合训练阶段 (epoch > warmup):
    - 解冻所有 LoRA 参数
    - fusion_module gate 恢复为中性/对称初始化
    - 重新构建优化器，启用完整的 param_groups
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import copy
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

# ========================= 数值稳定工具函数 =========================

def f_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    sign_x = torch.sign(x)
    abs_x = torch.abs(x) + eps
    return (sign_x * torch.log1p(abs_x)).to(x.dtype)


class RobustRegressionLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, scaling_c: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.c = scaling_c

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        with torch.no_grad():
            if valid_mask is not None and valid_mask.any():
                thresh = self.c * torch.median(abs_diff[valid_mask])
            else:
                thresh = self.c * torch.median(abs_diff) if abs_diff.numel() > 0 else torch.tensor(self.c)
            thresh = thresh.clamp_min(1e-6)
        quadratic = 0.5 * (diff ** 2) / thresh
        linear = abs_diff - 0.5 * thresh
        loss = torch.where(abs_diff <= thresh, quadratic, linear)
        if valid_mask is not None:
            return loss[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return loss.mean()


def so3_chordal_distance(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    trace = torch.clamp(trace, -3.0, 3.0)
    chordal_sq = torch.clamp(2.0 * (3.0 - trace), min=0.0)
    return torch.sqrt(chordal_sq + 1e-8)


def exclude_top_n_percent(loss_map: torch.Tensor, n_percent: float = 5.0, valid_mask: torch.Tensor = None) -> torch.Tensor:
    if valid_mask is not None and valid_mask.shape != loss_map.shape:
        if valid_mask.ndim == loss_map.ndim and valid_mask.shape[-1] != loss_map.shape[-1]:
            valid_mask = valid_mask.any(dim=-1, keepdim=True)
        if valid_mask.shape != loss_map.shape:
            try:
                valid_mask = valid_mask.expand_as(loss_map)
            except RuntimeError:
                valid_mask = None
    if n_percent <= 0:
        return loss_map.mean() if valid_mask is None else loss_map[valid_mask].mean()
    if valid_mask is not None:
        loss_vec = loss_map[valid_mask]
    else:
        loss_vec = loss_map.flatten()
    if loss_vec.numel() == 0:
        return torch.tensor(0.0, device=loss_map.device, dtype=loss_map.dtype)
    k = max(1, int(loss_vec.numel() * n_percent / 100.0))
    threshold = torch.topk(loss_vec, k, largest=True)[0][-1]
    keep_mask = loss_vec < threshold
    if keep_mask.any():
        return loss_vec[keep_mask].mean()
    return loss_vec.mean()


class ConfidenceLoss(nn.Module):
    def __init__(self, conf_alpha: float = 0.2):
        super().__init__()
        self.conf_alpha = conf_alpha

    def forward(self, confidence: torch.Tensor, loss_map: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.ones_like(loss_map, dtype=torch.bool)
        conf = confidence[valid_mask].flatten()
        loss_vals = loss_map[valid_mask].flatten().detach()
        if conf.numel() == 0:
            return torch.tensor(0.0, device=confidence.device)
        with torch.no_grad():
            loss_normalized = loss_vals / (loss_vals.mean() + 1e-8)
            loss_normalized = torch.clamp(loss_normalized, 0.0, 10.0)
            target_conf = torch.exp(-loss_normalized)
        conf_clamped = torch.clamp(conf, 1e-6, 1.0 - 1e-6)
        conf_loss = F.binary_cross_entropy(conf_clamped, target_conf.float())
        return self.conf_alpha * conf_loss


# ========================= LoRA =========================

class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, lora_alpha: int = 8):
        super().__init__()
        self.linear = linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, linear.out_features) * 0.01)
        std = 1.0 / math.sqrt(r)
        nn.init.normal_(self.lora_A, mean=0.0, std=std)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        lora = x.float() @ self.lora_A.float()
        lora = lora @ self.lora_B.float()
        lora = lora.to(out.dtype)
        return out + lora * self.scaling


def inject_lora_to_module(module: nn.Module, r: int = 8, lora_alpha: int = 8):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(child, r, lora_alpha))
        else:
            inject_lora_to_module(child, r, lora_alpha)


# ========================= EMA =========================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data


# ========================= 通用工具 =========================

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
        print(f"  -> Gradient Checkpointing 已启用: {wrapped_names}")


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


# ========================= 全局深度归一化 =========================
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
                 img_exts: tuple = None, use_lidar: bool = True):
        super().__init__()
        self.seq_root = seq_root
        self.seq_len = seq_len
        self.stride = stride
        self.img_size = img_size
        self.pcd_gen_size = pcd_gen_size
        self.tolerance = tolerance
        self.cache_dir = cache_dir
        self.use_lidar = use_lidar
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
            print(f"[Dataset] TUM 真值位姿: {len(self.gt_poses_list)} 帧")

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
                self.part_depths.append({'files': valid_files, 'timestamps': np.array(depth_ts, dtype=np.int64)})
            else:
                self.part_depths.append({'files': [], 'timestamps': np.array([], dtype=np.int64)})

        intrinsics_file = os.path.join(seq_root, "color_camera_intrinsics.txt")
        self.intrinsics = self._load_intrinsics(intrinsics_file) if os.path.exists(intrinsics_file) else np.eye(3, dtype=np.float32)

        self.all_views_meta = []
        timestamp_diffs = []
        for pidx, part in enumerate(self.part_folders):
            rgb_dir = os.path.join(seq_root, part, "rgb")
            if not os.path.exists(rgb_dir):
                continue
            img_paths = []
            for fname in sorted(os.listdir(rgb_dir)):
                if fname.lower().endswith(self.img_exts):
                    img_paths.append(os.path.join(rgb_dir, fname))
            for ipath in img_paths:
                basename = os.path.basename(ipath)
                match = re.search(r'color_(\d+)', basename)
                if not match:
                    match = re.search(r'(\d+)', basename)
                if not match:
                    continue
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
                else:
                    timestamp_diffs.append(diff)

        if not self.all_views_meta and is_main_process(int(os.environ.get('RANK', 0))):
            raise ValueError("没有任何图像通过时间戳匹配！")

        self.part_pcds = []
        if use_lidar:
            for part in self.part_folders:
                lidar_dir = os.path.join(seq_root, part, "lidar")
                pcd_files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd")))
                pcd_ts = []
                for pf in pcd_files:
                    m = re.search(r'(\d+)', os.path.basename(pf))
                    ts = int(m.group(1)) if m else int(os.path.getmtime(pf) * 1e9)
                    pcd_ts.append(ts)
                self.part_pcds.append({'files': pcd_files, 'timestamps': np.array(pcd_ts, dtype=np.int64)})
        else:
            for _ in self.part_folders:
                self.part_pcds.append({'files': [], 'timestamps': np.array([], dtype=np.int64)})

        if cache_dir and use_lidar:
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
            ipath, img_ts_ns, pidx, gt_idx = self.all_views_meta[meta_idx]
            img_ts_sec = img_ts_ns / 1e9

            img = Image.open(ipath).convert('RGB')
            img_np_color = np.array(img)
            if not np.isfinite(img_np_color).all():
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

            if self.use_lidar:
                pcd_7ch, z_d = self._get_pcd_feature(img_ts_sec, pidx)
                pcd_tensor = torch.from_numpy(pcd_7ch)
                if pcd_tensor.shape[1] != self.img_size or pcd_tensor.shape[2] != self.img_size:
                    pcd_tensor = F.interpolate(
                        pcd_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                pcd_tensor = pcd_tensor.permute(1, 2, 0).unsqueeze(0)
            else:
                pcd_tensor = torch.zeros((1, self.img_size, self.img_size, 7), dtype=torch.float32)
                z_d = 1.0

            gt_pose = torch.from_numpy(self.gt_poses_list[gt_idx])

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


# ========================= 损失函数 =========================

class MapAnythingLoss(nn.Module):
    def __init__(self, 
                 w_depth: float = 0.1,
                 w_pose_trans: float = 0.1,
                 w_pose_rot: float = 0.1,
                 w_ray: float = 0.1,
                 w_pts3d_cam: float = 0.1,
                 w_world_pts: float = 1.0,
                 w_confidence: float = 0.2,
                 w_scale: float = 0.1,
                 robust_alpha: float = 0.5,
                 robust_c: float = 0.05,
                 top_n_percent: float = 5.0,
                 use_log_depth: bool = True,
                 use_chordal_rot: bool = True):
        super().__init__()
        self.w_depth = w_depth
        self.w_pose_trans = w_pose_trans
        self.w_pose_rot = w_pose_rot
        self.w_ray = w_ray
        self.w_pts3d_cam = w_pts3d_cam
        self.w_world_pts = w_world_pts
        self.w_confidence = w_confidence
        self.w_scale = w_scale
        self.use_log_depth = use_log_depth
        self.use_chordal_rot = use_chordal_rot
        self.top_n_percent = top_n_percent
        self.robust_loss = RobustRegressionLoss(alpha=robust_alpha, scaling_c=robust_c)
        self.conf_loss_fn = ConfidenceLoss(conf_alpha=w_confidence)

    @staticmethod
    def _quat_to_rotmat(quats):
        norm = torch.norm(quats, dim=-1, keepdim=True)
        quats = quats / (norm + 1e-8)
        quats = torch.where(quats[:, 3:4] < 0, -quats, quats)
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
        T[:, :3, :3] = R
        T[:, :3, 3] = trans
        return T

    @staticmethod
    def _inv_transform(T):
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        T_inv = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
        R_T = R.transpose(-2, -1)
        T_inv[:, :3, :3] = R_T
        T_inv[:, :3, 3] = -(R_T @ t.unsqueeze(-1)).squeeze(-1)
        return T_inv

    def _compute_world_frame_points_loss(self, pred_pts3d_world, gt_pts3d_world, valid_mask=None):
        if valid_mask is None or not valid_mask.any():
            if pred_pts3d_world is None or gt_pts3d_world is None:
                return torch.tensor(0.0, device=pred_pts3d_world.device if pred_pts3d_world is not None else 'cuda')
            valid_mask = torch.isfinite(pred_pts3d_world).all(dim=-1) & torch.isfinite(gt_pts3d_world).all(dim=-1)
        pred_log = f_log(pred_pts3d_world)
        gt_log = f_log(gt_pts3d_world)
        loss_map = torch.abs(pred_log - gt_log).mean(dim=-1, keepdim=True)
        return exclude_top_n_percent(loss_map, self.top_n_percent, valid_mask)

    def forward(self, predictions: List[Dict], views: List[Dict], seq_len: int = 2):
        device = predictions[0]['pts3d'].device
        loss_components = []
        metrics = {}
        N = len(predictions)

        for i, (pred, view) in enumerate(zip(predictions, views)):
            H, W = pred['pts3d'].shape[1:3] if 'pts3d' in pred else (448, 448)
            B = 1

            if 'depth_along_ray' in pred and view.get('gt_depth') is not None:
                pred_d = pred['depth_along_ray']
                gt_d = view['gt_depth'].to(device)
                pred_d = pred_d.permute(0, 3, 1, 2)
                if pred_d.shape[-2:] != gt_d.shape[-2:]:
                    gt_d = F.interpolate(gt_d, size=pred_d.shape[-2:], mode='bilinear', align_corners=False)
                valid = (gt_d > 1e-4) & torch.isfinite(pred_d)
                if valid.any():
                    if self.use_log_depth:
                        pred_d_log = f_log(pred_d)
                        gt_d_log = f_log(gt_d)
                        loss_map = torch.abs(pred_d_log - gt_d_log)
                    else:
                        loss_map = torch.abs(pred_d - gt_d)
                    loss_d = exclude_top_n_percent(loss_map, self.top_n_percent, valid)
                    loss_components.append(self.w_depth * loss_d)
                    metrics[f'depth_{i}'] = loss_d.item()

            if 'ray_directions' in pred and view.get('gt_intrinsics') is not None:
                K = view['gt_intrinsics'].to(device)
                if K.dim() == 2:
                    K = K.unsqueeze(0)
                pred_ray = pred['ray_directions']
                B, H, W, _ = pred_ray.shape
                u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
                pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).float()
                with torch.cuda.amp.autocast(enabled=False):
                    K_f32 = K.float()
                    pixels_f32 = pixels.reshape(-1, 3).float()
                    K_inv = torch.linalg.inv(K_f32)
                    rays_cam = (K_inv @ pixels_f32.T.unsqueeze(0).expand(B, -1, -1))
                rays_cam = rays_cam.to(pred_ray.dtype)
                rays_cam = rays_cam.permute(0, 2, 1).reshape(B, H, W, 3)
                rays_cam = rays_cam / (torch.norm(rays_cam, dim=-1, keepdim=True) + 1e-8)
                cos_sim = F.cosine_similarity(pred_ray, rays_cam, dim=-1)
                loss_ray = (1 - cos_sim).mean()
                loss_components.append(self.w_ray * loss_ray)
                metrics[f'ray_{i}'] = loss_ray.item()

            gt_pts3d_cam = None
            if 'pts3d_cam' in pred and view.get('gt_depth') is not None and view.get('gt_intrinsics') is not None:
                gt_d = view['gt_depth'].to(device)
                if gt_d.dim() == 2:
                    gt_d = gt_d.unsqueeze(0).unsqueeze(0)
                elif gt_d.dim() == 3:
                    gt_d = gt_d.unsqueeze(0)
                K = view['gt_intrinsics'].to(device)
                if K.dim() == 2:
                    K = K.unsqueeze(0)
                B = K.shape[0]
                H, W = pred['pts3d_cam'].shape[1:3]
                if gt_d.shape[-2:] != (H, W):
                    gt_d = F.interpolate(gt_d, size=(H, W), mode='bilinear', align_corners=False)
                u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
                pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).float()
                with torch.cuda.amp.autocast(enabled=False):
                    K_f32 = K.float()
                    pixels_f32 = pixels.reshape(-1, 3).float()
                    K_inv = torch.linalg.inv(K_f32)
                    rays = (K_inv @ pixels_f32.T.unsqueeze(0).expand(B, -1, -1))
                rays = rays.to(pred['pts3d_cam'].dtype)
                rays = rays.permute(0, 2, 1).reshape(B, H, W, 3)
                rays = rays / (torch.norm(rays, dim=-1, keepdim=True) + 1e-8)
                gt_pts3d_cam = rays * gt_d.permute(0, 2, 3, 1)
                pred_pts3d_cam = pred['pts3d_cam']
                valid = (gt_d.permute(0, 2, 3, 1) > 1e-4).expand_as(pred_pts3d_cam)
                if valid.any():
                    pred_log = f_log(pred_pts3d_cam)
                    gt_log = f_log(gt_pts3d_cam)
                    loss_map = torch.abs(pred_log - gt_log).mean(dim=-1, keepdim=True)
                    loss_pc = exclude_top_n_percent(loss_map, self.top_n_percent, valid)
                    loss_components.append(self.w_pts3d_cam * loss_pc)
                    metrics[f'pts3d_cam_{i}'] = loss_pc.item()

            if 'pts3d' in pred and view.get('gt_pose') is not None and view.get('gt_depth') is not None and gt_pts3d_cam is not None:
                pred_world = pred['pts3d']
                gt_pose = view['gt_pose'].to(device)
                if gt_pose.dim() == 2:
                    gt_pose = gt_pose.unsqueeze(0)
                gt_pts3d_world = torch.matmul(gt_pts3d_cam.reshape(B, -1, 3),
                                               gt_pose[:, :3, :3].transpose(1, 2)) + gt_pose[:, :3, 3:4].transpose(1, 2)
                gt_pts3d_world = gt_pts3d_world.reshape(B, H, W, 3)
                valid_world = torch.isfinite(pred_world).all(dim=-1) & torch.isfinite(gt_pts3d_world).all(dim=-1)
                if valid_world.any():
                    loss_world = self._compute_world_frame_points_loss(pred_world, gt_pts3d_world, valid_world)
                    loss_components.append(self.w_world_pts * loss_world)
                    metrics[f'world_pts_{i}'] = loss_world.item()

            if 'confidence' in pred and ('depth_along_ray' in pred or 'pts3d_cam' in pred):
                if 'depth_along_ray' in pred and view.get('gt_depth') is not None:
                    pred_d = pred['depth_along_ray'].permute(0, 3, 1, 2)
                    gt_d = view['gt_depth'].to(device)
                    if pred_d.shape[-2:] != gt_d.shape[-2:]:
                        gt_d = F.interpolate(gt_d, size=pred_d.shape[-2:], mode='bilinear', align_corners=False)
                    valid = (gt_d > 1e-4) & torch.isfinite(pred_d)
                    if valid.any():
                        with torch.no_grad():
                            err_map = torch.abs(f_log(pred_d) - f_log(gt_d))
                        conf = pred['confidence'] if 'confidence' in pred else torch.ones_like(err_map)
                        if conf.shape != err_map.shape:
                            conf = F.interpolate(conf, size=err_map.shape[-2:], mode='bilinear', align_corners=False)
                        loss_conf = self.conf_loss_fn(conf, err_map, valid)
                        if loss_conf > 0:
                            loss_components.append(loss_conf)
                            metrics[f'conf_{i}'] = loss_conf.item()

        if seq_len >= 2 and N >= seq_len:
            batch_size = N // seq_len
            for b in range(batch_size):
                start = b * seq_len
                for k in range(seq_len - 1):
                    idx1 = start + k
                    idx2 = start + k + 1
                    pred1, pred2 = predictions[idx1], predictions[idx2]
                    view1, view2 = views[idx1], views[idx2]
                    required_keys = ['cam_trans', 'cam_quats']
                    if not all(k in pred1 and k in pred2 for k in required_keys):
                        continue
                    if view1.get('gt_pose') is None or view2.get('gt_pose') is None:
                        continue
                    gt_pose1 = view1['gt_pose'].to(device)
                    gt_pose2 = view2['gt_pose'].to(device)
                    if gt_pose1.dim() == 2:
                        gt_pose1 = gt_pose1.unsqueeze(0)
                    if gt_pose2.dim() == 2:
                        gt_pose2 = gt_pose2.unsqueeze(0)
                    T_pred1 = self._build_transform(pred1['cam_trans'], pred1['cam_quats'])
                    T_pred2 = self._build_transform(pred2['cam_trans'], pred2['cam_quats'])
                    T_rel_gt = self._inv_transform(gt_pose1) @ gt_pose2
                    T_rel_pred = self._inv_transform(T_pred1) @ T_pred2
                    if not (torch.isfinite(T_rel_pred).all() and torch.isfinite(T_rel_gt).all()):
                        continue
                    t_norm_pred = torch.norm(T_rel_pred[:, :3, 3], dim=-1).mean()
                    t_norm_gt = torch.norm(T_rel_gt[:, :3, 3], dim=-1).mean()
                    if not torch.isfinite(t_norm_pred) or t_norm_gt < 1e-12:
                        continue
                    trans_diff = T_rel_pred[:, :3, 3] - T_rel_gt[:, :3, 3]
                    trans_diff_clipped = torch.clamp(trans_diff, -1.0, 1.0)
                    loss_t = F.smooth_l1_loss(trans_diff_clipped, torch.zeros_like(trans_diff_clipped), beta=0.1)
                    if self.use_chordal_rot:
                        loss_r = so3_chordal_distance(T_rel_pred[:, :3, :3], T_rel_gt[:, :3, :3]).mean()
                    else:
                        R_diff = torch.bmm(T_rel_pred[:, :3, :3].transpose(1, 2), T_rel_gt[:, :3, :3])
                        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
                        trace = torch.clamp(trace, -1.0, 3.0)
                        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
                        loss_r = torch.acos(cos_angle).mean()
                    if torch.isfinite(loss_t) and torch.isfinite(loss_r):
                        loss_components.append(self.w_pose_trans * loss_t + self.w_pose_rot * loss_r)
                        metrics[f'rpe_trans_b{b}_k{k}'] = loss_t.item()
                    metrics[f'rpe_rot_b{b}_k{k}'] = loss_r.item()

        rpe_trans_vals = [v for k, v in metrics.items() if k.startswith('rpe_trans_b')]
        rpe_rot_vals = [v for k, v in metrics.items() if k.startswith('rpe_rot_b')]
        if rpe_trans_vals:
            metrics['rpe_trans'] = sum(rpe_trans_vals) / len(rpe_trans_vals)
            metrics['rpe_rot'] = sum(rpe_rot_vals) / len(rpe_rot_vals)
        depth_vals = [v for k, v in metrics.items() if k.startswith('depth_')]
        if depth_vals:
            metrics['depth'] = sum(depth_vals) / len(depth_vals)
        ray_vals = [v for k, v in metrics.items() if k.startswith('ray_')]
        if ray_vals:
            metrics['ray'] = sum(ray_vals) / len(ray_vals)
        world_vals = [v for k, v in metrics.items() if k.startswith('world_pts_')]
        if world_vals:
            metrics['world_pts'] = sum(world_vals) / len(world_vals)

        if loss_components:
            total_loss = sum(loss_components)
        else:
            metrics['total_loss'] = 0.0
            metrics['no_valid_loss'] = True
            return None, metrics

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics


# ========================= 模型构建 =========================

def build_model(model_dir: str, device: str, rank: int, use_compile: bool = True,
                lora_r: int = 8, lora_alpha: int = 8,
                use_lora: bool = True, use_lidar: bool = True,
                lidar_warmup_epochs: int = 0,
                lidar_warmup_gate_alpha: float = 0.70):
    """
    构建模型。LiDAR 编码器默认从随机初始化开始，并全参数参与训练。
    """
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    with open(config_path, 'r') as f:
        config = json.load(f)
    encoder_config = config.get("encoder_config", {}).copy()
    encoder_config.pop("pretrained", None); encoder_config.pop("weights", None)
    encoder_config["uses_torch_hub"] = False
    geometric_input_config = copy.deepcopy(config.get("geometric_input_config", {}))
    lidar_encoder_config = geometric_input_config.get("lidars_encoder_config", {})
    for key in ("pretrained", "weights", "pretrained_checkpoint_path", "checkpoint_path", "custom_ckpt_path", "load_pretrained_weights"):
        lidar_encoder_config.pop(key, None)
    lidar_encoder_config["pretrained"] = False
    lidar_encoder_config["weights"] = None
    lidar_encoder_config["uses_torch_hub"] = False
    geometric_input_config["lidars_encoder_config"] = lidar_encoder_config
    model = MapAnything(
        name=config.get("name", "mapanything"),
        encoder_config=encoder_config,
        info_sharing_config=config.get("info_sharing_config", {}),
        pred_head_config=config.get("pred_head_config", {}),
        geometric_input_config=geometric_input_config,
        pretrained_checkpoint_path=None,
        torch_hub_force_reload=False,
        info_sharing_mlp_layer_str="swiglufused"
    )
    if os.path.exists(weights_path):
        if is_main_process(rank):
            print(f"加载预训练权重: {weights_path}")
        state_dict = load_file(weights_path)
        if use_lidar:
            lidar_encoder_keys = [
                key for key in state_dict.keys()
                if key.replace('_orig_mod.', '').startswith('lidars_encoder.')
            ]
            if lidar_encoder_keys:
                for key in lidar_encoder_keys:
                    state_dict.pop(key, None)
                if is_main_process(rank):
                    print(f"  -> 已跳过 {len(lidar_encoder_keys)} 个 LiDAR encoder 权重键，保持随机初始化")
        model.load_state_dict(state_dict, strict=False)
    else:
        if is_main_process(rank):
            print("警告: 未找到预训练权重")

    if use_lidar:
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
                    if lidar_warmup_epochs > 0:
                        gate_mlp[-1].bias.fill_(_gate_bias_from_alpha(lidar_warmup_gate_alpha))
                        if is_main_process(rank):
                            print(f"  -> fusion_module warmup init: LiDAR-biased gate alpha={lidar_warmup_gate_alpha:.2f} (warmup={lidar_warmup_epochs} epochs)")
                    else:
                        if is_main_process(rank):
                            print("  -> fusion_module standard init: neutral gate")
                refine = getattr(fusion_module, "refine", None)
                if refine is not None and hasattr(refine, "weight"):
                    refine.weight.zero_()
        elif is_main_process(rank):
            print("  -> warning: fusion_module not found; skipping fusion initialization")
        if hasattr(model, 'lidars_encoder') and model.lidars_encoder is not None and is_main_process(rank):
            print("  -> LiDAR encoder initialized for training")
    else:
        if is_main_process(rank):
            print("  -> LiDAR disabled")

    if use_lora:
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
    else:
        if is_main_process(rank):
            print("  -> LoRA 已禁用")

    # 初始状态：全部冻结
    for param in model.parameters():
        param.requires_grad = False

    # LiDAR 参数始终可训练（如果启用）
    trainable_names = []
    if use_lidar:
        lidar_keywords = ('lidars_encoder', 'lidar_film', 'fusion_module', 'fusion_conv')
        for name, param in model.named_parameters():
            clean_name = name.replace('_orig_mod.', '')
            if clean_name.startswith(lidar_keywords):
                param.requires_grad = True
                trainable_names.append(name)

    # LoRA 参数在 warmup 阶段保持冻结，之后解冻
    if use_lora and lidar_warmup_epochs == 0:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
                trainable_names.append(name)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable == 0:
        raise RuntimeError("没有任何可训练参数，请至少启用 --lora 或 --lidar。")

    if is_main_process(rank):
        print(f"  -> 总参数量: {total/1e6:.2f}M, 可训练: {trainable/1e6:.2f}M ({len(trainable_names)} modules)")
        if use_lidar:
            print(f"     LiDAR+Fusion: {sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and any(k in n for k in ('lidars_encoder', 'lidar_film', 'fusion_module')))/1e6:.2f}M")
        if use_lora and lidar_warmup_epochs == 0:
            print(f"     LoRA: {sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and ('lora_A' in n or 'lora_B' in n))/1e6:.2f}M")
        if lidar_warmup_epochs > 0:
            print(f"     [Warmup 模式] LoRA 参数已冻结，前 {lidar_warmup_epochs} 个 epoch 只训练 LiDAR")

    enable_gradient_checkpointing_safe(model, rank)
    model = model.to(device)

    if is_main_process(rank):
        print("[验证] 执行 dummy forward...")
        model.eval()
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224, device=device)
            dummy_pcd = torch.randn(1, 224, 224, 7, device=device) if use_lidar else torch.zeros(1, 224, 224, 7, device=device)
            dummy_views = [{
                "img": dummy_img,
                "pcd": dummy_pcd,
                "lidar_depth_scale": torch.tensor([1.0], device=device),
                "data_norm_type": ["dinov2"],
                "confidence": torch.tensor(0.5, device=device),
            }]
            try:
                dummy_pred = model(dummy_views, use_lidar=use_lidar)
                if isinstance(dummy_pred, list) and len(dummy_pred) > 0:
                    print(f"  -> 输出keys: {list(dummy_pred[0].keys())}")
            except Exception as e:
                print(f"  -> dummy forward 失败: {e}")
        model.train()

    return model


# ========================= 显存清理 =========================

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


# ========================= 保存与恢复 =========================

def save_checkpoint(save_model, optimizer, scheduler, scaler, epoch, best_loss, path, is_main, ema=None):
    if not is_main:
        return
    full_state = {k: v.detach().cpu() for k, v in save_model.state_dict().items()}
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': full_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
        'best_loss': best_loss,
    }
    if ema is not None:
        checkpoint['ema_shadow'] = {k: v.clone() for k, v in ema.shadow.items()}
    
    tmp_path = path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"  -> 保存完整模型 ({len(full_state)} keys) | epoch={epoch} | {path}")
    except Exception as e:
        print(f"  -> 保存失败: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def smart_resume(model, optimizer, scheduler, scaler, ema, ckpt_path, rank, device):
    """从 checkpoint 恢复完整模型权重和训练状态"""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return 0, float('inf')

    if is_main_process(rank):
        print(f"[Resume] 恢复: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if is_main_process(rank):
            print(f"  -> 模型权重已恢复")
    else:
        if is_main_process(rank):
            print("  -> 警告: checkpoint 中没有 model_state_dict")

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if is_main_process(rank):
                print(f"  -> 优化器状态已恢复")
        except Exception as e:
            if is_main_process(rank):
                print(f"  -> 优化器恢复失败（结构不匹配）: {e}")
                print(f"     将使用新优化器，仅保留模型权重")

    if scheduler is not None and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if is_main_process(rank):
                print(f"  -> 调度器已恢复")
        except Exception as e:
            if is_main_process(rank):
                print(f"  -> 调度器恢复失败: {e}")

    if scaler is not None and 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
        try:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        except Exception as e:
            if is_main_process(rank):
                print(f"  -> Scaler 恢复失败: {e}")

    if ema is not None:
        ema._register(model)
        if is_main_process(rank):
            print(f"  -> EMA 重新初始化")

    epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))

    if is_main_process(rank):
        print(f"  -> 恢复 epoch {epoch}, best_loss={best_loss:.4f}")

    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return epoch, best_loss


# ========================= Warmup 工具函数 =========================

# 全局状态：保存 fusion_conv RGB 分支的 hook handle
_fusion_rgb_hook_handle = None


def _freeze_fusion_rgb_branch_hook(enc_dim):
    """返回一个 backward hook，用于冻结 fusion_conv 的 RGB 分支梯度"""
    def _hook(grad):
        if grad is not None:
            grad = grad.clone()
            grad[:, :enc_dim, 0, 0].zero_()
        return grad
    return _hook


def _gate_bias_from_alpha(alpha: float) -> float:
    """Convert a desired sigmoid gate value into the corresponding bias."""
    alpha = float(max(1e-4, min(1.0 - 1e-4, alpha)))
    return math.log(alpha / (1.0 - alpha))


def set_lidar_warmup_phase(model, enable_warmup, args, rank):
    global _fusion_rgb_hook_handle
    target_model = model.module if hasattr(model, 'module') else model

    unfrozen_count = 0
    frozen_count = 0

    for name, param in target_model.named_parameters():
        clean_name = name.replace('_orig_mod.', '')
        is_lora = 'lora_A' in clean_name or 'lora_B' in clean_name
        is_lidar = any(k in clean_name for k in ('lidars_encoder', 'lidar_film', 'fusion_module', 'fusion_conv'))
        is_head_base = any(h in clean_name for h in ('pose_head', 'dense_head', 'scale_head')) and not is_lora
        is_shared = any(k in clean_name for k in ('shared_linear', 'shared_decoder', 'output_proj'))

        if enable_warmup:
            # Warmup should teach the LiDAR branch and the fusion block together,
            # otherwise the fusion layer never learns to surface LiDAR evidence.
            if is_lidar or is_head_base or is_shared:
                param.requires_grad = True
                unfrozen_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        else:
            if is_lidar or is_lora:
                param.requires_grad = True
                unfrozen_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1

    if _fusion_rgb_hook_handle is not None:
        _fusion_rgb_hook_handle.remove()
        _fusion_rgb_hook_handle = None

    if is_main_process(rank):
        phase = "LiDAR-Warmup-V2(shared)" if enable_warmup else "Joint-Training-V2"
        print(f"\n{'='*60}")
        print(f"[phase switch] {phase}")
        print(f"  unfrozen: {unfrozen_count} | frozen: {frozen_count}")

    return unfrozen_count, frozen_count


def build_optimizer_for_phase(model, args, rank, phase='warmup'):
    """
    根据当前阶段构建优化器。
    phase='warmup': LiDAR + Head 基参数 + Shared 投影层
    phase='joint':  LoRA（所有线性层的 A/B）+ LiDAR
    """
    target_model = model.module if hasattr(model, 'module') else model
    
    lora_A_params_encoder = []
    lora_B_params_encoder = []
    lora_A_params_head = []
    lora_B_params_head = []
    lidar_params = []
    fusion_params = []
    head_base_params = []
    shared_params = []
    
    for name, param in target_model.named_parameters():
        if not param.requires_grad:
            continue
        
        clean_name = name.replace('_orig_mod.', '')
        is_encoder = 'encoder' in clean_name
        is_B = 'lora_B' in clean_name
        is_fusion = any(k in clean_name for k in ('fusion_module', 'fusion_conv'))
        is_lidar = any(k in clean_name for k in ('lidars_encoder', 'lidar_film'))
        is_head_base = any(h in clean_name for h in ('pose_head', 'dense_head', 'scale_head')) and 'lora' not in clean_name
        is_shared = any(k in clean_name for k in ('shared_linear', 'shared_decoder', 'output_proj'))
        
        # LoRA 参数优先级最高（无论它们在哪个模块里）
        if is_fusion:
            fusion_params.append(param)
        elif is_lidar:
            lidar_params.append(param)
        elif phase == 'joint' and 'lora_B' in clean_name:
            if is_encoder:
                lora_B_params_encoder.append(param)
            else:
                lora_B_params_head.append(param)
        elif phase == 'joint' and 'lora_A' in clean_name:
            if is_encoder:
                lora_A_params_encoder.append(param)
            else:
                lora_A_params_head.append(param)
        elif is_shared:
            shared_params.append(param)
        elif is_head_base:
            head_base_params.append(param)
    
    param_groups = []
    
    if phase == 'warmup':
        warmup_lr_scale = args.lidar_warmup_lr_scale
        if lidar_params:
            param_groups.append({
                'params': lidar_params,
                'lr': args.lr * max(args.lidar_lr_scale, 0.5) * warmup_lr_scale,
                'weight_decay': args.weight_decay,
                'name': 'lidar_warmup'
            })
        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': args.lr * max(args.fusion_lr_scale, 0.2) * warmup_lr_scale,
                'weight_decay': args.weight_decay,
                'name': 'fusion_warmup'
            })
        if shared_params:
            param_groups.append({
                'params': shared_params,
                'lr': args.lr * 0.5 * warmup_lr_scale,
                'weight_decay': args.weight_decay,
                'name': 'shared_proj'
            })
        if head_base_params:
            param_groups.append({
                'params': head_base_params,
                'lr': args.lr * warmup_lr_scale,
                'weight_decay': args.weight_decay,
                'name': 'head_base'
            })
    else:
        # 联合训练：只训 LoRA（所有线性层的 A/B）+ LiDAR
        # Head 和 Shared 的基参数冻结，只通过 LoRA 微调
        if lora_A_params_head:
            param_groups.append({'params': lora_A_params_head, 'lr': args.lr, 'weight_decay': args.weight_decay, 'name': 'head_lora_A'})
        if lora_B_params_head:
            param_groups.append({'params': lora_B_params_head, 'lr': args.lr * args.lr_b_multiplier, 'weight_decay': 0.0, 'name': 'head_lora_B'})
        if lora_A_params_encoder:
            param_groups.append({'params': lora_A_params_encoder, 'lr': args.lr * args.encoder_lr_ratio, 'weight_decay': args.weight_decay, 'name': 'enc_lora_A'})
        if lora_B_params_encoder:
            param_groups.append({'params': lora_B_params_encoder, 'lr': args.lr * args.lr_b_multiplier * args.encoder_lr_ratio, 'weight_decay': 0.0, 'name': 'enc_lora_B'})
        if lidar_params:
            param_groups.append({'params': lidar_params, 'lr': args.lr * args.lidar_lr_scale, 'weight_decay': args.weight_decay, 'name': 'lidar_full'})
        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': args.lr * args.fusion_lr_scale,
                'weight_decay': args.weight_decay,
                'name': 'fusion'
            })
    
    if not param_groups:
        raise RuntimeError(f"无可训练参数！phase={phase}")
    
    optimizer = AdamW(param_groups, betas=(0.9, 0.999))
    
    if is_main_process(rank):
        print(f"[优化器V2] phase={phase}, groups={len(param_groups)}")
        for g in param_groups:
            print(f"  {g['name']}: lr={g['lr']:.2e}, params={sum(p.numel() for p in g['params'])/1e6:.2f}M")
    
    return optimizer


def reinitialize_fusion_module_for_joint_training(model, rank, smooth_alpha=0.35):
    target_model = model.module if hasattr(model, 'module') else model
    fusion_module = getattr(target_model, "fusion_module", None)
    if fusion_module is None:
        if is_main_process(rank):
            print("  -> warning: fusion_module not found; skipping reinitialization")
        return

    with torch.no_grad():
        gate_mlp = getattr(fusion_module, "gate_mlp", None)
        if gate_mlp is not None and len(gate_mlp) >= 3 and hasattr(gate_mlp[-1], "weight"):
            gate_mlp[-1].weight.zero_()
            gate_mlp[-1].bias.fill_(_gate_bias_from_alpha(smooth_alpha))
        refine = getattr(fusion_module, "refine", None)
        if refine is not None and hasattr(refine, "weight"):
            refine.weight.zero_()
        fusion_conv = getattr(target_model, "fusion_conv", None)
        if fusion_conv is not None and hasattr(fusion_conv, "bias"):
            fusion_conv.bias.zero_()

    if is_main_process(rank):
        print(f"  -> fusion_module reinitialized to a moderate gate (alpha={smooth_alpha:.2f})")

def build_scheduler(optimizer, args, steps_per_epoch):
    """构建学习率调度器"""
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
    return scheduler, warmup_steps


# ========================= 训练循环 =========================

def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, epoch, args, rank, scheduler, ema=None, phase='joint'):
    model.train()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)

    lidar_dropout_count = 0
    lidar_total_count = 0
    b_first_norm = None
    nan_grad_batches = 0

    # ===================== 梯度监控系统 =====================
    CHECK_GRAD_ONLY = os.environ.get('CHECK_GRAD', '0') == '1'
    MAX_CHECK_BATCHES = 30
    check_grad_accum_count = 0

    from collections import defaultdict
    grad_history = defaultdict(list)
    grad_window = defaultdict(list)
    accum_boundary_count = 0

    def _collect_grad_norm(group_params):
        total_norm = 0.0
        num_params = 0
        for p in group_params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1
        if num_params == 0:
            return 0.0
        return total_norm ** 0.5

    def _print_grad_window(window, epoch, accum_count):
        if not window or not is_main_process(rank):
            return
        phase_tag = "WARM" if phase == 'warmup' else "JOINT"
        print(f"\n{'─'*55}")
        print(f"[Epoch {epoch} | {phase_tag} | accum_boundary {accum_count}] 最近50步梯度速报")
        print(f"{'─'*55}")
        for name, norms in sorted(window.items()):
            if not norms or name == 'total_loss':
                continue
            mean_norm = sum(norms) / len(norms)
            std_norm = (sum((x - mean_norm)**2 for x in norms) / len(norms)) ** 0.5
            cv = std_norm / (mean_norm + 1e-12)
            mid = len(norms) // 2
            trend = "?"
            if mid > 0:
                early = sum(norms[:mid]) / mid
                late = sum(norms[mid:]) / (len(norms) - mid)
                trend = "↓收敛" if late < early * 0.9 else ("↑活跃" if late > early * 1.1 else "→平稳")
            print(f"  {name:18s} mean={mean_norm:.4e} std={std_norm:.2e} CV={cv:.2f} | {trend}")

        lidar_n = window.get('lidar', [])
        lora_n = window.get('lora_A', []) + window.get('lora_B', [])
        if lidar_n and lora_n:
            ratio = (sum(lidar_n)/len(lidar_n)) / (sum(lora_n)/len(lora_n) + 1e-12)
            flag = "LiDAR>>LoRA" if ratio > 10 else ("LoRA>>LiDAR" if ratio < 0.1 else "平衡")
            print(f"  [梯度比] LiDAR/LoRA={ratio:.2f} {flag}")

        losses = window.get('total_loss', [])
        if losses:
            l_mean = sum(losses) / len(losses)
            l_mid = len(losses) // 2
            l_trend = "?"
            if l_mid > 0:
                el = sum(losses[:l_mid]) / l_mid
                ll = sum(losses[l_mid:]) / (len(losses) - l_mid)
                l_trend = "↓降" if ll < el * 0.95 else ("↑升" if ll > el * 1.05 else "→平")
            print(f"  [Loss] mean={l_mean:.4f} | {l_trend}")
        print(f"{'─'*55}")

    def _print_grad_diagnostics(grad_history, epoch):
        if not grad_history or not is_main_process(rank):
            return
        phase_tag = "WARM" if phase == 'warmup' else "JOINT"
        print(f"\n{'='*60}")
        print(f"[Epoch {epoch} {phase_tag} 梯度诊断 - 全 epoch 累积]")
        print(f"{'='*60}")
        for name, norms in sorted(grad_history.items()):
            if not norms or name == 'total_loss':
                continue
            mean_norm = sum(norms) / len(norms)
            std_norm = (sum((x - mean_norm)**2 for x in norms) / len(norms)) ** 0.5
            max_norm = max(norms)
            min_norm = min(norms)
            cv = std_norm / (mean_norm + 1e-12)
            mid = len(norms) // 2
            trend = "?"
            if mid > 0:
                early = sum(norms[:mid]) / mid
                late = sum(norms[mid:]) / (len(norms) - mid)
                trend = "↓收敛" if late < early * 0.9 else ("↑活跃" if late > early * 1.1 else "→平稳")
            print(f"  {name:18s} | mean={mean_norm:.4e} std={std_norm:.2e} | "
                  f"min={min_norm:.2e} max={max_norm:.2e} | CV={cv:.2f} | {trend}")

        lidar_norms = grad_history.get('lidar', [])
        lora_all = grad_history.get('lora_A', []) + grad_history.get('lora_B', [])
        if lidar_norms and lora_all:
            lidar_mean = sum(lidar_norms) / len(lidar_norms)
            lora_mean = sum(lora_all) / len(lora_all)
            ratio = lidar_mean / (lora_mean + 1e-12)
            flag = "LiDAR>>LoRA" if ratio > 10 else ("LoRA>>LiDAR" if ratio < 0.1 else "平衡")
            print(f"\n  [梯度比] LiDAR/LoRA={ratio:.2f} {flag}")

        all_norms = []
        for k, v in grad_history.items():
            if k != 'total_loss':
                all_norms.extend(v)
        if all_norms:
            overall_mean = sum(all_norms) / len(all_norms)
            if overall_mean < 1e-5:
                print(f"  → 梯度极小({overall_mean:.2e})，可能已收敛或学习率太小")
            elif overall_mean > 10.0:
                print(f"  → 梯度极大({overall_mean:.2e})，学习率过大或震荡")
            else:
                print(f"  → 梯度正常({overall_mean:.2e})")

        losses = grad_history.get('total_loss', [])
        if losses:
            mid = len(losses) // 2
            if mid > 0:
                early_l = sum(losses[:mid]) / mid
                late_l = sum(losses[mid:]) / (len(losses) - mid)
                print(f"  [Loss趋势] 前={early_l:.4f} 后={late_l:.4f} ", end="")
                print("↓下降" if late_l < early_l * 0.95 else ("↑上升" if late_l > early_l * 1.05 else "→持平"))
        print(f"{'='*60}\n")
    # ======================================================

    t_start = time.time()
    for batch_idx, views in enumerate(dataloader):
        for view in views:
            for key, val in view.items():
                if torch.is_tensor(val):
                    view[key] = val.to(device, non_blocking=True)

        # # ========== Warmup 阶段：RGB 输入置零，强制模型只依赖 LiDAR ==========
        rgb_zeroed = False
        if phase == 'warmup' and args.lidar_warmup_rgb_dropout_prob > 0.0:
            if random.random() < args.lidar_warmup_rgb_dropout_prob:
                for view in views:
                    if 'img' in view and torch.is_tensor(view['img']):
                        view['img'].zero_()
                rgb_zeroed = True
        # # ================================================================

        lidar_dropped = False
        if args.lidar and args.lidar_dropout_prob > 0.0 and random.random() < args.lidar_dropout_prob:
            lidar_dropped = True
            lidar_dropout_count += 1
            for view in views:
                if 'pcd' in view and torch.is_tensor(view['pcd']):
                    view['pcd'].zero_()
                if 'lidar_depth_scale' in view and torch.is_tensor(view['lidar_depth_scale']):
                    view['lidar_depth_scale'].fill_(1.0)
        lidar_total_count += 1

        with autocast(enabled=args.amp, dtype=torch.bfloat16):
            predictions = model(views, use_lidar=args.lidar)

            if batch_idx == 0 and is_main_process(rank):
                print(f"[诊断] Epoch{epoch} Batch0 keys: {list(predictions[0].keys()) if predictions else []}")

            loss, metrics = criterion(predictions, views, seq_len=args.seq_len)

        if loss is None:
            _cleanup_batch_tensors(views, predictions, None, None)
            t_start = time.time()
            continue

        loss_val = loss.item() if torch.isfinite(loss) else float('nan')

        if not loss.requires_grad or not torch.isfinite(loss):
            _cleanup_batch_tensors(views, predictions, loss, None)
            t_start = time.time()
            continue

        loss_scaled = loss / args.accum_iter
        if args.amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if batch_idx == 0 and b_first_norm is None:
            for name, p in model.named_parameters():
                if 'lora_B' in name and p.requires_grad:
                    b_first_norm = p.norm().item()
                    break

        _cleanup_batch_tensors(views, predictions, loss, loss_scaled)

        total_loss += loss_val
        num_batches += 1

        is_accum_boundary = ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(dataloader))
        if is_accum_boundary:
            has_nan_grad = False
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    break
            if has_nan_grad:
                optimizer.zero_grad(set_to_none=True)
                nan_grad_batches += 1
                torch.cuda.empty_cache()
                continue

            # 参数分组
            lora_A_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_A' in n]
            lora_B_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' in n]
            lidar_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ('lidars_encoder', 'lidar_film'))]
            fusion_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ('fusion_module', 'fusion_conv'))]
            pose_head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'pose_head' in n and 'lora' not in n]
            dense_head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'dense_head' in n and 'lora' not in n]
            shared_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ('shared_linear', 'shared_decoder', 'output_proj'))]

            # 收集梯度范数
            grad_snapshot = {}
            if lora_A_params: grad_snapshot['lora_A'] = _collect_grad_norm(lora_A_params)
            if lora_B_params: grad_snapshot['lora_B'] = _collect_grad_norm(lora_B_params)
            if lidar_params:  grad_snapshot['lidar'] = _collect_grad_norm(lidar_params)
            if fusion_params: grad_snapshot['fusion'] = _collect_grad_norm(fusion_params)
            if pose_head_params: grad_snapshot['pose_head'] = _collect_grad_norm(pose_head_params)
            if dense_head_params: grad_snapshot['dense_head'] = _collect_grad_norm(dense_head_params)
            if shared_params: grad_snapshot['shared'] = _collect_grad_norm(shared_params)
            grad_snapshot['total_loss'] = loss_val

            for k, v in grad_snapshot.items():
                grad_history[k].append(v)
                grad_window[k].append(v)
                if len(grad_window[k]) > 50:
                    grad_window[k] = grad_window[k][-50:]

            accum_boundary_count += 1

            if accum_boundary_count % 50 == 0:
                _print_grad_window(grad_window, epoch, accum_boundary_count)

            if args.amp:
                scaler.unscale_(optimizer)
                if lora_A_params:
                    torch.nn.utils.clip_grad_norm_(lora_A_params, max_norm=args.grad_clip)
                if lora_B_params:
                    torch.nn.utils.clip_grad_norm_(lora_B_params, max_norm=args.grad_clip * 5.0)
                if lidar_params:
                    torch.nn.utils.clip_grad_norm_(lidar_params, max_norm=0.3)
                if fusion_params:
                    torch.nn.utils.clip_grad_norm_(fusion_params, max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                if lora_A_params:
                    torch.nn.utils.clip_grad_norm_(lora_A_params, max_norm=args.grad_clip)
                if lora_B_params:
                    torch.nn.utils.clip_grad_norm_(lora_B_params, max_norm=args.grad_clip * 5.0)
                if lidar_params:
                    torch.nn.utils.clip_grad_norm_(lidar_params, max_norm=0.3)
                if fusion_params:
                    torch.nn.utils.clip_grad_norm_(fusion_params, max_norm=0.5)
                optimizer.step()

            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            check_grad_accum_count += 1
            if CHECK_GRAD_ONLY and check_grad_accum_count >= MAX_CHECK_BATCHES:
                if is_main_process(rank):
                    print(f"\n CHECK_GRAD 模式：已完成 {MAX_CHECK_BATCHES} 个 accum_boundary")
                    _print_grad_diagnostics(grad_history, epoch)
                    print(" 诊断完成，进程退出")
                cleanup_ddp(torch.distributed.is_initialized())
                sys.exit(0)

        if is_main_process(rank) and batch_idx % args.log_interval == 0:
            lr_str = "/".join([f"{g['lr']:.2e}" for g in optimizer.param_groups])
            drop_tag = "[DROP]" if lidar_dropped else "[LID]"
            phase_tag = "W" if phase == 'warmup' else "J"
            rgb_tag = "[ZERO-RGB]" if rgb_zeroed else ""
            log_str = (f"[Epoch {epoch}] [{batch_idx}/{len(dataloader)}] {phase_tag}{drop_tag}{rgb_tag} "
                       f"LR: {lr_str} Loss: {loss_val:.4f}")
            for key in ['rpe_trans', 'rpe_rot', 'depth', 'ray', 'world_pts']:
                if key in metrics:
                    log_str += f" {key}: {metrics[key]:.4f}"
            print(log_str)

        t_start = time.time()

    if is_main_process(rank) and lidar_total_count > 0:
        drop_rate = lidar_dropout_count / lidar_total_count
        print(f"[Epoch {epoch}] LiDAR Dropout: {lidar_dropout_count}/{lidar_total_count} ({drop_rate*100:.1f}%)")

    if is_main_process(rank):
        b_norms = [p.norm().item() for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
        avg_b = sum(b_norms) / len(b_norms) if b_norms else 0
        print(f"[Epoch {epoch} 结束] B_avg: {avg_b:.4e} | NaNGrad: {nan_grad_batches}")
        _print_grad_diagnostics(grad_history, epoch)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return total_loss / max(num_batches, 1)


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description="MapAnything LiDAR Warmup Training")
    parser.add_argument("--seq_root", type=str, nargs='+',
                        default=["/add02/users/xuyh/seq1/", "/add02/users/xuyh/seq3/"],
                        help="训练数据路径，可指定多个序列")
    parser.add_argument("--model_dir", type=str, default="/home/xuyh/mapanything/")
    parser.add_argument("--output_dir", type=str, default="/add02/users/xuyh/checkpoints/32_lora_lidar")
    parser.add_argument("--cache_dir", type=str, default="./cache/lidar_7ch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_b_multiplier", type=float, default=25.0)
    parser.add_argument("--encoder_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--accum_iter", type=int, default=8)
    parser.add_argument("--use_compile", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lidar_lr_scale", type=float, default=0.2)
    parser.add_argument("--fusion_lr_scale", type=float, default=0.05,
                        help="fusion module 的学习率缩放系数，默认更保守")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--lidar_dropout_prob", type=float, default=0.0)
    
    # ========== LiDAR Warmup 新参数 ==========
    parser.add_argument("--lidar_warmup_epochs", type=int, default=3,
                        help="保留兼容参数；当前默认直接 joint 训练，不执行 warmup")
    parser.add_argument("--lidar_warmup_lr_scale", type=float, default=0.5,
                        help="warmup 阶段的整体 LR 缩放系数，默认 0.5 更稳")
    parser.add_argument("--lidar_warmup_gate_alpha", type=float, default=0.70,
                        help="warmup phase fusion gate target for LiDAR emphasis")
    parser.add_argument("--fusion_smooth_alpha", type=float, default=0.55,
                        help="joint phase fusion gate target after warmup")
    parser.add_argument("--lidar_warmup_rgb_dropout_prob", type=float, default=0.50,
                        help="probability of zeroing RGB inputs during LiDAR warmup")
    # ========================================
    
    # 训练开关
    parser.add_argument("--lora", action="store_true", default=True, help="启用 LoRA 训练")
    parser.add_argument("--no-lora", action="store_false", dest="lora", help="禁用 LoRA 训练")
    parser.add_argument("--lidar", action="store_true", default=True, help="启用 LiDAR 训练")
    parser.add_argument("--no-lidar", action="store_false", dest="lidar", help="禁用 LiDAR 训练")
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
        print("=" * 65)
        print("MapAnything LiDAR Joint Training")
        print(f"LoRA: {'ON' if args.lora else 'OFF'} | LiDAR: {'ON' if args.lidar else 'OFF'}")
        if args.lidar_warmup_epochs > 0:
            print(f"LiDAR Warmup 参数已保留，仅用于兼容旧实验：{args.lidar_warmup_epochs}")
        print("=" * 65)

    # 加载模型（传入 warmup 配置，控制 fusion_module 初始化）
    model = build_model(
        args.model_dir, device, rank,
        use_compile=False,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        use_lora=args.lora, use_lidar=args.lidar,
        lidar_warmup_epochs=args.lidar_warmup_epochs,
        lidar_warmup_gate_alpha=args.lidar_warmup_gate_alpha
    )

    # 数据集
    if len(args.seq_root) == 1:
        dataset = Seq1LidarDataset(
            seq_root=args.seq_root[0], seq_len=args.seq_len, stride=args.stride,
            img_size=args.img_size, cache_dir=args.cache_dir, tolerance=args.tolerance,
            use_lidar=args.lidar
        )
    else:
        datasets = []
        for idx, root in enumerate(args.seq_root):
            cache_subdir = os.path.join(args.cache_dir, f"seq{idx}") if args.cache_dir else None
            ds = Seq1LidarDataset(
                seq_root=root, seq_len=args.seq_len, stride=args.stride,
                img_size=args.img_size, cache_dir=cache_subdir, tolerance=args.tolerance,
                use_lidar=args.lidar
            )
            datasets.append(ds)
        dataset = ConcatDataset(datasets)

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

    criterion = MapAnythingLoss(
        w_depth=0.1, w_pose_trans=2.0, w_pose_rot=0.5, w_ray=0.1,
        w_pts3d_cam=0.1, w_world_pts=0.1, w_confidence=0.1,
        w_scale=0.1, robust_alpha=0.5, robust_c=0.05,
        top_n_percent=5.0, use_log_depth=True, use_chordal_rot=True)

    scaler = GradScaler(enabled=args.amp)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, gradient_as_bucket_view=True
        )

    # 根据阶段构建初始优化器
    steps_per_epoch = len(dataloader) // args.accum_iter
    
    current_phase = 'warmup' if args.lidar_warmup_epochs > 0 else 'joint'
    set_lidar_warmup_phase(model, enable_warmup=(current_phase == 'warmup'), args=args, rank=rank)
    
    optimizer = build_optimizer_for_phase(model, args, rank, phase=current_phase)
    scheduler, warmup_steps = build_scheduler(optimizer, args, steps_per_epoch)

    # EMA
    ema = None
    if args.use_ema and is_main_process(rank):
        target_model = model.module if is_ddp else model
        ema = ModelEMA(target_model, decay=args.ema_decay)

    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    resumed_phase = current_phase
    
    if args.resume:
        target_model = model.module if is_ddp else model
        start_epoch, best_loss = smart_resume(
            target_model, optimizer, scheduler, scaler, ema,
            args.resume, rank, device
        )
        
        resumed_phase = 'warmup' if start_epoch < args.lidar_warmup_epochs else 'joint'
        set_lidar_warmup_phase(model, enable_warmup=(resumed_phase == 'warmup'), args=args, rank=rank)
        optimizer = build_optimizer_for_phase(model, args, rank, phase=resumed_phase)
        scheduler, warmup_steps = build_scheduler(optimizer, args, steps_per_epoch)
        
        if is_main_process(rank):
            print(f"[Resume] 从 epoch {start_epoch + 1} 继续训练 | phase={resumed_phase}")

    # ========== 训练循环（含阶段切换）==========
    for epoch in range(start_epoch + 1, args.epochs):
        
        # ---- 阶段切换检测 ----
        # 保持单阶段 joint 训练，不做 warmup -> joint 的硬切换
        
        desired_phase = 'warmup' if args.lidar_warmup_epochs > 0 and epoch <= args.lidar_warmup_epochs else 'joint'
        if desired_phase != current_phase:
            current_phase = desired_phase
            set_lidar_warmup_phase(model, enable_warmup=(current_phase == 'warmup'), args=args, rank=rank)
            if current_phase == 'joint' and args.lidar_warmup_epochs > 0:
                reinitialize_fusion_module_for_joint_training(
                    model, rank, smooth_alpha=args.fusion_smooth_alpha
                )
            optimizer = build_optimizer_for_phase(model, args, rank, phase=current_phase)
            scheduler, warmup_steps = build_scheduler(optimizer, args, steps_per_epoch)
            if ema is not None and current_phase == 'joint':
                target_model = model.module if is_ddp else model
                ema = ModelEMA(target_model, decay=args.ema_decay)

        if sampler is not None:
            sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, scaler, criterion,
            device, epoch, args, rank, scheduler, ema, phase=current_phase
        )
        epoch_time = time.time() - epoch_start

        if is_main_process(rank):
            phase_tag = "WARM" if current_phase == 'warmup' else "JOINT"
            print(f"Epoch {epoch} [{phase_tag}] 完成 | 平均损失: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")
            
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
                torch.cuda.empty_cache()
                save_model = model.module if is_ddp else model
                save_checkpoint(save_model, optimizer, scheduler, scaler, epoch, best_loss, ckpt_path, True, ema)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.cuda.empty_cache()
                save_model = model.module if is_ddp else model
                save_checkpoint(save_model, optimizer, scheduler, scaler, epoch, best_loss, best_path, True, ema)
                print(f"  -> 最佳模型 (loss={best_loss:.4f}): {best_path}")
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    if is_main_process(rank):
        print("训练完成!")
    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
