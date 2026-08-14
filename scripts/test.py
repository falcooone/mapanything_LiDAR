#!/usr/bin/env python3
# coding: utf-8
"""
MapAnything LoRA Evaluation Script (Fixed)
==========================================
关键修复：
  1. 图像预处理与训练脚本完全一致（PIL + /255 + bilinear resize），
     避免 load_images 内置 tvf.Normalize 导致的二次归一化。
  2. LoRA 权重加载兼容 lora_state_dict / model_state_dict。
  3. data_norm_type 统一为 list 格式。
  5. uses_torch_hub=False（与训练一致）。
  6. lora_alpha 默认 32.0（与训练一致）。
  7. cap 去掉 *255 缩放（与训练一致，值域 [0,1]）。
  8. pcd shape 修正为 [B, H, W, 9]（与训练一致，channel-last）。
  9. 加 fusion_module 权重诊断。
"""

import os
import sys
import json
import time
import argparse
import glob
import re
import math
import copy
import gc
import warnings
from typing import List, Dict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as SciR
import open3d as o3d
from PIL import Image

try:
    from scipy.spatial import cKDTree, procrustes
except ImportError:
    cKDTree = None
    procrustes = None

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("[警告] 未安装 openpyxl，xlsx 导出将不可用。请执行: pip install openpyxl")

from mapanything.models import MapAnything
from safetensors.torch import load_file

warnings.filterwarnings('ignore')

# ========================= 确定性设置 =========================
def setup_deterministic(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_deterministic(42)

# ========================= 环境设置 =========================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/tmp/hf_cache"

# ========================= LoRA (与训练脚本逐行一致) =========================
class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
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


def inject_lora_to_module(module: nn.Module, r: int = 8, lora_alpha: float = 16.0):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(child, r, lora_alpha))
        else:
            inject_lora_to_module(child, r, lora_alpha)


LEGACY_KEY_ALIASES = {
    "fusion_gate_mlp.0.weight": "fusion_module.gate_mlp.0.weight",
    "fusion_gate_mlp.0.bias": "fusion_module.gate_mlp.0.bias",
    "fusion_gate_mlp.2.weight": "fusion_module.gate_mlp.2.weight",
    "fusion_gate_mlp.2.bias": "fusion_module.gate_mlp.2.bias",
    "fusion_refine.weight": "fusion_module.refine.weight",
}


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    alias_hits = {}
    for key, value in state_dict.items():
        new_key = LEGACY_KEY_ALIASES.get(key, key)
        normalized[new_key] = value
        if new_key != key:
            alias_hits[key] = new_key
    return normalized, alias_hits


GLOBAL_DEPTH_MAX = 40.0
LIDAR_NUM_CHANNELS = 9
MAX_TIMESTAMP_TOLERANCE_SEC = 0.05


def _lidar_points_to_camera_axes(points_xyz: np.ndarray) -> np.ndarray:
    # Confirmed manually from projection overlays: X_cam=Y_lidar, Y_cam=Z_lidar, Z_cam=X_lidar.
    return np.stack([points_xyz[:, 1], points_xyz[:, 2], points_xyz[:, 0]], axis=1).astype(np.float32)


def _empty_pcd_feature_dict(H, W, device):
    depth = torch.zeros((H, W), dtype=torch.float32, device=device)
    normal = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    cap = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    valid = torch.zeros((H, W), dtype=torch.float32, device=device)
    edge = torch.zeros((H, W), dtype=torch.float32, device=device)
    scale = torch.ones((1,), dtype=torch.float32, device=device)
    return {
        'cap': cap,
        'depth': depth,
        'normal': normal,
        'valid': valid,
        'edge': edge,
        'scale': scale,
    }


def _compute_sparse_depth_edge(depth_rel, valid_mask):
    edge = np.zeros_like(depth_rel, dtype=np.float32)

    dx = np.abs(depth_rel[:, 1:] - depth_rel[:, :-1])
    valid_x = valid_mask[:, 1:] & valid_mask[:, :-1]
    dx = np.where(valid_x, dx, 0.0)
    edge[:, 1:] = np.maximum(edge[:, 1:], dx)
    edge[:, :-1] = np.maximum(edge[:, :-1], dx)

    dy = np.abs(depth_rel[1:, :] - depth_rel[:-1, :])
    valid_y = valid_mask[1:, :] & valid_mask[:-1, :]
    dy = np.where(valid_y, dy, 0.0)
    edge[1:, :] = np.maximum(edge[1:, :], dy)
    edge[:-1, :] = np.maximum(edge[:-1, :], dy)

    positive = edge[edge > 0]
    if positive.size > 0:
        scale = float(np.percentile(positive, 95))
        edge = edge / max(scale, 1e-6)
    return np.clip(edge, 0.0, 1.0).astype(np.float32)


LEGACY_KEY_ALIASES = {
    "fusion_gate_mlp.0.weight": "fusion_module.gate_mlp.0.weight",
    "fusion_gate_mlp.0.bias": "fusion_module.gate_mlp.0.bias",
    "fusion_gate_mlp.2.weight": "fusion_module.gate_mlp.2.weight",
    "fusion_gate_mlp.2.bias": "fusion_module.gate_mlp.2.bias",
    "fusion_refine.weight": "fusion_module.refine.weight",
}


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    alias_hits = {}
    for key, value in state_dict.items():
        new_key = LEGACY_KEY_ALIASES.get(key, key)
        normalized[new_key] = value
        if new_key != key:
            alias_hits[key] = new_key
    return normalized, alias_hits


# ========================= LiDAR 预处理 (与 pcd.py 逐行一致) =========================
def rotation_matrix_from_lookat(direction, up=np.array([0, 0, 1])):
    z_cam = direction / np.linalg.norm(direction)
    x_cam = np.cross(up, z_cam)
    norm_x = np.linalg.norm(x_cam)
    if norm_x < 1e-6:
        x_cam = np.cross(np.array([1, 0, 0]), z_cam)
        norm_x = np.linalg.norm(x_cam)
    x_cam = x_cam / norm_x
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

def _scale_intrinsics(K, src_size, dst_size):
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    K_scaled = K.astype(np.float32).copy()
    if src_w > 0 and src_h > 0:
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        K_scaled[0, 0] *= sx
        K_scaled[0, 2] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[1, 2] *= sy
    return K_scaled


def get_pcd_features(
    pcd_path,
    K,
    intrinsics_size,
    output_size=(448, 448),
    device="cuda",
    pcd_cache=None,
    pcd_idx=None,
):
    if pcd_cache is not None and pcd_idx is not None and pcd_idx in pcd_cache:
        cached = pcd_cache[pcd_idx]
        if device is None:
            return cached
        return {k: v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v for k, v in cached.items()}

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    W, H = output_size
    if K is None:
        result = _empty_pcd_feature_dict(H, W, device)
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}
        return result
    K_proj = _scale_intrinsics(K, intrinsics_size, (W, H))
    fx, fy = float(K_proj[0, 0]), float(K_proj[1, 1])
    cx, cy = float(K_proj[0, 2]), float(K_proj[1, 2])

    if len(points) == 0:
        result = _empty_pcd_feature_dict(H, W, device)
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}
        return result

    pts_cam = _lidar_points_to_camera_axes(points.astype(np.float32))
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid_mask = z > 1e-4
    if not np.any(valid_mask):
        result = _empty_pcd_feature_dict(H, W, device)
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}
        return result

    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_image):
        result = _empty_pcd_feature_dict(H, W, device)
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}
        return result

    u = u[in_image].astype(int)
    v = v[in_image].astype(int)
    z = z[in_image]
    valid_indices = np.where(valid_mask)[0][in_image]

    normals, curv, aniso, plan = compute_features_for_indices(pcd, valid_indices, radius=0.1, max_nn=30)
    good_mask = ~np.isnan(curv)
    if not np.any(good_mask):
        result = _empty_pcd_feature_dict(H, W, device)
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}
        return result

    u, v, z = u[good_mask], v[good_mask], z[good_mask]
    normals = normals[good_mask]
    curv = curv[good_mask]
    aniso = aniso[good_mask]
    plan = plan[good_mask]

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

    valid_img = np.isfinite(depth_img)
    if not np.any(valid_img):
        depth_img = np.zeros((H, W), dtype=np.float32)
        z_d = 1.0
    else:
        z_d = float(np.mean(depth_img[valid_img]))
        depth_img[~valid_img] = 0.0
        z_d = max(z_d, 1e-3)

    # ========== 修复：去掉 *255，与训练一致（值域 [0,1]）==========
    cap_np = np.stack([curv_img, aniso_img, plan_img], axis=-1)
    cap_np = np.clip(cap_np, 0.0, 1.0).astype(np.float32)
    # ============================================================

    depth_np = np.clip(depth_img / GLOBAL_DEPTH_MAX, 0.0, 1.0).astype(np.float32)
    valid_np = valid_img.astype(np.float32)
    edge_np = _compute_sparse_depth_edge(depth_np, valid_img)
    normal_np = normal_img.astype(np.float32)

    cap_tensor = torch.from_numpy(cap_np)
    depth_tensor = torch.from_numpy(depth_np)
    normal_tensor = torch.from_numpy(normal_np)
    valid_tensor = torch.from_numpy(valid_np)
    edge_tensor = torch.from_numpy(edge_np)
    scale_tensor = torch.tensor([z_d], dtype=torch.float32)

    result_cpu = {
        'cap': cap_tensor,
        'depth': depth_tensor,
        'normal': normal_tensor,
        'valid': valid_tensor,
        'edge': edge_tensor,
        'scale': scale_tensor,
    }
    if pcd_cache is not None and pcd_idx is not None:
        pcd_cache[pcd_idx] = result_cpu
    if device is None:
        return result_cpu
    return {k: v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v for k, v in result_cpu.items()}

# ========================= 健壮内参解析 =========================
def _parse_yaml_intrinsics(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        k_match = re.search(r"K:\s*\[\[(.*?)\]\]", content, re.DOTALL)
        if not k_match:
            return None
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", k_match.group(1))
        if len(numbers) >= 9:
            return np.array([float(num) for num in numbers[:9]], dtype=np.float32).reshape(3, 3)
    except Exception as e:
        print(f"解析YAML内参文件失败 {file_path}: {e}")
    return None

def _load_intrinsics_file(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        data = np.loadtxt(file_path)
        if data.shape == (3, 3):
            return data
        if data.size == 9:
            return data.reshape(3, 3)
    except:
        pass
    K = _parse_yaml_intrinsics(file_path)
    if K is not None:
        return K
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    numbers = []
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or ':' in line:
            continue
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
        numbers.extend([float(n) for n in nums])
    if len(numbers) >= 9:
        return np.array(numbers[:9]).reshape(3, 3)
    return None


def _load_intrinsics_with_size(file_path):
    K = _load_intrinsics_file(file_path)
    width = 0
    height = 0
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            width_match = re.search(r"width:\s*(\d+)", content)
            height_match = re.search(r"height:\s*(\d+)", content)
            if width_match:
                width = int(width_match.group(1))
            if height_match:
                height = int(height_match.group(1))
        except Exception:
            pass
    return K, (width, height)

# ========================= 模型加载 (保留 FiLM/LoRA 支持) =========================
def build_model_for_eval(model_dir: str, device: str,
                         trained_ckpt_path: str = None,
                         use_lora: bool = False,
                         lora_r: int = 8, lora_alpha: float = 16.0):
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    with open(config_path, 'r') as f:
        config = json.load(f)

    encoder_config = config.get("encoder_config", {}).copy()
    encoder_config.pop("pretrained", None)
    encoder_config.pop("weights", None)
    # ========== 修复：uses_torch_hub = False（与训练一致）==========
    encoder_config["uses_torch_hub"] = False
    # =============================================================
    geometric_input_config = copy.deepcopy(config.get("geometric_input_config", {}))
    lidar_encoder_config = geometric_input_config.get("lidars_encoder_config", {})
    lidar_encoder_config["in_chans"] = LIDAR_NUM_CHANNELS
    lidar_encoder_config["uses_torch_hub"] = False
    lidar_encoder_config.pop("pretrained", None)
    lidar_encoder_config.pop("weights", None)
    # Keep the LiDAR ResNet grid aligned with the 32x32 RGB token grid:
    # 512x512 input with stride 16 produces 32x32 features.
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
        info_sharing_mlp_layer_str="swiglufused"
    )
    fusion_conv = getattr(model, "fusion_conv", None)
    if fusion_conv is not None and hasattr(fusion_conv, "weight"):
        with torch.no_grad():
            fusion_conv.weight.zero_()
            if fusion_conv.bias is not None:
                fusion_conv.bias.zero_()

    if os.path.exists(weights_path):
        print(f"[Model] 加载预训练权重: {weights_path}")
        state_dict = load_file(weights_path)
        model_state = model.state_dict()
        filtered_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = len(state_dict) - len(filtered_state)
        if skipped:
            print(f"[Model] skipped pretrained keys with incompatible shape: {skipped}")
        model.load_state_dict(filtered_state, strict=False)
    else:
        print("[Model] 警告: 未找到预训练权重")

    if use_lora:
        print(f"[Model] 应用 LoRA (rank={lora_r}, alpha={lora_alpha})...")
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

        lora_layer_count = sum(1 for _ in model.modules() if isinstance(_, LinearWithLoRA))
        print(f"  -> 已注入 LoRA: {lora_layer_count} 个 Linear 层 (r={lora_r}, alpha={lora_alpha})")

        model = model.to(device)
        if trained_ckpt_path and os.path.exists(trained_ckpt_path):
            print(f"[Model] 加载训练权重: {trained_ckpt_path}")
            ckpt = torch.load(trained_ckpt_path, map_location='cpu')

            if 'lora_state_dict' in ckpt:
                state_dict = ckpt['lora_state_dict']
                print(f"[Model] 检测到 LoRA-only checkpoint ({len(state_dict)} 个参数)")
            elif 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
                print("[Model] 检测到完整模型 checkpoint")
            else:
                state_dict = ckpt
                print("[Model] 警告: 未识别标准 key，尝试直接加载整个 checkpoint")

            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '')
                new_state_dict[new_k] = v

            new_state_dict, alias_hits = _normalize_state_dict_keys(new_state_dict)

            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(new_state_dict.keys())
            common_keys = model_keys & ckpt_keys
            lora_common = [k for k in common_keys if 'lora_' in k]
            fusion_common = [k for k in common_keys if k.startswith(('fusion_module.', 'fusion_conv.'))]
            if alias_hits:
                print("[Fusion-Diag] legacy fusion key aliases detected:")
                for old_k, new_k in alias_hits.items():
                    print(f"  - {old_k} -> {new_k}")
            print(f"[LoRA-Diag] 模型 LoRA 参数总数: {sum(1 for k in model_keys if 'lora_' in k)}")
            print(f"[LoRA-Diag] checkpoint 中 LoRA 参数总数: {sum(1 for k in ckpt_keys if 'lora_' in k)}")
            print(f"[LoRA-Diag] 成功匹配的 LoRA 参数: {len(lora_common)}")

            if lora_common:
                sample_key = None
                for k in lora_common:
                    if new_state_dict[k].shape == model.state_dict()[k].shape:
                        sample_key = k
                        break
                if sample_key is not None:
                    diff = torch.abs(model.state_dict()[sample_key].cpu() - new_state_dict[sample_key].cpu()).max().item()
                    print(f"[LoRA-Diag] 抽样 {sample_key}: 权重差异 max={diff:.6f}")
                else:
                    print("[LoRA-Diag] 所有公共 LoRA key 的 shape 都不一致，跳过抽样 diff 检查")

                a_norms = [model.state_dict()[k].norm().item() for k in model_keys if 'lora_A' in k]
                b_norms = [model.state_dict()[k].norm().item() for k in model_keys if 'lora_B' in k]
                print(f"[LoRA-Diag] 加载前 LoRA_A 平均范数: {sum(a_norms)/len(a_norms):.4f} (初始化≈0.01~0.02)")
                print(f"[LoRA-Diag] 加载前 LoRA_B 平均范数: {sum(b_norms)/len(b_norms):.4f} (初始化≈0.0)")
            else:
                print("[LoRA-Diag] 警告: 未匹配到任何 LoRA 参数！请检查训练/测试脚本结构是否一致。")

            filtered_dict = {}
            for k in common_keys:
                if new_state_dict[k].shape == model.state_dict()[k].shape:
                    filtered_dict[k] = new_state_dict[k]
                else:
                    print(f"[LoRA-Diag] 形状不匹配跳过: {k} | ckpt={tuple(new_state_dict[k].shape)} model={tuple(model.state_dict()[k].shape)}")

            missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
            if missing:
                print(f"[Model] 缺失 keys: {len(missing)} (含 LoRA {sum(1 for k in missing if 'lora_' in k)} 个)")
            if unexpected:
                print(f"[Model] 意外 keys: {len(unexpected)}")

            if lora_common:
                a_norms_after = [model.state_dict()[k].norm().item() for k in model_keys if 'lora_A' in k]
                b_norms_after = [model.state_dict()[k].norm().item() for k in model_keys if 'lora_B' in k]
                print(f"[LoRA-Diag] 加载后 LoRA_A 平均范数: {sum(a_norms_after)/len(a_norms_after):.4f}")
                print(f"[LoRA-Diag] 加载后 LoRA_B 平均范数: {sum(b_norms_after)/len(b_norms_after):.4f}")

            # ========== 新增：fusion_module 权重诊断 ==========
            # ========== fusion_module ???? ==========
            with torch.no_grad():
                fusion_module = getattr(model, "fusion_module", None)
                if fusion_module is not None:
                    gate_mlp = getattr(fusion_module, "gate_mlp", None)
                    refine = getattr(fusion_module, "refine", None)
                    if gate_mlp is not None and len(gate_mlp) >= 3:
                        gate_bias = gate_mlp[-1].bias.mean().item()
                        gate_weight_norm = gate_mlp[-1].weight.norm().item()
                        gate_weight_abs_mean = gate_mlp[-1].weight.abs().mean().item()
                        gate_sigmoid = torch.sigmoid(gate_mlp[-1].bias.mean()).item()
                        print(
                            f"[Fusion-Diag] gate_bias_mean={gate_bias:.6f}, "
                            f"gate_weight_norm={gate_weight_norm:.6e}, "
                            f"gate_weight_abs_mean={gate_weight_abs_mean:.6e}, "
                            f"gate_sigmoid(mean_bias)={gate_sigmoid:.6f}"
                        )
                    if refine is not None and hasattr(refine, "weight"):
                        refine_norm = refine.weight.norm().item()
                        refine_abs_mean = refine.weight.abs().mean().item()
                        print(
                            f"[Fusion-Diag] refine_weight_norm={refine_norm:.6e}, "
                            f"refine_weight_abs_mean={refine_abs_mean:.6e}"
                        )
                    if fusion_common:
                        print(f"[Fusion-Diag] loaded fusion keys: {len(fusion_common)}")
                    fusion_conv = getattr(model, "fusion_conv", None)
                    if fusion_conv is not None and hasattr(fusion_conv, "weight"):
                        conv_norm = fusion_conv.weight.norm().item()
                        conv_abs_mean = fusion_conv.weight.abs().mean().item()
                        print(
                            f"[Fusion-Diag] fusion_conv_weight_norm={conv_norm:.6e}, "
                            f"fusion_conv_weight_abs_mean={conv_abs_mean:.6e}"
                        )
                else:
                    print("[Fusion-Diag] warning: fusion_module not found")
            # ================================================
    else:
        if trained_ckpt_path and os.path.exists(trained_ckpt_path):
            print(f"[Model] 加载训练权重: {trained_ckpt_path}")
            ckpt = torch.load(trained_ckpt_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '')
                new_state_dict[new_k] = v
            new_state_dict, alias_hits = _normalize_state_dict_keys(new_state_dict)
            model_keys = set(model.state_dict().keys())
            common_keys = model_keys & set(new_state_dict.keys())
            filtered_dict = {}
            for k in common_keys:
                if new_state_dict[k].shape == model.state_dict()[k].shape:
                    filtered_dict[k] = new_state_dict[k]
            missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
            fusion_common = [
                k for k in filtered_dict
                if k.startswith(('lidars_encoder.', 'lidar_film.', 'fusion_module.', 'fusion_conv.'))
            ]
            print(f"[Model] loaded non-LoRA checkpoint keys: {len(filtered_dict)}")
            print(f"[Model] loaded LiDAR/Fusion keys without LoRA: {len(fusion_common)}")
            if alias_hits:
                print(f"[Fusion-Diag] legacy fusion aliases: {len(alias_hits)}")
            if missing:
                print(f"[Model] missing keys without LoRA: {len(missing)}")
            if unexpected:
                print(f"[Model] unexpected keys without LoRA: {len(unexpected)}")
            fusion_conv = getattr(model, "fusion_conv", None)
            if fusion_conv is not None and hasattr(fusion_conv, "weight"):
                print(f"[Fusion-Diag] fusion_conv_weight_norm={fusion_conv.weight.norm().item():.6e}")

    model = model.to(device)
    model.eval()
    return model

# ========================= 数据加载 =========================
def _compute_rgb_quality_stats(image_path: str):
    """
    Compute simple RGB quality metrics on the grayscale image.
    Brightness: mean intensity in [0, 1].
    Contrast: RMS contrast / standard deviation in [0, 1].
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None
    img_norm = img_gray.astype(np.float32) / 255.0
    brightness = float(np.mean(img_norm))
    contrast = float(np.sqrt(np.mean((img_norm - brightness) ** 2)))
    return {
        "brightness": brightness,
        "contrast": contrast,
    }


def load_eval_data(seq_root: str, img_size: int = 448, use_lidar: bool = False,
                   max_images: int = None, tolerance: float = MAX_TIMESTAMP_TOLERANCE_SEC,
                   max_rgb_brightness: float = None, max_rgb_contrast: float = None):
    rgb_dir = os.path.join(seq_root, "rgb")
    if not os.path.exists(rgb_dir):
        raise ValueError(f"RGB 目录不存在: {rgb_dir}")

    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    img_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)
                        if f.lower().endswith(img_exts)])
    if not img_paths:
        raise ValueError("未找到图像")

    filtered_img_paths = []
    rejected_quality = []
    for img_path in img_paths:
        quality = _compute_rgb_quality_stats(img_path)
        if quality is None:
            continue
        brightness = quality["brightness"]
        contrast = quality["contrast"]
        if max_rgb_brightness is not None and brightness > max_rgb_brightness:
            rejected_quality.append((img_path, brightness, contrast, "brightness"))
            continue
        if max_rgb_contrast is not None and contrast > max_rgb_contrast:
            rejected_quality.append((img_path, brightness, contrast, "contrast"))
            continue
        filtered_img_paths.append(img_path)

    if max_rgb_brightness is not None or max_rgb_contrast is not None:
        print(
            f"[Data][Quality] kept={len(filtered_img_paths)}/{len(img_paths)} "
            f"max_brightness={max_rgb_brightness} max_contrast={max_rgb_contrast}"
        )
        if rejected_quality:
            sample_rejects = rejected_quality[:5]
            for img_path, brightness, contrast, reason in sample_rejects:
                print(
                    f"[Data][Quality] reject({reason}) {os.path.basename(img_path)} "
                    f"brightness={brightness:.4f} contrast={contrast:.4f}"
                )

    img_paths = filtered_img_paths
    if max_images:
        img_paths = img_paths[:max_images]
    print(f"[Data] 图像: {len(img_paths)} 张")

    intrinsics_file = os.path.join(seq_root, "color_camera_intrinsics.txt")
    intrinsics_raw, intrinsics_size = _load_intrinsics_with_size(intrinsics_file)
    if intrinsics_raw is None:
        print("[Data] 内参无法解析，使用单位阵")
        intrinsics_raw = np.eye(3, dtype=np.float32)
        intrinsics_size = (img_size, img_size)
    intrinsics = _scale_intrinsics(intrinsics_raw, intrinsics_size, (img_size, img_size))

    tum_file = os.path.join(seq_root, "extrinsics.tum")
    gt_poses = {}
    if os.path.exists(tum_file):
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
                T_w2c[:3, :3] = rot
                T_w2c[:3, 3] = [tx, ty, tz]
                gt_poses[ts] = np.linalg.inv(T_w2c)
        print(f"[Data] 真值位姿: {len(gt_poses)} 帧")
    else:
        print("[Data] 警告: 未找到 extrinsics.tum")

    gt_depths = {}
    is_part_dir = re.match(r'shuangchuang_seq\d+_(night|daytime)\d+th', os.path.basename(seq_root))

    if is_part_dir:
        depth_dirs = [os.path.join(seq_root, "depth")]
    else:
        depth_dirs = []
        for part in sorted(os.listdir(seq_root)):
            part_path = os.path.join(seq_root, part)
            if not os.path.isdir(part_path):
                continue
            if not re.match(r'shuangchuang_seq\d+_(night|daytime)\d+th', part):
                continue
            ddir = os.path.join(part_path, "depth")
            if os.path.exists(ddir):
                depth_dirs.append(ddir)

    for depth_dir in depth_dirs:
        if not os.path.exists(depth_dir):
            continue
        for df in glob.glob(os.path.join(depth_dir, "*")):
            if not os.path.isfile(df):
                continue
            ext = os.path.splitext(df)[1].lower()
            if ext not in ('.npy', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'):
                continue
            m = re.search(r'(\d+)', os.path.basename(df))
            if not m:
                continue
            ts = int(m.group(1))
            if ext == '.npy':
                gt_depths[ts] = np.load(df).astype(np.float32)
            else:
                img = Image.open(df)
                d = np.array(img).astype(np.float32)
                if d.ndim == 3:
                    d = d.mean(axis=2)
                d = d / 1000.0
                gt_depths[ts] = d

    if gt_depths:
        print(f"[Data] 真值深度图: {len(gt_depths)} 张 (扫描了 {len(depth_dirs)} 个 depth 目录)")
    else:
        print("[Data] 警告: 未找到真值深度图，深度评测将跳过")

    pcd_file_list = []
    pcd_timestamps = []
    if use_lidar:
        lidar_dir = os.path.join(seq_root, "lidar")
        if os.path.exists(lidar_dir):
            pcd_files = sorted([f for f in os.listdir(lidar_dir) if f.lower().endswith('.pcd')])
            pcd_file_list = [os.path.join(lidar_dir, f) for f in pcd_files]
            for f in pcd_files:
                m = re.search(r'(\d+)', f)
                if m:
                    ts = int(m.group(1))
                    pcd_timestamps.append(ts)
                else:
                    ts = int(os.path.getmtime(os.path.join(lidar_dir, f)) * 1e9)
                    pcd_timestamps.append(ts)
            print(f"[Data] LiDAR PCD: {len(pcd_file_list)} 个")
        else:
            print("[Data] 警告: 启用 LiDAR 但未找到 lidar 目录")

        if not pcd_timestamps:
            print("[Data] LiDAR 时间戳为空，将退化为 RGB-only 推理路径")

    return (
        img_paths,
        intrinsics,
        gt_poses,
        gt_depths,
        pcd_file_list,
        pcd_timestamps,
        intrinsics_raw,
        intrinsics_size,
    )

# ========================= 关键修复：手动图像加载（与训练一致） =========================
def load_images_manual(image_paths, img_size=448, device='cuda'):
    """
    与训练脚本 SeqRGBDataset.__getitem__ 逐行一致的图像加载。
    输出图像范围 [0, 1]，由模型内部根据 data_norm_type='dinov2' 做归一化。
    """
    views = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_np_color = np.array(img)

        # 与训练一致：transpose + /255
        img_np = img_np_color.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)

        # resize 到 img_size（与训练一致）
        if img_tensor.shape[1] != img_size or img_tensor.shape[2] != img_size:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # confidence 计算与训练一致
        img_gray = img_np_color.astype(np.float32).mean(axis=2) / 255.0
        mean = np.mean(img_gray)
        rms = np.sqrt(np.mean((img_gray - mean) ** 2))
        confidence = float(rms) if not np.isnan(rms) else 0.5

        views.append({
            'img': img_tensor.unsqueeze(0).to(device),
            'data_norm_type': ['dinov2'],  # 必须是 list，与训练一致
            'confidence': torch.tensor(confidence, dtype=torch.float32, device=device),
        })
    return views


def _timestamp_ns_from_image_path(image_path):
    match = re.search(r'color_(\d+)', image_path)
    if match:
        return int(match.group(1))
    return int(os.path.getmtime(image_path) * 1e9)


def _pcd_feature_has_valid_points(feat_dict) -> bool:
    valid = feat_dict.get("valid")
    if valid is None:
        return False
    if torch.is_tensor(valid):
        return bool(torch.isfinite(valid).all() and torch.count_nonzero(valid).item() > 0)
    valid_np = np.asarray(valid)
    return bool(np.isfinite(valid_np).all() and np.count_nonzero(valid_np) > 0)


# ========================= ????????????=========================
def run_inference_batch(model, image_paths, device, use_lidar, pcd_file_list, pcd_timestamps,
                        intrinsics_raw=None, intrinsics_size=(0, 0),
                        img_size=448, view_pcd_indices=None, pcd_cache=None):
    """Use the same preprocessing path as training."""
    target_h, target_w = img_size, img_size
    views = load_images_manual(image_paths, img_size=img_size, device=device)

    B = views[0]['img'].shape[0]
    if intrinsics_raw is not None:
        intrinsics_scaled = _scale_intrinsics(intrinsics_raw, intrinsics_size, (target_w, target_h))
        intrinsics_tensor = torch.from_numpy(intrinsics_scaled).to(device=device, dtype=torch.float32).unsqueeze(0)
        for view in views:
            view['intrinsics'] = intrinsics_tensor.expand(view['img'].shape[0], -1, -1).contiguous()

    # confidence is computed from the original image when absent.
    for i, view in enumerate(views):
        if 'confidence' not in view:
            try:
                img_gray = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
                if img_gray is not None:
                    img_norm = img_gray.astype(np.float32) / 255.0
                    mean = np.mean(img_norm)
                    rms_contrast = np.sqrt(np.mean((img_norm - mean) ** 2))
                    confidence = rms_contrast
                else:
                    confidence = 0.5
            except Exception:
                confidence = 0.5
            view['confidence'] = torch.tensor(confidence, dtype=torch.float32, device=device)

    # 处理激光雷达数据（如果启用）
    if use_lidar:
        if view_pcd_indices is None:
            view_pcd_indices = [0] * len(views)
        if len(view_pcd_indices) != len(views):
            raise ValueError(
                "view_pcd_indices must contain one entry per image/view"
            )

        valid_pcd_indices = {
            int(idx)
            for idx in view_pcd_indices
            if idx is not None and int(idx) >= 0 and int(idx) < len(pcd_file_list)
        }
        pcd_features = {}

        for idx in valid_pcd_indices:
            try:
                feat_dict = get_pcd_features(
                    pcd_file_list[idx],
                    intrinsics_raw,
                    intrinsics_size,
                    output_size=(target_w, target_h),
                    device=device,
                    pcd_cache=pcd_cache,
                    pcd_idx=idx,
                )
            except Exception as e:
                print(f"[LiDAR] PCD 异常，回退为零特征 idx={idx}: {pcd_file_list[idx]} | {e}")
                feat_dict = _empty_pcd_feature_dict(target_h, target_w, device)
            if not _pcd_feature_has_valid_points(feat_dict):
                print(f"[LiDAR] 空投影 PCD，回退为零特征 idx={idx}: {pcd_file_list[idx]}")

            # Build LiDAR feature map as (H, W, 9):
            # cap(3), depth(1), normal(3), valid(1), depth edge(1).
            cap = feat_dict['cap'].to(dtype=torch.float32)       # (H, W, 3)
            depth = feat_dict['depth'].to(dtype=torch.float32)   # (H, W)
            normal = feat_dict['normal'].to(dtype=torch.float32) # (H, W, 3)
            valid = feat_dict['valid'].to(dtype=torch.float32)   # (H, W)
            edge = feat_dict['edge'].to(dtype=torch.float32)     # (H, W)

            depth = depth.unsqueeze(-1)                           # (H, W, 1)
            valid = valid.unsqueeze(-1)
            edge = edge.unsqueeze(-1)
            pcd_lidar = torch.cat([cap, depth, normal, valid, edge], dim=-1)

            # Resize through channel-first layout, then convert back to channel-last.
            pcd_lidar = pcd_lidar.permute(2, 0, 1).unsqueeze(0)
            pcd_lidar = F.interpolate(
                pcd_lidar, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )
            pcd_lidar = pcd_lidar.squeeze(0).permute(1, 2, 0)
            pcd_features[idx] = {
                "pcd": pcd_lidar.unsqueeze(0),
                "scale": feat_dict["scale"].to(dtype=torch.float32),
                "valid": feat_dict.get("valid"),
            }
            # =====================================================================

        for image_path, view, pcd_idx in zip(image_paths, views, view_pcd_indices):
            if pcd_idx is None or int(pcd_idx) not in pcd_features:
                print(f"[LiDAR] 样本缺少可用 PCD，保留 RGB-only: {os.path.basename(image_path)}")
                continue
            pcd_idx = int(pcd_idx)
            # Expand batch dimension: final shape is (B, H, W, LIDAR_NUM_CHANNELS).
            view['pcd'] = pcd_features[pcd_idx]["pcd"].expand(B, -1, -1, -1).contiguous()
            view['lidar_depth_scale'] = pcd_features[pcd_idx]["scale"].expand(B).contiguous()

    # 统一所有张量到同一设备
    dev = torch.device(device)
    for view in views:
        for key, value in list(view.items()):
            if isinstance(value, torch.Tensor):
                view[key] = value.to(dev, dtype=torch.float32)
            elif isinstance(value, (list, tuple)) and any(isinstance(v, torch.Tensor) for v in value):
                view[key] = [v.to(dev, dtype=torch.float32) if isinstance(v, torch.Tensor) else v for v in value]

    # 调试打印（首次 batch 时）
    if not hasattr(run_inference_batch, '_printed'):
        img = views[0]['img']
        print(f"[Debug-Input] img shape: {img.shape}, "
              f"min={img.min():.3f}, max={img.max():.3f}, mean={img.mean():.3f}, "
              f"data_norm_type: {views[0].get('data_norm_type')}")
        if 'pcd' in views[0]:
            print(f"[Debug-Input] pcd shape: {views[0]['pcd'].shape}, "
                  f"min={views[0]['pcd'].min():.3f}, max={views[0]['pcd'].max():.3f}")
        run_inference_batch._printed = True

    try:
        with torch.no_grad():
            predictions = model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype="bf16" if dev.type == "cuda" else "fp32",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                use_lidar=use_lidar,
            )
        return predictions, views, image_paths
    except Exception as e:
        print(f"批次处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ========================= 输出提取 =========================
def extract_batch_outputs(predictions, image_paths, views):
    if predictions is None:
        return []
    batch_outputs = []
    for i, (pred, img_path, view) in enumerate(zip(predictions, image_paths, views)):
        conf_val = view.get('confidence', 0.5)
        if isinstance(conf_val, torch.Tensor):
            conf_val = conf_val.item()
        output = {
            'image_path': img_path,
            'image_name': os.path.basename(img_path),
            'batch_index': i,
            'confidence': conf_val,
        }
        for key in ["pts3d", "depth_z", "camera_poses", "intrinsics", "conf", "mask",
                    "img_no_norm", "ray_directions"]:
            if key in pred:
                value = pred[key]
                if torch.is_tensor(value):
                    value_np = value.cpu().numpy()
                    if value_np.ndim > 0 and value_np.shape[0] == 1:
                        value_np = value_np[0]
                    output[key] = value_np
        batch_outputs.append(output)
    return batch_outputs

# ========================= 全局 SE(3) 对齐工具 =========================
def align_trajectory_SE3(pred_poses, gt_poses):
    pred_trans = np.array([p[:3, 3] for p in pred_poses])
    gt_trans = np.array([p[:3, 3] for p in gt_poses])

    pred_center = np.mean(pred_trans, axis=0)
    gt_center = np.mean(gt_trans, axis=0)
    pred_centered = pred_trans - pred_center
    gt_centered = gt_trans - gt_center

    H = pred_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = gt_center - R @ pred_center

    aligned_poses = []
    for p in pred_poses:
        T_aligned = np.eye(4, dtype=np.float32)
        T_aligned[:3, :3] = R @ p[:3, :3]
        T_aligned[:3, 3] = (R @ p[:3, 3]) + t
        aligned_poses.append(T_aligned)

    return aligned_poses


POINTCLOUD_THRESHOLDS_M = (0.05, 0.10, 0.20)
POINTCLOUD_VOXEL_SIZE_M = 0.05
POINTCLOUD_MAX_POINTS_PER_FRAME = 20000
POINTCLOUD_MAX_POINTS_TOTAL = 300000


def _as_hw_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth


def _depth_to_camera_points(depth, K, valid_mask):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth[valid_mask].astype(np.float32)
    x = (u[valid_mask] - float(K[0, 2])) * z / float(K[0, 0])
    y = (v[valid_mask] - float(K[1, 2])) * z / float(K[1, 1])
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _transform_points(points_cam, pose_c2w):
    R = pose_c2w[:3, :3].astype(np.float32)
    t = pose_c2w[:3, 3].astype(np.float32)
    return (points_cam @ R.T) + t


def _subsample_points(points, max_points, rng):
    if points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _voxel_downsample_np(points, voxel_size):
    if points.size == 0 or voxel_size <= 0:
        return points
    voxel = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def _nearest_distances(src_points, dst_points):
    tree = cKDTree(dst_points)
    distances, _ = tree.query(src_points, k=1)
    return distances.astype(np.float32)


def _compute_pointcloud_reconstruction_metrics(
    matched_frames,
    pred_poses_aligned,
    gt_poses_raw,
    gt_depths,
    gt_intrinsics,
    tolerance=0.05,
):
    tolerance = max(float(tolerance), 0.0)
    if cKDTree is None or not gt_depths or gt_intrinsics is None:
        return {}

    depth_timestamps = np.array(sorted(gt_depths.keys()), dtype=np.int64)
    if len(depth_timestamps) == 0:
        return {}

    rng = np.random.default_rng(42)
    pred_clouds = []
    gt_clouds = []
    pc_frame_count = 0

    for m, pred_pose, gt_pose in zip(matched_frames, pred_poses_aligned, gt_poses_raw):
        pred_d = m.get("depth_z")
        if pred_d is None:
            continue

        img_ts_raw = int(round(m["pred_ts"] * 1e9))
        idx = np.argmin(np.abs(depth_timestamps - img_ts_raw))
        best_ts = int(depth_timestamps[idx])
        if abs(best_ts - img_ts_raw) > tolerance * 1e9:
            continue

        pred_d = _as_hw_depth(pred_d)
        gt_d = _as_hw_depth(gt_depths[best_ts])

        if pred_d.shape[:2] != gt_d.shape[:2]:
            gt_d = cv2.resize(
                gt_d,
                (pred_d.shape[1], pred_d.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        valid_pred = np.isfinite(pred_d) & (pred_d > 0)
        valid_gt = np.isfinite(gt_d) & (gt_d > 0)
        if not np.any(valid_pred) or not np.any(valid_gt):
            continue

        pred_points_cam = _depth_to_camera_points(pred_d, gt_intrinsics, valid_pred)
        gt_points_cam = _depth_to_camera_points(gt_d, gt_intrinsics, valid_gt)

        pred_points_world = _transform_points(pred_points_cam, pred_pose)
        gt_points_world = _transform_points(gt_points_cam, gt_pose)

        pred_points_world = _subsample_points(
            pred_points_world, POINTCLOUD_MAX_POINTS_PER_FRAME, rng
        )
        gt_points_world = _subsample_points(
            gt_points_world, POINTCLOUD_MAX_POINTS_PER_FRAME, rng
        )

        pred_clouds.append(pred_points_world)
        gt_clouds.append(gt_points_world)
        pc_frame_count += 1

    if not pred_clouds or not gt_clouds:
        return {}

    pred_points = np.concatenate(pred_clouds, axis=0).astype(np.float32)
    gt_points = np.concatenate(gt_clouds, axis=0).astype(np.float32)

    pred_points = _voxel_downsample_np(pred_points, POINTCLOUD_VOXEL_SIZE_M)
    gt_points = _voxel_downsample_np(gt_points, POINTCLOUD_VOXEL_SIZE_M)
    pred_points = _subsample_points(pred_points, POINTCLOUD_MAX_POINTS_TOTAL, rng)
    gt_points = _subsample_points(gt_points, POINTCLOUD_MAX_POINTS_TOTAL, rng)

    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return {}

    pred_to_gt = _nearest_distances(pred_points, gt_points)
    gt_to_pred = _nearest_distances(gt_points, pred_points)

    accuracy = float(np.mean(pred_to_gt))
    completeness = float(np.mean(gt_to_pred))
    metrics = {
        "pc_num_frames": int(pc_frame_count),
        "pc_num_pred_points": int(pred_points.shape[0]),
        "pc_num_gt_points": int(gt_points.shape[0]),
        "pc_accuracy_mean": accuracy,
        "pc_completeness_mean": completeness,
        "pc_chamfer_l1": float(accuracy + completeness),
        "pc_rmse": float(
            np.sqrt((np.mean(pred_to_gt ** 2) + np.mean(gt_to_pred ** 2)) / 2.0)
        ),
    }

    for tau in POINTCLOUD_THRESHOLDS_M:
        precision = float(np.mean(pred_to_gt < tau) * 100.0)
        recall = float(np.mean(gt_to_pred < tau) * 100.0)
        fscore = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
        suffix = f"{int(round(tau * 100))}cm"
        metrics[f"pc_precision_{suffix}"] = precision
        metrics[f"pc_recall_{suffix}"] = recall
        metrics[f"pc_fscore_{suffix}"] = float(fscore)
        metrics[f"pc_outlier_{suffix}"] = float(100.0 - precision)

    return metrics


# ========================= 评测 =========================
def run_comprehensive_validation(predictions, gt_poses, gt_depths, gt_intrinsics, output_dir,
                                 use_lora, use_lidar, dataset_id, tolerance = 0.05):
    if not predictions or not gt_poses:
        print("[Eval] 无预测结果或真值，跳过评测")
        return None

    tolerance = max(float(tolerance), 0.0)

    def ts_from_file(f):
        match = re.search(r'color_(\d+)', f)
        if match:
            return int(match.group(1)) / 1e9
        return None

    gt_timestamps = np.array(sorted(gt_poses.keys()))
    gt_pose_dict = gt_poses

    matched_frames = []
    for pred in predictions:
        name = pred['image_name']
        pred_ts = ts_from_file(name)
        if pred_ts is None:
            continue

        idx = np.searchsorted(gt_timestamps, pred_ts)
        if idx == 0:
            best_ts = gt_timestamps[0]
        elif idx == len(gt_timestamps):
            best_ts = gt_timestamps[-1]
        else:
            left_ts = gt_timestamps[idx - 1]
            right_ts = gt_timestamps[idx]
            best_ts = left_ts if abs(pred_ts - left_ts) < abs(pred_ts - right_ts) else right_ts
        
        # tolerance 单位是秒，pred_ts 单位是秒，不需要转换
        if abs(pred_ts - best_ts) > tolerance:
            continue

        pred_pose = pred.get('camera_poses')
        if pred_pose is None:
            continue

        gt_pose = gt_pose_dict[best_ts]
        matched_frames.append({
            'pred_ts': pred_ts,
            'best_ts': best_ts,
            'pred_pose': pred_pose,
            'gt_pose': gt_pose,
            'depth_z': pred.get('depth_z'),
            'ray_directions': pred.get('ray_directions'),
            'intrinsics': pred.get('intrinsics'),
        })

    if len(matched_frames) < 2:
        print("[Eval] 有效帧不足 2，无法计算 RPE/ATE")
        return None

    print(f"[Eval] 找到 {len(matched_frames)} 帧用于评测")
    matched_frames.sort(key=lambda x: x['pred_ts'])

    pred_poses_raw = [m['pred_pose'] for m in matched_frames]
    gt_poses_raw = [m['gt_pose'] for m in matched_frames]

    print(f"[Eval-Diag] 第一帧 GT 位姿平移: {gt_poses_raw[0][:3, 3]}, 旋转行列式: {np.linalg.det(gt_poses_raw[0][:3, :3]):.3f}")
    print(f"[Eval-Diag] 第一帧 Pred 位姿平移: {pred_poses_raw[0][:3, 3]}, 旋转行列式: {np.linalg.det(pred_poses_raw[0][:3, :3]):.3f}")

    pred_poses_aligned = align_trajectory_SE3(pred_poses_raw, gt_poses_raw)

    pred_trans = np.array([p[:3, 3] for p in pred_poses_aligned])
    gt_trans = np.array([p[:3, 3] for p in gt_poses_raw])
    ate = np.sqrt(np.mean(np.sum((gt_trans - pred_trans) ** 2, axis=1)))

    abs_rot_errors = []
    for p_pred, p_gt in zip(pred_poses_aligned, gt_poses_raw):
        R_pred = p_pred[:3, :3]
        R_gt = p_gt[:3, :3]
        R_rel = R_pred @ R_gt.T
        cos_angle = (np.trace(R_rel) - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        abs_rot_errors.append(angle)

    mean_abs_rot = np.mean(abs_rot_errors)
    std_abs_rot = np.std(abs_rot_errors)

    abs_trans_errors = [np.linalg.norm(p[:3, 3] - g[:3, 3]) for p, g in zip(pred_poses_aligned, gt_poses_raw)]
    mean_abs_trans = np.mean(abs_trans_errors)
    std_abs_trans = np.std(abs_trans_errors)

    rpe_trans_errors = []
    rpe_rot_errors = []
    for i in range(len(pred_poses_aligned) - 1):
        pred_rel = np.linalg.inv(pred_poses_aligned[i]) @ pred_poses_aligned[i + 1]
        gt_rel = np.linalg.inv(gt_poses_raw[i]) @ gt_poses_raw[i + 1]

        trans_err = np.linalg.norm(pred_rel[:3, 3] - gt_rel[:3, 3])
        rpe_trans_errors.append(trans_err)

        R_pred_rel = pred_rel[:3, :3]
        R_gt_rel = gt_rel[:3, :3]
        R_err = R_pred_rel @ R_gt_rel.T
        cos_angle = (np.trace(R_err) - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        rot_err = np.arccos(cos_angle) * 180 / np.pi
        rpe_rot_errors.append(rot_err)

    # ========== RRA 指标（与旧版完全一致，带 deg 后缀）==========
    rra_thresholds = [0.5, 1.0, 1.5]
    rra_results = {}
    for tau in rra_thresholds:
        rra = np.mean(np.array(rpe_rot_errors) < tau) * 100
        rra_results[tau] = rra
    # ==========================================================

    rta_thresholds = [0.1, 0.2, 0.3, 0.5]
    rta_results = {}
    for tau in rta_thresholds:
        rta = np.mean(np.array(rpe_trans_errors) < tau) * 100
        rta_results[tau] = rta

    depth_rels = []
    depth_taus = []
    depth_match_count = 0
    if gt_depths and gt_intrinsics is not None:
        depth_timestamps = np.array(sorted(gt_depths.keys()), dtype=np.int64)

        for m in matched_frames:
            pred_d = m.get('depth_z')
            if pred_d is None:
                continue

            img_ts_raw = int(round(m['pred_ts'] * 1e9))

            if len(depth_timestamps) == 0:
                continue

            idx = np.argmin(np.abs(depth_timestamps - img_ts_raw))
            best_ts = int(depth_timestamps[idx])
            diff = abs(best_ts - img_ts_raw)

            tolerance_ns = tolerance * 1e9

            if diff > tolerance_ns:
                continue

            gt_d = gt_depths[best_ts]
            depth_match_count += 1

            if pred_d.shape[:2] != gt_d.shape[:2]:
                import cv2
                pred_d = cv2.resize(pred_d, (gt_d.shape[1], gt_d.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
            valid = (gt_d > 0) & (pred_d > 0)
            if not np.any(valid):
                continue
            diff = np.abs(pred_d - gt_d)
            rel = diff / (gt_d + 1e-8)
            rel_valid = rel[valid]
            depth_rels.append(np.mean(rel_valid))
            depth_taus.append(np.mean(rel_valid < 0.0103) * 100)

        print(f"[Eval] 深度图匹配成功: {depth_match_count} / {len(matched_frames)} 帧")

    ray_errs = []
    for m in matched_frames:
        pred_rays = m.get('ray_directions')
        if pred_rays is None or gt_intrinsics is None:
            continue
        h, w = pred_rays.shape[:2]
        K = gt_intrinsics
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones_like(u)
        pix_hom = np.stack([u, v, ones], axis=-1).reshape(-1, 3).T
        ray_dir = np.linalg.inv(K) @ pix_hom
        ray_dir = ray_dir.T.reshape(h, w, 3)
        ray_dir = ray_dir / (np.linalg.norm(ray_dir, axis=-1, keepdims=True) + 1e-8)
        pred_rays = pred_rays / (np.linalg.norm(pred_rays, axis=-1, keepdims=True) + 1e-8)
        dot = np.sum(pred_rays * ray_dir, axis=-1)
        dot = np.clip(dot, -1, 1)
        angle = np.arccos(dot) * 180 / np.pi
        ray_errs.append(np.mean(angle))

    pc_stats = _compute_pointcloud_reconstruction_metrics(
        matched_frames,
        pred_poses_aligned,
        gt_poses_raw,
        gt_depths,
        gt_intrinsics,
        tolerance,
    )

    # ========== stats dict key 与旧版完全一致（带 deg 后缀）==========
    stats = {
        'num_frames': len(matched_frames),
        'ate_rmse': float(ate),
        'rra_0.5deg': float(rra_results[0.5]),
        'rra_1.0deg': float(rra_results[1.0]),
        'rra_1.5deg': float(rra_results[1.5]),
        'rta_0.1m': float(rta_results[0.1]),
        'rta_0.2m': float(rta_results[0.2]),
        'rta_0.3m': float(rta_results[0.3]),
        'rta_0.5m': float(rta_results[0.5]),
        'abs_rot_error_mean': float(mean_abs_rot),
        'abs_rot_error_std': float(std_abs_rot),
        'abs_trans_error_mean': float(mean_abs_trans),
        'abs_trans_error_std': float(std_abs_trans),
        'rpe_trans_mean': float(np.mean(rpe_trans_errors)) if rpe_trans_errors else None,
        'rpe_trans_std': float(np.std(rpe_trans_errors)) if rpe_trans_errors else None,
        'rpe_rot_mean': float(np.mean(rpe_rot_errors)) if rpe_rot_errors else None,
        'rpe_rot_std': float(np.std(rpe_rot_errors)) if rpe_rot_errors else None,
        'depth_rel_mean': float(np.mean(depth_rels)) if depth_rels else None,
        'depth_rel_std': float(np.std(depth_rels)) if depth_rels else None,
        'depth_tau_mean': float(np.mean(depth_taus)) if depth_taus else None,
        'depth_tau_std': float(np.std(depth_taus)) if depth_taus else None,
        'ray_error_mean': float(np.mean(ray_errs)) if ray_errs else None,
        'ray_error_std': float(np.std(ray_errs)) if ray_errs else None,
    }
    stats.update(pc_stats)
    # =================================================================

    # ========== 打印输出与旧版完全一致（带 ° 符号）==========
    print("\n论文评测结果:")
    print(f"  ATE RMSE (全局对齐): {stats['ate_rmse']:.4f} m")
    print(f"  RRA@0.5° (RPE-based): {stats['rra_0.5deg']:.2f} %")
    print(f"  RRA@1.0° (RPE-based): {stats['rra_1.0deg']:.2f} %")
    print(f"  RRA@1.5° (RPE-based): {stats['rra_1.5deg']:.2f} %")
    print(f"  RTA@0.1m (RPE-based): {stats['rta_0.1m']:.2f} %")
    print(f"  RTA@0.2m (RPE-based): {stats['rta_0.2m']:.2f} %")
    print(f"  RTA@0.3m (RPE-based): {stats['rta_0.3m']:.2f} %")
    print(f"  RTA@0.5m (RPE-based): {stats['rta_0.5m']:.2f} %")
    print(f"  绝对旋转误差 (参考): 均值 = {stats['abs_rot_error_mean']:.2f}°, 标准差 = {stats['abs_rot_error_std']:.2f}°")
    print(f"  绝对平移误差 (参考): 均值 = {stats['abs_trans_error_mean']:.4f} m, 标准差 = {stats['abs_trans_error_std']:.4f} m")
    if rpe_trans_errors:
        print(f"  RPE trans: 均值 = {stats['rpe_trans_mean']:.4f} m, 标准差 = {stats['rpe_trans_std']:.4f} m")
        print(f"  RPE rot: 均值 = {stats['rpe_rot_mean']:.2f}°, 标准差 = {stats['rpe_rot_std']:.2f}°")
    if depth_rels:
        print(f"  Depth rel: {stats['depth_rel_mean']:.4f} ± {stats['depth_rel_std']:.4f}")
        print(f"  Depth τ: {stats['depth_tau_mean']:.2f} ± {stats['depth_tau_std']:.2f} %")
    else:
        print(f"  Depth: 未计算（匹配失败 0 帧，检查时间戳单位）")
    if ray_errs:
        print(f"  Ray error: {stats['ray_error_mean']:.2f} ± {stats['ray_error_std']:.2f} deg")
    if pc_stats:
        print(
            f"  Point cloud Chamfer-L1: {stats['pc_chamfer_l1']:.4f} m "
            f"(Acc={stats['pc_accuracy_mean']:.4f} m, Comp={stats['pc_completeness_mean']:.4f} m)"
        )
        print(
            f"  Point cloud F-score@10cm: {stats['pc_fscore_10cm']:.2f} % "
            f"(P={stats['pc_precision_10cm']:.2f} %, R={stats['pc_recall_10cm']:.2f} %)"
        )
    else:
        print("  Point cloud: 未计算（缺少 GT depth / 有效点云 / scipy cKDTree）")
    # =========================================================

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{use_lora}_{use_lidar}_{dataset_id}_evaluation_results.json")
    with open(result_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[Eval] 结果已保存: {result_path}")
    return stats

# ========================= 数据集 ID 解析 =========================
def parse_dataset_id(seq_root: str) -> str:
    basename = os.path.basename(seq_root)
    m = re.search(r'seq(\d+)_(?:night|daytime)(\d+)th', basename)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    parent = os.path.basename(os.path.dirname(seq_root))
    m2 = re.search(r'seq(\d+)', parent)
    m3 = re.search(r'(?:night|daytime)(\d+)th', basename)
    if m2 and m3:
        return f"{m2.group(1)}_{m3.group(2)}"

    return "unknown"

# ========================= xlsx 导出 =========================
def save_results_to_xlsx(stats: dict, output_dir: str, use_lora: int, use_lidar: int, dataset_id: str):
    if not HAS_OPENPYXL:
        print("[xlsx] openpyxl 未安装，跳过 xlsx 导出")
        return

    filename = f"{use_lora}_{use_lidar}_{dataset_id}.xlsx"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Evaluation Results"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center")

    headers = ["Metric", "Value", "Unit", "Description"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align

    # ========== 指标名称与旧版完全一致（带 deg 后缀）==========
    rows = [
        ("ATE_RMSE", stats.get('ate_rmse'), "m", "全局 SE(3) 对齐后的轨迹 RMSE"),
        ("RRA@0.5°", stats.get('rra_0.5deg'), "%", "RPE 相对旋转误差 < 0.5° 的比例"),
        ("RRA@1.0°", stats.get('rra_1.0deg'), "%", "RPE 相对旋转误差 < 1.0° 的比例"),
        ("RRA@1.5°", stats.get('rra_1.5deg'), "%", "RPE 相对旋转误差 < 1.5° 的比例"),
        ("RTA@0.1m", stats.get('rta_0.1m'), "%", "RPE 相对平移误差 < 0.1m 的比例"),
        ("RTA@0.2m", stats.get('rta_0.2m'), "%", "RPE 相对平移误差 < 0.2m 的比例"),
        ("RTA@0.3m", stats.get('rta_0.3m'), "%", "RPE 相对平移误差 < 0.3m 的比例"),
        ("RTA@0.5m", stats.get('rta_0.5m'), "%", "RPE 相对平移误差 < 0.5m 的比例"),
        ("Abs_Rot_Mean", stats.get('abs_rot_error_mean'), "deg", "绝对旋转误差均值（参考）"),
        ("Abs_Rot_Std", stats.get('abs_rot_error_std'), "deg", "绝对旋转误差标准差"),
        ("Abs_Trans_Mean", stats.get('abs_trans_error_mean'), "m", "绝对平移误差均值"),
        ("Abs_Trans_Std", stats.get('abs_trans_error_std'), "m", "绝对平移误差标准差"),
        ("RPE_Trans_Mean", stats.get('rpe_trans_mean'), "m", "RPE 平移误差均值"),
        ("RPE_Trans_Std", stats.get('rpe_trans_std'), "m", "RPE 平移误差标准差"),
        ("RPE_Rot_Mean", stats.get('rpe_rot_mean'), "deg", "RPE 旋转误差均值"),
        ("RPE_Rot_Std", stats.get('rpe_rot_std'), "deg", "RPE 旋转误差标准差"),
        ("Depth_Rel_Mean", stats.get('depth_rel_mean'), "-", "相对深度误差均值"),
        ("Depth_Rel_Std", stats.get('depth_rel_std'), "-", "相对深度误差标准差"),
        ("Depth_Tau_Mean", stats.get('depth_tau_mean'), "%", "深度 τ 阈值通过率"),
        ("Depth_Tau_Std", stats.get('depth_tau_std'), "%", "深度 τ 阈值通过率标准差"),
        ("Ray_Error_Mean", stats.get('ray_error_mean'), "deg", "射线方向误差均值"),
        ("Ray_Error_Std", stats.get('ray_error_std'), "deg", "射线方向误差标准差"),
        ("PC_Chamfer_L1", stats.get('pc_chamfer_l1'), "m", "预测点云到 GT 点云的双向最近邻 L1 距离"),
        ("PC_Accuracy", stats.get('pc_accuracy_mean'), "m", "预测点到 GT 点云的平均最近邻距离"),
        ("PC_Completeness", stats.get('pc_completeness_mean'), "m", "GT 点到预测点云的平均最近邻距离"),
        ("PC_RMSE", stats.get('pc_rmse'), "m", "预测/GT 双向最近邻 RMSE"),
        ("PC_Precision@5cm", stats.get('pc_precision_5cm'), "%", "预测点中距离 GT 小于 5cm 的比例"),
        ("PC_Recall@5cm", stats.get('pc_recall_5cm'), "%", "GT 点中距离预测点小于 5cm 的比例"),
        ("PC_Fscore@5cm", stats.get('pc_fscore_5cm'), "%", "5cm 阈值下的点云 F-score"),
        ("PC_Outlier@5cm", stats.get('pc_outlier_5cm'), "%", "预测点中距离 GT 大于等于 5cm 的比例"),
        ("PC_Precision@10cm", stats.get('pc_precision_10cm'), "%", "预测点中距离 GT 小于 10cm 的比例"),
        ("PC_Recall@10cm", stats.get('pc_recall_10cm'), "%", "GT 点中距离预测点小于 10cm 的比例"),
        ("PC_Fscore@10cm", stats.get('pc_fscore_10cm'), "%", "10cm 阈值下的点云 F-score"),
        ("PC_Outlier@10cm", stats.get('pc_outlier_10cm'), "%", "预测点中距离 GT 大于等于 10cm 的比例"),
        ("PC_Precision@20cm", stats.get('pc_precision_20cm'), "%", "预测点中距离 GT 小于 20cm 的比例"),
        ("PC_Recall@20cm", stats.get('pc_recall_20cm'), "%", "GT 点中距离预测点小于 20cm 的比例"),
        ("PC_Fscore@20cm", stats.get('pc_fscore_20cm'), "%", "20cm 阈值下的点云 F-score"),
        ("PC_Outlier@20cm", stats.get('pc_outlier_20cm'), "%", "预测点中距离 GT 大于等于 20cm 的比例"),
        ("PC_Num_Frames", stats.get('pc_num_frames'), "-", "参与点云指标计算的帧数"),
        ("PC_Num_Pred_Points", stats.get('pc_num_pred_points'), "-", "下采样后的预测点数"),
        ("PC_Num_GT_Points", stats.get('pc_num_gt_points'), "-", "下采样后的 GT 点数"),
        ("Num_Frames", stats.get('num_frames'), "-", "参与评测的有效帧数"),
    ]
    # ==========================================================

    for metric, value, unit, desc in rows:
        if value is None:
            value = "N/A"
        else:
            if isinstance(value, float):
                if abs(value) < 0.01:
                    value = f"{value:.6f}"
                elif abs(value) < 1:
                    value = f"{value:.4f}"
                else:
                    value = f"{value:.2f}"
        ws.append([metric, value, unit, desc])

    ws.column_dimensions['A'].width = 18
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 8
    ws.column_dimensions['D'].width = 45

    ws.freeze_panes = 'A2'

    wb.save(filepath)
    print(f"[xlsx] 结果已保存: {filepath}")

# ========================= 主函数 =========================
def main():
    parser = argparse.ArgumentParser(description="MapAnything LoRA Evaluation (Fixed)")
    parser.add_argument("--seq_root", type=str, default="/add02/users/xuyh/seq2/shuangchuang_seq2_night1th")
    parser.add_argument("--model_dir", type=str, default="/home/xuyh/mapanything/")
    parser.add_argument("--trained_ckpt", type=str, default="/add02/users/xuyh/checkpoints/32_lora_lidar/checkpoints/epoch_009_full.pt")
    parser.add_argument("--output_dir", type=str, default="/add02/users/xuyh/mapanything/output/")
    parser.add_argument("--use_lidar", type=int, default=1)
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="测试时每次输入模型的视图数。注意：若 use_lora=1 且 info_sharing 被 LoRA，建议设为训练时的 seq_len(4)")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--max_rgb_brightness",
        type=float,
        default=None,
        help="only keep RGB images whose grayscale mean brightness is <= this value in [0, 1]",
    )
    parser.add_argument(
        "--max_rgb_contrast",
        type=float,
        default=None,
        help="only keep RGB images whose grayscale RMS contrast is <= this value in [0, 1]",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="maximum RGB/GT/depth/LiDAR timestamp error in seconds",
    )
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()
    args.tolerance = max(float(args.tolerance), 0.0)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = False

    dataset_id = parse_dataset_id(args.seq_root)
    print(f"[Info] 数据集标识: {dataset_id}")

    print("\n[1/4] 加载模型...")
    model = build_model_for_eval(
        args.model_dir, device,
        trained_ckpt_path=args.trained_ckpt if args.trained_ckpt else None,
        use_lora=bool(args.use_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    print("\n[2/4] 加载数据...")
    (
        img_paths,
        intrinsics,
        gt_poses,
        gt_depths,
        pcd_file_list,
        pcd_timestamps,
        intrinsics_raw,
        intrinsics_size,
    ) = load_eval_data(
        args.seq_root, img_size=args.img_size,
        use_lidar=bool(args.use_lidar),
        max_images=args.max_images,
        tolerance=args.tolerance,
        max_rgb_brightness=args.max_rgb_brightness,
        max_rgb_contrast=args.max_rgb_contrast,
    )

    print("\n[3/4] 开始推理...")
    t0 = time.time()
    all_predictions = []
    num_batches = (len(img_paths) + args.batch_size - 1) // args.batch_size
    pcd_cache = {}
    pcd_timestamps_np = np.array(pcd_timestamps) if pcd_timestamps else np.array([])
    pcd_match_total = 0
    pcd_match_valid = 0
    pcd_nearest_min_sec = float('inf')
    pcd_nearest_max_sec = 0.0
    pcd_nearest_sum_sec = 0.0

    for b in range(num_batches):
        batch_paths = img_paths[b * args.batch_size:(b + 1) * args.batch_size]
        print(f"[Infer] Batch {b+1}/{num_batches}: {len(batch_paths)} 张图像")

        view_pcd_indices = []
        if args.use_lidar and len(pcd_file_list) > 0:
            for img_path in batch_paths:
                img_ts = _timestamp_ns_from_image_path(img_path)
                pcd_diff_ns = np.abs(pcd_timestamps_np - img_ts)
                closest_idx = int(np.argmin(pcd_diff_ns))
                nearest_diff_sec = float(pcd_diff_ns[closest_idx]) / 1e9
                pcd_match_total += 1
                pcd_nearest_min_sec = min(pcd_nearest_min_sec, nearest_diff_sec)
                pcd_nearest_max_sec = max(pcd_nearest_max_sec, nearest_diff_sec)
                pcd_nearest_sum_sec += nearest_diff_sec
                if pcd_diff_ns[closest_idx] <= args.tolerance * 1e9:
                    pcd_match_valid += 1
                view_pcd_indices.append(closest_idx)
        else:
            view_pcd_indices = [0] * len(batch_paths) if len(pcd_file_list) > 0 else [None] * len(batch_paths)
            if args.use_lidar:
                pcd_match_total += len(batch_paths)

        predictions, views, kept_batch_paths = run_inference_batch(
            model, batch_paths, device,
            use_lidar=bool(args.use_lidar),
            pcd_file_list=pcd_file_list,
            pcd_timestamps=pcd_timestamps,
            intrinsics_raw=intrinsics_raw,
            intrinsics_size=intrinsics_size,
            img_size=args.img_size,
            view_pcd_indices=view_pcd_indices,
            pcd_cache=pcd_cache,
        )

        if predictions is None:
            print(f"[Infer] Batch {b+1} 推理失败, 跳过")
            torch.cuda.empty_cache()
            continue
        batch_outputs = extract_batch_outputs(predictions, kept_batch_paths, views)
        all_predictions.extend(batch_outputs)

        del predictions
        if views is not None:
            del views
        del batch_outputs
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[3/4] 推理完成: {len(all_predictions)} 帧, 耗时 {time.time()-t0:.1f}s")
    if args.use_lidar:
        if pcd_match_total > 0 and pcd_timestamps_np.size > 0:
            print(
                f"[TimeMatch] RGB-PCD valid={pcd_match_valid}/{pcd_match_total} "
                f"within {args.tolerance:.3f}s; nearest diff "
                f"min={pcd_nearest_min_sec:.6f}s, "
                f"mean={pcd_nearest_sum_sec / pcd_match_total:.6f}s, "
                f"max={pcd_nearest_max_sec:.6f}s"
            )
        else:
            print("[TimeMatch] no PCD timestamps available; all frames use RGB-only path")

    print("\n[4/4] 开始评测...")
    stats = run_comprehensive_validation(
        all_predictions, gt_poses, gt_depths, intrinsics, args.output_dir,
        use_lora=args.use_lora, use_lidar=args.use_lidar, dataset_id=dataset_id,
        tolerance=args.tolerance)

    if stats:
        save_results_to_xlsx(
            stats,
            output_dir=args.output_dir,
            use_lora=args.use_lora,
            use_lidar=args.use_lidar,
            dataset_id=dataset_id
        )

    print("\n全部完成!")

if __name__ == "__main__":
    main()
