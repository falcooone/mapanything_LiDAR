#!/usr/bin/env python3
#coding=gbk
"""
MapAnything LoRA Evaluation Script (Fixed)
==========================================
关键修复：
  1. 图像预处理与训练脚本完全一致（PIL + /255 + bilinear resize），
     避免 load_images 内置 tvf.Normalize 导致的二次归一化。
  2. LoRA 权重加载兼容 lora_state_dict / model_state_dict。
  3. data_norm_type 统一为 list 格式。
  4. 支持 --use_load_images 开关用于对比实验。
"""

import os
import sys
import json
import time
import argparse
import glob
import re
import math
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
    from scipy.spatial import procrustes
except ImportError:
    procrustes = None

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("[警告] 未安装 openpyxl，xlsx 导出将不可用。请执行: pip install openpyxl")

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
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
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

setup_deterministic(42)

# ========================= 环境设置 =========================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/tmp/hf_cache"

# ========================= LoRA (与训练脚本逐行一致) =========================
class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, lora_alpha: float = 2.0):
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


def inject_lora_to_module(module: nn.Module, r: int = 8, lora_alpha: float = 2.0):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(child, r, lora_alpha))
        else:
            inject_lora_to_module(child, r, lora_alpha)


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

def get_pcd_features(pcd_path, device="cuda", pcd_cache=None, pcd_idx=None):
    if pcd_cache is not None and pcd_idx is not None and pcd_idx in pcd_cache:
        return pcd_cache[pcd_idx]

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    img_size = (448, 448)
    f = img_size[0] / 2
    cx, cy = img_size[0] / 2, img_size[1] / 2
    H, W = img_size[1], img_size[0]

    direction = np.array([1, 0, 0])
    R = rotation_matrix_from_lookat(direction)

    if len(points) == 0:
        depth = torch.zeros((H, W), dtype=torch.float32, device=device)
        normal = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        cap = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        result = {'cap': cap, 'depth': depth, 'normal': normal}
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = result
        return result

    pts_cam = points @ R.T
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid_mask = z > 0
    if not np.any(valid_mask):
        depth = torch.zeros((H, W), dtype=torch.float32, device=device)
        normal = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        cap = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        result = {'cap': cap, 'depth': depth, 'normal': normal}
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = result
        return result

    x, y, z = x[valid_mask], y[valid_mask], -z[valid_mask]
    u = (f * x / z) + cx
    v = (f * y / z) + cy
    in_image = (u >= 0) & (u < img_size[0]) & (v >= 0) & (v < img_size[1])
    if not np.any(in_image):
        depth = torch.zeros((H, W), dtype=torch.float32, device=device)
        normal = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        cap = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        result = {'cap': cap, 'depth': depth, 'normal': normal}
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = result
        return result

    u = u[in_image].astype(int)
    v = v[in_image].astype(int)
    z = z[in_image]
    valid_indices = np.where(valid_mask)[0][in_image]

    normals, curv, aniso, plan = compute_features_for_indices(pcd, valid_indices, radius=0.1, max_nn=30)
    good_mask = ~np.isnan(curv)
    if not np.any(good_mask):
        depth = torch.zeros((H, W), dtype=torch.float32, device=device)
        normal = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        cap = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        result = {'cap': cap, 'depth': depth, 'normal': normal}
        if pcd_cache is not None and pcd_idx is not None:
            pcd_cache[pcd_idx] = result
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

    finite = np.isfinite(depth_img)
    if not np.any(finite):
        depth_img = np.zeros((H, W), dtype=np.float32)
    else:
        depth_img[~finite] = 0.0

    cap_np = np.stack([curv_img, aniso_img, plan_img], axis=-1)
    cap_np = np.clip(cap_np * 255, 0, 255).astype(np.uint8)
    depth_np = depth_img.astype(np.float32)
    normal_np = normal_img.astype(np.float32)

    cap_tensor = torch.from_numpy(cap_np).to(device)
    depth_tensor = torch.from_numpy(depth_np).to(device)
    normal_tensor = torch.from_numpy(normal_np).to(device)

    result = {'cap': cap_tensor, 'depth': depth_tensor, 'normal': normal_tensor}
    if pcd_cache is not None and pcd_idx is not None:
        pcd_cache[pcd_idx] = result
    return result

# ========================= 健壮内参解析 =========================
def _parse_yaml_intrinsics(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        k_pattern = r'K:\s*\[\[(.*?)\]\s*\[(.*?)\]\s*\[(.*?)\]\]'
        match = re.search(k_pattern, content, re.DOTALL)
        if match:
            rows = []
            for i in range(1, 4):
                row_str = match.group(i).replace('\n', ' ').replace('\r', ' ')
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', row_str)
                if len(numbers) >= 3:
                    rows.append([float(num) for num in numbers[:3]])
            if len(rows) == 3:
                return np.array(rows)
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)
        if len(numbers) >= 9:
            return np.array([float(num) for num in numbers[:9]]).reshape(3, 3)
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
    with open(file_path, 'r') as f:
        content = f.read()
    numbers = []
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        nums = re.findall(r'[-+]?\d*\.\d+|\d+', line)
        numbers.extend([float(n) for n in nums])
    if len(numbers) >= 9:
        return np.array(numbers[:9]).reshape(3, 3)
    return None

# ========================= 模型加载 (保留 FiLM/LoRA 支持) =========================
def build_model_for_eval(model_dir: str, device: str,
                         trained_ckpt_path: str = None,
                         use_lora: bool = False,
                         lora_r: int = 8, lora_alpha: float = 2.0):
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    with open(config_path, 'r') as f:
        config = json.load(f)

    encoder_config = config.get("encoder_config", {}).copy()
    encoder_config.pop("pretrained", None)
    encoder_config.pop("weights", None)
    encoder_config["uses_torch_hub"] = True

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
        print(f"[Model] 加载预训练权重: {weights_path}")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
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

            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(new_state_dict.keys())
            common_keys = model_keys & ckpt_keys
            lora_common = [k for k in common_keys if 'lora_' in k]
            print(f"[LoRA-Diag] 模型 LoRA 参数总数: {sum(1 for k in model_keys if 'lora_' in k)}")
            print(f"[LoRA-Diag] checkpoint 中 LoRA 参数总数: {sum(1 for k in ckpt_keys if 'lora_' in k)}")
            print(f"[LoRA-Diag] 成功匹配的 LoRA 参数: {len(lora_common)}")
            
            if lora_common:
                sample_key = lora_common[0]
                diff = torch.abs(model.state_dict()[sample_key].cpu() - new_state_dict[sample_key].cpu()).max().item()
                print(f"[LoRA-Diag] 抽样 {sample_key}: 权重差异 max={diff:.6f}")
                
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
        else:
            print("[Model] 未提供训练权重，仅使用预训练模型")
    else:
        if trained_ckpt_path and os.path.exists(trained_ckpt_path):
            print(f"[Model] 加载训练权重: {trained_ckpt_path}")
            ckpt = torch.load(trained_ckpt_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '')
                new_state_dict[new_k] = v
            model.load_state_dict(new_state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model

# ========================= 数据加载 =========================
def load_eval_data(seq_root: str, img_size: int = 448, use_lidar: bool = False,
                   max_images: int = None):
    rgb_dir = os.path.join(seq_root, "rgb")
    if not os.path.exists(rgb_dir):
        raise ValueError(f"RGB 目录不存在: {rgb_dir}")

    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    img_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)
                        if f.lower().endswith(img_exts)])
    if not img_paths:
        raise ValueError("未找到图像")
    if max_images:
        img_paths = img_paths[:max_images]
    print(f"[Data] 图像: {len(img_paths)} 张")

    intrinsics_file = os.path.join(seq_root, "color_camera_intrinsics.txt")
    intrinsics = _load_intrinsics_file(intrinsics_file)
    if intrinsics is None:
        print("[Data] 内参无法解析，使用单位阵")
        intrinsics = np.eye(3, dtype=np.float32)

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

    return img_paths, intrinsics, gt_poses, gt_depths, pcd_file_list, pcd_timestamps

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

# ========================= 推理（双路径：手动 / load_images） =========================
def run_inference_batch(model, image_paths, device, use_lidar, pcd_file_list, pcd_timestamps,
                        img_size=448, view_pcd_indices=None, pcd_cache=None,
                        use_load_images=False):
    """
    支持两种图像加载模式：
      - use_load_images=False（默认）：手动 PIL 加载，与训练预处理完全一致。
      - use_load_images=True：使用原始 load_images（若需对比基线 behavior）。
    """
    target_h, target_w = img_size, img_size
    
    if use_load_images:
        # 原始路径（保留用于对比实验）
        views = load_images(
            image_paths,
            norm_type="dinov2",
            resolution_set=518,
            patch_size=14
        )
        # 确保 data_norm_type 为 list
        for view in views:
            dnt = view.get('data_norm_type')
            if isinstance(dnt, str):
                view['data_norm_type'] = [dnt]
            elif dnt is None:
                view['data_norm_type'] = ['dinov2']
        
        # 强制 resize 到 448
        orig_h, orig_w = views[0]['img'].shape[2:4]
        if orig_h != target_h or orig_w != target_w:
            for view in views:
                view['img'] = F.interpolate(
                    view['img'], size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                )
    else:
        # 修复路径：与训练一致
        views = load_images_manual(image_paths, img_size=img_size, device=device)
    
    B = views[0]['img'].shape[0]
    
    # confidence 补充（手动加载时已计算，load_images 模式需补充）
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
        needed_pcd_idxs = set(view_pcd_indices)
        pcd_features = {}

        for idx in needed_pcd_idxs:
            feat_dict = get_pcd_features(pcd_file_list[idx], device=device,
                                         pcd_cache=pcd_cache, pcd_idx=idx)

            cap = feat_dict['cap'].to(dtype=torch.float32)
            if cap.dim() == 3 and cap.shape[-1] == 3:
                cap = cap.permute(2, 0, 1)
            elif cap.dim() == 3 and cap.shape[0] == 3:
                pass
            else:
                raise ValueError(f"Unexpected cap shape: {cap.shape} for idx {idx}")
            cap_resized = F.interpolate(
                cap.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
            ).squeeze(0)

            depth = feat_dict['depth'].to(dtype=torch.float32)
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            elif depth.dim() == 3 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1).unsqueeze(0)
            elif depth.dim() == 3 and depth.shape[0] == 1:
                pass
            else:
                raise ValueError(f"Unexpected depth shape: {depth.shape} for idx {idx}")
            depth_resized = F.interpolate(
                depth.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
            ).squeeze(0)

            normal = feat_dict['normal'].to(dtype=torch.float32)
            if normal.dim() == 3 and normal.shape[-1] == 3:
                normal = normal.permute(2, 0, 1)
            elif normal.dim() == 3 and normal.shape[0] == 3:
                pass
            else:
                raise ValueError(f"Unexpected normal shape: {normal.shape} for idx {idx}")
            normal_resized = F.interpolate(
                normal.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
            ).squeeze(0)

            pcd_7ch = torch.cat([cap_resized, depth_resized, normal_resized], dim=0)
            pcd_features[idx] = pcd_7ch.unsqueeze(0)

        for view, pcd_idx in zip(views, view_pcd_indices):
            if pcd_idx not in pcd_features:
                raise KeyError(f"PCD index {pcd_idx} not found in pcd_features")
            view['pcd'] = pcd_features[pcd_idx].expand(B, -1, -1, -1).contiguous()

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
        return predictions, views
    except Exception as e:
        print(f"批次处理失败: {e}")
        return None, None

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

# ========================= 评测 =========================
def run_comprehensive_validation(predictions, gt_poses, gt_depths, gt_intrinsics, output_dir,
                                 use_lora, use_lidar, dataset_id):
    if not predictions or not gt_poses:
        print("[Eval] 无预测结果或真值，跳过评测")
        return None

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
        if idx == 0 or idx == len(gt_timestamps):
            continue

        left_ts = gt_timestamps[idx - 1]
        right_ts = gt_timestamps[idx]
        best_ts = left_ts if abs(pred_ts - left_ts) < abs(pred_ts - right_ts) else right_ts

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

    rra_thresholds = [0.5, 1.0, 1.5]
    rra_results = {}
    for tau in rra_thresholds:
        rra = np.mean(np.array(rpe_rot_errors) < tau) * 100
        rra_results[tau] = rra

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
            
            tolerance_ns = 5e7
            
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
        ("Num_Frames", stats.get('num_frames'), "-", "参与评测的有效帧数"),
    ]

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
    parser.add_argument("--seq_root", type=str, default="/add02/users/xuyh/seq1/shuangchuang_seq1_night2th")
    parser.add_argument("--model_dir", type=str, default="/home/xuyh/mapanything/")
    parser.add_argument("--trained_ckpt", type=str, default="/add02/users/xuyh/checkpoints/lora/checkpoints/best.pt")
    parser.add_argument("--output_dir", type=str, default="/add02/users/xuyh/mapanything/final_output/")
    parser.add_argument("--use_lidar", type=int, default=0)
    parser.add_argument("--use_lora", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="测试时每次输入模型的视图数。注意：若 use_lora=1 且 info_sharing 被 LoRA，建议设为训练时的 seq_len(4)")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--use_load_images", action="store_true", default=False,
                        help="使用原始 load_images（会先做 DINOv2 归一化）。默认使用手动加载（与训练一致）。")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = False

    dataset_id = parse_dataset_id(args.seq_root)
    print(f"[Info] 数据集标识: {dataset_id}")
    print(f"[Info] 图像加载模式: {'load_images (原始)' if args.use_load_images else 'manual (与训练一致)'}")

    print("\n[1/4] 加载模型...")
    model = build_model_for_eval(
        args.model_dir, device,
        trained_ckpt_path=args.trained_ckpt if args.trained_ckpt else None,
        use_lora=bool(args.use_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    print("\n[2/4] 加载数据...")
    img_paths, intrinsics, gt_poses, gt_depths, pcd_file_list, pcd_timestamps = load_eval_data(
        args.seq_root, img_size=args.img_size,
        use_lidar=bool(args.use_lidar),
        max_images=args.max_images
    )

    print("\n[3/4] 开始推理...")
    t0 = time.time()
    all_predictions = []
    num_batches = (len(img_paths) + args.batch_size - 1) // args.batch_size
    pcd_cache = {}
    pcd_timestamps_np = np.array(pcd_timestamps) if pcd_timestamps else np.array([])

    for b in range(num_batches):
        batch_paths = img_paths[b * args.batch_size:(b + 1) * args.batch_size]
        print(f"[Infer] Batch {b+1}/{num_batches}: {len(batch_paths)} 张图像")

        view_pcd_indices = []
        if args.use_lidar and len(pcd_file_list) > 0:
            for img_path in batch_paths:
                match = re.search(r'color_(\d+)', img_path)
                if not match:
                    img_ts = int(os.path.getmtime(img_path) * 1e9)
                else:
                    img_ts = int(match.group(1))
                closest_idx = np.argmin(np.abs(pcd_timestamps_np - img_ts))
                view_pcd_indices.append(closest_idx)
        else:
            view_pcd_indices = [0] * len(batch_paths)

        predictions, views = run_inference_batch(
            model, batch_paths, device,
            use_lidar=bool(args.use_lidar),
            pcd_file_list=pcd_file_list,
            pcd_timestamps=pcd_timestamps,
            img_size=args.img_size,
            view_pcd_indices=view_pcd_indices,
            pcd_cache=pcd_cache,
            use_load_images=args.use_load_images
        )

        if predictions is None:
            print(f"[Infer] Batch {b+1} 推理失败, 跳过")
            torch.cuda.empty_cache()
            pcd_cache.clear()
            continue

        batch_outputs = extract_batch_outputs(predictions, batch_paths, views)
        all_predictions.extend(batch_outputs)

        del predictions
        if views is not None:
            del views
        if 'batch_outputs' in locals():
            del batch_outputs
        torch.cuda.empty_cache()
        pcd_cache.clear()

    print(f"[3/4] 推理完成: {len(all_predictions)} 帧, 耗时 {time.time()-t0:.1f}s")

    print("\n[4/4] 开始评测...")
    stats = run_comprehensive_validation(
        all_predictions, gt_poses, gt_depths, intrinsics, args.output_dir,
        use_lora=args.use_lora, use_lidar=args.use_lidar, dataset_id=dataset_id
    )

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