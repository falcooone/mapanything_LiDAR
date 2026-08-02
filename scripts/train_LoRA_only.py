#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MapAnything LoRA-Only Training Script (Pure RGB, No LiDAR) - v3 Final
=====================================================================
全部修复整合:
  [HOTFIX-1] B初始化: randn*0.01 (非零起点,打破A/B死锁)
  [HOTFIX-2] B权重衰减: 0.0 (消除wd对B的抑制)
  [HOTFIX-3] B梯度裁剪: 5.0 (A的5倍,更宽松)
  [HOTFIX-4] 保存: 不再ema.apply_shadow覆盖,直接存真实参数
  [HOTFIX-5] 恢复: 智能检测B=0→重置B+清空B动量+跳过warmup
  [P0] 深度/点云损失移至对数空间(f_log)
  [P0] 所有单项损失权重改为0.1(对标MapAnything官方)
  [P0] RPE旋转: acos→chordal距离
  [P0] 四元数双覆盖处理
  [P1] 鲁棒回归损失(Huber)+top-N像素排除
  [P1] 世界坐标系点云损失
  [P1] LoRA+: B矩阵LR=16x A矩阵
  [P1] 编码器/预测头LR分离(10x差距)
  [P1] 梯度裁剪: A=1.0, B=5.0
  [P2] LoRA rank=8, accum_iter=4
  [P2] EMA(只用于评估参考,不覆盖保存)
"""

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
import warnings
import copy
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

from mapanything.models import MapAnything
from safetensors.torch import load_file

warnings.filterwarnings('ignore')

# Kept consistent with the LiDAR+LoRA model configuration. This script runs
# with use_lidar=False, so the channel tensor is only a construction fallback.
LIDAR_NUM_CHANNELS = 9
MAX_TIMESTAMP_TOLERANCE_SEC = 0.01


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


def _parse_camera_intrinsics_file(file_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    K = None
    width = None
    height = None
    if not file_path or not os.path.exists(file_path):
        return np.eye(3, dtype=np.float32), (0, 0)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        width_match = re.search(r"width:\s*(\d+)", content)
        height_match = re.search(r"height:\s*(\d+)", content)
        if width_match:
            width = int(width_match.group(1))
        if height_match:
            height = int(height_match.group(1))

        k_match = re.search(r"K:\s*\[\[(.*?)\]\]", content, re.DOTALL)
        if k_match:
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", k_match.group(1))
            if len(nums) >= 9:
                K = np.array([float(x) for x in nums[:9]], dtype=np.float32).reshape(3, 3)
    except Exception:
        pass

    if K is None:
        try:
            data = np.loadtxt(file_path)
            if data.shape == (3, 3):
                K = data.astype(np.float32)
            elif data.size == 9:
                K = data.reshape(3, 3).astype(np.float32)
        except Exception:
            K = np.eye(3, dtype=np.float32)

    return K.astype(np.float32), (int(width or 0), int(height or 0))


def _scale_intrinsics(
    K: np.ndarray,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> np.ndarray:
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


# ========================= 数值稳定工具函数 =========================

def f_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """MapAnything官方对数空间变换: f_log(x) = sign(x) * log(1+|x|)"""
    x = x.float()
    sign_x = torch.sign(x)
    abs_x = torch.abs(x) + eps
    return (sign_x * torch.log1p(abs_x)).to(x.dtype)


class RobustRegressionLoss(nn.Module):
    """Huber损失变体: 小误差二次惩罚, 大误差线性惩罚"""
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
    """SO(3) chordal距离: d = sqrt(2*(3-trace(R^T @ R_gt)))"""
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    trace = torch.clamp(trace, -3.0, 3.0)
    chordal_sq = torch.clamp(2.0 * (3.0 - trace), min=0.0)
    return torch.sqrt(chordal_sq + 1e-8)


def exclude_top_n_percent(loss_map: torch.Tensor, n_percent: float = 5.0, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """排除损失最高的n_percent像素"""
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


# ========================= LoRA (HOTFIX-1: B非零初始化) =========================

class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, lora_alpha: int = 8):
        super().__init__()
        self.linear = linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, r))
        # [HOTFIX-1] B不再初始化为0！使用极小随机值打破对称性
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


# ========================= EMA (HOTFIX-4: 只参考不覆盖) =========================

class ModelEMA:
    """EMA只用于评估时参考，绝不覆盖保存的真实参数"""
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


# ========================= Gradient Checkpointing =========================

def enable_gradient_checkpointing_safe(model, rank):
    import torch.utils.checkpoint as cp
    wrapped_names = []
    def _make_checkpoint_fn(orig_forward, name):
        def _forward(*args, **kwargs):
            return cp.checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)
        return _forward
    candidates = ['info_sharing', 'info_sharing_module', 'transformer', 'encoder',
                  'dense_head', 'pose_head', 'scale_head']
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

            # ---- 深度损失 (对数空间) ----
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

            # ---- 射线方向损失 ----
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

            # ---- 相机坐标系3D点损失 ----
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

            # ---- 世界坐标系点云损失 ----
            if self.w_world_pts > 0.0 and 'pts3d' in pred and view.get('gt_pose') is not None and view.get('gt_depth') is not None and gt_pts3d_cam is not None:
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

            # ---- 置信度损失 ----
            if self.w_confidence > 0.0 and 'confidence' in pred and ('depth_along_ray' in pred or 'pts3d_cam' in pred):
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

        # ---- 相对姿态估计损失 (RPE) ----
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
                    loss_t = F.smooth_l1_loss(T_rel_pred[:, :3, 3], T_rel_gt[:, :3, 3], beta=0.1)
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


# ========================= 数据集 =========================

class SeqRGBDataset(Dataset):
    def __init__(self, seq_root: str, seq_len: int = 2, stride: int = 1,
                 img_size: int = 448, tolerance: float = 0.01,
                 img_exts: tuple = None):
        super().__init__()
        self.seq_root = seq_root
        self.seq_len = seq_len
        self.stride = stride
        self.img_size = img_size
        requested_tolerance = float(tolerance)
        self.tolerance = min(
            max(requested_tolerance, 0.0),
            MAX_TIMESTAMP_TOLERANCE_SEC,
        )
        if requested_tolerance > MAX_TIMESTAMP_TOLERANCE_SEC and is_main_process(int(os.environ.get('RANK', 0))):
            print(
                f"[Data] timestamp tolerance {requested_tolerance:.6f}s exceeds the hard "
                f"limit; clamped to {MAX_TIMESTAMP_TOLERANCE_SEC:.6f}s"
            )
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

        root_intrinsics_file = os.path.join(seq_root, "color_camera_intrinsics.txt")
        root_K, root_size = _parse_camera_intrinsics_file(root_intrinsics_file)
        self.part_intrinsics = []
        for part in self.part_folders:
            part_intrinsics_file = os.path.join(seq_root, part, "color_camera_intrinsics.txt")
            if os.path.exists(part_intrinsics_file):
                self.part_intrinsics.append(_parse_camera_intrinsics_file(part_intrinsics_file))
            else:
                self.part_intrinsics.append((root_K, root_size))
        self.intrinsics = self.part_intrinsics[0][0] if self.part_intrinsics else root_K

        self.all_views_meta = []
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
                if diff <= self.tolerance:
                    self.all_views_meta.append((ipath, img_ts_ns, pidx, nearest_idx))

        if not self.all_views_meta and is_main_process(int(os.environ.get('RANK', 0))):
            raise ValueError("没有任何图像通过时间戳匹配！")
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
                T_w2c[:3, :3] = rot
                T_w2c[:3, 3] = [tx, ty, tz]
                poses[ts] = np.linalg.inv(T_w2c)
        return poses

    def _find_closest_depth(self, img_ts_ns: int, part_idx: int):
        depth_info = self.part_depths[part_idx]
        if len(depth_info['timestamps']) == 0:
            return None
        idx = np.argmin(np.abs(depth_info['timestamps'] - img_ts_ns))
        matched_ts = int(depth_info['timestamps'][idx])
        diff_ns = abs(matched_ts - img_ts_ns)
        if diff_ns <= self.tolerance * 1e9:
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
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        return d

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
            img = Image.open(ipath).convert('RGB')
            img_np_color = np.array(img)
            if not np.isfinite(img_np_color).all():
                img_np_color = np.zeros_like(img_np_color)
            img_np = img_np_color.transpose(2, 0, 1).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            if img_tensor.shape[1] != self.img_size or img_tensor.shape[2] != self.img_size:
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                                           mode='bilinear', align_corners=False).squeeze(0)
            img_gray_np = img_np_color.astype(np.float32).mean(axis=2) / 255.0
            mean = np.mean(img_gray_np)
            rms = np.sqrt(np.mean((img_gray_np - mean) ** 2))
            confidence = float(rms) if not np.isnan(rms) else 0.5
            gt_pose = torch.from_numpy(self.gt_poses_list[gt_idx])
            gt_depth = torch.zeros((1, 1, self.img_size, self.img_size), dtype=torch.float32)
            depth_file = self._find_closest_depth(img_ts_ns, pidx)
            if depth_file is not None:
                d = self._load_depth_from_path(depth_file)
                d_tensor = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
                if d_tensor.shape[-2:] != (self.img_size, self.img_size):
                    d_tensor = F.interpolate(d_tensor, size=(self.img_size, self.img_size),
                                             mode='bilinear', align_corners=False)
                d_tensor = torch.nan_to_num(d_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                gt_depth = d_tensor
            K, src_size = self.part_intrinsics[pidx]
            K_scaled = _scale_intrinsics(K, src_size, (self.img_size, self.img_size))
            gt_intrinsics = torch.from_numpy(K_scaled).unsqueeze(0)
            views.append({
                "img": img_tensor.unsqueeze(0),
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


# ========================= 模型构建 =========================
def build_model(
    model_dir: str,
    device: str,
    rank: int,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_target_mode: str = "all",
):
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
    lidar_encoder_config["in_chans"] = LIDAR_NUM_CHANNELS
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

    lora_targets = []
    if lora_target_mode not in {"all", "no_encoder", "heads_only"}:
        raise ValueError(f"Invalid lora_target_mode: {lora_target_mode}")
    if lora_target_mode == "all" and hasattr(model, 'encoder') and model.encoder is not None:
        lora_targets.append(model.encoder)
    if lora_target_mode in {"all", "no_encoder"} and hasattr(model, 'info_sharing') and model.info_sharing is not None:
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
        print(f"  -> 已注入 LoRA: {lora_layer_count} 个 Linear 层 (r={lora_r}, alpha={lora_alpha}, mode={lora_target_mode})")

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("没有任何可训练参数")
    if is_main_process(rank):
        print(f"  -> 总参数量: {total/1e6:.2f}M, 可训练(LoRA): {trainable/1e6:.2f}M")

    enable_gradient_checkpointing_safe(model, rank)
    model = model.to(device)
    return model


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
        torch.cuda.empty_cache()


# ========================= 训练循环 =========================

def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, epoch, args, rank, scheduler, ema=None):
    model.train()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)
    skipped_batches = 0
    nan_grad_batches = 0
    b_first_norm = None  # 记录首个batch后B的范数

    for batch_idx, views in enumerate(dataloader):
        for view in views:
            for key, val in view.items():
                if torch.is_tensor(val):
                    view[key] = val.to(device, non_blocking=True)

        skip_batch = False
        for i, view in enumerate(views):
            img = view.get('img')
            if img is not None and not torch.isfinite(img).all():
                if is_main_process(rank):
                    print(f"[Epoch {epoch}] batch {batch_idx}: view[{i}].img 包含NaN/Inf，跳过")
                skip_batch = True
                break
        if skip_batch:
            skipped_batches += 1
            continue

        with autocast(enabled=args.amp, dtype=torch.bfloat16):
            predictions = model(views)
            has_nan_pred = False
            for i, pred in enumerate(predictions):
                for k, v in pred.items():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():
                        has_nan_pred = True
            if has_nan_pred:
                skipped_batches += 1
                _cleanup_batch_tensors(views, predictions, None, None)
                continue
            loss, metrics = criterion(predictions, views, seq_len=args.seq_len)

        if loss is None or not torch.isfinite(loss) or not loss.requires_grad:
            _cleanup_batch_tensors(views, predictions, loss, None)
            skipped_batches += 1
            continue

        loss_val = loss.item()
        loss_scaled = loss / args.accum_iter

        if args.amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # [HOTFIX-诊断] 第0个batch后记录B范数
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

            # [HOTFIX-3] A/B分离梯度裁剪
            a_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_A' in n]
            b_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' in n]

            if args.amp:
                scaler.unscale_(optimizer)
                if a_params:
                    torch.nn.utils.clip_grad_norm_(a_params, max_norm=args.grad_clip)
                if b_params:
                    torch.nn.utils.clip_grad_norm_(b_params, max_norm=args.grad_clip * 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if a_params:
                    torch.nn.utils.clip_grad_norm_(a_params, max_norm=args.grad_clip)
                if b_params:
                    torch.nn.utils.clip_grad_norm_(b_params, max_norm=args.grad_clip * 5.0)
                optimizer.step()

            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if is_main_process(rank) and batch_idx % args.log_interval == 0:
            lr_str = "/".join([f"{g['lr']:.2e}" for g in optimizer.param_groups])
            log_str = (f"[Epoch{epoch}][{batch_idx}/{len(dataloader)}] "
                       f"LR:{lr_str} Loss:{loss_val:.4f}")
            for key in ['rpe_trans', 'rpe_rot', 'depth', 'ray', 'world_pts']:
                if key in metrics:
                    log_str += f" {key}:{metrics[key]:.4f}"
            print(log_str)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    # Epoch结束: 打印B范数诊断
    if is_main_process(rank):
        b_norms = [p.norm().item() for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
        avg_b = sum(b_norms) / len(b_norms) if b_norms else 0
        print(f"[Epoch {epoch} 结束] B平均范数: {avg_b:.4e} | "
              f"首个batch后B: {b_first_norm if b_first_norm else 'N/A'} | "
              f"跳过batch: {skipped_batches} | NaNGrad: {nan_grad_batches}")

    return total_loss / max(num_batches, 1)


# ========================= 保存 (HOTFIX-4: 不覆盖) =========================

def get_lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters()
            if 'lora_A' in k or 'lora_B' in k}


def save_checkpoint(save_model, optimizer, scheduler, scaler, epoch, best_loss, path, is_main, ema=None):
    if not is_main:
        return
    lora_state = get_lora_state_dict(save_model)
    checkpoint = {
        'epoch': epoch,
        'lora_state_dict': lora_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
        'best_loss': best_loss,
    }
    # [HOTFIX-4] EMA shadow不再覆盖真实参数，只作为参考保存
    if ema is not None:
        checkpoint['ema_shadow'] = {k: v.clone() for k, v in ema.shadow.items()}

    tmp_path = path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        # 快速验证
        b_norms = [v.norm().item() for k, v in lora_state.items() if 'lora_B' in k]
        avg_b = sum(b_norms) / len(b_norms) if b_norms else 0
        print(f"  -> 保存 ({len(lora_state)}参数) | B平均范数: {avg_b:.4e} | {path}")
    except Exception as e:
        print(f"  -> 保存失败: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ========================= 智能恢复 (HOTFIX-5) =========================

def smart_resume(model, optimizer, scheduler, scaler, ema, ckpt_path, rank, device):
    """智能恢复：保A、检测B=0→重置B、清空B动量、跳过warmup"""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return 0, float('inf')

    if is_main_process(rank):
        print(f"[HOTFIX-5] 智能恢复: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'lora_state_dict' in ckpt:
        state_dict = ckpt['lora_state_dict']
    else:
        state_dict = {k: v for k, v in ckpt.get('model_state_dict', {}).items() if 'lora_A' in k or 'lora_B' in k}

    a_state = {k: v for k, v in state_dict.items() if 'lora_A' in k}
    b_state = {k: v for k, v in state_dict.items() if 'lora_B' in k}

    # 1. 加载A（永远保留）
    model.load_state_dict(a_state, strict=False)

    # 2. 检测B状态
    b_norms = [v.norm().item() for v in b_state.values()]
    avg_b = sum(b_norms) / len(b_norms) if b_norms else 0
    b_reinit = avg_b < 1e-10

    if b_reinit:
        # B全为0 → 不加载B，让模型用新的非零初始化
        if is_main_process(rank):
            print(f"  -> [HOTFIX-5a] B全为0(avg={avg_b:.4e})，重新初始化B为小随机值")
            print(f"  -> A范数保留: {[f'{v.norm().item():.2f}' for v in a_state.values()][:4]}...")
    else:
        # B有值 → 正常加载
        model.load_state_dict(b_state, strict=False)
        if is_main_process(rank):
            print(f"  -> B非零(avg={avg_b:.4e})，正常加载")

    # 3. 优化器状态：选择性处理
    if 'optimizer_state_dict' in ckpt:
        opt_state = ckpt['optimizer_state_dict']

        if b_reinit and 'state' in opt_state:
            # [HOTFIX-5b] 清空B参数的Adam动量(exp_avg/exp_avg_sq)
            b_cleared = 0
            # 建立 param_id → shape 映射
            b_shapes = set(tuple(v.shape) for k, v in b_state.items())
            for param_id, state in opt_state['state'].items():
                if 'exp_avg' in state and 'exp_avg_sq' in state:
                    shape = tuple(state['exp_avg'].shape)
                    if shape in b_shapes and state['exp_avg'].numel() > 0:
                        # 判断是否为B参数（B是瘦高矩阵r×out，A是宽矮矩阵in×r）
                        if shape[0] <= 32 and shape[1] > 32:  # r=8, out=1536/4608/8192
                            state['exp_avg'].zero_()
                            state['exp_avg_sq'].fill_(1e-8)
                            b_cleared += 1

            optimizer.load_state_dict(opt_state)
            if is_main_process(rank):
                print(f"  -> [HOTFIX-5b] 已清空{b_cleared}个B参数的Adam动量")
        else:
            optimizer.load_state_dict(opt_state)
            if is_main_process(rank):
                print(f"  -> 优化器状态正常加载")

    # 4. 学习率调度：直接恢复（跳过warmup）
    if 'scheduler_state_dict' in ckpt and scheduler is not None and ckpt['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        current_lr = optimizer.param_groups[0]['lr']
        if is_main_process(rank):
            print(f"  -> [HOTFIX-5c] 调度器恢复，当前LR={current_lr:.2e}（跳过warmup）")

    # 5. Scaler
    if 'scaler_state_dict' in ckpt and scaler is not None and ckpt['scaler_state_dict'] is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    # 6. EMA：不恢复旧shadow，重新注册当前参数
    if ema is not None:
        ema._register(model)
        if is_main_process(rank):
            print(f"  -> [HOTFIX-5d] EMA重新初始化（不使用旧shadow）")

    epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))

    if is_main_process(rank):
        print(f"  -> 恢复 epoch {epoch}, best_loss={best_loss:.4f}")

    del ckpt, state_dict
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return epoch, best_loss


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description="MapAnything LoRA Training (v3 Final)")
    parser.add_argument("--seq_root", type=str, nargs='+',
                        default=["/add02/users/xuyh/seq1/", "/add02/users/xuyh/seq3/"])
    parser.add_argument("--model_dir", type=str, default="/home/xuyh/mapanything/")
    parser.add_argument("--output_dir", type=str, default="/add02/users/xuyh/checkpoints/32_lora/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_b_multiplier", type=float, default=16.0)
    parser.add_argument("--encoder_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=MAX_TIMESTAMP_TOLERANCE_SEC,
        help="maximum RGB/GT/depth timestamp error in seconds (hard-capped at 0.01)",
    )
    parser.add_argument("--accum_iter", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_mode", type=str, default="all",
                        choices=["all", "no_encoder", "heads_only"],
                        help="LoRA injection scope; default all matches the stronger reconstruction setting")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    args = parser.parse_args()

    rank, local_rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_per_process_memory_fraction(1.00, device)
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.backends.cudnn.benchmark = True

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    if is_main_process(rank):
        print("=" * 65)
        print("MapAnything LoRA Training v3 Final")
        print("=" * 65)
        print(f"[HOTFIX-1] B初始化: randn*0.01 (非零起点)")
        print(f"[HOTFIX-2] B权重衰减: 0.0")
        print(f"[HOTFIX-3] B梯度裁剪: {args.grad_clip * 5.0} (A={args.grad_clip})")
        print(f"[HOTFIX-4] 保存: 不再用EMA覆盖")
        print(f"[HOTFIX-5] 恢复: 智能检测B=0→重置B+清动量+跳过warmup")
        print(f"  A LR: {args.lr}, B LR: {args.lr * args.lr_b_multiplier:.2e}")
        print(f"  Encoder LR ratio: {args.encoder_lr_ratio}")
        print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"  LoRA target mode: {args.lora_target_mode}")
        print(f"  Accum: {args.accum_iter}, GradClip A/B: {args.grad_clip}/{args.grad_clip*5.0}")
        print("=" * 65)

    model = build_model(
        args.model_dir,
        device,
        rank,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_mode=args.lora_target_mode,
    )

    # [HOTFIX-2] B的weight_decay=0
    lora_A_params_encoder = []
    lora_B_params_encoder = []
    lora_A_params_head = []
    lora_B_params_head = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = 'encoder' in name
        is_B = 'lora_B' in name

        if is_encoder and is_B:
            lora_B_params_encoder.append(param)
        elif is_encoder and not is_B:
            lora_A_params_encoder.append(param)
        elif not is_encoder and is_B:
            lora_B_params_head.append(param)
        else:
            lora_A_params_head.append(param)

    param_groups = [
        {'params': lora_A_params_head, 'lr': args.lr, 'weight_decay': args.weight_decay, 'name': 'head_lora_A'},
        {'params': lora_B_params_head, 'lr': args.lr * args.lr_b_multiplier, 'weight_decay': 0.0, 'name': 'head_lora_B'},  # wd=0
        {'params': lora_A_params_encoder, 'lr': args.lr * args.encoder_lr_ratio, 'weight_decay': args.weight_decay, 'name': 'enc_lora_A'},
        {'params': lora_B_params_encoder, 'lr': args.lr * args.lr_b_multiplier * args.encoder_lr_ratio, 'weight_decay': 0.0, 'name': 'enc_lora_B'},  # wd=0
    ]
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    if is_main_process(rank):
        total_trainable = sum(len(g['params']) for g in param_groups)
        print(f"优化器: {total_trainable} 参数, {len(param_groups)} 组")
        for g in param_groups:
            print(f"  -> {g['name']}: {len(g['params'])} params, lr={g['lr']:.2e}, wd={g.get('weight_decay', 0)}")

    optimizer = AdamW(param_groups, betas=(0.9, 0.999))

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True)

    # 数据集
    if len(args.seq_root) == 1:
        dataset = SeqRGBDataset(seq_root=args.seq_root[0], seq_len=args.seq_len,
                                stride=args.stride, img_size=args.img_size, tolerance=args.tolerance)
    else:
        datasets = [SeqRGBDataset(seq_root=root, seq_len=args.seq_len, stride=args.stride,
                                  img_size=args.img_size, tolerance=args.tolerance) for root in args.seq_root]
        dataset = ConcatDataset(datasets)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                            sampler=sampler, num_workers=min(4, args.num_workers),
                            pin_memory=True, collate_fn=collate_fn, persistent_workers=False,
                            prefetch_factor=2 if args.num_workers > 0 else None)

    # 学习率调度
    steps_per_epoch = len(dataloader) // args.accum_iter
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_steps, total_steps // 2)
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # 损失函数
    criterion = MapAnythingLoss(
        w_depth=0.1, w_pose_trans=0.1, w_pose_rot=0.1, w_ray=0.1,
        w_pts3d_cam=0.1, w_world_pts=1.0, w_confidence=0.2,
        w_scale=0.1, robust_alpha=0.5, robust_c=0.05,
        top_n_percent=5.0, use_log_depth=True, use_chordal_rot=True)

    scaler = GradScaler(enabled=args.amp)
    ema = None
    if args.use_ema and is_main_process(rank):
        target_model = model.module if is_ddp else model
        ema = ModelEMA(target_model, decay=args.ema_decay)
        print(f"[EMA] 已启用, decay={args.ema_decay} (仅参考,不覆盖保存)")

    # [HOTFIX-5] 智能恢复
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        target_model = model.module if is_ddp else model
        start_epoch, best_loss = smart_resume(
            target_model, optimizer, scheduler, scaler, ema,
            args.resume, rank, device)
        start_epoch += 1  # 从下一个epoch开始

    if is_main_process(rank):
        print(f"训练: epoch {start_epoch}~{args.epochs}, steps/epoch={steps_per_epoch}")

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, dataloader, optimizer, scaler, criterion,
                                   device, epoch, args, rank, scheduler, ema)
        if is_main_process(rank):
            print(f"Epoch {epoch} 完成 | 平均损失: {avg_loss:.4f}")

            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch:03d}.pt")
                torch.cuda.empty_cache()
                save_model = model.module if is_ddp else model
                save_checkpoint(save_model, optimizer, scheduler, scaler, epoch, best_loss, ckpt_path, True, ema)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, "checkpoints", "best.pt")
                torch.cuda.empty_cache()
                save_model = model.module if is_ddp else model
                save_checkpoint(save_model, optimizer, scheduler, scaler, epoch, best_loss, best_path, True, ema)
                print(f"  -> 最佳模型 (loss={best_loss:.4f})")

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    if is_main_process(rank):
        print("训练完成!")
    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
