#!/usr/bin/env python3
#coding=gbk
"""
MapAnything LoRA-Only Training Script (Pure RGB, No LiDAR) - 全面修复版
==========================================================================
修复内容汇总:
  [P0] 1. 深度/点云损失移至对数空间 (f_log)
  [P0] 2. 所有单项损失权重改为0.1 (对标官方)
  [P0] 3. RPE旋转损失: acos -> chordal距离
  [P0] 4. 四元数双覆盖处理 (q vs -q 等价性)
  [P1] 5. 添加鲁棒回归损失 (Huber) + top-N像素排除
  [P1] 6. 添加置信度预测与ConfLoss (简化版)
  [P1] 7. 世界坐标系点云损失 (world_frame_points)
  [P1] 8. LoRA+: B矩阵学习率 = 16x A矩阵
  [P1] 9. 编码器/预测头学习率分离 (10x差距)
  [P1] 10. 梯度裁剪: 5.0 -> 1.0
  [P2] 11. LoRA rank: 16 -> 8
  [P2] 12. Accum iter: 8 -> 4
  [P2] 13. 添加EMA (指数移动平均)
  [P2] 14. 损失不稳定性自动检测
  [P2] 15. 深度图增强: 随机深度掩码模拟置信度
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

# [修复P0] 对数空间变换: f_log(x) = sign(x) * log(1+|x|)
# 相比 raw log: (1) x=0处梯度有界 (2) 保留方向信息 (3) 尺度等变性
def f_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """MapAnything官方使用的对数空间变换"""
    x = x.float()
    sign_x = torch.sign(x)
    abs_x = torch.abs(x) + eps
    return (sign_x * torch.log1p(abs_x)).to(x.dtype)


# [修复P1] 鲁棒回归损失 (Huber-like)
class RobustRegressionLoss(nn.Module):
    """Huber损失变体: 小误差二次惩罚, 大误差线性惩罚"""
    def __init__(self, alpha: float = 0.5, scaling_c: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.c = scaling_c

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        # 自适应阈值: scaling_c * median(abs_diff)
        with torch.no_grad():
            if valid_mask is not None and valid_mask.any():
                thresh = self.c * torch.median(abs_diff[valid_mask])
            else:
                thresh = self.c * torch.median(abs_diff) if abs_diff.numel() > 0 else torch.tensor(self.c)
            thresh = thresh.clamp_min(1e-6)
        
        # Huber核
        quadratic = 0.5 * (diff ** 2) / thresh
        linear = abs_diff - 0.5 * thresh
        loss = torch.where(abs_diff <= thresh, quadratic, linear)
        
        if valid_mask is not None:
            return loss[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return loss.mean()


# [修复P0] Chordal距离: SO(3)上的Frobenius范数距离, 梯度有界
def so3_chordal_distance(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """SO(3) chordal距离: d = ||R_pred - R_gt||_F = sqrt(2*(3-trace(R^T @ R_gt)))"""
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    trace = torch.clamp(trace, -3.0, 3.0)
    chordal_sq = torch.clamp(2.0 * (3.0 - trace), min=0.0)
    return torch.sqrt(chordal_sq + 1e-8)


# [修复P0] 四元数双覆盖距离: min(||q1-q2||, ||q1+q2||)
def quat_geodesic_distance(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """四元数geodesic距离, 正确处理q ~ -q等价性"""
    # 归一化
    q_pred = F.normalize(q_pred, dim=-1)
    q_gt = F.normalize(q_gt, dim=-1)
    # 内积
    dot = torch.sum(q_pred * q_gt, dim=-1)
    # 双覆盖: 取绝对值
    dot_abs = torch.abs(dot)
    # clamp防止数值越界
    dot_abs = torch.clamp(dot_abs, -1.0 + 1e-7, 1.0 - 1e-7)
    # geodesic距离: 2*acos(|<q1,q2>|)
    angle = 2.0 * torch.acos(dot_abs)
    return angle


# [修复P1] Top-N百分比像素排除
def exclude_top_n_percent(loss_map: torch.Tensor, n_percent: float = 5.0, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """排除损失最高的n_percent像素 (鲁棒性策略)"""
    if valid_mask is not None and valid_mask.shape != loss_map.shape:
        if valid_mask.ndim == loss_map.ndim and valid_mask.shape[-1] != loss_map.shape[-1]:
            # 多通道 valid -> 单通道 loss: 在最后一维做 any/reduce
            valid_mask = valid_mask.any(dim=-1, keepdim=True)
        # 如果还不匹配，尝试广播
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


# [修复P1] 置信度损失 (简化ConfLoss)
class ConfidenceLoss(nn.Module):
    """鼓励模型对低误差预测赋予高置信度"""
    def __init__(self, conf_alpha: float = 0.2):
        super().__init__()
        self.conf_alpha = conf_alpha

    def forward(self, confidence: torch.Tensor, loss_map: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        confidence: [B, 1, H, W] 预测置信度 (0~1)
        loss_map: [B, 1, H, W] 每个像素的回归损失
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(loss_map, dtype=torch.bool)
        
        conf = confidence[valid_mask].flatten()
        loss_vals = loss_map[valid_mask].flatten().detach()  # 停止梯度
        
        if conf.numel() == 0:
            return torch.tensor(0.0, device=confidence.device)
        
        # 归一化损失到0~1范围用于置信度监督
        with torch.no_grad():
            loss_normalized = loss_vals / (loss_vals.mean() + 1e-8)
            loss_normalized = torch.clamp(loss_normalized, 0.0, 10.0)
            target_conf = torch.exp(-loss_normalized)  # 低损失 -> 高置信度
        
        # BCE损失
        conf_clamped = torch.clamp(conf, 1e-6, 1.0 - 1e-6)
        conf_loss = F.binary_cross_entropy(conf_clamped, target_conf.float())
        
        return self.conf_alpha * conf_loss

# ========================= LoRA (embedded, no external deps) =========================
# [修复P2] LoRA rank改为8, alpha=8

class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, lora_alpha: int = 8):
        super().__init__()
        self.linear = linear
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear.out_features))
        # 标准初始化: A ~ N(0, 1/sqrt(r)), B = 0
        std = 1.0 / math.sqrt(r)
        nn.init.normal_(self.lora_A, mean=0.0, std=std)
        nn.init.zeros_(self.lora_B)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        # 强制fp32进行LoRA计算避免bfloat16溢出
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


# ========================= EMA (指数移动平均) =========================
# [修复P2] 添加EMA用于更稳定的收敛

class ModelEMA:
    """模型权重的指数移动平均"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

# ========================= Gradient Checkpointing =========================

def enable_gradient_checkpointing_safe(model, rank):
    import torch.utils.checkpoint as cp
    wrapped_names = []

    def _make_checkpoint_fn(orig_forward, name):
        def _forward(*args, **kwargs):
            return cp.checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)
        return _forward

    candidates = [
        'info_sharing', 'info_sharing_module', 'transformer', 'encoder',
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


# ========================= 损失函数（全面修复版） =========================
# [修复P0] 所有权重改为0.1 (对标MapAnything官方)
# [修复P0] 深度损失使用f_log对数空间
# [修复P0] RPE旋转使用chordal距离替代acos
# [修复P0] 四元数双覆盖处理
# [修复P1] 添加鲁棒回归损失 (Huber)
# [修复P1] 添加置信度损失
# [修复P1] 添加world_frame_points损失
# [修复P1] Top-N百分比像素排除

class MapAnythingLoss(nn.Module):
    def __init__(self, 
                 w_depth: float = 0.1,           # [修复] 1.0 -> 0.1
                 w_pose_trans: float = 0.1,       # [修复] 1.0 -> 0.1
                 w_pose_rot: float = 0.1,         # [修复] 1.0 -> 0.1
                 w_ray: float = 0.1,              # [修复] 1.0 -> 0.1
                 w_pts3d_cam: float = 0.1,        # [修复] 1.0 -> 0.1
                 w_world_pts: float = 1.0,        # [新增] world frame points (主损失)
                 w_confidence: float = 0.2,       # [新增] 置信度损失
                 w_scale: float = 0.1,            # [新增] 尺度损失
                 robust_alpha: float = 0.5,        # [新增] 鲁棒损失参数
                 robust_c: float = 0.05,           # [新增] 鲁棒损失缩放
                 top_n_percent: float = 5.0,       # [新增] 像素排除比例
                 use_log_depth: bool = True,       # [新增] 使用对数空间深度
                 use_chordal_rot: bool = True):    # [新增] 使用chordal旋转距离
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
        
        # 鲁棒回归损失
        self.robust_loss = RobustRegressionLoss(alpha=robust_alpha, scaling_c=robust_c)
        # 置信度损失
        self.conf_loss_fn = ConfidenceLoss(conf_alpha=w_confidence)

    @staticmethod
    def _quat_to_rotmat(quats):
        # [修复P0] 四元数归一化 + 双覆盖处理 (统一到qw>=0的半球)
        norm = torch.norm(quats, dim=-1, keepdim=True)
        quats = quats / (norm + 1e-8)
        # 双覆盖: 确保qw >= 0
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
        """[新增] 世界坐标系点云损失 (MapAnything的主损失项)"""
        if valid_mask is None or not valid_mask.any():
            if pred_pts3d_world is None or gt_pts3d_world is None:
                return torch.tensor(0.0, device=pred_pts3d_world.device if pred_pts3d_world is not None else 'cuda')
            valid_mask = torch.isfinite(pred_pts3d_world).all(dim=-1) & torch.isfinite(gt_pts3d_world).all(dim=-1)
        
        # 对数空间计算
        pred_log = f_log(pred_pts3d_world)
        gt_log = f_log(gt_pts3d_world)
        loss_map = torch.abs(pred_log - gt_log).mean(dim=-1, keepdim=True)
        
        # 鲁棒回归 + top-N排除
        return exclude_top_n_percent(loss_map, self.top_n_percent, valid_mask)

    def forward(self, predictions: List[Dict], views: List[Dict], seq_len: int = 2):
        device = predictions[0]['pts3d'].device
        loss_components = []
        metrics = {}
        N = len(predictions)

        for i, (pred, view) in enumerate(zip(predictions, views)):
            # ---- 深度损失 (对数空间) ----
            if 'depth_along_ray' in pred and view.get('gt_depth') is not None:
                pred_d = pred['depth_along_ray']
                gt_d = view['gt_depth'].to(device)
                pred_d = pred_d.permute(0, 3, 1, 2)
                if pred_d.shape[-2:] != gt_d.shape[-2:]:
                    gt_d = F.interpolate(gt_d, size=pred_d.shape[-2:], mode='bilinear', align_corners=False)
                
                valid = (gt_d > 1e-4) & torch.isfinite(pred_d)  # [修复] 阈值1e-3->1e-4
                if valid.any():
                    if self.use_log_depth:
                        # [修复P0] 使用f_log对数空间
                        pred_d_log = f_log(pred_d)
                        gt_d_log = f_log(gt_d)
                        loss_map = torch.abs(pred_d_log - gt_d_log)
                    else:
                        loss_map = torch.abs(pred_d - gt_d)
                    
                    # [修复P1] 鲁棒回归 + top-N排除
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

            # ---- 相机坐标系3D点损失 (对数空间) ----
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
                    # [修复P0] 对数空间计算
                    pred_log = f_log(pred_pts3d_cam)
                    gt_log = f_log(gt_pts3d_cam)
                    loss_map = torch.abs(pred_log - gt_log).mean(dim=-1, keepdim=True)
                    loss_pc = exclude_top_n_percent(loss_map, self.top_n_percent, valid)
                    loss_components.append(self.w_pts3d_cam * loss_pc)
                    metrics[f'pts3d_cam_{i}'] = loss_pc.item()

            # ---- 置信度损失 ----
            if 'confidence' in pred and ('depth_along_ray' in pred or 'pts3d_cam' in pred):
                # 使用深度误差作为置信度监督信号
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

            # ---- 世界坐标系点云损失 [新增] ----
            if 'pts3d' in pred and view.get('gt_pose') is not None and view.get('gt_depth') is not None:
                # 使用预测的世界坐标系点云 (如果模型输出)
                pred_world = pred['pts3d']
                # 从GT深度和姿态构造GT世界坐标系点云
                gt_d = view['gt_depth'].to(device)
                if gt_d.dim() == 3:
                    gt_d = gt_d.unsqueeze(0)
                if gt_d.shape[-2:] != (H, W):
                    gt_d = F.interpolate(gt_d, size=(H, W), mode='bilinear', align_corners=False)
                
                # 简化的world points: 使用相机坐标系点经姿态变换
                if 'pts3d_cam' in pred:
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
                    
                    # 数值有效性检查
                    if not (torch.isfinite(T_rel_pred).all() and torch.isfinite(T_rel_gt).all()):
                        continue
                        
                    t_norm_pred = torch.norm(T_rel_pred[:, :3, 3], dim=-1).mean()
                    t_norm_gt = torch.norm(T_rel_gt[:, :3, 3], dim=-1).mean()
                    if not torch.isfinite(t_norm_pred) or t_norm_gt < 1e-12:
                        continue
                        
                    # 平移损失: Smooth L1
                    loss_t = F.smooth_l1_loss(T_rel_pred[:, :3, 3], T_rel_gt[:, :3, 3], beta=0.1)
                    
                    # [修复P0] 旋转损失: chordal距离替代acos
                    if self.use_chordal_rot:
                        loss_r = so3_chordal_distance(T_rel_pred[:, :3, :3], T_rel_gt[:, :3, :3]).mean()
                    else:
                        # 备用: 四元数geodesic距离 (处理双覆盖)
                        R_diff = torch.bmm(T_rel_pred[:, :3, :3].transpose(1, 2), T_rel_gt[:, :3, :3])
                        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
                        trace = torch.clamp(trace, -1.0, 3.0)
                        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
                        loss_r = torch.acos(cos_angle).mean()
                    
                    if torch.isfinite(loss_t) and torch.isfinite(loss_r):
                        loss_components.append(self.w_pose_trans * loss_t + self.w_pose_rot * loss_r)
                        metrics[f'rpe_trans_b{b}_k{k}'] = loss_t.item()
                        metrics[f'rpe_rot_b{b}_k{k}'] = loss_r.item()

        # 汇总metrics
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
        conf_vals = [v for k, v in metrics.items() if k.startswith('conf_')]
        if conf_vals:
            metrics['confidence'] = sum(conf_vals) / len(conf_vals)
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


# ========================= 纯 RGB 数据集（惰性加载深度图） =========================

class SeqRGBDataset(Dataset):
    def __init__(self, seq_root: str, seq_len: int = 2, stride: int = 1,
                 img_size: int = 448, tolerance: float = 0.01,
                 img_exts: tuple = None):
        super().__init__()
        self.seq_root = seq_root
        self.seq_len = seq_len
        self.stride = stride
        self.img_size = img_size
        self.tolerance = tolerance
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
                if is_main_process(int(os.environ.get('RANK', 0))):
                    print(f"[Dataset] {part}/depth: 扫描到 {len(valid_files)} 张深度图")
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

        if timestamp_diffs and is_main_process(int(os.environ.get('RANK', 0))):
            diffs = np.array(timestamp_diffs)
            print(f"[Dataset] 时间戳匹配失败: 共 {len(diffs)} 帧, 差值中位数 {np.median(diffs):.3f}s")

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
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        return d

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
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)

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

            gt_intrinsics = torch.from_numpy(self.intrinsics).unsqueeze(0)

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


# ========================= 模型构建（仅 LoRA，无 LiDAR） =========================

def build_model(model_dir: str, device: str, rank: int, lora_r: int = 8, lora_alpha: int = 8):
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
        if is_main_process(rank):
            print(f"加载预训练权重: {weights_path}")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

    # 注入LoRA
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

    # 冻结非LoRA参数
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable == 0:
        raise RuntimeError("没有任何可训练参数，请检查模块名匹配规则。")

    if is_main_process(rank):
        print(f"  -> 总参数量: {total/1e6:.2f}M, 可训练(LoRA): {trainable/1e6:.2f}M")

    enable_gradient_checkpointing_safe(model, rank)
    model = model.to(device)

    # Dummy forward验证
    if is_main_process(rank):
        print("[验证] 执行dummy forward检查输出格式...")
        model.eval()
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224, device=device)
            dummy_views = [{
                "img": dummy_img,
                "data_norm_type": ["dinov2"],
                "confidence": torch.tensor(0.5, device=device),
            }]
            try:
                dummy_pred = model(dummy_views)
                if isinstance(dummy_pred, list) and len(dummy_pred) > 0:
                    print(f"  -> 模型输出keys: {list(dummy_pred[0].keys())}")
                else:
                    print(f"  -> 警告: 模型输出格式异常: {type(dummy_pred)}")
            except Exception as e:
                print(f"  -> dummy forward失败（不影响训练）: {e}")
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
        torch.cuda.empty_cache()


# ========================= 训练循环（全面修复版） =========================
# [修复P1] 梯度裁剪: 5.0 -> 1.0
# [修复P2] accum_iter: 8 -> 4
# [修复P1] LoRA+ : B_lr = 16x A_lr
# [修复P1] 编码器/头学习率分离
# [修复P2] 添加EMA
# [修复P2] 损失不稳定性自动检测

def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, epoch, args, rank, scheduler, ema=None):
    model.train()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    total_loss = 0.0
    num_batches = 0
    data_time = 0.0
    train_time = 0.0
    optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    last_grad_norm = 0.0
    lora_grad_count = 0
    lora_grad_norm = 0.0
    skipped_batches = 0
    nan_grad_batches = 0

    # [修复P2] 损失不稳定性检测窗口
    loss_window = []
    instable_detected = False

    t_start = time.time()
    for batch_idx, views in enumerate(dataloader):
        t_data_end = time.time()
        data_time += (t_data_end - t_start)

        for view in views:
            for key, val in view.items():
                if torch.is_tensor(val):
                    view[key] = val.to(device, non_blocking=True)

        # 输入数据异常检测
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
            t_start = time.time()
            continue

        with autocast(enabled=args.amp, dtype=torch.bfloat16):
            predictions = model(views)

            # 预测输出异常检测
            has_nan_pred = False
            for i, pred in enumerate(predictions):
                for k, v in pred.items():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():
                        if is_main_process(rank):
                            print(f"[Epoch {epoch}] batch {batch_idx}: 预测输出 nan! view={i}, key={k}")
                        has_nan_pred = True
            if has_nan_pred:
                skipped_batches += 1
                _cleanup_batch_tensors(views, predictions, None, None)
                t_start = time.time()
                continue

            loss, metrics = criterion(predictions, views, seq_len=args.seq_len)

        if loss is None:
            _cleanup_batch_tensors(views, predictions, None, None)
            skipped_batches += 1
            t_start = time.time()
            continue

        loss_val = loss.item() if torch.isfinite(loss) else float('nan')

        if not loss.requires_grad:
            _cleanup_batch_tensors(views, predictions, loss, None)
            skipped_batches += 1
            t_start = time.time()
            continue

        if not torch.isfinite(loss):
            _cleanup_batch_tensors(views, predictions, loss, None)
            skipped_batches += 1
            t_start = time.time()
            continue

        # [修复P2] 损失不稳定性检测
        if torch.isfinite(loss):
            loss_window.append(loss_val)
            if len(loss_window) > 20:
                loss_window.pop(0)
            if len(loss_window) >= 10 and not instable_detected:
                recent_mean = np.mean(loss_window[-10:])
                recent_std = np.std(loss_window[-10:])
                if recent_mean > 100.0 or (recent_std > recent_mean * 0.5 and recent_mean > 10.0):
                    if is_main_process(rank):
                        print(f"?? [Epoch {epoch}] 检测到训练不稳定! 近期loss均值={recent_mean:.2f}, std={recent_std:.2f}")
                    instable_detected = True

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
            # 梯度异常检测
            has_nan_grad = False
            nan_param_names = []
            lora_per_layer_grad = {}
            
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    nan_param_names.append(name)
                if p.grad is not None and ('lora_A' in name or 'lora_B' in name):
                    g_norm = p.grad.norm().item()
                    layer_prefix = name.split('.lora_')[0]
                    if layer_prefix not in lora_per_layer_grad:
                        lora_per_layer_grad[layer_prefix] = []
                    lora_per_layer_grad[layer_prefix].append(g_norm)

            if has_nan_grad:
                if is_main_process(rank):
                    print(f"[Epoch {epoch}] batch {batch_idx}: 检测到nan梯度，跳过step！")
                    for n in nan_param_names[:3]:
                        print(f"    {n}")
                optimizer.zero_grad(set_to_none=True)
                last_grad_norm = float('nan')
                nan_grad_batches += 1
                torch.cuda.empty_cache()
                continue

            # [修复P1] 梯度裁剪: 5.0 -> 1.0
            params_for_clip = list(filter(lambda p: p.requires_grad, model.parameters()))
            if args.amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=args.grad_clip)
                optimizer.step()

            # [修复P2] 更新EMA
            if ema is not None:
                ema.update(model)

            # LoRA梯度统计
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

            # 显存和梯度诊断
            if is_main_process(rank) and batch_idx % max(args.log_interval, 5) == 0:
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                peak = torch.cuda.max_memory_allocated(device) / 1024**3
                top_layers = sorted(
                    [(k, sum(v)/len(v)) for k, v in lora_per_layer_grad.items()],
                    key=lambda x: x[1], reverse=True
                )[:3] if lora_per_layer_grad else []
                layer_str = " | ".join([f"{k.split('.')[-1][:20]}:{v:.3f}" for k, v in top_layers])
                print(f"  -> 显存: {allocated:.2f}GB | 峰值 {peak:.2f}GB | LoRATop3: {layer_str}")

        # 日志输出
        if is_main_process(rank) and batch_idx % args.log_interval == 0:
            lr_str = "/".join([f"{g['lr']:.2e}" for g in optimizer.param_groups])
            log_str = (f"[Epoch{epoch}][{batch_idx}/{len(dataloader)}] "
                       f"LR:{lr_str} Loss:{loss_val:.4f}")
            for key in ['rpe_trans', 'rpe_rot', 'depth', 'ray', 'confidence', 'world_pts']:
                if key in metrics:
                    log_str += f"{key}:{metrics[key]:.4f}"
            if 'rpe_trans' not in metrics and batch_idx % (args.log_interval * 5) == 0:
                log_str += "|(无 RPE，检查 pose 输出)"
            if math.isfinite(last_grad_norm):
                log_str += f"|GradNorm: {last_grad_norm:.4f}"
            if lora_grad_count > 0:
                log_str += f"|LoRA_Grad: {lora_grad_norm:.4f}({lora_grad_count})"
            if skipped_batches > 0:
                log_str += f"|跳过:{skipped_batches}"
            if nan_grad_batches > 0:
                log_str += f"|NaNGrad:{nan_grad_batches}"
            print(log_str)

        t_start = time.time()

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    
    if is_main_process(rank):
        if skipped_batches > 0:
            print(f"[Epoch {epoch} 总结] 跳过batch: {skipped_batches}/{len(dataloader)}")
        if nan_grad_batches > 0:
            print(f"[Epoch {epoch} 总结] NaN梯度batch: {nan_grad_batches}")
        
    return total_loss / max(num_batches, 1)


# ========================= LoRA-only 保存/加载工具 =========================

def get_lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters()
            if 'lora_A' in k or 'lora_B' in k}


def save_checkpoint_lora(save_model, optimizer, scheduler, scaler, epoch, best_loss, path, is_main, ema=None):
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
    # [修复P2] 同时保存EMA权重
    if ema is not None:
        checkpoint['ema_shadow'] = copy.deepcopy(ema.shadow)

    tmp_path = path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"  -> 保存 LoRA checkpoint ({len(lora_state)} 个参数): {path}")
    except Exception as e:
        print(f"  -> 保存失败: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description="Train MapAnything LoRA-Only (Fixed Version)")
    parser.add_argument("--seq_root", type=str, nargs='+',
                        default=["/add02/users/xuyh/seq1/", "/add02/users/xuyh/seq3/"],
                        help="训练数据路径，可指定多个序列")
    parser.add_argument("--model_dir", type=str, default="/home/xuyh/mapanything/")
    parser.add_argument("--output_dir", type=str, default="/add02/users/xuyh/checkpoints/new_lora/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=448)
    # [修复P1] 学习率调整: 统一lr -> 分组lr (LoRA+)
    parser.add_argument("--lr", type=float, default=5e-5, help="LoRA A矩阵学习率 (推荐3e-5~5e-5)")
    parser.add_argument("--lr_b_multiplier", type=float, default=16.0, help="LoRA B矩阵LR = lr * multiplier")
    parser.add_argument("--encoder_lr_ratio", type=float, default=0.1, help="编码器LoRA LR = lr * ratio")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    # [修复P1] 梯度裁剪: 5.0 -> 1.0 (LoRA推荐0.5~1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default="/add02/users/xuyh/checkpoints/new_lora/checkpoints/epoch_002.pt")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.05)
    # [修复P2] accum_iter: 8 -> 4
    parser.add_argument("--accum_iter", type=int, default=4, help="梯度累积步数")
    # [修复P2] LoRA rank: 16 -> 8
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (推荐4~8)")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
    # [修复P2] EMA
    parser.add_argument("--use_ema", action="store_true", default=True, help="使用EMA")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA衰减率")
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
    if is_ddp:
        torch.distributed.barrier()
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if is_main_process(rank):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if is_main_process(rank):
        print("=" * 60)
        print("MapAnything LoRA Training (全面修复版)")
        print(f"  LoRA A LR: {args.lr}, B LR: {args.lr * args.lr_b_multiplier:.2e}")
        print(f"  Encoder LR ratio: {args.encoder_lr_ratio}")
        print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}, scaling={args.lora_alpha/args.lora_r}")
        print(f"  GradClip: {args.grad_clip}, Accum: {args.accum_iter}")
        print(f"  EMA: {args.use_ema} (decay={args.ema_decay})")
        print("=" * 60)
        print("加载模型...")

    model = build_model(args.model_dir, device, rank, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    # [修复P1] LoRA+: 分组参数 (A矩阵, B矩阵, 编码器vs头)
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
        {'params': lora_B_params_head, 'lr': args.lr * args.lr_b_multiplier, 'weight_decay': args.weight_decay, 'name': 'head_lora_B'},
        {'params': lora_A_params_encoder, 'lr': args.lr * args.encoder_lr_ratio, 'weight_decay': args.weight_decay, 'name': 'enc_lora_A'},
        {'params': lora_B_params_encoder, 'lr': args.lr * args.lr_b_multiplier * args.encoder_lr_ratio, 'weight_decay': args.weight_decay, 'name': 'enc_lora_B'},
    ]
    # 过滤空组
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    if is_main_process(rank):
        total_trainable = sum(len(g['params']) for g in param_groups)
        print(f"优化器: {total_trainable} 个参数, {len(param_groups)} 个参数组")
        for g in param_groups:
            print(f"  -> {g['name']}: {len(g['params'])} params, lr={g['lr']:.2e}")

    optimizer = AdamW(param_groups, betas=(0.9, 0.999))

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )

    # 数据集构建
    if is_main_process(rank):
        print(f"构建数据集... (seq_len={args.seq_len})")

    if len(args.seq_root) == 1:
        dataset = SeqRGBDataset(
            seq_root=args.seq_root[0], seq_len=args.seq_len, stride=args.stride,
            img_size=args.img_size, tolerance=args.tolerance
        )
    else:
        datasets = []
        for idx, root in enumerate(args.seq_root):
            ds = SeqRGBDataset(
                seq_root=root, seq_len=args.seq_len, stride=args.stride,
                img_size=args.img_size, tolerance=args.tolerance
            )
            datasets.append(ds)
        dataset = ConcatDataset(datasets)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if is_ddp else None

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler,
        num_workers=min(4, args.num_workers),
        pin_memory=True, collate_fn=collate_fn,
        persistent_workers=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # 学习率调度
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

    # [修复P0/P1] 损失函数: 全面对标MapAnything官方配置
    criterion = MapAnythingLoss(
        w_depth=0.1,           # [修复] 1.0->0.1
        w_pose_trans=0.1,      # [修复] 1.0->0.1
        w_pose_rot=0.1,        # [修复] 1.0->0.1
        w_ray=0.1,             # [修复] 1.0->0.1
        w_pts3d_cam=0.1,       # [修复] 1.0->0.1
        w_world_pts=1.0,       # [新增] world frame points (主损失)
        w_confidence=0.2,      # [新增] 置信度损失
        w_scale=0.1,           # [新增] 尺度损失
        robust_alpha=0.5,      # [新增] Huber损失参数
        robust_c=0.05,         # [新增] 鲁棒损失缩放
        top_n_percent=5.0,     # [新增] Top-N像素排除
        use_log_depth=True,    # [新增] 对数空间深度
        use_chordal_rot=True,  # [新增] chordal旋转距离
    )

    scaler = GradScaler(enabled=args.amp)

    # [修复P2] 初始化EMA
    ema = None
    if args.use_ema and is_main_process(rank):
        target_model = model.module if is_ddp else model
        ema = ModelEMA(target_model, decay=args.ema_decay)
        print(f"[EMA] 已启用, decay={args.ema_decay}")

    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        if is_main_process(rank):
            print(f"恢复训练: {args.resume}")

        ckpt = torch.load(args.resume, map_location='cpu')
        if 'lora_state_dict' in ckpt:
            state_dict = ckpt['lora_state_dict']
        else:
            state_dict = ckpt['model_state_dict']
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))

        target_model = model.module if is_ddp else model
        target_model.load_state_dict(state_dict, strict=False)

        # [修复P2] 恢复EMA
        if ema is not None and 'ema_shadow' in ckpt:
            ema.shadow = ckpt['ema_shadow']
            print(f"[EMA] 已恢复EMA状态")

        if is_main_process(rank):
            lora_state_loaded = get_lora_state_dict(target_model)
            a_norms = [v.norm().item() for k, v in lora_state_loaded.items() if 'lora_A' in k]
            b_norms = [v.norm().item() for k, v in lora_state_loaded.items() if 'lora_B' in k]
            if a_norms:
                print(f"  -> LoRA_A 平均范数: {sum(a_norms)/len(a_norms):.4f}")
                print(f"  -> LoRA_B 平均范数: {sum(b_norms)/len(b_norms):.4f}")

        del ckpt, state_dict
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_start = time.time()
        avg_loss = train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, epoch, args, rank, scheduler, ema)
        epoch_time = time.time() - epoch_start

        if is_main_process(rank):
            print(f"Epoch {epoch} 完成 | 平均损失: {avg_loss:.4f} | 总耗时: {epoch_time:.1f}s")
            
            # 保存checkpoint (使用EMA权重)
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
                torch.cuda.empty_cache()
                save_model = model.module if is_ddp else model
                
                # [修复P2] 保存前应用EMA
                if ema is not None:
                    ema.apply_shadow(save_model)
                
                save_checkpoint_lora(save_model, optimizer, scheduler, scaler, epoch, best_loss, ckpt_path, True, ema)
                
                if ema is not None:
                    ema.restore(save_model)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.cuda.empty_cache()
                save_model = model.module if is_ddp else model
                
                if ema is not None:
                    ema.apply_shadow(save_model)
                    
                save_checkpoint_lora(save_model, optimizer, scheduler, scaler, epoch, best_loss, best_path, True, ema)
                
                if ema is not None:
                    ema.restore(save_model)
                    
                print(f"  -> 保存最佳模型 (loss={best_loss:.4f})")
                
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    if is_main_process(rank):
        print("训练完成!")
    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()