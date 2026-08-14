from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairInfoNCELoss(nn.Module):
    """Symmetric InfoNCE for paired RGB-LiDAR embeddings.

    This is the simplest cross-modal objective for one positive pair per sample
    and batch-wise negatives. It is the same core idea as OLIVINE's NCELoss,
    but normalized, symmetric, and masked for invalid samples.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True,
        rgb_to_lidar_weight: float = 0.3,
        lidar_to_rgb_weight: float = 0.7,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)
        self.rgb_to_lidar_weight = float(rgb_to_lidar_weight)
        self.lidar_to_rgb_weight = float(lidar_to_rgb_weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        rgb_embed: torch.Tensor,
        lidar_embed: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rgb_embed.ndim != 2 or lidar_embed.ndim != 2:
            raise ValueError("PairInfoNCELoss expects 2D tensors shaped [B, D].")
        if rgb_embed.shape != lidar_embed.shape:
            raise ValueError(
                f"rgb_embed and lidar_embed must have the same shape, got "
                f"{tuple(rgb_embed.shape)} vs {tuple(lidar_embed.shape)}"
            )

        rgb_embed = F.normalize(rgb_embed, dim=-1)
        lidar_embed = F.normalize(lidar_embed, dim=-1)

        if valid_mask is not None:
            valid_mask = valid_mask.to(dtype=torch.bool, device=rgb_embed.device)
            if valid_mask.ndim != 1 or valid_mask.shape[0] != rgb_embed.shape[0]:
                raise ValueError("valid_mask must be a 1D tensor with length B.")
            rgb_embed = rgb_embed[valid_mask]
            lidar_embed = lidar_embed[valid_mask]

        if rgb_embed.shape[0] == 0:
            return torch.zeros((), device=rgb_embed.device, dtype=rgb_embed.dtype)

        logits = rgb_embed @ lidar_embed.t()
        logits = logits / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)

        loss_rgb_to_lidar = self.ce(logits, targets)
        if not self.symmetric:
            return loss_rgb_to_lidar
        loss_lidar_to_rgb = self.ce(logits.t(), targets)
        weight_sum = self.rgb_to_lidar_weight + self.lidar_to_rgb_weight
        if weight_sum <= 0:
            return 0.5 * (loss_rgb_to_lidar + loss_lidar_to_rgb)
        return (
            self.rgb_to_lidar_weight * loss_rgb_to_lidar
            + self.lidar_to_rgb_weight * loss_lidar_to_rgb
        ) / weight_sum


class SupConLoss(nn.Module):
    """Supervised contrastive loss.

    Expected input:
        features: [B, V, D] or [B, D] if V=1
        labels:   [B]
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError("SupConLoss expects features shaped [B, V, D] or [B, D].")

        device = features.device
        labels = labels.to(device=device, dtype=torch.long).view(-1, 1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have the same batch size as features.")

        batch_size, view_count, dim = features.shape
        features = F.normalize(features, dim=-1)
        features = features.view(batch_size * view_count, dim)

        similarity = torch.matmul(features, features.t()) / self.temperature
        logits_max = similarity.max(dim=1, keepdim=True).values.detach()
        logits = similarity - logits_max

        # Mask out self-contrast.
        logits_mask = torch.ones_like(logits)
        logits_mask.fill_diagonal_(0)

        # Positive mask: same label, excluding self.
        mask = torch.eq(labels, labels.t()).float().to(device)
        mask = mask.repeat_interleave(view_count, dim=0).repeat_interleave(view_count, dim=1)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        pos_count = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
        valid = pos_count > 0
        if not valid.any():
            return torch.zeros((), device=device, dtype=features.dtype)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


@dataclass
class ContrastiveLossOutput:
    total: torch.Tensor
    rgb_to_lidar: torch.Tensor
    lidar_to_rgb: torch.Tensor
