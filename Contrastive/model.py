from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mapanything.models import MapAnything
from uniception.models.encoders import ViTEncoderInput, ViTEncoderNonImageInput

LIDAR_NUM_CHANNELS = 9


def _strip_incompatible_keys(state_dict: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]):
    return {
        k: v
        for k, v in state_dict.items()
        if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
    }


def build_mapanything_from_dir(
    model_dir: str | Path,
    device: Optional[torch.device] = None,
) -> MapAnything:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    encoder_config = copy.deepcopy(config.get("encoder_config", {}))
    encoder_config.pop("pretrained", None)
    encoder_config.pop("weights", None)
    encoder_config["uses_torch_hub"] = False

    geometric_input_config = copy.deepcopy(config.get("geometric_input_config", {}))
    lidar_encoder_config = geometric_input_config.get("lidars_encoder_config", {})
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

    model = MapAnything(
        name=config.get("name", "mapanything"),
        encoder_config=encoder_config,
        info_sharing_config=config.get("info_sharing_config", {}),
        pred_head_config=config.get("pred_head_config", {}),
        geometric_input_config=geometric_input_config,
        pretrained_checkpoint_path=None,
        torch_hub_force_reload=False,
        use_register_tokens_from_encoder=bool(
            config.get("use_register_tokens_from_encoder", False)
        ),
        info_sharing_mlp_layer_str=config.get(
            "info_sharing_mlp_layer_str",
            "swiglufused",
        ),
    )

    if weights_path.exists():
        state_dict = None
        try:
            from safetensors.torch import load_file  # type: ignore

            state_dict = load_file(str(weights_path))
        except Exception as exc:
            warnings.warn(
                f"Skip loading pretrained weights from {weights_path} because safetensors is unavailable "
                f"or failed to load: {exc}",
                RuntimeWarning,
            )

        if state_dict is not None:
            model_state = model.state_dict()
            filtered_state = _strip_incompatible_keys(state_dict, model_state)
            model.load_state_dict(filtered_state, strict=False)

    if device is not None:
        model = model.to(device)
    return model


def _pool_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
    if features.ndim == 3:
        # Common cases:
        #   [B, N, C] token map -> pool over N
        #   [B, C, N] token map -> pool over N
        if features.shape[-1] >= features.shape[1]:
            return features.mean(dim=1)
        return features.mean(dim=2)
    if features.ndim == 2:
        return features
    raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        out_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)


class ContrastivePairModel(nn.Module):
    """RGB-DINO and LiDAR-ResNet contrastive wrapper.

    The underlying MapAnything model is only used as an encoder container; the
    original project code does not need to be changed.
    """

    def __init__(
        self,
        model: MapAnything,
        proj_dim: int = 256,
        proj_hidden_dim: int = 1024,
        share_projector: bool = False,
        train_rgb_encoder: bool = True,
        train_lidar_encoder: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.model = model
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.rgb_proj = ProjectionHead(
            in_dim=model.encoder.enc_embed_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_dim,
        )
        if share_projector:
            self.lidar_proj = self.rgb_proj
        else:
            self.lidar_proj = ProjectionHead(
                in_dim=model.lidars_encoder.enc_embed_dim,
                hidden_dim=proj_hidden_dim,
                out_dim=proj_dim,
            )

        self._freeze_non_contrastive_parts()
        self._set_encoder_trainability(train_rgb_encoder, train_lidar_encoder)

    def _freeze_non_contrastive_parts(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _set_encoder_trainability(self, train_rgb: bool, train_lidar: bool):
        for param in self.model.encoder.parameters():
            param.requires_grad = bool(train_rgb)
        for param in self.model.lidars_encoder.parameters():
            param.requires_grad = bool(train_lidar)
        for param in self.rgb_proj.parameters():
            param.requires_grad = True
        for param in self.lidar_proj.parameters():
            param.requires_grad = True

    @staticmethod
    def _should_checkpoint(module: nn.Module) -> bool:
        return torch.is_grad_enabled() and any(p.requires_grad for p in module.parameters())

    @staticmethod
    def _stack_view_field(views, key: str, *, cat_dim: int = 0):
        tensors = []
        for view in views:
            value = view[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            tensors.append(value)
        return torch.cat(tensors, dim=cat_dim)

    @staticmethod
    def _extract_data_norm_type(views) -> str:
        first = views[0].get("data_norm_type", "dinov2")
        if isinstance(first, (list, tuple)):
            return str(first[0])
        return str(first)

    @staticmethod
    def _extract_rgb_features(encoder, images: torch.Tensor, data_norm_type: str, use_amp: bool):
        ctx = torch.autocast("cuda", enabled=False) if use_amp and images.is_cuda else nullcontext()
        def _forward(image_tensor: torch.Tensor):
            with ctx:
                encoder_input = ViTEncoderInput(image=image_tensor, data_norm_type=data_norm_type)
                output = encoder(encoder_input)
            feats = output.features if hasattr(output, "features") else output
            return _pool_features(feats)

        if torch.is_grad_enabled() and any(p.requires_grad for p in encoder.parameters()):
            return checkpoint(_forward, images, use_reentrant=False)
        return _forward(images)

    def _extract_lidar_features(self, lidars: torch.Tensor, use_amp: bool):
        if lidars.ndim != 4:
            raise ValueError(f"Expected LiDAR tensor shaped [B, C, H, W], got {tuple(lidars.shape)}")
        ctx = torch.autocast("cuda", enabled=False) if use_amp and lidars.is_cuda else nullcontext()
        def _forward(lidar_tensor: torch.Tensor):
            with ctx:
                output = self.model.lidars_encoder(ViTEncoderNonImageInput(data=lidar_tensor))
            feats = output.features if hasattr(output, "features") else output
            return _pool_features(feats)

        if self.use_gradient_checkpointing and torch.is_grad_enabled() and any(
            p.requires_grad for p in self.model.lidars_encoder.parameters()
        ):
            return checkpoint(_forward, lidars, use_reentrant=False)
        return _forward(lidars)

    def forward(self, views, use_amp: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(views) == 0:
            raise ValueError("views is empty")

        data_norm_type = self._extract_data_norm_type(views)
        images = self._stack_view_field(views, "img", cat_dim=0)
        lidars = self._stack_view_field(views, "pcd", cat_dim=0)
        if lidars.ndim != 4:
            raise ValueError(f"Expected pcd tensors shaped [B, H, W, C], got {tuple(lidars.shape)}")

        lidars = lidars.permute(0, 3, 1, 2).contiguous()
        if hasattr(self.model, "lidar_input_size") and self.model.lidar_input_size > 0:
            target = int(self.model.lidar_input_size)
            if lidars.shape[-1] != target or lidars.shape[-2] != target:
                lidars = F.interpolate(lidars, size=(target, target), mode="bilinear", align_corners=False)

        rgb_feat = self._extract_rgb_features(self.model.encoder, images, data_norm_type, use_amp)
        lidar_feat = self._extract_lidar_features(lidars, use_amp)

        rgb_embed = self.rgb_proj(rgb_feat)
        lidar_embed = self.lidar_proj(lidar_feat)
        return rgb_embed, lidar_embed


def build_trainable_contrastive_model(
    model_dir: str | Path,
    device: torch.device,
    proj_dim: int = 256,
    proj_hidden_dim: int = 1024,
    share_projector: bool = False,
    train_rgb_encoder: bool = True,
    train_lidar_encoder: bool = True,
    use_gradient_checkpointing: bool = True,
):
    model = build_mapanything_from_dir(model_dir=model_dir, device=device)
    contrastive_model = ContrastivePairModel(
        model=model,
        proj_dim=proj_dim,
        proj_hidden_dim=proj_hidden_dim,
        share_projector=share_projector,
        train_rgb_encoder=train_rgb_encoder,
        train_lidar_encoder=train_lidar_encoder,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return contrastive_model
