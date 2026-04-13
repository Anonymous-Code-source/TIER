#!/usr/bin/env python3
"""
Stage3 V1 No-Reasoner model:
    Stage2 backbone feature -> three classification heads
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from model.stage2_model import Stage2Model, load_stage1_encoder


@dataclass
class Stage2BackboneMeta:
    stage1_checkpoint: str
    projection_dim: int
    domain_projection_dim: int
    freeze_blocks: int
    has_discriminator: bool
    has_matching_head: bool
    has_domain_classifier: bool
    missing_keys: int
    unexpected_keys: int


def _infer_output_dim_from_state(state_dict: Dict[str, torch.Tensor]) -> int:
    key = "projection_head.net.3.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    # fallback to default used in Stage2
    return 128


def _extract_stage2_build_args(
    stage2_ckpt: Dict[str, object],
    stage1_checkpoint_override: str = "",
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    if "model" in stage2_ckpt and isinstance(stage2_ckpt["model"], dict):
        state_dict = stage2_ckpt["model"]
    elif isinstance(stage2_ckpt, dict):
        # tolerate raw model state_dict checkpoint
        state_dict = stage2_ckpt
    else:
        raise ValueError("Unsupported Stage2 checkpoint format.")

    ckpt_args = stage2_ckpt.get("args", None) if isinstance(stage2_ckpt, dict) else None
    projection_dim = _infer_output_dim_from_state(state_dict)
    domain_projection_dim = projection_dim
    freeze_blocks = 8
    num_sensors = 3
    discriminator_hidden_dim = 64
    discriminator_dropout = 0.1

    if ckpt_args is not None:
        projection_dim = int(getattr(ckpt_args, "projection_dim", projection_dim))
        domain_projection_dim = int(
            getattr(ckpt_args, "domain_projection_dim", projection_dim)
        )
        freeze_blocks = int(getattr(ckpt_args, "freeze_blocks", freeze_blocks))
        num_sensors = int(getattr(ckpt_args, "num_sensors", num_sensors))
        discriminator_hidden_dim = int(
            getattr(ckpt_args, "discriminator_hidden_dim", discriminator_hidden_dim)
        )
        discriminator_dropout = float(
            getattr(ckpt_args, "discriminator_dropout", discriminator_dropout)
        )

    has_discriminator = any(k.startswith("discriminator.") for k in state_dict.keys())
    has_matching_head = any(k.startswith("match_head.") for k in state_dict.keys())
    has_domain_classifier = any(
        k.startswith("domain_classifier.") for k in state_dict.keys()
    )

    stage1_ckpt = stage1_checkpoint_override
    if stage1_ckpt == "" and ckpt_args is not None:
        stage1_ckpt = str(getattr(ckpt_args, "stage1_checkpoint", ""))
    if stage1_ckpt == "":
        raise ValueError(
            "stage1_checkpoint is required. Pass --stage1_checkpoint or use a Stage2 checkpoint that stores args.stage1_checkpoint."
        )

    build_args = {
        "stage1_checkpoint": stage1_ckpt,
        "projection_dim": projection_dim,
        "domain_projection_dim": domain_projection_dim,
        "freeze_blocks": freeze_blocks,
        "num_sensors": num_sensors,
        "discriminator_hidden_dim": discriminator_hidden_dim,
        "discriminator_dropout": discriminator_dropout,
        "has_discriminator": has_discriminator,
        "has_matching_head": has_matching_head,
        "has_domain_classifier": has_domain_classifier,
    }
    return build_args, state_dict


def load_stage2_backbone(
    stage2_checkpoint_path: str,
    stage1_checkpoint_override: str = "",
    device: str = "cpu",
) -> Tuple[Stage2Model, Stage2BackboneMeta]:
    stage2_ckpt = torch.load(stage2_checkpoint_path, map_location="cpu", weights_only=False)
    build_args, state_dict = _extract_stage2_build_args(
        stage2_ckpt=stage2_ckpt,
        stage1_checkpoint_override=stage1_checkpoint_override,
    )

    stage1_model = load_stage1_encoder(
        checkpoint_path=build_args["stage1_checkpoint"],
        device=device,
    )

    model = Stage2Model(
        stage1_model=stage1_model,
        freeze_blocks=build_args["freeze_blocks"],
        projection_dim=build_args["projection_dim"],
        domain_projection_dim=build_args["domain_projection_dim"],
        use_discriminator=build_args["has_discriminator"],
        discriminator_hidden_dim=build_args["discriminator_hidden_dim"],
        discriminator_dropout=build_args["discriminator_dropout"],
        num_sensors=build_args["num_sensors"],
        use_matching_head=build_args["has_matching_head"],
        use_domain_classifier=build_args["has_domain_classifier"],
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.to(device)

    meta = Stage2BackboneMeta(
        stage1_checkpoint=build_args["stage1_checkpoint"],
        projection_dim=build_args["projection_dim"],
        domain_projection_dim=build_args["domain_projection_dim"],
        freeze_blocks=build_args["freeze_blocks"],
        has_discriminator=build_args["has_discriminator"],
        has_matching_head=build_args["has_matching_head"],
        has_domain_classifier=build_args["has_domain_classifier"],
        missing_keys=len(incompatible.missing_keys),
        unexpected_keys=len(incompatible.unexpected_keys),
    )
    return model, meta


def _infer_encoder_output_dim(backbone: Stage2Model) -> int:
    stage1_model = getattr(backbone, "stage1_model", None)
    touch_projection = getattr(stage1_model, "touch_projection", None)
    if isinstance(touch_projection, nn.Linear):
        return int(touch_projection.out_features)
    return int(getattr(backbone, "projection_dim", 128))


class FusionTrunk(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.residual = nn.Linear(input_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return self.net(x) + self.residual(x)


class TaskHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage3NoReasonerModel(nn.Module):
    """
    Improved V1:
      x -> frozen Stage2 backbone -> concat(h, z_contact)
        -> shared fusion trunk -> three task heads
    """

    def __init__(
        self,
        backbone: Stage2Model,
        num_material: int,
        num_hardness: int,
        num_roughness: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.contact_dim = int(getattr(backbone, "projection_dim", 128))
        self.encoder_dim = _infer_encoder_output_dim(backbone)
        self.fused_dim = self.encoder_dim + self.contact_dim
        self.hidden_dim = int(hidden_dim)

        self.encoder_norm = nn.LayerNorm(self.encoder_dim)
        self.contact_norm = nn.LayerNorm(self.contact_dim)
        self.shared_trunk = FusionTrunk(
            input_dim=self.fused_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )

        # Material is the weakest task, so it gets a wider and deeper head.
        material_input_dim = self.hidden_dim + self.encoder_dim
        contact_task_input_dim = self.hidden_dim + self.contact_dim
        self.head_material = TaskHead(
            input_dim=material_input_dim,
            hidden_dims=(self.hidden_dim * 2, self.hidden_dim),
            output_dim=num_material,
            dropout=dropout,
        )
        self.head_hardness = TaskHead(
            input_dim=contact_task_input_dim,
            hidden_dims=(self.hidden_dim,),
            output_dim=num_hardness,
            dropout=dropout,
        )
        self.head_roughness = TaskHead(
            input_dim=contact_task_input_dim,
            hidden_dims=(self.hidden_dim,),
            output_dim=num_roughness,
            dropout=dropout,
        )

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen backbone in eval mode to avoid dropout/bn randomness.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.freeze_backbone:
            with torch.no_grad():
                contact_feat, encoder_feat = self.backbone(x, return_encoder_output=True)
        else:
            contact_feat, encoder_feat = self.backbone(x, return_encoder_output=True)

        encoder_feat = self.encoder_norm(encoder_feat)
        contact_feat = self.contact_norm(contact_feat)
        fused_feat = torch.cat([encoder_feat, contact_feat], dim=-1)
        shared_feat = self.shared_trunk(fused_feat)

        material_feat = torch.cat([shared_feat, encoder_feat], dim=-1)
        contact_task_feat = torch.cat([shared_feat, contact_feat], dim=-1)

        logits_material = self.head_material(material_feat)
        logits_hardness = self.head_hardness(contact_task_feat)
        logits_roughness = self.head_roughness(contact_task_feat)

        return {
            "feat": shared_feat,
            "feat_encoder": encoder_feat,
            "feat_contact": contact_feat,
            "feat_fused": fused_feat,
            "logits_material": logits_material,
            "logits_hardness": logits_hardness,
            "logits_roughness": logits_roughness,
        }
