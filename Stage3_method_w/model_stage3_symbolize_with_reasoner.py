#!/usr/bin/env python3
"""
Stage3 V3 Symbolize + Reasoner model:
    V1 fusion backbone + Symbolize + pairwise attribute reasoner
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.stage2_model import Stage2Model

from Stage3_method_w.model_stage3_no_reasoner import Stage2BackboneMeta, load_stage2_backbone


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


class SymbolizeLayer(nn.Module):
    """
    Differentiable symbolization with a learnable codebook.
    """

    def __init__(
        self,
        input_dim: int,
        symbol_k: int = 64,
        symbol_dim: int = 64,
        init_std: float = 0.02,
        use_gumbel: bool = True,
    ):
        super().__init__()
        self.symbol_k = int(symbol_k)
        self.symbol_dim = int(symbol_dim)
        self.use_gumbel = bool(use_gumbel)

        self.logit_proj = nn.Linear(input_dim, self.symbol_k)
        self.codebook = nn.Parameter(torch.randn(self.symbol_k, self.symbol_dim) * float(init_std))

    def forward(self, feat: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        tau = float(max(temperature, 1e-4))
        symbol_logits = self.logit_proj(feat)

        if self.training and self.use_gumbel:
            symbol_probs = F.gumbel_softmax(symbol_logits, tau=tau, hard=False, dim=-1)
        else:
            symbol_probs = torch.softmax(symbol_logits / tau, dim=-1)

        symbol_embed = torch.matmul(symbol_probs, self.codebook)
        symbol_id = symbol_probs.argmax(dim=-1)
        return {
            "symbol_logits": symbol_logits,
            "symbol_probs": symbol_probs,
            "symbol_embed": symbol_embed,
            "symbol_id": symbol_id,
        }


class PairwiseReasoner(nn.Module):
    """
    Message-passing reasoner over material / hardness / roughness logits.
    """

    def __init__(
        self,
        num_material: int,
        num_hardness: int,
        num_roughness: int,
        reason_steps: int = 2,
        init_std: float = 1e-3,
        freeze_reasoner: bool = False,
    ):
        super().__init__()
        self.num_material = int(num_material)
        self.num_hardness = int(num_hardness)
        self.num_roughness = int(num_roughness)
        self.reason_steps = int(reason_steps)
        self.init_std = float(init_std)
        self.freeze_reasoner = bool(freeze_reasoner)

        self.psi_mh = nn.Parameter(torch.empty(self.num_material, self.num_hardness))
        self.psi_mr = nn.Parameter(torch.empty(self.num_material, self.num_roughness))
        self.psi_hr = nn.Parameter(torch.empty(self.num_hardness, self.num_roughness))
        self.reset_parameters()

        if self.freeze_reasoner:
            for param in self.parameters():
                param.requires_grad = False

    def reset_parameters(self) -> None:
        if self.init_std > 0.0:
            nn.init.normal_(self.psi_mh, mean=0.0, std=self.init_std)
            nn.init.normal_(self.psi_mr, mean=0.0, std=self.init_std)
            nn.init.normal_(self.psi_hr, mean=0.0, std=self.init_std)
        else:
            nn.init.zeros_(self.psi_mh)
            nn.init.zeros_(self.psi_mr)
            nn.init.zeros_(self.psi_hr)

    def relation_l1(self) -> torch.Tensor:
        return self.psi_mh.abs().sum() + self.psi_mr.abs().sum() + self.psi_hr.abs().sum()

    def relation_abs_mean(self) -> torch.Tensor:
        values = torch.cat(
            [
                self.psi_mh.abs().reshape(-1),
                self.psi_mr.abs().reshape(-1),
                self.psi_hr.abs().reshape(-1),
            ],
            dim=0,
        )
        return values.mean()

    def relation_sparsity(self, threshold: float = 1e-3) -> torch.Tensor:
        values = torch.cat(
            [
                self.psi_mh.abs().reshape(-1),
                self.psi_mr.abs().reshape(-1),
                self.psi_hr.abs().reshape(-1),
            ],
            dim=0,
        )
        if int(values.numel()) == 0:
            return torch.zeros((), dtype=values.dtype, device=values.device)
        return (values < float(threshold)).to(values.dtype).mean()

    def export_state(self) -> Dict[str, torch.Tensor]:
        return {
            "psi_mh": self.psi_mh.detach().cpu(),
            "psi_mr": self.psi_mr.detach().cpu(),
            "psi_hr": self.psi_hr.detach().cpu(),
        }

    def forward(
        self,
        unary_logits_material: torch.Tensor,
        unary_logits_hardness: torch.Tensor,
        unary_logits_roughness: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits_material = unary_logits_material
        logits_hardness = unary_logits_hardness
        logits_roughness = unary_logits_roughness

        probs_material = torch.softmax(logits_material, dim=-1)
        probs_hardness = torch.softmax(logits_hardness, dim=-1)
        probs_roughness = torch.softmax(logits_roughness, dim=-1)

        prev_probs_material = probs_material
        prev_probs_hardness = probs_hardness
        prev_probs_roughness = probs_roughness

        zero_mh = torch.zeros_like(unary_logits_material)
        zero_mr = torch.zeros_like(unary_logits_material)
        zero_hm = torch.zeros_like(unary_logits_hardness)
        zero_hr = torch.zeros_like(unary_logits_hardness)
        zero_rm = torch.zeros_like(unary_logits_roughness)
        zero_rh = torch.zeros_like(unary_logits_roughness)

        msg_material_from_hardness = zero_mh
        msg_material_from_roughness = zero_mr
        msg_hardness_from_material = zero_hm
        msg_hardness_from_roughness = zero_hr
        msg_roughness_from_material = zero_rm
        msg_roughness_from_hardness = zero_rh

        for _ in range(self.reason_steps):
            msg_material_from_hardness = torch.matmul(
                probs_hardness,
                self.psi_mh.transpose(0, 1),
            )
            msg_material_from_roughness = torch.matmul(
                probs_roughness,
                self.psi_mr.transpose(0, 1),
            )

            msg_hardness_from_material = torch.matmul(probs_material, self.psi_mh)
            msg_hardness_from_roughness = torch.matmul(
                probs_roughness,
                self.psi_hr.transpose(0, 1),
            )

            msg_roughness_from_material = torch.matmul(probs_material, self.psi_mr)
            msg_roughness_from_hardness = torch.matmul(probs_hardness, self.psi_hr)

            logits_material = (
                unary_logits_material
                + msg_material_from_hardness
                + msg_material_from_roughness
            )
            logits_hardness = (
                unary_logits_hardness
                + msg_hardness_from_material
                + msg_hardness_from_roughness
            )
            logits_roughness = (
                unary_logits_roughness
                + msg_roughness_from_material
                + msg_roughness_from_hardness
            )

            prev_probs_material = probs_material
            prev_probs_hardness = probs_hardness
            prev_probs_roughness = probs_roughness

            probs_material = torch.softmax(logits_material, dim=-1)
            probs_hardness = torch.softmax(logits_hardness, dim=-1)
            probs_roughness = torch.softmax(logits_roughness, dim=-1)

        return {
            "logits_material": logits_material,
            "logits_hardness": logits_hardness,
            "logits_roughness": logits_roughness,
            "probs_material": probs_material,
            "probs_hardness": probs_hardness,
            "probs_roughness": probs_roughness,
            "prev_probs_material": prev_probs_material,
            "prev_probs_hardness": prev_probs_hardness,
            "prev_probs_roughness": prev_probs_roughness,
            "reason_message_material_from_hardness": msg_material_from_hardness,
            "reason_message_material_from_roughness": msg_material_from_roughness,
            "reason_message_hardness_from_material": msg_hardness_from_material,
            "reason_message_hardness_from_roughness": msg_hardness_from_roughness,
            "reason_message_roughness_from_material": msg_roughness_from_material,
            "reason_message_roughness_from_hardness": msg_roughness_from_hardness,
        }


class Stage3SymbolizeWithReasonerModel(nn.Module):
    """
    Stage3 V3:
      x -> frozen Stage2 backbone -> concat(encoder_h, z_contact)
        -> Symbolize(fusion)
        -> concat(fusion, s_x)
        -> shared fusion trunk
        -> unary heads
        -> pairwise reasoner
    """

    def __init__(
        self,
        backbone: Stage2Model,
        num_material: int,
        num_hardness: int,
        num_roughness: int,
        symbol_k: int = 64,
        symbol_dim: int = 64,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        reason_steps: int = 2,
        reasoner_init_std: float = 1e-3,
        freeze_reasoner: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone_is_frozen = bool(freeze_backbone)
        self.contact_dim = int(getattr(backbone, "projection_dim", 128))
        self.encoder_dim = _infer_encoder_output_dim(backbone)
        self.fused_dim = self.encoder_dim + self.contact_dim
        self.symbol_dim = int(symbol_dim)
        self.hidden_dim = int(hidden_dim)
        self._backbone_default_requires_grad = [bool(p.requires_grad) for p in self.backbone.parameters()]

        self.encoder_norm = nn.LayerNorm(self.encoder_dim)
        self.contact_norm = nn.LayerNorm(self.contact_dim)
        self.symbol_norm = nn.LayerNorm(self.symbol_dim)
        self.symbolize = SymbolizeLayer(
            input_dim=self.fused_dim,
            symbol_k=symbol_k,
            symbol_dim=symbol_dim,
        )

        self.fused_with_symbol_dim = self.fused_dim + self.symbol_dim
        self.shared_trunk = FusionTrunk(
            input_dim=self.fused_with_symbol_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )

        material_input_dim = self.hidden_dim + self.encoder_dim + self.symbol_dim
        contact_task_input_dim = self.hidden_dim + self.contact_dim + self.symbol_dim
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

        self.reasoner = PairwiseReasoner(
            num_material=num_material,
            num_hardness=num_hardness,
            num_roughness=num_roughness,
            reason_steps=reason_steps,
            init_std=reasoner_init_std,
            freeze_reasoner=freeze_reasoner,
        )

        if self.freeze_backbone:
            self.set_backbone_trainable(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_is_frozen:
            self.backbone.eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        trainable = bool(trainable)
        for param, default_flag in zip(self.backbone.parameters(), self._backbone_default_requires_grad):
            param.requires_grad = bool(default_flag) if trainable else False
        self.freeze_backbone = not trainable
        self.backbone_is_frozen = not trainable
        if self.backbone_is_frozen:
            self.backbone.eval()

    def set_reasoner_trainable(self, trainable: bool) -> None:
        trainable = bool(trainable)
        for param in self.reasoner.parameters():
            param.requires_grad = trainable
        self.reasoner.freeze_reasoner = not trainable

    def relation_l1(self) -> torch.Tensor:
        return self.reasoner.relation_l1()

    def relation_abs_mean(self) -> torch.Tensor:
        return self.reasoner.relation_abs_mean()

    def relation_sparsity(self, threshold: float = 1e-3) -> torch.Tensor:
        return self.reasoner.relation_sparsity(threshold=threshold)

    def export_relation_state(self) -> Dict[str, torch.Tensor]:
        return self.reasoner.export_state()

    def forward(self, x: torch.Tensor, symbol_temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        if self.backbone_is_frozen:
            with torch.no_grad():
                contact_feat, encoder_feat = self.backbone(x, return_encoder_output=True)
        else:
            contact_feat, encoder_feat = self.backbone(x, return_encoder_output=True)

        encoder_feat = self.encoder_norm(encoder_feat)
        contact_feat = self.contact_norm(contact_feat)
        fused_feat = torch.cat([encoder_feat, contact_feat], dim=-1)

        symbol_out = self.symbolize(fused_feat, temperature=symbol_temperature)
        symbol_feat = self.symbol_norm(symbol_out["symbol_embed"])
        fused_with_symbol = torch.cat([fused_feat, symbol_feat], dim=-1)
        shared_feat = self.shared_trunk(fused_with_symbol)

        material_feat = torch.cat([shared_feat, encoder_feat, symbol_feat], dim=-1)
        contact_task_feat = torch.cat([shared_feat, contact_feat, symbol_feat], dim=-1)

        unary_logits_material = self.head_material(material_feat)
        unary_logits_hardness = self.head_hardness(contact_task_feat)
        unary_logits_roughness = self.head_roughness(contact_task_feat)

        reason_out = self.reasoner(
            unary_logits_material=unary_logits_material,
            unary_logits_hardness=unary_logits_hardness,
            unary_logits_roughness=unary_logits_roughness,
        )

        return {
            "feat": shared_feat,
            "feat_encoder": encoder_feat,
            "feat_contact": contact_feat,
            "feat_fused_base": fused_feat,
            "feat_symbol": symbol_feat,
            "feat_fused": fused_with_symbol,
            "symbol_logits": symbol_out["symbol_logits"],
            "symbol_probs": symbol_out["symbol_probs"],
            "symbol_embed": symbol_feat,
            "symbol_id": symbol_out["symbol_id"],
            "logits_material_unary": unary_logits_material,
            "logits_hardness_unary": unary_logits_hardness,
            "logits_roughness_unary": unary_logits_roughness,
            "logits_material": reason_out["logits_material"],
            "logits_hardness": reason_out["logits_hardness"],
            "logits_roughness": reason_out["logits_roughness"],
            "probs_material": reason_out["probs_material"],
            "probs_hardness": reason_out["probs_hardness"],
            "probs_roughness": reason_out["probs_roughness"],
            "prev_probs_material": reason_out["prev_probs_material"],
            "prev_probs_hardness": reason_out["prev_probs_hardness"],
            "prev_probs_roughness": reason_out["prev_probs_roughness"],
            "reason_message_material_from_hardness": reason_out["reason_message_material_from_hardness"],
            "reason_message_material_from_roughness": reason_out["reason_message_material_from_roughness"],
            "reason_message_hardness_from_material": reason_out["reason_message_hardness_from_material"],
            "reason_message_hardness_from_roughness": reason_out["reason_message_hardness_from_roughness"],
            "reason_message_roughness_from_material": reason_out["reason_message_roughness_from_material"],
            "reason_message_roughness_from_hardness": reason_out["reason_message_roughness_from_hardness"],
        }


__all__ = [
    "Stage2BackboneMeta",
    "load_stage2_backbone",
    "Stage3SymbolizeWithReasonerModel",
]
