#!/usr/bin/env python3
"""
Hierarchical Stage3 model:
    x -> E(x) -> s_x -> s^p -> s^i -> s^c
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

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


class BeliefProjector(nn.Module):
    def __init__(self, input_dims: Sequence[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(sum(int(v) for v in input_dims))
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
        )

    def forward(self, probs: Iterable[torch.Tensor]) -> torch.Tensor:
        return self.net(torch.cat(list(probs), dim=-1))


class GenericTriReasoner(nn.Module):
    def __init__(
        self,
        num_a: int,
        num_b: int,
        num_c: int,
        reason_steps: int = 2,
        init_std: float = 1e-3,
    ):
        super().__init__()
        self.num_a = int(num_a)
        self.num_b = int(num_b)
        self.num_c = int(num_c)
        self.reason_steps = int(reason_steps)
        self.init_std = float(init_std)

        self.psi_ab = nn.Parameter(torch.empty(self.num_a, self.num_b))
        self.psi_ac = nn.Parameter(torch.empty(self.num_a, self.num_c))
        self.psi_bc = nn.Parameter(torch.empty(self.num_b, self.num_c))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_std > 0.0:
            nn.init.normal_(self.psi_ab, mean=0.0, std=self.init_std)
            nn.init.normal_(self.psi_ac, mean=0.0, std=self.init_std)
            nn.init.normal_(self.psi_bc, mean=0.0, std=self.init_std)
        else:
            nn.init.zeros_(self.psi_ab)
            nn.init.zeros_(self.psi_ac)
            nn.init.zeros_(self.psi_bc)

    def relation_values(self) -> torch.Tensor:
        return torch.cat(
            [
                self.psi_ab.reshape(-1),
                self.psi_ac.reshape(-1),
                self.psi_bc.reshape(-1),
            ],
            dim=0,
        )

    def relation_l1(self) -> torch.Tensor:
        return self.relation_values().abs().sum()

    def relation_abs_mean(self) -> torch.Tensor:
        values = self.relation_values().abs()
        if int(values.numel()) == 0:
            return torch.zeros((), dtype=self.psi_ab.dtype, device=self.psi_ab.device)
        return values.mean()

    def relation_sparsity(self, threshold: float = 1e-3) -> torch.Tensor:
        values = self.relation_values().abs()
        if int(values.numel()) == 0:
            return torch.zeros((), dtype=self.psi_ab.dtype, device=self.psi_ab.device)
        return (values < float(threshold)).to(values.dtype).mean()

    def export_state(self, prefix: str) -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}_psi_ab": self.psi_ab.detach().cpu(),
            f"{prefix}_psi_ac": self.psi_ac.detach().cpu(),
            f"{prefix}_psi_bc": self.psi_bc.detach().cpu(),
        }

    def forward(
        self,
        logits_a_unary: torch.Tensor,
        logits_b_unary: torch.Tensor,
        logits_c_unary: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits_a = logits_a_unary
        logits_b = logits_b_unary
        logits_c = logits_c_unary

        probs_a = torch.softmax(logits_a, dim=-1)
        probs_b = torch.softmax(logits_b, dim=-1)
        probs_c = torch.softmax(logits_c, dim=-1)

        prev_probs_a = probs_a
        prev_probs_b = probs_b
        prev_probs_c = probs_c

        zero_a = torch.zeros_like(logits_a_unary)
        zero_b = torch.zeros_like(logits_b_unary)
        zero_c = torch.zeros_like(logits_c_unary)

        msg_a_from_b = zero_a
        msg_a_from_c = zero_a
        msg_b_from_a = zero_b
        msg_b_from_c = zero_b
        msg_c_from_a = zero_c
        msg_c_from_b = zero_c

        for _ in range(self.reason_steps):
            msg_a_from_b = torch.matmul(probs_b, self.psi_ab.transpose(0, 1))
            msg_a_from_c = torch.matmul(probs_c, self.psi_ac.transpose(0, 1))
            msg_b_from_a = torch.matmul(probs_a, self.psi_ab)
            msg_b_from_c = torch.matmul(probs_c, self.psi_bc.transpose(0, 1))
            msg_c_from_a = torch.matmul(probs_a, self.psi_ac)
            msg_c_from_b = torch.matmul(probs_b, self.psi_bc)

            logits_a = logits_a_unary + msg_a_from_b + msg_a_from_c
            logits_b = logits_b_unary + msg_b_from_a + msg_b_from_c
            logits_c = logits_c_unary + msg_c_from_a + msg_c_from_b

            prev_probs_a = probs_a
            prev_probs_b = probs_b
            prev_probs_c = probs_c

            probs_a = torch.softmax(logits_a, dim=-1)
            probs_b = torch.softmax(logits_b, dim=-1)
            probs_c = torch.softmax(logits_c, dim=-1)

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "logits_c": logits_c,
            "probs_a": probs_a,
            "probs_b": probs_b,
            "probs_c": probs_c,
            "prev_probs_a": prev_probs_a,
            "prev_probs_b": prev_probs_b,
            "prev_probs_c": prev_probs_c,
            "msg_a_from_b": msg_a_from_b,
            "msg_a_from_c": msg_a_from_c,
            "msg_b_from_a": msg_b_from_a,
            "msg_b_from_c": msg_b_from_c,
            "msg_c_from_a": msg_c_from_a,
            "msg_c_from_b": msg_c_from_b,
        }


class MixedContactReasoner(nn.Module):
    """
    Message passing over slippage / rotation classification logits and a
    continuous force value.
    """

    def __init__(
        self,
        num_slippage: int,
        num_rotation: int,
        reason_steps: int = 2,
        init_std: float = 1e-3,
    ):
        super().__init__()
        self.num_slippage = int(num_slippage)
        self.num_rotation = int(num_rotation)
        self.reason_steps = int(reason_steps)
        self.init_std = float(init_std)

        self.psi_sr = nn.Parameter(torch.empty(self.num_slippage, self.num_rotation))
        self.force_to_slippage = nn.Linear(1, self.num_slippage, bias=False)
        self.force_to_rotation = nn.Linear(1, self.num_rotation, bias=False)
        self.slippage_to_force = nn.Linear(self.num_slippage, 1, bias=False)
        self.rotation_to_force = nn.Linear(self.num_rotation, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_std > 0.0:
            nn.init.normal_(self.psi_sr, mean=0.0, std=self.init_std)
            for module in (
                self.force_to_slippage,
                self.force_to_rotation,
                self.slippage_to_force,
                self.rotation_to_force,
            ):
                nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
        else:
            nn.init.zeros_(self.psi_sr)
            for module in (
                self.force_to_slippage,
                self.force_to_rotation,
                self.slippage_to_force,
                self.rotation_to_force,
            ):
                nn.init.zeros_(module.weight)

    def relation_values(self) -> torch.Tensor:
        return torch.cat(
            [
                self.psi_sr.reshape(-1),
                self.force_to_slippage.weight.reshape(-1),
                self.force_to_rotation.weight.reshape(-1),
                self.slippage_to_force.weight.reshape(-1),
                self.rotation_to_force.weight.reshape(-1),
            ],
            dim=0,
        )

    def relation_l1(self) -> torch.Tensor:
        return self.relation_values().abs().sum()

    def relation_abs_mean(self) -> torch.Tensor:
        values = self.relation_values().abs()
        if int(values.numel()) == 0:
            return torch.zeros((), dtype=self.psi_sr.dtype, device=self.psi_sr.device)
        return values.mean()

    def relation_sparsity(self, threshold: float = 1e-3) -> torch.Tensor:
        values = self.relation_values().abs()
        if int(values.numel()) == 0:
            return torch.zeros((), dtype=self.psi_sr.dtype, device=self.psi_sr.device)
        return (values < float(threshold)).to(values.dtype).mean()

    def export_state(self, prefix: str) -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}_psi_slippage_rotation": self.psi_sr.detach().cpu(),
            f"{prefix}_force_to_slippage": self.force_to_slippage.weight.detach().cpu(),
            f"{prefix}_force_to_rotation": self.force_to_rotation.weight.detach().cpu(),
            f"{prefix}_slippage_to_force": self.slippage_to_force.weight.detach().cpu(),
            f"{prefix}_rotation_to_force": self.rotation_to_force.weight.detach().cpu(),
        }

    def forward(
        self,
        logits_slippage_unary: torch.Tensor,
        logits_rotation_unary: torch.Tensor,
        force_unary: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits_slippage = logits_slippage_unary
        logits_rotation = logits_rotation_unary
        force_value = F.softplus(force_unary)

        probs_slippage = torch.softmax(logits_slippage, dim=-1)
        probs_rotation = torch.softmax(logits_rotation, dim=-1)

        prev_probs_slippage = probs_slippage
        prev_probs_rotation = probs_rotation
        prev_force = force_value

        msg_slippage_from_rotation = torch.zeros_like(logits_slippage_unary)
        msg_slippage_from_force = torch.zeros_like(logits_slippage_unary)
        msg_rotation_from_slippage = torch.zeros_like(logits_rotation_unary)
        msg_rotation_from_force = torch.zeros_like(logits_rotation_unary)
        msg_force_from_slippage = torch.zeros_like(force_unary)
        msg_force_from_rotation = torch.zeros_like(force_unary)

        for _ in range(self.reason_steps):
            msg_slippage_from_rotation = torch.matmul(
                probs_rotation,
                self.psi_sr.transpose(0, 1),
            )
            msg_rotation_from_slippage = torch.matmul(probs_slippage, self.psi_sr)
            msg_slippage_from_force = self.force_to_slippage(force_value)
            msg_rotation_from_force = self.force_to_rotation(force_value)
            msg_force_from_slippage = self.slippage_to_force(probs_slippage)
            msg_force_from_rotation = self.rotation_to_force(probs_rotation)

            logits_slippage = (
                logits_slippage_unary
                + msg_slippage_from_rotation
                + msg_slippage_from_force
            )
            logits_rotation = (
                logits_rotation_unary
                + msg_rotation_from_slippage
                + msg_rotation_from_force
            )
            force_value = F.softplus(force_unary + msg_force_from_slippage + msg_force_from_rotation)

            prev_probs_slippage = probs_slippage
            prev_probs_rotation = probs_rotation
            prev_force = force_value

            probs_slippage = torch.softmax(logits_slippage, dim=-1)
            probs_rotation = torch.softmax(logits_rotation, dim=-1)

        return {
            "logits_slippage": logits_slippage,
            "logits_rotation": logits_rotation,
            "force": force_value,
            "probs_slippage": probs_slippage,
            "probs_rotation": probs_rotation,
            "prev_probs_slippage": prev_probs_slippage,
            "prev_probs_rotation": prev_probs_rotation,
            "prev_force": prev_force,
            "msg_slippage_from_rotation": msg_slippage_from_rotation,
            "msg_slippage_from_force": msg_slippage_from_force,
            "msg_rotation_from_slippage": msg_rotation_from_slippage,
            "msg_rotation_from_force": msg_rotation_from_force,
            "msg_force_from_slippage": msg_force_from_slippage,
            "msg_force_from_rotation": msg_force_from_rotation,
        }


class PhysicalModule(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        contact_dim: int,
        fused_dim: int,
        obs_symbol_dim: int,
        symbol_k: int,
        symbol_dim: int,
        hidden_dim: int,
        num_material: int,
        num_hardness: int,
        num_roughness: int,
        dropout: float,
        reason_steps: int,
        reasoner_init_std: float,
    ):
        super().__init__()
        self.trunk = FusionTrunk(
            input_dim=fused_dim + obs_symbol_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.symbolize = SymbolizeLayer(
            input_dim=hidden_dim,
            symbol_k=symbol_k,
            symbol_dim=symbol_dim,
        )
        self.symbol_norm = nn.LayerNorm(symbol_dim)

        material_input_dim = hidden_dim + encoder_dim + obs_symbol_dim + symbol_dim
        contact_task_input_dim = hidden_dim + contact_dim + obs_symbol_dim + symbol_dim
        self.head_material = TaskHead(
            input_dim=material_input_dim,
            hidden_dims=(hidden_dim * 2, hidden_dim),
            output_dim=num_material,
            dropout=dropout,
        )
        self.head_hardness = TaskHead(
            input_dim=contact_task_input_dim,
            hidden_dims=(hidden_dim,),
            output_dim=num_hardness,
            dropout=dropout,
        )
        self.head_roughness = TaskHead(
            input_dim=contact_task_input_dim,
            hidden_dims=(hidden_dim,),
            output_dim=num_roughness,
            dropout=dropout,
        )

        self.reasoner = GenericTriReasoner(
            num_a=num_material,
            num_b=num_hardness,
            num_c=num_roughness,
            reason_steps=reason_steps,
            init_std=reasoner_init_std,
        )
        self.belief_projector = BeliefProjector(
            input_dims=(num_material, num_hardness, num_roughness),
            output_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        fused_feat: torch.Tensor,
        encoder_feat: torch.Tensor,
        contact_feat: torch.Tensor,
        obs_symbol_embed: torch.Tensor,
        symbol_temperature: float,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([fused_feat, obs_symbol_embed], dim=-1))
        symbol_out = self.symbolize(hidden, temperature=symbol_temperature)
        symbol_embed = self.symbol_norm(symbol_out["symbol_embed"])

        material_feat = torch.cat([hidden, encoder_feat, obs_symbol_embed, symbol_embed], dim=-1)
        contact_task_feat = torch.cat([hidden, contact_feat, obs_symbol_embed, symbol_embed], dim=-1)

        logits_material_unary = self.head_material(material_feat)
        logits_hardness_unary = self.head_hardness(contact_task_feat)
        logits_roughness_unary = self.head_roughness(contact_task_feat)

        reason_out = self.reasoner(
            logits_a_unary=logits_material_unary,
            logits_b_unary=logits_hardness_unary,
            logits_c_unary=logits_roughness_unary,
        )
        belief_feat = self.belief_projector(
            [reason_out["probs_a"], reason_out["probs_b"], reason_out["probs_c"]]
        )
        return {
            "hidden": hidden,
            "symbol_logits": symbol_out["symbol_logits"],
            "symbol_probs": symbol_out["symbol_probs"],
            "symbol_embed": symbol_embed,
            "symbol_id": symbol_out["symbol_id"],
            "belief_feat": belief_feat,
            "logits": {
                "material": reason_out["logits_a"],
                "hardness": reason_out["logits_b"],
                "roughness": reason_out["logits_c"],
            },
            "logits_unary": {
                "material": logits_material_unary,
                "hardness": logits_hardness_unary,
                "roughness": logits_roughness_unary,
            },
            "probs": {
                "material": reason_out["probs_a"],
                "hardness": reason_out["probs_b"],
                "roughness": reason_out["probs_c"],
            },
            "prev_probs": {
                "material": reason_out["prev_probs_a"],
                "hardness": reason_out["prev_probs_b"],
                "roughness": reason_out["prev_probs_c"],
            },
            "reason_messages": {
                "material_from_hardness": reason_out["msg_a_from_b"],
                "material_from_roughness": reason_out["msg_a_from_c"],
                "hardness_from_material": reason_out["msg_b_from_a"],
                "hardness_from_roughness": reason_out["msg_b_from_c"],
                "roughness_from_material": reason_out["msg_c_from_a"],
                "roughness_from_hardness": reason_out["msg_c_from_b"],
            },
        }


class InteractionModule(nn.Module):
    def __init__(
        self,
        fused_dim: int,
        obs_symbol_dim: int,
        physical_symbol_dim: int,
        physical_belief_dim: int,
        symbol_k: int,
        symbol_dim: int,
        hidden_dim: int,
        num_deformation: int,
        dropout: float,
    ):
        super().__init__()
        self.trunk = FusionTrunk(
            input_dim=fused_dim + obs_symbol_dim + physical_symbol_dim + physical_belief_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.symbolize = SymbolizeLayer(
            input_dim=hidden_dim,
            symbol_k=symbol_k,
            symbol_dim=symbol_dim,
        )
        self.symbol_norm = nn.LayerNorm(symbol_dim)
        self.head_deformation = TaskHead(
            input_dim=hidden_dim + obs_symbol_dim + physical_belief_dim + symbol_dim,
            hidden_dims=(hidden_dim,),
            output_dim=num_deformation,
            dropout=dropout,
        )
        self.belief_projector = BeliefProjector(
            input_dims=(num_deformation,),
            output_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        fused_feat: torch.Tensor,
        obs_symbol_embed: torch.Tensor,
        physical_symbol_embed: torch.Tensor,
        physical_belief_feat: torch.Tensor,
        symbol_temperature: float,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(
            torch.cat([fused_feat, obs_symbol_embed, physical_symbol_embed, physical_belief_feat], dim=-1)
        )
        symbol_out = self.symbolize(hidden, temperature=symbol_temperature)
        symbol_embed = self.symbol_norm(symbol_out["symbol_embed"])
        head_input = torch.cat([hidden, obs_symbol_embed, physical_belief_feat, symbol_embed], dim=-1)
        logits_deformation = self.head_deformation(head_input)
        probs_deformation = torch.softmax(logits_deformation, dim=-1)
        belief_feat = self.belief_projector([probs_deformation])
        return {
            "hidden": hidden,
            "symbol_logits": symbol_out["symbol_logits"],
            "symbol_probs": symbol_out["symbol_probs"],
            "symbol_embed": symbol_embed,
            "symbol_id": symbol_out["symbol_id"],
            "belief_feat": belief_feat,
            "logits": {"deformation": logits_deformation},
            "probs": {"deformation": probs_deformation},
        }


class ContactModule(nn.Module):
    def __init__(
        self,
        fused_dim: int,
        obs_symbol_dim: int,
        interaction_symbol_dim: int,
        interaction_belief_dim: int,
        physical_symbol_dim: int,
        physical_belief_dim: int,
        symbol_k: int,
        symbol_dim: int,
        hidden_dim: int,
        num_slippage: int,
        num_rotation: int,
        num_force: int,
        dropout: float,
        reason_steps: int,
        reasoner_init_std: float,
    ):
        super().__init__()
        self.main_trunk = FusionTrunk(
            input_dim=fused_dim + obs_symbol_dim + interaction_symbol_dim + interaction_belief_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.skip_projector = nn.Sequential(
            nn.LayerNorm(physical_symbol_dim + physical_belief_dim),
            nn.Linear(physical_symbol_dim + physical_belief_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.symbolize = SymbolizeLayer(
            input_dim=hidden_dim,
            symbol_k=symbol_k,
            symbol_dim=symbol_dim,
        )
        self.symbol_norm = nn.LayerNorm(symbol_dim)
        contact_input_dim = hidden_dim + interaction_belief_dim + physical_belief_dim + symbol_dim
        self.head_slippage = TaskHead(
            input_dim=contact_input_dim,
            hidden_dims=(hidden_dim,),
            output_dim=num_slippage,
            dropout=dropout,
        )
        self.head_rotation = TaskHead(
            input_dim=contact_input_dim,
            hidden_dims=(hidden_dim,),
            output_dim=num_rotation,
            dropout=dropout,
        )
        self.head_force = TaskHead(
            input_dim=contact_input_dim,
            hidden_dims=(hidden_dim,),
            output_dim=1,
            dropout=dropout,
        )
        self.reasoner = MixedContactReasoner(
            num_slippage=num_slippage,
            num_rotation=num_rotation,
            reason_steps=reason_steps,
            init_std=reasoner_init_std,
        )

    def forward(
        self,
        fused_feat: torch.Tensor,
        obs_symbol_embed: torch.Tensor,
        interaction_symbol_embed: torch.Tensor,
        interaction_belief_feat: torch.Tensor,
        physical_symbol_embed: torch.Tensor,
        physical_belief_feat: torch.Tensor,
        symbol_temperature: float,
    ) -> Dict[str, torch.Tensor]:
        hidden_main = self.main_trunk(
            torch.cat([fused_feat, obs_symbol_embed, interaction_symbol_embed, interaction_belief_feat], dim=-1)
        )
        hidden_skip = self.skip_projector(torch.cat([physical_symbol_embed, physical_belief_feat], dim=-1))
        gate = torch.sigmoid(self.gate_net(torch.cat([hidden_main, hidden_skip], dim=-1)))
        hidden = hidden_main + gate * hidden_skip

        symbol_out = self.symbolize(hidden, temperature=symbol_temperature)
        symbol_embed = self.symbol_norm(symbol_out["symbol_embed"])
        head_input = torch.cat([hidden, interaction_belief_feat, physical_belief_feat, symbol_embed], dim=-1)

        logits_slippage_unary = self.head_slippage(head_input)
        logits_rotation_unary = self.head_rotation(head_input)
        force_unary = self.head_force(head_input)

        reason_out = self.reasoner(
            logits_slippage_unary=logits_slippage_unary,
            logits_rotation_unary=logits_rotation_unary,
            force_unary=force_unary,
        )
        return {
            "hidden": hidden,
            "skip_hidden": hidden_skip,
            "skip_gate": gate,
            "symbol_logits": symbol_out["symbol_logits"],
            "symbol_probs": symbol_out["symbol_probs"],
            "symbol_embed": symbol_embed,
            "symbol_id": symbol_out["symbol_id"],
            "logits": {
                "slippage": reason_out["logits_slippage"],
                "rotation": reason_out["logits_rotation"],
                "force": reason_out["force"],
            },
            "logits_unary": {
                "slippage": logits_slippage_unary,
                "rotation": logits_rotation_unary,
                "force": force_unary,
            },
            "probs": {
                "slippage": reason_out["probs_slippage"],
                "rotation": reason_out["probs_rotation"],
            },
            "prev_probs": {
                "slippage": reason_out["prev_probs_slippage"],
                "rotation": reason_out["prev_probs_rotation"],
            },
            "reason_messages": {
                "slippage_from_rotation": reason_out["msg_slippage_from_rotation"],
                "slippage_from_force": reason_out["msg_slippage_from_force"],
                "rotation_from_slippage": reason_out["msg_rotation_from_slippage"],
                "rotation_from_force": reason_out["msg_rotation_from_force"],
                "force_from_slippage": reason_out["msg_force_from_slippage"],
                "force_from_rotation": reason_out["msg_force_from_rotation"],
            },
        }


class Stage3HierarchicalModel(nn.Module):
    def __init__(
        self,
        backbone: Stage2Model,
        num_material: int,
        num_hardness: int,
        num_roughness: int,
        num_deformation: int,
        num_slippage: int,
        num_rotation: int,
        num_force: int,
        obs_symbol_k: int = 64,
        obs_symbol_dim: int = 64,
        physical_symbol_k: int = 32,
        physical_symbol_dim: int = 64,
        interaction_symbol_k: int = 16,
        interaction_symbol_dim: int = 64,
        contact_symbol_k: int = 32,
        contact_symbol_dim: int = 64,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        reason_steps_physical: int = 2,
        reason_steps_contact: int = 2,
        reasoner_init_std: float = 1e-3,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone_is_frozen = bool(freeze_backbone)
        self.contact_dim = int(getattr(backbone, "projection_dim", 128))
        self.encoder_dim = _infer_encoder_output_dim(backbone)
        self.fused_dim = self.encoder_dim + self.contact_dim
        self.hidden_dim = int(hidden_dim)
        self.obs_symbol_dim = int(obs_symbol_dim)
        self.physical_symbol_dim = int(physical_symbol_dim)
        self.interaction_symbol_dim = int(interaction_symbol_dim)
        self.contact_symbol_dim = int(contact_symbol_dim)
        self._backbone_default_requires_grad = [bool(p.requires_grad) for p in self.backbone.parameters()]

        self.encoder_norm = nn.LayerNorm(self.encoder_dim)
        self.contact_norm = nn.LayerNorm(self.contact_dim)
        self.obs_symbolize = SymbolizeLayer(
            input_dim=self.fused_dim,
            symbol_k=obs_symbol_k,
            symbol_dim=obs_symbol_dim,
        )
        self.obs_symbol_norm = nn.LayerNorm(obs_symbol_dim)

        self.physical = PhysicalModule(
            encoder_dim=self.encoder_dim,
            contact_dim=self.contact_dim,
            fused_dim=self.fused_dim,
            obs_symbol_dim=obs_symbol_dim,
            symbol_k=physical_symbol_k,
            symbol_dim=physical_symbol_dim,
            hidden_dim=hidden_dim,
            num_material=num_material,
            num_hardness=num_hardness,
            num_roughness=num_roughness,
            dropout=dropout,
            reason_steps=reason_steps_physical,
            reasoner_init_std=reasoner_init_std,
        )
        self.interaction = InteractionModule(
            fused_dim=self.fused_dim,
            obs_symbol_dim=obs_symbol_dim,
            physical_symbol_dim=physical_symbol_dim,
            physical_belief_dim=hidden_dim,
            symbol_k=interaction_symbol_k,
            symbol_dim=interaction_symbol_dim,
            hidden_dim=hidden_dim,
            num_deformation=num_deformation,
            dropout=dropout,
        )
        self.contact = ContactModule(
            fused_dim=self.fused_dim,
            obs_symbol_dim=obs_symbol_dim,
            interaction_symbol_dim=interaction_symbol_dim,
            interaction_belief_dim=hidden_dim,
            physical_symbol_dim=physical_symbol_dim,
            physical_belief_dim=hidden_dim,
            symbol_k=contact_symbol_k,
            symbol_dim=contact_symbol_dim,
            hidden_dim=hidden_dim,
            num_slippage=num_slippage,
            num_rotation=num_rotation,
            num_force=num_force,
            dropout=dropout,
            reason_steps=reason_steps_contact,
            reasoner_init_std=reasoner_init_std,
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

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad = bool(trainable)

    def set_physical_trainable(self, trainable: bool) -> None:
        self._set_module_trainable(self.physical, trainable)

    def set_interaction_trainable(self, trainable: bool) -> None:
        self._set_module_trainable(self.interaction, trainable)

    def set_contact_trainable(self, trainable: bool) -> None:
        self._set_module_trainable(self.contact, trainable)

    def set_reasoners_trainable(self, trainable: bool) -> None:
        self._set_module_trainable(self.physical.reasoner, trainable)
        self._set_module_trainable(self.contact.reasoner, trainable)

    def set_physical_reasoner_trainable(self, trainable: bool) -> None:
        self._set_module_trainable(self.physical.reasoner, trainable)

    def set_contact_reasoner_trainable(self, trainable: bool) -> None:
        self._set_module_trainable(self.contact.reasoner, trainable)

    def _relation_values(self) -> torch.Tensor:
        values = [
            self.physical.reasoner.relation_values().abs(),
            self.contact.reasoner.relation_values().abs(),
        ]
        return torch.cat(values, dim=0)

    def relation_l1(self) -> torch.Tensor:
        return self.physical.reasoner.relation_l1() + self.contact.reasoner.relation_l1()

    def relation_abs_mean(self) -> torch.Tensor:
        values = self._relation_values()
        return values.mean()

    def relation_sparsity(self, threshold: float = 1e-3) -> torch.Tensor:
        values = self._relation_values()
        return (values < float(threshold)).to(values.dtype).mean()

    def export_relation_state(self) -> Dict[str, torch.Tensor]:
        payload = {}
        payload.update(self.physical.reasoner.export_state(prefix="physical"))
        payload.update(self.contact.reasoner.export_state(prefix="contact"))
        return payload

    def forward(self, x: torch.Tensor, symbol_temperature: float = 1.0) -> Dict[str, object]:
        if self.backbone_is_frozen:
            with torch.no_grad():
                contact_feat, encoder_feat = self.backbone(x, return_encoder_output=True)
        else:
            contact_feat, encoder_feat = self.backbone(x, return_encoder_output=True)

        encoder_feat = self.encoder_norm(encoder_feat)
        contact_feat = self.contact_norm(contact_feat)
        fused_feat = torch.cat([encoder_feat, contact_feat], dim=-1)

        obs_symbol_out = self.obs_symbolize(fused_feat, temperature=symbol_temperature)
        obs_symbol_embed = self.obs_symbol_norm(obs_symbol_out["symbol_embed"])

        physical_out = self.physical(
            fused_feat=fused_feat,
            encoder_feat=encoder_feat,
            contact_feat=contact_feat,
            obs_symbol_embed=obs_symbol_embed,
            symbol_temperature=symbol_temperature,
        )
        interaction_out = self.interaction(
            fused_feat=fused_feat,
            obs_symbol_embed=obs_symbol_embed,
            physical_symbol_embed=physical_out["symbol_embed"],
            physical_belief_feat=physical_out["belief_feat"],
            symbol_temperature=symbol_temperature,
        )
        contact_out = self.contact(
            fused_feat=fused_feat,
            obs_symbol_embed=obs_symbol_embed,
            interaction_symbol_embed=interaction_out["symbol_embed"],
            interaction_belief_feat=interaction_out["belief_feat"],
            physical_symbol_embed=physical_out["symbol_embed"],
            physical_belief_feat=physical_out["belief_feat"],
            symbol_temperature=symbol_temperature,
        )

        logits = {}
        probs = {}
        prev_probs = {}
        logits_unary = {}
        reason_messages = {}
        for module_out in (physical_out, interaction_out, contact_out):
            logits.update(module_out.get("logits", {}))
            probs.update(module_out.get("probs", {}))
            prev_probs.update(module_out.get("prev_probs", {}))
            logits_unary.update(module_out.get("logits_unary", {}))
            reason_messages.update(module_out.get("reason_messages", {}))

        return {
            "feat_encoder": encoder_feat,
            "feat_contact": contact_feat,
            "feat_fused_base": fused_feat,
            "symbols": {
                "obs": {
                    "symbol_logits": obs_symbol_out["symbol_logits"],
                    "symbol_probs": obs_symbol_out["symbol_probs"],
                    "symbol_embed": obs_symbol_embed,
                    "symbol_id": obs_symbol_out["symbol_id"],
                },
                "physical": {
                    "symbol_logits": physical_out["symbol_logits"],
                    "symbol_probs": physical_out["symbol_probs"],
                    "symbol_embed": physical_out["symbol_embed"],
                    "symbol_id": physical_out["symbol_id"],
                },
                "interaction": {
                    "symbol_logits": interaction_out["symbol_logits"],
                    "symbol_probs": interaction_out["symbol_probs"],
                    "symbol_embed": interaction_out["symbol_embed"],
                    "symbol_id": interaction_out["symbol_id"],
                },
                "contact": {
                    "symbol_logits": contact_out["symbol_logits"],
                    "symbol_probs": contact_out["symbol_probs"],
                    "symbol_embed": contact_out["symbol_embed"],
                    "symbol_id": contact_out["symbol_id"],
                },
            },
            "layers": {
                "physical": physical_out,
                "interaction": interaction_out,
                "contact": contact_out,
            },
            "logits": logits,
            "logits_unary": logits_unary,
            "probs": probs,
            "prev_probs": prev_probs,
            "reason_messages": reason_messages,
            "aux": {
                "physical_belief_feat": physical_out["belief_feat"],
                "interaction_belief_feat": interaction_out["belief_feat"],
                "contact_skip_gate_mean": contact_out["skip_gate"].mean(dim=-1),
            },
        }


__all__ = [
    "Stage2BackboneMeta",
    "load_stage2_backbone",
    "Stage3HierarchicalModel",
]
