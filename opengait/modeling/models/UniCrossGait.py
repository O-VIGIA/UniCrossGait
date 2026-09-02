"""Reference OpenGait implementation of UniCrossGait.

Paper: UniCrossGait: Unified Cross-Modal Gait Recognition Based on
Knowledge Distillation, IEEE Transactions on Multimedia, 2026.

This file is an overlay for OpenGait, not a standalone training framework.
It intentionally presents the paper's final method without the commented
ablation branches and machine-specific code in the authors' research snapshot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..base_model import BaseModel
from ..modules import (
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
)
from ..losses.unicrossgait_math import distillation_balance, linear_warmup


class CrossModalAttention2D(nn.Module):
    """Multi-head cross-attention over spatial tokens, Eqs. (3)-(5)."""

    def __init__(
        self,
        in_channels: int,
        attention_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(in_channels, attention_dim)
        self.key = nn.Linear(in_channels, attention_dim)
        self.value = nn.Linear(in_channels, attention_dim)
        self.output = nn.Linear(attention_dim, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query_map: torch.Tensor, context_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if query_map.ndim != 4 or context_map.ndim != 4:
            raise ValueError("cross-attention inputs must be [batch, channels, height, width]")
        if query_map.shape[0] != context_map.shape[0]:
            raise ValueError("query and context batch sizes must match")

        batch, _, height, width = query_map.shape
        query_tokens = query_map.flatten(2).transpose(1, 2)
        context_tokens = context_map.flatten(2).transpose(1, 2)

        query = self._split_heads(self.query(query_tokens))
        key = self._split_heads(self.key(context_tokens))
        value = self._split_heads(self.value(context_tokens))

        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        attended = torch.matmul(self.dropout(attention), value)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch, height * width, -1)
        attended = self.output(attended).transpose(1, 2)
        return attended.reshape(batch, -1, height, width), attention

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        return tokens.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)


class BidirectionalCrossAttentionFusion(nn.Module):
    """Bidirectional camera/depth fusion followed by Eq. (6)'s MLP."""

    def __init__(
        self,
        channels: int = 512,
        attention_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        negative_slope: float = 0.02,
    ) -> None:
        super().__init__()
        self.cross_2d_to_3d = CrossModalAttention2D(
            channels, attention_dim, num_heads, dropout
        )
        self.cross_3d_to_2d = CrossModalAttention2D(
            channels, attention_dim, num_heads, dropout
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        features_2d_to_3d, _ = self.cross_2d_to_3d(features_2d, features_3d)
        features_3d_to_2d, _ = self.cross_3d_to_2d(features_3d, features_2d)
        return self.fusion_mlp(
            torch.cat((features_2d_to_3d, features_3d_to_2d), dim=1)
        )


class MultiModalTeacher(nn.Module):
    """Multimodal teacher network from Eqs. (2)-(7)."""

    def __init__(
        self,
        camera_backbone: nn.Module,
        depth_backbone: nn.Module,
        bin_num,
        separate_fcs: Mapping,
        separate_bn_necks: Mapping,
        fusion_cfg: Mapping,
    ) -> None:
        super().__init__()
        self.camera_encoder = SetBlockWrapper(camera_backbone)
        self.depth_encoder = SetBlockWrapper(depth_backbone)
        self.temporal_pool = PackSequenceWrapper(torch.max)
        self.fusion = BidirectionalCrossAttentionFusion(**dict(fusion_cfg))
        self.hpp = HorizontalPoolingPyramid(bin_num=bin_num)
        self.fcs = SeparateFCs(**dict(separate_fcs))
        self.bn_necks = SeparateBNNecks(**dict(separate_bn_necks))

    def forward(
        self,
        camera: torch.Tensor,
        depth: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features_2d = self.camera_encoder(camera)
        features_3d = self.depth_encoder(depth)
        features_2d = self.temporal_pool(
            features_2d, sequence_lengths, options={"dim": 2}
        )[0]
        features_3d = self.temporal_pool(
            features_3d, sequence_lengths, options={"dim": 2}
        )[0]
        fused = self.fusion(features_2d, features_3d)
        embedding = self.fcs(self.hpp(fused))
        _, logits = self.bn_necks(embedding)
        return embedding, logits


class UniCrossGait(BaseModel):
    """Two-stage UniCrossGait model integrated with OpenGait's ``BaseModel``.

    ``model_cfg.stage`` controls construction:

    - ``teacher`` trains the multimodal fusion teacher.
    - ``student`` trains both unimodal students when ``training=True`` and
      builds only the student branches for inference.
    """

    def __init__(self, cfgs, training: bool):
        object.__setattr__(self, "_constructing_for_training", bool(training))
        super().__init__(cfgs, training)

    def build_network(self, model_cfg: Mapping) -> None:
        self.stage = str(model_cfg.get("stage", "student")).lower()
        if self.stage not in {"teacher", "student"}:
            raise ValueError("model_cfg.stage must be 'teacher' or 'student'")

        self.camera_input_index = int(model_cfg.get("camera_input_index", 1))
        self.depth_input_index = int(model_cfg.get("depth_input_index", 0))
        self.bin_num = list(model_cfg["bin_num"])
        self.temperature = float(model_cfg.get("temperature", 4.0))
        self.warmup_start = float(model_cfg.get("warmup_start", 0.1))
        self.warmup_end = float(model_cfg.get("warmup_end", 1.5))
        self.warmup_iterations = int(model_cfg.get("warmup_iterations", 40000))
        self.detach_balance = bool(model_cfg.get("detach_balance", False))
        self.use_student_ce = bool(model_cfg.get("use_student_ce", False))
        self._teacher_frozen = False

        fcs_cfg = dict(model_cfg["SeparateFCs"])
        bn_cfg = dict(model_cfg["SeparateBNNecks"])
        expected_parts = sum(self.bin_num)
        if fcs_cfg["parts_num"] != expected_parts or bn_cfg["parts_num"] != expected_parts:
            raise ValueError(
                "SeparateFCs/SeparateBNNecks parts_num must equal sum(bin_num)"
            )

        if self.stage == "teacher":
            self.teacher = self._build_teacher(model_cfg, fcs_cfg, bn_cfg)
            return

        self.camera_student = SetBlockWrapper(
            self.get_backbone(model_cfg["backbone_cfg_2d"])
        )
        self.depth_student = SetBlockWrapper(
            self.get_backbone(model_cfg["backbone_cfg_3d"])
        )
        self.temporal_pool = PackSequenceWrapper(torch.max)
        self.hpp = HorizontalPoolingPyramid(bin_num=self.bin_num)
        # These two heads are shared across modalities, as specified in Sec. III-C2.
        self.shared_fcs = SeparateFCs(**fcs_cfg)
        self.shared_bn_necks = SeparateBNNecks(**bn_cfg)

        if self._constructing_for_training:
            self.teacher = self._build_teacher(model_cfg, fcs_cfg, bn_cfg)
            self.teacher_checkpoint = str(model_cfg.get("teacher_checkpoint", ""))
            self.teacher_checkpoint_strict = bool(
                model_cfg.get("teacher_checkpoint_strict", True)
            )

    def _build_teacher(
        self, model_cfg: Mapping, fcs_cfg: Mapping, bn_cfg: Mapping
    ) -> MultiModalTeacher:
        return MultiModalTeacher(
            camera_backbone=self.get_backbone(model_cfg["backbone_cfg_2d"]),
            depth_backbone=self.get_backbone(model_cfg["backbone_cfg_3d"]),
            bin_num=self.bin_num,
            separate_fcs=fcs_cfg,
            separate_bn_necks=bn_cfg,
            fusion_cfg=model_cfg["fusion_cfg"],
        )

    def init_parameters(self) -> None:
        super().init_parameters()
        if self.stage == "student" and self._constructing_for_training:
            self._load_teacher_checkpoint(
                self.teacher_checkpoint, self.teacher_checkpoint_strict
            )
            self._freeze_teacher()

    def _load_teacher_checkpoint(self, checkpoint_path: str, strict: bool) -> None:
        if not checkpoint_path:
            raise ValueError(
                "student training requires model_cfg.teacher_checkpoint; "
                "train the stage-1 teacher first"
            )
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"teacher checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, Mapping):
            raise TypeError("teacher checkpoint does not contain a state dictionary")

        expected = self.teacher.state_dict()
        teacher_state: Dict[str, torch.Tensor] = {}
        for raw_key, value in state.items():
            key = raw_key[7:] if raw_key.startswith("module.") else raw_key
            if key.startswith("teacher."):
                key = key[len("teacher.") :]
            if key in expected:
                teacher_state[key] = value

        if not teacher_state:
            raise KeyError(
                "no teacher parameters matched; use a checkpoint produced by "
                "configs/unicrossgait_sustech1k_teacher.yaml"
            )
        incompatible = self.teacher.load_state_dict(teacher_state, strict=strict)
        if not strict:
            self.msg_mgr.log_warning(
                "Non-strict teacher restore: missing=%s, unexpected=%s"
                % (incompatible.missing_keys, incompatible.unexpected_keys)
            )

    def _freeze_teacher(self) -> None:
        self._teacher_frozen = True
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._teacher_frozen and hasattr(self, "teacher"):
            # BaseModel calls train(True) after construction; keep BatchNorm and
            # Dropout in the frozen teacher deterministic throughout stage 2.
            self.teacher.eval()
        return self

    @staticmethod
    def _as_open_gait_sequence(sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim == 4:  # [N, S, H, W]
            return sequence.unsqueeze(1)
        if sequence.ndim == 5:  # [N, S, C, H, W]
            return sequence.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError("each modality must be [N,S,H,W] or [N,S,C,H,W]")

    def _unpack_inputs(self, inputs):
        modalities, labels, _, _, sequence_lengths = inputs
        largest_index = max(self.camera_input_index, self.depth_input_index)
        if len(modalities) <= largest_index:
            raise IndexError(
                "modal input indices do not match data_cfg.data_in_use/preprocessing order"
            )
        camera = self._as_open_gait_sequence(modalities[self.camera_input_index])
        depth = self._as_open_gait_sequence(modalities[self.depth_input_index])
        return camera, depth, labels, sequence_lengths

    def _student_forward(
        self, camera: torch.Tensor, depth: torch.Tensor, sequence_lengths
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features_2d = self.camera_student(camera)
        features_3d = self.depth_student(depth)
        features_2d = self.temporal_pool(
            features_2d, sequence_lengths, options={"dim": 2}
        )[0]
        features_3d = self.temporal_pool(
            features_3d, sequence_lengths, options={"dim": 2}
        )[0]
        embedding_2d = self.shared_fcs(self.hpp(features_2d))
        embedding_3d = self.shared_fcs(self.hpp(features_3d))
        _, logits_2d = self.shared_bn_necks(embedding_2d)
        _, logits_3d = self.shared_bn_necks(embedding_3d)
        return embedding_2d, embedding_3d, logits_2d, logits_3d

    def forward(self, inputs):
        camera, depth, labels, sequence_lengths = self._unpack_inputs(inputs)

        if self.stage == "teacher":
            embedding, logits = self.teacher(camera, depth, sequence_lengths)
            if self.training:
                return {
                    "training_feat": {
                        "triplet_fusion": {
                            "embeddings_2d": embedding,
                            "embeddings_3d": embedding,
                            "labels": labels,
                        },
                        "softmax_fusion": {"logits": logits, "labels": labels},
                    },
                    "visual_summary": {},
                }
            return {
                "inference_feat": {
                    "embeddings_2d": embedding,
                    "embeddings_3d": embedding,
                }
            }

        embedding_2d, embedding_3d, logits_2d, logits_3d = self._student_forward(
            camera, depth, sequence_lengths
        )
        if not self.training:
            return {
                "inference_feat": {
                    "embeddings_2d": embedding_2d,
                    "embeddings_3d": embedding_3d,
                }
            }
        if not hasattr(self, "teacher"):
            raise RuntimeError("the frozen teacher is required for student training")

        with torch.no_grad():
            teacher_embedding, _ = self.teacher(camera, depth, sequence_lengths)

        mu_2d, mu_3d, theta_2d, theta_3d = distillation_balance(
            logits_2d,
            logits_3d,
            labels,
            temperature=self.temperature,
            detach=self.detach_balance,
        )
        phi = linear_warmup(
            self.iteration,
            self.warmup_iterations,
            self.warmup_start,
            self.warmup_end,
        )
        weight_2d = phi * mu_2d
        weight_3d = phi * mu_3d

        training_features = {
            # Two directions, weighted by 0.5 each in the YAML, form L_tri_cross.
            "triplet_2d3d": {
                "embeddings_2d": embedding_2d,
                "embeddings_3d": embedding_3d,
                "labels": labels,
            },
            "triplet_3d2d": {
                "embeddings_2d": embedding_3d,
                "embeddings_3d": embedding_2d,
                "labels": labels,
            },
            "dfi_2d": {
                "teacher_features": teacher_embedding,
                "student_features": embedding_2d,
                "ada_weights": weight_2d,
            },
            "dfi_3d": {
                "teacher_features": teacher_embedding,
                "student_features": embedding_3d,
                "ada_weights": weight_3d,
            },
            "intra_c_2d": {
                "teacher_features": teacher_embedding,
                "student_features": embedding_2d,
                "ada_weights": 0.5 * weight_2d,
            },
            "intra_c_3d": {
                "teacher_features": teacher_embedding,
                "student_features": embedding_3d,
                "ada_weights": 0.5 * weight_3d,
            },
            "inter_c_2d": {
                "teacher_features": teacher_embedding,
                "student_features": embedding_2d,
                "ada_weights": 0.5 * weight_2d,
            },
            "inter_c_3d": {
                "teacher_features": teacher_embedding,
                "student_features": embedding_3d,
                "ada_weights": 0.5 * weight_3d,
            },
        }
        if self.use_student_ce:
            training_features.update(
                {
                    "softmax_2d": {"logits": logits_2d, "labels": labels},
                    "softmax_3d": {"logits": logits_3d, "labels": labels},
                }
            )

        return {
            "training_feat": training_features,
            "visual_summary": {
                "scalar/distill_phi": torch.as_tensor(phi, device=labels.device),
                "scalar/distill_mu_2d": mu_2d.detach(),
                "scalar/distill_mu_3d": mu_3d.detach(),
                "scalar/distill_theta_2d": theta_2d.detach(),
                "scalar/distill_theta_3d": theta_3d.detach(),
            },
        }


__all__ = ["UniCrossGait"]
