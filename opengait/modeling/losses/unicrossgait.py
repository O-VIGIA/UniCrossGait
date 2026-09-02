"""Feature-distillation losses for UniCrossGait.

Copy this file and ``unicrossgait_math.py`` into
``opengait/modeling/losses``.  OpenGait's dynamic loss discovery then exposes
the classes below to YAML configurations.
"""

from __future__ import annotations

import torch

from .base import BaseLoss
from .unicrossgait_math import (
    direction_feature_imitation,
    inter_sample_affinity,
    intra_sample_affinity,
)


class DirectionFeatureImitationLoss(BaseLoss):
    """Cosine-direction imitation from Eq. (11)."""

    def forward(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        ada_weights=1.0,
    ):
        loss = direction_feature_imitation(teacher_features, student_features)
        loss = loss * ada_weights
        self.info.update({"loss": loss.detach().clone()})
        return loss, self.info


class InterSampleCorrelationLoss(BaseLoss):
    """Match global sample relationships from Eqs. (12)-(14)."""

    def forward(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        ada_weights=1.0,
    ):
        teacher_affinity = inter_sample_affinity(teacher_features)
        student_affinity = inter_sample_affinity(student_features)
        loss = (teacher_affinity - student_affinity).abs().mean()
        loss = loss * ada_weights
        self.info.update({"loss": loss.detach().clone()})
        return loss, self.info


class IntraSampleCorrelationLoss(BaseLoss):
    """Match local part relationships from Eqs. (15)-(16)."""

    def forward(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        ada_weights=1.0,
    ):
        teacher_affinity = intra_sample_affinity(teacher_features)
        student_affinity = intra_sample_affinity(student_features)
        loss = (teacher_affinity - student_affinity).abs().mean()
        loss = loss * ada_weights
        self.info.update({"loss": loss.detach().clone()})
        return loss, self.info
