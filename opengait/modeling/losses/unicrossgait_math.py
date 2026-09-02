"""Pure PyTorch operators used by UniCrossGait.

This module deliberately has no OpenGait imports.  Keeping the equations here
makes them easy to audit and unit test independently from the training engine.
Feature tensors follow OpenGait's ``[batch, channels, parts]`` convention and
classifier logits use ``[batch, classes, parts]``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def linear_warmup(
    iteration: int,
    total_iterations: int,
    start: float = 0.1,
    end: float = 1.5,
) -> float:
    """Return the clamped warm-up factor from Eq. (20) of the paper."""

    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if end < start:
        raise ValueError("end must be greater than or equal to start")

    progress = min(max(float(iteration), 0.0), float(total_iterations))
    return start + (end - start) * progress / float(total_iterations)


def _true_class_confidence(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, classes, parts]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    probabilities = F.softmax(logits / temperature, dim=1)
    indices = labels.to(device=logits.device, dtype=torch.long)
    indices = indices[:, None, None].expand(-1, 1, logits.shape[2])
    return probabilities.gather(dim=1, index=indices).mean()


def distillation_balance(
    logits_2d: torch.Tensor,
    logits_3d: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    eps: float = 1e-12,
    detach: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Distillation Balancing (DB) factors from Eqs. (17)-(19).

    The common ``1 / num_classes`` factor in Eq. (18) cancels in Eq. (19), so
    the implementation averages the ground-truth probabilities over batch and
    parts.  ``mu_2d`` is driven by the *3D* intensity (and vice versa), which
    increases supervision for the weaker branch.

    Returns:
        ``(mu_2d, mu_3d, theta_2d, theta_3d)`` as scalar tensors.
    """

    if logits_2d.shape != logits_3d.shape:
        raise ValueError("2D and 3D logits must have identical shapes")

    theta_2d = _true_class_confidence(logits_2d, labels, temperature)
    theta_3d = _true_class_confidence(logits_3d, labels, temperature)
    if detach:
        theta_2d = theta_2d.detach()
        theta_3d = theta_3d.detach()

    denominator = (theta_2d + theta_3d).clamp_min(eps)
    mu_2d = 0.5 * (1.0 + torch.tanh(theta_3d / denominator))
    mu_3d = 0.5 * (1.0 + torch.tanh(theta_2d / denominator))
    return mu_2d, mu_3d, theta_2d, theta_3d


def direction_feature_imitation(
    teacher_features: torch.Tensor,
    student_features: torch.Tensor,
) -> torch.Tensor:
    """Direction-level Feature Imitation (DFI), Eq. (11)."""

    _check_feature_pair(teacher_features, student_features)
    cosine = F.cosine_similarity(
        teacher_features.float(), student_features.float(), dim=1
    )
    return 1.0 - cosine.mean()


def inter_sample_affinity(features: torch.Tensor) -> torch.Tensor:
    """Return the holistic sample affinity matrix from Eqs. (12)-(13)."""

    _check_features(features)
    holistic = features.float().amax(dim=2)
    holistic = F.normalize(holistic, p=2, dim=1)
    return holistic @ holistic.transpose(0, 1)


def intra_sample_affinity(features: torch.Tensor) -> torch.Tensor:
    """Return the part-to-part affinity tensor from Eq. (15)."""

    _check_features(features)
    parts = F.normalize(features.float().transpose(1, 2), p=2, dim=2)
    return torch.bmm(parts, parts.transpose(1, 2))


def _check_features(features: torch.Tensor) -> None:
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, channels, parts]")


def _check_feature_pair(
    teacher_features: torch.Tensor,
    student_features: torch.Tensor,
) -> None:
    _check_features(teacher_features)
    _check_features(student_features)
    if teacher_features.shape != student_features.shape:
        raise ValueError("teacher and student features must have identical shapes")
