"""Unit tests for the paper equations that do not require OpenGait."""

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "opengait/modeling/losses/unicrossgait_math.py"
)
SPEC = importlib.util.spec_from_file_location("unicrossgait_math", MODULE_PATH)
math_ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math_ops)


def test_linear_warmup_is_clamped_and_hits_paper_endpoints():
    assert math_ops.linear_warmup(-5, 40000) == pytest.approx(0.1)
    assert math_ops.linear_warmup(20000, 40000) == pytest.approx(0.8)
    assert math_ops.linear_warmup(40000, 40000) == pytest.approx(1.5)
    assert math_ops.linear_warmup(50000, 40000) == pytest.approx(1.5)


def test_equal_branches_receive_equal_db_weights():
    logits = torch.tensor(
        [[[2.0, 1.0], [0.0, 0.5], [-1.0, -0.5]], [[0.0, 0.0], [2.0, 1.0], [-1.0, -0.5]]]
    )
    labels = torch.tensor([0, 1])
    mu_2d, mu_3d, _, _ = math_ops.distillation_balance(logits, logits, labels)
    expected = 0.5 * (1.0 + torch.tanh(torch.tensor(0.5)))
    assert mu_2d.item() == pytest.approx(expected.item())
    assert mu_3d.item() == pytest.approx(expected.item())


def test_stronger_3d_branch_increases_2d_distillation_weight():
    labels = torch.tensor([0, 1])
    logits_2d = torch.zeros(2, 3, 2)
    logits_3d = torch.tensor(
        [[[12.0, 12.0], [-8.0, -8.0], [-8.0, -8.0]], [[-8.0, -8.0], [12.0, 12.0], [-8.0, -8.0]]]
    )
    mu_2d, mu_3d, theta_2d, theta_3d = math_ops.distillation_balance(
        logits_2d, logits_3d, labels
    )
    assert theta_3d > theta_2d
    assert mu_2d > mu_3d
    assert 0.5 < mu_3d.item() < mu_2d.item() < 0.8808


def test_dfi_ignores_positive_feature_scale():
    torch.manual_seed(7)
    teacher = torch.randn(4, 8, 16)
    student = 5.0 * teacher
    loss = math_ops.direction_feature_imitation(teacher, student)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_affinity_shapes_symmetry_and_unit_diagonal():
    torch.manual_seed(11)
    features = torch.randn(3, 8, 4)
    inter = math_ops.inter_sample_affinity(features)
    intra = math_ops.intra_sample_affinity(features)
    assert inter.shape == (3, 3)
    assert intra.shape == (3, 4, 4)
    assert torch.allclose(inter, inter.transpose(0, 1), atol=1e-6)
    assert torch.allclose(torch.diagonal(inter), torch.ones(3), atol=1e-6)
    assert torch.allclose(
        torch.diagonal(intra, dim1=1, dim2=2), torch.ones(3, 4), atol=1e-6
    )
