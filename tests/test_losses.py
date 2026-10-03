"""
tests/test_losses.py
=====================
Unit tests for hqnn_forge.utils.imbalance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hqnn_forge.utils.imbalance import FocalLoss, compute_class_weights, weighted_bce_loss


@pytest.fixture
def balanced_labels() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])


@pytest.fixture
def imbalanced_labels() -> torch.Tensor:
    return torch.tensor([0.0] * 95 + [1.0] * 5)


@pytest.fixture
def sample_logits() -> torch.Tensor:
    return torch.tensor([0.8, -0.3, 1.2, -1.5, 0.1, -0.5, 0.9, -0.8])


@pytest.fixture
def sample_targets() -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


class TestFocalLoss:
    def test_gamma_zero_matches_bce(
        self, sample_logits: torch.Tensor, sample_targets: torch.Tensor
    ) -> None:
        focal_val = FocalLoss(alpha=0.5, gamma=0.0, reduction="mean")(
            sample_logits, sample_targets
        )
        bce_val = F.binary_cross_entropy_with_logits(
            sample_logits, sample_targets, reduction="mean"
        )
        torch.testing.assert_close(focal_val, 0.5 * bce_val, atol=1e-5, rtol=1e-5)

    def test_gradient_flows_to_logits(self) -> None:
        logits = torch.randn(16, requires_grad=True)
        targets = (torch.rand(16) > 0.5).float()
        loss = FocalLoss(alpha=0.25, gamma=2.0)(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0.0

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            FocalLoss(alpha=0.0)
        with pytest.raises(ValueError, match="gamma"):
            FocalLoss(gamma=-1.0)
        with pytest.raises(ValueError, match="reduction"):
            FocalLoss(reduction="avg")


class TestClassWeights:
    def test_balanced_weights_approximately_equal(self, balanced_labels: torch.Tensor) -> None:
        weights = compute_class_weights(balanced_labels)
        assert abs(weights[0].item() - weights[1].item()) < 0.3

    def test_extreme_ratio(self, imbalanced_labels: torch.Tensor) -> None:
        weights = compute_class_weights(imbalanced_labels)
        assert (weights[1].item() / weights[0].item()) > 5.0

    def test_exact_values(self) -> None:
        # w_c = N / (2 * n_c + smooth), with N = 10, n_neg = 9, n_pos = 1
        y = torch.tensor([0] * 9 + [1])
        torch.testing.assert_close(compute_class_weights(y), torch.tensor([10 / 19, 10 / 3]))
        torch.testing.assert_close(
            compute_class_weights(y, smooth=0.0), torch.tensor([10 / 18, 10 / 2])
        )


class TestWeightedBCELoss:
    # Asymmetric on purpose, so swapping w_neg and w_pos changes the result
    CLASS_WEIGHTS = torch.tensor([0.3, 2.5])

    def _reference(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_sample = torch.where(targets == 1.0, self.CLASS_WEIGHTS[1], self.CLASS_WEIGHTS[0])
        return F.binary_cross_entropy_with_logits(
            logits, targets, weight=per_sample, reduction="none"
        )

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_matches_weighted_bce(
        self, sample_logits: torch.Tensor, sample_targets: torch.Tensor, reduction: str
    ) -> None:
        expected = self._reference(sample_logits, sample_targets)
        expected = {"mean": expected.mean(), "sum": expected.sum(), "none": expected}[reduction]
        got = weighted_bce_loss(
            sample_logits, sample_targets, self.CLASS_WEIGHTS, reduction=reduction
        )
        torch.testing.assert_close(got, expected)

    def test_weights_apply_to_the_right_class(self) -> None:
        # One negative and one positive sample with identical unweighted loss
        logits = torch.tensor([0.7, -0.7])
        targets = torch.tensor([0.0, 1.0])
        per_sample = weighted_bce_loss(logits, targets, self.CLASS_WEIGHTS, reduction="none")
        unweighted = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        torch.testing.assert_close(per_sample, unweighted * self.CLASS_WEIGHTS)

    # "Mean" and "avg" used to fall through and return the per-sample tensor.
    @pytest.mark.parametrize("reduction", ["Mean", "avg", "", "None"])
    def test_rejects_unknown_reduction(
        self, sample_logits: torch.Tensor, sample_targets: torch.Tensor, reduction: str
    ) -> None:
        with pytest.raises(ValueError, match="reduction must be 'mean', 'sum', or 'none'"):
            weighted_bce_loss(
                sample_logits, sample_targets, self.CLASS_WEIGHTS, reduction=reduction
            )

    def test_column_logits_are_flattened(
        self, sample_logits: torch.Tensor, sample_targets: torch.Tensor
    ) -> None:
        flat = weighted_bce_loss(sample_logits, sample_targets, self.CLASS_WEIGHTS)
        column = weighted_bce_loss(
            sample_logits.unsqueeze(-1), sample_targets.unsqueeze(-1), self.CLASS_WEIGHTS
        )
        torch.testing.assert_close(column, flat)
