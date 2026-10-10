"""
tests/test_linear_classifier.py
===============================
``LinearClassifier`` (#501): one affine map to a logit, the linear head of
the Rydberg feature benchmark and, trained with the cross-entropy, logistic
regression.

The numbers are checked against a logit worked out by hand, the closed-form
maximum-likelihood fit of a one-feature binary problem, and the Xavier bound
of the initialisation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from hqnn_forge.models import BinaryClassifierBase, LinearClassifier
from hqnn_forge.training import train_model
from hqnn_forge.utils import load_checkpoint, save_checkpoint


def _set(model: LinearClassifier, weight: list[float], bias: float) -> None:
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([weight]))
        model.head.bias.fill_(bias)


class TestLogit:
    def test_closed_form_logit(self) -> None:
        """``z = Σ_j w_j x_j + b``, by hand: 0.5·1 − 2·3 + 0.25·(−4) + 1.5 = −5."""
        model = LinearClassifier(3)
        _set(model, [0.5, -2.0, 0.25], 1.5)
        x = torch.tensor([[1.0, 3.0, -4.0], [0.0, 0.0, 0.0], [2.0, 1.0, 4.0]])
        logits = model(x)
        assert logits.shape == (3, 1)
        assert logits.squeeze(-1).tolist() == [-5.0, 1.5, 1.5]

    def test_probability_is_the_logistic_function_of_the_logit(self) -> None:
        model = LinearClassifier(2)
        _set(model, [1.0, -1.0], math.log(3.0))  # σ(ln 3) = 3/4
        x = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0], [0.0, 2 * math.log(3.0)]])
        # logits ln 3, 2 ln 3, −ln 3: probabilities 3/4, 9/10, 1/4
        torch.testing.assert_close(
            model.predict_proba(x), torch.tensor([0.75, 0.9, 0.25]), rtol=0, atol=1e-6
        )
        assert model.predict(x).tolist() == [1, 1, 0]
        assert model.predict(x, threshold=0.8).tolist() == [0, 1, 0]

    def test_the_decision_boundary_is_the_hyperplane_w_x_plus_b_equal_zero(self) -> None:
        """A step along a direction orthogonal to ``w`` does not change the logit."""
        model = LinearClassifier(2)
        _set(model, [3.0, 4.0], -5.0)
        on_plane = torch.tensor([[3.0, -1.0]])  # 9 − 4 − 5 = 0
        along = torch.tensor([[4.0, -3.0]])  # orthogonal to (3, 4)
        assert model(on_plane).item() == 0.0
        assert model(on_plane + 7 * along).item() == 0.0
        assert model(on_plane + torch.tensor([[0.3, 0.4]])).item() == pytest.approx(2.5)


class TestParameters:
    @pytest.mark.parametrize("n", [1, 4, 6, 30])
    def test_parameter_count_is_n_plus_one(self, n: int) -> None:
        model = LinearClassifier(n)
        assert model.count_parameters() == n + 1
        assert [tuple(p.shape) for p in model.parameters()] == [(1, n), (1,)]

    def test_is_a_binary_classifier_with_a_linear_head(self) -> None:
        model = LinearClassifier(5)
        assert isinstance(model, BinaryClassifierBase)
        assert isinstance(model.head, nn.Linear)
        assert model.n_input_features == 5
        # The head is the whole model: no other module holds a parameter.
        assert [name for name, _ in model.named_parameters()] == ["head.weight", "head.bias"]

    def test_initialisation_is_xavier_uniform_with_a_zero_bias(self) -> None:
        """Weights within ``±√(6 / (n + 1))`` and filling that range; bias 0."""
        n = 400
        model = LinearClassifier(n, init_seed=0)
        bound = math.sqrt(6.0 / (n + 1))
        weight = model.head.weight.detach()
        assert float(weight.abs().max()) <= bound
        # A uniform draw on (−a, a) has standard deviation a/√3; 400 draws
        # estimate it to about 1/√800 = 3.5 %.
        assert float(weight.std()) == pytest.approx(bound / math.sqrt(3.0), rel=0.15)
        assert float(weight.abs().max()) > 0.9 * bound
        assert model.head.bias.item() == 0.0

    def test_init_seed(self) -> None:
        """As on the other classifiers: reproducible, the global RNG untouched."""
        torch.manual_seed(1)
        a = LinearClassifier(4, init_seed=np.int64(3))  # type: ignore[arg-type]
        torch.manual_seed(2)
        before = torch.random.get_rng_state()
        b = LinearClassifier(4, init_seed=3)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert type(a.get_config()["init_seed"]) is int
        assert torch.equal(a.head.weight, b.head.weight)
        assert not torch.equal(a.head.weight, LinearClassifier(4, init_seed=4).head.weight)
        with pytest.raises(TypeError, match="init_seed"):
            LinearClassifier(4, init_seed=1.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -2, 2.0, True, "3", None])
    def test_invalid_width(self, bad: object) -> None:
        torch.manual_seed(5)
        before = torch.random.get_rng_state()
        with pytest.raises(ValueError, match="n_input_features must be an integer >= 1"):
            LinearClassifier(bad, init_seed=1)  # type: ignore[arg-type]
        assert torch.equal(torch.random.get_rng_state(), before)


class TestConfig:
    def test_get_config_rebuilds_the_model(self) -> None:
        model = LinearClassifier(6, init_seed=11)
        assert model.get_config() == {"n_input_features": 6, "init_seed": 11}
        rebuilt = LinearClassifier(**model.get_config())
        assert torch.equal(rebuilt.head.weight, model.head.weight)

    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        model = LinearClassifier(3)
        _set(model, [0.5, -2.0, 0.25], 1.5)
        save_checkpoint(model, tmp_path / "linear.pt")
        loaded = load_checkpoint(tmp_path / "linear.pt")
        assert isinstance(loaded, LinearClassifier)
        assert loaded(torch.tensor([[1.0, 3.0, -4.0]])).item() == -5.0


class TestLogisticRegression:
    def test_cross_entropy_training_reaches_the_closed_form_fit(self) -> None:
        """
        One binary feature has a closed-form maximum-likelihood fit: the model
        has two parameters and two groups, so it reproduces each group's
        positive rate exactly, ``σ(b) = p_0`` and ``σ(w + b) = p_1``.  With 2
        of 8 positives at ``x = 0`` and 9 of 12 at ``x = 1``:
        ``b = ln(1/3)``, ``w = ln 3 − ln(1/3) = 2 ln 3``.
        """
        x = torch.tensor([0.0] * 8 + [1.0] * 12, dtype=torch.float64).reshape(-1, 1)
        y = torch.tensor([1.0] * 2 + [0.0] * 6 + [1.0] * 9 + [0.0] * 3, dtype=torch.float64)
        model = LinearClassifier(1, init_seed=0).double()
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            max_iter=500,
            tolerance_grad=1e-14,
            tolerance_change=1e-16,
            line_search_fn="strong_wolfe",
        )
        loss_fn = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            return loss

        optimizer.step(closure)
        assert model.head.bias.item() == pytest.approx(math.log(1 / 3), abs=1e-7)
        assert model.head.weight.item() == pytest.approx(2 * math.log(3.0), abs=1e-7)
        torch.testing.assert_close(
            model.predict_proba(torch.tensor([[0.0], [1.0]], dtype=torch.float64)),
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            rtol=0,
            atol=1e-8,
        )

    def test_trains_under_train_model(self) -> None:
        """A linearly separable problem is learnt: the validation MCC reaches 1."""
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(200, 3, generator=generator)
        y = (x @ torch.tensor([1.0, -2.0, 0.5]) > 0.2).float()
        model = LinearClassifier(3, init_seed=0)
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.1),
            x[:150],
            y[:150],
            x[150:],
            y[150:],
            max_epochs=60,
            batch_size=32,
            generator=torch.Generator().manual_seed(0),
        )
        assert history.best_value == 1.0
        # The fitted direction is the one the labels were made from.
        weight = model.head.weight.detach().squeeze(0)
        cosine = float(weight @ torch.tensor([1.0, -2.0, 0.5])) / float(
            weight.norm() * math.sqrt(5.25)
        )
        assert cosine > 0.95
