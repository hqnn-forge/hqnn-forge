"""
tests/test_noise.py
===================
hqnn_forge.noise: post-hoc depolarizing noise and noise sweeps.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.noise import NoiseSweepPoint, apply_depolarizing_noise, noise_sweep

N_QUBITS = 3
Layer = QuantumEncodingLayer | IQPEncodingLayer
Model = HybridBinaryClassifier | ParallelHybridClassifier


# The noisy QNode is rebuilt on default.mixed with backprop, which drops the
# broadcast_expand wrapper the other differentiation methods need. These are the
# configurations where that rebuild actually changes something, including
# lightning.qubit/adjoint (the default until #349, and what "auto" picks above
# 12 qubits).
NON_BACKPROP_CONFIGS = [
    pytest.param("default.qubit", "parameter-shift", id="default.qubit/parameter-shift"),
    pytest.param(
        "lightning.qubit",
        "adjoint",
        id="lightning.qubit/adjoint",
        marks=pytest.mark.requires_lightning,
    ),
]


def _layer(
    cls: type[Layer] = QuantumEncodingLayer,
    diff_method: DiffMethod = "backprop",
    device_name: DeviceName = "default.qubit",
) -> Layer:
    torch.manual_seed(0)
    return cls(n_qubits=N_QUBITS, n_layers=2, device_name=device_name, diff_method=diff_method)


def _weights(layer: Layer) -> torch.Tensor:
    """The layer's variational weights, narrowed from ``nn.Module.__getattr__``."""
    weights = layer.qlayer.weights
    assert isinstance(weights, torch.Tensor)
    return weights


def _model(cls: type[Model] = HybridBinaryClassifier) -> Model:
    torch.manual_seed(0)
    return cls(
        n_input_features=5,
        n_qubits=N_QUBITS,
        n_layers=2,
        device_name="default.qubit",
        diff_method="backprop",
    )


@pytest.fixture
def x() -> torch.Tensor:
    return (torch.rand(8, N_QUBITS, generator=torch.Generator().manual_seed(1)) * 2 - 1) * math.pi


class TestNoiseEffect:
    def test_zero_noise_is_bit_identical(self, x: torch.Tensor) -> None:
        layer = _layer()
        with torch.no_grad():
            clean = layer(x)
            qnode = layer.qlayer.qnode
            with apply_depolarizing_noise(layer, 0.0):
                assert layer.qlayer.qnode is qnode
                torch.testing.assert_close(layer(x), clean, rtol=0, atol=0)

    @pytest.mark.parametrize("cls", [QuantumEncodingLayer, IQPEncodingLayer])
    @pytest.mark.parametrize("p", [0.05, 0.3, 0.75])
    def test_end_noise_damps_by_exactly_one_minus_four_thirds_p(
        self, cls: type, p: float, x: torch.Tensor
    ) -> None:
        layer = _layer(cls)
        with torch.no_grad():
            clean = layer(x)
            with apply_depolarizing_noise(layer, p, position="end"):
                noisy = layer(x)
        torch.testing.assert_close(noisy, (1 - 4 * p / 3) * clean, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(("device_name", "diff_method"), NON_BACKPROP_CONFIGS)
    def test_end_noise_damps_exactly_for_non_backprop_layers(
        self, device_name: DeviceName, diff_method: DiffMethod, x: torch.Tensor
    ) -> None:
        # The noisy QNode ignores the layer's own device and differentiation
        # method, so the analytic damping has to hold for a layer that was not
        # built for backprop, and the layer has to be unchanged afterwards.
        layer = _layer(diff_method=diff_method, device_name=device_name)
        p = 0.3
        with torch.no_grad():
            clean = layer(x)
            with apply_depolarizing_noise(layer, p, position="end"):
                noisy = layer(x)
            restored = layer(x)
        torch.testing.assert_close(noisy, (1 - 4 * p / 3) * clean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(restored, clean, rtol=0, atol=0)

    def test_gate_noise_monotonically_dampens(self, x: torch.Tensor) -> None:
        layer = _layer()
        magnitudes = []
        with torch.no_grad():
            for p in (0.0, 0.02, 0.05, 0.1, 0.2):
                with apply_depolarizing_noise(layer, p):
                    magnitudes.append(layer(x).abs().mean().item())
        assert all(a > b for a, b in itertools.pairwise(magnitudes)), magnitudes

    def test_full_depolarization_zeroes_every_expectation(self, x: torch.Tensor) -> None:
        layer = _layer()
        with torch.no_grad(), apply_depolarizing_noise(layer, 0.75, position="end"):
            torch.testing.assert_close(layer(x), torch.zeros(8, N_QUBITS), rtol=0, atol=1e-6)

    def test_weights_are_untouched_and_qnode_restored(self, x: torch.Tensor) -> None:
        layer = _layer(diff_method="parameter-shift")
        qnode = layer.qlayer.qnode
        before = _weights(layer).detach().clone()
        with torch.no_grad():
            clean = layer(x)
            with apply_depolarizing_noise(layer, 0.1):
                assert layer.qlayer.qnode is not qnode
            assert layer.qlayer.qnode is qnode
            torch.testing.assert_close(layer(x), clean, rtol=0, atol=0)
        torch.testing.assert_close(_weights(layer).detach(), before, rtol=0, atol=0)

    def test_restored_after_exception(self) -> None:
        layer = _layer()
        qnode = layer.qlayer.qnode
        with pytest.raises(KeyError):
            with apply_depolarizing_noise(layer, 0.1):
                raise KeyError("x")
        assert layer.qlayer.qnode is qnode
        with apply_depolarizing_noise(layer, 0.1):  # usable again
            pass

    def test_gradients_flow_through_the_noisy_circuit(self, x: torch.Tensor) -> None:
        layer = _layer()
        with apply_depolarizing_noise(layer, 0.1):
            layer(x).sum().backward()
        grad = _weights(layer).grad
        assert grad is not None
        assert grad.abs().sum() > 0

    @pytest.mark.parametrize("cls", [HybridBinaryClassifier, ParallelHybridClassifier])
    def test_models(self, cls: type[Model]) -> None:
        model = _model(cls)
        X = torch.randn(4, 5)
        with apply_depolarizing_noise(model, 0.1) as m:
            assert m is model
            probs = model.predict_proba(X)
        assert probs.shape == (4,)


class TestValidation:
    @pytest.mark.parametrize("p", [-0.1, 0.76])
    def test_p_range(self, p: float) -> None:
        with pytest.raises(ValueError, match=r"p must lie in \[0, 0.75\]"):
            with apply_depolarizing_noise(_layer(), p):
                pass

    def test_position(self) -> None:
        with pytest.raises(ValueError, match="position must be"):
            with apply_depolarizing_noise(_layer(), 0.1, position="start"):  # type: ignore[arg-type]
                pass

    def test_unsupported_model(self) -> None:
        with pytest.raises(TypeError, match="got Linear"):
            with apply_depolarizing_noise(torch.nn.Linear(2, 1), 0.1):
                pass

    def test_nesting(self) -> None:
        layer = _layer()
        qnode = layer.qlayer.qnode
        with apply_depolarizing_noise(layer, 0.1):
            with pytest.raises(RuntimeError, match="cannot be nested"):
                with apply_depolarizing_noise(layer, 0.2):
                    pass
        assert layer.qlayer.qnode is qnode

    def test_zero_noise_never_trips_the_nesting_guard(self, x: torch.Tensor) -> None:
        # p = 0 replaces nothing, so it is a no-op wherever it appears rather
        # than a nested block: a sweep starting at 0 must not raise.
        layer = _layer()
        with apply_depolarizing_noise(layer, 0.1):
            noisy = layer.qlayer.qnode
            with apply_depolarizing_noise(layer, 0.0):
                assert layer.qlayer.qnode is noisy

    def test_a_failed_replacement_leaves_the_layer_usable(
        self, monkeypatch: pytest.MonkeyPatch, x: torch.Tensor
    ) -> None:
        # Building the noisy QNode can fail (default.mixed refuses more than 23
        # wires). The layer must come out of that as it went in, not stuck
        # reporting a nested block for the rest of the process.
        layer = _layer()
        qnode = layer.qlayer.qnode

        def boom(*args: object, **kwargs: object) -> None:
            raise ValueError("device refused the wires")

        monkeypatch.setattr("hqnn_forge.noise._noisy_qnode", boom)
        with pytest.raises(ValueError, match="device refused the wires"):
            with apply_depolarizing_noise(layer, 0.1):
                pass
        monkeypatch.undo()

        assert layer.qlayer.qnode is qnode
        with torch.no_grad(), apply_depolarizing_noise(layer, 0.1):
            assert layer.qlayer.qnode is not qnode
            layer(x)
        assert layer.qlayer.qnode is qnode


class TestSweep:
    def test_sweep_points_and_scores(self) -> None:
        model = _model()
        X, y = torch.randn(6, 5), torch.tensor([0, 1, 0, 1, 1, 0])
        spread = lambda _y, probs: float((probs - 0.5).abs().mean())
        points = noise_sweep(model, X, [0.0, 0.1, 0.4], y=y, score_fn=spread)
        assert [pt.p for pt in points] == [0.0, 0.1, 0.4]
        assert all(
            isinstance(pt, NoiseSweepPoint) and pt.probabilities.shape == (6,) for pt in points
        )
        torch.testing.assert_close(points[0].probabilities, model.predict_proba(X), rtol=0, atol=0)
        # More noise pushes the quantum features towards 0, so the output depends
        # less on the input: the probabilities spread less around 0.5
        first, last = points[0].score, points[2].score
        assert first is not None and last is not None
        assert first > last

    def test_sweep_without_scoring_accepts_an_iterator(self) -> None:
        points = noise_sweep(_model(), torch.randn(2, 5), (p for p in [0.2]))
        assert [pt.p for pt in points] == [0.2]
        assert points[0].score is None

    def test_sweep_rejects_a_bad_level_before_evaluating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Each level costs an O(4^n) mixed-state run, so a bad one at the end
        # has to be caught before the first of them, not after: building any
        # noisy circuit at all fails this test.
        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("a level was evaluated before the sweep was validated")

        monkeypatch.setattr("hqnn_forge.noise._noisy_qnode", boom)
        with pytest.raises(ValueError, match=r"every p must lie in \[0, 0.75\]"):
            noise_sweep(_model(), torch.randn(2, 5), [0.1, 1.5])

    def test_sweep_argument_errors(self) -> None:
        with pytest.raises(ValueError, match="both y and score_fn"):
            noise_sweep(_model(), torch.randn(2, 5), [0.1], y=torch.tensor([0, 1]))
        with pytest.raises(TypeError, match="predict_proba"):
            noise_sweep(_layer(), torch.randn(2, N_QUBITS), [0.1])
