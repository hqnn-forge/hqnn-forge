"""
tests/test_shots.py
===================
Finite shots and any PennyLane device (#314).

Sampled outputs are random, so they are checked statistically against the
exact values of an identical shot-free layer: the mean within ``Z`` standard
errors, and the variance against the binomial ``(1 − ⟨Z⟩²) / shots``.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any

import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding._common import resolve_device
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.noise import apply_depolarizing_noise, apply_shots, shot_sweep
from hqnn_forge.utils import load_checkpoint, save_checkpoint

Z = 5.0
SHIFT: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "parameter-shift"}
EXACT: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}

LAYERS = [
    pytest.param(QuantumEncodingLayer, 3, id="angle"),
    pytest.param(IQPEncodingLayer, 3, id="iqp"),
    pytest.param(partial(AmplitudeEncodingLayer, n_features=5), 5, id="amplitude"),
    pytest.param(partial(DataReuploadingLayer, trainable_input_scaling=True), 3, id="reuploading"),
]


def _on_shot_lattice(values: torch.Tensor, shots: int) -> bool:
    """Whether every ⟨Z⟩ in ``values`` is ``(2k − shots) / shots``: a sampled estimate."""
    k = (values.detach().double() * shots + shots) / 2
    return bool(torch.allclose(k, k.round(), rtol=0, atol=1e-6))


def _pair(factory: Any, shots: int) -> tuple[Any, Any]:
    """A sampled layer and an exact one with the same weights."""
    torch.manual_seed(0)
    sampled = factory(n_qubits=3, n_layers=2, shots=shots, **SHIFT)
    exact = factory(n_qubits=3, n_layers=2, **EXACT)
    exact.load_state_dict(sampled.state_dict())
    return sampled, exact


@pytest.mark.parametrize("factory, width", LAYERS)
class TestSampledLayers:
    def test_mean_and_variance_are_the_binomial_ones(self, factory: Any, width: int) -> None:
        shots, repeats = 200, 300
        sampled, exact = _pair(factory, shots)
        x = torch.rand(1, width, generator=torch.Generator().manual_seed(1)) + 0.1
        with torch.no_grad():
            z = exact(x)[0].double()
            runs = torch.stack([sampled(x)[0] for _ in range(repeats)]).double()
        expected_var = (1 - z**2) / shots
        se_mean = expected_var.sqrt() / math.sqrt(repeats)
        assert ((runs.mean(0) - z).abs() <= Z * se_mean + 1e-9).all()
        # The sample variance of ``repeats`` near-Gaussian draws has relative
        # standard error sqrt(2 / (repeats - 1)).
        ratio = runs.var(0) / expected_var
        assert ((ratio - 1).abs() <= Z * math.sqrt(2 / (repeats - 1))).all(), ratio

    def test_repr_shows_the_shots(self, factory: Any, width: int) -> None:
        sampled, exact = _pair(factory, 123)
        assert "shots=123" in repr(sampled) and "shots" not in repr(exact)
        assert sampled.shots == 123 and exact.shots is None


def test_sampled_gradients_are_unbiased() -> None:
    shots, repeats = 500, 60
    sampled, exact = _pair(QuantumEncodingLayer, shots)
    x = torch.rand(2, 3, generator=torch.Generator().manual_seed(2))
    (g_exact,) = torch.autograd.grad(exact(x).sum(), exact.qlayer.weights)
    grads = torch.stack(
        [torch.autograd.grad(sampled(x).sum(), sampled.qlayer.weights)[0] for _ in range(repeats)]
    )
    se = grads.std(0) / math.sqrt(repeats)
    assert ((grads.mean(0) - g_exact).abs() <= Z * se + 1e-6).all()


class TestValidation:
    @pytest.mark.parametrize("diff_method", ["adjoint", "backprop", "finite-diff"])
    @pytest.mark.parametrize("factory, width", LAYERS)
    def test_exact_value_methods_are_refused(
        self, diff_method: str, factory: Any, width: int
    ) -> None:
        with pytest.raises(ValueError, match="use diff_method='parameter-shift'"):
            factory(
                n_qubits=3,
                n_layers=1,
                shots=100,
                device_name="default.qubit",
                diff_method=diff_method,
            )

    @pytest.mark.parametrize("shots", [0, -5, True, 2.5])
    def test_shots_must_be_a_positive_int(self, shots: Any) -> None:
        with pytest.raises(ValueError, match="shots must be None or a positive int"):
            QuantumEncodingLayer(n_qubits=3, n_layers=1, shots=shots, **SHIFT)

    @pytest.mark.parametrize(
        "cls", [HybridBinaryClassifier, ParallelHybridClassifier, MulticlassHybridClassifier]
    )
    def test_classifiers_validate_too(self, cls: type) -> None:
        with pytest.raises(ValueError, match="needs exact ones"):
            cls(n_input_features=4, n_qubits=3, shots=100, **EXACT)

    def test_density_training_noise_ignores_shots_and_is_refused(self) -> None:
        with pytest.raises(ValueError, match="use noise_method='trajectories'"):
            QuantumEncodingLayer(n_qubits=3, n_layers=1, shots=100, noise_level=0.1, **SHIFT)

    def test_trajectory_training_noise_samples_on_the_shot_qnode(self) -> None:
        # Five shots put every sampled ⟨Z⟩ on a coarse lattice no exact value
        # of random weights lands on.
        layer = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=1,
            shots=5,
            noise_level=0.1,
            noise_method="trajectories",
            **SHIFT,
        )
        layer.train()
        out = layer(torch.rand(4, 3))
        assert _on_shot_lattice(out, 5)
        out.sum().backward()
        assert layer.qlayer.weights.grad is not None


@pytest.mark.parametrize(
    "cls, extra",
    [
        (HybridBinaryClassifier, {}),
        (ParallelHybridClassifier, {}),
        (MulticlassHybridClassifier, {"n_classes": 3}),
    ],
)
class TestClassifiers:
    def test_the_quantum_layer_is_sampled(self, cls: type, extra: dict[str, Any]) -> None:
        torch.manual_seed(0)
        model = cls(n_input_features=4, n_qubits=3, n_layers=1, shots=50, **SHIFT, **extra)
        assert model.quantum_layer.shots == 50 and model.get_config()["shots"] == 50
        x = torch.randn(6, 4)
        with torch.no_grad():
            assert not torch.equal(model.eval()(x), model(x))  # two samples differ

    def test_checkpoint_round_trip_and_a_weight_safe_shots_override(
        self, cls: type, extra: dict[str, Any], tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        model = cls(n_input_features=4, n_qubits=3, n_layers=1, **EXACT, **extra)
        save_checkpoint(model, tmp_path / "m.pt")
        # Running a model trained on exact values with shots is an override,
        # like a device change, not an architecture change.
        sampled: Any = load_checkpoint(
            tmp_path / "m.pt", shots=4000, diff_method="parameter-shift"
        )
        assert sampled.quantum_layer.shots == 4000
        for key, value in model.state_dict().items():
            torch.testing.assert_close(sampled.state_dict()[key], value, rtol=0, atol=0)
        x = torch.randn(5, 4)
        with torch.no_grad():
            torch.testing.assert_close(sampled(x), model.eval()(x), atol=0.15, rtol=0)


class TestApplyShots:
    def _model(self) -> Any:
        torch.manual_seed(0)
        return HybridBinaryClassifier(n_input_features=4, n_qubits=3, n_layers=2, **EXACT)

    def test_samples_inside_and_restores_the_exact_model(self) -> None:
        model = self._model()
        x = torch.randn(8, 4)
        before = model.predict_proba(x)
        original = model.quantum_layer.qlayer.qnode
        with apply_shots(model, 30):
            a, b = model.predict_proba(x), model.predict_proba(x)
            assert not torch.equal(a, b)
        assert model.quantum_layer.qlayer.qnode is original
        torch.testing.assert_close(model.predict_proba(x), before, rtol=0, atol=0)

    def test_none_is_the_exact_reference(self) -> None:
        model = self._model()
        x = torch.randn(8, 4)
        with apply_shots(model, None):
            inside = model.predict_proba(x)
        torch.testing.assert_close(inside, model.predict_proba(x), atol=1e-6, rtol=0)

    def test_restored_when_the_block_raises(self) -> None:
        model = self._model()
        original = model.quantum_layer.qlayer.qnode
        with pytest.raises(KeyError), apply_shots(model, 10):
            raise KeyError("inside")
        assert model.quantum_layer.qlayer.qnode is original

    def test_nesting_and_combining_with_noise_are_refused(self) -> None:
        model = self._model()
        with apply_shots(model, 10), pytest.raises(RuntimeError, match="cannot be nested"):
            with apply_shots(model, 20):
                pass
        with (
            apply_depolarizing_noise(model, 0.1),
            pytest.raises(RuntimeError, match="exact channel"),
        ):
            with apply_shots(model, 20):
                pass

    def test_noise_block_refuses_a_sampled_layer(self) -> None:
        # The noise block's default.mixed QNode is exact, so it would silently
        # drop the shots of a layer built with them or inside apply_shots.
        model = self._model()
        original = model.quantum_layer.qlayer.qnode
        with (
            apply_shots(model, 20),
            pytest.raises(RuntimeError, match="ignore the layer's shots=20"),
        ):
            with apply_depolarizing_noise(model, 0.1):
                pass
        assert model.quantum_layer.qlayer.qnode is original
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, shots=50, **SHIFT)
        with pytest.raises(RuntimeError, match="ignore the layer's shots=50"):
            with apply_depolarizing_noise(layer, 0.1):
                pass
        assert getattr(layer.qlayer, "_hqnn_noise_depth", 0) == 0
        # p = 0 replaces nothing, so it stays allowed.
        with apply_depolarizing_noise(layer, 0.0):
            pass

    def test_train_mode_trajectory_noise_samples_with_the_block_shots(self) -> None:
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=3, n_layers=1, noise_level=0.05, noise_method="trajectories", **EXACT
        )
        noise_qnode = layer._training_noise_qnode
        x = torch.rand(4, 3)
        layer.train()
        assert not _on_shot_lattice(layer(x), 5)
        with apply_shots(layer, 5):
            out = layer(x)
            assert _on_shot_lattice(out, 5)
            out.sum().backward()
            assert layer.qlayer.weights.grad is not None
        assert layer._training_noise_qnode is noise_qnode

    def test_train_mode_density_noise_is_refused_inside(self) -> None:
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, noise_level=0.05, **EXACT)
        x = torch.rand(4, 3)
        with apply_shots(layer, 5):
            assert _on_shot_lattice(layer.eval()(x), 5)
            with pytest.raises(RuntimeError, match="would ignore apply_shots"):
                layer.train()(x)

    def test_amplitude_input_gradients_stay_refused(self) -> None:
        # Built under backprop, whose input gradient is exact; inside the
        # block parameter-shift would differentiate the state preparation.
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=4, n_qubits=2, n_layers=1, encoding_type="amplitude", **EXACT
        )
        x = torch.randn(4, 4)
        with apply_shots(model, 50):
            model.predict_proba(x)  # no input gradient: allowed
            with pytest.raises(RuntimeError, match="cannot differentiate with respect"):
                model(x)
        model(x).sum().backward()  # backprop again outside

    def test_shots_attribute_follows_the_block(self) -> None:
        model = self._model()
        assert model.quantum_layer.shots is None
        with apply_shots(model, 40):
            assert model.quantum_layer.shots == 40
            assert "shots=40" in repr(model.quantum_layer)
        assert model.quantum_layer.shots is None

    def test_errors_name_apply_shots(self) -> None:
        with (
            pytest.raises(TypeError, match="apply_shots expects"),
            apply_shots(torch.nn.Linear(2, 1), 5),
        ):
            pass
        with pytest.raises(ValueError, match="positive int"), apply_shots(self._model(), 0):
            pass


class TestShotSweep:
    def test_spread_falls_like_one_over_root_shots(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(n_input_features=4, n_qubits=3, n_layers=2, **EXACT)
        x = torch.randn(20, 4)
        exact, few, many = shot_sweep(model, x, [None, 40, 4000], n_repeats=6)
        assert exact.probabilities.shape == (6, 20)
        assert (exact.probabilities == exact.probabilities[0]).all()
        spread_few = few.probabilities.std(0).mean()
        spread_many = many.probabilities.std(0).mean()
        # sqrt(4000 / 40) = 10; the estimate of a mean std over 6 repeats is
        # rough, so ask for half of it.
        assert spread_few > 5 * spread_many > 0

    def test_scores_per_repeat_and_validation_up_front(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(n_input_features=4, n_qubits=3, n_layers=1, **EXACT)
        x, y = torch.randn(10, 4), torch.randint(0, 2, (10,))
        (point,) = shot_sweep(
            model,
            x,
            [100],
            n_repeats=3,
            y=y,
            score_fn=lambda y, p: float((p > 0.5).eq(y).float().mean()),
        )
        assert point.scores is not None and len(point.scores) == 3
        with pytest.raises(ValueError, match="positive int"):
            shot_sweep(model, x, [100, 0])
        with pytest.raises(ValueError, match="both y and score_fn"):
            shot_sweep(model, x, [100], y=y)


def test_any_pennylane_device_runs_a_layer() -> None:
    # default.mixed is outside the fallback chain: constructed as given.
    torch.manual_seed(0)
    mixed = QuantumEncodingLayer(
        n_qubits=2, n_layers=1, device_name="default.mixed", diff_method="parameter-shift"
    )
    exact = QuantumEncodingLayer(n_qubits=2, n_layers=1, **EXACT)
    exact.load_state_dict(mixed.state_dict())
    assert mixed.qlayer.qnode.device.name == "default.mixed"
    x = torch.rand(3, 2)
    with torch.no_grad():
        torch.testing.assert_close(mixed(x), exact(x), atol=1e-6, rtol=0)


@pytest.mark.filterwarnings("ignore:Setting shots on device is deprecated")
@pytest.mark.parametrize(
    "layer_cls, module_name, kwargs, in_features",
    [
        (
            QuantumEncodingLayer,
            "hqnn_forge.encoding.angle_embedding",
            {"n_qubits": 3, "n_layers": 1},
            3,
        ),
        (
            IQPEncodingLayer,
            "hqnn_forge.encoding.iqp_embedding",
            {"n_qubits": 3, "n_layers": 1},
            3,
        ),
        (
            AmplitudeEncodingLayer,
            "hqnn_forge.encoding.amplitude_embedding",
            {"n_features": 4, "n_qubits": 2, "n_layers": 1},
            4,
        ),
        (
            DataReuploadingLayer,
            "hqnn_forge.encoding.data_reuploading",
            {"n_qubits": 3, "n_layers": 1},
            3,
        ),
    ],
)
def test_finite_shot_device_rejection_and_sampled_execution(
    monkeypatch: pytest.MonkeyPatch,
    layer_cls: Any,
    module_name: str,
    kwargs: dict[str, Any],
    in_features: int,
) -> None:
    def fake_resolve(
        device_name: str, n_qubits: int, *, seed: int | None = None
    ) -> qml.devices.Device:
        if device_name == "custom.sampling.device":
            return qml.device("default.qubit", wires=n_qubits, shots=10_000)
        return resolve_device(device_name, n_qubits, seed=seed)

    monkeypatch.setattr(f"{module_name}.resolve_device", fake_resolve)

    with pytest.raises(ValueError, match="device samples"):
        layer_cls(**kwargs, device_name="custom.sampling.device")

    torch.manual_seed(0)
    exact = layer_cls(**kwargs)
    sampled = layer_cls(
        **kwargs,
        device_name="custom.sampling.device",
        shots=10_000,
        diff_method="parameter-shift",
    )
    sampled.load_state_dict(exact.state_dict())
    x = torch.rand(2, in_features)
    with torch.no_grad():
        # each <Z> estimate has standard deviation <= 1/sqrt(10 000) = 0.01; allow 5 sigma
        torch.testing.assert_close(sampled(x), exact(x), atol=0.05, rtol=0)
