"""
tests/test_noise_aware_training.py
==================================
Training-time depolarizing noise (``noise_level`` on the encoding layers and
classifiers): the noiseless default is untouched, train mode runs the noisy
circuit with a known effect, eval mode is noiseless, gradients flow through
the noise, and the post-hoc wrapper takes precedence.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, TypedDict

import pytest
import torch

from hqnn_forge.diagnostics import effective_dimension, gradient_variance
from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.noise import (
    TrainingNoiseMixin,
    apply_depolarizing_noise,
    noise_sweep,
    run_with_training_noise,
    training_noise_qnode,
)

N_QUBITS = 3


class _Backend(TypedDict):
    device_name: DeviceName
    diff_method: DiffMethod


CPU: _Backend = {"device_name": "default.qubit", "diff_method": "backprop"}
Layer = QuantumEncodingLayer | IQPEncodingLayer | AmplitudeEncodingLayer | DataReuploadingLayer
LayerFactory = Callable[..., Layer]
Model = HybridBinaryClassifier | ParallelHybridClassifier | MulticlassHybridClassifier
# Every encoding layer (#227).  The amplitude layer is built for N_QUBITS
# features, padded to 2**N_QUBITS amplitudes, so it takes the same inputs.
LAYERS = [
    pytest.param(QuantumEncodingLayer, id="angle"),
    pytest.param(IQPEncodingLayer, id="iqp"),
    pytest.param(partial(AmplitudeEncodingLayer, n_features=N_QUBITS), id="amplitude"),
    pytest.param(DataReuploadingLayer, id="reuploading"),
    pytest.param(
        partial(DataReuploadingLayer, trainable_input_scaling=True), id="reuploading-scaled"
    ),
]
MODELS: list[type[Model]] = [HybridBinaryClassifier, ParallelHybridClassifier]


def _layer(cls: LayerFactory = QuantumEncodingLayer, **kwargs: Any) -> Layer:
    torch.manual_seed(0)
    return cls(n_qubits=N_QUBITS, n_layers=2, **CPU, **kwargs)


def _weights(layer: Layer) -> torch.Tensor:
    """The layer's variational weights, narrowed from ``nn.Module.__getattr__``."""
    weights = layer.qlayer.weights
    assert isinstance(weights, torch.Tensor)
    return weights


def _pair(cls: LayerFactory = QuantumEncodingLayer, **noise: Any) -> tuple[Layer, Layer]:
    """A noisy layer and a noiseless one with identical weights."""
    noisy = _layer(cls, **noise)
    clean = _layer(cls)
    with torch.no_grad():
        _weights(clean).copy_(_weights(noisy))
    return noisy, clean


@pytest.fixture
def x() -> torch.Tensor:
    return (torch.rand(6, N_QUBITS, generator=torch.Generator().manual_seed(1)) * 2 - 1) * math.pi


# ---------------------------------------------------------------------------
# noise_level = 0 is the existing path, exactly
# ---------------------------------------------------------------------------


class TestNoiselessDefault:
    @pytest.mark.parametrize("cls", LAYERS)
    def test_zero_noise_is_bit_identical_in_both_modes(
        self, cls: LayerFactory, x: torch.Tensor
    ) -> None:
        explicit, default = _pair(cls, noise_level=0.0)
        assert explicit._training_noise_qnode is None
        for mode in (True, False):
            explicit.train(mode)
            default.train(mode)
            out_a, out_b = explicit(x), default(x)
            torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
        explicit.train()
        default.train()
        explicit(x).sum().backward()
        default(x).sum().backward()
        torch.testing.assert_close(
            explicit.qlayer.weights.grad, default.qlayer.weights.grad, rtol=0, atol=0
        )

    @pytest.mark.parametrize("cls", [*MODELS, MulticlassHybridClassifier])
    def test_classifier_default_is_noiseless(self, cls: type[Model]) -> None:
        torch.manual_seed(0)
        model = cls(n_input_features=4, n_qubits=N_QUBITS, n_layers=1, **CPU)
        assert model.quantum_layer.noise_level == 0.0
        assert model.quantum_layer._training_noise_qnode is None


# ---------------------------------------------------------------------------
# Effect of the noise in train mode
# ---------------------------------------------------------------------------


class TestTrainingNoise:
    @pytest.mark.parametrize("cls", LAYERS)
    @pytest.mark.parametrize("p", [0.05, 0.3])
    def test_end_noise_damps_train_output_by_one_minus_four_thirds_p(
        self, cls: LayerFactory, p: float, x: torch.Tensor
    ) -> None:
        noisy, clean = _pair(cls, noise_level=p, noise_position="end")
        noisy.train()
        with torch.no_grad():
            torch.testing.assert_close(noisy(x), (1 - 4 * p / 3) * clean(x), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("cls", LAYERS)
    def test_eval_mode_is_noiseless(self, cls: LayerFactory, x: torch.Tensor) -> None:
        noisy, clean = _pair(cls, noise_level=0.3)
        noisy.eval()
        with torch.no_grad():
            torch.testing.assert_close(noisy(x), clean(x), rtol=0, atol=0)
        assert noisy.qlayer.qnode is not noisy._training_noise_qnode

    def test_gate_noise_changes_the_train_output(self, x: torch.Tensor) -> None:
        noisy, clean = _pair(noise_level=0.1)
        noisy.train()
        with torch.no_grad():
            assert not torch.allclose(noisy(x), clean(x), atol=1e-3)

    @pytest.mark.parametrize("cls", LAYERS)
    @pytest.mark.parametrize("p", [0.05, 0.3])
    def test_gate_noise_train_output_matches_the_post_hoc_channel(
        self, cls: LayerFactory, p: float, x: torch.Tensor
    ) -> None:
        """
        The default position="all": train mode must equal the noiseless twin
        run under apply_depolarizing_noise with the same p and position, which
        pins where channels go and with what p, not only that the output moved.
        """
        noisy, clean = _pair(cls, noise_level=p)
        noisy.train()
        with torch.no_grad():
            out = noisy(x)
            with apply_depolarizing_noise(clean, p, position="all"):
                expected = clean(x)
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-7)
        assert not torch.allclose(out, clean(x).detach(), atol=1e-3)

    def test_gradients_flow_through_the_noisy_circuit(self, x: torch.Tensor) -> None:
        noisy, clean = _pair(noise_level=0.2, noise_position="end")
        noisy.train()
        clean.train()
        noisy(x).sum().backward()
        clean(x).sum().backward()
        noisy_grad, clean_grad = _weights(noisy).grad, _weights(clean).grad
        assert noisy_grad is not None and clean_grad is not None
        # End-position noise scales every ⟨Z⟩ by a constant, hence the gradient too.
        torch.testing.assert_close(
            noisy_grad,
            (1 - 4 * 0.2 / 3) * clean_grad,
            rtol=1e-4,
            atol=1e-6,
        )

    def test_qnode_is_restored_after_the_forward_pass(self, x: torch.Tensor) -> None:
        noisy = _layer(noise_level=0.1)
        original = noisy.qlayer.qnode
        noisy.train()
        noisy(x)
        assert noisy.qlayer.qnode is original

    def test_qnode_is_restored_when_the_forward_pass_raises(self) -> None:
        noisy = _layer(noise_level=0.1)
        original = noisy.qlayer.qnode
        noisy.train()
        with pytest.raises(Exception):  # noqa: B017 - any error from the bad input
            run_with_training_noise(
                noisy.qlayer, noisy._training_noise_qnode, torch.zeros(2, N_QUBITS + 1)
            )
        assert noisy.qlayer.qnode is original

    @pytest.mark.parametrize("cls", LAYERS)
    def test_extra_repr_mentions_the_noise(self, cls: LayerFactory) -> None:
        assert "noise_level=0.1, noise_position='all'" in _layer(cls, noise_level=0.1).extra_repr()
        end = _layer(cls, noise_level=0.1, noise_position="end").extra_repr()
        assert "noise_position='end'" in end
        assert "noise_level" not in _layer(cls).extra_repr()

    @pytest.mark.requires_lightning
    @pytest.mark.parametrize("cls", LAYERS)
    def test_lightning_adjoint_layer_damps_output_and_gradient(
        self, cls: LayerFactory, x: torch.Tensor
    ) -> None:
        """
        The noiseless QNode on lightning.qubit + adjoint (the default until
        #349, and what "auto" picks above 12 qubits) is wrapped for batching,
        and the train-mode one is rebuilt from its circuit function.  End noise must still scale output and gradient by
        exactly 1 - 4p/3 relative to the noiseless layer.
        """
        p = 0.2
        torch.manual_seed(0)
        lightning = {"device_name": "lightning.qubit", "diff_method": "adjoint"}
        noisy = cls(
            n_qubits=N_QUBITS, n_layers=2, noise_level=p, noise_position="end", **lightning
        )
        clean = cls(n_qubits=N_QUBITS, n_layers=2, **lightning)
        assert clean.qlayer.qnode.device.name == "lightning.qubit"
        with torch.no_grad():
            clean.qlayer.weights.copy_(noisy.qlayer.weights)
        noisy.train()
        clean.train()
        out_noisy, out_clean = noisy(x), clean(x)
        torch.testing.assert_close(out_noisy, (1 - 4 * p / 3) * out_clean, rtol=1e-5, atol=1e-6)
        out_noisy.sum().backward()
        out_clean.sum().backward()
        torch.testing.assert_close(
            noisy.qlayer.weights.grad,
            (1 - 4 * p / 3) * clean.qlayer.weights.grad,
            rtol=1e-4,
            atol=1e-6,
        )
        noisy.eval()
        with torch.no_grad():
            torch.testing.assert_close(noisy(x), clean(x), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Interaction with the post-hoc wrapper and the classifiers
# ---------------------------------------------------------------------------


class TestInteractions:
    def test_post_hoc_wrapper_wins_in_train_mode(self, x: torch.Tensor) -> None:
        """Inside apply_depolarizing_noise the sweep's channel is used, not the training one."""
        noisy, clean = _pair(noise_level=0.3, noise_position="end")
        noisy.train()
        with torch.no_grad(), apply_depolarizing_noise(noisy, 0.6, position="end"):
            torch.testing.assert_close(
                noisy(x), (1 - 4 * 0.6 / 3) * clean(x), rtol=1e-5, atol=1e-6
            )

    def test_zero_noise_wrapper_is_noiseless_in_train_mode(self, x: torch.Tensor) -> None:
        """p = 0 counts as the wrapper too: it suppresses the training channel."""
        noisy, clean = _pair(noise_level=0.3, noise_position="end")
        noisy.train()
        with torch.no_grad():
            with apply_depolarizing_noise(noisy, 0.0):
                torch.testing.assert_close(noisy(x), clean(x), rtol=0, atol=0)
            assert noisy.qlayer._hqnn_noise_depth == 0
            # Outside the block the training channel is back.
            torch.testing.assert_close(noisy(x), 0.6 * clean(x), rtol=1e-5, atol=1e-6)

    def test_train_mode_sweep_over_a_noisy_layer_is_monotone(self, x: torch.Tensor) -> None:
        noisy, clean = _pair(noise_level=0.3, noise_position="end")
        noisy.train()
        reference = clean(x).detach()
        for p in (0.0, 0.1, 0.4):
            with torch.no_grad(), apply_depolarizing_noise(noisy, p, position="end"):
                torch.testing.assert_close(
                    noisy(x), (1 - 4 * p / 3) * reference, rtol=1e-5, atol=1e-6
                )

    def test_zero_noise_wrapper_inside_an_open_block_keeps_that_block(
        self, x: torch.Tensor
    ) -> None:
        noisy, clean = _pair()
        with torch.no_grad(), apply_depolarizing_noise(noisy, 0.3, position="end"):
            with apply_depolarizing_noise(noisy, 0.0):
                torch.testing.assert_close(noisy(x), 0.6 * clean(x), rtol=1e-5, atol=1e-6)
            # The inner p = 0 block must not have disarmed the outer one.
            assert noisy.qlayer._hqnn_noise_original is not None
        assert noisy.qlayer._hqnn_noise_original is None

    def test_noisy_block_inside_a_zero_noise_block(self, x: torch.Tensor) -> None:
        """A p = 0 baseline block wrapped around a sweep must not trip the nesting guard."""
        noisy, clean = _pair(noise_level=0.3, noise_position="end")
        noisy.train()
        with torch.no_grad(), apply_depolarizing_noise(noisy, 0.0):
            with apply_depolarizing_noise(noisy, 0.6, position="end"):
                torch.testing.assert_close(
                    noisy(x), (1 - 4 * 0.6 / 3) * clean(x), rtol=1e-5, atol=1e-6
                )
            # Back in the outer p = 0 block: noiseless, training channel still off.
            torch.testing.assert_close(noisy(x), clean(x), rtol=0, atol=0)
        torch.testing.assert_close(noisy(x), 0.6 * clean(x), rtol=1e-5, atol=1e-6)
        assert noisy.qlayer._hqnn_noise_depth == 0

    # gradient_variance measures one trainable tensor here, so not the layer
    # with input_scaling.
    @pytest.mark.parametrize("cls", [p for p in LAYERS if p.id != "reuploading-scaled"])
    def test_gradient_variance_measures_the_noiseless_circuit(self, cls: LayerFactory) -> None:
        noisy, clean = _pair(cls, noise_level=0.3)
        noisy.train()
        a = gradient_variance(noisy, n_samples=4, generator=torch.Generator().manual_seed(0))
        b = gradient_variance(clean, n_samples=4, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(a.per_parameter, b.per_parameter, rtol=0, atol=0)
        assert noisy.training

    def test_gradient_variance_under_the_wrapper_sees_the_noise(self) -> None:
        """End noise scales every gradient by 1 - 4p/3, so the variance by its square."""
        noisy, clean = _pair(noise_level=0.3)
        noisy.train()
        with apply_depolarizing_noise(noisy, 0.3, position="end"):
            a = gradient_variance(noisy, n_samples=4, generator=torch.Generator().manual_seed(0))
        b = gradient_variance(clean, n_samples=4, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(
            a.per_parameter, (1 - 4 * 0.3 / 3) ** 2 * b.per_parameter, rtol=1e-4, atol=1e-10
        )

    @pytest.mark.slow
    def test_effective_dimension_measures_the_noiseless_circuit(self) -> None:
        """Agrees with gradient_variance: a train-mode noisy model is measured noiselessly."""
        torch.manual_seed(0)
        noisy = HybridBinaryClassifier(4, N_QUBITS, 1, noise_level=0.3, **CPU)
        torch.manual_seed(0)
        clean = HybridBinaryClassifier(4, N_QUBITS, 1, **CPU)
        clean.load_state_dict(noisy.state_dict())
        X = torch.randn(80, 4, generator=torch.Generator().manual_seed(2))
        assert noisy.training
        a = effective_dimension(
            noisy, X, n_theta_samples=2, generator=torch.Generator().manual_seed(0)
        )
        b = effective_dimension(
            clean, X, n_theta_samples=2, generator=torch.Generator().manual_seed(0)
        )
        assert a.effective_dimension == b.effective_dimension
        assert noisy.training
        with apply_depolarizing_noise(noisy, 0.3):
            c = effective_dimension(
                noisy, X, n_theta_samples=2, generator=torch.Generator().manual_seed(0)
            )
        assert c.effective_dimension != b.effective_dimension

    @pytest.mark.parametrize("cls", MODELS)
    def test_classifier_trains_and_sweeps(self, cls: type[Model]) -> None:
        torch.manual_seed(0)
        model = cls(n_input_features=4, n_qubits=N_QUBITS, n_layers=1, noise_level=0.1, **CPU)
        assert model.quantum_layer.noise_level == 0.1
        X = torch.randn(6, 4)
        y = torch.randint(0, 2, (6,)).float()
        model.train()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X).squeeze(-1), y)
        loss.backward()
        assert _weights(model.quantum_layer).grad is not None
        # predict_proba runs in eval mode: noiseless, so the sweep's p = 0 point
        # equals the plain prediction and larger p change it.
        points = noise_sweep(model, X, [0.0, 0.3])
        torch.testing.assert_close(points[0].probabilities, model.predict_proba(X), rtol=0, atol=0)
        assert not torch.allclose(points[1].probabilities, points[0].probabilities, atol=1e-4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("p", [-0.1, 0.8])
    def test_noise_level_range(self, p: float) -> None:
        with pytest.raises(ValueError, match="noise_level must lie"):
            _layer(noise_level=p)

    def test_noise_position(self) -> None:
        with pytest.raises(ValueError, match="noise_position must be"):
            _layer(noise_level=0.1, noise_position="middle")

    def test_classifier_validates_too(self) -> None:
        with pytest.raises(ValueError, match="noise_level must lie"):
            HybridBinaryClassifier(n_input_features=4, n_qubits=N_QUBITS, noise_level=1.0, **CPU)

    @pytest.mark.parametrize("cls", LAYERS)
    def test_warns_above_the_practical_qubit_count(self, cls: LayerFactory) -> None:
        with pytest.warns(RuntimeWarning, match="n_qubits=7") as record:
            cls(n_qubits=7, n_layers=1, noise_level=0.1, **CPU)
        # Attributed to the line that built the layer, not to library code.
        assert record[0].filename == __file__

    @pytest.mark.parametrize("kwargs", [{"n_qubits": 6, "noise_level": 0.1}, {"n_qubits": 7}])
    def test_no_warning_at_the_limit_or_without_noise(self, kwargs: dict[str, Any]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            QuantumEncodingLayer(n_layers=1, **kwargs, **CPU)

    def test_training_noise_qnode_rejects_zero(self) -> None:
        layer = _layer()
        with pytest.raises(ValueError, match="p > 0"):
            training_noise_qnode(layer.qlayer.qnode, N_QUBITS, 0.0, channel="depolarizing")


class TestOneImplementation:
    """#227: construction and dispatch live in hqnn_forge.noise, once."""

    @pytest.mark.parametrize(
        "cls",
        [QuantumEncodingLayer, IQPEncodingLayer, AmplitudeEncodingLayer, DataReuploadingLayer],
    )
    def test_layers_inherit_the_mixin_and_copy_nothing(self, cls: type[torch.nn.Module]) -> None:
        assert issubclass(cls, TrainingNoiseMixin)
        for name in ("_init_training_noise", "_run_circuit", "_noise_repr"):
            assert name not in vars(cls)
        # No hand-written train-mode branch left in any forward.
        assert "_training_noise_qnode" not in inspect.getsource(cls.forward)

    @pytest.mark.parametrize(
        "cls",
        [QuantumEncodingLayer, IQPEncodingLayer, AmplitudeEncodingLayer, DataReuploadingLayer],
    )
    def test_every_layer_takes_the_same_noise_arguments(self, cls: type) -> None:
        parameters = inspect.signature(cls).parameters
        for name, default in (
            ("noise_level", 0.0),
            ("noise_position", "all"),
            ("noise_method", "density"),
            ("noise_trajectories", 1),
        ):
            assert parameters[name].default == default
