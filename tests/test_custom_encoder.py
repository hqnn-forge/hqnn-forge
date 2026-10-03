"""
tests/test_custom_encoder.py
============================
A user-supplied ``classical_encoder`` on HybridBinaryClassifier and
ParallelHybridClassifier (#198): the model applies ``encoder_activation`` and
π on top of the module, checks its output width once, never re-initialises
it, and refuses to checkpoint it.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hqnn_forge.diagnostics import gradient_variance
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.utils import disable_quantum_layer, load_checkpoint, save_checkpoint

CPU = {"device_name": "default.qubit", "diff_method": "backprop"}
N_IN, N_QUBITS = 6, 3
Model = HybridBinaryClassifier | ParallelHybridClassifier
CLASSES = pytest.mark.parametrize(
    "cls", [HybridBinaryClassifier, ParallelHybridClassifier], ids=["serial", "parallel"]
)


def _mlp(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(N_IN, 5), nn.ReLU(), nn.Linear(5, N_QUBITS))


def _model(cls: type[Model], encoder: nn.Module | None = None, **kwargs: object) -> Model:
    torch.manual_seed(1)
    # Two layers: at n_qubits * n_layers <= 3 the restricted init warns (#167).
    return cls(N_IN, N_QUBITS, 2, classical_encoder=encoder, **CPU, **kwargs)  # type: ignore[arg-type]


def _custom(model: Model) -> nn.Module:
    """The user's module inside ``Sequential(module, activation)``."""
    encoder = model.classical_encoder
    assert isinstance(encoder, nn.Sequential)
    return encoder[0]


def _first_linear(module: nn.Module) -> nn.Linear:
    assert isinstance(module, nn.Sequential)
    linear = module[0]
    assert isinstance(linear, nn.Linear)
    return linear


def _x(n: int = 4) -> torch.Tensor:
    return torch.randn(n, N_IN, generator=torch.Generator().manual_seed(2))


@CLASSES
class TestForward:
    @pytest.mark.parametrize(
        ("activation", "squash"),
        [("tanh", torch.tanh), ("sigmoid", torch.sigmoid)],
    )
    def test_circuit_sees_the_activation_of_the_module_times_pi(
        self, cls: type[Model], activation: str, squash: object
    ) -> None:
        encoder = _mlp()
        model = _model(cls, encoder, encoder_activation=activation)
        seen: list[torch.Tensor] = []
        model.quantum_layer.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
        x = _x()
        logits = model(x)
        assert logits.shape == (4, 1)
        with torch.no_grad():
            expected = squash(encoder(x)) * math.pi  # type: ignore[operator]
        torch.testing.assert_close(seen[0], expected)

    def test_the_module_is_used_as_given(self, cls: type[Model]) -> None:
        encoder = _mlp()
        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        model = _model(cls, encoder)
        # The same object, so the caller's reference trains with the model,
        # and not re-initialised, so pretrained weights survive.
        assert _custom(model) is encoder
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_the_rest_is_still_initialised(self, cls: type[Model]) -> None:
        with torch.no_grad():
            head_bias_default = _model(cls).head.bias.clone()
        model = _model(cls, _mlp())
        torch.testing.assert_close(model.head.bias, head_bias_default)
        assert torch.count_nonzero(model.head.bias) == 0  # the model's own zero-bias init

    def test_parameters_are_counted(self, cls: type[Model]) -> None:
        builtin = _model(cls)
        encoder = _mlp()
        custom = _model(cls, encoder)
        linear = N_IN * N_QUBITS + N_QUBITS
        own = sum(p.numel() for p in encoder.parameters())
        assert custom.count_parameters() == builtin.count_parameters() - linear + own

    def test_gradients_reach_the_module_and_a_step_trains_it(self, cls: type[Model]) -> None:
        encoder = _mlp()
        model = _model(cls, encoder)
        first = _first_linear(encoder).weight.detach().clone()
        optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
        nn.functional.binary_cross_entropy_with_logits(
            model(_x()), torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        ).backward()
        for name, param in encoder.named_parameters():
            assert param.grad is not None and param.grad.abs().max() > 0, name
        optimiser.step()
        assert not torch.equal(_first_linear(encoder).weight, first)


@CLASSES
class TestValidation:
    def test_wrong_output_width(self, cls: type[Model]) -> None:
        with pytest.raises(
            ValueError, match=r"n_qubits=3\); on a batch of 2 it returned \(2, 4\)"
        ):
            _model(cls, nn.Linear(N_IN, 4))

    def test_module_that_cannot_take_the_input(self, cls: type[Model]) -> None:
        with pytest.raises(ValueError, match=r"failed on an input of shape \(2, 6\)") as info:
            _model(cls, nn.Linear(N_IN + 1, N_QUBITS))
        assert isinstance(info.value.__cause__, RuntimeError)

    def test_non_tensor_output(self, cls: type[Model]) -> None:
        class Pair(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x[:, :N_QUBITS], x[:, :N_QUBITS]

        with pytest.raises(ValueError, match="it returned tuple"):
            _model(cls, Pair())

    def test_not_a_module(self, cls: type[Model]) -> None:
        with pytest.raises(TypeError, match="must be an nn.Module; got function"):
            _model(cls, lambda x: x)  # type: ignore[arg-type]

    def test_needs_use_classical_encoder(self, cls: type[Model]) -> None:
        with pytest.raises(ValueError, match="needs use_classical_encoder=True"):
            _model(cls, _mlp(), use_classical_encoder=False)

    @pytest.mark.parametrize("last", [nn.Tanh(), nn.Sigmoid()], ids=["tanh", "sigmoid"])
    def test_a_final_activation_warns(self, cls: type[Model], last: nn.Module) -> None:
        with pytest.warns(UserWarning, match="squashed twice"):
            _model(cls, nn.Sequential(nn.Linear(N_IN, N_QUBITS), last))

    def test_no_warning_otherwise(self, cls: type[Model]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _model(cls, _mlp())

    def test_the_width_probe_leaves_the_module_untouched(self, cls: type[Model]) -> None:
        # BatchNorm would update its running statistics on a training-mode
        # forward, and the probe must not flip the caller's mode.
        encoder = nn.Sequential(nn.Linear(N_IN, N_QUBITS), nn.BatchNorm1d(N_QUBITS))
        bn = encoder[1]
        assert isinstance(bn, nn.BatchNorm1d)
        assert bn.running_mean is not None and bn.running_var is not None
        mean, var = bn.running_mean.clone(), bn.running_var.clone()
        _model(cls, encoder)
        assert encoder.training and bn.training
        torch.testing.assert_close(bn.running_mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(bn.running_var, var, rtol=0, atol=0)
        assert bn.num_batches_tracked == 0


@CLASSES
class TestConfigAndCheckpoint:
    def test_config_rebuilds_with_a_copy_not_the_same_module(self, cls: type[Model]) -> None:
        encoder = _mlp()
        model = _model(cls, encoder)
        config = model.get_config()
        copied = config["classical_encoder"]
        assert isinstance(copied, nn.Sequential) and copied is not encoder
        for name, value in copied.state_dict().items():
            torch.testing.assert_close(value, encoder.state_dict()[name], rtol=0, atol=0)
        rebuilt = cls(**config)
        assert _custom(rebuilt) is copied
        # Not shared: training one model leaves the other's encoder alone.
        with torch.no_grad():
            _first_linear(encoder).weight.add_(1.0)
        assert not torch.equal(_first_linear(copied).weight, _first_linear(encoder).weight)

    def test_builtin_config_records_none(self, cls: type[Model]) -> None:
        assert _model(cls).get_config()["classical_encoder"] is None

    def test_save_checkpoint_refuses_a_custom_encoder(
        self, cls: type[Model], tmp_path: Path
    ) -> None:
        path = tmp_path / "model.pt"
        with pytest.raises(ValueError, match="custom classical_encoder.*state_dict"):
            save_checkpoint(_model(cls, _mlp()), path)
        assert not path.exists()

    def test_builtin_encoder_round_trips(self, cls: type[Model], tmp_path: Path) -> None:
        model = _model(cls)
        save_checkpoint(model, tmp_path / "model.pt")
        loaded = load_checkpoint(tmp_path / "model.pt")
        assert loaded.get_config()["classical_encoder"] is None
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(_x()), model(_x()), rtol=0, atol=0)

    def test_state_dict_moves_weights_between_models_with_the_same_encoder(
        self, cls: type[Model]
    ) -> None:
        """The documented alternative to a checkpoint."""
        trained = _model(cls, _mlp(seed=0))
        fresh = _model(cls, _mlp(seed=5))
        fresh.load_state_dict(trained.state_dict())
        trained.eval()
        fresh.eval()
        with torch.no_grad():
            torch.testing.assert_close(fresh(_x()), trained(_x()), rtol=0, atol=0)


@CLASSES
class TestDiagnostics:
    def test_gradient_variance_reads_the_quantum_layer(self, cls: type[Model]) -> None:
        model = _model(cls, _mlp())
        whole = gradient_variance(model, n_samples=4)
        layer = gradient_variance(model.quantum_layer, n_samples=4)
        torch.testing.assert_close(whole.per_parameter, layer.per_parameter, rtol=0, atol=0)

    def test_disable_quantum_layer_replaces_only_the_quantum_output(
        self, cls: type[Model]
    ) -> None:
        model = _model(cls, _mlp())
        model.eval()
        with torch.no_grad(), disable_quantum_layer(model, fill=0.0):
            ablated = model(_x())
        with torch.no_grad():
            full = model(_x())
        assert not torch.equal(ablated, full)
        if cls is HybridBinaryClassifier:
            # The quantum layer is the only path to the head: constant output.
            assert torch.unique(ablated).numel() == 1
