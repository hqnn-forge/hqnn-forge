"""
tests/test_ablation.py
======================
Unit tests for hqnn_forge.utils.disable_quantum_layer and permute_quantum_layer.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, TypedDict, TypeVar

import pytest
import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.utils import disable_quantum_layer, permute_quantum_layer


class _Backend(TypedDict):
    device_name: DeviceName
    diff_method: DiffMethod


CPU: _Backend = {"device_name": "default.qubit", "diff_method": "backprop"}
N_FEATURES, N_QUBITS = 6, 3


Model = HybridBinaryClassifier | ParallelHybridClassifier
M = TypeVar("M", bound=Model)
_Ablation = Callable[[nn.Module], AbstractContextManager[nn.Module]]


def _model(cls: type[M], **kw: Any) -> M:
    torch.manual_seed(0)
    return cls(n_input_features=N_FEATURES, n_qubits=N_QUBITS, n_layers=2, **CPU, **kw)


def _qweights(model: Model) -> torch.Tensor:
    """The quantum layer's weights, narrowed from ``nn.Module.__getattr__``."""
    weights = model.quantum_layer.qlayer.weights
    assert isinstance(weights, torch.Tensor)
    return weights


MODELS = [
    pytest.param(HybridBinaryClassifier, id="serial"),
    pytest.param(ParallelHybridClassifier, id="parallel"),
]


@pytest.fixture
def x() -> torch.Tensor:
    return torch.randn(5, N_FEATURES, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("cls", MODELS)
class TestAblation:
    def test_output_ignores_quantum_weights(self, cls: type[Model], x: torch.Tensor) -> None:
        model = _model(cls)
        with disable_quantum_layer(model):
            before = model(x).detach()
            with torch.no_grad():
                _qweights(model).add_(1.0)
            after = model(x).detach()
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_no_gradient_reaches_the_quantum_branch(
        self, cls: type[Model], x: torch.Tensor
    ) -> None:
        model = _model(cls)
        with disable_quantum_layer(model):
            model(x).sum().backward()
        assert _qweights(model).grad is None
        for p in model.classical_encoder.parameters():
            assert p.grad is None
        assert model.head.weight.grad is not None

    def test_circuit_is_not_executed(
        self, cls: type[Model], x: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _model(cls)

        def boom(*_: object, **__: object) -> None:
            raise AssertionError("circuit executed")

        monkeypatch.setattr(model.quantum_layer.qlayer, "forward", boom)
        with disable_quantum_layer(model):
            model(x)

    def test_forward_is_restored(self, cls: type[Model], x: torch.Tensor) -> None:
        model = _model(cls)
        model.eval()
        with torch.no_grad():
            expected = model(x)
            with disable_quantum_layer(model):
                assert not torch.allclose(model(x), expected)
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        assert "forward" not in vars(model.quantum_layer)

    def test_restored_after_an_exception(self, cls: type[Model]) -> None:
        model = _model(cls)
        with pytest.raises(KeyError):
            with disable_quantum_layer(model):
                raise KeyError("inside")
        assert "forward" not in vars(model.quantum_layer)

    def test_predict_proba_works_inside(self, cls: type[Model], x: torch.Tensor) -> None:
        model = _model(cls)
        with disable_quantum_layer(model):
            probs = model.predict_proba(x)
        assert probs.shape == (5,)

    def test_nested_use_raises(self, cls: type[Model]) -> None:
        model = _model(cls)
        with disable_quantum_layer(model):
            with pytest.raises(RuntimeError, match="cannot be nested"):
                with disable_quantum_layer(model):
                    pass
        assert "forward" not in vars(model.quantum_layer)


class TestExactReplacement:
    def test_serial_output_is_head_of_constant(self, x: torch.Tensor) -> None:
        model = _model(HybridBinaryClassifier)
        with torch.no_grad(), disable_quantum_layer(model, fill=0.25):
            out = model(x)
            expected = model.head(torch.full((5, N_QUBITS), 0.25))
        torch.testing.assert_close(out, expected)

    def test_parallel_output_keeps_the_classical_branch(self, x: torch.Tensor) -> None:
        model = _model(ParallelHybridClassifier)
        with torch.no_grad(), disable_quantum_layer(model):
            out = model(x)
            fused = torch.cat([model.classical_branch(x), torch.zeros(5, N_QUBITS)], dim=-1)
            expected = model.head(fused)
        torch.testing.assert_close(out, expected)

    def test_serial_ablation_is_a_constant_predictor(self, x: torch.Tensor) -> None:
        # The quantum layer is the serial model's only input -> head path, so
        # ablating it leaves head(full(n_qubits, fill)): the same probability for
        # every sample.  The module docstring says so; this pins it.
        model = _model(HybridBinaryClassifier)
        with disable_quantum_layer(model):
            probs = model.predict_proba(x)
        assert torch.unique(probs).numel() == 1

    def test_iqp_layer_is_supported(self, x: torch.Tensor) -> None:
        model = _model(HybridBinaryClassifier, encoding_type="iqp")
        with torch.no_grad(), disable_quantum_layer(model) as layer:
            assert layer is model.quantum_layer
            torch.testing.assert_close(layer(torch.randn(2, N_QUBITS)), torch.zeros(2, N_QUBITS))

    def test_readout_first_is_filled_to_the_readout_width(self, x: torch.Tensor) -> None:
        # With readout="first" the layer emits one number, not n_qubits of them,
        # and the head is built for that width: filling n_qubits wide would make
        # the ablated forward fail where the real one works.
        model = _model(HybridBinaryClassifier, entangler="strongly_entangling", readout="first")
        with torch.no_grad(), disable_quantum_layer(model, fill=0.25) as layer:
            assert layer(torch.randn(5, N_QUBITS)).shape == (5, 1)
            torch.testing.assert_close(model(x), model.head(torch.full((5, 1), 0.25)))

    def test_training_inside_updates_only_live_parameters(self, x: torch.Tensor) -> None:
        model = _model(ParallelHybridClassifier)
        quantum_before = _qweights(model).detach().clone()
        encoder_before = [p.detach().clone() for p in model.classical_encoder.parameters()]
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with disable_quantum_layer(model):
            for _ in range(3):
                opt.zero_grad()
                model(x).pow(2).mean().backward()
                opt.step()
        torch.testing.assert_close(_qweights(model).detach(), quantum_before, rtol=0, atol=0)
        for before, p in zip(encoder_before, model.classical_encoder.parameters()):
            torch.testing.assert_close(p.detach(), before, rtol=0, atol=0)


class TestValidation:
    def test_model_without_quantum_layer(self) -> None:
        with pytest.raises(TypeError, match="expects a model with a quantum_layer.*got Linear"):
            with disable_quantum_layer(nn.Linear(2, 1)):
                pass

    def test_input_width_is_rejected_inside_the_block_too(self) -> None:
        # use_classical_encoder=False is the only way a wrong width reaches the
        # quantum layer; the ablated layer must refuse it exactly as the real
        # one does, or an ablation run reports numbers for rejected input.
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=N_QUBITS,
            n_qubits=N_QUBITS,
            n_layers=2,
            use_classical_encoder=False,
            **CPU,
        )
        wrong = torch.zeros(2, N_QUBITS + 1)
        with pytest.raises(ValueError, match="does not match n_qubits"):
            model(wrong)
        with disable_quantum_layer(model):
            with pytest.raises(ValueError, match="does not match n_qubits"):
                model(wrong)
        assert "forward" not in vars(model.quantum_layer)

    @pytest.mark.parametrize("fill", [-1.5, 1.01])
    def test_fill_out_of_range(self, fill: float) -> None:
        model = _model(HybridBinaryClassifier)
        with pytest.raises(ValueError, match=r"fill must lie in \[-1, 1\]"):
            with disable_quantum_layer(model, fill=fill):
                pass


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("cls", MODELS)
class TestPermutation:
    """#179: the circuit runs; its rows are shuffled across the batch."""

    def test_output_is_a_row_permutation_of_the_real_output(
        self, cls: type[Model], x: torch.Tensor
    ) -> None:
        model = _model(cls).eval()
        layer = model.quantum_layer
        q = torch.randn(5, N_QUBITS, generator=_gen(3))
        with torch.no_grad():
            real = layer(q)
            with permute_quantum_layer(model, generator=_gen()):
                shuffled = layer(q)
        perm = torch.randperm(5, generator=_gen())
        assert not torch.equal(perm, torch.arange(5))
        torch.testing.assert_close(shuffled, real[perm], rtol=0, atol=0)

    def test_same_seed_same_permutation(self, cls: type[Model], x: torch.Tensor) -> None:
        model = _model(cls).eval()
        outs = []
        for _ in range(2):
            with torch.no_grad(), permute_quantum_layer(model, generator=_gen(7)):
                outs.append(model(x))
        torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)

    def test_no_gradient_reaches_the_quantum_layer_or_upstream(
        self, cls: type[Model], x: torch.Tensor
    ) -> None:
        model = _model(cls)
        with permute_quantum_layer(model, generator=_gen()):
            model(x).sum().backward()
        assert model.quantum_layer.qlayer.weights.grad is None
        encoder = model.classical_encoder
        assert isinstance(encoder, nn.Sequential)
        assert encoder[0].weight.grad is None

    def test_forward_is_restored_including_on_exception(self, cls: type[Model]) -> None:
        model = _model(cls)
        with pytest.raises(RuntimeError, match="boom"), permute_quantum_layer(model):
            raise RuntimeError("boom")
        assert "forward" not in vars(model.quantum_layer)

    def test_cannot_nest_with_either_ablation(self, cls: type[Model]) -> None:
        model = _model(cls)
        ablations: list[tuple[_Ablation, _Ablation]] = [
            (permute_quantum_layer, permute_quantum_layer),
            (permute_quantum_layer, disable_quantum_layer),
            (disable_quantum_layer, permute_quantum_layer),
        ]
        for outer, inner in ablations:
            with outer(model), pytest.raises(RuntimeError, match="cannot be nested"):
                with inner(model):
                    pass
            assert "forward" not in vars(model.quantum_layer)


class TestPermutationDegenerateCases:
    def test_batch_of_one_warns(self) -> None:
        model = _model(HybridBinaryClassifier).eval()
        with (
            torch.no_grad(),
            permute_quantum_layer(model, generator=_gen()),
            pytest.warns(RuntimeWarning, match="batch of 1 unchanged.*nothing to swap"),
        ):
            model(torch.randn(1, N_FEATURES))

    def test_identity_draw_warns(self) -> None:
        seed = next(
            s
            for s in range(100)
            if torch.equal(torch.randperm(2, generator=_gen(s)), torch.arange(2))
        )
        model = _model(HybridBinaryClassifier).eval()
        with (
            torch.no_grad(),
            permute_quantum_layer(model, generator=_gen(seed)),
            pytest.warns(RuntimeWarning, match="identity permutation"),
        ):
            model(torch.randn(2, N_FEATURES))

    def test_a_real_permutation_is_silent(self, x: torch.Tensor) -> None:
        model = _model(HybridBinaryClassifier).eval()
        with torch.no_grad(), permute_quantum_layer(model, generator=_gen()):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                model(x)

    def test_serial_permuted_model_is_not_a_constant_predictor(self, x: torch.Tensor) -> None:
        """Unlike the constant null, the permutation is meaningful for the serial topology."""
        model = _model(HybridBinaryClassifier)
        with permute_quantum_layer(model, generator=_gen()):
            proba = model.predict_proba(x)
        assert proba.unique().numel() > 1

    def test_model_without_quantum_layer(self) -> None:
        with pytest.raises(TypeError, match="permute_quantum_layer expects"):
            with permute_quantum_layer(nn.Linear(2, 1)):
                pass
