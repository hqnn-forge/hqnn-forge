"""
tests/test_parameter_shift.py
=============================
``diff_method="parameter-shift"`` -- the only gradient method that runs on
quantum hardware -- against backprop, for every encoder and circuit option (#315).

For each configuration, the same layer is built twice with identical weights,
once with ``backprop`` and once with ``parameter-shift`` on ``default.qubit``,
and the gradient of one fixed scalar cost is compared for every trainable
tensor and, where the layer supports it, for the inputs (a classical encoder
upstream always needs those).  The two are independent computations -- autograd
through the state vector against two shifted circuit evaluations per parameter
-- so agreement to float64 round-off pins the shift rules and the classical
processing around them (the IQP phases ``x_i x_j``, the re-uploading
``input_scaling``).  Adjoint differentiation on ``lightning.qubit``, the
default backend, is compared against the same backprop reference.

A gate without an analytic shift rule would make PennyLane fall back to finite
differences without an error; the method it picks for each parameter is
recorded during a real backward pass and checked.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pennylane as qml
import pennylane.gradients.parameter_shift as parameter_shift_module
import pytest
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)


def _lightning_available() -> bool:
    try:
        qml.device("lightning.qubit", wires=1)
    except Exception:  # noqa: BLE001 - any failure means "not installed"
        return False
    return True


requires_lightning = pytest.mark.skipif(
    not _lightning_available(), reason="pennylane-lightning not installed"
)

N_QUBITS = 3
# Everything runs in float64, where the methods agree to a few 1e-16
# (measured at most 1e-15 over these configurations).  1e-10 leaves margin
# while still catching a shift rule off by 1e-5 relative, which float32
# round-off (a few 1e-7) would hide in some configurations.
ATOL = 1e-10
# lightning.qubit's own forward pass differs from default.qubit's by ~1e-8
# in float64 (measured even for a single RX, under adjoint and
# parameter-shift alike), so its gradients can only agree to that.
LIGHTNING_ATOL = 1e-6


def _angle(**kw: Any) -> Any:
    return partial(QuantumEncodingLayer, **kw)


def _iqp(**kw: Any) -> Any:
    return partial(IQPEncodingLayer, **kw)


def _reuploading(**kw: Any) -> Any:
    return partial(DataReuploadingLayer, **kw)


ENCODERS = [
    pytest.param(_angle(), True, id="angle"),
    pytest.param(_angle(rotation="Y"), True, id="angle-y"),
    # No rotation="Z": on |0⟩ an RZ embedding is a phase, so the inputs have no
    # effect and their gradient is 0 under both methods (#212; #275 refuses it).
    pytest.param(_angle(entangler="strongly_entangling"), True, id="angle-strongly"),
    pytest.param(_angle(entangler="brickwork"), True, id="angle-brickwork"),
    pytest.param(_angle(readout="first"), True, id="angle-first"),
    pytest.param(
        _angle(rotation="Y", entangler="strongly_entangling", readout="first"),
        True,
        id="angle-published-shnn",
    ),
    pytest.param(_iqp(), True, id="iqp"),
    pytest.param(_iqp(n_repeats=2, entangler="strongly_entangling"), True, id="iqp-repeats"),
    pytest.param(_iqp(entangler="brickwork"), True, id="iqp-brickwork"),
    pytest.param(_iqp(readout="first"), True, id="iqp-first"),
    pytest.param(_reuploading(), True, id="reuploading"),
    pytest.param(_reuploading(trainable_input_scaling=True), True, id="reuploading-scaled"),
    pytest.param(
        _reuploading(trainable_input_scaling=True, rotation="Z"), True, id="reuploading-scaled-z"
    ),
    pytest.param(
        _reuploading(entangler="strongly_entangling", readout="first"),
        True,
        id="reuploading-strongly-first",
    ),
    pytest.param(_reuploading(entangler="brickwork"), True, id="reuploading-brickwork"),
    # Input gradients are refused under every method but backprop (see the
    # amplitude module docstring), so only the weights are compared.
    pytest.param(partial(AmplitudeEncodingLayer, n_features=5), False, id="amplitude"),
]


def _build(factory: Any, device_name: str, diff_method: str) -> Any:
    return factory(
        n_qubits=N_QUBITS, n_layers=2, device_name=device_name, diff_method=diff_method
    ).double()


def _pair(
    factory: Any, device_name: str = "default.qubit", diff_method: str = "parameter-shift"
) -> tuple[Any, Any]:
    torch.manual_seed(0)
    backprop = _build(factory, "default.qubit", "backprop")
    other = _build(factory, device_name, diff_method)
    other.load_state_dict(backprop.state_dict())
    # Trained-looking weights rather than the init's small angles, so no
    # gradient is near zero by construction.
    with torch.no_grad():
        for p_b, p_o in zip(backprop.parameters(), other.parameters(), strict=True):
            values = torch.empty_like(p_b).uniform_(-torch.pi, torch.pi)
            p_b.copy_(values)
            p_o.copy_(values)
    return backprop, other


def _inputs(layer: Any, requires_grad: bool) -> torch.Tensor:
    width = getattr(layer, "n_features", N_QUBITS)
    x = torch.rand(4, width, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    return (x * 2 - 1).requires_grad_(requires_grad)


def _gradients(layer: Any, x: torch.Tensor) -> list[torch.Tensor]:
    out = layer(x)
    assert out.dtype == torch.float64
    # A fixed, sign-varying weighting, so no cancellation hides a wrong term.
    weighting = torch.linspace(-1.0, 1.0, out.numel(), dtype=out.dtype).reshape(out.shape)
    targets = [*layer.parameters(), *([x] if x.requires_grad else [])]
    return list(torch.autograd.grad((out * weighting).sum(), targets))


def _assert_matches_backprop(
    factory: Any, input_grads: bool, device_name: str, diff_method: str, atol: float
) -> None:
    backprop, other = _pair(factory, device_name, diff_method)
    x = _inputs(backprop, input_grads)
    expected = _gradients(backprop, x)
    got = _gradients(other, x.detach().requires_grad_(input_grads))
    assert len(got) == len(expected) >= 1
    for g_other, g_backprop in zip(got, expected, strict=True):
        assert g_backprop.abs().max() > 1e-3  # a gradient worth comparing
        torch.testing.assert_close(g_other, g_backprop, rtol=0, atol=atol)


@pytest.mark.parametrize("factory, input_grads", ENCODERS)
class TestAgainstBackprop:
    def test_every_gradient_matches(self, factory: Any, input_grads: bool) -> None:
        _assert_matches_backprop(factory, input_grads, "default.qubit", "parameter-shift", ATOL)

    @requires_lightning
    def test_adjoint_on_lightning_matches(self, factory: Any, input_grads: bool) -> None:
        # The default backend's method, against the same reference.
        _assert_matches_backprop(
            factory, input_grads, "lightning.qubit", "adjoint", LIGHTNING_ATOL
        )

    def test_no_parameter_falls_back_to_finite_differences(
        self, factory: Any, input_grads: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # param_shift picks a method per trainable parameter -- "A" (analytic
        # shift rule), "0" (provably zero) or "F" (finite differences) -- and
        # runs "F" without an error.  Record the choices it actually makes
        # during a real backward pass and refuse any "F".
        chosen: list[dict[int, str]] = []
        original = parameter_shift_module.find_and_validate_gradient_methods

        def spy(*args: Any, **kwargs: Any) -> dict[int, str]:
            methods = original(*args, **kwargs)
            chosen.append(dict(methods))
            return methods  # type: ignore[no-any-return]

        monkeypatch.setattr(parameter_shift_module, "find_and_validate_gradient_methods", spy)
        _, shift = _pair(factory)
        _gradients(shift, _inputs(shift, input_grads))
        assert chosen, "the parameter-shift transform never ran"
        methods = {m for per_tape in chosen for m in per_tape.values()}
        assert "F" not in methods and "A" in methods, methods


def _binary(cls: type) -> Any:
    def build(encoding_type: str, diff_method: str) -> Any:
        return cls(
            n_input_features=4,
            n_qubits=2,
            n_layers=2,
            encoding_type=encoding_type,
            device_name="default.qubit",
            diff_method=diff_method,
            init_seed=0,
        )

    return build


def _multiclass(strategy: str) -> Any:
    def build(encoding_type: str, diff_method: str) -> Any:
        return MulticlassHybridClassifier(
            n_input_features=4,
            n_qubits=2,
            n_layers=2,
            n_classes=3,
            strategy=strategy,  # type: ignore[arg-type]
            encoding_type=encoding_type,
            device_name="default.qubit",
            diff_method=diff_method,  # type: ignore[arg-type]
            init_seed=0,
        )

    return build


CLASSIFIERS = [
    pytest.param(_binary(HybridBinaryClassifier), "binary", id="HybridBinaryClassifier"),
    pytest.param(_binary(ParallelHybridClassifier), "binary", id="ParallelHybridClassifier"),
    pytest.param(_multiclass("softmax"), "softmax", id="Multiclass-softmax"),
    pytest.param(_multiclass("one_vs_rest"), "one_vs_rest", id="Multiclass-one_vs_rest"),
]


@pytest.mark.parametrize("build, task", CLASSIFIERS)
@pytest.mark.parametrize("encoding_type", ["angle", "iqp"])
def test_classifier_trains_under_parameter_shift(
    build: Any, task: str, encoding_type: str
) -> None:
    # The whole model -- encoder, circuit, head -- trained end to end with the
    # hardware-compatible method, and its first step equal to backprop's.
    shift = build(encoding_type, "parameter-shift").double()
    backprop = build(encoding_type, "backprop").double()
    x = torch.randn(16, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    if task == "binary":
        y = (x[:, 0] > 0).double()

        def loss(model: Any) -> torch.Tensor:
            return torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)

    else:
        labels = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()

        def loss(model: Any) -> torch.Tensor:
            if task == "softmax":
                return torch.nn.functional.cross_entropy(model(x), labels)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                model(x), model.one_hot(labels)
            )

    for g_s, g_b in zip(
        torch.autograd.grad(loss(shift), list(shift.parameters())),
        torch.autograd.grad(loss(backprop), list(backprop.parameters())),
        strict=True,
    ):
        torch.testing.assert_close(g_s, g_b, rtol=0, atol=ATOL)

    opt = torch.optim.Adam(shift.parameters(), lr=0.1)
    before = loss(shift).item()
    for _ in range(10):
        opt.zero_grad()
        loss(shift).backward()
        opt.step()
    assert loss(shift).item() < before


@pytest.mark.parametrize("model", ["serial", "parallel"])
def test_sklearn_estimator_fits_under_parameter_shift(model: str) -> None:
    # The estimator passes diff_method through to the model it builds; a fit
    # under parameter-shift follows the backprop fit from the same seed (the
    # two differ only by float32 round-off accumulated over the steps).
    # scikit-learn is an optional extra: skip this test alone without it.
    pytest.importorskip("sklearn")
    from hqnn_forge.sklearn import HybridClassifierEstimator

    rng = np.random.default_rng(3)
    X = rng.normal(size=(32, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    def fit(diff_method: str) -> HybridClassifierEstimator:
        return HybridClassifierEstimator(
            model=model,  # type: ignore[arg-type]
            n_qubits=2,
            n_layers=2,
            device_name="default.qubit",
            diff_method=diff_method,
            loss="bce",
            lr=0.05,
            max_epochs=3,
            batch_size=16,
            threshold=0.5,
            random_state=0,
        ).fit(X, y)

    shift, backprop = fit("parameter-shift"), fit("backprop")
    assert shift.model_.quantum_layer.qlayer.qnode.diff_method == "parameter-shift"
    np.testing.assert_allclose(shift.predict_proba(X), backprop.predict_proba(X), atol=1e-4)
