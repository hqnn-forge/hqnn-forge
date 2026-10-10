"""
tests/test_binary_classifiers.py
=================================
Tests every binary classifier must pass, run against each one.

``BinaryClassifierBase`` implements ``predict_proba``, ``predict`` and
``count_parameters`` once; the two models below inherit them and share the
encoder-bypass and encoding-type constructor options.  Parametrising over the
classes means a fix to the shared API is tested on every model, and a third
classifier is added to the suite by extending ``CLASSIFIERS``.

Model-specific tests (exact parameter counts, branch wiring, gradient flow,
init-strategy variance) stay in the per-model files.
"""

from __future__ import annotations

from typing import Any, TypedDict, get_type_hints

import pytest
import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod
from hqnn_forge.models import (
    BinaryClassifierBase,
    ClassicalBaseline,
    HybridBinaryClassifier,
    LinearClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.utils.modes import eval_mode

BATCH = 8
N_QUBITS = 4
N_LAYERS = 2
N_RAW_FEATURES = 12
FIXTURE_SEED = 0

# The concrete classes. Fixtures and tests are typed with these rather than
# BinaryClassifierBase, which has no __init__ of its own (so mypy would accept any
# constructor arguments) and declares only the head: ClassicalBaseline has no
# classical_encoder or quantum_layer, so reads of those need the hybrid classes.
ConcreteClassifier = HybridBinaryClassifier | ParallelHybridClassifier

CLASSIFIERS = [
    pytest.param(HybridBinaryClassifier, id="serial"),
    pytest.param(ParallelHybridClassifier, id="parallel"),
]


# Constructor arguments common to both classes; CI-portable device and diff method.
class _DeviceKwargs(TypedDict):
    device_name: DeviceName
    diff_method: DiffMethod


DEVICE_KWARGS: _DeviceKwargs = {"device_name": "default.qubit", "diff_method": "parameter-shift"}


@pytest.fixture(scope="module", params=CLASSIFIERS)
def model_cls(request: pytest.FixtureRequest) -> type[ConcreteClassifier]:
    return request.param


@pytest.fixture(scope="module")
def classifier(model_cls: type[ConcreteClassifier]) -> ConcreteClassifier:
    """Small classifier of the parametrised class, with the classical encoder on."""
    torch.manual_seed(FIXTURE_SEED)
    return model_cls(
        n_input_features=N_RAW_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        use_classical_encoder=True,
        init_strategy="restricted",
        **DEVICE_KWARGS,
    )


@pytest.fixture
def random_raw_batch() -> torch.Tensor:
    """Seeded raw-feature batch, shape (BATCH, N_RAW_FEATURES)."""
    generator = torch.Generator().manual_seed(FIXTURE_SEED)
    return torch.randn(BATCH, N_RAW_FEATURES, generator=generator)


class TestBaseClass:
    def test_models_subclass_the_base(self, model_cls: type[ConcreteClassifier]) -> None:
        assert issubclass(model_cls, BinaryClassifierBase)
        assert issubclass(model_cls, nn.Module)

    def test_shared_methods_are_not_overridden(self, model_cls: type[ConcreteClassifier]) -> None:
        """A model that re-implements the shared API defeats the point of the base."""
        for name in ("predict_proba", "predict", "count_parameters"):
            assert getattr(model_cls, name) is getattr(BinaryClassifierBase, name), name

    def test_forward_is_abstract(self) -> None:
        class Incomplete(BinaryClassifierBase):
            pass

        with pytest.raises(NotImplementedError, match="Incomplete must implement forward"):
            Incomplete()(torch.zeros(1, 3))

    def test_declared_attributes_exist_on_every_subclass(self) -> None:
        """What the base declares, mypy accepts on any subclass; it must exist at runtime.

        Every ``BinaryClassifierBase`` subclass shipped in ``hqnn_forge`` is
        built and checked against every annotation on the base, so declaring an
        attribute one model lacks (``quantum_layer`` on ``ClassicalBaseline``)
        fails here instead of as an ``AttributeError`` in user code.
        """
        build_kwargs: dict[type[BinaryClassifierBase], dict[str, Any]] = {
            HybridBinaryClassifier: dict(n_input_features=4, n_qubits=2, **DEVICE_KWARGS),
            ParallelHybridClassifier: dict(n_input_features=4, n_qubits=2, **DEVICE_KWARGS),
            ClassicalBaseline: dict(n_input_features=4, hidden_dims=[3]),
            LinearClassifier: dict(n_input_features=4),
        }
        pending = list(BinaryClassifierBase.__subclasses__())
        shipped: set[type[BinaryClassifierBase]] = set()
        while pending:
            cls = pending.pop()
            pending.extend(cls.__subclasses__())
            if cls.__module__.startswith("hqnn_forge."):
                shipped.add(cls)
        assert shipped == set(build_kwargs), "add the new subclass to build_kwargs"

        hints = get_type_hints(BinaryClassifierBase)
        assert "head" in hints
        for cls, kwargs in build_kwargs.items():
            model = cls(**kwargs)
            for name, hint in hints.items():
                assert hasattr(model, name), f"{cls.__name__} lacks declared {name!r}"
                if isinstance(hint, type):
                    assert isinstance(getattr(model, name), hint), (cls.__name__, name)


class TestForwardShape:
    def test_output_shape(
        self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        assert classifier(random_raw_batch).shape == (BATCH, 1)

    def test_single_sample(self, classifier: ConcreteClassifier) -> None:
        assert classifier(torch.randn(1, N_RAW_FEATURES)).shape == (1, 1)


class TestPredictProba:
    def test_output_in_zero_one(
        self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        assert probs.min().item() >= 0.0 - 1e-6
        assert probs.max().item() <= 1.0 + 1e-6

    def test_output_shape(
        self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        assert classifier.predict_proba(random_raw_batch).shape == (BATCH,)

    def test_no_grad(self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor) -> None:
        assert not classifier.predict_proba(random_raw_batch).requires_grad

    def test_is_sigmoid_of_forward(
        self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        # predict_proba runs its forward under eval_mode, so the expectation has to
        # as well; a bare classifier.eval() would also leak eval mode into every later
        # test through the module-scoped fixture.
        with eval_mode(classifier), torch.no_grad():
            expected = torch.sigmoid(classifier(random_raw_batch)).squeeze(-1)
        torch.testing.assert_close(classifier.predict_proba(random_raw_batch), expected)


class TestPredict:
    def test_returns_binary(
        self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        preds = classifier.predict(random_raw_batch)
        assert set(torch.unique(preds).tolist()) <= {0, 1}

    def test_output_shape_and_dtype(
        self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        preds = classifier.predict(random_raw_batch)
        assert preds.shape == (BATCH,)
        assert preds.dtype == torch.long

    def test_threshold_is_applied(
        self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        probs = classifier.predict_proba(random_raw_batch)
        torch.testing.assert_close(
            classifier.predict(random_raw_batch, threshold=0.0),
            torch.ones(BATCH, dtype=torch.long),
        )
        torch.testing.assert_close(
            classifier.predict(random_raw_batch, threshold=1.01),
            torch.zeros(BATCH, dtype=torch.long),
        )
        mid = probs.median().item()
        torch.testing.assert_close(
            classifier.predict(random_raw_batch, threshold=mid), (probs >= mid).long()
        )


def _dropout_classifier(model_cls: type[ConcreteClassifier]) -> ConcreteClassifier:
    """Classifier with active dropout, in train mode as left by construction."""
    torch.manual_seed(FIXTURE_SEED)
    return model_cls(
        n_input_features=N_RAW_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        dropout_p=0.5,
        **DEVICE_KWARGS,
    )


class TestInferenceMode:
    """
    ``@torch.no_grad()`` does not disable ``nn.Dropout``.  Mode handling itself
    is tested in ``tests/test_modes.py``; this checks ``predict_proba`` uses it.
    """

    def test_train_mode_matches_eval_forward(
        self, model_cls: type[ConcreteClassifier], random_raw_batch: torch.Tensor
    ) -> None:
        """Repeated calls in train mode give the dropout-free eval probabilities."""
        model = _dropout_classifier(model_cls)
        model.eval()
        with torch.no_grad():
            expected = torch.sigmoid(model(random_raw_batch)).squeeze(-1)

        model.train()
        for _ in range(2):
            torch.testing.assert_close(model.predict_proba(random_raw_batch), expected)
        assert all(module.training for module in model.modules())


class TestParameterCount:
    def test_positive_count(self, classifier: ConcreteClassifier) -> None:
        assert classifier.count_parameters() > 0

    def test_matches_torch(self, classifier: ConcreteClassifier) -> None:
        assert classifier.count_parameters() == sum(p.numel() for p in classifier.parameters())
        assert classifier.count_parameters(trainable_only=False) == classifier.count_parameters()

    def test_trainable_only_excludes_frozen(self, model_cls: type[ConcreteClassifier]) -> None:
        model = model_cls(
            n_input_features=N_RAW_FEATURES, n_qubits=N_QUBITS, n_layers=N_LAYERS, **DEVICE_KWARGS
        )
        total = model.count_parameters(trainable_only=False)
        model.head.weight.requires_grad_(False)
        assert model.count_parameters() == total - model.head.weight.numel()
        assert model.count_parameters(trainable_only=False) == total


def _circuit_input(model: ConcreteClassifier, x: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return the tensor handed to the quantum layer."""
    captured: list[torch.Tensor] = []
    handle = model.quantum_layer.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    try:
        model(x)
    finally:
        handle.remove()
    return captured[0]


class TestEncoderBypass:
    def test_mismatched_dims_raises(self, model_cls: type[ConcreteClassifier]) -> None:
        with pytest.raises(ValueError, match="n_input_features"):
            model_cls(
                n_input_features=10, n_qubits=4, use_classical_encoder=False, **DEVICE_KWARGS
            )

    def test_matching_dims_works(self, model_cls: type[ConcreteClassifier]) -> None:
        model = model_cls(
            n_input_features=4,
            n_qubits=4,
            n_layers=1,
            use_classical_encoder=False,
            **DEVICE_KWARGS,
        )
        assert model(torch.randn(2, 4)).shape == (2, 1)

    def test_bypassed_input_reaches_circuit_unscaled(
        self, model_cls: type[ConcreteClassifier]
    ) -> None:
        """Bypassed input is already in (-π, π); a second π factor aliases angles mod 2π."""
        model = model_cls(
            n_input_features=4,
            n_qubits=4,
            n_layers=1,
            use_classical_encoder=False,
            **DEVICE_KWARGS,
        )
        x = torch.linspace(-3.0, 3.0, 8).reshape(2, 4)
        torch.testing.assert_close(_circuit_input(model, x), x)

    def test_encoder_output_scaled_by_pi(
        self, classifier: ConcreteClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        expected = classifier.classical_encoder(random_raw_batch).detach() * torch.pi
        torch.testing.assert_close(_circuit_input(classifier, random_raw_batch), expected)


class TestEncodingTypes:
    @pytest.mark.parametrize("encoding_type", ["angle", "iqp"])
    def test_supported_encodings(
        self, model_cls: type[ConcreteClassifier], encoding_type: str
    ) -> None:
        model = model_cls(
            n_input_features=4,
            n_qubits=4,
            n_layers=1,
            use_classical_encoder=False,
            encoding_type=encoding_type,
            **DEVICE_KWARGS,
        )
        assert model(torch.randn(2, 4)).shape == (2, 1)

    def test_invalid_encoding(self, model_cls: type[ConcreteClassifier]) -> None:
        with pytest.raises(ValueError, match="Unsupported encoding_type"):
            model_cls(n_input_features=4, n_qubits=4, n_layers=1, encoding_type="unknown_encoding")
