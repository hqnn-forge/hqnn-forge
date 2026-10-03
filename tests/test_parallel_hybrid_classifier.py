"""
tests/test_parallel_hybrid_classifier.py
=========================================
Tests specific to hqnn_forge.models.ParallelHybridClassifier: exact parameter
count, branch wiring, gradient flow and init strategies.  The inference, bypass
and encoding-type tests shared with every classifier live in
tests/test_binary_classifiers.py.
"""

from __future__ import annotations

import math

import pytest
import torch

from hqnn_forge.models import ParallelHybridClassifier

BATCH = 8
N_QUBITS = 4
N_LAYERS = 2
N_RAW_FEATURES = 12
CLASSICAL_HIDDEN_DIM = 6
FIXTURE_SEED = 0


@pytest.fixture(scope="module")
def classifier() -> ParallelHybridClassifier:
    """Small ParallelHybridClassifier for unit tests."""
    torch.manual_seed(FIXTURE_SEED)
    return ParallelHybridClassifier(
        n_input_features=N_RAW_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        classical_hidden_dim=CLASSICAL_HIDDEN_DIM,
        use_classical_encoder=True,
        device_name="default.qubit",
        diff_method="parameter-shift",
        init_strategy="restricted",
    )


@pytest.fixture
def random_raw_batch() -> torch.Tensor:
    """Seeded raw-feature batch, shape (BATCH, N_RAW_FEATURES).

    Seeded so the non-zero gradient assertions check the same draw every run.
    """
    generator = torch.Generator().manual_seed(FIXTURE_SEED)
    return torch.randn(BATCH, N_RAW_FEATURES, generator=generator)


class TestParameterCount:
    def test_exact_count(self, classifier: ParallelHybridClassifier) -> None:
        """
        Pin the exact parameter count, block by block.  An inequality against
        the serial model passes for almost any MLP width, so it would not catch
        a mis-sized head or a wrongly wired branch — this does.
        """
        branch = (
            (N_RAW_FEATURES * CLASSICAL_HIDDEN_DIM + CLASSICAL_HIDDEN_DIM)  # Linear 1
            + (CLASSICAL_HIDDEN_DIM * CLASSICAL_HIDDEN_DIM + CLASSICAL_HIDDEN_DIM)  # Linear 2
        )
        encoder = N_RAW_FEATURES * N_QUBITS + N_QUBITS
        quantum = N_LAYERS * N_QUBITS * 3
        head = (CLASSICAL_HIDDEN_DIM + N_QUBITS) * 1 + 1

        expected = branch + encoder + quantum + head
        assert expected == 207, "test constants drifted from the documented config"
        assert classifier.count_parameters() == expected

    def test_head_consumes_both_branches(self, classifier: ParallelHybridClassifier) -> None:
        """The head must be wired to the *concatenated* width, not one branch."""
        assert classifier.head.in_features == CLASSICAL_HIDDEN_DIM + N_QUBITS

    def test_exceeds_serial_classifier(self, classifier: ParallelHybridClassifier) -> None:
        """Parallel topology adds an MLP branch, so it must have strictly more
        parameters than the equivalent serial HybridBinaryClassifier."""
        from hqnn_forge.models import HybridBinaryClassifier

        serial = HybridBinaryClassifier(
            n_input_features=N_RAW_FEATURES,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            use_classical_encoder=True,
            device_name="default.qubit",
            diff_method="parameter-shift",
            init_strategy="restricted",
        )
        assert classifier.count_parameters() > serial.count_parameters()


class TestGradientFlow:
    def test_gradients_reach_classical_branch(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        for param in classifier.classical_branch.parameters():
            assert param.grad is not None
            assert torch.any(param.grad != 0)

    def test_gradients_reach_quantum_branch(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        quantum_weights = classifier.quantum_layer.qlayer.weights
        assert quantum_weights.grad is not None
        assert torch.any(quantum_weights.grad != 0)

    def test_gradients_reach_classical_encoder(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        """
        The encoder's only path to the loss runs back through the circuit, so
        this also checks that *input* gradients cross the quantum layer — not
        just gradients w.r.t. its weights.
        """
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        encoder_params = list(classifier.classical_encoder.parameters())
        assert encoder_params, "fixture must build with use_classical_encoder=True"
        for param in encoder_params:
            assert param.grad is not None
            assert torch.any(param.grad != 0)

    def test_gradients_reach_head(
        self, classifier: ParallelHybridClassifier, random_raw_batch: torch.Tensor
    ) -> None:
        classifier.zero_grad()
        out = classifier(random_raw_batch)
        out.sum().backward()

        assert classifier.head.weight.grad is not None
        assert torch.any(classifier.head.weight.grad != 0)


# Wider/deeper than the shared fixture: each layer holds n_qubits * 3 weights,
# and at 4 qubits the per-layer std estimate is far too noisy to distinguish the
# two strategies without flaking (cf. #21).
#
# ``restricted`` is checked by the slope of log(std_l) against log(l + 1),
# fitted over all layers: 0 for one shared sigma.  At 16 qubits / 16 layers the
# slope has sd 0.034, so the +/-0.25 band sits ~7 sd out — no failures over
# 500 000 simulated draws.  ``block_local`` is checked exactly against
# ``restricted`` built from the same seed, so it needs no tolerance.  The
# margin, not INIT_SEED, is what keeps these stable when __init__ changes
# re-roll the RNG.
INIT_N_QUBITS = 16
INIT_N_LAYERS = 16
INIT_SEED = 0
SLOPE_TOLERANCE = 0.25


def _build_for_init(strategy: str) -> ParallelHybridClassifier:
    torch.manual_seed(INIT_SEED)
    return ParallelHybridClassifier(
        n_input_features=INIT_N_QUBITS,
        n_qubits=INIT_N_QUBITS,
        n_layers=INIT_N_LAYERS,
        use_classical_encoder=False,
        device_name="default.qubit",
        diff_method="parameter-shift",
        init_strategy=strategy,
    )


def _log_std_slope(model: ParallelHybridClassifier) -> float:
    """Least-squares slope of log(per-layer std) against log(layer_index + 1)."""
    w = model.quantum_layer.qlayer.weights.data.double()
    y = torch.log(w.flatten(start_dim=1).std(dim=1))
    x = torch.log(torch.arange(1, w.shape[0] + 1, dtype=torch.float64))
    x, y = x - x.mean(), y - y.mean()
    return ((x * y).sum() / (x * x).sum()).item()


class TestInitStrategies:
    def test_restricted_variance_is_flat_across_layers(self) -> None:
        """``restricted`` uses one shared sigma, so log-std has zero slope in depth."""
        slope = _log_std_slope(_build_for_init("restricted"))
        assert abs(slope) < SLOPE_TOLERANCE

    def test_restricted_matches_documented_sigma(self) -> None:
        """sigma = pi / sqrt(n_qubits * n_layers), per the initializer docstring."""
        model = _build_for_init("restricted")
        expected = math.pi / math.sqrt(INIT_N_QUBITS * INIT_N_LAYERS)
        actual = model.quantum_layer.qlayer.weights.data.std().item()
        assert actual == pytest.approx(expected, rel=0.25)

    def test_block_local_is_restricted_tapered_by_layer(self) -> None:
        """
        ``block_local`` uses sigma_l = pi / sqrt(n_qubits * (L + l)) (#166).
        From the same seed that is exactly the ``restricted`` weights times
        sqrt(L / (L + l)) on layer l.  This checks the model passes the
        initialiser its whole weight tensor, whose first dimension is the
        depth L.
        """
        restricted = _build_for_init("restricted").quantum_layer.qlayer.weights.data
        tapered = _build_for_init("block_local").quantum_layer.qlayer.weights.data
        factor = torch.tensor(
            [math.sqrt(INIT_N_LAYERS / (INIT_N_LAYERS + l)) for l in range(INIT_N_LAYERS)]
        )
        torch.testing.assert_close(tapered, restricted * factor.view(-1, 1, 1), rtol=1e-6, atol=0)

    def test_strategies_produce_different_weights(self) -> None:
        """
        Wiring check: identical seeds, different strategy — the weights must
        differ.  If ``init_strategy`` were silently ignored these would match.
        """
        restricted = _build_for_init("restricted").quantum_layer.qlayer.weights.data
        block_local = _build_for_init("block_local").quantum_layer.qlayer.weights.data
        assert not torch.allclose(restricted, block_local)
