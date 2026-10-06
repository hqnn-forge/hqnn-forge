"""
tests/test_wire_patterns.py
===========================
The CNOT wire pattern -- order and direction, not only counts -- of every
entangler the encoders build (#183).  Gate counts, depth and parameter counts
are all invariant under a reversed ring, a changed range or a permuted wire
order, and the topology is load-bearing: which features a readout sees (#150)
follows from the ring's direction.  Each test compares the full list of
``(control, target)`` pairs, so a change fails with a readable diff.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import pytest
import torch

from hqnn_forge.diagnostics.circuit import _logical_tape
from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier

CPU: dict[str, Any] = dict(device_name="default.qubit", diff_method="backprop")
Pairs = Callable[[object], list[tuple[int, int]]]


def _ring(n: int, r: int = 1) -> list[tuple[int, int]]:
    """CNOT(i → i+r mod n) for i = 0 … n-1, applied in that order."""
    return [(i, (i + r) % n) for i in range(n)]


class TestAngleEncoder:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_ring_is_range_one_forward_in_every_layer(
        self, n_qubits: int, cnot_pairs: Pairs
    ) -> None:
        layer = QuantumEncodingLayer(n_qubits=n_qubits, n_layers=3, **CPU)
        assert cnot_pairs(layer) == _ring(n_qubits) * 3

    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_strongly_entangling_range_grows_with_the_layer(
        self, n_qubits: int, cnot_pairs: Pairs
    ) -> None:
        """Range r = ℓ mod (n-1) + 1: at 4 qubits and 4 layers, 1, 2, 3, then 1 again."""
        layer = QuantumEncodingLayer(
            n_qubits=n_qubits, n_layers=4, entangler="strongly_entangling", **CPU
        )
        expected = [p for ell in range(4) for p in _ring(n_qubits, ell % (n_qubits - 1) + 1)]
        assert cnot_pairs(layer) == expected


class TestIQPEncoder:
    @pytest.mark.parametrize(("n_qubits", "n_repeats"), [(3, 1), (4, 2)])
    def test_embedding_block_then_ring(
        self, n_qubits: int, n_repeats: int, cnot_pairs: Pairs
    ) -> None:
        """
        Each repeat: CNOT(i, j) · RZ(x_i x_j) on j · CNOT(i, j) for every pair
        i < j in lexicographic order, then the angle encoder's ring per layer.
        """
        layer = IQPEncodingLayer(n_qubits=n_qubits, n_layers=2, n_repeats=n_repeats, **CPU)
        block = [p for i, j in combinations(range(n_qubits), 2) for p in ((i, j), (i, j))]
        assert cnot_pairs(layer) == block * n_repeats + _ring(n_qubits) * 2


class TestOtherEncoders:
    def test_data_reuploading_uses_the_ring_between_uploads(self, cnot_pairs: Pairs) -> None:
        layer = DataReuploadingLayer(n_qubits=4, n_layers=3, **CPU)
        assert cnot_pairs(layer) == _ring(4) * 3

    def test_amplitude_encoder_ends_in_the_ring(self, cnot_pairs: Pairs) -> None:
        """The state preparation has CNOTs of its own; the ansatz is the last n·L."""
        layer = AmplitudeEncodingLayer(n_qubits=3, n_layers=2, **CPU)
        x = torch.rand(8, dtype=torch.float64)
        tape = _logical_tape(layer, inputs=x)
        assert cnot_pairs(tape)[-6:] == _ring(3) * 2


class TestClassifiersShareTheEncoders:
    @pytest.mark.parametrize("cls", [HybridBinaryClassifier, ParallelHybridClassifier])
    @pytest.mark.parametrize("encoding_type", ["angle", "iqp"])
    def test_classifier_circuit_is_its_encoder(
        self, cls: type, encoding_type: str, cnot_pairs: Pairs
    ) -> None:
        model = cls(n_input_features=5, n_qubits=3, n_layers=2, encoding_type=encoding_type, **CPU)
        block = [(0, 1), (0, 1), (0, 2), (0, 2), (1, 2), (1, 2)] if encoding_type == "iqp" else []
        assert cnot_pairs(model.quantum_layer) == block + _ring(3) * 2
