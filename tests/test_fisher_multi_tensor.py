"""
tests/test_fisher_multi_tensor.py
=================================
Fisher diagnostics over several trainable tensors (#308): the re-uploading
layer's ``weights`` and ``input_scaling``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from hqnn_forge.diagnostics import effective_dimension, fisher_information_matrix
from hqnn_forge.encoding import DataReuploadingLayer

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}


def _layer() -> Any:
    torch.manual_seed(0)
    layer = DataReuploadingLayer(n_qubits=3, n_layers=2, trainable_input_scaling=True, **CPU)
    with torch.no_grad():  # away from the identity start, so every block is non-trivial
        layer.qlayer.input_scaling.uniform_(0.5, 1.5)
    return layer


def _x() -> torch.Tensor:
    return torch.rand(5, 3, generator=torch.Generator().manual_seed(1)) * 2 - 1


def test_the_matrix_spans_both_tensors_in_argument_order() -> None:
    layer = _layer()
    spectrum = fisher_information_matrix(layer, _x())
    assert list(spectrum.parameter_slices) == ["weights", "input_scaling"]
    assert spectrum.parameter_slices["weights"] == slice(0, 18)
    assert spectrum.parameter_slices["input_scaling"] == slice(18, 24)
    assert spectrum.n_params == 24 and spectrum.matrix.shape == (24, 24)


def test_the_matrix_equals_an_independent_jacobian_computation() -> None:
    # Gaussian likelihood: F = E_x[Jᵀ J] with J = d outputs / d (weights, input_scaling).
    layer = _layer()
    x = _x()
    w0 = layer.qlayer.weights.detach().double()
    s0 = layer.qlayer.input_scaling.detach().double()
    qnode = layer.qlayer.qnode

    def outputs(flat: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        w = flat[:18].reshape(w0.shape)
        s = flat[18:].reshape(s0.shape)
        return torch.stack(qnode(xi, weights=w, input_scaling=s))

    flat = torch.cat([w0.reshape(-1), s0.reshape(-1)])
    expected = torch.zeros(24, 24, dtype=torch.float64)
    for xi in x.double():
        jac = torch.autograd.functional.jacobian(lambda f, xi=xi: outputs(f, xi), flat)
        expected += jac.T @ jac
    expected /= x.shape[0]
    got = fisher_information_matrix(layer, x).matrix
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
    # Both blocks and the coupling carry information.
    assert expected[:18, :18].abs().max() > 1e-3
    assert expected[18:, 18:].abs().max() > 1e-3
    assert expected[:18, 18:].abs().max() > 1e-3


@pytest.mark.parametrize("name", ["weights", "input_scaling"])
def test_a_subset_is_the_corresponding_block(name: str) -> None:
    layer = _layer()
    x = _x()
    full = fisher_information_matrix(layer, x)
    part = fisher_information_matrix(layer, x, parameters=[name])
    assert list(part.parameter_slices) == [name]
    torch.testing.assert_close(part.matrix, full.block(name), rtol=0, atol=1e-12)


def test_frozen_tensors_are_measured_and_stay_frozen() -> None:
    layer = _layer()
    layer.qlayer.input_scaling.requires_grad_(False)
    x = _x()
    spectrum = fisher_information_matrix(layer, x)
    assert not layer.qlayer.input_scaling.requires_grad
    assert spectrum.block("input_scaling").abs().max() > 1e-3


@pytest.mark.parametrize("parameters", [["scale"], [], ["weights", "weights"]])
def test_unknown_or_empty_parameters(parameters: list[str]) -> None:
    with pytest.raises(
        ValueError, match="parameters must name some of 'weights', 'input_scaling'"
    ):
        fisher_information_matrix(_layer(), _x(), parameters=parameters)


def test_effective_dimension_counts_every_tensor_and_draws_only_the_angles() -> None:
    layer = _layer()
    scaling = layer.qlayer.input_scaling.detach().clone()
    weights = layer.qlayer.weights.detach().clone()
    result = effective_dimension(layer, _x(), n_data=100, n_theta_samples=3)
    assert result.n_params == 24
    assert result.mean_normalized_spectrum.shape == (24,)
    assert 0 < result.effective_dimension <= 1.4 * 24
    torch.testing.assert_close(layer.qlayer.input_scaling.detach(), scaling, rtol=0, atol=0)
    torch.testing.assert_close(layer.qlayer.weights.detach(), weights, rtol=0, atol=0)
