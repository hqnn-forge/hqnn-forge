"""
tests/test_published_shnn_default_backend.py
============================================
``published_shnn()`` on ``lightning.qubit`` with ``adjoint`` (#186).

The preset leaves ``device_name`` and ``diff_method`` at the library defaults.
Those were ``lightning.qubit`` with ``adjoint`` until #349; now they are
``"auto"``, which picks ``default.qubit`` with ``backprop`` at the preset's
8 qubits and ``lightning.qubit`` with ``adjoint`` above 12, so this module
names lightning/adjoint explicitly.  Every other test of the preset uses
``default.qubit`` with ``backprop``.  The combination that is otherwise
untested is ``readout="first"`` (a single ``qml.expval`` instead of one per
wire) and ``entangler="strongly_entangling"`` under adjoint, feeding a
``Linear(1 → 1)`` head, or the concatenated head of the parallel model.

Each test builds the preset twice with identical weights, once with no
override and once on ``default.qubit``/``backprop``, and compares them: the
quantum readout, the logits, every parameter's gradient, and the model after
one optimiser step.  Checking only that shapes come back and gradients are
non-zero would pass with adjoint gradients that are wrong.

The module skips without ``pennylane-lightning``: ``_resolve_device`` would
otherwise fall back to ``default.qubit`` and the comparison would measure
nothing.  The device the QNode is bound to is asserted as well, so a fallback
for any other reason fails instead of passing.  The tests take well under a
second, so they run in the default suite rather than behind an opt-in marker.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier

pytestmark = pytest.mark.requires_lightning

Model = HybridBinaryClassifier | ParallelHybridClassifier

LIGHTNING = {"device_name": "lightning.qubit", "diff_method": "adjoint"}
REFERENCE = {"device_name": "default.qubit", "diff_method": "backprop"}
BATCH = 5
# The two simulators agree to about 2e-9 here.  The tolerance leaves room for
# float32 rounding while still failing on a 1e-3 change to a single quantum
# weight, which moves the logits and gradients by about 1e-5.
ATOL, RTOL = 1e-7, 1e-6


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(BATCH, 8, generator=g)
    y = torch.tensor([[0.0], [1.0], [0.0], [1.0], [1.0]])
    return x, y


@pytest.fixture(
    params=[HybridBinaryClassifier, ParallelHybridClassifier], ids=lambda c: c.__name__
)
def pair(request: pytest.FixtureRequest) -> tuple[Model, Model]:
    """The preset on lightning/adjoint, and a ``default.qubit`` copy with its weights."""
    cls = request.param
    torch.manual_seed(0)
    # A failed lightning.qubit would warn and fall back; fail on that instead.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = cls.published_shnn(**LIGHTNING)
    reference = cls.published_shnn(**REFERENCE)
    reference.load_state_dict(model.state_dict())
    return model, reference


def _loss(model: Model) -> torch.Tensor:
    x, y = _inputs()
    return F.binary_cross_entropy_with_logits(model(x), y)


def test_default_preset_resolves_auto_to_backprop(
    pair: tuple[Model, Model],
) -> None:
    """At 8 qubits, "auto" gives the unmodified preset default.qubit/backprop."""
    qnode = type(pair[0]).published_shnn().quantum_layer.qlayer.qnode
    assert qnode.device.name == "default.qubit"
    assert qnode.diff_method == "backprop"


def test_preset_is_bound_to_lightning_adjoint(
    pair: tuple[Model, Model],
) -> None:
    model, _ = pair
    qnode = model.quantum_layer.qlayer.qnode
    assert qnode.device.name == "lightning.qubit"
    assert qnode.diff_method == "adjoint"
    assert model.quantum_layer.readout == "first"
    assert model.quantum_layer.entangler == "strongly_entangling"


def test_single_readout_and_logits_match_reference(
    pair: tuple[Model, Model],
) -> None:
    model, reference = pair
    x, _ = _inputs()
    with torch.no_grad():
        encoded = model.classical_encoder(x) * torch.pi
        readout = model.quantum_layer(encoded)
        expected_readout = reference.quantum_layer(encoded)
        logits, expected_logits = model(x), reference(x)

    # One ⟨Z_0⟩ per sample, not a flat (batch,) and not one per wire.
    assert readout.shape == (BATCH, 1)
    torch.testing.assert_close(readout, expected_readout, atol=ATOL, rtol=RTOL)
    assert logits.shape == (BATCH, 1)
    torch.testing.assert_close(logits, expected_logits, atol=ATOL, rtol=RTOL)


def test_every_parameter_gets_the_reference_gradient(
    pair: tuple[Model, Model],
) -> None:
    model, reference = pair
    _loss(model).backward()
    _loss(reference).backward()

    expected = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        # Nothing reaching a tensor at all would mean a detached path; a
        # tensor of zeros would mean the adjoint gradient never arrived.
        assert param.grad is not None, name
        assert param.grad.abs().max() > 0, name
        reference_grad = expected[name].grad
        assert reference_grad is not None, name
        torch.testing.assert_close(param.grad, reference_grad, atol=ATOL, rtol=RTOL, msg=name)


def test_one_optimiser_step_matches_reference(
    pair: tuple[Model, Model],
) -> None:
    """
    The step must update the weights the lightning QNode reads.  SGD rather
    than Adam: Adam's first step is about ``lr·sign(g)``, which turns float32
    rounding in a near-zero gradient into a full-sized difference.
    """
    model, reference = pair
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for m in (model, reference):
        optimiser = torch.optim.SGD(m.parameters(), lr=0.1)
        optimiser.zero_grad()
        _loss(m).backward()
        optimiser.step()

    expected = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        assert not torch.equal(param, before[name]), name
        torch.testing.assert_close(param, expected[name], atol=ATOL, rtol=RTOL, msg=name)
    with torch.no_grad():
        torch.testing.assert_close(_loss(model), _loss(reference), atol=ATOL, rtol=RTOL)
