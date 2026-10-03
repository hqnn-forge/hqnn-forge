"""
tests/test_classical_baseline.py
================================
``ClassicalBaseline`` and ``classical_baseline(model)`` (#178): the classical
control of an ablation study, matched in parameter count to the hybrid model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from hqnn_forge.diagnostics import circuit_summary
from hqnn_forge.models import (
    ClassicalBaseline,
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.models.classical_baseline import mlp_parameter_count
from hqnn_forge.training import train_model
from hqnn_forge.utils import classical_baseline, load_checkpoint, save_checkpoint

CPU: dict[str, Any] = dict(device_name="default.qubit", diff_method="backprop")


def _hybrid(cls: type, **kwargs: Any) -> Any:
    return cls(**{"n_input_features": 30, "n_qubits": 8, "n_layers": 2, **CPU, **kwargs})


def _live(hybrid: Any) -> int:
    """The count ``classical_baseline`` matches: total minus inert circuit weights."""
    return int(hybrid.count_parameters() - circuit_summary(hybrid).n_inert_params)


def _same_weights(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def _step(control: ClassicalBaseline) -> int:
    """Parameters one more unit of hidden width adds to the control."""
    dims = control.get_config()["hidden_dims"]
    n_in = control.get_config()["n_input_features"]
    wider = [d + 1 for d in dims]
    return mlp_parameter_count(n_in, wider) - mlp_parameter_count(n_in, dims)


class TestClassicalBaseline:
    def test_parameter_count_formula(self) -> None:
        model = ClassicalBaseline(7, [5, 3])
        assert (
            model.count_parameters()
            == mlp_parameter_count(7, [5, 3])
            == 7 * 5 + 5 + 5 * 3 + 3 + 3 + 1
        )

    def test_forward_is_one_logit_per_sample(self) -> None:
        assert ClassicalBaseline(4, [6]).forward(torch.randn(5, 4)).shape == (5, 1)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (dict(hidden_dims=[]), "at least one"),
            (dict(hidden_dims=[0]), "at least one"),
            (dict(hidden_dims=[3], activation="gelu"), "activation"),
            (dict(hidden_dims=[3], dropout_p=1.0), "dropout_p"),
        ],
    )
    def test_invalid_arguments(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ClassicalBaseline(4, **kwargs)

    def test_init_seed(self) -> None:
        """As on the hybrid classifiers (#175): reproducible from the seed, the
        global RNG left exactly as it was, and non-integers refused."""
        torch.manual_seed(1)
        a = ClassicalBaseline(4, [6, 2], init_seed=np.int64(3))  # type: ignore[arg-type]
        torch.manual_seed(2)
        before = torch.random.get_rng_state()
        b = ClassicalBaseline(4, [6, 2], init_seed=3)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert type(a.get_config()["init_seed"]) is int
        assert _same_weights(a, b)
        assert not _same_weights(a, ClassicalBaseline(4, [6, 2], init_seed=4))
        with pytest.raises(TypeError, match="init_seed"):
            ClassicalBaseline(4, [6], init_seed=1.5)  # type: ignore[arg-type]

    def test_a_failed_build_leaves_the_global_rng_alone(self) -> None:
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        with pytest.raises(ValueError, match="dropout_p"):
            ClassicalBaseline(4, [6], dropout_p=1.0, init_seed=3)
        assert torch.equal(torch.random.get_rng_state(), before)

    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        model = ClassicalBaseline(4, [6, 2], activation="tanh", dropout_p=0.1).eval()
        save_checkpoint(model, tmp_path / "c.pt")
        loaded = load_checkpoint(tmp_path / "c.pt")
        x = torch.randn(3, 4)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "cls", [HybridBinaryClassifier, ParallelHybridClassifier], ids=["serial", "parallel"]
)
class TestBuilder:
    def test_has_no_quantum_layer(self, cls: type) -> None:
        control = classical_baseline(_hybrid(cls))
        assert isinstance(control, ClassicalBaseline)
        assert not hasattr(control, "quantum_layer")
        assert all(
            type(m).__module__.startswith("torch.nn")
            for m in control.modules()
            if m is not control
        )

    @pytest.mark.parametrize(
        "sizes",
        [
            dict(),
            dict(n_input_features=5, n_qubits=3, n_layers=1),
            dict(n_layers=6),
            dict(readout="first"),
        ],
        ids=["default", "small", "deep", "first"],
    )
    def test_count_within_half_a_width_step(self, cls: type, sizes: dict) -> None:
        """The documented tolerance, around the live count: no integer width
        gets closer."""
        hybrid = _hybrid(cls, **sizes)
        control = classical_baseline(hybrid)
        gap = abs(control.count_parameters() - _live(hybrid))
        assert gap <= _step(control) / 2

    def test_matched_on_the_live_count_not_the_total(self, cls: type) -> None:
        """#234: the inert circuit weights are left out of the target."""
        hybrid = _hybrid(cls, readout="first")
        n_inert = circuit_summary(hybrid).n_inert_params
        assert n_inert > 0
        control = classical_baseline(hybrid)
        target = hybrid.count_parameters() - n_inert
        assert abs(control.count_parameters() - target) <= _step(control) / 2

    def test_no_neighbouring_width_is_closer(self, cls: type) -> None:
        hybrid = _hybrid(cls)
        control = classical_baseline(hybrid)
        n_in, dims = 30, control.get_config()["hidden_dims"]
        target = _live(hybrid)
        gap = abs(control.count_parameters() - target)
        for delta in (-1, 1):
            other = [d + delta for d in dims]
            assert gap <= abs(mlp_parameter_count(n_in, other) - target)

    def test_carries_input_width_and_dropout(self, cls: type) -> None:
        control = classical_baseline(_hybrid(cls, n_input_features=11, dropout_p=0.2))
        assert control.get_config()["n_input_features"] == 11
        assert control.get_config()["dropout_p"] == 0.2

    def test_a_seeded_hybrid_gets_a_seeded_control(self, cls: type) -> None:
        """The hybrid's init_seed is carried over: same weights whatever the
        global state, and the global RNG is neither reseeded nor advanced."""
        hybrid = _hybrid(cls, init_seed=7)
        torch.manual_seed(1)
        a = classical_baseline(hybrid)
        torch.manual_seed(2)
        before = torch.random.get_rng_state()
        b = classical_baseline(hybrid)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert a.get_config()["init_seed"] == 7
        assert _same_weights(a, b)
        assert _same_weights(a, ClassicalBaseline(**a.get_config()))

    def test_an_unseeded_hybrid_gets_an_unseeded_control(self, cls: type) -> None:
        hybrid = _hybrid(cls)
        torch.manual_seed(5)
        a = classical_baseline(hybrid)
        torch.manual_seed(5)
        b = classical_baseline(hybrid)
        assert a.get_config()["init_seed"] is None
        assert _same_weights(a, b)
        assert not _same_weights(a, classical_baseline(hybrid))

    def test_it_trains(self, cls: type) -> None:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(200, 30, generator=g)
        y = (x[:, 0] - x[:, 1] > 0).float()
        torch.manual_seed(0)
        control = classical_baseline(_hybrid(cls))
        history = train_model(
            control,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(control.parameters(), lr=0.01),
            x,
            y,
            max_epochs=30,
            batch_size=32,
        )
        assert history.epochs[-1].train_loss < 0.5 * history.epochs[0].train_loss
        assert (control.predict(x) == y.long()).float().mean() > 0.9


class TestShapes:
    def test_serial_is_one_tanh_hidden_layer(self) -> None:
        control = classical_baseline(_hybrid(HybridBinaryClassifier))
        assert control.get_config() == {
            "n_input_features": 30,
            "hidden_dims": [9],
            "activation": "tanh",
            "dropout_p": 0.0,
            "init_seed": None,
        }

    def test_parallel_is_the_widened_branch(self) -> None:
        control = classical_baseline(_hybrid(ParallelHybridClassifier))
        assert control.get_config()["hidden_dims"] == [20, 20]
        assert control.get_config()["activation"] == "relu"

    def test_published_shnn(self) -> None:
        """122 trainable, 102 live by autograd (test_published_shnn_parity), 106
        by the structural lower bound on the inert ones.  Both targets give the
        same width, h = 10: 101 parameters, against 121 matched on the total."""
        hybrid = HybridBinaryClassifier.published_shnn(**CPU)
        assert hybrid.count_parameters() == 122
        assert _live(hybrid) == 106
        control = classical_baseline(hybrid)
        assert control.get_config()["hidden_dims"] == [10]
        assert control.count_parameters() == 101
        n_in = control.get_config()["n_input_features"]
        exact = min(range(1, 20), key=lambda h: (abs(mlp_parameter_count(n_in, [h]) - 102), h))
        assert exact == 10

    def test_published_parallel(self) -> None:
        hybrid = ParallelHybridClassifier.published_shnn(**CPU)
        assert (hybrid.count_parameters(), _live(hybrid)) == (554, 538)
        assert classical_baseline(hybrid).count_parameters() == 523

    def test_other_models_are_refused(self) -> None:
        multiclass = MulticlassHybridClassifier(
            n_input_features=4, n_qubits=2, n_layers=1, n_classes=3, **CPU
        )
        with pytest.raises(TypeError, match="classical_baseline"):
            classical_baseline(multiclass)
