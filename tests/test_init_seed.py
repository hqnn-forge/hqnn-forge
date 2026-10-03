"""
tests/test_init_seed.py
=======================
``init_seed`` on the classifiers and a seeded ``HybridClassifierEstimator.fit``
(#175): the weights are reproducible from the seed, and the global torch RNG is
left exactly as the caller had it -- not reseeded, and not advanced either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.utils import load_checkpoint, save_checkpoint
from hqnn_forge.utils.rng import seeded_rng

CPU = dict(device_name="default.qubit", diff_method="backprop")
CLASSES = [
    pytest.param(HybridBinaryClassifier, {}, id="serial"),
    pytest.param(ParallelHybridClassifier, {}, id="parallel"),
    pytest.param(MulticlassHybridClassifier, {"n_classes": 3}, id="multiclass"),
]


def _build(cls: type, extra: dict, **kwargs: object) -> Any:
    return cls(n_input_features=5, n_qubits=3, n_layers=2, **CPU, **extra, **kwargs)


def _same_weights(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


@pytest.mark.parametrize(("cls", "extra"), CLASSES)
class TestInitSeed:
    def test_same_seed_same_weights_whatever_the_global_state(
        self, cls: type, extra: dict
    ) -> None:
        torch.manual_seed(1)
        a = _build(cls, extra, init_seed=7)
        torch.manual_seed(2)
        b = _build(cls, extra, init_seed=7)
        assert _same_weights(a, b)

    def test_different_seeds_differ(self, cls: type, extra: dict) -> None:
        assert not _same_weights(_build(cls, extra, init_seed=7), _build(cls, extra, init_seed=8))

    def test_the_global_rng_is_left_exactly_as_it_was(self, cls: type, extra: dict) -> None:
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        _build(cls, extra, init_seed=7)
        assert torch.equal(torch.random.get_rng_state(), before)

    def test_none_keeps_drawing_from_the_global_rng(self, cls: type, extra: dict) -> None:
        torch.manual_seed(5)
        a = _build(cls, extra)
        torch.manual_seed(5)
        b = _build(cls, extra)
        assert _same_weights(a, b)
        assert a.get_config()["init_seed"] is None

    def test_numpy_integer_is_stored_as_int(self, cls: type, extra: dict) -> None:
        model = _build(cls, extra, init_seed=np.int64(7))
        assert type(model.get_config()["init_seed"]) is int
        assert _same_weights(model, _build(cls, extra, init_seed=7))

    @pytest.mark.parametrize("bad", [1.5, "7", True])
    def test_non_integers_are_refused(self, cls: type, extra: dict, bad: object) -> None:
        with pytest.raises(TypeError, match="init_seed"):
            _build(cls, extra, init_seed=bad)

    def test_checkpoint_round_trips_it(self, cls: type, extra: dict, tmp_path: Path) -> None:
        model = _build(cls, extra, init_seed=np.int64(3))
        save_checkpoint(model, tmp_path / "m.pt")
        assert load_checkpoint(tmp_path / "m.pt").get_config()["init_seed"] == 3


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        # The classical branch is built before the width check raises.
        pytest.param(
            ParallelHybridClassifier,
            dict(n_input_features=5, n_qubits=3, use_classical_encoder=False),
            id="parallel-width",
        ),
        # The classical encoder is built before the IQP rotation check raises.
        pytest.param(
            HybridBinaryClassifier,
            dict(n_input_features=5, n_qubits=3, encoding_type="iqp", embedding_rotation="Y"),
            id="serial-iqp-rotation",
        ),
    ],
)
def test_a_failed_build_leaves_the_global_rng_alone(cls: type, kwargs: dict) -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    with pytest.raises(ValueError):
        cls(**CPU, **kwargs, init_seed=7)
    assert torch.equal(torch.random.get_rng_state(), before)


class TestSeededRng:
    def test_draws_are_those_of_manual_seed(self) -> None:
        torch.manual_seed(9)
        expected = torch.rand(5)
        torch.manual_seed(0)
        with seeded_rng(9) as reseed:
            first = torch.rand(5)
            reseed()
            again = torch.rand(5)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(again, expected, rtol=0, atol=0)

    def test_state_is_restored_when_the_block_raises(self) -> None:
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        with pytest.raises(RuntimeError), seeded_rng(9):
            torch.rand(5)
            raise RuntimeError
        assert torch.equal(torch.random.get_rng_state(), before)

    def test_none_runs_on_the_global_rng(self) -> None:
        torch.manual_seed(4)
        expected = torch.rand(5)
        torch.manual_seed(4)
        with seeded_rng(None) as reseed:
            reseed()
            torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)

    @pytest.mark.may_skip  # no CUDA device on the CI runners
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_state_is_restored(self) -> None:
        torch.cuda.init()
        torch.cuda.manual_seed_all(123)
        before = torch.cuda.get_rng_state_all()
        with seeded_rng(7):
            torch.rand(5, device="cuda")
        after = torch.cuda.get_rng_state_all()
        assert all(torch.equal(a, b) for a, b in zip(after, before))


class TestEstimator:
    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 4)).astype(np.float32)
        y = (X[:, 0] + 0.3 * rng.normal(size=40) > 0).astype(int)
        return X, y

    def _estimator(self, **kwargs: Any) -> Any:
        from hqnn_forge.sklearn import HybridClassifierEstimator

        return HybridClassifierEstimator(
            n_qubits=2,
            n_layers=1,
            max_epochs=2,
            batch_size=16,
            dropout_p=0.3,
            **CPU,  # type: ignore[arg-type]
            **kwargs,
        )

    def test_seeded_fit_leaves_the_global_rng_alone(self, data: tuple) -> None:
        X, y = data
        torch.manual_seed(123)
        expected = torch.randn(3)
        torch.manual_seed(123)
        self._estimator(random_state=0).fit(X, y)
        torch.testing.assert_close(torch.randn(3), expected, rtol=0, atol=0)

    def test_seeded_fit_is_reproducible_with_dropout(self, data: tuple) -> None:
        """Dropout masks are torch draws too; they come from the private RNG."""
        X, y = data
        torch.manual_seed(1)
        a = self._estimator(random_state=np.int64(4)).fit(X, y).predict_proba(X)
        torch.manual_seed(2)
        b = self._estimator(random_state=4).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)

    def test_the_fitted_model_records_the_seed(self, data: tuple) -> None:
        X, y = data
        assert self._estimator(random_state=4).fit(X, y).model_.get_config()["init_seed"] == 4

    def test_training_draws_are_not_the_init_stream(
        self, data: tuple, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Dropout and batch order run on seeds spawned from random_state, not on
        random_state itself -- else the first dropout masks would replay the
        numbers the initial weights were drawn from.
        """
        import hqnn_forge.sklearn as sk

        seen: dict[str, Any] = {}
        real = sk.train_model

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen["draw"] = torch.rand(8)
            seen["generator"] = kwargs["generator"]
            return real(*args, **kwargs)

        monkeypatch.setattr(sk, "train_model", spy)
        X, y = data
        self._estimator(random_state=4).fit(X, y)
        init_stream = torch.rand(8, generator=torch.Generator().manual_seed(4))
        assert not torch.equal(seen["draw"], init_stream)
        assert seen["generator"].initial_seed() != 4
