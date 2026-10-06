"""
tests/test_training.py
======================
Unit tests for hqnn_forge.training.train_model.

Most tests use a plain logistic-regression module so they run in
milliseconds; one test trains a small hybrid classifier end to end.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from hqnn_forge.training import EpochRecord, TrainingHistory, train_model

N_FEATURES = 4


def _separable(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N_FEATURES, generator=g)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).float()
    return x, y


@pytest.fixture
def data() -> tuple[torch.Tensor, ...]:
    return (*_separable(200, 0), *_separable(80, 1))


def _logreg(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(N_FEATURES, 1)


def _running_stats(bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
    """``bn``'s running mean and variance, narrowed from ``Tensor | None``."""
    assert bn.running_mean is not None and bn.running_var is not None
    return bn.running_mean, bn.running_var


class TestTraining:
    def test_loss_decreases_on_separable_data(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, yv = data
        model = _logreg()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.1),
            X,
            y,
            Xv,
            yv,
            max_epochs=30,
            batch_size=32,
            patience=None,
        )
        assert history.n_epochs == 30
        assert history.train_loss[-1] < 0.5 * history.train_loss[0]
        assert history.best_value is not None and history.best_value > 0.9  # validation MCC

    def test_without_validation_runs_all_epochs(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, _, _ = data
        model = _logreg()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.0),
            X,
            y,
            max_epochs=4,
            patience=1,
        )
        assert history.n_epochs == 4 and not history.stopped_early
        assert history.best_epoch is None and history.epochs[0].val_score is None

    def test_generator_makes_runs_reproducible(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, yv = data
        runs = []
        for _ in range(2):
            model = _logreg()
            train_model(
                model,
                nn.BCEWithLogitsLoss(),
                torch.optim.SGD(model.parameters(), lr=0.1),
                X,
                y,
                Xv,
                yv,
                max_epochs=3,
                batch_size=16,
                generator=torch.Generator().manual_seed(7),
            )
            runs.append(model.weight.detach().clone())
        torch.testing.assert_close(runs[0], runs[1], rtol=0, atol=0)

    def test_accepts_logits_of_shape_batch(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, yv = data

        class Flat(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(N_FEATURES, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(x).squeeze(-1)

        model = Flat()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.1),
            X,
            y,
            Xv,
            yv,
            max_epochs=2,
        )
        assert history.n_epochs == 2

    def test_on_epoch_end_receives_every_record(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, yv = data
        model = _logreg()
        seen: list[EpochRecord] = []
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.1),
            X,
            y,
            Xv,
            yv,
            max_epochs=3,
            patience=None,
            on_epoch_end=seen.append,
        )
        assert seen == history.epochs
        assert [r.epoch for r in seen] == [1, 2, 3]
        assert all(r.val_loss is not None and r.val_threshold is not None for r in seen)


class TestEarlyStopping:
    def test_halts_once_the_metric_stops_improving(self, data: tuple[torch.Tensor, ...]) -> None:
        """lr=0: epoch 1 sets the best score and nothing ever improves on it."""
        X, y, Xv, yv = data
        model = _logreg()
        before = model.weight.detach().clone()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.0),
            X,
            y,
            Xv,
            yv,
            max_epochs=50,
            patience=3,
        )
        assert history.stopped_early
        assert history.n_epochs == 1 + 3
        assert history.best_epoch == 1
        # Rolled back to epoch 1, whose weights are the untouched initial ones
        assert history.restored_best
        torch.testing.assert_close(model.weight.detach(), before, rtol=0, atol=0)

    def test_monitored_value_is_the_score_not_the_threshold(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        """Pin the unpacking of find_optimal_threshold's (threshold, score) result."""
        from hqnn_forge.evaluation import find_optimal_threshold

        X, y, Xv, yv = data
        model = _logreg()
        records: list[EpochRecord] = []
        train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.1),
            X,
            y,
            Xv,
            yv,
            max_epochs=5,
            patience=None,
            restore_best=False,
            on_epoch_end=records.append,
        )
        with torch.no_grad():
            probs = torch.sigmoid(model(Xv).squeeze(-1))
        expected = find_optimal_threshold(yv.long(), probs)
        assert records[-1].val_score == pytest.approx(expected.score)
        assert records[-1].val_threshold == pytest.approx(expected.threshold)

    @pytest.mark.parametrize("monitor", ["mcc", "f1", "balanced_accuracy", "val_loss"])
    def test_every_monitor_stops(self, monitor: str, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, yv = data
        model = _logreg()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.0),
            X,
            y,
            Xv,
            yv,
            max_epochs=20,
            patience=2,
            monitor=monitor,
        )
        assert history.stopped_early and history.n_epochs == 3
        assert history.monitor == monitor

    def test_patience_reaching_max_epochs_is_not_early(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        X, y, Xv, yv = data
        model = _logreg()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.0),
            X,
            y,
            Xv,
            yv,
            max_epochs=3,
            patience=2,
        )
        assert history.n_epochs == 3 and not history.stopped_early

    def test_restores_the_best_weights(self, data: tuple[torch.Tensor, ...]) -> None:
        """
        Train well for a few epochs, then sabotage the weights from a callback:
        validation loss gets worse, patience runs out, and the returned model
        must carry the snapshot from the best epoch, not the sabotaged one.
        """
        X, y, Xv, yv = data
        model = _logreg()
        snapshots: dict[int, torch.Tensor] = {}

        def sabotage(record: EpochRecord) -> None:
            if record.epoch >= 5:
                with torch.no_grad():
                    model.weight.mul_(-1.0)
            snapshots[record.epoch] = model.weight.detach().clone()

        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.05),
            X,
            y,
            Xv,
            yv,
            max_epochs=40,
            patience=3,
            monitor="val_loss",
            on_epoch_end=sabotage,
        )
        assert history.stopped_early and history.restored_best
        assert history.best_epoch is not None and history.best_epoch < history.n_epochs
        # The snapshot is taken at validation time, before the callback runs
        best_record = history.epochs[history.best_epoch - 1]
        with torch.no_grad():
            restored_loss = float(nn.BCEWithLogitsLoss()(model(Xv).squeeze(-1), yv))
        assert restored_loss == pytest.approx(best_record.val_loss, rel=1e-6)

    def test_restore_best_false_keeps_last_weights(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, yv = data
        model = _logreg()

        def sabotage(record: EpochRecord) -> None:
            if record.epoch >= 3:
                with torch.no_grad():
                    model.weight.mul_(-1.0)

        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.05),
            X,
            y,
            Xv,
            yv,
            max_epochs=40,
            patience=2,
            monitor="val_loss",
            restore_best=False,
            on_epoch_end=sabotage,
        )
        assert history.stopped_early and not history.restored_best

    def test_min_delta_ignores_small_improvements(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, yv = data
        model = _logreg()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=1e-4),
            X,
            y,
            Xv,
            yv,
            max_epochs=50,
            patience=2,
            monitor="val_loss",
            min_delta=1.0,
        )
        assert history.stopped_early and history.best_epoch == 1


class TestDivergence:
    def test_nan_logits_do_not_abort_a_metric_monitor(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        """
        A diverged model gives NaN probabilities, which find_optimal_threshold
        rejects.  Under the default monitor that used to raise out of the loop
        and lose both the history and the best-epoch snapshot.
        """
        X, y, Xv, yv = data
        model = _logreg()

        def diverge(record: EpochRecord) -> None:
            if record.epoch == 2:
                with torch.no_grad():
                    model.weight.fill_(float("nan"))

        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.05),
            X,
            y,
            Xv,
            yv,
            max_epochs=20,
            patience=2,
            on_epoch_end=diverge,
        )
        assert history.stopped_early and history.n_epochs == 4
        # The NaN epochs never improve, so the best epoch predates the divergence
        assert history.best_epoch is not None and history.best_epoch <= 2
        assert history.best_value is not None and not math.isnan(history.best_value)
        assert all(math.isnan(r.val_score) for r in history.epochs[2:] if r.val_score is not None)
        # and the snapshot survives, so the returned model is usable again
        assert history.restored_best
        with torch.no_grad():
            assert torch.isfinite(model(Xv)).all()


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(monitor="auc"), "unknown monitor"),
            (dict(max_epochs=0), "max_epochs must be"),
            (dict(batch_size=0), "batch_size must be"),
            (dict(patience=0), "patience must be"),
        ],
    )
    def test_bad_arguments(self, kwargs: dict, match: str, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, yv = data
        model = _logreg()
        with pytest.raises(ValueError, match=match):
            train_model(
                model,
                nn.BCEWithLogitsLoss(),
                torch.optim.SGD(model.parameters(), lr=0.1),
                X,
                y,
                Xv,
                yv,
                **kwargs,
            )

    def test_val_pair_must_be_complete(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, Xv, _ = data
        model = _logreg()
        with pytest.raises(ValueError, match="both X_val and y_val"):
            train_model(
                model,
                nn.BCEWithLogitsLoss(),
                torch.optim.SGD(model.parameters(), lr=0.1),
                X,
                y,
                Xv,
            )

    def test_length_mismatch(self, data: tuple[torch.Tensor, ...]) -> None:
        X, y, _, _ = data
        model = _logreg()
        with pytest.raises(ValueError, match="X_train and y_train differ"):
            train_model(
                model,
                nn.BCEWithLogitsLoss(),
                torch.optim.SGD(model.parameters(), lr=0.1),
                X,
                y[:-1],
            )

    def test_single_class_val_split_is_rejected_by_a_metric_monitor(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        """Otherwise every epoch scores alike, so epoch 1 "wins" and the run rolls back to it."""
        X, y, Xv, _ = data
        model = _logreg()
        with pytest.raises(ValueError, match="single class"):
            train_model(
                model,
                nn.BCEWithLogitsLoss(),
                torch.optim.SGD(model.parameters(), lr=0.1),
                X,
                y,
                Xv,
                torch.zeros(Xv.shape[0]),
            )

    def test_single_class_val_split_is_allowed_for_val_loss(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        """val_loss ranks epochs on a single-class split perfectly well."""
        X, y, Xv, _ = data
        model = _logreg()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.1),
            X,
            y,
            Xv,
            torch.zeros(Xv.shape[0]),
            max_epochs=3,
            patience=None,
            monitor="val_loss",
        )
        assert history.n_epochs == 3 and history.best_epoch is not None

    def test_bad_output_shape(self, data: tuple[torch.Tensor, ...]) -> None:
        # (batch, n_classes) is a multiclass output now (#309); a 3-D one is
        # neither kind.
        X, y, _, _ = data
        model = nn.Sequential(nn.Linear(N_FEATURES, 4), nn.Unflatten(1, (2, 2)))
        with pytest.raises(ValueError, match=r"shape \(batch,\) or \(batch, 1\)"):
            train_model(
                model, nn.BCEWithLogitsLoss(), torch.optim.SGD(model.parameters(), lr=0.1), X, y
            )


class TestModes:
    def test_validation_runs_without_dropout_and_restores_entry_modes(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        X, y, Xv, yv = data
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(N_FEATURES, 8), nn.Dropout(0.9), nn.Linear(8, 1))
        records: list[EpochRecord] = []
        train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.0),
            X,
            y,
            Xv,
            yv,
            max_epochs=3,
            patience=None,
            on_epoch_end=records.append,
        )
        # With lr=0 and dropout off in validation, every epoch sees the same val loss
        assert len({r.val_loss for r in records}) == 1
        assert all(m.training for m in model.modules())

    def test_a_frozen_batchnorm_stays_frozen(self, data: tuple[torch.Tensor, ...]) -> None:
        """#174: the caller froze the statistics; training must not move them."""
        X, y, Xv, yv = data
        torch.manual_seed(0)
        first, bn = nn.Linear(N_FEATURES, 8), nn.BatchNorm1d(8)
        model = nn.Sequential(first, bn, nn.Linear(8, 1))
        bn.eval()
        mean, var = (t.clone() for t in _running_stats(bn))
        weight = first.weight.clone()
        train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            y,
            Xv,
            yv,
            max_epochs=2,
            batch_size=32,
            patience=None,
        )
        assert model.training and not bn.training
        assert torch.equal(_running_stats(bn)[0], mean)
        assert torch.equal(_running_stats(bn)[1], var)
        assert not torch.equal(first.weight, weight), "the rest of the model did train"

    def test_a_frozen_body_under_an_eval_root_stays_frozen(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        """model.eval() then the head back to train: only the head may train."""
        X, y, _, _ = data
        torch.manual_seed(0)
        bn = nn.BatchNorm1d(8)
        model = nn.Sequential(nn.Linear(N_FEATURES, 8), bn, nn.Linear(8, 1))
        model.eval()
        model[2].train()
        mean, var = (t.clone() for t in _running_stats(bn))
        train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            y,
            max_epochs=2,
            batch_size=32,
        )
        assert torch.equal(_running_stats(bn)[0], mean)
        assert torch.equal(_running_stats(bn)[1], var)
        assert [m.training for m in model] == [False, False, True]

    def test_every_epoch_trains_in_the_entry_modes_after_a_callback_calls_eval(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        """A callback evaluating the model must not leave later epochs in eval mode."""
        X, y, Xv, yv = data
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(N_FEATURES, 8), nn.BatchNorm1d(8), nn.Dropout(0.5), nn.Linear(8, 1)
        )
        model[1].eval()
        seen: dict[int, set[tuple[bool, bool]]] = {}
        epoch = [1]

        def record(_: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            if torch.is_grad_enabled():  # training batches only, not validation
                seen.setdefault(epoch[0], set()).add((model[1].training, model[2].training))

        def on_epoch_end(_: EpochRecord) -> None:
            model.eval()
            epoch[0] += 1

        model.register_forward_pre_hook(record)
        train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            y,
            Xv,
            yv,
            max_epochs=3,
            batch_size=64,
            patience=None,
            on_epoch_end=on_epoch_end,
        )
        # Batch norm frozen, dropout active, in every epoch
        assert seen == {1: {(False, True)}, 2: {(False, True)}, 3: {(False, True)}}
        assert model.training and not model[1].training and model[2].training

    def test_an_eval_model_trains_in_train_mode_and_is_returned_in_eval(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        """A model fresh from load_checkpoint is in eval mode; dropout must still act."""
        X, y, _, _ = data
        model = nn.Sequential(nn.Linear(N_FEATURES, 8), nn.Dropout(0.5), nn.Linear(8, 1)).eval()
        seen: list[bool] = []
        model[1].register_forward_pre_hook(lambda m, _: seen.append(m.training))
        train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            y,
            max_epochs=1,
            batch_size=64,
        )
        assert seen and all(seen)
        assert not any(m.training for m in model.modules())


class TestBatching:
    """#174: a trailing batch of one sample is merged into the batch before it."""

    @staticmethod
    def _batch_sizes(n: int, batch_size: int) -> list[int]:
        X, y = _separable(n, 0)
        model = _logreg()
        sizes: list[int] = []
        model.register_forward_pre_hook(lambda _, args: sizes.append(args[0].shape[0]))
        train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            y,
            max_epochs=1,
            batch_size=batch_size,
        )
        return sizes

    @pytest.mark.parametrize(
        ("n", "batch_size", "expected"),
        [
            (201, 200, [201]),
            (401, 200, [200, 201]),
            (400, 200, [200, 200]),
            (402, 200, [200, 200, 2]),
            (199, 200, [199]),
            (1, 5, [1]),
            (5, 1, [1, 1, 1, 1, 1]),
        ],
    )
    def test_batch_boundaries(self, n: int, batch_size: int, expected: list[int]) -> None:
        assert self._batch_sizes(n, batch_size) == expected

    def test_batchnorm_model_survives_a_remainder_of_one(self) -> None:
        X, y = _separable(201, 0)
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(N_FEATURES, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            y,
            max_epochs=2,
            batch_size=200,
        )
        assert history.n_epochs == 2


class TestHybridClassifier:
    def test_trains_a_hybrid_classifier_with_focal_loss(
        self, data: tuple[torch.Tensor, ...]
    ) -> None:
        from hqnn_forge.models import HybridBinaryClassifier
        from hqnn_forge.utils import FocalLoss

        X, y, Xv, yv = data
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=N_FEATURES,
            n_qubits=2,
            n_layers=1,
            device_name="default.qubit",
            diff_method="backprop",
        )
        history = train_model(
            model,
            FocalLoss(),
            torch.optim.Adam(model.parameters(), lr=0.05),
            X[:64],
            y[:64],
            Xv[:32],
            yv[:32],
            max_epochs=4,
            batch_size=32,
            patience=None,
        )
        assert isinstance(history, TrainingHistory)
        assert history.train_loss[-1] < history.train_loss[0]
        assert history.best_threshold is not None and 0.0 <= history.best_threshold <= 1.0
