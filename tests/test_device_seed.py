"""
tests/test_device_seed.py
=========================
Reproducible shot sampling through a device seed (#354): the ``seed`` option on
layers and classifiers, and SPSA's common random numbers for shot noise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding._common import resolve_device
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import apply_shots
from hqnn_forge.training import SPSA, train_model
from hqnn_forge.training.spsa import device_generators
from hqnn_forge.utils import load_checkpoint, save_checkpoint

SHOTS = 50


def _model(seed: int | None, *, torch_seed: int = 0, **kwargs) -> HybridBinaryClassifier:
    torch.manual_seed(torch_seed)
    options = {"shots": SHOTS, "device_name": "default.qubit", **kwargs}
    return HybridBinaryClassifier(n_input_features=3, n_qubits=2, n_layers=1, seed=seed, **options)


def _samples(model: torch.nn.Module, n_calls: int = 3) -> list[torch.Tensor]:
    x = torch.linspace(-1, 1, 12).reshape(4, 3)
    with torch.no_grad():
        return [model.predict_proba(x) for _ in range(n_calls)]  # type: ignore[operator]


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(u, v) for u, v in zip(a, b, strict=True))


class TestSeed:
    def test_seeded_models_sample_identically(self) -> None:
        assert _same(_samples(_model(3)), _samples(_model(3)))

    def test_different_seeds_sample_differently(self) -> None:
        assert not _same(_samples(_model(3)), _samples(_model(4)))

    def test_unseeded_models_differ_although_torch_is_seeded(self) -> None:
        # The point of the option: torch.manual_seed fixes the weights (the
        # same torch_seed below) but not the device's sampling.
        a, b = _model(None), _model(None)
        assert torch.equal(a.quantum_layer.qlayer.weights, b.quantum_layer.qlayer.weights)
        assert not _same(_samples(a), _samples(b))

    def test_seed_does_not_touch_the_weights(self) -> None:
        a, b = _model(1), _model(2)
        for (name, p), q in zip(a.state_dict().items(), b.state_dict().values(), strict=True):
            assert torch.equal(p, q), name

    def test_seed_reaches_the_device(self) -> None:
        (rng,) = device_generators(_model(11))
        reference = np.random.default_rng(11)
        assert rng.bit_generator.state == reference.bit_generator.state

    def test_layer_takes_a_seed_and_shows_it(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=2, n_layers=1, shots=SHOTS, seed=5)
        assert layer.seed == 5
        assert ", shots=50, seed=5" in layer.extra_repr()
        assert "seed" not in QuantumEncodingLayer(n_qubits=2, n_layers=1).extra_repr()

    def test_negative_seed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="seed must be None or a non-negative int"):
            resolve_device("default.qubit", 2, seed=-1)

    @pytest.mark.parametrize("bad", [1.5, True, "0"])
    def test_non_integer_seed_is_refused(self, bad: object) -> None:
        with pytest.raises(TypeError, match="seed must be an int or None"):
            resolve_device("default.qubit", 2, seed=bad)  # type: ignore[arg-type]

    def test_numpy_integer_seed_is_converted(self) -> None:
        # As init_seed: a NumPy integer, as scikit-learn tools pass around, is
        # stored as the plain int a weights_only checkpoint load accepts.
        model = _model(np.int64(3))  # type: ignore[arg-type]
        assert type(model.get_config()["seed"]) is int
        assert type(model.quantum_layer.seed) is int
        assert _same(_samples(model), _samples(_model(3)))

    def test_seed_is_honoured_after_a_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pennylane as qml

        from hqnn_forge.encoding import angle_embedding as ae

        real = qml.device

        def device(name: str, *args, **kwargs):
            if name == "lightning.qubit":
                raise ImportError("lightning unavailable in this test")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(ae.qml, "device", device)
        with pytest.warns(RuntimeWarning):
            dev = resolve_device("lightning.qubit", 2, seed=8)
        assert dev.name == "default.qubit"
        assert dev._rng.bit_generator.state == np.random.default_rng(8).bit_generator.state

    def test_config_records_the_seed_and_checkpoints_round_trip(self, tmp_path) -> None:
        model = _model(7)
        assert model.get_config()["seed"] == 7
        save_checkpoint(model, tmp_path / "m.pt")
        assert load_checkpoint(tmp_path / "m.pt").get_config()["seed"] == 7

    def test_seed_is_weight_safe(self, tmp_path) -> None:
        save_checkpoint(_model(7), tmp_path / "m.pt")
        # No allow_architecture_override needed: the seed only seeds sampling.
        assert load_checkpoint(tmp_path / "m.pt", seed=9).get_config()["seed"] == 9

    def test_trajectory_noise_shares_the_layer_s_generator(self) -> None:
        model = _model(
            2, diff_method="parameter-shift", noise_level=0.05, noise_method="trajectories"
        )
        assert model.quantum_layer._training_noise_qnode is not None
        assert len(device_generators(model)) == 1

    def test_apply_shots_samples_from_the_seeded_device(self) -> None:
        def run() -> list[torch.Tensor]:
            model = _model(4, shots=None, diff_method="backprop")
            with apply_shots(model, SHOTS):
                return _samples(model)

        assert _same(run(), run())


class TestSPSACommonShotNoise:
    def _estimates(self, model: HybridBinaryClassifier, sync: bool, n: int = 60) -> np.ndarray:
        x = torch.linspace(-1, 1, 24).reshape(8, 3)
        y = torch.tensor([0.0, 1.0] * 4)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            return loss_fn(model(x).squeeze(-1), y)

        opt = SPSA(model.parameters(), perturbation=0.05, model=model if sync else None)
        rows = []
        for _ in range(n):
            # The same direction every time, so only the shot noise varies.
            opt.generator.manual_seed(0)
            rows.append(torch.cat([g.flatten() for g in opt.gradient_estimate(closure)]).numpy())
        return np.stack(rows)

    @pytest.mark.slow
    def test_synchronised_shots_cut_the_estimate_s_variance(self) -> None:
        model = _model(0, shots=200)
        independent = self._estimates(model, sync=False).var(axis=0).sum()
        synchronised = self._estimates(model, sync=True).var(axis=0).sum()
        # Measured: a 4.3- to 4.9-fold drop over device seeds 0 to 2.  The bound
        # leaves room for the sampling error of 60 estimates.
        assert synchronised < independent / 2.5, (synchronised, independent)

    def test_generators_advance_after_a_synchronised_step(self) -> None:
        # Restoring the state before the second evaluation must not freeze the
        # stream: the next step draws new samples.
        model = _model(0)
        (rng,) = device_generators(model)
        before = rng.bit_generator.state
        self._estimates(model, sync=True, n=1)
        assert rng.bit_generator.state != before

    def test_train_model_gives_spsa_the_model(self) -> None:
        # Without model=, train_model passes the model it trains, so the
        # documented train_model(model, loss, SPSA(model.parameters())) call
        # gets the common shot noise too.
        model = _model(0)
        x = torch.linspace(-1, 1, 24).reshape(8, 3)
        opt = SPSA(model.parameters(), lr=0.2)
        train_model(
            model, torch.nn.BCEWithLogitsLoss(), opt, x, (x[:, 0] > 0).float(), max_epochs=1
        )
        assert opt.model is model

    def test_two_seeded_spsa_fits_are_identical(self) -> None:
        def fit(seed: int | None) -> dict[str, torch.Tensor]:
            model = _model(seed)
            x = torch.linspace(-1, 1, 48).reshape(16, 3)
            y = (x[:, 0] > 0).float()
            train_model(
                model,
                torch.nn.BCEWithLogitsLoss(),
                SPSA(model.parameters(), lr=0.2, model=model),
                x,
                y,
                max_epochs=2,
                batch_size=8,
                generator=torch.Generator().manual_seed(0),
            )
            return model.state_dict()

        a, b = fit(21), fit(21)
        assert all(torch.equal(a[k], b[k]) for k in a)
        c, d = fit(None), fit(None)
        assert not all(torch.equal(c[k], d[k]) for k in c)
