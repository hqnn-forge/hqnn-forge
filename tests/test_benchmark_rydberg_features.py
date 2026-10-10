"""
tests/test_benchmark_rydberg_features.py
========================================
``examples/benchmark_rydberg_features.py`` (#501): the runner that compares
Rydberg features with and without interactions against classical baselines.

What the numbers are checked against:

* the model page's closed forms, typed in: the spacing table of
  ``docs/rydberg-model.md``, the single-atom feature ``f(√(1 + 3 σ(x)²))``
  and the damped Rabi oscillation after the pulse;
* a single-atom map, for arm B under dephasing;
* a fold rebuilt by hand from the library's own functions
  (``stratified_kfold``, ``PCANormalizer``, ``smote``, ``train_model``), not
  through the script, for the recorded indices, seeds, scores and separation
  measures of a ``--quick`` run;
* a circuit written out gate by gate in PennyLane, for arm D;
* hand computations, for the positive control's labels, the parameter counts
  and the JSON mapping.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch
import torch.nn as nn

from hqnn_forge.benchmark import _sample_configs, fingerprint
from hqnn_forge.data import load_iranian_churn
from hqnn_forge.diagnostics import separation_measures
from hqnn_forge.evaluation import matthews_corrcoef, pr_auc
from hqnn_forge.experiment import environment
from hqnn_forge.models import ClassicalBaseline, HybridBinaryClassifier, LinearClassifier
from hqnn_forge.preprocessing import PCANormalizer, smote, stratified_kfold
from hqnn_forge.rydberg import DEFAULT_C6, AtomRegister, PulseEncoding, RydbergFeatureMap
from hqnn_forge.training import train_model

ROOT = Path(__file__).resolve().parent.parent
OMEGA = 4.0 * math.pi  # the page's reference Rabi frequency, rad/µs


def _script() -> ModuleType:
    path = ROOT / "examples" / "benchmark_rydberg_features.py"
    spec = importlib.util.spec_from_file_location("benchmark_rydberg_features", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


bench = _script()
QUICK = bench.QUICK
#: Arms that need no PennyLane circuit, for runs that only check the plumbing.
FAST = ("A", "B", "C")


def _replace(settings: Any, **changes: Any) -> Any:
    import dataclasses

    return dataclasses.replace(settings, **changes)


# ---------------------------------------------------------------------------
# A fold rebuilt by hand, from the library and not from the script
# ---------------------------------------------------------------------------


def _draw(root: np.random.SeedSequence, n: int) -> list[int]:
    return [int(child.generate_state(1)[0]) for child in root.spawn(n)]


def _fold_by_hand(X: np.ndarray, y: np.ndarray, settings: Any, dataset_index: int) -> Any:
    """Fold 0 of ``settings``: seeds, row indices, inputs and the oversampled training rows."""
    root = np.random.SeedSequence([settings.random_state, dataset_index])
    split_root, *fold_roots = root.spawn(settings.n_splits + 1)
    split_seed = int(split_root.generate_state(1)[0])
    inner_seed, smote_seed, init_seed, batch_seed, tune_seed = _draw(fold_roots[0], 5)
    train_part, test_idx = stratified_kfold(y, settings.n_splits, random_state=split_seed)[0]
    tr, va = stratified_kfold(y[train_part], settings.validation_folds, random_state=inner_seed)[0]
    train_idx, val_idx = train_part[tr], train_part[va]

    mean, std = X[train_part].mean(0), X[train_part].std(0)
    std[std == 0] = 1.0
    scaled = (X - mean) / std
    pca = PCANormalizer(n_components=settings.n_atoms, scale_to_pi=True).fit(scaled[train_part])
    inputs = pca.transform(scaled).double().numpy()
    oversampled = smote(inputs[train_idx], y[train_idx], random_state=smote_seed)
    return SimpleNamespace(
        seeds=dict(
            split_seed=split_seed,
            inner_seed=inner_seed,
            smote_seed=smote_seed,
            init_seed=init_seed,
            batch_seed=batch_seed,
            tune_seed=tune_seed,
        ),
        train_part=train_part,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        inputs=inputs,
        train_inputs=oversampled.X,
        train_labels=np.asarray(oversampled.y, dtype=np.int64),
        n_synthetic=len(oversampled.sources),
    )


def _map_by_hand(n_atoms: int, v: float, gamma_ratio: float, n_steps: int, **kwargs: Any) -> Any:
    """The page's map at ``(V/Ω, γ/Ω)``: ``a = (C6/Ω)^(1/6) (V/Ω)^(−1/6)``, ``ΩT = π``."""
    a = (DEFAULT_C6 / OMEGA) ** (1 / 6) * v ** (-1 / 6)
    return RydbergFeatureMap(
        AtomRegister.chain(n_atoms, a),
        PulseEncoding(n_atoms, omega=OMEGA),
        c6=DEFAULT_C6,
        gamma=gamma_ratio * OMEGA,
        n_steps=n_steps if gamma_ratio > 0 else None,
        **kwargs,
    )


def _single_atom_feature(x: np.ndarray) -> np.ndarray:
    """``f(√(1 + 3 σ(x)²))`` with ``f(s) = sin²(π s/2)/s²``: the page's closed form."""
    s = np.sqrt(1.0 + 3.0 / (1.0 + np.exp(-x)) ** 2)
    return np.asarray(np.sin(math.pi * s / 2) ** 2 / s**2)


@pytest.fixture(scope="module")
def quick(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One ``--quick`` run through the command line, and its fold rebuilt by hand."""
    directory = tmp_path_factory.mktemp("quick")
    out = directory / "quick.jsonl"
    records = bench.main(["--quick", "--out", str(out), "--cache-dir", str(directory / "cache")])
    X, y, data_seed = bench.load_dataset("positive-control", QUICK)
    return SimpleNamespace(
        records=records,
        lines=bench.read_records(out),
        text=out.read_text(encoding="utf-8"),
        cache=directory / "cache",
        X=X,
        y=y,
        data_seed=data_seed,
        fold=_fold_by_hand(X, y, QUICK, 0),
    )


def _line(run: Any, arm: str, head: str) -> dict[str, Any]:
    (found,) = [r for r in run.lines if r["arm"] == arm and r["head"] == head]
    return dict(found)


# ---------------------------------------------------------------------------
# The grid: spacing, dephasing rate, pulse
# ---------------------------------------------------------------------------


class TestGrid:
    @pytest.mark.parametrize(
        ("v", "a"),
        [(0.1, 12.76), (0.3, 10.62), (1.0, 8.69), (3.0, 7.24), (10.0, 5.92), (30.0, 4.93)],
    )
    def test_spacing_is_the_table_of_the_model_page(self, v: float, a: float) -> None:
        assert bench.spacing(v, OMEGA, DEFAULT_C6) == pytest.approx(a, abs=0.006)

    @pytest.mark.parametrize("v", [0.1, 1.0, 7.5, 100.0])
    def test_spacing_gives_the_nearest_neighbour_interaction_asked_for(self, v: float) -> None:
        """``C6/a⁶ = (V/Ω) Ω``, and the next-nearest coupling is ``1/64`` of it."""
        settings = bench.Settings(n_atoms=4)
        fmap = bench.feature_map(settings, v, 0.0, interactions=True)
        a = bench.spacing(v, settings.omega, settings.c6)
        assert a == pytest.approx((DEFAULT_C6 / OMEGA) ** (1 / 6) * v ** (-1 / 6), rel=1e-13)
        assert DEFAULT_C6 / a**6 / OMEGA == pytest.approx(v, rel=1e-12)
        couplings = fmap.register.interaction_matrix(DEFAULT_C6)
        assert float(couplings[0, 1]) / OMEGA == pytest.approx(v, rel=1e-12)
        assert float(couplings[0, 2]) / OMEGA == pytest.approx(v / 64, rel=1e-12)
        # an open chain in the order of the inputs: atom i at (i a, 0)
        expected = np.array([[i * a, 0.0] for i in range(4)])
        np.testing.assert_allclose(fmap.register.positions.numpy(), expected, rtol=1e-13)

    def test_spacing_refuses_a_non_positive_ratio(self) -> None:
        with pytest.raises(ValueError, match="v_over_omega must be > 0"):
            bench.spacing(0.0, OMEGA, DEFAULT_C6)

    def test_the_grid_is_the_product_and_holds_gamma_zero_for_every_spacing(self) -> None:
        settings = bench.Settings()
        points = bench.grid(settings)
        assert len(points) == len(settings.v_over_omega) * len(settings.gamma_over_omega) == 28
        assert set(points) == {
            (v, g) for v in settings.v_over_omega for g in settings.gamma_over_omega
        }
        assert {v for v, g in points if g == 0.0} == set(settings.v_over_omega)
        assert bench.grid(QUICK) == [(1.0, 0.1)]

    def test_the_pulse_area_is_the_same_at_every_grid_point(self) -> None:
        """``T = ΩT/Ω``: 0.25 µs at the reference Ω, whatever ``V`` and γ."""
        settings = bench.Settings(n_atoms=2)
        assert settings.evolution_time == pytest.approx(0.25, rel=1e-15)
        for v, gamma in bench.grid(settings):
            for interactions in (True, False):
                fmap = bench.feature_map(settings, v, gamma, interactions=interactions)
                assert fmap.evolution_time * fmap.encoding.omega == pytest.approx(math.pi)
                assert fmap.encoding.delta_max == pytest.approx(math.sqrt(3) * OMEGA)
        longer = bench.Settings(n_atoms=2, omega_t=2 * math.pi)
        assert bench.feature_map(longer, 1.0, 0.0, interactions=True).evolution_time == (
            pytest.approx(0.5)
        )

    def test_gamma_is_in_units_of_omega_and_zero_needs_no_steps(self) -> None:
        settings = bench.Settings(n_atoms=2, n_steps=123)
        noiseless = bench.feature_map(settings, 1.0, 0.0, interactions=True)
        assert noiseless.gamma == 0.0
        assert noiseless.n_steps is None
        noisy = bench.feature_map(settings, 1.0, 0.25, interactions=True)
        assert noisy.gamma == pytest.approx(math.pi)  # 0.25 · 4π
        assert noisy.n_steps == 123

    @pytest.mark.parametrize("ratio", [0.1, 1.0])
    def test_dephasing_sweep_against_the_damped_rabi_oscillation(self, ratio: float) -> None:
        """
        A resonant atom (``x → −∞``, so ``Δ → 0``) after the π pulse, from the
        page: ``⟨n⟩ = ½ − ½ e^(−γT/4) [cos λT + γ/(4λ) sin λT]``,
        ``λ = √(Ω² − γ²/16)``.  At ``γ = 0`` the same atom is fully excited.
        """
        settings = bench.Settings(n_atoms=2, n_steps=400)
        x = torch.full((1, 2), -40.0, dtype=torch.float64)
        gamma, t = ratio * OMEGA, math.pi / OMEGA
        lam = math.sqrt(OMEGA**2 - gamma**2 / 16)
        expected = 0.5 - 0.5 * math.exp(-gamma * t / 4) * (
            math.cos(lam * t) + gamma / (4 * lam) * math.sin(lam * t)
        )
        noisy = bench.feature_map(settings, 1.0, ratio, interactions=False).transform(x)
        assert noisy[0].tolist() == pytest.approx([expected, expected], abs=1e-6)
        assert expected < 1 - 0.03  # the dephasing is visible, not a rounding effect
        noiseless = bench.feature_map(settings, 1.0, 0.0, interactions=False).transform(x)
        assert noiseless[0].tolist() == pytest.approx([1.0, 1.0], abs=1e-12)

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            (dict(v_over_omega=(1.0, 0.0)), "v_over_omega"),
            (dict(v_over_omega=()), "v_over_omega"),
            (dict(gamma_over_omega=(-0.1,)), "gamma_over_omega"),
            (dict(arms=("A", "E")), "arms"),
            (dict(arms=()), "arms"),
            (dict(loss="hinge"), "loss"),
            (dict(n_splits=1), "n_splits"),
            (dict(max_folds=0), "max_folds"),
            (dict(n_trials=-1), "n_trials"),
            (dict(inner_folds=1), "inner_folds"),
            (dict(search_space={"hidden": (4, 8)}), "cannot be tuned"),
            (dict(search_space={"lr": ()}), "no values"),
        ],
    )
    def test_invalid_settings(self, changes: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            bench.Settings(**changes)


# ---------------------------------------------------------------------------
# Arms A and B: the features
# ---------------------------------------------------------------------------


class TestRydbergArms:
    X = np.random.default_rng(3).uniform(-math.pi, math.pi, size=(9, 4))

    def _features(self, settings: Any, arm: str, v: float, gamma: float) -> np.ndarray:
        (rep,) = [
            r
            for r in bench.representations(_replace(settings, arms=(arm,)))
            if (r.v_over_omega, r.gamma_over_omega) == (v, gamma)
        ]
        assert rep.compute is not None
        return np.asarray(rep.compute(self.X))

    def test_arm_b_is_the_closed_form_single_atom_feature(self) -> None:
        """Feature ``i`` is ``f(√(1 + 3 σ(x_i)²))`` of input ``i`` alone, at every spacing."""
        settings = bench.Settings(v_over_omega=(0.1, 1.0, 30.0), gamma_over_omega=(0.0,))
        for v in settings.v_over_omega:
            features = self._features(settings, "B", v, 0.0)
            np.testing.assert_allclose(features, _single_atom_feature(self.X), rtol=0, atol=1e-12)

    def test_arm_b_under_dephasing_is_a_single_atom_evolved_alone(self) -> None:
        """Column ``i`` equals a one-atom map (same pulse, γ and steps) applied to input ``i``."""
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.5,), n_steps=60)
        features = self._features(settings, "B", 1.0, 0.5)
        alone = RydbergFeatureMap(
            AtomRegister.chain(1, 1.0),
            PulseEncoding(1, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.5 * OMEGA,
            n_steps=60,
        )
        for i in range(4):
            column = alone.transform(self.X[:, [i]]).numpy()[:, 0]
            np.testing.assert_allclose(features[:, i], column, rtol=0, atol=1e-12)
        # and it differs from the noiseless closed form by far more than that
        assert np.abs(features - _single_atom_feature(self.X)).max() > 0.05

    def test_arm_a_is_the_interacting_map_and_mixes_neighbouring_inputs(self) -> None:
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.0,))
        features = self._features(settings, "A", 1.0, 0.0)
        by_hand = _map_by_hand(4, 1.0, 0.0, 1).transform(self.X).numpy()
        np.testing.assert_allclose(features, by_hand, rtol=0, atol=1e-12)
        control = self._features(settings, "B", 1.0, 0.0)
        assert np.abs(features - control).max() > 0.1
        # Input 1 moves the feature of atom 0 in A and not in B.
        moved = self.X.copy()
        moved[:, 1] += 0.7
        (rep_a,) = bench.representations(_replace(settings, arms=("A",)))
        (rep_b,) = bench.representations(_replace(settings, arms=("B",)))
        assert np.abs(rep_a.compute(moved)[:, 0] - features[:, 0]).max() > 1e-2
        assert np.abs(rep_b.compute(moved)[:, 0] - control[:, 0]).max() < 1e-12

    def test_the_two_arms_differ_in_the_interactions_flag_only(self) -> None:
        settings = bench.Settings(v_over_omega=(3.0,), gamma_over_omega=(1.0,))
        rep_a, rep_b = bench.representations(_replace(settings, arms=("A", "B")))
        assert (rep_a.arm, rep_b.arm) == ("A", "B")
        config_a, config_b = rep_a.config["config"], rep_b.config["config"]
        assert config_a["interactions"] is True
        assert config_b["interactions"] is False
        assert {k: v for k, v in config_a.items() if k != "interactions"} == {
            k: v for k, v in config_b.items() if k != "interactions"
        }
        assert config_a == _map_by_hand(4, 3.0, 1.0, 200).get_config()
        assert rep_a.heads == rep_b.heads == ("linear", "mlp")

    def test_shots_are_drawn_with_the_recorded_seed(self) -> None:
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.0,), shots=64)
        features = self._features(settings, "A", 1.0, 0.0)
        shot_seed = bench.run_seeds(settings.random_state)["shot_seed"]
        by_hand = _map_by_hand(4, 1.0, 0.0, 1).transform(
            self.X, shots=64, generator=torch.Generator().manual_seed(shot_seed)
        )
        np.testing.assert_array_equal(features, by_hand.numpy())
        # multiples of 1/64, and within the binomial error of the exact features
        np.testing.assert_allclose(features * 64, np.round(features * 64), atol=1e-9)
        exact = _map_by_hand(4, 1.0, 0.0, 1).transform(self.X).numpy()
        assert 0 < np.abs(features - exact).max() < 5 * 0.5 / math.sqrt(64)


# ---------------------------------------------------------------------------
# Arm D: the frozen circuit
# ---------------------------------------------------------------------------


class TestCircuitArm:
    def test_weights_are_uniform_draws_from_the_circuit_seed_and_frozen(self) -> None:
        settings = bench.Settings(n_atoms=3, circuit_layers=2)
        layer = bench.frozen_circuit(settings, 11)
        weights = layer.qlayer.weights
        draw = torch.rand(
            (2, 3, 3), generator=torch.Generator().manual_seed(11), dtype=torch.float64
        )
        expected = ((2 * draw - 1) * math.pi).to(weights.dtype)
        assert torch.equal(weights.detach(), expected)
        assert float(weights.abs().max()) <= math.pi
        assert all(not p.requires_grad for p in layer.parameters())
        assert not torch.equal(weights, bench.frozen_circuit(settings, 12).qlayer.weights)

    def test_features_are_the_z_expectations_of_the_circuit_written_out(self) -> None:
        """
        ``RX(x_i)`` on qubit ``i``, then per layer a CNOT ring ``i → i+1`` and
        ``Rot`` on every qubit, then ``⟨Z_i⟩``: feature ``i`` is qubit ``i``.
        """
        settings = bench.Settings(n_atoms=3, circuit_layers=2)
        (rep,) = [
            r for r in bench.representations(_replace(settings, arms=("D",))) if r.arm == "D"
        ]
        seed = bench.run_seeds(settings.random_state)["circuit_seed"]
        weights = bench.frozen_circuit(settings, seed).qlayer.weights.detach().double().numpy()
        device = qml.device("default.qubit", wires=3)

        @qml.qnode(device)
        def circuit(x: np.ndarray) -> Any:
            for i in range(3):
                qml.RX(x[i], wires=i)
            for layer in range(2):
                for i in range(3):
                    qml.CNOT(wires=[i, (i + 1) % 3])
                for i in range(3):
                    qml.Rot(*weights[layer, i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        X = np.random.default_rng(5).uniform(-math.pi, math.pi, size=(6, 3))
        by_hand = np.array([[float(v) for v in circuit(row)] for row in X])
        features = rep.compute(X)
        np.testing.assert_allclose(features, by_hand, rtol=0, atol=1e-9)
        assert np.abs(features).max() <= 1.0
        assert rep.n_frozen_parameters == 2 * 3 * 3
        assert rep.config["config"]["circuit_seed"] == seed

    def test_the_hybrid_row_is_a_trained_circuit_on_the_inputs(self) -> None:
        settings = bench.Settings(n_atoms=3, circuit_layers=2)
        reps = bench.representations(_replace(settings, arms=("D",)))
        assert [(r.arm, r.kind, r.heads) for r in reps] == [
            ("D", "frozen-circuit", ("linear", "mlp")),
            ("D-hybrid", "inputs", ("hybrid",)),
        ]
        assert reps[1].compute is None
        assert reps[1].n_frozen_parameters == 0


# ---------------------------------------------------------------------------
# Heads and parameter counts
# ---------------------------------------------------------------------------


class TestModels:
    @pytest.mark.parametrize(("d", "hidden"), [(4, 8), (6, 5), (10, 3)])
    def test_parameter_counts_by_hand(self, d: int, hidden: int) -> None:
        settings = bench.Settings(hidden=hidden, circuit_layers=2)
        linear = bench.build_model("linear", d, settings, 0)
        mlp = bench.build_model("mlp", d, settings, 0)
        assert isinstance(linear, LinearClassifier)
        assert isinstance(mlp, ClassicalBaseline)
        assert linear.count_parameters() == d + 1
        assert mlp.count_parameters() == hidden * (d + 2) + 1 == d * hidden + hidden + hidden + 1
        assert mlp.get_config()["hidden_dims"] == [hidden]

    def test_hybrid_parameter_count_by_hand(self) -> None:
        model = bench.build_model("hybrid", 4, bench.Settings(circuit_layers=2), 0)
        assert isinstance(model, HybridBinaryClassifier)
        assert model.count_parameters() == 3 * 2 * 4 + 4 + 1
        config = model.get_config()
        assert config["use_classical_encoder"] is False
        assert (config["n_input_features"], config["n_qubits"], config["n_layers"]) == (4, 4, 2)

    def test_the_same_seed_gives_every_arm_the_same_initial_head(self) -> None:
        a = bench.build_model("mlp", 4, bench.Settings(), 17)
        b = bench.build_model("mlp", 4, bench.Settings(), 17)
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            assert torch.equal(p, q)

    def test_unknown_head(self) -> None:
        with pytest.raises(ValueError, match="unknown head"):
            bench.build_model("svm", 4, bench.Settings(), 0)

    def test_candidates_are_drawn_as_the_benchmark_tuning_draws_them(self) -> None:
        space = {"lr": (0.003, 0.01, 0.03), "batch_size": (32, 128)}
        for seed in (0, 1, 99):
            assert bench.sample_configs(space, 4, seed) == _sample_configs(space, 4, seed)
            assert len(bench.sample_configs(space, 4, seed)) == 4
        assert len(bench.sample_configs(space, 10, 0)) == 6  # the whole grid, no more


# ---------------------------------------------------------------------------
# Data: the positive control and a UCI loader
# ---------------------------------------------------------------------------


class TestPositiveControl:
    def test_labels_by_hand(self) -> None:
        latent = [
            [1.0, 2.0, 3.0, 4.0],  # 2 + 6 + 12 = 20
            [1.0, -2.0, 3.0, -4.0],  # −2 − 6 − 12 = −20
            [1.0, 1.0, -3.0, 0.5],  # 1 − 3 − 1.5 = −3.5
            [-1.0, -1.0, 0.1, 5.0],  # 1 − 0.1 + 0.5 = 1.4
            [5.0, 0.1, -1.0, 9.0],  # 0.5 − 0.1 − 9 = −8.6: not the first bond alone
            [1.0, 1.0, -1.0, 1.0],  # 1 − 1 − 1 = −1
        ]
        assert bench.product_labels(latent).tolist() == [1, 0, 0, 1, 0, 0]
        assert bench.product_labels([[2.0, 3.0], [2.0, -3.0]]).tolist() == [1, 0]

    def test_only_neighbours_enter(self) -> None:
        """Atoms 0 and 2 are not neighbours: their product alone decides nothing."""
        assert bench.product_labels([[1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]).tolist() == [0, 0]
        with pytest.raises(ValueError, match="N >= 2"):
            bench.product_labels([[1.0], [2.0]])

    def test_flipping_one_parity_flips_every_label(self) -> None:
        """The symmetry behind ``P(y = 1 | z_i) = ½``: the even latents stay as they are."""
        control = bench.positive_control(5, 400, seed=2)
        flipped = control.latent * np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        np.testing.assert_array_equal(bench.product_labels(flipped), 1 - control.y)

    def test_columns_are_noisy_copies_in_blocks_of_halving_size(self) -> None:
        control = bench.positive_control(4, 3000, seed=0, noise=0.1)
        assert control.X.shape == (3000, 15)  # 8 + 4 + 2 + 1
        assert control.X.dtype == np.float64
        assert control.y.dtype == np.int64
        np.testing.assert_array_equal(control.y, bench.product_labels(control.latent))
        owner = [0] * 8 + [1] * 4 + [2] * 2 + [3]
        residual = control.X - control.latent[:, owner]
        assert residual.std(axis=0) == pytest.approx(np.full(15, 0.1), rel=0.1)
        assert abs(control.y.mean() - 0.5) < 0.05
        again = bench.positive_control(4, 3000, seed=0, noise=0.1)
        np.testing.assert_array_equal(again.X, control.X)

    def test_leading_eigenvalues_are_the_closed_form(self) -> None:
        """``λ_i = 1 + (m_i − 1)/(1 + noise²)`` for the standardised columns."""
        noise = 0.3
        control = bench.positive_control(4, 20000, seed=1, noise=noise)
        scaled = (control.X - control.X.mean(0)) / control.X.std(0)
        pca = PCANormalizer(n_components=4).fit(scaled)
        rho = 1 / (1 + noise**2)
        expected = [1 + (m - 1) * rho for m in (8, 4, 2, 1)]
        assert pca.explained_variance_.tolist() == pytest.approx(expected, rel=0.03)

    @pytest.mark.parametrize("n_atoms", [4, 6])
    def test_latent_i_ends_up_on_atom_i_with_a_positive_sign(self, n_atoms: int) -> None:
        """Input column ``i`` of the pipeline follows latent ``i`` and no other."""
        control = bench.positive_control(n_atoms, 600, seed=0)
        rows = np.arange(600)
        inputs = bench.preprocess(control.X, rows, n_atoms)
        correlation = np.corrcoef(inputs.T, control.latent.T)[:n_atoms, n_atoms:]
        assert np.diag(correlation).min() > 0.93  # π tanh of a normal variable: about 0.96
        off_diagonal = correlation - np.diag(np.diag(correlation))
        assert np.abs(off_diagonal).max() < 0.15
        # increasing: the ranks agree almost perfectly
        for i in range(n_atoms):
            order = np.argsort(control.latent[:, i])
            ranked = np.argsort(np.argsort(inputs[order, i]))
            assert np.corrcoef(ranked, np.arange(600))[0, 1] > 0.98

    def test_arm_a_separates_the_classes_and_arm_b_cannot(self) -> None:
        """
        The Fisher ratio of arm B's features, and of the inputs, sits at its
        no-information level ``(M − 2)/(M − d − 3) · d M/(n0 n1)`` (the class
        means coincide); with interactions at ``V = Ω`` it is far above it.
        """
        m, d = 600, 4
        control = bench.positive_control(d, m, seed=0)
        inputs = bench.preprocess(control.X, np.arange(m), d)
        n1 = int(control.y.sum())
        null = (m - 2) / (m - d - 3) * d * m / (n1 * (m - n1))
        fisher = {
            "inputs": separation_measures(inputs, control.y).fisher_ratio,
            "A": separation_measures(
                _map_by_hand(d, 1.0, 0.0, 1).transform(inputs), control.y
            ).fisher_ratio,
            "B": separation_measures(
                _map_by_hand(d, 1.0, 0.0, 1, interactions=False).transform(inputs), control.y
            ).fisher_ratio,
        }
        assert fisher["B"] < 3 * null
        assert fisher["inputs"] < 3 * null
        assert fisher["A"] > 20 * null
        assert fisher["A"] > 20 * fisher["B"]

    def test_load_dataset_draws_the_control_from_the_run_seed(self) -> None:
        settings = _replace(QUICK, random_state=5)
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        assert data_seed == bench.run_seeds(5)["data_seed"]
        by_hand = bench.positive_control(settings.n_atoms, settings.n_samples, data_seed)
        np.testing.assert_array_equal(X, by_hand.X)
        np.testing.assert_array_equal(y, by_hand.y)
        with pytest.raises(ValueError, match="unknown dataset"):
            bench.load_dataset("mnist", settings)


CHURN_HEADER = [
    "Call  Failure",
    "Complains",
    "Subscription  Length",
    "Charge  Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Status",
    "Age",
    "Customer Value",
    "Churn",
]


def _write_churn(directory: Path, n: int = 96, n_pos: int = 24) -> Path:
    """A small file in the format of UCI's ``Customer Churn.csv``; the dataset is not fetched."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 500, (n, 13)).astype(float)
    labels = np.zeros(n, dtype=int)
    labels[:n_pos] = 1
    data[labels == 1, :3] += 150.0  # churners differ, so there is something to learn
    lines = [",".join(CHURN_HEADER)]
    lines += [",".join(f"{v:g}" for v in row) + f",{label}" for row, label in zip(data, labels)]
    path = directory / "Customer Churn.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestUciData:
    def test_a_uci_loader_runs_through_the_command_line(self, tmp_path: Path) -> None:
        _write_churn(tmp_path)
        out = tmp_path / "churn.jsonl"
        arguments = ["--dataset", "iranian-churn", "--data-path", str(tmp_path), "--no-cache"]
        arguments += ["--arms", "C", "--folds", "3", "--max-folds", "1", "--trials", "0"]
        arguments += ["--epochs", "3", "--out", str(out)]
        records = bench.main(arguments)
        assert [(r["arm"], r["head"]) for r in records] == [("C", "linear"), ("C", "mlp")]
        assert not (tmp_path / "rydberg_features_cache").exists()

        data = load_iranian_churn(tmp_path)
        settings = bench.Settings(n_splits=3)
        fold = _fold_by_hand(data.X, data.y, settings, bench.DATASETS.index("iranian-churn"))
        for record in bench.read_records(out):
            assert record["dataset"] == "iranian-churn"
            assert record["data_sha256"] == fingerprint(data.X, data.y)
            assert (record["n_samples"], record["n_positives"]) == (96, 24)
            assert record["n_raw_features"] == 13
            assert record["seeds"]["data_seed"] is None
            assert record["test_idx"] == np.sort(fold.test_idx).tolist()
            assert record["tuning"] is None
            assert record["hyperparameters"]["max_epochs"] == 3
            # The 13 columns reduced to 4 bounded inputs, fitted on the training part.
            by_hand = separation_measures(
                fold.inputs[fold.test_idx], data.y[fold.test_idx]
            ).to_dict()
            assert record["separation"]["test"] == pytest.approx(by_hand, rel=1e-9)

    def test_too_few_columns_for_the_atoms(self) -> None:
        X = np.random.default_rng(0).normal(size=(40, 3))
        y = np.array([0, 1] * 20)
        records = bench.run_dataset(
            "iranian-churn", X, y, bench.Settings(), bench.FeatureCache(None)
        )
        with pytest.raises(ValueError, match="cannot be reduced to n_atoms=4"):
            next(records)


# ---------------------------------------------------------------------------
# Inputs: fitted on the training rows, oversampled once
# ---------------------------------------------------------------------------


class TestInputs:
    def _data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(4)
        X = rng.normal(size=(120, 7)) * np.array([1.0, 50.0, 0.01, 3.0, 1.0, 1.0, 9.0])
        X[:, 5] = 2.5  # a constant column, as in the bankruptcy data
        y = (rng.random(120) < 0.25).astype(np.int64)
        return X, y

    def test_inputs_are_bounded_and_fitted_on_the_given_rows_only(self) -> None:
        X, _ = self._data()
        fit_rows = np.arange(0, 90)
        inputs = bench.preprocess(X, fit_rows, 4)
        assert inputs.shape == (120, 4)
        assert inputs.dtype == np.float64
        assert np.abs(inputs).max() <= float(np.float32(math.pi))
        # Changing rows outside the fit changes no other row's inputs.
        changed = X.copy()
        changed[90:] = changed[90:] * 3.0 + 1.0
        again = bench.preprocess(changed, fit_rows, 4)
        np.testing.assert_array_equal(again[:90], inputs[:90])
        assert np.abs(again[90:] - inputs[90:]).max() > 0.1
        # ... and a change inside it does.
        changed = X.copy()
        changed[:10] += 1.0
        assert np.abs(bench.preprocess(changed, fit_rows, 4)[90:] - inputs[90:]).max() > 1e-3

    def test_inputs_are_standardised_components_squashed_to_pi(self) -> None:
        """By hand: standardise, project on the leading eigenvectors, ``π tanh(· / std)``."""
        X, _ = self._data()
        rows = np.arange(0, 90)
        mean, std = X[rows].mean(0), X[rows].std(0)
        std[std == 0] = 1.0
        scaled = (X - mean) / std
        centred = scaled[rows] - scaled[rows].mean(0)
        values, vectors = np.linalg.eigh(np.cov(centred, rowvar=False))
        leading = vectors[:, np.argsort(values)[::-1][:4]]
        leading = leading * np.sign(leading[np.abs(leading).argmax(0), np.arange(4)])
        projected = (scaled - scaled[rows].mean(0)) @ leading
        by_hand = math.pi * np.tanh(projected / projected[rows].std(0, ddof=1))
        np.testing.assert_allclose(bench.preprocess(X, rows, 4), by_hand, rtol=0, atol=2e-6)

    def test_split_blocks_and_synthetic_rows(self) -> None:
        X, y = self._data()
        train, val, test = np.arange(0, 70), np.arange(70, 90), np.arange(90, 120)
        fit_rows = np.arange(0, 90)
        split = bench.make_split(X, y, fit_rows, train, val, test, n_components=4, smote_seed=3)
        inputs = bench.preprocess(X, fit_rows, 4)
        n_pos = int(y[train].sum())
        n_synthetic = (70 - n_pos) - n_pos  # SMOTE balances the training rows
        assert (split.n_train, split.n_synthetic) == (70 + n_synthetic, n_synthetic)
        assert (split.n_val, split.n_test) == (20, 30)
        np.testing.assert_array_equal(split.inputs[:70], inputs[train])
        np.testing.assert_array_equal(split.labels[:70], y[train])
        np.testing.assert_array_equal(split.inputs[split.val], inputs[val])
        np.testing.assert_array_equal(split.labels[split.val], y[val])
        np.testing.assert_array_equal(split.inputs[split.test], inputs[test])
        np.testing.assert_array_equal(split.labels[split.test], y[test])
        assert split.labels[70 : split.n_train].tolist() == [1] * n_synthetic
        by_hand = smote(inputs[train], y[train], random_state=3)
        np.testing.assert_array_equal(split.inputs[split.train], by_hand.X)
        # The real rows of the training part: training rows, then validation rows.
        np.testing.assert_array_equal(split.inputs[split.real], inputs[fit_rows])
        # A tuning fold scores its validation rows.
        tuning = bench.make_split(X, y, train, train, val, None, n_components=4, smote_seed=3)
        assert tuning.n_test == 0
        assert tuning.test == tuning.val
        assert tuning.inputs.shape[0] == tuning.n_train + 20


# ---------------------------------------------------------------------------
# The feature cache
# ---------------------------------------------------------------------------


class TestCache:
    inputs = np.random.default_rng(8).uniform(-3, 3, size=(12, 4))
    labels = np.array([0, 1] * 6)

    def _counting(self) -> tuple[Any, list[int]]:
        calls: list[int] = []
        fmap = _map_by_hand(4, 1.0, 0.0, 1)

        def compute(x: np.ndarray) -> np.ndarray:
            calls.append(x.shape[0])
            return np.asarray(fmap.transform(x).numpy())

        return compute, calls

    def test_a_second_request_reads_the_same_features_without_computing(
        self, tmp_path: Path
    ) -> None:
        compute, calls = self._counting()
        config = {
            "feature_map": "RydbergFeatureMap",
            "config": _map_by_hand(4, 1.0, 0.0, 1).get_config(),
        }
        cache = bench.FeatureCache(tmp_path / "cache")
        first = cache.fetch(self.inputs, self.labels, config, compute)
        assert (first.hit, calls) == (False, [12])
        np.testing.assert_array_equal(
            first.features, _map_by_hand(4, 1.0, 0.0, 1).transform(self.inputs)
        )
        second = bench.FeatureCache(tmp_path / "cache").fetch(
            self.inputs, self.labels, config, compute
        )
        assert (second.hit, calls) == (True, [12])
        np.testing.assert_array_equal(second.features, first.features)
        assert second.features.dtype == np.float64
        # the simulation time of the first computation travels with the entry
        assert second.seconds == first.seconds > 0
        assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
        assert not list((tmp_path / "cache").glob("*.part"))

    def test_a_change_of_the_data_is_a_miss(self, tmp_path: Path) -> None:
        compute, calls = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        cache = bench.FeatureCache(tmp_path)
        cache.fetch(self.inputs, self.labels, config, compute)
        moved = self.inputs.copy()
        moved[7, 2] = np.nextafter(moved[7, 2], np.inf)  # one bit of one entry
        assert cache.fetch(moved, self.labels, config, compute).hit is False
        other_labels = self.labels.copy()
        other_labels[0] = 1
        assert cache.fetch(self.inputs, other_labels, config, compute).hit is False
        assert cache.fetch(self.inputs[:-1], self.labels[:-1], config, compute).hit is False
        assert calls == [12, 12, 12, 11]
        assert cache.fetch(self.inputs, self.labels, config, compute).hit is True

    def test_a_change_of_the_config_is_a_miss(self, tmp_path: Path) -> None:
        """Every setting ``get_config()`` records, and the shots, enter the key."""
        compute, calls = self._counting()
        cache = bench.FeatureCache(tmp_path)
        maps = [
            _map_by_hand(4, 1.0, 0.0, 1),
            _map_by_hand(4, 1.0, 0.0, 1, interactions=False),
            _map_by_hand(4, 3.0, 0.0, 1),  # another spacing
            _map_by_hand(4, 1.0, 0.5, 50),  # another γ
            _map_by_hand(4, 1.0, 0.5, 100),  # another step count
            _map_by_hand(4, 1.0, 0.0, 1, correlations=True),
        ]
        configs = [{"config": m.get_config(), "shots": None, "shot_seed": None} for m in maps]
        configs.append({**configs[0], "shots": 100, "shot_seed": 1})
        configs.append({**configs[0], "shots": 100, "shot_seed": 2})
        paths = {cache.path(self.inputs, self.labels, c) for c in configs}
        assert len(paths) == len(configs)
        for config in configs:
            assert cache.fetch(self.inputs, self.labels, config, compute).hit is False
        assert len(calls) == len(configs)
        # The same settings rebuilt from scratch are the same entry.
        rebuilt = {
            "config": _map_by_hand(4, 1.0, 0.0, 1).get_config(),
            "shots": None,
            "shot_seed": None,
        }
        assert cache.fetch(self.inputs, self.labels, rebuilt, compute).hit is True

    def test_the_key_is_the_hash_of_the_fingerprint_and_the_config(self, tmp_path: Path) -> None:
        config = {"config": _map_by_hand(4, 1.0, 0.5, 50).get_config()}
        text = json.dumps(
            {"data": fingerprint(self.inputs, self.labels), "config": config}, sort_keys=True
        )
        expected = hashlib.sha256(text.encode()).hexdigest()
        path = bench.FeatureCache(tmp_path).path(self.inputs, self.labels, config)
        assert path == tmp_path / f"{expected}.npz"
        assert bench.cache_key(fingerprint(self.inputs, self.labels), config) == expected

    def test_without_a_directory_every_request_computes(self) -> None:
        compute, calls = self._counting()
        cache = bench.FeatureCache(None)
        assert cache.path(self.inputs, self.labels, {}) is None
        assert cache.fetch(self.inputs, self.labels, {}, compute).hit is False
        assert cache.fetch(self.inputs, self.labels, {}, compute).hit is False
        assert calls == [12, 12]

    def test_a_repeated_run_reads_every_feature_matrix_and_scores_the_same(
        self, tmp_path: Path
    ) -> None:
        settings = _replace(QUICK, arms=("A", "B"))
        data = {"positive-control": bench.load_dataset("positive-control", settings)}
        first = bench.run(data, settings, tmp_path / "a.jsonl", cache_dir=tmp_path / "cache")
        stored = sorted(p.name for p in (tmp_path / "cache").iterdir())
        # per arm: the fold's rows and one matrix per tuning fold
        assert len(stored) == 2 * (1 + settings.inner_folds)
        second = bench.run(data, settings, tmp_path / "b.jsonl", cache_dir=tmp_path / "cache")
        assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == stored
        assert not any(r["feature_cache_hit"] for r in first)
        assert all(r["feature_cache_hit"] for r in second)
        for before, after in zip(first, second, strict=True):
            assert after["feature_seconds"] == before["feature_seconds"] > 0
            assert (after["mcc"], after["pr_auc"], after["threshold"]) == (
                before["mcc"],
                before["pr_auc"],
                before["threshold"],
            )
            assert after["separation"] == before["separation"]


# ---------------------------------------------------------------------------
# Strict JSON
# ---------------------------------------------------------------------------


def _strict_loads(line: str) -> Any:
    def refuse(token: str) -> Any:
        raise AssertionError(f"not strict JSON: {token}")

    return json.loads(line, parse_constant=refuse)


class TestJson:
    def test_infinity_and_nan_have_a_stated_mapping(self) -> None:
        record = {
            "a": math.inf,
            "b": -math.inf,
            "c": math.nan,
            "d": np.float64(np.inf),
            "e": [1.5, math.inf, (np.float32(-np.inf), math.nan)],
            "f": {"g": np.int64(3), "h": np.array([1, 2]), "i": None, "j": "inf", "k": True},
        }
        line = bench.to_json_line(record)
        assert "Infinity" not in line
        assert "NaN" not in line
        assert _strict_loads(line) == {
            "a": "inf",
            "b": "-inf",
            "c": None,
            "d": "inf",
            "e": [1.5, "inf", ["-inf", None]],
            "f": {"g": 3, "h": [1, 2], "i": None, "j": "inf", "k": True},
        }

    def test_the_separation_measures_survive_the_file(self, tmp_path: Path) -> None:
        """Perfect separation (``inf``) and identical rows (NaN), as #500 returns them."""
        perfect = separation_measures([[0.0], [0.0], [1.0], [1.0]], [0, 0, 1, 1]).to_dict()
        identical = separation_measures([[0.5], [0.5], [0.5], [0.5]], [0, 0, 1, 1]).to_dict()
        assert math.isinf(perfect["fisher_ratio"])
        assert math.isinf(perfect["distance_ratio"])
        assert math.isnan(identical["fisher_ratio"])
        assert math.isnan(identical["distance_ratio"])
        record = {
            "arm": "C",
            "v_over_omega": None,
            "separation": {"train": perfect, "test": identical},
        }
        path = tmp_path / "records.jsonl"
        path.write_text(bench.to_json_line(record) + "\n\n", encoding="utf-8")
        written = _strict_loads(path.read_text(encoding="utf-8").strip())
        assert written["separation"]["train"]["fisher_ratio"] == "inf"
        assert written["separation"]["test"]["fisher_ratio"] is None
        (read,) = bench.read_records(path)
        assert read["v_over_omega"] is None  # "does not apply" stays None
        for part, original in (("train", perfect), ("test", identical)):
            restored = read["separation"][part]
            assert set(restored) == set(original)
            assert type(restored["n_samples"]) is int
            for key, value in original.items():
                if isinstance(value, float) and math.isnan(value):
                    assert math.isnan(restored[key]), key
                else:
                    assert restored[key] == value, key


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


class TestTuning:
    def test_every_arm_gets_the_same_trials_on_folds_of_the_training_part(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _replace(
            QUICK, arms=FAST, n_trials=3, search_space={"lr": (0.01, 0.03, 0.1, 0.3)}
        )
        X, y, _ = bench.load_dataset("positive-control", settings)
        fold = _fold_by_hand(X, y, settings, 0)

        splits: list[tuple[Any, ...]] = []
        original_split = bench.make_split

        def spy_split(
            X_: Any, y_: Any, fit: Any, train: Any, val: Any, test: Any, **kw: Any
        ) -> Any:
            splits.append((fit, train, val, test, kw["smote_seed"]))
            return original_split(X_, y_, fit, train, val, test, **kw)

        fits_per_tune: list[int] = []
        fits: list[int] = [0]
        original_fit, original_tune = bench.fit_and_score, bench.tune

        def spy_fit(*args: Any, **kw: Any) -> Any:
            fits[0] += 1
            return original_fit(*args, **kw)

        def spy_tune(*args: Any, **kw: Any) -> Any:
            before = fits[0]
            result = original_tune(*args, **kw)
            fits_per_tune.append(fits[0] - before)
            return result

        monkeypatch.setattr(bench, "make_split", spy_split)
        monkeypatch.setattr(bench, "fit_and_score", spy_fit)
        monkeypatch.setattr(bench, "tune", spy_tune)
        records = list(
            bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None))
        )

        # The budget: the same number of trainings for each of the six models.
        assert len(records) == 6
        assert fits_per_tune == [3 * settings.inner_folds] * 6
        assert fits[0] == 6 * (3 * settings.inner_folds + 1)
        expected = _sample_configs({"lr": (0.01, 0.03, 0.1, 0.3)}, 3, fold.seeds["tune_seed"])
        for record in records:
            assert record["tuning"]["n_trials"] == 3
            assert record["tuning"]["inner_folds"] == settings.inner_folds
            assert record["tuning"]["candidates"] == expected
            scores = record["tuning"]["scores"]
            assert len(scores) == 3
            chosen = expected[scores.index(max(scores))]  # the first of the best
            assert record["hyperparameters"] == {**settings.defaults, **chosen}
            assert record["tuning_seconds"] > 0

        # The data: one outer split, then the tuning folds, which never touch a test row.
        outer, *inner = splits
        np.testing.assert_array_equal(outer[0], fold.train_part)
        np.testing.assert_array_equal(outer[3], fold.test_idx)
        assert len(inner) == settings.inner_folds
        by_hand = stratified_kfold(
            y[fold.train_part], settings.inner_folds, random_state=fold.seeds["tune_seed"]
        )
        held_out: list[int] = []
        for (fit, train, val, test, seed), (fit_pos, val_pos) in zip(inner, by_hand, strict=True):
            assert test is None
            assert seed == fold.seeds["tune_seed"]
            np.testing.assert_array_equal(fit, fold.train_part[fit_pos])
            np.testing.assert_array_equal(train, fit)  # fitted and trained on the same rows
            np.testing.assert_array_equal(val, fold.train_part[val_pos])
            assert not set(fit.tolist()) & set(val.tolist())
            assert not (set(fit.tolist()) | set(val.tolist())) & set(fold.test_idx.tolist())
            held_out += val.tolist()
        assert sorted(held_out) == fold.train_part.tolist()

    def test_the_best_mean_score_wins_and_the_first_wins_a_tie(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table = {0.1: [0.2, 0.4], 0.2: [0.5, 0.3], 0.3: [0.4, 0.4], 0.4: [0.1, 0.1]}
        calls: list[float] = []

        def fake_fit(model: Any, features: Any, split: Any, config: Any, **kw: Any) -> Any:
            calls.append(config["lr"])
            return bench.Fit(table[config["lr"]][calls.count(config["lr"]) - 1], 0.0, 0.5, 0.0, 1)

        monkeypatch.setattr(bench, "fit_and_score", fake_fit)
        candidates = [{"lr": lr} for lr in table]
        features = [np.zeros((4, 2)), np.zeros((4, 2))]
        chosen, scores = bench.tune("linear", [None, None], features, candidates, QUICK, 0)
        assert scores == pytest.approx([0.3, 0.4, 0.4, 0.1])
        assert chosen == {"lr": 0.2}
        assert calls == [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]

    def test_no_trials_means_the_default_settings(self, tmp_path: Path) -> None:
        settings = _replace(QUICK, arms=("C",), n_trials=0)
        X, y, _ = bench.load_dataset("positive-control", settings)
        records = list(
            bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None))
        )
        for record in records:
            assert record["tuning"] is None
            assert record["tuning_seconds"] == 0.0
            assert record["hyperparameters"] == settings.defaults


# ---------------------------------------------------------------------------
# --quick: one fold of one small grid point, every field of every line
# ---------------------------------------------------------------------------

ROWS = [
    ("A", "linear"),
    ("A", "mlp"),
    ("B", "linear"),
    ("B", "mlp"),
    ("C", "linear"),
    ("C", "mlp"),
    ("D", "linear"),
    ("D", "mlp"),
    ("D-hybrid", "hybrid"),
]
FIELDS = {
    "dataset",
    "data_sha256",
    "n_samples",
    "n_positives",
    "n_raw_features",
    "arm",
    "head",
    "representation",
    "feature_map_trained",
    "v_over_omega",
    "gamma_over_omega",
    "spacing_um",
    "omega",
    "omega_t",
    "evolution_time_us",
    "n_steps",
    "feature_map",
    "measurement",
    "shots",
    "fold",
    "n_splits",
    "train_idx",
    "val_idx",
    "test_idx",
    "n_synthetic",
    "seeds",
    "feature_dim",
    "n_parameters",
    "n_frozen_parameters",
    "model",
    "hyperparameters",
    "tuning",
    "mcc",
    "pr_auc",
    "threshold",
    "epochs",
    "feature_seconds",
    "feature_cache_hit",
    "train_seconds",
    "tuning_seconds",
    "separation",
    "settings",
    "environment",
}


@pytest.mark.slow
class TestQuick:
    def test_one_line_per_arm_and_head_of_one_fold_and_one_grid_point(self, quick: Any) -> None:
        assert [(r["arm"], r["head"]) for r in quick.lines] == ROWS
        assert {r["fold"] for r in quick.lines} == {0}
        assert {(r["v_over_omega"], r["gamma_over_omega"]) for r in quick.lines[:4]} == {
            (1.0, 0.1)
        }
        assert {(r["v_over_omega"], r["gamma_over_omega"]) for r in quick.lines[4:]} == {
            (None, None)
        }
        lines = quick.text.splitlines()
        assert len(lines) == len(ROWS)
        for line, record in zip(lines, quick.lines, strict=True):
            assert set(_strict_loads(line)) == FIELDS == set(record)

    def test_every_arm_has_the_same_fold_rows_and_seeds(self, quick: Any) -> None:
        """Compared between the lines, and against the fold rebuilt by hand."""
        fold = quick.fold
        first = quick.lines[0]
        for record in quick.lines:
            for key in ("train_idx", "val_idx", "test_idx", "n_synthetic", "seeds", "n_splits"):
                assert record[key] == first[key], (record["arm"], key)
        assert first["train_idx"] == np.sort(fold.train_idx).tolist()
        assert first["val_idx"] == np.sort(fold.val_idx).tolist()
        assert first["test_idx"] == np.sort(fold.test_idx).tolist()
        assert first["n_synthetic"] == fold.n_synthetic
        rows = first["train_idx"] + first["val_idx"] + first["test_idx"]
        assert sorted(rows) == list(range(QUICK.n_samples))  # a partition of the dataset
        # a quarter of each class, the first fold taking a remainder: 40 or 41 of 160 rows
        assert len(first["test_idx"]) in (40, 41)
        assert set(first["train_idx"]).isdisjoint(first["val_idx"])
        run_root = np.random.SeedSequence([QUICK.random_state, 1 << 16])
        data_seed, shot_seed, circuit_seed = _draw(run_root, 3)
        assert first["seeds"] == {
            "random_state": 0,
            **fold.seeds,
            "data_seed": data_seed,
            "shot_seed": shot_seed,
            "circuit_seed": circuit_seed,
        }
        assert quick.data_seed == data_seed
        assert len(set(first["seeds"].values())) == len(first["seeds"])

    def test_dataset_settings_and_environment(self, quick: Any) -> None:
        for record in quick.lines:
            assert record["dataset"] == "positive-control"
            assert record["data_sha256"] == fingerprint(quick.X, quick.y)
            assert record["n_samples"] == 160
            assert record["n_positives"] == int(quick.y.sum())
            assert record["n_raw_features"] == 15
            assert record["environment"] == environment()
            assert record["settings"] == json.loads(json.dumps(QUICK.as_dict()))
            assert record["measurement"] == "exact"
            assert record["shots"] is None

    def test_grid_fields(self, quick: Any) -> None:
        for record in quick.lines[:4]:
            assert record["omega"] == pytest.approx(OMEGA)
            assert record["omega_t"] == pytest.approx(math.pi)
            assert record["evolution_time_us"] == pytest.approx(0.25)  # π / 4π
            assert record["spacing_um"] == pytest.approx(8.69, abs=0.005)  # R_b of the page
            assert record["n_steps"] == 40
            by_hand = _map_by_hand(4, 1.0, 0.1, 40, interactions=record["arm"] == "A")
            assert record["feature_map"] == {
                "feature_map": "RydbergFeatureMap",
                "config": by_hand.get_config(),
                "shots": None,
                "shot_seed": None,
            }
            assert record["representation"] == (
                "rydberg" if record["arm"] == "A" else "rydberg-noninteracting"
            )
        for record in quick.lines[4:]:
            for key in ("spacing_um", "omega", "omega_t", "evolution_time_us", "n_steps"):
                assert record[key] is None
        assert [r["feature_map"] for r in quick.lines[4:6]] == [None, None]
        assert quick.lines[6]["feature_map"]["feature_map"] == "QuantumEncodingLayer"
        assert quick.lines[8]["feature_map"] is None

    def test_dimensions_and_parameter_counts(self, quick: Any) -> None:
        """Linear ``d + 1 = 5``; MLP ``hidden (d + 2) + 1 = 25``; hybrid ``24 + 5``."""
        expected = {"linear": 5, "mlp": 25, "hybrid": 29}
        classes = {
            "linear": "LinearClassifier",
            "mlp": "ClassicalBaseline",
            "hybrid": "HybridBinaryClassifier",
        }
        for record in quick.lines:
            assert record["feature_dim"] == 4
            assert record["n_parameters"] == expected[record["head"]]
            assert record["model"]["class"] == classes[record["head"]]
            assert record["model"]["config"]["init_seed"] == record["seeds"]["init_seed"]
            assert record["n_frozen_parameters"] == (24 if record["arm"] == "D" else 0)
            assert record["feature_map_trained"] is (record["arm"] == "D-hybrid")
        # The MLP on the inputs has the parameter count of the MLP head in A.
        assert _line(quick, "C", "mlp")["n_parameters"] == _line(quick, "A", "mlp")["n_parameters"]
        assert _line(quick, "C", "mlp")["model"] == _line(quick, "A", "mlp")["model"]
        assert _line(quick, "C", "linear")["model"] == _line(quick, "B", "linear")["model"]

    def test_tuning_budget_is_the_same_for_every_line(self, quick: Any) -> None:
        for record in quick.lines:
            assert record["tuning"]["n_trials"] == 2
            assert record["tuning"]["inner_folds"] == 2
            assert record["tuning"]["candidates"] == [{"lr": 0.01}, {"lr": 0.05}]
            scores = record["tuning"]["scores"]
            assert len(scores) == 2
            assert all(-1.0 <= s <= 1.0 for s in scores)
            best = 0.01 if scores[0] >= scores[1] else 0.05
            assert record["hyperparameters"] == {
                "lr": best,
                "max_epochs": 6,
                "batch_size": 32,
                "patience": None,
            }

    def test_seconds_are_recorded_separately(self, quick: Any) -> None:
        for record in quick.lines:
            assert record["train_seconds"] > 0
            assert record["tuning_seconds"] > 0
            assert record["epochs"] == 6
            simulated = record["arm"] in ("A", "B", "D")
            assert (record["feature_seconds"] > 0) is simulated
            assert record["feature_cache_hit"] is False  # a fresh cache directory
        # Both heads of an arm read the same features: the same simulation time.
        for arm in ("A", "B", "D"):
            assert (
                _line(quick, arm, "linear")["feature_seconds"]
                == _line(quick, arm, "mlp")["feature_seconds"]
            )
        # per simulated arm: the fold's rows and one matrix per tuning fold
        assert len(list(quick.cache.glob("*.npz"))) == 3 * (1 + QUICK.inner_folds)

    def _features(self, quick: Any, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The arm's features of the training part's real rows and of the test rows, by hand."""
        fold = quick.fold
        real = np.concatenate([fold.train_idx, fold.val_idx])
        if arm == "C":
            return fold.inputs[real], fold.inputs[fold.test_idx], real
        fmap = _map_by_hand(4, 1.0, 0.1, 40, interactions=arm == "A")
        return (
            fmap.transform(fold.inputs[real]).numpy(),
            fmap.transform(fold.inputs[fold.test_idx]).numpy(),
            real,
        )

    @pytest.mark.parametrize("arm", ["A", "B", "C"])
    def test_separation_measures_of_the_matrix_the_head_receives(
        self, quick: Any, arm: str
    ) -> None:
        train, test, real = self._features(quick, arm)
        expected = {
            "train": separation_measures(train, quick.y[real]).to_dict(),
            "test": separation_measures(test, quick.y[quick.fold.test_idx]).to_dict(),
        }
        for head in ("linear", "mlp"):
            recorded = _line(quick, arm, head)["separation"]
            assert set(recorded) == {"train", "test"}
            for part in ("train", "test"):
                assert recorded[part] == pytest.approx(expected[part], rel=1e-7, abs=1e-12)
        n_test = quick.fold.test_idx.size
        assert expected["train"]["n_samples"] == 160 - n_test  # no synthetic row among them
        assert expected["test"]["n_samples"] == n_test
        assert expected["train"]["n_features"] == 4

    def test_arm_b_separation_is_that_of_single_atom_features(self, quick: Any) -> None:
        """The recorded B line, from four one-atom maps: no interaction entered it."""
        fold = quick.fold
        alone = RydbergFeatureMap(
            AtomRegister.chain(1, 1.0),
            PulseEncoding(1, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.1 * OMEGA,
            n_steps=40,
        )
        test = fold.inputs[fold.test_idx]
        features = np.column_stack([alone.transform(test[:, [i]]).numpy()[:, 0] for i in range(4)])
        expected = separation_measures(features, quick.y[fold.test_idx]).to_dict()
        recorded = _line(quick, "B", "linear")["separation"]["test"]
        assert recorded == pytest.approx(expected, rel=1e-7, abs=1e-12)
        other = _line(quick, "A", "linear")["separation"]["test"]
        assert other["fisher_ratio"] != pytest.approx(expected["fisher_ratio"], rel=1e-3)

    @pytest.mark.parametrize(("arm", "head"), [("C", "linear"), ("A", "mlp"), ("B", "linear")])
    def test_scores_of_a_model_trained_by_hand(self, quick: Any, arm: str, head: str) -> None:
        """
        MCC, PR-AUC, threshold and epochs of a line, from a model built and
        trained here with the recorded hyperparameters on the fold rebuilt by
        hand.  For arm C with the linear head this is the logistic regression.
        """
        fold = quick.fold
        record = _line(quick, arm, head)

        # One matrix for the fold's rows, as the script evolves them: a batch
        # of another size may round differently in the last digit (#509).
        stacked = np.concatenate(
            [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
        )
        if arm == "C":
            features = torch.from_numpy(stacked.astype(np.float32))
        else:
            fmap = _map_by_hand(4, 1.0, 0.1, 40, interactions=arm == "A")
            features = fmap.transform(stacked).to(torch.float32)
        n_train, n_val = fold.train_inputs.shape[0], fold.val_idx.size
        x_train, x_val, x_test = features.split([n_train, n_val, fold.test_idx.size])

        seed = fold.seeds["init_seed"]
        model: nn.Module = (
            LinearClassifier(4, init_seed=seed)
            if head == "linear"
            else ClassicalBaseline(4, [QUICK.hidden], init_seed=seed)
        )
        settings = record["hyperparameters"]
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=settings["lr"]),
            x_train,
            torch.from_numpy(fold.train_labels.astype(np.float32)),
            x_val,
            torch.from_numpy(quick.y[fold.val_idx].astype(np.float32)),
            max_epochs=settings["max_epochs"],
            batch_size=settings["batch_size"],
            monitor="mcc",
            patience=settings["patience"],
            generator=torch.Generator().manual_seed(fold.seeds["batch_seed"]),
        )
        assert isinstance(model, LinearClassifier | ClassicalBaseline)
        probability = model.predict_proba(x_test)
        truth = quick.y[fold.test_idx]
        assert history.best_threshold is not None
        assert record["threshold"] == pytest.approx(history.best_threshold, abs=1e-6)
        prediction = (probability >= history.best_threshold).long()
        assert record["mcc"] == pytest.approx(matthews_corrcoef(truth, prediction), abs=1e-9)
        assert record["pr_auc"] == pytest.approx(pr_auc(truth, probability), abs=1e-6)
        assert record["epochs"] == history.n_epochs
        assert 0.0 <= record["pr_auc"] <= 1.0

    def test_records_returned_are_the_lines_written(self, quick: Any) -> None:
        assert len(quick.records) == len(quick.lines)
        for returned, read in zip(quick.records, quick.lines, strict=True):
            assert returned["mcc"] == read["mcc"]
            assert returned["test_idx"].tolist() == read["test_idx"]
        table = bench.summarise(quick.records)
        assert len(table.splitlines()) == 1 + len(ROWS)


class TestShotsRun:
    def test_the_measurement_assumption_is_recorded_per_line(self, tmp_path: Path) -> None:
        settings = _replace(QUICK, arms=("A", "C"), shots=32, n_trials=0)
        data = {"positive-control": bench.load_dataset("positive-control", settings)}
        records = bench.run(data, settings, tmp_path / "shots.jsonl", cache_dir=tmp_path / "c")
        read = bench.read_records(tmp_path / "shots.jsonl")
        assert [(r["arm"], r["measurement"], r["shots"]) for r in read] == [
            ("A", "shots", 32),
            ("A", "shots", 32),
            ("C", "exact", None),
            ("C", "exact", None),
        ]
        X, y, _ = data["positive-control"]
        fold = _fold_by_hand(X, y, settings, 0)
        shot_seed = records[0]["seeds"]["shot_seed"]
        assert read[0]["feature_map"]["shots"] == 32
        assert read[0]["feature_map"]["shot_seed"] == shot_seed
        # The script's features of the fold's rows, drawn by hand with that seed.
        stacked = np.concatenate(
            [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
        )
        sampled = _map_by_hand(4, 1.0, 0.1, 40).transform(
            stacked, shots=32, generator=torch.Generator().manual_seed(shot_seed)
        )
        test = sampled[-fold.test_idx.size :].numpy()
        expected = separation_measures(test, y[fold.test_idx]).to_dict()
        assert read[0]["separation"]["test"] == pytest.approx(expected, rel=1e-9)
        exact = _map_by_hand(4, 1.0, 0.1, 40).transform(fold.inputs[fold.test_idx]).numpy()
        assert np.abs(test - exact).max() > 0.01  # sampled, not exact
