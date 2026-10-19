import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")  # the example imports hqnn_forge.sklearn

ROOT = Path(__file__).resolve().parent.parent


def _script():
    path = ROOT / "examples" / "does_the_quantum_layer_help.py"
    examples_dir = str(ROOT / "examples")

    spec = importlib.util.spec_from_file_location("does_the_quantum_layer_help", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    sys.path.insert(0, examples_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(examples_dir)

    return module


def test_near_constant_column_is_only_centred():
    example = _script()

    rng = np.random.default_rng(0)
    near_constant = 0.5 + rng.integers(-1, 2, size=200) * 2.0**-53
    varying = 0.5 + 1e-6 * rng.standard_normal(200)
    features = np.column_stack((near_constant, varying))

    fit_rows = np.arange(150)
    val_rows = np.arange(150, 175)
    test_rows = np.arange(175, 200)

    fitted, validation, test = example.standardise(features, fit_rows, val_rows, test_rows)

    expected = near_constant - near_constant[fit_rows].mean()
    np.testing.assert_allclose(fitted[:, 0], expected[fit_rows], atol=0)
    assert np.max(np.abs(fitted[:, 0])) < 2e-16

    np.testing.assert_allclose(validation[:, 0], expected[val_rows], atol=0)
    np.testing.assert_allclose(test[:, 0], expected[test_rows], atol=0)
    np.testing.assert_allclose(fitted[:, 1].std(), 1.0, rtol=1e-9)
    fit = varying[fit_rows]
    expected_varying = (varying - fit.mean()) / fit.std()

    np.testing.assert_allclose(
        validation[:, 1],
        expected_varying[val_rows],
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        test[:, 1],
        expected_varying[test_rows],
        rtol=0,
        atol=1e-8,
    )
