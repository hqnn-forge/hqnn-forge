"""
tests/test_seeded_reference.py
==============================
Seeded models initialise exactly as the stored reference says (#310).

See ``tests/seeded_reference.py`` for what is stored and how to regenerate it
after an intended change.  The comparison is at a float32-round-off tolerance:
a reordered construction, an added module or a changed number of draws moves
the weights by O(1), not by 1e-6.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from seeded_reference import CONFIGS, REFERENCE, SEED, summarise

# float32 weights, summed in float64: the sums of O(10) values agree to about
# 1e-6; any change to the draws changes them by far more.
ATOL = 1e-5
RTOL = 1e-6


def _stored() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text())  # type: ignore[no-any-return]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= ATOL + RTOL * abs(b)


def _compare(where: str, got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    if got["shape"] != want["shape"]:
        return [f"{where}: shape {got['shape']} != stored {want['shape']}"]
    problems = []
    for field in ("sum", "sum_abs"):
        if not _close(got[field], want[field]):
            problems.append(f"{where}: {field} {got[field]!r} != stored {want[field]!r}")
    for i, (g, w) in enumerate(zip(got["values"], want["values"], strict=True)):
        if not _close(g, w):
            problems.append(f"{where}: value[{i}] {g!r} != stored {w!r}")
    return problems


def test_the_reference_covers_every_configuration() -> None:
    stored = _stored()
    assert stored["seed"] == SEED
    assert set(stored["configs"]) == set(CONFIGS), (
        "tests/data/seeded_reference.json is out of date with CONFIGS; "
        "run `uv run python tests/seeded_reference.py`"
    )


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_seeded_initialisation_matches_the_reference(name: str) -> None:
    want = _stored()["configs"][name]
    got = summarise(name)
    assert set(got["state_dict"]) == set(want["state_dict"]), (
        f"{name}: state-dict keys changed: {sorted(set(got['state_dict']) ^ set(want['state_dict']))}"
    )
    problems = []
    for key in sorted(want["state_dict"]):
        problems += _compare(f"{name} {key}", got["state_dict"][key], want["state_dict"][key])
    problems += _compare(f"{name} output", got["output"], want["output"])
    assert not problems, (
        f"{name} no longer initialises as recorded (seed {SEED}, reference written with "
        f"torch {_stored()['torch']}, running {torch.__version__}).  If the change is "
        f"intended, run `uv run python tests/seeded_reference.py` and say so in the PR.\n  "
        + "\n  ".join(problems)
    )


def test_the_summary_is_sensitive_to_one_extra_draw() -> None:
    # The property the reference relies on: one extra draw from the global RNG
    # before construction moves every summary well past the tolerance.
    name = "serial"
    baseline = summarise(name)
    build, _ = CONFIGS[name]
    original = CONFIGS[name]
    try:
        CONFIGS[name] = (lambda: (torch.rand(1), build())[1], original[1])
        shifted = summarise(name)
    finally:
        CONFIGS[name] = original
    moved = 0
    for key, want in baseline["state_dict"].items():
        if want["sum_abs"] == 0.0:  # zero-initialised biases do not depend on the seed
            assert shifted["state_dict"][key]["sum_abs"] == 0.0, key
            continue
        assert _compare(key, shifted["state_dict"][key], want), key
        moved += 1
    assert moved >= 3  # encoder, circuit and head weights


@pytest.mark.parametrize("name", sorted(n for n in CONFIGS if n.endswith("-init-seed")))
def test_init_seed_configurations_ignore_the_global_rng(name: str) -> None:
    # The *-init-seed entries pin the private-RNG path (#175): an extra draw
    # before construction must leave them exactly where they are.
    baseline = summarise(name)
    build, width = CONFIGS[name]
    try:
        CONFIGS[name] = (lambda: (torch.rand(1), build())[1], width)
        shifted = summarise(name)
    finally:
        CONFIGS[name] = (build, width)
    assert shifted == baseline
