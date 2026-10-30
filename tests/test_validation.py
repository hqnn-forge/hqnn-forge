"""
tests/test_validation.py
========================
Unit tests for the shared integer validator in ``hqnn_forge._validation`` (#662).
"""

from __future__ import annotations

import numpy as np
import pytest

from hqnn_forge._validation import check_int


class TestAccepted:
    def test_python_int(self) -> None:
        assert check_int(5, "shots", minimum=1) == 5

    def test_minimum_is_inclusive(self) -> None:
        assert check_int(1, "shots", minimum=1) == 1
        assert check_int(0, "n_steps", minimum=0) == 0

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint16, np.uint64])
    def test_numpy_integers_are_returned_as_plain_int(self, dtype: type) -> None:
        result = check_int(dtype(100), "shots", minimum=1)
        assert result == 100
        assert type(result) is int

    def test_none_only_when_allowed(self) -> None:
        assert check_int(None, "max_rows", minimum=1, allow_none=True) is None


class TestRefused:
    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_booleans(self, value: object) -> None:
        # bool is a numbers.Integral, so it has to be refused explicitly.
        with pytest.raises(ValueError, match=r"shots must be an integer >= 0; got "):
            check_int(value, "shots", minimum=0)

    def test_message_names_the_value_for_a_python_bool(self) -> None:
        with pytest.raises(ValueError, match=r"shots must be an integer >= 1; got True\."):
            check_int(True, "shots", minimum=1)

    @pytest.mark.parametrize("value", [2.0, 2.5, "3", b"3", [3], (3,), 1 + 0j, np.float64(2.0)])
    def test_non_integers(self, value: object) -> None:
        with pytest.raises(ValueError, match=r"batch_size must be an integer >= 1; got "):
            check_int(value, "batch_size", minimum=1)

    @pytest.mark.parametrize("value", [0, -1, np.int64(0), np.int64(-7)])
    def test_below_minimum(self, value: object) -> None:
        with pytest.raises(ValueError, match=r"n_atoms must be an integer >= 1; got "):
            check_int(value, "n_atoms", minimum=1)

    def test_none_is_refused_by_default(self) -> None:
        with pytest.raises(ValueError, match=r"max_rows must be an integer >= 1; got None\."):
            check_int(None, "max_rows", minimum=1)

    def test_message_marks_none_as_allowed(self) -> None:
        with pytest.raises(ValueError, match=r"max_rows must be an integer >= 2 or None; got 1\."):
            check_int(1, "max_rows", minimum=2, allow_none=True)

    def test_minimum_appears_in_the_message(self) -> None:
        with pytest.raises(ValueError, match=r"n_steps must be an integer >= 0; got -1\."):
            check_int(-1, "n_steps", minimum=0)


def test_check_int_numpy_and_bool() -> None:
    # NumPy integers accepted and normalised to int.
    assert check_int(np.int64(100), "shots", minimum=1) == 100
    assert isinstance(check_int(np.int64(100), "shots", minimum=1), int)

    # Booleans refused even though isinstance(True, numbers.Integral) is True.
    with pytest.raises(ValueError, match=r"shots must be an integer >= 1; got True"):
        check_int(True, "shots", minimum=1)