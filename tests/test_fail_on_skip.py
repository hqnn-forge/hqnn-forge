"""
tests/test_fail_on_skip.py
==========================
The ``HQNN_FORGE_FAIL_ON_SKIP`` hooks in ``conftest.py`` (#187), run against
small throwaway suites that load the same ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ENV = "HQNN_FORGE_FAIL_ON_SKIP"

SUITE = """
import pytest

def test_passes():
    pass

def test_skips_inside():
    pytest.skip("needs a thing")

@pytest.mark.skipif(True, reason="condition held")
def test_skipif():
    pass

@pytest.mark.may_skip
def test_allowed_to_skip():
    pytest.skip("no gpu here")

@pytest.mark.xfail(reason="known", strict=True)
def test_expected_failure():
    assert False

def test_importorskip_in_test():
    pytest.importorskip("hqnn_forge_no_such_module")
"""

MODULE_SKIP = """
import pytest
pytest.importorskip("hqnn_forge_no_such_module")

def test_never_collected():
    pass
"""


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest((Path(__file__).parent / "conftest.py").read_text())
    pytester.makepyfile(test_suite=SUITE, test_module_skip=MODULE_SKIP)
    return pytester


def test_skips_stay_skips_without_the_variable(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    result = suite.runpytest()
    result.assert_outcomes(passed=1, skipped=5, xfailed=1)
    assert result.ret == pytest.ExitCode.OK


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_only_the_value_1_turns_it_on(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(ENV, value)
    suite.runpytest().assert_outcomes(passed=1, skipped=5, xfailed=1)


def test_every_unmarked_skip_fails_with_its_reason(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "1")
    # Past the module-level skip, which would otherwise stop collection.
    result = suite.runpytest("test_suite.py")

    # skipif is decided at setup, so it fails as a setup error.
    result.assert_outcomes(passed=1, failed=2, errors=1, skipped=1, xfailed=1)
    result.stdout.fnmatch_lines(
        [
            f"*test_skips_inside skipped, and {ENV}=1 makes a skip fail.",
            "Skipped: needs a thing",
            "Mark the test `may_skip` if the skip is expected in CI.",
        ]
    )
    result.stdout.fnmatch_lines(["Skipped: condition held"])
    result.stdout.fnmatch_lines(["Skipped: could not import 'hqnn_forge_no_such_module'*"])
    result.stdout.no_fnmatch_line("*test_allowed_to_skip skipped, and*")


def test_a_module_skipped_at_import_fails_collection(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "1")
    result = suite.runpytest("test_module_skip.py")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        [
            f"test_module_skip.py skipped, and {ENV}=1 makes a skip fail.",
            "Skipped: could not import 'hqnn_forge_no_such_module'*",
        ]
    )


def test_deselected_tests_are_not_skips(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pytest -m "not slow", the quick local run (#323): deselected tests never
    # run and produce no report, so the fail-on-skip hooks must not see them.
    monkeypatch.setenv(ENV, "1")
    suite.makepyfile(
        test_marked="""
import pytest

@pytest.mark.slow
def test_slow():
    pass

def test_fast():
    pass
"""
    )
    result = suite.runpytest("-m", "not slow", "test_marked.py")
    result.assert_outcomes(passed=1, deselected=1)
