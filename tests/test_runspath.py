"""
One path-traversal guard, shared by both artifact stores.

`backtest.artifacts` and `modeling.artifacts` each had their own copy of the
identifier validator and the runs-directory lookup: same regex, same
message, written twice. Nothing was wrong with either, which is exactly the
problem -- two copies of a security check are equal only until one of them
is improved, and then the codebase reads as fixed everywhere while being
fixed in one place.

The tests below are mostly about that IDENTITY. The traversal cases are
already covered against each store individually in
tests/backtest/test_artifacts.py; what was untestable before is the claim
that the two stores agree, because there was nothing shared to point at.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from standard_quant_tools import _runspath
from standard_quant_tools.backtest import artifacts as backtest_artifacts
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as modeling_artifacts

TRAVERSAL = [
    "../escape",
    "..",
    "a/b",
    "a\\b",
    "/abs",
    "C:\\Windows",
    "",
    "with space",
    "semi;colon",
    "null\x00byte",
    "dot.dot",
]


class TestBothStoresUseTheOneGuard:
    def test_the_validator_is_the_same_object_in_both(self):
        """Not "equivalent" -- the same function. An equivalence test passes
        for two copies that have not drifted yet, which is the state this
        started in."""
        assert (
            backtest_artifacts._validate_identifier
            is modeling_artifacts._validate_identifier
            is _runspath.validate_identifier
        )

    def test_the_runs_dir_lookup_is_the_same_object_in_both(self):
        assert (
            backtest_artifacts._runs_dir
            is modeling_artifacts._runs_dir
            is _runspath.runs_dir
        )

    def test_the_containment_check_is_shared_rather_than_open_coded(self):
        """`modeling.artifacts.run_dir` used to inline this check instead of
        sharing it, so the two stores defended the same attack with two
        pieces of code."""
        assert (
            backtest_artifacts._resolved_within_runs_dir
            is modeling_artifacts._resolved_within_runs_dir
            is _runspath.resolve_within_runs_dir
        )


class TestWhatTheGuardRefuses:
    @pytest.mark.parametrize("value", TRAVERSAL)
    def test_a_non_slug_is_refused(self, value):
        with pytest.raises(ValidationError, match="not a valid identifier"):
            _runspath.validate_identifier(value, "run_id")

    @pytest.mark.parametrize("value", TRAVERSAL)
    def test_and_both_stores_refuse_it_identically(self, value, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        with pytest.raises(ValidationError):
            modeling_artifacts.run_dir(value)
        with pytest.raises(ValidationError):
            backtest_artifacts._validate_identifier(value, "run_id")

    @pytest.mark.parametrize(
        "value", ["run", "RUN-1", "ds_abc123", "mdl_0011aabb", "a-b_c", "9"]
    )
    def test_a_plain_slug_is_accepted(self, value):
        _runspath.validate_identifier(value, "run_id")


class TestContainment:
    def test_a_path_inside_the_root_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        resolved = _runspath.resolve_within_runs_dir(tmp_path / "run" / "x.parquet")
        assert resolved.is_relative_to(tmp_path.resolve())

    def test_a_path_outside_the_root_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        with pytest.raises(ValidationError, match="escapes SQT_RUNS_DIR"):
            _runspath.resolve_within_runs_dir(tmp_path / "elsewhere" / "x.parquet")

    def test_the_env_var_relocates_the_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "custom"))
        assert _runspath.runs_dir() == Path(str(tmp_path / "custom"))

    def test_without_the_env_var_it_falls_back_under_home(self, monkeypatch):
        monkeypatch.delenv("SQT_RUNS_DIR", raising=False)
        assert _runspath.runs_dir().is_relative_to(Path.home())
