"""
`compare_artifacts` took two labels and never read them.

The tool exists to answer "did this reproduce, and where did it move" -- so
the two sides are almost always something like baseline and candidate, and
`label_a`/`label_b` are how a caller says so. They were declared with no
description, passed nothing, and read nowhere: every report came back
talking about "a" and "b" whatever the caller called them.

The tool had no tests at all, which is how a field can be inert from the
day it was written.

The difference entries still key on `a` and `b` -- that is structure, and
renaming keys would break every consumer -- so the labels name the sides in
the prose and are echoed in the result, which is what says which key is
which.
"""

from __future__ import annotations

import pytest

from standard_quant_tools.agent.runtimes import resolve


@pytest.fixture
def meta():
    return resolve("meta")


def _compare(meta, a, b, **kwargs):
    return meta.dispatch("compare_artifacts", {"a": a, "b": b, **kwargs})


class TestTheLabelsAreRead:
    def test_they_come_back_in_the_result(self, meta):
        result = _compare(
            meta,
            {"sharpe": 1.0},
            {"sharpe": 1.4},
            label_a="baseline",
            label_b="candidate",
        )
        assert result["label_a"] == "baseline"
        assert result["label_b"] == "candidate"

    def test_the_identical_message_uses_them(self, meta):
        result = _compare(
            meta, {"s": 1.0}, {"s": 1.0}, label_a="run_01", label_b="run_02"
        )
        assert any("run_01" in w and "run_02" in w for w in result["warnings"]), result[
            "warnings"
        ]

    def test_the_structural_message_uses_them(self, meta):
        result = _compare(
            meta,
            {"sharpe": 1.0, "only_here": 1},
            {"sharpe": 1.0},
            label_a="baseline",
            label_b="candidate",
        )
        assert any("baseline" in w and "candidate" in w for w in result["warnings"])

    def test_it_says_which_key_is_which_side(self, meta):
        """The labels name the sides; the entries still key on a/b. Without
        this sentence a reader has the names and no way to map them."""
        result = _compare(
            meta,
            {"x": 1, "gone": 2},
            {"x": 1},
            label_a="baseline",
            label_b="candidate",
        )
        assert any("`a` is baseline" in w for w in result["warnings"])

    def test_the_defaults_still_read_naturally(self, meta):
        result = _compare(meta, {"s": 1.0}, {"s": 1.0})
        assert result["label_a"] == "a"
        assert "a and b are identical" in " ".join(result["warnings"])


class TestTheComparisonItselfStillWorks:
    """It had no tests, so these are the first."""

    def test_identical_artifacts_are_identical(self, meta):
        result = _compare(meta, {"a": 1.0, "b": "x"}, {"a": 1.0, "b": "x"})
        assert result["identical"] is True
        assert result["n_differences"] == 0

    def test_a_value_change_is_found_and_measured(self, meta):
        result = _compare(meta, {"sharpe": 1.0}, {"sharpe": 1.5})
        assert result["identical"] is False
        assert result["largest_relative_change"] == pytest.approx(1 / 3, rel=1e-6)

    def test_a_change_inside_the_tolerance_is_not_a_difference(self, meta):
        result = _compare(meta, {"x": 1.0}, {"x": 1.0 + 1e-12}, tolerance=1e-9)
        assert result["identical"] is True

    def test_a_missing_field_is_structural_not_numerical(self, meta):
        result = _compare(meta, {"x": 1, "y": 2}, {"x": 1})
        kinds = {d["kind"] for d in result["differences"]}
        assert kinds == {"only_in_a"}

    def test_a_type_change_is_reported_as_one(self, meta):
        result = _compare(meta, {"flag": True}, {"flag": 1})
        assert {d["kind"] for d in result["differences"]} == {"type_changed"}

    def test_nested_fields_are_compared_by_path(self, meta):
        result = _compare(
            meta,
            {"metrics": {"sharpe": 1.0, "dd": -0.2}},
            {"metrics": {"sharpe": 1.4, "dd": -0.2}},
        )
        paths = [d["path"] for d in result["differences"]]
        assert any("sharpe" in p for p in paths)
        assert not any("dd" in p for p in paths)

    def test_identical_is_evidence_not_proof(self, meta):
        """The warning exists because an identical result from an identical
        cached input says nothing about the computation."""
        result = _compare(meta, {"s": 1.0}, {"s": 1.0})
        assert any("not proof" in w for w in result["warnings"])
