"""
Tests for backtest/sizing.py — SCORE -> TARGET_WEIGHT construction methods.

Each test checks the invariant the function promises (gross-leverage bound,
dollar-neutrality, degenerate-row handling) on small synthetic score panels
rather than re-deriving expected numeric weights by hand.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.sizing import (
    rank_weighted, equal_weight_top_bottom, zscore_normalized,
    vol_scaled, dollar_neutral,
)
from standard_quant_tools.error import ValidationError


@pytest.fixture
def scores():
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    return pd.DataFrame(
        {"A": [1.0, 2.0, -1.0], "B": [2.0, 1.0, 0.0], "C": [-1.0, -3.0, 1.0], "D": [0.5, 0.0, -0.5]},
        index=dates,
    )


class TestValidation:
    def test_empty_scores_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            rank_weighted(pd.DataFrame())

    def test_nan_scores_raises(self, scores):
        bad = scores.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValidationError, match="NaN"):
            rank_weighted(bad)


class TestRankWeighted:
    def test_gross_leverage_matches(self, scores):
        w = rank_weighted(scores, gross_leverage=2.0)
        assert w.abs().sum(axis=1).to_numpy() == pytest.approx([2.0, 2.0, 2.0])

    def test_sum_is_zero(self, scores):
        w = rank_weighted(scores)
        assert w.sum(axis=1).to_numpy() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_highest_score_gets_most_positive_weight(self, scores):
        w = rank_weighted(scores)
        row0 = w.iloc[0]
        assert row0.idxmax() == "B"  # B=2.0 is the highest score in row 0


class TestEqualWeightTopBottom:
    def test_gross_leverage_and_counts(self, scores):
        w = equal_weight_top_bottom(scores, n_long=1, n_short=1, gross_leverage=1.0)
        row0 = w.iloc[0]
        assert (row0 > 0).sum() == 1
        assert (row0 < 0).sum() == 1
        assert row0.abs().sum() == pytest.approx(1.0)
        assert row0.idxmax() == "B"   # highest score row 0
        assert row0.idxmin() == "C"   # lowest score row 0

    def test_too_many_names_raises(self, scores):
        with pytest.raises(ValidationError, match="exceeds"):
            equal_weight_top_bottom(scores, n_long=3, n_short=3)

    def test_zero_both_sides_raises(self, scores):
        with pytest.raises(ValidationError, match="n_long"):
            equal_weight_top_bottom(scores, n_long=0, n_short=0)


class TestZscoreNormalized:
    def test_gross_leverage_matches(self, scores):
        w = zscore_normalized(scores, gross_leverage=1.5)
        assert w.abs().sum(axis=1).to_numpy() == pytest.approx([1.5, 1.5, 1.5])

    def test_degenerate_row_is_all_zero(self):
        dates = pd.date_range("2023-01-02", periods=1, freq="B")
        flat = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]}, index=dates)
        w = zscore_normalized(flat)
        assert (w.iloc[0] == 0.0).all()


class TestVolScaled:
    def test_missing_returns_column_raises(self, scores):
        returns_df = pd.DataFrame({"A": [0.01] * 3, "B": [0.01] * 3}, index=scores.index)
        with pytest.raises(ValidationError, match="missing"):
            vol_scaled(scores, returns_df, lookback=2)

    def test_high_vol_name_gets_smaller_weight_for_equal_score(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        equal_scores = pd.DataFrame({"LOWVOL": [1.0] * 5, "HIGHVOL": [1.0] * 5}, index=dates)
        rng = np.random.default_rng(0)
        returns_df = pd.DataFrame(
            {"LOWVOL": rng.normal(0, 0.001, 5), "HIGHVOL": rng.normal(0, 0.05, 5)},
            index=dates,
        )
        w = vol_scaled(equal_scores, returns_df, lookback=3, gross_leverage=1.0)
        last = w.iloc[-1]
        assert abs(last["LOWVOL"]) > abs(last["HIGHVOL"])


class TestDollarNeutral:
    def test_sum_is_zero(self, scores):
        raw = rank_weighted(scores) + 0.1  # shift off-neutral
        neutral = dollar_neutral(raw)
        assert neutral.sum(axis=1).to_numpy() == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_pairwise_differences_preserved(self, scores):
        raw = equal_weight_top_bottom(scores, n_long=1, n_short=1)
        neutral = dollar_neutral(raw)
        diff_before = raw.iloc[0]["B"] - raw.iloc[0]["C"]
        diff_after = neutral.iloc[0]["B"] - neutral.iloc[0]["C"]
        assert diff_after == pytest.approx(diff_before)
