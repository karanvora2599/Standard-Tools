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
    dollar_neutral,
    equal_weight_top_bottom,
    rank_weighted,
    vol_scaled,
    zscore_normalized,
)
from standard_quant_tools.error import ValidationError


@pytest.fixture
def scores():
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, -1.0],
            "B": [2.0, 1.0, 0.0],
            "C": [-1.0, -3.0, 1.0],
            "D": [0.5, 0.0, -0.5],
        },
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
        assert row0.idxmax() == "B"  # highest score row 0
        assert row0.idxmin() == "C"  # lowest score row 0

    def test_too_many_names_raises(self, scores):
        with pytest.raises(ValidationError, match="exceeds"):
            equal_weight_top_bottom(scores, n_long=3, n_short=3)

    def test_zero_both_sides_raises(self, scores):
        with pytest.raises(ValidationError, match="n_long"):
            equal_weight_top_bottom(scores, n_long=0, n_short=0)

    def test_long_only_allocates_full_gross_leverage(self, scores):
        """
        Regression test (high-severity item 2): a long-only request
        (n_short=0) must allocate the FULL gross_leverage across the long
        side, not half of it. The old formula unconditionally used
        gross_leverage / (2 * n_long), silently sizing a long-only
        portfolio at half the requested gross exposure.
        """
        w = equal_weight_top_bottom(scores, n_long=2, n_short=0, gross_leverage=1.0)
        row0 = w.iloc[0]
        assert (row0 < 0).sum() == 0
        assert row0.abs().sum() == pytest.approx(1.0)

    def test_short_only_allocates_full_gross_leverage(self, scores):
        w = equal_weight_top_bottom(scores, n_long=0, n_short=2, gross_leverage=1.0)
        row0 = w.iloc[0]
        assert (row0 > 0).sum() == 0
        assert row0.abs().sum() == pytest.approx(1.0)


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
        returns_df = pd.DataFrame(
            {"A": [0.01] * 3, "B": [0.01] * 3}, index=scores.index
        )
        with pytest.raises(ValidationError, match="missing"):
            vol_scaled(scores, returns_df, lookback=2)

    def test_high_vol_name_gets_smaller_weight_for_equal_score(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        equal_scores = pd.DataFrame(
            {"LOWVOL": [1.0] * 5, "HIGHVOL": [1.0] * 5}, index=dates
        )
        rng = np.random.default_rng(0)
        returns_df = pd.DataFrame(
            {"LOWVOL": rng.normal(0, 0.001, 5), "HIGHVOL": rng.normal(0, 0.05, 5)},
            index=dates,
        )
        w = vol_scaled(equal_scores, returns_df, lookback=3, gross_leverage=1.0)
        last = w.iloc[-1]
        assert abs(last["LOWVOL"]) > abs(last["HIGHVOL"])

    def test_rolling_window_uses_return_frequency_not_score_frequency(self):
        """
        Regression test (high-severity item 2): the rolling volatility
        window must be computed on returns_df's own (daily) frequency
        BEFORE reindexing onto the (possibly sparse) score dates -- not
        reindexed first, which would silently turn a `lookback`-bar window
        into `lookback` SCORE-DATE observations (e.g. ~20 months instead of
        20 days for monthly scores against daily returns).

        Two tickers with IDENTICAL daily volatility for the first ~40 days,
        then LOWVOL's daily vol drops sharply for the last 10 days while
        HIGHVOL's stays the same. Scores are submitted only on the LAST
        bar (a single sparse "monthly" date). With a 10-bar lookback
        computed on the correct (daily) frequency, this last bar's rolling
        window is entirely within the post-day-40 regime, so LOWVOL's
        realized vol is clearly lower there and it must get the larger
        weight. Under the reindex-before-rolling bug, reindexing daily
        returns onto the single sparse score date first collapses
        returns_df to ONE row before rolling(10) ever runs, producing an
        all-NaN std (only one observation) -- so both names would get the
        same all-zero-filled weight instead.
        """
        dates = pd.date_range("2023-01-02", periods=50, freq="B")
        rng = np.random.default_rng(0)
        highvol_returns = rng.normal(0, 0.02, 50)
        lowvol_returns = np.concatenate(
            [rng.normal(0, 0.02, 40), rng.normal(0, 0.001, 10)]
        )
        returns_df = pd.DataFrame(
            {"LOWVOL": lowvol_returns, "HIGHVOL": highvol_returns}, index=dates
        )
        sparse_scores = pd.DataFrame(
            {"LOWVOL": [1.0], "HIGHVOL": [1.0]}, index=[dates[-1]]
        )
        w = vol_scaled(sparse_scores, returns_df, lookback=10, gross_leverage=1.0)
        last = w.iloc[-1]
        assert last.notna().all()
        assert abs(last["LOWVOL"]) > abs(last["HIGHVOL"])


class TestDollarNeutral:
    def test_sum_is_zero(self, scores):
        raw = rank_weighted(scores) + 0.1  # shift off-neutral
        neutral = dollar_neutral(raw)
        assert neutral.sum(axis=1).to_numpy() == pytest.approx(
            [0.0, 0.0, 0.0], abs=1e-9
        )

    def test_pairwise_differences_preserved(self, scores):
        raw = equal_weight_top_bottom(scores, n_long=1, n_short=1)
        neutral = dollar_neutral(raw)
        diff_before = raw.iloc[0]["B"] - raw.iloc[0]["C"]
        diff_after = neutral.iloc[0]["B"] - neutral.iloc[0]["C"]
        assert diff_after == pytest.approx(diff_before)

    def test_gross_leverage_preserved_after_centering(self, scores):
        """
        Regression test (high-severity item 2): dollar_neutral must
        preserve the input panel's own gross exposure (sum(|weight|) per
        row) after mean-centering, not just make sum(weight) == 0.
        Mean-centering alone changes gross exposure whenever the panel
        isn't already symmetric long/short -- every OTHER sizing function
        in this module already centers on its own mean by construction
        (rank_weighted, zscore_normalized) or splits long/short into equal
        magnitudes (equal_weight_top_bottom with both sides active), so
        sum(weight) == 0 for those already and dollar_neutral would be a
        no-op on their output. A long-only panel (n_short=0, all weights
        positive, sum(weight) != 0) is the case that actually exercises
        real centering -- and is exactly the case the old formula's gross
        exposure would silently drift on.
        """
        raw = equal_weight_top_bottom(scores, n_long=2, n_short=0, gross_leverage=1.0)
        original_gross = raw.abs().sum(axis=1)
        neutral = dollar_neutral(raw)
        # Mean-centering must have actually changed something here (not a
        # no-op), otherwise this test wouldn't be exercising the bug.
        assert not neutral.equals(raw)
        assert neutral.sum(axis=1).to_numpy() == pytest.approx(
            [0.0, 0.0, 0.0], abs=1e-9
        )
        assert neutral.abs().sum(axis=1).to_numpy() == pytest.approx(
            original_gross.to_numpy()
        )
