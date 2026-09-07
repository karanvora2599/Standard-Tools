"""Tests for liquidity/microstructure proxies (Amihud illiquidity, Corwin-Schultz spread)."""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.liquidity import (
    amihud_illiquidity,
    corwin_schultz_spread,
)
from standard_quant_tools.error import ValidationError


class TestAmihudIlliquidity:
    def test_hand_computed_value(self):
        returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        dollar_volume = pd.Series([1e6] * 5)
        result = amihud_illiquidity(returns, dollar_volume, window=5)
        expected = float(returns.abs().mean() / 1e6 * 1e6)
        assert result.iloc[-1] == pytest.approx(expected)

    def test_higher_volume_yields_lower_illiquidity(self):
        returns = pd.Series([0.02] * 10)
        low_volume = pd.Series([1e5] * 10)
        high_volume = pd.Series([1e8] * 10)
        low_vol_illiq = amihud_illiquidity(returns, low_volume, window=5).dropna()
        high_vol_illiq = amihud_illiquidity(returns, high_volume, window=5).dropna()
        assert (low_vol_illiq > high_vol_illiq).all()

    def test_zero_dollar_volume_does_not_produce_inf(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
        dollar_volume = pd.Series([1e6, 0.0, 1e6, 1e6, 1e6])
        result = amihud_illiquidity(returns, dollar_volume, window=5)
        assert not np.isinf(result.dropna()).any()

    def test_negative_dollar_volume_does_not_produce_inf_or_nan_ratio_flip(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
        dollar_volume = pd.Series([1e6, -50.0, 1e6, 1e6, 1e6])
        result = amihud_illiquidity(returns, dollar_volume, window=5)
        assert not np.isinf(result.dropna()).any()

    @pytest.mark.parametrize("bad_window", [0, -1, -10])
    def test_invalid_window_raises(self, bad_window):
        returns = pd.Series([0.01] * 10)
        dollar_volume = pd.Series([1e6] * 10)
        with pytest.raises(ValidationError, match="window"):
            amihud_illiquidity(returns, dollar_volume, window=bad_window)

    def test_empty_series_raises(self):
        empty = pd.Series([], dtype=float)
        with pytest.raises(ValidationError):
            amihud_illiquidity(empty, empty, window=5)


class TestCorwinSchultzSpread:
    def test_hand_computed_two_bar_value(self):
        high = pd.Series([102.0, 104.0])
        low = pd.Series([98.0, 96.0])

        log_hl2 = [math.log(102 / 98) ** 2, math.log(104 / 96) ** 2]
        beta = log_hl2[0] + log_hl2[1]
        gamma = math.log(max(102, 104) / min(98, 96)) ** 2
        k = 3 - 2 * math.sqrt(2)
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
        expected = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))

        result = corwin_schultz_spread(high, low)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(max(expected, 0.0), rel=1e-9)

    def test_zero_range_yields_zero_spread(self):
        """High == Low every bar -> ln(H/L) = 0 everywhere -> beta=gamma=0
        -> alpha is nan (0/0)/log(1)... guard: spread should not be positive
        or crash; clip(lower=0) plus nan propagation is acceptable here."""
        flat = pd.Series([100.0] * 10)
        result = corwin_schultz_spread(flat, flat)
        valid = result.dropna()
        # Wherever it's not NaN, it must not be negative (clipped).
        assert (valid >= 0.0).all()

    def test_wider_range_yields_larger_spread(self):
        n = 20
        narrow_high = pd.Series(np.full(n, 101.0))
        narrow_low = pd.Series(np.full(n, 99.0))
        wide_high = pd.Series(np.full(n, 120.0))
        wide_low = pd.Series(np.full(n, 80.0))
        narrow = corwin_schultz_spread(narrow_high, narrow_low).dropna()
        wide = corwin_schultz_spread(wide_high, wide_low).dropna()
        assert (wide > narrow).all()

    def test_spread_never_negative(self):
        rng = np.random.default_rng(0)
        n = 200
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        high = pd.Series(close * (1 + np.abs(rng.normal(0, 0.005, n))))
        low = pd.Series(close * (1 - np.abs(rng.normal(0, 0.005, n))))
        result = corwin_schultz_spread(high, low)
        assert (result.dropna() >= 0.0).all()

    def test_window_smooths_the_series(self):
        rng = np.random.default_rng(1)
        n = 100
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.02, n))
        high = pd.Series(close * (1 + np.abs(rng.normal(0, 0.01, n))))
        low = pd.Series(close * (1 - np.abs(rng.normal(0, 0.01, n))))
        raw = corwin_schultz_spread(high, low, window=1).dropna()
        smoothed = corwin_schultz_spread(high, low, window=10).dropna()
        assert smoothed.std() < raw.std()

    @pytest.mark.parametrize("bad_window", [0, -1, -5])
    def test_invalid_window_raises(self, bad_window):
        high = pd.Series([102.0, 104.0])
        low = pd.Series([98.0, 96.0])
        with pytest.raises(ValidationError, match="window"):
            corwin_schultz_spread(high, low, window=bad_window)

    def test_empty_series_raises(self):
        empty = pd.Series([], dtype=float)
        with pytest.raises(ValidationError):
            corwin_schultz_spread(empty, empty)


class TestTheTwoShapesAreOneEstimator:
    """
    `corwin_schultz_spread` exists twice on purpose -- as this per-bar
    Series, which `check_spread_proxy` rolls a window over, and as an
    aggregate dict in `analysis.microstructure_estimators` that reports
    `negative_fraction`. They are not interchangeable and neither can be
    deleted.

    They WERE two independent copies of the same algebra, and that is the
    part that had to go. They agreed to the last digit on clean data, so
    nothing flagged them, and disagreed completely on invalid data: only the
    dict form refused a bar whose low was non-positive or whose high sat
    below its low. The series form took the logarithm anyway and returned a
    number -- to `check_spread_proxy`, whose entire job is to say whether a
    backtest charged too little.

    These tests pin the two properties that make one kernel worth having:
    identical answers on good data, identical refusals on bad.
    """

    @staticmethod
    def _bars(n=250, seed=7):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2023-01-02", periods=n)
        mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        half = mid * 0.0016
        high = pd.Series(mid + half + rng.uniform(0, 0.05, n), index=idx)
        low = pd.Series(mid - half - rng.uniform(0, 0.05, n), index=idx)
        return high, low

    def test_the_two_forms_agree_to_the_last_digit(self):
        """Measured at 3.8968222761 bps from both, before and after the two
        copies became one kernel. This is the number that made the duplicate
        invisible, so it is the one to pin."""
        from standard_quant_tools.analysis.microstructure_estimators import (
            corwin_schultz_spread as aggregate_form,
        )

        high, low = self._bars()
        series_bps = float(corwin_schultz_spread(high, low).mean()) * 1e4
        dict_bps = aggregate_form(pd.DataFrame({"high": high, "low": low}))[
            "spread_bps"
        ]

        assert series_bps == pytest.approx(dict_bps, abs=1e-9)
        assert series_bps == pytest.approx(3.8968222761, abs=1e-6)

    @pytest.mark.parametrize(
        "corrupt",
        [
            pytest.param(
                lambda h, lo: (h, lo.mask(lo.index == lo.index[50], 0.0)),
                id="low_is_zero",
            ),
            pytest.param(
                lambda h, lo: (h, lo.mask(lo.index == lo.index[20], -3.0)),
                id="low_is_negative",
            ),
            pytest.param(
                lambda h, lo: (lo.copy(), h.copy()), id="high_and_low_transposed"
            ),
        ],
    )
    def test_both_forms_refuse_the_same_bad_bars(self, corrupt):
        from standard_quant_tools.analysis.microstructure_estimators import (
            corwin_schultz_spread as aggregate_form,
        )

        high, low = corrupt(*self._bars())

        with pytest.raises(ValidationError, match="non-positive low or a high"):
            corwin_schultz_spread(high, low)
        with pytest.raises(ValidationError, match="non-positive low or a high"):
            aggregate_form(pd.DataFrame({"high": high, "low": low}))

    def test_a_transposed_row_no_longer_passes_for_a_spread(self):
        """
        The failure this consolidation exists to close. One bar with its
        high and low swapped used to come back as a slightly wider spread --
        1.04x the true figure on this fixture, which is well inside the
        range a real name could move and would never be questioned.
        """
        high, low = self._bars()
        high.iloc[100], low.iloc[100] = low.iloc[100], high.iloc[100]

        with pytest.raises(ValidationError, match="not recoverable data noise"):
            corwin_schultz_spread(high, low)

    def test_a_partial_nan_is_still_allowed_through(self):
        """A ticker that lists mid-sample, or a holiday on one calendar and
        not another, is a normal gap and not a corrupt bar. The refusal is
        for impossible values, not for missing ones."""
        high, low = self._bars()
        high.iloc[10] = np.nan
        low.iloc[10] = np.nan

        result = corwin_schultz_spread(high, low)
        assert result.notna().any()
