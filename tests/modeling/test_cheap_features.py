"""
Four price/volume features that needed no new source, each planted:

- realized semivariance is zero on a series that only rises and the
  downside half of a symmetric series;
- bipower variation stays near zero across a single jump that sends the
  ordinary realized volatility up;
- Amihud illiquidity halves when volume doubles;
- volume surprise is zero at the trailing mean and log 2 at twice it.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.features.base import FeatureContext
from standard_quant_tools.modeling.features.registry import get_feature


def _ohlcv(close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=len(close))
    close = np.asarray(close, dtype=float)
    volume = (
        np.full(len(close), 1_000_000.0)
        if volume is None
        else np.asarray(volume, dtype=float)
    )
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


CONTEXT = FeatureContext(interval="1d")


class TestRealizedSemivariance:
    def test_a_series_that_only_rises_has_none_and_a_symmetric_one_has_half(self):
        rising = np.exp(np.cumsum(np.full(60, 0.01)))
        semivariance = get_feature("risk.realized_semivariance").fn(
            _ohlcv(rising), CONTEXT, period=20
        )
        assert semivariance.iloc[-1] == pytest.approx(0.0)
        assert semivariance.iloc[:20].isna().all()
        steps = np.tile([0.02, -0.02], 30)
        symmetric = np.exp(np.cumsum(steps))
        semivariance = get_feature("risk.realized_semivariance").fn(
            _ohlcv(symmetric), CONTEXT, period=20
        )
        # Half the squared returns are downside: the annualized figure is
        # 0.02 * sqrt(252 / 2).
        assert semivariance.iloc[-1] == pytest.approx(0.02 * np.sqrt(252 / 2), rel=1e-9)


class TestBipowerVariation:
    def test_a_single_jump_moves_realized_variance_but_barely_bipower(self):
        rng = np.random.default_rng(0)
        steps = rng.normal(scale=0.005, size=80)
        smooth = np.exp(np.cumsum(steps))
        jumped_steps = steps.copy()
        jumped_steps[60] = 0.20
        jumped = np.exp(np.cumsum(jumped_steps))
        bipower = get_feature("risk.bipower_variation").fn

        def close_to_close(close):
            r = np.diff(np.log(close))[-20:]
            return float(np.sqrt(np.mean(r**2) * 252))

        smooth_bp = bipower(_ohlcv(smooth), CONTEXT, period=20).iloc[-1]
        jumped_bp = bipower(_ohlcv(jumped), CONTEXT, period=20).iloc[-1]
        smooth_cc, jumped_cc = close_to_close(smooth), close_to_close(jumped)
        # Without a jump the two estimate the same diffusion volatility.
        assert abs(smooth_bp / smooth_cc - 1.0) < 0.35
        # With one, the squared-return estimator explodes and bipower does
        # not: the jump enters it only through two cross products.
        assert jumped_cc > 6 * smooth_cc
        assert jumped_bp < 3 * smooth_bp
        assert jumped_bp < jumped_cc / 2

    def test_it_annualizes_and_refuses_intraday_without_a_calendar(self):
        from standard_quant_tools.error import ValidationError

        close = np.exp(np.cumsum(np.full(40, 0.01)))
        with pytest.raises(ValidationError, match="calendar"):
            get_feature("risk.bipower_variation").fn(
                _ohlcv(close), FeatureContext(interval="1h"), period=20
            )


class TestAmihud:
    def test_twice_the_volume_is_half_the_illiquidity(self):
        close = np.exp(np.cumsum(np.tile([0.01, -0.01], 30)))
        thin = get_feature("volume.amihud_illiquidity").fn(
            _ohlcv(close, np.full(60, 1e5)), CONTEXT, period=20
        )
        thick = get_feature("volume.amihud_illiquidity").fn(
            _ohlcv(close, np.full(60, 2e5)), CONTEXT, period=20
        )
        assert thin.iloc[-1] > 0
        assert thick.iloc[-1] == pytest.approx(thin.iloc[-1] / 2, rel=1e-9)
        # A bar with no volume is left out rather than made infinite.
        volume = np.full(60, 1e5)
        volume[-1] = 0.0
        gapped = get_feature("volume.amihud_illiquidity").fn(
            _ohlcv(close, volume), CONTEXT, period=20
        )
        assert np.isfinite(gapped.iloc[-1])


class TestVolumeSurprise:
    def test_zero_at_the_trailing_mean_and_log_two_at_twice_it(self):
        close = np.linspace(100, 110, 60)
        volume = np.full(60, 1e6)
        volume[-1] = 2e6
        surprise = get_feature("volume.volume_surprise").fn(
            _ohlcv(close, volume), CONTEXT, period=20
        )
        assert surprise.iloc[-2] == pytest.approx(0.0)
        assert surprise.iloc[-1] == pytest.approx(np.log(2.0))
        assert surprise.iloc[:20].isna().all()
        volume[-1] = 0.0
        silent = get_feature("volume.volume_surprise").fn(
            _ohlcv(close, volume), CONTEXT, period=20
        )
        assert np.isnan(silent.iloc[-1])
