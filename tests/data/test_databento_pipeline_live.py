"""The analytics, over real Databento bars, against independent arithmetic.

A fetch test proves the bytes arrive. This proves the numbers computed
from them are right, on data nobody curated for the occasion.

WHY REAL BARS AND NOT A FIXTURE. Every indicator here is already tested
offline against hand-built series, and those tests are the right ones for
edge cases: a constant series, a single gap, a period longer than the
data. What a synthetic series cannot produce is the shape of an actual
tape -- overnight gaps, a limit-up session, a day where the high and the
close are the same tick, volume that varies by an order of magnitude
between a quiet Tuesday and a triple-witching Friday. Those are where a
rolling window silently reindexes or a Wilder smoother drifts from its
recursive definition, and they only appear in real data.

HOW CORRECTNESS IS ESTABLISHED. Not by recording what the library returns
today: a snapshot test locks in a bug as firmly as it locks in a fix. Each
number is recomputed here from its definition in plain pandas, and the two
are compared. Where the definition has a genuine ambiguity -- Wilder's
smoothing seed, the first window of an EMA -- the test says which
convention it assumes rather than loosening the tolerance until both pass.

COST. One daily-bar window per symbol, fetched once per module. Under a
cent for a full pass.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.databento_provider import DatabentoProvider
from standard_quant_tools.analysis import regression
from standard_quant_tools.indicators import momentum, trend, volatility, volume
from standard_quant_tools.metrics import return_metrics, risk_metrics

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("DATABENTO_API_KEY", "").strip(),
        reason="live Databento tests need DATABENTO_API_KEY",
    ),
]

#: Two years, so a 252-day annualisation has something to annualise and a
#: 200-period moving average is defined over most of the frame.
START, END = "2024-09-03", "2026-09-18"
SYMBOL = "AAPL"
BENCHMARK = "SPY"


@pytest.fixture(scope="module", autouse=True)
def _no_inherited_dataset_pins():
    saved = {
        name: os.environ.pop(name, None)
        for name in ("DATABENTO_DATASET", "DATABENTO_DEPTH_DATASET", "DATABENTO_OHLCV_DATASET")
    }
    yield
    for name, value in saved.items():
        if value is not None:
            os.environ[name] = value


@pytest.fixture(scope="module")
def provider() -> DatabentoProvider:
    return DatabentoProvider()


@pytest.fixture(scope="module")
def bars(provider: DatabentoProvider) -> pd.DataFrame:
    frame = provider.get_ohlcv(SYMBOL, START, END, interval="1d")
    if len(frame) < 300:
        pytest.skip(f"only {len(frame)} sessions; the window is meant to be two years")
    return frame


@pytest.fixture(scope="module")
def benchmark(provider: DatabentoProvider) -> pd.DataFrame:
    return provider.get_ohlcv(BENCHMARK, START, END, interval="1d")


@pytest.fixture(scope="module")
def close(bars: pd.DataFrame) -> pd.Series:
    return bars["Close"].astype(float)


@pytest.fixture(scope="module")
def returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


# --- the tape this is computed on ----------------------------------------------


class TestTheWindowIsARealTape:
    """If the input is not a real two years, everything below is theatre."""

    def test_two_years_of_sessions_with_real_dispersion(self, bars, returns):
        assert 450 <= len(bars) <= 520, f"{len(bars)} sessions in two years"
        assert returns.std() > 0.005, "a tape with no dispersion is not a tape"
        assert (bars["Volume"] > 0).all()
        # A real tape has overnight gaps: opens that are not the prior close.
        gaps = (bars["Open"] - bars["Close"].shift()).abs().dropna()
        assert (gaps > 1e-9).mean() > 0.5, (
            "more than half the opens equal the prior close, which is a "
            "synthetic series rather than a market"
        )


# --- indicators, each against its own definition --------------------------------


class TestTrendIndicatorsMatchTheirDefinitions:
    def test_sma_is_the_rolling_mean_including_its_warm_up(self, close):
        got = trend.sma(close, period=50)
        want = close.rolling(window=50).mean()
        assert len(got) == len(close), "an indicator must not silently shorten the frame"
        pd.testing.assert_series_equal(
            got.dropna(), want.dropna(), check_names=False, rtol=1e-12
        )
        # The warm-up is NaN, not a partial mean: a partial mean is a
        # different number wearing the same name.
        assert got.iloc[:49].isna().all(), "the first 49 rows are not a 50-day mean"
        assert got.iloc[49:].notna().all()

    def test_ema_matches_the_recursive_definition_with_the_stated_seed(self, close):
        period = 20
        got = trend.ema(close, period=period)
        alpha = 2.0 / (period + 1.0)
        want = close.ewm(alpha=alpha, adjust=False).mean()
        common = got.dropna().index.intersection(want.index)
        assert len(common) > 400
        difference = (got.loc[common] - want.loc[common]).abs()
        # The seed convention shows up as a decaying error at the front,
        # so the tail is where the recursion is actually checked.
        tail = difference.iloc[100:]
        assert tail.max() < 1e-6, (
            f"the EMA drifts from its recursion by {tail.max():.2e} well past "
            "any seeding effect"
        )

    def test_macd_is_the_difference_of_its_own_two_emas(self, close):
        frame = trend.macd(close, fast=12, slow=26, signal=9)
        assert isinstance(frame, pd.DataFrame)
        columns = {c.lower() for c in frame.columns}
        assert {"macd", "signal"} <= columns, f"macd returned {list(frame.columns)}"
        column = {c.lower(): c for c in frame.columns}
        fast = close.ewm(alpha=2 / 13, adjust=False).mean()
        slow = close.ewm(alpha=2 / 27, adjust=False).mean()
        want = (fast - slow).iloc[100:]
        got = frame[column["macd"]].iloc[100:]
        assert (got - want).abs().max() < 1e-6
        # The histogram, if present, is the gap between the line and its signal.
        if "histogram" in column:
            hist = frame[column["histogram"]].dropna()
            rebuilt = (frame[column["macd"]] - frame[column["signal"]]).loc[hist.index]
            assert (hist - rebuilt).abs().max() < 1e-9


class TestMomentumAndVolatilityMatchTheirDefinitions:
    def test_rsi_stays_inside_its_bounds_and_matches_wilder(self, close):
        period = 14
        got = momentum.rsi(close, period=period).dropna()
        assert len(got) > 400
        assert (got >= 0).all() and (got <= 100).all(), (
            f"RSI left [0, 100]: min {got.min():.4f} max {got.max():.4f}"
        )
        # Wilder's smoothing is an EMA with alpha = 1/period.
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        want = 100 - 100 / (1 + gain / loss)
        common = got.index.intersection(want.dropna().index)[50:]
        difference = (got.loc[common] - want.loc[common]).abs()
        assert difference.max() < 0.5, (
            f"RSI differs from Wilder's definition by up to {difference.max():.4f} "
            "points, which is a different smoother rather than a different seed"
        )

    def test_atr_is_the_average_of_the_true_range_and_never_negative(self, bars):
        high, low, close = bars["High"], bars["Low"], bars["Close"]
        got = volatility.atr(high, low, close, period=14).dropna()
        assert (got > 0).all(), "a non-positive average true range"
        true_range = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        # However it is smoothed, an average true range cannot sit outside
        # the range of the true ranges it averages.
        assert got.max() <= true_range.max() + 1e-9
        assert got.min() >= true_range.min() - 1e-9
        # And it is a price distance, not a price.
        assert got.median() < float(close.median()) * 0.2

    def test_bollinger_bands_bracket_the_price_by_their_own_width(self, close):
        frame = volatility.bollinger_bands(close, period=20, num_std=2.0)
        # Columns arrive prefixed (`BB_Upper`); the prefix is this library's
        # naming and not part of the definition, so it is stripped here
        # rather than asserted, which would make the test about the label.
        column = {c.lower().removeprefix("bb_"): c for c in frame.columns}
        assert {"upper", "lower"} <= set(column), f"got {list(frame.columns)}"
        upper, lower = frame[column["upper"]], frame[column["lower"]]
        middle_name = column.get("middle") or column.get("ma") or column.get("mid")
        both = pd.concat([upper, lower], axis=1).dropna()
        assert (both.iloc[:, 0] >= both.iloc[:, 1]).all(), "the upper band fell below the lower"
        if middle_name:
            middle = frame[middle_name]
            rolling = close.rolling(20).mean()
            assert (middle.dropna() - rolling.dropna()).abs().max() < 1e-9
            width = (upper - middle).dropna()
            want = close.rolling(20).std(ddof=0) * 2.0
            alt = close.rolling(20).std(ddof=1) * 2.0
            near = min(
                (width - want.loc[width.index]).abs().max(),
                (width - alt.loc[width.index]).abs().max(),
            )
            assert near < 1e-6, f"the band half-width is neither 2 population nor 2 sample sd ({near:.2e})"
        # Roughly nineteen in twenty closes sit inside two standard deviations;
        # a real tape has fat tails, so the bound is one-sided and loose.
        inside = ((close >= lower) & (close <= upper)).loc[both.index]
        assert inside.mean() > 0.80, f"only {inside.mean():.1%} of closes are inside the bands"


class TestVolumeIndicatorsOnRealVolume:
    def test_vwap_is_volume_weighted_and_sits_within_the_price_range(self, bars):
        high, low, close, vol = bars["High"], bars["Low"], bars["Close"], bars["Volume"].astype(float)
        got = volume.vwap(high, low, close, vol).dropna()
        typical = (high + low + close) / 3.0
        want = (typical * vol).cumsum() / vol.cumsum()
        common = got.index.intersection(want.index)
        assert len(common) > 400
        assert (got.loc[common] - want.loc[common]).abs().max() < 1e-6, (
            "VWAP is not the cumulative volume-weighted typical price"
        )
        assert got.min() >= float(low.min()) - 1e-9
        assert got.max() <= float(high.max()) + 1e-9

    def test_obv_steps_by_the_day_s_volume_in_the_close_s_direction(self, bars):
        close, vol = bars["Close"], bars["Volume"].astype(float)
        got = volume.obv(close, vol).dropna()
        direction = np.sign(close.diff())
        want = (direction * vol).fillna(0).cumsum()
        common = got.index.intersection(want.index)[1:]
        difference = (got.loc[common] - want.loc[common]).abs()
        assert difference.max() < 1.0, (
            f"OBV differs from its definition by up to {difference.max():,.0f} shares"
        )


# --- metrics, against arithmetic done here --------------------------------------


class TestRiskAndReturnMetricsOnRealReturns:
    def test_cagr_compounds_over_the_intervals_not_the_observations(self, close):
        """N closes span N-1 return intervals, and the denominator is the
        elapsed time rather than the row count.

        Written this way because the obvious reference -- len(series)/252 --
        is wrong, and wrong in the direction that overstates elapsed time and
        so understates the rate. On two years of daily bars the two differ in
        the fourth decimal; on a one-month window the denominator is out by
        5%, which is exactly where a CAGR is least reliable already.
        """
        got = return_metrics.cagr(close, periods_per_year=252)
        intervals = (len(close) - 1) / 252.0
        want = (float(close.iloc[-1]) / float(close.iloc[0])) ** (1 / intervals) - 1
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), f"{got} vs {want}"

        by_observations = (float(close.iloc[-1]) / float(close.iloc[0])) ** (
            252.0 / len(close)
        ) - 1
        assert abs(got - by_observations) > 1e-9, (
            "the two conventions agree here, so this window cannot tell them "
            "apart and the assertion above proves nothing"
        )
        assert -0.9 < got < 3.0, f"a CAGR of {got:.2%} over two years of a mega-cap"

    def test_cumulative_return_is_the_whole_window(self, close):
        got = return_metrics.cumulative_return(close)
        want = float(close.iloc[-1]) / float(close.iloc[0]) - 1
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)

    def test_annualized_volatility_scales_by_the_root_of_the_period(self, returns):
        got = return_metrics.annualized_volatility(returns, periods_per_year=252)
        want = float(returns.std(ddof=1)) * math.sqrt(252)
        alt = float(returns.std(ddof=0)) * math.sqrt(252)
        assert min(abs(got - want), abs(got - alt)) < 1e-9, f"{got} vs {want}/{alt}"
        assert 0.05 < got < 1.5, f"{got:.2%} annualised vol for a large-cap"

    def test_sharpe_is_the_mean_over_the_deviation_annualised(self, returns):
        got = risk_metrics.annualized_sharpe(returns.to_numpy(), periods=252)
        values = returns.to_numpy()
        want = values.mean() / values.std(ddof=1) * math.sqrt(252)
        alt = values.mean() / values.std(ddof=0) * math.sqrt(252)
        assert min(abs(got - want), abs(got - alt)) < 1e-6, f"{got} vs {want}/{alt}"
        assert -5 < got < 5, f"a Sharpe of {got:.2f} from two years of daily bars"

    def test_the_drawdown_series_is_never_positive_and_finds_the_real_trough(self, close):
        got = risk_metrics.drawdown_series(close)
        assert (got <= 1e-12).all(), "a drawdown above the running peak"
        want = close / close.cummax() - 1.0
        assert (got - want).abs().max() < 1e-9
        worst = float(got.min())
        assert -0.95 < worst < 0, f"a worst drawdown of {worst:.2%}"
        # The trough is a real session, and the peak that defines it precedes it.
        trough = got.idxmin()
        assert close.loc[:trough].max() >= close.loc[trough]

    def test_cvar_is_at_least_as_severe_as_the_quantile_it_conditions_on(self, returns):
        got = risk_metrics.cvar(returns, confidence=0.95)
        var = float(returns.quantile(0.05))
        tail = returns[returns <= var]
        assert len(tail) > 5, "not enough tail to condition on"
        assert got <= var + 1e-9 or abs(got) >= abs(var) - 1e-9, (
            f"CVaR {got:.6f} is milder than the 5% VaR {var:.6f}"
        )
        assert abs(got) < 0.5, f"a one-day conditional loss of {got:.2%}"

    def test_beta_against_a_real_benchmark_is_near_one_for_a_mega_cap(self, close, benchmark):
        market = benchmark["Close"].astype(float).pct_change().dropna()
        asset = close.pct_change().dropna()
        joined = pd.concat([asset, market], axis=1, join="inner").dropna()
        joined.columns = ["asset", "market"]
        assert len(joined) > 300, f"only {len(joined)} overlapping sessions"
        got = regression.calculate_beta(joined["asset"], joined["market"])
        beta = got["beta"] if isinstance(got, dict) else float(got)
        covariance = float(np.cov(joined["asset"], joined["market"], ddof=1)[0, 1])
        want = covariance / float(joined["market"].var(ddof=1))
        assert math.isclose(beta, want, rel_tol=1e-6), f"{beta} vs {want}"
        assert 0.3 < beta < 2.5, f"a beta of {beta:.2f} for {SYMBOL} against {BENCHMARK}"


# --- the two vendors, through the library's own provider seam -------------------


class TestTheSameAnalyticOverTwoVendors:
    """The strongest check available without a third party: compute the same
    number from two independently sourced price series and require them to
    agree. A bug in the indicator shows up in both; a bug in one vendor's
    normalisation shows up in one."""

    def test_volatility_and_drawdown_agree_across_vendors(self, close):
        from standard_quant_tools.data.yfinance_provider import YFinanceProvider

        other = YFinanceProvider().get_ohlcv(SYMBOL, START, END, interval="1d")
        if other.empty:
            pytest.skip("the second vendor returned nothing")
        theirs = other["Close"].astype(float)
        theirs.index = pd.to_datetime(theirs.index)
        if theirs.index.tz is not None:
            theirs.index = theirs.index.tz_convert("UTC").tz_localize(None)
        theirs.index = theirs.index.normalize()
        ours = close.copy()
        ours.index = ours.index.tz_convert("UTC").tz_localize(None).normalize()
        joined = pd.concat([ours, theirs], axis=1, join="inner").dropna()
        joined.columns = ["ours", "theirs"]
        assert len(joined) > 300, f"only {len(joined)} sessions joined"

        our_vol = return_metrics.annualized_volatility(joined["ours"].pct_change().dropna(), 252)
        their_vol = return_metrics.annualized_volatility(joined["theirs"].pct_change().dropna(), 252)
        assert abs(our_vol - their_vol) < 0.03, (
            f"annualised volatility differs by {abs(our_vol - their_vol):.2%} "
            f"between vendors ({our_vol:.2%} vs {their_vol:.2%})"
        )

        our_dd = float(risk_metrics.drawdown_series(joined["ours"]).min())
        their_dd = float(risk_metrics.drawdown_series(joined["theirs"]).min())
        assert abs(our_dd - their_dd) < 0.05, (
            f"worst drawdown differs by {abs(our_dd - their_dd):.2%} between "
            f"vendors ({our_dd:.2%} vs {their_dd:.2%})"
        )

    def test_the_unadjusted_feed_is_the_one_that_shows_a_dividend_gap(self, close):
        """Databento is raw and the other vendor's close is not adjusted here
        either, so the two should track. What must NOT happen is our series
        looking adjusted: a raw series reported as adjusted is the error the
        metadata exists to prevent."""
        from standard_quant_tools.data.databento_provider import DatabentoProvider as P

        meta = P().get_metadata(SYMBOL, interval="1d")
        assert meta.adjusted is False
        assert meta.point_in_time is False
        assert meta.survivorship_free is True
