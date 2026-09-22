"""
What these research tools computed and could not report.

WHAT THESE ARE FOR. Most cases below are a per-bar series the library
already computed, summarized into a handful of scalars, and then dropped on
the floor -- so the summary was the only answer available and nothing could
check it:

    the rolling Sharpe          twenty-one statistics described it and the
                                docstring promised it twice
    the GARCH variance path     the fit's own state, used for its last
                                element and thrown away
    the regime labels           on the library's dict, dropped at the
                                handler; `n_switches` said how many times
                                the regime changed and nothing said when
    the Kalman path             875 rows of hedge ratio, intercept, spread
                                and gain, reduced to six numbers
    the PC scores               factor RETURNS, which every return-scoring
                                tool in the library could have read
    the full PCA spectrum       truncated to `n_components` before anyone
                                saw what the truncation cost
    the Amihud rolling series   the percentile and the trend are both
                                comparisons within it
    the basis history           the percentile and the half-life are both
                                statements about it

Each is now published on request, as a reference any runtime resolves. The
reference is OPT-IN -- `run_id` and `name` together, neither by itself --
so the payload is unchanged for a caller who only wants the summary, and
half an address is refused rather than ignored.

The last class here is the same failure in a scalar: the KPSS long-run
variance chose its own bandwidth from the data and reported neither the
choice nor a way to override it, so the statistic could not be reproduced
and the tool's `lags` -- the ADF's augmentation -- looked as if it were the
KPSS's.

See the CHANGELOG entry of 2026-09-22.

THE ANSWERS ARE RECOMPUTED, NOT SHAPE-CHECKED. The rolling Sharpe is
re-derived from the returns with pandas; the labels are compared against the
library's own dict; the Amihud series is counted against `n - window + 1`.
A ref that resolved to the wrong series would pass a shape check and fail
every test here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import (
    GarchVolatilityForecastInput,
    KalmanHedgeRatioInput,
    PCAInput,
    RegimeDetectionInput,
)
from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.data.models import DataSource
from standard_quant_tools.agent.runtimes.delta_one.models import BasisHistoryInput
from standard_quant_tools.agent.runtimes.delta_one.tools import analyze_basis_history
from standard_quant_tools.agent.runtimes.microstructure.estimator_tools import (
    AmihudInput,
    get_amihud_illiquidity,
)
from standard_quant_tools.agent.runtimes.research.diagnostic_tools import (
    SharpeStabilityInput,
    get_sharpe_stability,
)
from standard_quant_tools.agent.runtimes.research.reference_tools import (
    SeriesMetricsInput,
    calculate_series_metrics,
)
from standard_quant_tools.agent.runtimes.research.tools import (
    detect_regimes,
    run_garch_volatility_forecast,
    run_kalman_hedge_ratio,
    run_pca_analysis,
)
from standard_quant_tools.error import ValidationError

START = "2022-01-03"
END = "2023-12-29"

#: Long enough for a 252-day window to leave windows to compare, and for
#: every asymptotic statistic here to have a distribution.
N_BARS = 500


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A private artifact store, so a published ref belongs to one test."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _bars(seed: int, drift: float = 0.0004, vol: float = 0.012) -> pd.DataFrame:
    """One name's OHLCV, its own random walk rather than a shared one."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(START, periods=N_BARS, freq="B")
    close = 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, N_BARS))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": rng.uniform(1e6, 5e6, N_BARS),
        },
        index=index,
    )


#: Three names with genuinely different paths. A provider that answers every
#: symbol with one frame makes a PCA of three assets rank one, which would
#: pass a shape check on the scores and say nothing about them.
_UNIVERSE = {"AAA": _bars(1), "BBB": _bars(2, drift=0.0002), "CCC": _bars(3, vol=0.02)}


def _factor_universe() -> dict:
    """
    Four names driven by ONE common factor, so PC1 dominates.

    Power iteration verifies its own eigenpairs and falls back to SVD when
    two components are weakly separated -- which three independent random
    walks are. A universe with a real factor is what actually exercises the
    iterative path, and `_UNIVERSE` above deliberately does not have one.
    """
    rng = np.random.default_rng(101)
    index = pd.date_range(START, periods=N_BARS, freq="B")
    factor = rng.normal(0.0003, 0.011, N_BARS)
    frames = {}
    for i, ticker in enumerate(["FAC1", "FAC2", "FAC3", "FAC4"]):
        returns = 0.95 * factor + rng.normal(0.0, 0.0015 * (i + 1), N_BARS)
        close = 100.0 * np.cumprod(1.0 + returns)
        frames[ticker] = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.004,
                "Low": close * 0.996,
                "Close": close,
                "Volume": np.full(N_BARS, 2e6),
            },
            index=index,
        )
    return frames


_FACTOR_UNIVERSE = _factor_universe()


def _stub_provider(monkeypatch, frames: dict):
    from unittest.mock import AsyncMock, MagicMock

    from standard_quant_tools.data.factory import DataFactory

    stub = MagicMock()
    stub.get_ohlcv.side_effect = lambda symbol, *a, **kw: frames[symbol]
    stub.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, *a, **kw: frames[symbol]
    )
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: stub)
    return stub


@pytest.fixture
def provider(monkeypatch):
    return _stub_provider(monkeypatch, _UNIVERSE)


@pytest.fixture
def factor_provider(monkeypatch):
    return _stub_provider(monkeypatch, _FACTOR_UNIVERSE)


def _returns(name: str = "AAA") -> pd.Series:
    return _UNIVERSE[name]["Close"].pct_change(fill_method=None).dropna()


# ── the rolling Sharpe ──────────────────────────────────────────────────


class TestRollingSharpeReference:
    WINDOW = 120

    def _call(self, runs_dir, **extra):
        return get_sharpe_stability(
            SharpeStabilityInput(
                returns=[float(r) for r in _returns()],
                window=self.WINDOW,
                periods_per_year=252,
                **extra,
            )
        )

    def test_the_published_series_is_the_rolling_sharpe(self, runs_dir):
        result = self._call(runs_dir, run_id="sharpe_stability", name="rolling")
        assert result.rolling_sharpe_ref == (
            "sqt://analytic_series/sharpe_stability/rolling"
        )
        published = handoff.resolve(result.rolling_sharpe_ref)

        returns = pd.Series([float(r) for r in _returns()])
        expected = (
            returns.rolling(self.WINDOW).mean()
            / returns.rolling(self.WINDOW).std(ddof=1)
            * math.sqrt(252)
        ).dropna()
        expected = expected[np.isfinite(expected)]

        assert len(published) == len(expected) == result.n_windows
        assert np.allclose(published.to_numpy(), expected.to_numpy())

    def test_the_summary_is_a_summary_of_exactly_that_series(self, runs_dir):
        result = self._call(runs_dir, run_id="sharpe_stability", name="rolling")
        published = handoff.resolve(result.rolling_sharpe_ref)
        assert result.mean_rolling_sharpe == pytest.approx(float(published.mean()))
        assert result.min_rolling_sharpe == pytest.approx(float(published.min()))
        assert result.max_rolling_sharpe == pytest.approx(float(published.max()))

    def test_without_an_address_nothing_is_written(self, runs_dir):
        """The reference is opt-in: a caller who wants the summary gets
        exactly the payload it always got, and no artifact appears."""
        result = self._call(runs_dir)
        assert result.rolling_sharpe_ref is None
        store = runs_dir / "runs"
        assert not store.exists() or not list(store.rglob("*.*"))

    def test_half_an_address_is_refused_naming_the_other_half(self, runs_dir):
        with pytest.raises(ValidationError, match="name"):
            self._call(runs_dir, run_id="sharpe_stability")
        with pytest.raises(ValidationError, match="run_id"):
            self._call(runs_dir, name="rolling")

    def test_a_reused_address_fails_loudly(self, runs_dir):
        self._call(runs_dir, run_id="sharpe_stability", name="rolling")
        with pytest.raises(ValidationError, match="already published"):
            self._call(runs_dir, run_id="sharpe_stability", name="rolling")


# ── the GARCH conditional volatility path ───────────────────────────────


class TestConditionalVolatilityReference:
    def test_the_published_series_is_the_root_of_the_fitted_variance(
        self, provider, runs_dir
    ):
        from standard_quant_tools.analysis.garch import garch_volatility_forecast

        result = run_garch_volatility_forecast(
            GarchVolatilityForecastInput(
                symbol="AAA",
                start_date=START,
                end_date=END,
                run_id="garch_fit",
                name="conditional_vol",
            )
        )
        published = handoff.resolve(result.conditional_vol_ref)
        expected = np.sqrt(
            garch_volatility_forecast(_returns())["conditional_variance"]
        )
        assert len(published) == result.n_obs
        assert np.allclose(published.to_numpy(), expected.to_numpy())

    def test_the_reported_volatility_is_one_recursion_step_past_the_path(
        self, provider, runs_dir
    ):
        """
        `current_annualized_vol` is the ONE-STEP-AHEAD forecast, and the
        path's last value is the last OBSERVED bar's conditional
        volatility. They are one recursion apart, not the same number, and
        the path is what makes that checkable.
        """
        result = run_garch_volatility_forecast(
            GarchVolatilityForecastInput(
                symbol="AAA",
                start_date=START,
                end_date=END,
                run_id="garch_fit",
                name="conditional_vol",
            )
        )
        published = handoff.resolve(result.conditional_vol_ref)
        returns = _returns()
        last_shock = float(returns.iloc[-1] - returns.mean()) ** 2
        next_variance = (
            result.omega
            + result.alpha * last_shock
            + result.beta * float(published.iloc[-1]) ** 2
        )
        assert math.sqrt(next_variance * 252) == pytest.approx(
            result.current_annualized_vol, rel=1e-4
        )

    def test_half_an_address_is_refused(self, provider, runs_dir):
        with pytest.raises(ValidationError, match="name"):
            run_garch_volatility_forecast(
                GarchVolatilityForecastInput(
                    symbol="AAA", start_date=START, end_date=END, run_id="garch_fit"
                )
            )
        with pytest.raises(ValidationError, match="run_id"):
            run_garch_volatility_forecast(
                GarchVolatilityForecastInput(
                    symbol="AAA",
                    start_date=START,
                    end_date=END,
                    name="conditional_vol",
                )
            )


# ── the regime labels ───────────────────────────────────────────────────


class TestRegimeLabelReference:
    def test_the_published_labels_are_the_library_s_own(self, provider, runs_dir):
        from standard_quant_tools.analysis.stationarity import (
            detect_regimes as _detect_regimes,
        )

        result = detect_regimes(
            RegimeDetectionInput(
                symbol="AAA",
                start_date=START,
                end_date=END,
                n_regimes=2,
                run_id="regimes",
                name="labels",
            )
        )
        published = handoff.resolve(result.labels_ref)
        expected = _detect_regimes(_returns(), n_regimes=2)["labels"]

        assert [int(v) for v in published] == expected
        assert int(published.iloc[-1]) == result.current_regime
        assert int((published.diff().dropna() != 0).sum()) == result.n_switches

    def test_the_labels_carry_the_dates_they_describe(self, provider, runs_dir):
        result = detect_regimes(
            RegimeDetectionInput(
                symbol="AAA",
                start_date=START,
                end_date=END,
                run_id="regimes",
                name="labels",
            )
        )
        published = handoff.resolve(result.labels_ref)
        assert list(published.index) == list(_returns().index)

    def test_half_an_address_is_refused(self, provider, runs_dir):
        with pytest.raises(ValidationError, match="name"):
            detect_regimes(
                RegimeDetectionInput(
                    symbol="AAA", start_date=START, end_date=END, run_id="regimes"
                )
            )


# ── the Kalman path ─────────────────────────────────────────────────────


class TestKalmanPathReference:
    def _call(self, **extra):
        return run_kalman_hedge_ratio(
            KalmanHedgeRatioInput(
                symbol_a="AAA", symbol_b="BBB", start_date=START, end_date=END, **extra
            )
        )

    def test_the_path_carries_all_four_columns(self, provider, runs_dir):
        result = self._call(run_id="kalman", name="path")
        path = handoff.resolve(result.path_ref)
        assert list(path.columns) == [
            "Hedge_Ratio",
            "Intercept",
            "Spread",
            "Kalman_Gain",
        ]
        assert len(path) == result.n_obs
        assert float(path["Hedge_Ratio"].iloc[-1]) == pytest.approx(
            result.current_hedge_ratio, abs=5e-5
        )
        assert float(path["Hedge_Ratio"].std()) == pytest.approx(
            result.hedge_ratio_std, abs=5e-5
        )

    def test_a_slope_only_fit_says_the_intercept_column_is_all_zero(
        self, provider, runs_dir
    ):
        result = self._call(run_id="kalman", name="path", include_intercept=False)
        path = handoff.resolve(result.path_ref)
        assert (path["Intercept"] == 0.0).all()
        assert any("Intercept" in w and "zero" in w for w in result.warnings)

    def test_a_fitted_intercept_carries_no_such_warning(self, provider, runs_dir):
        result = self._call(run_id="kalman", name="path", include_intercept=True)
        path = handoff.resolve(result.path_ref)
        assert not (path["Intercept"] == 0.0).all()
        assert not any("Intercept" in w for w in result.warnings)

    def test_half_an_address_is_refused(self, provider, runs_dir):
        with pytest.raises(ValidationError, match="run_id"):
            self._call(name="path")


# ── the PC scores and the full spectrum ─────────────────────────────────


class TestPCAReferenceAndSpectrum:
    TICKERS = ["AAA", "BBB", "CCC"]

    def _call(self, n_components=2, **extra):
        return run_pca_analysis(
            PCAInput(
                tickers=self.TICKERS,
                start_date=START,
                end_date=END,
                n_components=n_components,
                **extra,
            )
        )

    def test_the_scores_resolve_as_a_returns_panel(self, provider, runs_dir):
        result = self._call(run_id="pca", name="scores")
        assert result.factor_returns_ref == "sqt://returns_panel/pca/scores"
        panel = handoff.resolve(result.factor_returns_ref, expect="returns_panel")
        assert list(panel.columns) == ["PC1", "PC2"]
        assert len(panel) == result.n_obs

    def test_every_score_column_is_scored_by_the_returns_tool(self, provider, runs_dir):
        """A PC score IS a return series, which is why it goes out as one."""
        result = self._call(run_id="pca", name="scores")
        panel = handoff.resolve(result.factor_returns_ref, expect="returns_panel")
        for column in panel.columns:
            scored = calculate_series_metrics(
                SeriesMetricsInput(
                    series=DataSource(values=[float(v) for v in panel[column]]),
                    metrics=["sharpe_ratio", "annualized_volatility"],
                )
            )
            assert scored.n_observations == len(panel)
            assert math.isfinite(scored.values["annualized_volatility"])

    def test_the_full_spectrum_sums_to_one_and_prefixes_the_kept_ratios(
        self, provider, runs_dir
    ):
        result = self._call(n_components=2)
        assert len(result.explained_variance_ratio_full) == len(self.TICKERS)
        assert sum(result.explained_variance_ratio_full) == pytest.approx(1.0, abs=1e-6)
        kept = [result.explained_variance_ratio[f"PC{i + 1}"] for i in range(2)]
        assert result.explained_variance_ratio_full[:2] == pytest.approx(kept, abs=1e-4)

    def test_the_spectrum_says_what_the_truncation_cost(self, provider, runs_dir):
        """The tail is the point: two components out of three leave a
        remainder, and the number is now readable rather than inferred."""
        result = self._call(n_components=2)
        tail = sum(result.explained_variance_ratio_full[2:])
        assert tail > 0.0
        assert result.cumulative_variance_ratio["PC2"] + tail == pytest.approx(
            1.0, abs=1e-3
        )

    def test_power_iteration_says_why_the_spectrum_is_empty(
        self, factor_provider, runs_dir
    ):
        """
        The iterative path never forms the discarded part of the spectrum,
        which is the reason to use it. An empty list plus a warning naming
        the remedy, rather than a padded vector that would sum to less than
        one with nothing saying why.
        """
        result = run_pca_analysis(
            PCAInput(
                tickers=["FAC1", "FAC2", "FAC3", "FAC4"],
                start_date=START,
                end_date=END,
                n_components=2,
                method="power_iteration",
            )
        )
        assert result.explained_variance_ratio_full == []
        assert any("power_iteration" in w and "svd" in w for w in result.warnings)

    def test_the_same_universe_under_svd_does_report_the_spectrum(
        self, factor_provider, runs_dir
    ):
        result = run_pca_analysis(
            PCAInput(
                tickers=["FAC1", "FAC2", "FAC3", "FAC4"],
                start_date=START,
                end_date=END,
                n_components=2,
                method="svd",
            )
        )
        assert len(result.explained_variance_ratio_full) == 4
        assert sum(result.explained_variance_ratio_full) == pytest.approx(1.0, abs=1e-6)
        assert result.warnings == []

    def test_half_an_address_is_refused(self, provider, runs_dir):
        with pytest.raises(ValidationError, match="name"):
            self._call(run_id="pca")


# ── the Amihud rolling series ───────────────────────────────────────────


class TestAmihudRollingReference:
    WINDOW = 21

    def _call(self, **extra):
        bars = _UNIVERSE["AAA"]
        return get_amihud_illiquidity(
            AmihudInput(
                close=[float(v) for v in bars["Close"]],
                volume=[float(v) for v in bars["Volume"]],
                window=self.WINDOW,
                **extra,
            )
        )

    def test_the_series_has_one_value_per_completed_window(self, runs_dir):
        result = self._call(run_id="amihud", name="rolling")
        published = handoff.resolve(result.rolling_ref)
        assert len(published) == result.n_observations - self.WINDOW + 1

    def test_the_current_reading_is_the_series_last_value(self, runs_dir):
        result = self._call(run_id="amihud", name="rolling")
        published = handoff.resolve(result.rolling_ref)
        assert float(published.iloc[-1]) == pytest.approx(result.current_illiquidity)
        assert float((published < published.iloc[-1]).mean() * 100.0) == pytest.approx(
            result.current_percentile
        )

    def test_the_series_is_not_inlined_when_no_address_is_given(self, runs_dir):
        result = self._call()
        assert result.rolling_ref is None
        # extra="allow" on this result, so a series left on the dict would
        # have arrived as a field rather than as an error.
        assert "rolling" not in result.model_dump()

    def test_half_an_address_is_refused(self, runs_dir):
        with pytest.raises(ValidationError, match="run_id"):
            self._call(name="rolling")


# ── the basis history ───────────────────────────────────────────────────


class TestBasisHistoryReference:
    def _series(self):
        rng = np.random.default_rng(11)
        spot = 4000.0 + np.cumsum(rng.normal(0.0, 3.0, 250))
        futures = spot + 8.0 + rng.normal(0.0, 0.5, 250)
        return [float(v) for v in spot], [float(v) for v in futures]

    def _call(self, **extra):
        spot, futures = self._series()
        return analyze_basis_history(
            BasisHistoryInput(spot_prices=spot, futures_prices=futures, **extra)
        )

    def test_the_history_carries_its_three_series(self, runs_dir):
        result = self._call(run_id="basis", name="history")
        assert result.history_ref == "sqt://analytic_frame/basis/history"
        history = handoff.resolve(result.history_ref)
        assert list(history.columns) == ["basis_points", "basis_bps", "zscore"]
        assert len(history) == result.n_observations

    def test_the_summary_is_a_summary_of_that_frame(self, runs_dir):
        result = self._call(run_id="basis", name="history")
        history = handoff.resolve(result.history_ref)
        assert float(history["basis_bps"].iloc[-1]) == pytest.approx(
            result.current_basis_bps
        )
        assert float(history["basis_points"].iloc[-1]) == pytest.approx(
            result.current_basis_points
        )
        assert float(history["basis_bps"].mean()) == pytest.approx(result.mean_bps)

    def test_an_annualized_history_carries_the_annualized_column_too(self, runs_dir):
        spot, futures = self._series()
        result = analyze_basis_history(
            BasisHistoryInput(
                spot_prices=spot,
                futures_prices=futures,
                time_to_expiry=[0.25] * len(spot),
                run_id="basis",
                name="annualized",
            )
        )
        history = handoff.resolve(result.history_ref)
        assert "annualized_bps" in history.columns
        assert result.annualized is True

    def test_half_an_address_is_refused(self, runs_dir):
        with pytest.raises(ValidationError, match="name"):
            self._call(run_id="basis")


# ── the KPSS bandwidth and the variance-ratio horizons ──────────────────


def _ar1(phi: float, n: int = 400, seed: int = 0) -> pd.Series:
    """A stationary AR(1). Its KPSS null is TRUE, so a rejection is a
    false positive and the rate is measurable."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 1.0, n)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + shocks[t]
    return pd.Series(values)


class TestKpssBandwidthIsChosenAndReported:
    """
    The KPSS statistic is not comparable across bandwidths, and the
    bandwidth was neither settable nor reported: `lags` on the tool is the
    ADF's augmentation, and the KPSS long-run variance quietly picked its
    own. A number that cannot be reproduced is not a test result.
    """

    def _tests(self, series, **kw):
        from standard_quant_tools.analysis.stationarity import (
            run_stationarity_tests as _tests,
        )

        return _tests(series, vr_periods=(2,), **kw)

    def test_a_persistent_stationary_series_over_rejects_at_a_fixed_lag(self):
        """
        The reason the automatic rule is the default, measured rather than
        asserted: 100 stationary AR(1) draws at phi=0.9, where the correct
        rejection rate is 5%.
        """
        automatic = sum(
            self._tests(_ar1(0.9, seed=s))["kpss_rejects_stationarity"]
            for s in range(100)
        )
        fixed = sum(
            self._tests(_ar1(0.9, seed=s), kpss_lags=4)["kpss_rejects_stationarity"]
            for s in range(100)
        )
        assert automatic <= 30, automatic
        assert fixed >= 45, fixed
        assert fixed > automatic * 2

    def test_iid_noise_rejects_at_about_the_nominal_rate_under_either_lag(self):
        """The null case. A truncation that is too short only hurts where
        the autocorrelation extends past it, and on iid noise it does not."""
        for kwargs in ({}, {"kpss_lags": 4}):
            rejections = sum(
                self._tests(
                    pd.Series(np.random.default_rng(2000 + s).normal(0.0, 1.0, 400)),
                    **kwargs,
                )["kpss_rejects_stationarity"]
                for s in range(100)
            )
            assert rejections <= 12, (kwargs, rejections)

    def test_the_reported_bandwidth_is_the_one_andrews_picks(self):
        from standard_quant_tools.analysis.stationarity import andrews_bandwidth

        series = _ar1(0.9, seed=7)
        values = series.to_numpy()
        result = self._tests(series)
        assert result["kpss_lags_source"] == "andrews"
        assert result["kpss_lags_used"] == andrews_bandwidth(values - values.mean())

    def test_a_supplied_bandwidth_is_reported_as_the_caller_s(self):
        result = self._tests(_ar1(0.9, seed=7), kpss_lags=4)
        assert result["kpss_lags_source"] == "caller"
        assert result["kpss_lags_used"] == 4

    def test_a_bandwidth_the_sample_cannot_carry_is_clamped_and_said_so(self):
        short = _ar1(0.5, n=30, seed=3)
        result = self._tests(short, kpss_lags=100)
        assert result["kpss_lags_used"] == 29
        assert any(
            "clamped from 100 to 29" in w and "Andrews" in w for w in result["warnings"]
        )

    def test_the_tool_reports_the_bandwidth_it_ran(self, provider):
        from standard_quant_tools.agent.models import StationarityInput
        from standard_quant_tools.agent.runtimes.research.tools import (
            run_stationarity_tests,
        )

        automatic = run_stationarity_tests(
            StationarityInput(symbol="AAA", start_date=START, end_date=END)
        )
        assert automatic.kpss_lags_source == "andrews"
        assert automatic.kpss_lags_used >= 1

        supplied = run_stationarity_tests(
            StationarityInput(symbol="AAA", start_date=START, end_date=END, kpss_lags=4)
        )
        assert supplied.kpss_lags_source == "caller"
        assert supplied.kpss_lags_used == 4
        assert supplied.kpss_statistic != automatic.kpss_statistic

    def test_the_tool_runs_the_variance_ratio_horizons_it_was_given(self, provider):
        from standard_quant_tools.agent.models import StationarityInput
        from standard_quant_tools.agent.runtimes.research.tools import (
            run_stationarity_tests,
        )

        result = run_stationarity_tests(
            StationarityInput(
                symbol="AAA",
                start_date=START,
                end_date=END,
                vr_periods=[2, 4, 8, 16],
            )
        )
        assert [v.period for v in result.variance_ratios] == [2, 4, 8, 16]

        default = run_stationarity_tests(
            StationarityInput(symbol="AAA", start_date=START, end_date=END)
        )
        assert [v.period for v in default.variance_ratios] == [2, 4, 8]
