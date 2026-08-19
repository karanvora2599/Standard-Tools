"""
Regression tests for Pass 2 of the full-codebase audit: the shared numerical
input contract.

The audit's own diagnosis was that ~40 of its findings were one problem
wearing different clothes — `@validate_series` checked emptiness and nothing
else (its all-NaN check sat in the body as commented-out code, and there was
no infinity check at all), so every metric wearing it had its own accidental
behaviour for the same invalid input. These tests pin the shared contract
rather than each symptom, plus the specific places where a missing check
produced a confidently wrong number.
"""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import (
    require_aligned,
    require_finite_covariance,
    require_finite_scalar,
    require_finite_series,
    require_periods_per_year,
    require_positive_int,
    require_positive_price_series,
    require_positive_start_level,
)

IDX = pd.date_range("2023-01-02", periods=60, freq="B")


def _clean(seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0.0005, 0.01, 60), index=IDX)


class TestSharedSeriesContract:
    """
    Before this, the SAME invalid input produced four different behaviours
    across four metrics:

        sharpe_ratio(all-NaN)       -> nan
        sortino_ratio(all-NaN)      -> +inf   (reads as "no losing bars")
        var_historical(all-NaN)     -> IndexError
        max_drawdown(contains inf)  -> -1.703437775179145

    The last one is why this belongs in the shared decorator: an infinity does
    not stay visibly wrong. It came back as a drawdown that looks measured.
    """

    @pytest.mark.parametrize(
        "metric", ["sharpe_ratio", "sortino_ratio", "max_drawdown", "var_historical"]
    )
    def test_all_nan_rejected_uniformly(self, metric):
        import standard_quant_tools.metrics.risk_metrics as RM

        with pytest.raises(ValidationError, match="no observations"):
            getattr(RM, metric)(pd.Series([np.nan] * 60, index=IDX))

    @pytest.mark.parametrize(
        "metric", ["sharpe_ratio", "sortino_ratio", "max_drawdown", "var_historical"]
    )
    def test_infinity_rejected_uniformly(self, metric):
        import standard_quant_tools.metrics.risk_metrics as RM

        bad = _clean()
        bad.iloc[5] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
            getattr(RM, metric)(bad)

    def test_partial_nan_is_still_allowed(self):
        """
        The deliberate limit of the contract. Warm-up windows, a ticker that
        lists mid-sample and a benchmark on a different holiday calendar all
        produce legitimate gaps, and many callers drop them internally on
        purpose — making partial NaN fatal would break correct code to catch
        a problem it has already handled.
        """
        import standard_quant_tools.metrics.risk_metrics as RM

        partial = _clean()
        partial.iloc[3] = np.nan
        assert np.isfinite(RM.sharpe_ratio(partial))

    def test_clean_input_is_unaffected(self):
        import standard_quant_tools.metrics.risk_metrics as RM

        assert np.isfinite(RM.sharpe_ratio(_clean()))


class TestSortinoNoLongerConflatesTwoStates:
    def test_genuinely_no_downside_is_still_positive_infinity(self):
        """A strategy that never lost has an infinite Sortino. That is a real
        answer and must survive."""
        from standard_quant_tools.metrics.risk_metrics import sortino_ratio

        assert sortino_ratio(pd.Series([0.01] * 60, index=IDX)) == np.inf

    def test_unusable_input_no_longer_reports_infinite_sortino(self):
        """
        +inf used to mean both "no losing bars" and "I could not compute the
        deviation" — the single most flattering possible misreading of bad
        data.
        """
        from standard_quant_tools.metrics.risk_metrics import sortino_ratio

        with pytest.raises(ValidationError):
            sortino_ratio(pd.Series([np.nan] * 60, index=IDX))


class TestLevelSeriesNeedAPositiveStart:
    """
    Weaker than the price-series rule, deliberately. A leveraged position CAN
    be wiped out, so an equity curve legitimately reaches zero or goes
    negative at its tail. What must hold is that the DENOMINATOR is positive:
    cumulative_return divides by the first value, and the drawdown ratio
    divides by a running maximum seeded from it.
    """

    def test_wiped_out_curve_remains_supported(self):
        from standard_quant_tools.metrics.return_metrics import cagr, cumulative_return

        wiped = pd.Series([10000.0, 8000.0, 3000.0, -500.0, -800.0])
        assert cumulative_return(wiped) < -1.0
        assert cagr(wiped) == -1.0

    @pytest.mark.parametrize("start", [0.0, -100.0])
    def test_non_positive_start_rejected(self, start):
        from standard_quant_tools.metrics.return_metrics import cumulative_return

        series = pd.Series([start, 100.0, 120.0])
        with pytest.raises(ValidationError, match="denominator"):
            cumulative_return(series)

    def test_drawdown_rejects_a_non_positive_open(self):
        from standard_quant_tools.metrics.risk_metrics import max_drawdown

        with pytest.raises(ValidationError, match="denominator"):
            max_drawdown(pd.Series([-50.0, 100.0, 120.0]))


class TestAnnualizationFactor:
    """
    periods_per_year is a bare multiplier, so an invalid value produced a
    confidently wrong number rather than an error: -252 returned a CAGR of
    -0.5350151890419428, which reads as an ordinary annual loss.
    """

    @pytest.mark.parametrize("bad", [0, -252, 2.5, float("nan")])
    def test_invalid_factor_rejected(self, bad):
        from standard_quant_tools.metrics.return_metrics import cagr

        equity = pd.Series(np.linspace(10000, 12000, 60), index=IDX)
        with pytest.raises(ValidationError, match="periods_per_year"):
            cagr(equity, periods_per_year=bad)

    def test_sharpe_and_volatility_check_it_too(self):
        from standard_quant_tools.metrics.return_metrics import annualized_volatility
        from standard_quant_tools.metrics.risk_metrics import sharpe_ratio

        with pytest.raises(ValidationError, match="periods_per_year"):
            sharpe_ratio(_clean(), periods_per_year=0)
        with pytest.raises(ValidationError, match="periods_per_year"):
            annualized_volatility(_clean(), periods_per_year=-252)


class TestCagrCountsIntervalsNotObservations:
    def test_elapsed_span_uses_n_minus_one(self):
        """N levels contain N-1 returns. Using N overstated the elapsed time
        and so understated the growth rate — negligible over a decade,
        material on short windows."""
        from standard_quant_tools.metrics.return_metrics import cagr, cumulative_return

        equity = pd.Series(np.linspace(100.0, 200.0, 22))
        expected = (1 + cumulative_return(equity)) ** (1 / (21 / 252)) - 1
        assert cagr(equity) == pytest.approx(expected, rel=1e-12)

    def test_single_observation_spans_no_time(self):
        from standard_quant_tools.metrics.return_metrics import cagr

        assert cagr(pd.Series([100.0])) == 0.0


class TestPricesMustBePositive:
    """
    run_strategy checked finiteness only, and 0.0 and -5.0 are perfectly
    finite. A single -5.0 close produced a total_return of +0.397914 — a
    plausible profit computed through a negative price — while a 0.0 close
    produced a silent -1.0 wipeout.
    """

    def _frame(self, close):
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
            index=close.index,
        )

    @pytest.mark.parametrize("bad_value", [0.0, -5.0])
    def test_non_positive_close_rejected(self, bad_value):
        from standard_quant_tools.backtest.engine import run_strategy

        close = pd.Series(np.linspace(100.0, 140.0, 60), index=IDX)
        close.iloc[10] = bad_value
        with pytest.raises(ValidationError, match="non-positive"):
            run_strategy(self._frame(close), pd.Series(1.0, index=IDX))

    def test_normal_prices_still_run(self):
        from standard_quant_tools.backtest.engine import run_strategy

        close = pd.Series(np.linspace(100.0, 140.0, 60), index=IDX)
        assert (
            run_strategy(self._frame(close), pd.Series(1.0, index=IDX))["total_return"]
            > 0
        )

    def test_disjoint_price_and_signal_calendars_rejected(self):
        from standard_quant_tools.backtest.engine import run_strategy

        close = pd.Series(np.linspace(100.0, 140.0, 60), index=IDX)
        other = pd.Series(1.0, index=pd.date_range("2030-01-02", periods=60, freq="B"))
        with pytest.raises(ValidationError, match="share no dates"):
            run_strategy(self._frame(close), other)


class TestCostPrimitivesRejectCredits:
    """
    Every function in costs.py is a bare arithmetic expression, so a negative
    rate returned a NEGATIVE COST — indistinguishable downstream from a
    rebate. A backtest charging negative commission earns money by trading,
    flattering exactly the strategies that turn over most.
    """

    @pytest.mark.parametrize(
        "call",
        [
            ("percentage_commission", (1e6, -0.001)),
            ("fixed_bps_spread", (1e6, -10.0)),
            ("short_borrow_cost", (1e6, -500.0, 30.0)),
            ("margin_interest", (1e6, -0.05, 30.0)),
        ],
    )
    def test_negative_rate_rejected(self, call):
        from standard_quant_tools.backtest import costs as C

        name, args = call
        with pytest.raises(ValidationError, match=">= 0"):
            getattr(C, name)(*args)

    def test_nan_rate_rejected_before_the_sign_check(self):
        """`value < 0` is False for NaN, so a comparison-shaped guard would
        have passed it through to produce a NaN cost."""
        from standard_quant_tools.backtest.costs import percentage_commission

        with pytest.raises(ValidationError, match="finite"):
            percentage_commission(1e6, float("nan"))

    def test_inverted_bar_range_rejected(self):
        from standard_quant_tools.backtest.costs import pct_of_range_spread

        with pytest.raises(ValidationError, match="below low"):
            pct_of_range_spread(1e6, high=90.0, low=100.0, close=95.0, pct=0.1)

    def test_ordinary_costs_still_compute(self):
        from standard_quant_tools.backtest.costs import (
            percentage_commission,
            short_borrow_cost,
        )

        assert percentage_commission(1e6, 0.001) == pytest.approx(1000.0)
        assert short_borrow_cost(1e6, 500.0, 30.0) > 0


class TestSizingHygiene:
    def test_infinite_score_rejected(self):
        """
        NaN was rejected, infinity was not — and infinity is worse here: it
        makes the column's mean and std NaN, so EVERY weight in that
        cross-section becomes NaN rather than just the offending one.
        """
        from standard_quant_tools.backtest.sizing import zscore_normalized

        scores = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=IDX[:2])
        scores.iloc[0, 0] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
            zscore_normalized(scores)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_invalid_gross_leverage_rejected(self, bad):
        """It scales the whole vector, so a negative flips every position —
        turning the strategy into its own opposite while each individual
        weight still looks well-formed."""
        from standard_quant_tools.backtest.sizing import rank_weighted

        scores = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=IDX[:2])
        with pytest.raises(ValidationError, match="gross_leverage"):
            rank_weighted(scores, gross_leverage=bad)


class TestDiagnosticsSemantics:
    def test_breakeven_is_neither_a_win_nor_a_loss(self):
        """
        A 0.0 trade fell into `losses` via `~is_win`, dragging avg_loser
        toward zero and — worse — extending max_consecutive_losses through
        trades that were actually flat. On a win/breakeven/loss triple it
        reported avg_loser -0.5 and 2 consecutive losses.
        """
        from standard_quant_tools.metrics.diagnostics import trade_expectancy

        result = trade_expectancy(pd.DataFrame({"return_pct": [1.0, 0.0, -1.0]}))
        assert result["avg_loser_pct"] == pytest.approx(-1.0)
        assert result["max_consecutive_losses"] == 1

    def test_a_flat_trade_breaks_a_losing_streak(self):
        from standard_quant_tools.metrics.diagnostics import trade_expectancy

        result = trade_expectancy(pd.DataFrame({"return_pct": [-1.0, 0.0, -1.0, -1.0]}))
        assert result["max_consecutive_losses"] == 2

    def test_nan_position_is_not_time_in_market(self):
        """A NaN satisfies `!= 0`, so a missing position counted as exposure
        and made every exposure average NaN: time_in_market 0.6667 with
        avg_gross_exposure NaN."""
        from standard_quant_tools.metrics.diagnostics import exposure_stats

        positions = pd.Series([1.0, np.nan, 0.0], index=IDX[:3])
        with pytest.raises(ValidationError, match="non-finite"):
            exposure_stats(positions, None)

    def test_unmeasurable_excursion_is_nan_not_zero(self):
        """0.0 excursion reads as 'never moved against me' — the most
        flattering possible answer for a trade whose prices are missing."""
        from standard_quant_tools.metrics.diagnostics import trade_excursions

        prices = pd.DataFrame(
            {"High": [100.0, 101.0], "Low": [99.0, 98.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        log = pd.DataFrame(
            [
                {
                    "entry_date": pd.Timestamp("2030-01-01"),
                    "exit_date": pd.Timestamp("2030-01-02"),
                    "direction": "long",
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "position_size": 1.0,
                    "return_pct": 1.0,
                }
            ]
        )
        row = trade_excursions(log, prices).iloc[0]
        assert math.isnan(row["mae_pct"])
        assert math.isnan(row["mfe_pct"])


class TestContractHelpersThemselves:
    def test_aligned_rejects_equal_length_different_index(self):
        """Equal length is not alignment: pandas label-aligns while NumPy
        paths pair positionally, so the same inputs mean different things on
        different execution paths."""
        left = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3))
        right = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-02", periods=3))
        with pytest.raises(ValidationError, match="different indexes"):
            require_aligned(left, right, "left", "right", "test")

    def test_positive_int_names_look_ahead_for_negatives(self):
        with pytest.raises(ValidationError, match="FORWARD window"):
            require_positive_int(-5, "window", "test")

    def test_finite_scalar_checked_before_range(self):
        with pytest.raises(ValidationError, match="finite"):
            require_finite_scalar(float("nan"), "rate", "test", minimum=0.0)

    def test_covariance_rejects_nan_and_asymmetry(self):
        with pytest.raises(ValidationError, match="non-finite"):
            require_finite_covariance(
                np.array([[1.0, np.nan], [np.nan, 1.0]]), "cov", "test"
            )
        with pytest.raises(ValidationError, match="not symmetric"):
            require_finite_covariance(np.array([[1.0, 0.5], [0.2, 1.0]]), "cov", "test")

    def test_covariance_accepts_a_real_one(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.05]])
        assert require_finite_covariance(cov, "cov", "test").shape == (2, 2)

    def test_price_series_rejects_zero_and_negative(self):
        with pytest.raises(ValidationError, match="non-positive"):
            require_positive_price_series(
                pd.Series([100.0, 0.0, 101.0]), "Close", "test"
            )

    def test_start_level_allows_a_later_wipeout(self):
        assert (
            require_positive_start_level(
                pd.Series([100.0, 50.0, -10.0]), "equity", "test"
            )
            is not None
        )

    def test_finite_series_names_the_offending_positions(self):
        """A count alone is not actionable on a 2,000-bar series."""
        bad = pd.Series(
            [1.0, np.inf, 3.0], index=pd.date_range("2024-01-01", periods=3)
        )
        with pytest.raises(ValidationError, match="2024-01-02"):
            require_finite_series(bad, "returns", "test")
