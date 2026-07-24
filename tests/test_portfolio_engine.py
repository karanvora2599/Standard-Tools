"""
Hand-verified tests for the true portfolio simulation engine
(backtest/portfolio_engine.py). Expected numbers were computed by hand and
cross-checked with a standalone script (see the plan for the derivation)
before being encoded here as exact assertions.

Reference scenario: 2 tickers (AAPL, MSFT), 5 bars, rebalance at bar 0
({AAPL: 0.5, MSFT: 0.3}) and bar 3 ({AAPL: 0.2, MSFT: 0.6}).
"""

import pandas as pd
import pytest

from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.error import ValidationError


def _price_df(prices, dates):
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [1_000_000.0] * len(prices),
        },
        index=dates,
    )


@pytest.fixture
def two_ticker_price_data():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    aapl = [100.0, 105.0, 102.0, 110.0, 108.0]
    msft = [50.0, 48.0, 52.0, 55.0, 57.0]
    return {
        "AAPL": _price_df(aapl, dates),
        "MSFT": _price_df(msft, dates),
    }, dates


class TestZeroCostReferenceCase:
    def test_hand_verified_equity_curve(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {
                "AAPL": [0.5, None, None, 0.2, None],
                "MSFT": [0.3, None, None, 0.6, None],
            },
            index=dates,
        ).dropna()

        result = run_portfolio_simulation(
            price_data,
            target_weights,
            initial_capital=10_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        equity = result["equity_curve"]
        expected = [10000.0, 10130.0, 10220.0, 10800.0, 10996.363636363636]
        for actual, exp in zip(equity.tolist(), expected):
            assert actual == pytest.approx(exp, abs=1e-6)

    def test_equity_unchanged_immediately_across_zero_cost_rebalance(
        self, two_ticker_price_data
    ):
        """A zero-cost rebalance shouldn't create or destroy equity — only
        redistribute it across positions/cash."""
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.2], "MSFT": [0.3, 0.6]},
            index=[dates[0], dates[3]],
        )
        result = run_portfolio_simulation(
            price_data,
            target_weights,
            initial_capital=10_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        equity = result["equity_curve"]
        assert equity.iloc[0] == pytest.approx(10000.0)
        # bar 3 equity == bar 2's mark-to-market equity (pre-rebalance level preserved)
        assert equity.iloc[3] == pytest.approx(10800.0)

    def test_rebalance_log_shape(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.2], "MSFT": [0.3, 0.6]},
            index=[dates[0], dates[3]],
        )
        result = run_portfolio_simulation(
            price_data, target_weights, commission_pct=0.0, slippage_pct=0.0
        )
        log = result["rebalance_log"]
        assert len(log) == 2
        assert set(log.columns) == {
            "date",
            "turnover_pct",
            "gross_leverage_after",
            "n_positions",
        }
        assert log.iloc[0]["n_positions"] == 2
        assert log.iloc[0]["gross_leverage_after"] == pytest.approx(0.8, abs=1e-4)


class TestWithCostsReferenceCase:
    def test_hand_verified_equity_curve_with_costs(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.2], "MSFT": [0.3, 0.6]},
            index=[dates[0], dates[3]],
        )
        result = run_portfolio_simulation(
            price_data,
            target_weights,
            initial_capital=10_000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
        )
        equity = result["equity_curve"]
        expected = [9988.0, 10118.0, 10208.0, 10778.2272, 10974.372654545454]
        for actual, exp in zip(equity.tolist(), expected):
            assert actual == pytest.approx(exp, abs=1e-4)


class TestShortPosition:
    def test_short_entry_credits_cash_and_preserves_equity(self):
        dates = pd.date_range("2023-01-02", periods=2, freq="B")
        price_data = {"AAPL": _price_df([100.0, 102.0], dates)}
        target_weights = pd.DataFrame({"AAPL": [-0.5]}, index=[dates[0]])

        result = run_portfolio_simulation(
            price_data,
            target_weights,
            initial_capital=10_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        assert result["cash_curve"].iloc[0] == pytest.approx(15_000.0)
        assert result["equity_curve"].iloc[0] == pytest.approx(10_000.0)
        # price rose 100 -> 102 while short 50 shares: lose 50*2 = 100
        assert result["equity_curve"].iloc[1] == pytest.approx(9_900.0)


class TestLeverageCurve:
    def test_leverage_curve_matches_gross_over_equity(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.2], "MSFT": [0.3, 0.6]},
            index=[dates[0], dates[3]],
        )
        result = run_portfolio_simulation(
            price_data, target_weights, commission_pct=0.0, slippage_pct=0.0
        )
        expected = result["gross_exposure_curve"] / result["equity_curve"]
        for actual, exp in zip(result["leverage_curve"].tolist(), expected.tolist()):
            assert actual == pytest.approx(exp, abs=1e-9)


class TestNextOpenFill:
    def test_rebalance_executes_at_following_bars_open(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        prices = pd.DataFrame(
            {
                "Open": [100.0, 103.0, 106.0],
                "High": [101.0, 104.0, 107.0],
                "Low": [99.0, 102.0, 105.0],
                "Close": [100.0, 103.0, 106.0],
                "Volume": [1_000_000.0] * 3,
            },
            index=dates,
        )
        price_data = {"AAPL": prices}
        target_weights = pd.DataFrame({"AAPL": [1.0]}, index=[dates[0]])

        result = run_portfolio_simulation(
            price_data,
            target_weights,
            initial_capital=10_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            fill_price="next_open",
        )
        # Fill happens at dates[1]'s Open (103.0), not dates[0]'s Close (100.0):
        # shares bought = 10000 / 103.0.
        expected_shares = 10_000.0 / 103.0
        # Equity at dates[0] is still all-cash (rebalance hasn't executed yet).
        assert result["equity_curve"].iloc[0] == pytest.approx(10_000.0)
        # Equity at dates[2] = expected_shares * Close[2] (106.0).
        assert result["equity_curve"].iloc[2] == pytest.approx(expected_shares * 106.0)

    def test_last_bar_rebalance_without_following_bar_raises(
        self, two_ticker_price_data
    ):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[-1]])
        with pytest.raises(ValidationError, match="next_open"):
            run_portfolio_simulation(price_data, target_weights, fill_price="next_open")

    def test_missing_open_column_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        no_open = {t: df.drop(columns=["Open"]) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="Open"):
            run_portfolio_simulation(no_open, target_weights, fill_price="next_open")


class TestHl2ExploratoryFill:
    def test_rebalance_executes_at_same_bar_hl2(self):
        dates = pd.date_range("2023-01-02", periods=2, freq="B")
        prices = pd.DataFrame(
            {
                "Open": [100.0, 105.0],
                "High": [104.0, 106.0],
                "Low": [96.0, 104.0],
                "Close": [100.0, 105.0],
                "Volume": [1_000_000.0] * 2,
            },
            index=dates,
        )
        price_data = {"AAPL": prices}
        target_weights = pd.DataFrame({"AAPL": [1.0]}, index=[dates[0]])

        result = run_portfolio_simulation(
            price_data,
            target_weights,
            initial_capital=10_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            fill_price="hl2_exploratory",
        )
        # HL2 of bar 0 = (104 + 96) / 2 = 100.0 -> shares = 100.
        # Equity still marked to Close at bar 0 (100.0 == HL2 here, so
        # equity is unaffected either way); bar 1 equity = 100 shares * 105.
        assert result["equity_curve"].iloc[1] == pytest.approx(100.0 * 105.0)

    def test_missing_high_low_columns_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        no_hl = {t: df.drop(columns=["High", "Low"]) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="High"):
            run_portfolio_simulation(no_hl, target_weights, fill_price="hl2_exploratory")


class TestCostModels:
    def test_per_share_commission_hand_verified(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        result = run_portfolio_simulation(
            price_data,
            target_weights,
            initial_capital=10_000.0,
            commission_model="per_share",
            per_share_rate=0.01,
            slippage_pct=0.0,
        )
        # AAPL: 50 shares @ 0.01 = 0.5; MSFT: 60 shares @ 0.01 = 0.6; total 1.1
        assert result["cash_curve"].iloc[0] == pytest.approx(
            10_000.0 - 5_000.0 - 3_000.0 - 1.1
        )
        assert result["equity_curve"].iloc[0] == pytest.approx(9_998.9)

    def test_min_commission_floor_applies(self):
        dates = pd.date_range("2023-01-02", periods=1, freq="B")
        price_data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [100.0],
                    "Low": [100.0],
                    "Close": [100.0],
                    "Volume": [1_000_000.0],
                },
                index=dates,
            )
        }
        target_weights = pd.DataFrame({"AAPL": [0.01]}, index=dates)
        result = run_portfolio_simulation(
            price_data,
            target_weights,
            initial_capital=10_000.0,
            commission_model="per_share",
            per_share_rate=0.01,
            min_commission=1.0,
            slippage_pct=0.0,
        )
        # notional = 100 -> 1 share; raw commission 1*0.01=0.01, floored to 1.0.
        assert result["equity_curve"].iloc[0] == pytest.approx(10_000.0 - 1.0)

    def test_impact_model_increases_cost_vs_baseline(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        low_volume = {t: df.assign(Volume=1_000.0) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[3]])

        baseline = run_portfolio_simulation(
            low_volume,
            target_weights,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        with_impact = run_portfolio_simulation(
            low_volume,
            target_weights,
            commission_pct=0.0,
            slippage_pct=0.0,
            use_impact_model=True,
            impact_coefficient=5.0,
        )
        assert with_impact["final_equity"] < baseline["final_equity"]

    def test_impact_model_requires_volume_column(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        no_volume = {t: df.drop(columns=["Volume"]) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="Volume"):
            run_portfolio_simulation(no_volume, target_weights, use_impact_model=True)

    def test_borrow_fee_reduces_equity_for_short_position(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        price_data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": [100.0] * 3,
                    "High": [100.0] * 3,
                    "Low": [100.0] * 3,
                    "Close": [100.0] * 3,
                    "Volume": [1_000_000.0] * 3,
                },
                index=dates,
            )
        }
        target_weights = pd.DataFrame({"AAPL": [-0.5]}, index=[dates[0]])

        no_fee = run_portfolio_simulation(
            price_data,
            target_weights,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        with_fee = run_portfolio_simulation(
            price_data,
            target_weights,
            commission_pct=0.0,
            slippage_pct=0.0,
            borrow_fee_bps=500.0,
        )
        assert with_fee["final_equity"] < no_fee["final_equity"]

    def test_margin_interest_accrues_on_negative_cash(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        price_data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": [100.0] * 3,
                    "High": [100.0] * 3,
                    "Low": [100.0] * 3,
                    "Close": [100.0] * 3,
                    "Volume": [1_000_000.0] * 3,
                },
                index=dates,
            )
        }
        # max_gross_leverage > 1.0 needed for cash to go negative on a long-only book.
        target_weights = pd.DataFrame({"AAPL": [1.5]}, index=[dates[0]])

        no_interest = run_portfolio_simulation(
            price_data,
            target_weights,
            commission_pct=0.0,
            slippage_pct=0.0,
            max_gross_leverage=1.5,
            max_position_pct=1.5,
        )
        with_interest = run_portfolio_simulation(
            price_data,
            target_weights,
            commission_pct=0.0,
            slippage_pct=0.0,
            max_gross_leverage=1.5,
            max_position_pct=1.5,
            margin_interest_rate=0.10,
        )
        assert with_interest["final_cash"] < no_interest["final_cash"]

    def test_max_adv_participation_exceeded_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        low_volume = {t: df.assign(Volume=10.0) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="ADV participation"):
            run_portfolio_simulation(
                low_volume, target_weights, max_adv_participation=0.01
            )

    def test_max_adv_participation_requires_volume_column(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        no_volume = {t: df.drop(columns=["Volume"]) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="Volume"):
            run_portfolio_simulation(
                no_volume, target_weights, max_adv_participation=0.5
            )

    def test_invalid_commission_model_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="commission_model"):
            run_portfolio_simulation(
                price_data, target_weights, commission_model="bogus"
            )

    def test_zero_delta_rebalance_charges_no_commission(self):
        """
        Regression test: a "rebalance" whose target weight is 0.0 for a
        ticker that was never bought (shares start at 0, so delta = 0 - 0
        = 0 exactly — a genuine, unambiguous zero-size trade, not merely a
        small one) must cost nothing, even under commission_model=
        'per_share' with a large minimum-commission floor —
        _apply_rebalance must skip zero-size trades entirely rather than
        calling _trade_cost (whose minimum floor would otherwise charge a
        ticker that never actually traded).
        """
        dates = pd.date_range("2023-01-02", periods=2, freq="B")
        price_data = {"AAPL": _price_df([100.0, 100.0], dates)}
        target_weights = pd.DataFrame({"AAPL": [0.0]}, index=[dates[0]])

        result = run_portfolio_simulation(
            price_data,
            target_weights,
            commission_model="per_share",
            per_share_rate=0.01,
            min_commission=100.0,
        )
        assert result["final_equity"] == pytest.approx(10_000.0)
        assert result["final_cash"] == pytest.approx(10_000.0)
        assert result["rebalance_log"].iloc[0]["turnover_pct"] == 0.0

    def test_use_impact_model_zero_volume_raises(self, two_ticker_price_data):
        """
        Regression test: use_impact_model=True with an actual (nonzero-
        delta) trade against zero Volume must fail closed — a missing
        liquidity baseline can't be silently treated as "no impact."
        """
        price_data, dates = two_ticker_price_data
        zero_volume = {t: df.assign(Volume=0.0) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="average dollar volume"):
            run_portfolio_simulation(zero_volume, target_weights, use_impact_model=True)

    def test_max_adv_participation_nan_volume_raises(self, two_ticker_price_data):
        import numpy as np

        price_data, dates = two_ticker_price_data
        nan_volume = {t: df.assign(Volume=np.nan) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="average dollar volume"):
            run_portfolio_simulation(
                nan_volume, target_weights, max_adv_participation=0.5
            )

    def test_unconstrained_backtest_unaffected_by_zero_volume(
        self, two_ticker_price_data
    ):
        """
        Without max_adv_participation/use_impact_model set, zero Volume is
        fine — proves the fail-closed fix only fires when the caller
        explicitly opted into a liquidity-aware feature, not for every
        backtest.
        """
        price_data, dates = two_ticker_price_data
        zero_volume = {t: df.assign(Volume=0.0) for t, df in price_data.items()}
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        result = run_portfolio_simulation(zero_volume, target_weights)
        assert result["final_equity"] > 0

    def test_boundary_weights_with_nonzero_cost_succeed_not_rejected(
        self, two_ticker_price_data
    ):
        """
        Regression test: weights summing to exactly max_gross_leverage pass
        the upfront (pre-cost) validation. Nonzero commission/slippage then
        shrinks equity_after below equity_now while gross exposure (shares
        priced at exec_prices) is unchanged -- so the *reported*
        gross_leverage_after in rebalance_log is mechanically pushed above
        max_gross_leverage on every such trade. That is expected, unavoidable
        cost drag, not a sizing bug, and must NOT raise. (An earlier version
        of this fix compared realized leverage against equity_after with a
        tight tolerance, which made every fully-invested backtest with
        realistic costs fail -- this test guards against reintroducing that.)
        """
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.6], "MSFT": [0.4]}, index=[dates[0]]
        )  # gross == 1.0 exactly
        result = run_portfolio_simulation(
            price_data,
            target_weights,
            commission_pct=0.01,
            slippage_pct=0.01,
            max_gross_leverage=1.0,
        )
        assert result["final_equity"] > 0
        # The cost-inflated ratio is still honestly reported for auditability...
        assert result["rebalance_log"].iloc[0]["gross_leverage_after"] > 1.0
        # ...even though it does not (and should not) block the backtest.

    def test_zero_cost_leverage_at_exact_limit_succeeds(self, two_ticker_price_data):
        """Same weights as above, but zero costs -> realized leverage equals
        (not exceeds) the limit, and the rebalance must still succeed."""
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.6], "MSFT": [0.4]}, index=[dates[0]])
        result = run_portfolio_simulation(
            price_data,
            target_weights,
            commission_pct=0.0,
            slippage_pct=0.0,
            max_gross_leverage=1.0,
        )
        assert result["final_equity"] > 0
        assert result["rebalance_log"].iloc[0]["gross_leverage_after"] == pytest.approx(
            1.0
        )


class TestValidation:
    def test_invalid_fill_price_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="fill_price"):
            run_portfolio_simulation(price_data, target_weights, fill_price="bogus")

    def test_missing_ticker_in_price_data_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "GOOGL": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="GOOGL"):
            run_portfolio_simulation({"AAPL": price_data["AAPL"]}, target_weights)

    def test_gross_leverage_exceeded_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.8], "MSFT": [0.8]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="leverage"):
            run_portfolio_simulation(price_data, target_weights, max_gross_leverage=1.0)

    def test_position_bound_exceeded_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.9], "MSFT": [0.05]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="max_position_pct"):
            run_portfolio_simulation(price_data, target_weights, max_position_pct=0.5)

    def test_rebalance_date_outside_price_calendar_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        bad_date = pd.Timestamp("2023-06-01")
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[bad_date])
        with pytest.raises(ValidationError, match="no price data"):
            run_portfolio_simulation(price_data, target_weights)

    def test_empty_ticker_universe_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(index=[dates[0]])  # no columns
        with pytest.raises(ValidationError, match="empty universe"):
            run_portfolio_simulation(price_data, target_weights)

    def test_empty_rebalance_panel_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(columns=["AAPL", "MSFT"])  # no rows
        with pytest.raises(ValidationError, match="no rebalance dates"):
            run_portfolio_simulation(price_data, target_weights)

    def test_duplicate_rebalance_dates_raise(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.5], "MSFT": [0.3, 0.3]}, index=[dates[0], dates[0]]
        )
        with pytest.raises(ValidationError, match="duplicate"):
            run_portfolio_simulation(price_data, target_weights)

    def test_unsorted_rebalance_dates_raise(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.2], "MSFT": [0.3, 0.6]}, index=[dates[3], dates[0]]
        )
        with pytest.raises(ValidationError, match="sorted"):
            run_portfolio_simulation(price_data, target_weights)

    def test_infinite_weight_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [float("inf")], "MSFT": [0.3]}, index=[dates[0]]
        )
        with pytest.raises(ValidationError, match="infinite"):
            run_portfolio_simulation(price_data, target_weights)

    def test_nonpositive_close_price_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        bad_data = {t: df.copy() for t, df in price_data.items()}
        bad_data["AAPL"].loc[dates[2], "Close"] = 0.0
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="nonpositive or non-finite"):
            run_portfolio_simulation(bad_data, target_weights)

    def test_nonfinite_close_price_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        bad_data = {t: df.copy() for t, df in price_data.items()}
        bad_data["MSFT"].loc[dates[1], "Close"] = float("nan")
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="nonpositive or non-finite"):
            run_portfolio_simulation(bad_data, target_weights)

    def test_nonpositive_initial_capital_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="initial_capital"):
            run_portfolio_simulation(price_data, target_weights, initial_capital=0.0)

    def test_negative_commission_pct_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="commission_pct"):
            run_portfolio_simulation(price_data, target_weights, commission_pct=-0.01)

    def test_nonpositive_impact_lookback_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="impact_lookback"):
            run_portfolio_simulation(price_data, target_weights, impact_lookback=0)

    def test_empty_common_trading_calendar_raises(self):
        dates_a = pd.date_range("2023-01-02", periods=3, freq="B")
        dates_b = pd.date_range("2024-01-02", periods=3, freq="B")  # no overlap
        price_data = {
            "AAPL": _price_df([100.0, 101.0, 102.0], dates_a),
            "MSFT": _price_df([50.0, 51.0, 52.0], dates_b),
        }
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[dates_a[0]])
        with pytest.raises(ValidationError, match="no common trading dates"):
            run_portfolio_simulation(price_data, target_weights)


class TestInsolvency:
    def test_negative_equity_from_rebalance_cost_raises(self, two_ticker_price_data):
        """
        Regression test (high-severity item 7): a rebalance whose costs
        alone exceed the account's equity must raise, not silently continue
        with negative equity (which would make leverage_curve divide by a
        negative number, producing a nonsensical negative leverage value,
        and would make downstream annualized-return calculations raise on
        a negative base raised to a fractional power).
        """
        dates = pd.date_range("2023-01-02", periods=2, freq="B")
        price_data = {"AAPL": _price_df([100.0, 100.0], dates)}
        target_weights = pd.DataFrame({"AAPL": [1.0]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="insolvent"):
            run_portfolio_simulation(
                price_data, target_weights,
                initial_capital=100.0, commission_pct=2.0, slippage_pct=0.0,
            )

    def test_negative_equity_from_price_move_raises(self):
        """A short position whose losses exceed initial capital purely from
        price drift (no further rebalance) must also raise, not silently
        mark a negative-equity bar into the returned equity_curve."""
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        price_data = {"AAPL": _price_df([100.0, 100.0, 500.0], dates)}
        target_weights = pd.DataFrame({"AAPL": [-1.0]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="insolvent"):
            run_portfolio_simulation(
                price_data, target_weights,
                initial_capital=10_000.0, commission_pct=0.0, slippage_pct=0.0,
            )


class TestFinancingCalendarGaps:
    def test_weekend_gap_charges_three_days_not_one(self):
        """
        Regression test (high-severity item 6): financing (borrow fee on a
        short position) must accrue based on the actual elapsed CALENDAR
        days since the prior bar, not a hardcoded 1 -- a Friday-to-Monday
        gap must charge 3 days' worth, not 1. Business-day bars
        (freq="B") naturally include this gap: Thu 2023-01-05, Fri
        2023-01-06, Mon 2023-01-09.

        Price held constant (no further rebalance after bar 0) isolates
        the cash change on each subsequent bar to financing alone, so the
        Fri->Mon cash delta must be exactly 3x the Thu->Fri cash delta.
        """
        dates = pd.to_datetime(["2023-01-05", "2023-01-06", "2023-01-09", "2023-01-10"])
        price_data = {"AAPL": _price_df([100.0] * 4, dates)}
        target_weights = pd.DataFrame({"AAPL": [-1.0]}, index=[dates[0]])
        result = run_portfolio_simulation(
            price_data, target_weights,
            initial_capital=10_000.0, commission_pct=0.0, slippage_pct=0.0,
            borrow_fee_bps=500.0,
        )
        cash = result["cash_curve"]
        weekday_gap_cost = float(cash.iloc[0] - cash.iloc[1])  # Thu -> Fri (1 day)
        weekend_gap_cost = float(cash.iloc[1] - cash.iloc[2])  # Fri -> Mon (3 days)
        assert weekday_gap_cost > 0
        assert weekend_gap_cost == pytest.approx(3.0 * weekday_gap_cost, rel=1e-6)
