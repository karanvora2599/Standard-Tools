"""Tests for historical stress-test scenario replay."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.stress_test import (
    list_stress_scenarios,
    replay_stress_scenario,
    scenario_dates,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import max_drawdown


class TestListStressScenarios:
    def test_returns_expected_named_scenarios(self):
        scenarios = list_stress_scenarios()
        expected = {
            "black_monday_1987",
            "dotcom_2000",
            "gfc_2008",
            "volmageddon_2018",
            "covid_2020",
            "rate_shock_2022",
        }
        assert set(scenarios.keys()) == expected

    def test_each_scenario_has_start_before_end(self):
        for name, dates in list_stress_scenarios().items():
            assert dates["start"] < dates["end"], name


class TestScenarioDates:
    def test_known_scenario_returns_tuple(self):
        start, end = scenario_dates("covid_2020")
        assert start == "2020-02-19"
        assert end == "2020-03-23"

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValidationError, match="Unknown scenario"):
            scenario_dates("not_a_real_crash")


class TestReplayStressScenario:
    def test_hand_computed_equal_weight_two_assets(self):
        dates = pd.date_range("2020-02-19", periods=3, freq="B")
        returns_df = pd.DataFrame(
            {"A": [0.05, -0.10, 0.02], "B": [0.05, -0.10, 0.02]}, index=dates
        )
        result = replay_stress_scenario(returns_df, weights=[0.5, 0.5])

        equity = (1 + returns_df["A"]).cumprod()  # identical to portfolio since A==B
        expected_total_return = float(equity.iloc[-1] - 1.0)
        expected_mdd = float(max_drawdown(equity))

        assert result["portfolio_return_pct"] == pytest.approx(expected_total_return)
        assert result["max_drawdown_pct"] == pytest.approx(expected_mdd)
        assert result["n_trading_days"] == 3

    def test_worst_and_best_day_identified_correctly(self):
        dates = pd.date_range("2020-02-19", periods=4, freq="B")
        returns_df = pd.DataFrame(
            {"A": [0.01, -0.20, 0.15, -0.02], "B": [0.01, -0.20, 0.15, -0.02]},
            index=dates,
        )
        result = replay_stress_scenario(returns_df, weights=[1.0, 0.0])
        assert result["worst_day_return_pct"] == pytest.approx(-0.20)
        assert result["worst_day_date"] == str(dates[1].date())
        assert result["best_day_return_pct"] == pytest.approx(0.15)
        assert result["best_day_date"] == str(dates[2].date())

    def test_unequal_weights_change_portfolio_return(self):
        dates = pd.date_range("2020-02-19", periods=3, freq="B")
        returns_df = pd.DataFrame(
            {"A": [0.10, 0.10, 0.10], "B": [-0.10, -0.10, -0.10]}, index=dates
        )
        heavy_a = replay_stress_scenario(returns_df, weights=[0.9, 0.1])
        heavy_b = replay_stress_scenario(returns_df, weights=[0.1, 0.9])
        assert heavy_a["portfolio_return_pct"] > 0
        assert heavy_b["portfolio_return_pct"] < 0

    def test_empty_returns_raises(self):
        empty = pd.DataFrame({"A": [], "B": []}, dtype=float)
        with pytest.raises(ValidationError, match="empty"):
            replay_stress_scenario(empty, weights=[0.5, 0.5])

    def test_max_drawdown_is_nonpositive(self):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2008-09-01", periods=50, freq="B")
        returns_df = pd.DataFrame(
            {"A": rng.normal(-0.01, 0.03, 50), "B": rng.normal(-0.01, 0.03, 50)},
            index=dates,
        )
        result = replay_stress_scenario(returns_df, weights=[0.5, 0.5])
        assert result["max_drawdown_pct"] <= 0.0
