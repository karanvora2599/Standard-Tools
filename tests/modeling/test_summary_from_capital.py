"""
A model portfolio's growth and drawdown are measured from the capital it
opened with.

`evaluate_model_portfolio` summarized the simulation from the equity
curve's first point, which is the first bar after any rebalance on it, net
of that rebalance's costs. A book that paid 10 bps to put itself on then
read 10 bps richer than it was, and its drawdown started from a peak below
the one it fell from. `run_portfolio_simulation` (the tool) has measured
both from the capital since its day-0 fix, so the same simulation read
differently through the two doors.

The planted book holds one asset whole from the first bar at 10 bps of
commission and nothing else: $100,000 buys 1,000 shares at 100 and pays
$100, so the curve opens at 99,900. The price falls to 80 and ends at 110,
so the account reads 79,900 at its low and 109,900 at the end -- a total
return of +9.9% and a maximum drawdown of -20.1%, where the first point
gave +10.01% and -20.02%.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes.backtest.tools import (
    _metrics_with_day0_cost,
)
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.return_metrics import (
    annualized_volatility,
    cagr,
    cumulative_return,
)
from standard_quant_tools.metrics.risk_metrics import calmar_ratio, max_drawdown
from standard_quant_tools.modeling.agent import (
    BuildModelDatasetInput,
    RunModelExperimentInput,
    build_model_dataset,
    run_model_experiment,
)
from standard_quant_tools.modeling.portfolio_eval import (
    _summarize_simulation,
    evaluate_model_portfolio,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PortfolioSimSpec,
    TargetSpec,
    ValidationSpec,
)

CAPITAL = 100_000.0
N_BARS = 60


def _planted_simulation(first_rebalance: int = 0):
    close = np.full(N_BARS, 80.0)
    close[: first_rebalance + 1] = 100.0
    close[-1] = 110.0
    index = pd.bdate_range("2024-01-02", periods=N_BARS)
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": np.full(N_BARS, 1_000_000.0),
        },
        index=index,
    )
    weights = pd.DataFrame(
        {"AAA": [1.0]}, index=index[first_rebalance : first_rebalance + 1]
    )
    return run_portfolio_simulation(
        {"AAA": frame},
        weights,
        initial_capital=CAPITAL,
        commission_pct=0.001,
        slippage_pct=0.0,
        fill_price="close",
    )


def _summary(simulation, initial_capital=CAPITAL):
    return _summarize_simulation(
        simulation, 252, 0.0, 0.001, 0.0, initial_capital=initial_capital
    )


class TestMeasuredFromTheCapital:
    def test_the_planted_book(self):
        simulation = _planted_simulation()
        assert float(simulation["equity_curve"].iloc[0]) == pytest.approx(
            99_900.0, abs=1e-6
        )
        metrics = _summary(simulation)
        assert metrics["cumulative_return"] == pytest.approx(0.099, abs=1e-12)
        assert metrics["max_drawdown"] == pytest.approx(-0.201, abs=1e-12)
        # The capital and the opening fill are one bar: 60 bars span 59.
        growth = 1.099 ** (252 / (N_BARS - 1)) - 1.0
        assert metrics["cagr"] == pytest.approx(growth, rel=1e-12)
        assert metrics["calmar_ratio"] == pytest.approx(growth / 0.201, rel=1e-12)

    def test_the_first_point_gave_the_cost_back(self):
        """What the summary read before, so the size of the change is on
        record: from 99,900 the book gained 10.01% and fell 20.02%."""
        metrics = _summary(_planted_simulation())
        before_total, before_drawdown = 109_900 / 99_900 - 1, 79_900 / 99_900 - 1
        assert before_total == pytest.approx(0.1001001, abs=1e-7)
        assert before_drawdown == pytest.approx(-0.2002002, abs=1e-7)
        assert metrics["cumulative_return"] < before_total
        assert metrics["max_drawdown"] < before_drawdown

    def test_the_two_doors_agree_to_the_bit(self):
        """The evaluation's total return and drawdown are the backtest
        tool's, from the same simulation."""
        simulation = _planted_simulation()
        metrics = _summary(simulation)
        _, total, _, with_start = _metrics_with_day0_cost(
            simulation["equity_curve"], CAPITAL
        )
        assert metrics["cumulative_return"] == total
        assert metrics["max_drawdown"] == float(max_drawdown(with_start))

    def test_a_first_bar_that_trades_nothing_moves_nothing(self):
        """Rebalanced first on the sixth bar, the account is all cash on
        the first, so its equity there IS the capital and every figure is
        the curve's own, to the bit. This is how the evaluation's own
        simulations open: its prices start before the first rebalance."""
        simulation = _planted_simulation(first_rebalance=5)
        equity = simulation["equity_curve"]
        assert float(equity.iloc[0]) == CAPITAL
        metrics = _summary(simulation)
        assert metrics["cumulative_return"] == float(cumulative_return(equity))
        assert metrics["cagr"] == float(cagr(equity, 252))
        assert metrics["max_drawdown"] == float(max_drawdown(equity))
        assert metrics["calmar_ratio"] == float(calmar_ratio(equity, 252))

    def test_the_per_bar_figures_still_read_the_curve_s_own_bars(self):
        """Volatility and the ratios read the bars after the opening one;
        their risk-free benchmark accrues from the second bar, as the
        engine credits cash, so the opening trade is not one of them."""
        simulation = _planted_simulation()
        returns = simulation["equity_curve"].pct_change(fill_method=None).dropna()
        metrics = _summary(simulation)
        assert metrics["annualized_volatility"] == float(
            annualized_volatility(returns, 252)
        )

    @pytest.mark.parametrize("capital", [0.0, -1.0, math.nan, math.inf])
    def test_a_capital_that_cannot_be_a_denominator_is_refused(self, capital):
        with pytest.raises(ValidationError, match="initial_capital"):
            _summary(_planted_simulation(), initial_capital=capital)


UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


@pytest.fixture
def registered_model(patched_multi_factory):
    dataset = build_model_dataset(
        BuildModelDatasetInput(
            spec=DatasetSpec(
                universe=UNIVERSE,
                start="2022-01-01",
                end="2023-12-31",
                features=[
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 20}, alias="mom_20"
                    ),
                ],
                target=TargetSpec(type="forward_return", horizon=5),
            )
        )
    )
    experiment = run_model_experiment(
        RunModelExperimentInput(
            dataset_id=dataset.dataset_id,
            spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(
                    train_window=150, test_window=40, embargo=5, min_folds=2
                ),
                random_seed=1,
            ),
        )
    )
    return experiment.model_id


def test_the_evaluation_reads_its_total_return_from_the_spec_s_capital(
    registered_model,
):
    """End to end: the summary is handed the capital the spec opened the
    account with, so the total return is the final equity over it."""
    portfolio = PortfolioSimSpec(initial_capital=250_000.0)
    metrics = evaluate_model_portfolio(registered_model, portfolio=portfolio)["metrics"]
    assert metrics["cumulative_return"] == metrics["final_equity"] / 250_000.0 - 1.0
