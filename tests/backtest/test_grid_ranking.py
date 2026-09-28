"""
Which parameter sets may win a grid, and how many were run.

A combination that never traded has no drawdown, no volatility and no loss,
and its Sortino and Calmar used to be +inf, so it won `backtest_grid` under
Calmar, Sortino, max_drawdown and annualized_volatility while a genuine +50%
strategy in the same grid ranked last. Only rows that traded AND have a
finite metric are ranked now; the rest follow in grid order and are
counted. Every test runs on the native path and the Python path.
"""

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as engine
from standard_quant_tools.backtest.engine import backtest_grid, run_strategy
from standard_quant_tools.backtest.ranking import rank_order, rank_rows


@pytest.fixture(params=["native", "python"])
def engine_path(request, monkeypatch):
    if request.param == "native":
        if not engine.HAS_CPP:
            pytest.skip("the C++ extension is not built")
    else:
        monkeypatch.setattr(engine, "HAS_CPP", False)
    return request.param


@pytest.fixture(scope="module")
def uptrend() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2021-01-04", periods=252, freq="B")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0016, 0.01, len(dates)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1e6,
        },
        index=dates,
    )


def by_mode(price_data: pd.DataFrame, mode: str) -> pd.Series:
    """'none' never trades; 'hold' is long throughout; 'half' is long for the
    second half only."""
    signal = pd.Series(0.0, index=price_data.index)
    if mode == "hold":
        signal[:] = 1.0
    elif mode == "half":
        signal.iloc[len(signal) // 2 :] = 1.0
    return signal


def _grid(frame, sort_by, ascending=False, modes=("none", "hold")):
    return backtest_grid(
        frame,
        strategy=by_mode,
        param_grid={"mode": list(modes)},
        commission_pct=0.0,
        slippage_pct=0.0,
        sort_by=sort_by,
        ascending=ascending,
        n_workers=1,
    )


class TestADoNothingRowCannotWin:
    @pytest.mark.parametrize(
        "sort_by, ascending",
        [
            ("sortino_ratio", False),
            ("calmar_ratio", False),
            ("max_drawdown", False),
            ("annualized_volatility", True),
        ],
    )
    def test_the_strategy_that_traded_ranks_first(
        self, engine_path, uptrend, sort_by, ascending
    ):
        out = _grid(uptrend, sort_by, ascending)
        assert list(out["mode"]) == ["hold", "none"]
        assert out.attrs["n_unrankable"] == 1
        assert any("never traded" in w for w in out.attrs["warnings"])

    def test_the_no_trade_row_reports_undefined_ratios(self, engine_path, uptrend):
        """Its values are kept, and they are honest: 0/0 is NaN, and a
        profit factor with no trade is NaN rather than 0.0 ("every trade
        lost")."""
        row = _grid(uptrend, "sharpe_ratio").set_index("mode").loc["none"]
        assert row["num_trades"] == 0
        for column in ("sortino_ratio", "calmar_ratio", "profit_factor"):
            assert np.isnan(row[column]), column

    def test_sharpe_ordering_is_what_it_was(self, engine_path, uptrend):
        """Null case: with every row rankable the order is a plain sort."""
        out = _grid(uptrend, "sharpe_ratio", modes=("half", "hold"))
        assert out.attrs["n_unrankable"] == 0
        assert list(out["sharpe_ratio"]) == sorted(out["sharpe_ratio"], reverse=True)

    def test_unrankable_rows_keep_grid_order(self, engine_path, uptrend):
        """A +inf metric and a no-trade row are both unranked; they follow
        the ranked row in the order the grid listed them."""
        rising = uptrend.copy()
        rising["Close"] = np.linspace(100.0, 150.0, len(rising))
        out = _grid(rising, "sortino_ratio", modes=("none", "hold", "half"))
        # hold and half never lose on a straight line: Sortino is +inf.
        assert np.isinf(out.set_index("mode").loc["hold", "sortino_ratio"])
        assert list(out["mode"]) == ["none", "hold", "half"]
        assert out.attrs["n_unrankable"] == 3


class TestBothPathsAgree:
    def test_a_zero_signal_row_is_the_same_on_both_paths(self, uptrend, monkeypatch):
        if not engine.HAS_CPP:
            pytest.skip("the C++ extension is not built")
        signal = pd.Series(0.0, index=uptrend.index)
        native = run_strategy(uptrend, signal)
        monkeypatch.setattr(engine, "HAS_CPP", False)
        python = run_strategy(uptrend, signal)
        for key in ("sortino_ratio", "calmar_ratio", "profit_factor", "sharpe_ratio"):
            assert np.isnan(native[key]) and np.isnan(python[key]), key


class TestRepeatedGridValuesRunOnce:
    def test_a_fat_fingered_grid_is_the_clean_grid(self, engine_path, uptrend):
        fat = backtest_grid(
            uptrend,
            "sma_crossover",
            {"fast_period": [10, 10, 20], "slow_period": [50, 50]},
            n_workers=1,
        )
        clean = backtest_grid(
            uptrend,
            "sma_crossover",
            {"fast_period": [10, 20], "slow_period": [50]},
            n_workers=1,
        )
        assert len(fat) == 2
        assert fat.attrs["n_combinations"] == 2
        assert fat.attrs["duplicate_values_dropped"] == {
            "fast_period": 1,
            "slow_period": 1,
        }
        assert any("repeated" in w for w in fat.attrs["warnings"])
        pd.testing.assert_frame_equal(fat, clean)

    def test_ten_and_ten_point_zero_are_one_value(self, engine_path, uptrend):
        out = backtest_grid(
            uptrend,
            "sma_crossover",
            {"fast_period": [10, 10.0], "slow_period": [50]},
            n_workers=1,
        )
        assert len(out) == 1

    def test_a_grid_without_repeats_records_nothing(self, engine_path, uptrend):
        out = backtest_grid(
            uptrend,
            "sma_crossover",
            {"fast_period": [5, 10], "slow_period": [30, 50]},
            n_workers=1,
        )
        assert len(out) == 4
        assert out.attrs["duplicate_values_dropped"] == {}
        assert not any("repeated" in w for w in out.attrs["warnings"])


class TestTheRankingRule:
    def test_rank_order(self):
        values = [1.0, np.inf, 3.0, np.nan, 2.0, 3.0, None]
        trades = [4, 4, 4, 4, 0, 4, 4]
        order, n_unrankable = rank_order(values, trades)
        # 3.0 twice (a tie, kept in order), then 1.0; then the inf, NaN,
        # no-trade and None rows in their original order.
        assert order.tolist() == [2, 5, 0, 1, 3, 4, 6]
        assert n_unrankable == 4
        ascending, _ = rank_order(values, trades, ascending=True)
        assert ascending.tolist()[:3] == [0, 2, 5]

    def test_rank_rows_without_the_column_leaves_the_frame(self):
        frame = pd.DataFrame({"a": [3, 1, 2]})
        out = rank_rows(frame, "missing", ascending=False)
        assert out["a"].tolist() == [3, 1, 2]
        assert out.attrs["n_unrankable"] == 0
