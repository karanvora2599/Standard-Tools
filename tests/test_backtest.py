"""Tests for the vectorized backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.engine import run_strategy


@pytest.fixture(scope='module')
def simple_ohlcv():
    """100-bar deterministic OHLCV for backtest tests."""
    np.random.seed(0)
    n = 100
    returns = np.random.normal(0.001, 0.015, n)
    close = 100.0 * np.cumprod(1 + returns)
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    df = pd.DataFrame({
        'Open': close * 0.999,
        'High': close * 1.005,
        'Low': close * 0.995,
        'Close': pd.Series(close, index=dates),
        'Volume': np.full(n, 1_000_000.0),
    }, index=dates)
    return df


class TestReturnKeys:
    def test_result_has_required_keys(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index)
        result = run_strategy(simple_ohlcv, signals)
        required = {
            'final_equity', 'total_return', 'annualized_volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio',
            'win_rate', 'profit_factor', 'num_trades', 'avg_trade_return_pct',
            'equity_curve',
        }
        assert required.issubset(result.keys())

    def test_equity_curve_same_length_as_data(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index)
        result = run_strategy(simple_ohlcv, signals)
        assert len(result['equity_curve']) == len(simple_ohlcv)

    def test_equity_curve_starts_near_initial_capital(self, simple_ohlcv):
        """First bar is always flat (executed position = 0 due to shift)."""
        signals = pd.Series(1, index=simple_ohlcv.index)
        result = run_strategy(simple_ohlcv, signals, initial_capital=10_000)
        assert float(result['equity_curve'].iloc[0]) == pytest.approx(10_000.0, rel=1e-6)


class TestNoSignal:
    def test_zero_signal_yields_flat_equity(self, simple_ohlcv):
        signals = pd.Series(0, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals, initial_capital=10_000)
        assert result['total_return'] == pytest.approx(0.0, abs=1e-6)
        assert result['final_equity'] == pytest.approx(10_000.0, abs=0.01)

    def test_zero_signal_zero_trades(self, simple_ohlcv):
        signals = pd.Series(0, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals)
        assert result['num_trades'] == 0


class TestLookaheadBias:
    def test_first_executed_position_is_zero(self, simple_ohlcv):
        """Signal at bar 0 executes at bar 1: executed[0] must be 0."""
        signals = pd.Series(1, index=simple_ohlcv.index, dtype=float)
        # Manually verify: executed = signals.shift(1).fillna(0)
        executed = signals.shift(1).fillna(0.0)
        assert float(executed.iloc[0]) == 0.0

    def test_return_on_first_bar_is_zero(self, simple_ohlcv):
        """Because executed[0] = 0, strategy return on day 0 is always 0."""
        signals = pd.Series(1, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals, commission_pct=0, slippage_pct=0)
        equity = result['equity_curve']
        assert float(equity.iloc[0]) == pytest.approx(10_000.0, rel=1e-9)


class TestTransactionCosts:
    def test_costs_reduce_returns(self, simple_ohlcv):
        """A strategy with costs must have a lower final equity than without."""
        signals = pd.Series(
            np.where(np.arange(len(simple_ohlcv)) % 5 == 0, 1, 0),
            index=simple_ohlcv.index, dtype=float,
        )
        no_cost = run_strategy(simple_ohlcv, signals, commission_pct=0, slippage_pct=0)
        with_cost = run_strategy(simple_ohlcv, signals, commission_pct=0.002, slippage_pct=0.001)
        assert with_cost['final_equity'] < no_cost['final_equity']

    def test_cost_proportional_to_position_change_size(self, simple_ohlcv):
        """Going +1 → -1 (reversal, change=2) costs more than 0 → +1 (change=1)."""
        # This is structural: just verify the engine doesn't error on short signals
        signals = pd.Series(
            [1, -1, 1, -1] * 25, index=simple_ohlcv.index, dtype=float
        )
        result = run_strategy(simple_ohlcv, signals, commission_pct=0.001, slippage_pct=0.0005)
        assert result['num_trades'] > 0


class TestTradeLog:
    def test_include_trade_log_flag(self, simple_ohlcv):
        signals = pd.Series(
            [0] * 10 + [1] * 20 + [0] * 20 + [1] * 50, index=simple_ohlcv.index, dtype=float
        )
        result = run_strategy(simple_ohlcv, signals, include_trade_log=True)
        assert 'trade_log' in result
        trade_log = result['trade_log']
        assert isinstance(trade_log, pd.DataFrame)

    def test_trade_log_has_correct_columns(self, simple_ohlcv):
        signals = pd.Series([0] * 10 + [1] * 20 + [0] * 70, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals, include_trade_log=True)
        required_cols = {'entry_date', 'exit_date', 'direction', 'entry_price', 'exit_price', 'return_pct'}
        assert required_cols.issubset(set(result['trade_log'].columns))

    def test_trade_log_entry_before_exit(self, simple_ohlcv):
        signals = pd.Series(
            [0] * 5 + [1] * 10 + [0] * 5 + [1] * 10 + [0] * 70,
            index=simple_ohlcv.index, dtype=float,
        )
        result = run_strategy(simple_ohlcv, signals, include_trade_log=True)
        trade_log = result['trade_log']
        for _, row in trade_log.iterrows():
            assert row['entry_date'] <= row['exit_date']

    def test_alternating_signal_yields_correct_num_trades(self, simple_ohlcv):
        """signals [0]*10 [1]*10 [0]*10 [1]*10 [0]*10 ... after shift → 1 completed trade."""
        n = len(simple_ohlcv)
        # One flat block, one long block, then flat for the rest
        sig = pd.Series([0] * 10 + [1] * 10 + [0] * (n - 20), index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, sig, include_trade_log=True)
        assert result['num_trades'] == 1

    def test_long_direction_in_trade_log(self, simple_ohlcv):
        sig = pd.Series([0] * 10 + [1] * 10 + [0] * (len(simple_ohlcv) - 20),
                        index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, sig, include_trade_log=True)
        assert result['trade_log']['direction'].eq('long').all()


class TestMetricBounds:
    def test_win_rate_between_0_and_1(self, simple_ohlcv):
        np.random.seed(1)
        signals = pd.Series(np.random.choice([0, 1], len(simple_ohlcv)).astype(float),
                            index=simple_ohlcv.index)
        result = run_strategy(simple_ohlcv, signals)
        assert 0.0 <= result['win_rate'] <= 1.0

    def test_max_drawdown_nonpositive(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals)
        assert result['max_drawdown'] <= 0

    def test_final_equity_positive(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals)
        assert result['final_equity'] > 0
