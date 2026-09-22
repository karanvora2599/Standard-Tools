"""
The per-bar state every backtest engine builds, and used to throw away.

Each engine here computes a family of daily series -- cash, gross and net
exposure, leverage, posted margin, contracts held, a spread state machine
-- and, until the CHANGELOG entry of 2026-09-22, shipped the equity curve
and a handful of scalars. Four consequences, one test class each:

  NEUTRALITY WAS AN INPUT WITH NO OUTPUT. `make_dollar_neutral` is an
  argument; nothing in any result said whether the book stayed neutral as
  prices moved it. `net_exposure_min/max/mean` say so inline, and the
  curve behind them is publishable.

  A CAP THAT NAMED NOTHING. The engine records WHICH tickers it sized down
  to the participation limit; only the count crossed the boundary.

  ZERO MARGIN CALLS MEANT TWO DIFFERENT QUARTERS. A futures account that
  never went near its maintenance requirement and one that spent the
  quarter a tick above it both reported `n_margin_calls == 0`.
  `min_margin_cushion` separates them.

  A CHAIN BREAK. `run_signal_panel_backtest` blended a portfolio return
  series, reduced it to a metrics dict and dropped the series, which made
  it the one backtest tool whose output could not be fed to anything on
  the return-consuming side of the surface.

References are OPT-IN throughout: pass `run_id` and the curves are
published, omit it and nothing is written and every ref is None. The two
kinds they travel as -- `analytic_series` and `analytic_frame` -- are
pinned here as well, because a kind an agent is told about must be one the
package can actually mint.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import (
    CompareStrategiesInput,
    ListReferenceKindsInput,
    PairTradeBacktestInput,
    PortfolioSimulationInput,
    SignalPanelBacktestInput,
    WalkForwardInput,
)
from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.backtest.futures_tools import (
    FuturesBacktestInput,
    run_futures_backtest,
)
from standard_quant_tools.agent.runtimes.backtest.tools import (
    compare_strategies,
    run_pair_trade_backtest,
    run_portfolio_simulation,
    run_signal_panel_backtest,
    run_walk_forward_backtest,
)
from standard_quant_tools.agent.runtimes.meta.tools import list_reference_kinds
from standard_quant_tools.agent.runtimes.research.reference_tools import (
    SeriesMetricsInput,
    calculate_series_metrics,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError

START = "2023-01-02"


@pytest.fixture(autouse=True)
def runs_root(tmp_path, monkeypatch):
    """Every publish in this module lands under a directory of its own, so
    'nothing was written' is a statement about an empty tree."""
    root = tmp_path / "runs"
    monkeypatch.setenv("SQT_RUNS_DIR", str(root))
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    return root


def _ohlcv(close: pd.Series, volume: float = 5_000_000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": pd.Series(volume, index=close.index),
        }
    )


def _serve(frames, monkeypatch):
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda symbol, *a, **kw: frames[symbol]
    provider.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, *a, **kw: frames[symbol]
    )
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


def _written(root) -> list:
    return [] if not root.exists() else sorted(p.name for p in root.rglob("*.parquet"))


# ── the book that was built neutral ──────────────────────────────────────


@pytest.fixture
def swinging_pair(monkeypatch):
    """
    Two names at 100. The second never moves; the first alternates 130 /
    70 from the third bar on, so a book held +0.5 / -0.5 is genuinely long
    on half the bars and genuinely short on the other half while its
    AVERAGE net exposure is close to zero. That is exactly the situation
    no field used to describe.
    """
    dates = pd.date_range(START, periods=20, freq="B")
    swing = [100.0, 100.0, 100.0] + [130.0 if i % 2 == 0 else 70.0 for i in range(17)]
    frames = {
        "AAPL": _ohlcv(pd.Series(swing, index=dates)),
        "MSFT": _ohlcv(pd.Series(100.0, index=dates)),
    }
    _serve(frames, monkeypatch)
    return dates


def _simulation(dates, weights, rebalance_bar=2, **overrides):
    rebalance = str(dates[rebalance_bar].date())
    payload = dict(
        tickers=sorted(weights),
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        target_weights={t: {rebalance: w} for t, w in weights.items()},
        initial_capital=10_000.0,
        commission_pct=0.0,
        slippage_pct=0.0,
    )
    payload.update(overrides)
    return run_portfolio_simulation(PortfolioSimulationInput(**payload))


class TestWhetherNeutralityHeld:
    def test_a_dollar_neutral_book_reports_both_sides_of_zero(self, swinging_pair):
        result = _simulation(swinging_pair, {"AAPL": 0.5, "MSFT": -0.5})

        assert result.net_exposure_max > 0 > result.net_exposure_min
        # The average is near zero while the extremes are not: the book
        # was neutral ON AVERAGE and meaningfully directional on any given
        # bar, and both halves of that are now readable.
        gross = result.avg_gross_leverage
        assert abs(result.net_exposure_mean) < gross / 10
        assert (
            max(abs(result.net_exposure_max), abs(result.net_exposure_min)) > gross / 10
        )

    def test_a_single_long_is_net_long_by_its_whole_gross(self, swinging_pair):
        result = _simulation(swinging_pair, {"AAPL": 1.0}, rebalance_bar=0)

        # One long, no cash left over, invested from the first bar: net
        # exposure IS gross exposure on every bar, so the three scalars
        # collapse onto one another.
        assert result.net_exposure_min == pytest.approx(result.net_exposure_max)
        assert result.net_exposure_mean == pytest.approx(result.net_exposure_max)
        assert result.net_exposure_max == pytest.approx(
            result.max_gross_leverage_used, abs=1e-6
        )
        assert not any("neutral" in w.lower() for w in result.warnings)

    def test_the_scalars_are_the_published_curve(self, swinging_pair):
        result = _simulation(
            swinging_pair, {"AAPL": 0.5, "MSFT": -0.5}, run_id="neutrality-check"
        )

        net = handoff.resolve(result.net_exposure_curve_ref, expect="analytic_series")
        equity = pd.Series(result.equity_curve, index=net.index)
        fraction = net / equity
        assert float(fraction.min()) == pytest.approx(result.net_exposure_min)
        assert float(fraction.max()) == pytest.approx(result.net_exposure_max)
        assert float(fraction.mean()) == pytest.approx(result.net_exposure_mean)


# ── which tickers hit the cap ────────────────────────────────────────────


@pytest.fixture
def thin_tape(monkeypatch):
    """Two names whose whole daily dollar volume is a rounding error next
    to the book, so any participation limit binds."""
    dates = pd.date_range(START, periods=10, freq="B")
    frames = {
        "AAPL": _ohlcv(pd.Series(100.0, index=dates), volume=10.0),
        "MSFT": _ohlcv(pd.Series(50.0, index=dates), volume=10.0),
    }
    _serve(frames, monkeypatch)
    return dates


@pytest.fixture
def deep_tape(monkeypatch):
    dates = pd.date_range(START, periods=10, freq="B")
    frames = {
        "AAPL": _ohlcv(pd.Series(100.0, index=dates)),
        "MSFT": _ohlcv(pd.Series(50.0, index=dates)),
    }
    _serve(frames, monkeypatch)
    return dates


class TestTheCapNamesTheTickers:
    def test_a_forced_cap_names_every_ticker_it_sized_down(self, thin_tape):
        result = _simulation(
            thin_tape,
            {"AAPL": 0.5, "MSFT": 0.3},
            max_adv_participation=0.01,
        )
        event = result.rebalance_log[0]
        assert event.n_capped == 2
        assert sorted(event.capped) == ["AAPL", "MSFT"]
        # The count and the names agree, which is what makes the list
        # readable as the cap's own record rather than a second opinion.
        assert len(event.capped) == event.n_capped

    def test_a_limit_that_does_not_bind_names_nobody(self, deep_tape):
        result = _simulation(
            deep_tape,
            {"AAPL": 0.5, "MSFT": 0.3},
            max_adv_participation=0.5,
        )
        event = result.rebalance_log[0]
        assert event.n_capped == 0
        assert event.capped == []


# ── how close the futures account came ───────────────────────────────────


def _futures(**overrides):
    payload = dict(
        prices={"2024-01-02": 100.0, "2024-01-03": 100.0, "2024-01-04": 100.0},
        target_contracts={"2024-01-02": 1.0},
        multiplier=1.0,
        initial_capital=5_200.0,
        initial_margin=5_000.0,
    )
    payload.update(overrides)
    return run_futures_backtest(FuturesBacktestInput(**payload))


class TestHowCloseTheAccountCame:
    def test_no_margin_call_and_a_cushion_under_five_percent(self):
        """$5,200 of equity against a $5,000 maintenance requirement: the
        line is never crossed, and the account is 3.8% of its equity from
        crossing it for the whole run."""
        result = _futures()

        assert result.n_margin_calls == 0
        assert result.min_margin_cushion == pytest.approx(200.0 / 5_200.0, rel=1e-6)
        assert result.min_margin_cushion < 0.05

    def test_a_comfortable_account_reports_a_wide_cushion(self):
        result = _futures(initial_capital=100_000.0)

        assert result.n_margin_calls == 0
        assert result.min_margin_cushion > 0.9

    def test_an_unmargined_account_has_no_line_to_be_near(self):
        result = _futures(initial_capital=100_000.0, initial_margin=0.0)

        assert result.min_margin_cushion is None
        assert any("initial_margin is zero" in w for w in result.warnings)

    def test_the_five_curves_publish_and_resolve(self):
        result = _futures(run_id="margin-cushion-run")

        for ref in (
            result.cash_curve_ref,
            result.margin_curve_ref,
            result.position_curve_ref,
            result.exposure_curve_ref,
            result.leverage_curve_ref,
        ):
            assert handoff.parse(ref).kind == "analytic_series"

        margin = handoff.resolve(result.margin_curve_ref, expect="analytic_series")
        contracts = handoff.resolve(result.position_curve_ref, expect="analytic_series")
        assert margin.tolist() == [5_000.0, 5_000.0, 5_000.0]
        assert contracts.tolist() == [1.0, 1.0, 1.0]

    def test_without_a_run_id_nothing_is_written(self, runs_root):
        result = _futures()

        assert result.cash_curve_ref is None
        assert result.margin_curve_ref is None
        assert result.position_curve_ref is None
        assert result.exposure_curve_ref is None
        assert result.leverage_curve_ref is None
        assert _written(runs_root) == []


# ── the panel backtest's blended return series ───────────────────────────


@pytest.fixture
def three_names(monkeypatch):
    rng = np.random.default_rng(20260922)
    dates = pd.date_range(START, periods=180, freq="B")
    frames = {}
    for ticker, drift in (("AAPL", 0.0006), ("MSFT", 0.0003), ("GOOGL", -0.0002)):
        close = 100.0 * np.cumprod(1.0 + rng.normal(drift, 0.012, len(dates)))
        frames[ticker] = _ohlcv(pd.Series(close, index=dates))
    _serve(frames, monkeypatch)
    return frames, dates


def _panel_signals(frame: pd.DataFrame) -> dict:
    signal = (frame["Close"].pct_change(5) > 0).astype(float)
    return {str(d.date()): float(v) for d, v in signal.items()}


def _panel_backtest(frames, dates, **overrides):
    payload = dict(
        tickers=sorted(frames),
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        signal_panel={t: _panel_signals(f) for t, f in frames.items()},
    )
    payload.update(overrides)
    return run_signal_panel_backtest(SignalPanelBacktestInput(**payload))


class TestThePanelBacktestJoinsTheChain:
    def test_the_published_returns_reproduce_the_reported_sharpe(self, three_names):
        frames, dates = three_names
        result = _panel_backtest(frames, dates, run_id="panel-chain-run")

        assert handoff.parse(result.portfolio_returns_ref).kind == "returns_panel"
        frame = handoff.resolve(result.portfolio_returns_ref, expect="returns_panel")
        assert list(frame.columns) == ["portfolio"]

        scored = calculate_series_metrics(
            SeriesMetricsInput(
                series={"ref": result.portfolio_returns_ref},
                metrics=["sharpe_ratio"],
            )
        )
        # The summary was always a reduction of this series; now the
        # series is reachable and the two agree. The panel rounds its
        # Sharpe to four places, so the comparison rounds too.
        assert round(scored.values["sharpe_ratio"], 4) == pytest.approx(
            result.portfolio_metrics["sharpe_ratio"], abs=1e-6
        )

    def test_the_engines_caveats_reach_the_portfolio_result(self, three_names):
        frames, dates = three_names
        result = _panel_backtest(frames, dates)

        # Every per-ticker result carried the fill_price='close'
        # look-ahead caveat and the portfolio result carried none.
        assert result.warnings
        assert any(
            "look-ahead" in w.lower() or "close" in w.lower() for w in result.warnings
        )

    def test_without_a_run_id_nothing_is_written(self, three_names, runs_root):
        frames, dates = three_names
        result = _panel_backtest(frames, dates)

        assert result.portfolio_returns_ref is None
        assert _written(runs_root) == []


# ── the shared-cash simulation's own curves ──────────────────────────────


class TestTheAccountCurvesAreTheEnginesOwn:
    def test_every_ref_resolves_to_the_curve_behind_the_scalars(self, swinging_pair):
        result = _simulation(
            swinging_pair, {"AAPL": 0.5, "MSFT": -0.5}, run_id="state-curves-run"
        )

        kinds = {
            result.cash_curve_ref: "analytic_series",
            result.gross_exposure_curve_ref: "analytic_series",
            result.net_exposure_curve_ref: "analytic_series",
            result.leverage_curve_ref: "analytic_series",
            result.portfolio_returns_ref: "returns_panel",
        }
        for ref, kind in kinds.items():
            assert handoff.parse(ref).kind == kind
            assert handoff.describe(ref)["producer"] == (
                "backtest.run_portfolio_simulation"
            )

        cash = handoff.resolve(result.cash_curve_ref, expect="analytic_series")
        gross = handoff.resolve(result.gross_exposure_curve_ref, "analytic_series")
        net = handoff.resolve(result.net_exposure_curve_ref, "analytic_series")
        leverage = handoff.resolve(result.leverage_curve_ref, "analytic_series")
        equity = pd.Series(result.equity_curve, index=cash.index)

        # The engine's own identity for a cash book: equity is the cash
        # balance plus the signed value of what is held.
        pd.testing.assert_series_equal(cash + net, equity, check_names=False, rtol=1e-9)
        pd.testing.assert_series_equal(
            gross / equity, leverage, check_names=False, rtol=1e-9
        )
        assert float(leverage.mean()) == pytest.approx(
            result.avg_gross_leverage, abs=5e-5
        )

    def test_a_state_curve_is_not_an_equity_curve(self, swinging_pair):
        result = _simulation(
            swinging_pair, {"AAPL": 0.5, "MSFT": -0.5}, run_id="kind-check-run"
        )
        with pytest.raises(ValidationError, match="equity_curve"):
            handoff.resolve(result.leverage_curve_ref, expect="equity_curve")

    def test_without_a_run_id_nothing_is_written(self, swinging_pair, runs_root):
        result = _simulation(swinging_pair, {"AAPL": 0.5, "MSFT": -0.5})

        assert result.cash_curve_ref is None
        assert result.gross_exposure_curve_ref is None
        assert result.net_exposure_curve_ref is None
        assert result.leverage_curve_ref is None
        assert result.portfolio_returns_ref is None
        assert _written(runs_root) == []

    def test_the_published_returns_are_a_return_series(self, swinging_pair):
        result = _simulation(
            swinging_pair, {"AAPL": 0.5, "MSFT": -0.5}, run_id="returns-run"
        )
        scored = calculate_series_metrics(
            SeriesMetricsInput(
                series={"ref": result.portfolio_returns_ref},
                metrics=["annualized_volatility"],
            )
        )
        assert scored.values["annualized_volatility"] == pytest.approx(
            result.annualized_volatility, abs=1e-6
        )


# ── the pair's spread state machine ──────────────────────────────────────


@pytest.fixture
def diverging_pair(monkeypatch):
    """One name dips then spikes against a flat partner, so the state
    machine visits all three of its states."""
    dates = pd.date_range(START, periods=20, freq="B")
    close_a = [100.0] * 5 + [60.0] * 5 + [100.0] * 5 + [140.0] * 5
    frames = {
        "A": _ohlcv(pd.Series(close_a, index=dates)),
        "B": _ohlcv(pd.Series(100.0, index=dates)),
    }
    _serve(frames, monkeypatch)
    return dates


def _pair(dates, **overrides):
    payload = dict(
        symbol_a="A",
        symbol_b="B",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        hedge_ratio=1.0,
        entry_z=1.0,
        exit_z=0.3,
        commission_pct=0.0,
        slippage_pct=0.0,
        zscore_window=None,
    )
    payload.update(overrides)
    return run_pair_trade_backtest(PairTradeBacktestInput(**payload))


class TestTheSpreadStateMachine:
    def test_the_state_resolves_to_long_flat_and_short(self, diverging_pair):
        result = _pair(diverging_pair, run_id="spread-state-run")

        assert handoff.parse(result.state_ref).kind == "analytic_series"
        state = handoff.resolve(result.state_ref, expect="analytic_series")
        assert set(state.unique()) <= {-1.0, 0.0, 1.0}
        # n_round_trips reduced this to one integer; the series says when.
        assert len(state) == len(result.equity_curve)
        assert state.abs().sum() > 0

    def test_the_four_account_curves_travel_with_it(self, diverging_pair):
        result = _pair(diverging_pair, run_id="pair-curves-run")

        for ref in (
            result.cash_curve_ref,
            result.gross_exposure_curve_ref,
            result.net_exposure_curve_ref,
            result.leverage_curve_ref,
        ):
            assert handoff.parse(ref).kind == "analytic_series"
            assert handoff.describe(ref)["producer"] == (
                "backtest.run_pair_trade_backtest"
            )

    def test_without_a_run_id_nothing_is_written(self, diverging_pair, runs_root):
        result = _pair(diverging_pair)

        assert result.state_ref is None
        assert result.cash_curve_ref is None
        assert result.leverage_curve_ref is None
        assert _written(runs_root) == []


# ── the walk-forward's stitched curve ────────────────────────────────────


@pytest.fixture
def one_symbol(monkeypatch):
    rng = np.random.default_rng(4242)
    dates = pd.date_range(START, periods=600, freq="B")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.013, len(dates)))
    frames = {"AAPL": _ohlcv(pd.Series(close, index=dates))}
    _serve(frames, monkeypatch)
    return dates


def _walk_forward(dates, **overrides):
    payload = dict(
        symbol="AAPL",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        strategy="sma_crossover",
        param_grid={"fast_period": [5, 10], "slow_period": [40, 60]},
        train_bars=200,
        test_bars=100,
    )
    payload.update(overrides)
    return run_walk_forward_backtest(WalkForwardInput(**payload))


class TestTheStitchedOutOfSampleCurve:
    def test_the_curve_behind_the_five_scalars_is_publishable(self, one_symbol):
        result = _walk_forward(one_symbol, run_id="walk-forward-run")

        assert handoff.parse(result.stitched_equity_curve_ref).kind == "equity_curve"
        curve = handoff.resolve(result.stitched_equity_curve_ref, "equity_curve")
        assert len(curve) == result.n_windows * 100
        # The scalar is a reduction of this curve, so the two agree.
        assert float(curve.iloc[-1] / 10_000.0 - 1.0) == pytest.approx(
            result.stitched_oos_return, abs=1e-6
        )

    def test_without_a_run_id_nothing_is_written(self, one_symbol, runs_root):
        result = _walk_forward(one_symbol)

        assert result.stitched_equity_curve_ref is None
        assert _written(runs_root) == []


# ── the comparison's sort direction ──────────────────────────────────────


def _compare(dates, sort_by):
    return compare_strategies(
        CompareStrategiesInput(
            symbol="AAPL",
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            sort_by=sort_by,
        )
    )


class TestTheComparisonSortsInTheRightDirection:
    def test_by_volatility_the_quietest_wins(self, one_symbol):
        result = _compare(one_symbol, "annualized_volatility")

        volatilities = [s.annualized_volatility for s in result.strategies]
        assert all(v is not None for v in volatilities)
        assert volatilities == sorted(volatilities)
        assert result.best_strategy == result.strategies[0].strategy
        # The field existed nowhere on the row before, so this sort was a
        # tie between four identical absences.
        assert len(set(volatilities)) > 1

    def test_by_sharpe_the_highest_still_wins(self, one_symbol):
        result = _compare(one_symbol, "sharpe_ratio")

        sharpes = [s.sharpe_ratio for s in result.strategies]
        assert sharpes == sorted(sharpes, reverse=True)
        assert result.best_strategy == result.strategies[0].strategy

    def test_the_other_two_silent_ties_are_fields_now(self, one_symbol):
        for metric in ("profit_factor", "avg_trade_return_pct"):
            result = _compare(one_symbol, metric)
            values = [getattr(s, metric) for s in result.strategies]
            assert all(v is not None for v in values)
            assert values == sorted(values, reverse=True)


# ── the refusal that named an argument the caller did not have ───────────


class TestTheRefusalNamesTheArgumentTheCallerSet:
    def test_a_stated_weight_over_the_cap_names_max_position_pct(self, swinging_pair):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError) as excinfo:
            _simulation(
                swinging_pair, {"AAPL": 0.9, "MSFT": -0.05}, max_position_pct=0.5
            )

        message = str(excinfo.value)
        assert "max_position_pct=0.5" in message
        # The shared validator's own spelling still appears, named as
        # what it is rather than as an argument to pass.
        assert "max_abs_weight" in message
        assert "signal_type='score'" in message


# ── the two kinds the curves travel as ───────────────────────────────────


class TestTheKindTableAdvertisesWhatItCanMint:
    def test_both_analytic_kinds_are_listed(self):
        listed = {
            k.kind: k.description
            for k in list_reference_kinds(ListReferenceKindsInput()).kinds
        }

        assert "analytic_series" in listed
        assert "analytic_frame" in listed
        assert listed["analytic_series"].strip()
        assert listed["analytic_frame"].strip()

    def test_their_storage_matches_what_resolve_returns(self):
        assert handoff.KINDS["analytic_series"]["storage"] == "series"
        assert handoff.KINDS["analytic_frame"]["storage"] == "frame"

    def test_neither_is_external(self):
        # Both have producers in this library, so they publish from
        # memory rather than being registered by path.
        assert "analytic_series" not in handoff.EXTERNAL_KINDS
        assert "analytic_frame" not in handoff.EXTERNAL_KINDS

    def test_a_frame_round_trips_as_an_analytic_frame(self):
        frame = pd.DataFrame(
            {"Beta": [1.0, 1.1], "Kalman_Gain": [0.2, 0.3]},
            index=pd.date_range(START, periods=2, freq="B"),
        )
        ref = handoff.publish(frame, "analytic_frame", "analytic-frame-run", "path")
        resolved = handoff.resolve(ref, expect="analytic_frame")
        # check_freq=False: Parquet stores the timestamps, not the
        # DatetimeIndex's inferred frequency.
        pd.testing.assert_frame_equal(resolved, frame, check_freq=False)
