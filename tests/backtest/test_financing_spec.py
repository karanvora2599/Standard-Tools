"""
The full financing terms: a benchmark that may move, the spreads a broker
keeps around it, a borrow rate per asset, and where short proceeds sit.

Each test plants an answer that follows from the arithmetic of a financed
account -- $5m borrowed for 30/360 of a year at 5.80% is 24,166.67, a
constant k-times book returns k*R - (k - 1)*f on every bar whatever f does,
two shorts at 100 and 300 bps pay notional * bps * days / basis each -- and
pins the refusals: a dated benchmark with a gap, a bar before the first
rate, a short with no borrow rate, 30/360, and the scalar rates passed
beside the full terms. With no terms and zero rates nothing moves: the cash
of a book is constant between rebalances to the bit, on both engine paths,
under every fill.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as engine
import standard_quant_tools.backtest.portfolio_engine as pe
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.backtest.financing import (
    FinancingSpec,
    FinancingSpecInput,
    calendar_day_gaps,
)
from standard_quant_tools.backtest.pairs import run_pair_backtest
from standard_quant_tools.backtest.panel import run_signal_panel_backtest
from standard_quant_tools.error import ValidationError

requires_cpp = pytest.mark.skipif(not pe.HAS_CPP, reason="_sqt_core not built")


# ── fixtures ────────────────────────────────────────────────────────────


def _frame(close, index):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(close), 1e7),
        },
        index=index,
    )


def _walk(n=90, seed=3, start="2024-01-02"):
    index = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    return _frame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index), index


def _moving_rates(index, start=0.03, step=0.0004):
    """A benchmark that moves every day, on every bar date."""
    return {str(d.date()): start + step * i for i, d in enumerate(index)}


def _both(fn, module, *args, **kwargs):
    """Run once natively and once through the Python fallback."""
    native = fn(*args, **kwargs)
    saved = module.HAS_CPP
    module.HAS_CPP = False
    try:
        looped = fn(*args, **kwargs)
    finally:
        module.HAS_CPP = saved
    return native, looped


_FREE = dict(commission_pct=0.0, slippage_pct=0.0)


# ── the planted debit ───────────────────────────────────────────────────


class TestADebitAccruesBenchmarkPlusSpread:
    """$10m long on $5m of equity borrows $5m. At a 5.30% benchmark with a
    50 bp debit spread, 30 calendar days on ACT/360 cost 5,000,000 x 5.80%
    x 30 / 360 = 24,166.67."""

    def test_on_both_paths(self):
        index = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])
        data = {"A": _frame([100.0, 100.0], index)}
        weights = pd.DataFrame({"A": [2.0]}, index=index[:1])
        spec = FinancingSpec(0.053, "ACT/360", debit_spread_bps=50.0)
        for result in _both(
            pe.run_portfolio_simulation,
            pe,
            data,
            weights,
            initial_capital=5_000_000.0,
            max_gross_leverage=2.0,
            max_position_pct=2.0,
            financing=spec,
            **_FREE,
        ):
            charged = result["financing"]["interest_charged"]
            assert abs(charged - 5_000_000.0 * 0.058 * 30 / 360) < 1e-9
            assert round(charged, 2) == 24_166.67
            assert result["final_equity"] == pytest.approx(
                5_000_000.0 - 24_166.666666666667, abs=1e-6
            )
            report = result["financing"]
            assert report["source"] == "spec"
            assert report["margin_interest_rate"] == pytest.approx(0.058)
            assert report["debit_spread_bps"] == 50.0
            assert report["benchmark"]["kind"] == "scalar"


# ── leverage invariance under a moving benchmark ────────────────────────


class TestLeverageInvarianceUnderADatedBenchmark:
    """A k-times book rebalanced to k at every close, financed at the
    benchmark with no spreads, returns k*R - (k - 1)*f on every bar, where
    f is the benchmark in force on the previous bar over the calendar days
    since it. It holds whether the k-1 is borrowed (k > 1) or lent (k < 1),
    and on a benchmark that moves every day."""

    @pytest.mark.parametrize("k", [0.0, 0.5, 2.0, 3.0])
    def test_the_shared_cash_engine(self, k):
        frame, index = _walk(n=40)
        rates = _moving_rates(index)
        spec = FinancingSpec(rates, "ACT/360")
        weights = pd.DataFrame({"A": np.full(len(index), k)}, index=index)
        r = frame["Close"].pct_change().to_numpy()
        days = calendar_day_gaps(index)
        prev_rate = np.array([rates[str(d.date())] for d in index])
        for result in _both(
            pe.run_portfolio_simulation,
            pe,
            {"A": frame},
            weights,
            max_gross_leverage=3.0,
            max_position_pct=3.0,
            financing=spec,
            **_FREE,
        ):
            equity = result["equity_curve"].to_numpy()
            got = equity[1:] / equity[:-1] - 1.0
            f = prev_rate[:-1] * days[1:] / 360.0
            want = k * r[1:] - (k - 1.0) * f
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("k", [0.0, 0.5, 2.0, 3.0])
    def test_run_strategy(self, k):
        frame, index = _walk(n=40, seed=8)
        rates = _moving_rates(index, start=0.04, step=-0.0003)
        spec = FinancingSpec(rates, "ACT/360")
        r = frame["Close"].pct_change().to_numpy()
        days = calendar_day_gaps(index)
        prev_rate = np.array([rates[str(d.date())] for d in index])
        signal = pd.Series(k, index=index)
        for result in _both(
            engine.run_strategy, engine, frame, signal, financing=spec, **_FREE
        ):
            equity = result["equity_curve"].to_numpy()
            got = equity[1:] / equity[:-1] - 1.0
            f = prev_rate[:-1] * days[1:] / 360.0
            want = k * r[1:] - (k - 1.0) * f
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


# ── a constant dated benchmark is the scalar ────────────────────────────


class TestAConstantDatedBenchmarkIsTheScalar:
    """The same rate stated once or on every date accrues the same doubles:
    the bits of every curve and every total agree."""

    def _specs(self, index):
        common = dict(
            credit_spread_bps=25.0,
            debit_spread_bps=50.0,
            default_borrow_bps=100.0,
        )
        return (
            FinancingSpec(0.05, "ACT/360", **common),
            FinancingSpec({str(d.date()): 0.05 for d in index}, "ACT/360", **common),
        )

    def test_the_shared_cash_engine(self):
        frame, index = _walk(n=60)
        weights = pd.DataFrame({"A": [1.5, 0.4, -0.7]}, index=index[[0, 20, 40]])
        kwargs = dict(max_gross_leverage=2.0, max_position_pct=2.0)
        for native in (True, False):
            saved = pe.HAS_CPP
            pe.HAS_CPP = native and saved
            try:
                scalar, dated = (
                    pe.run_portfolio_simulation(
                        {"A": frame}, weights, financing=s, **kwargs
                    )
                    for s in self._specs(index)
                )
            finally:
                pe.HAS_CPP = saved
            for key in ("equity_curve", "cash_curve"):
                assert np.array_equal(scalar[key].to_numpy(), dated[key].to_numpy())
            for key in ("interest_credited", "interest_charged", "borrow_charged"):
                assert scalar["financing"][key] == dated["financing"][key]
            assert np.array_equal(
                scalar["benchmark_accrual"].to_numpy(),
                dated["benchmark_accrual"].to_numpy(),
            )

    def test_run_strategy(self):
        frame, index = _walk(n=60, seed=4)
        signal = pd.Series(np.where(np.arange(60) % 15 < 8, 2.0, -0.5), index=index)
        for native in (True, False):
            saved = engine.HAS_CPP
            engine.HAS_CPP = native and saved
            try:
                scalar, dated = (
                    engine.run_strategy(frame, signal, financing=s)
                    for s in self._specs(index)
                )
            finally:
                engine.HAS_CPP = saved
            assert np.array_equal(
                scalar["equity_curve"].to_numpy(), dated["equity_curve"].to_numpy()
            )
            assert scalar["sharpe_ratio"] == dated["sharpe_ratio"]


# ── borrow per asset ────────────────────────────────────────────────────


class TestEachShortPaysItsOwnBorrowRate:
    """Two shorts at constant prices, one at 100 bps and one at 300: each
    accrues notional x bps x calendar days / basis, the account pays both,
    and the report says which paid what."""

    def test_on_both_paths(self):
        index = pd.DatetimeIndex(["2024-01-02", "2024-02-01", "2024-03-03"])
        data = {
            "A": _frame([100.0] * 3, index),
            "B": _frame([40.0] * 3, index),
        }
        weights = pd.DataFrame({"A": [-0.5], "B": [-0.5]}, index=index[:1])
        spec = FinancingSpec(0.0, "ACT/365F", borrow_bps={"A": 100.0, "B": 300.0})
        for result in _both(
            pe.run_portfolio_simulation,
            pe,
            data,
            weights,
            initial_capital=1_000_000.0,
            financing=spec,
            **_FREE,
        ):
            report = result["financing"]
            by_asset = report["borrow_charged_by_asset"]
            assert by_asset["A"] == pytest.approx(500_000 * 0.01 * 61 / 365, rel=1e-12)
            assert by_asset["B"] == pytest.approx(500_000 * 0.03 * 61 / 365, rel=1e-12)
            assert report["borrow_charged"] == pytest.approx(
                by_asset["A"] + by_asset["B"], rel=1e-14
            )
            assert report["borrow_bps_by_asset"] == {"A": 100.0, "B": 300.0}
            assert report["borrow_fee_bps"] is None  # two rates, not one

    def test_each_leg_of_a_pair_pays_its_own(self):
        index = pd.bdate_range("2024-01-02", periods=80)
        rng = np.random.default_rng(0)
        common = rng.normal(0.0, 0.01, 80)
        data = {
            "A": _frame(100 * np.cumprod(1 + common + rng.normal(0, 0.004, 80)), index),
            "B": _frame(100 * np.cumprod(1 + common + rng.normal(0, 0.004, 80)), index),
        }
        spec = FinancingSpec(0.02, "ACT/360", borrow_bps={"A": 100.0, "B": 300.0})
        out = run_pair_backtest(
            data, "A", "B", 1.0, entry_z=1.0, exit_z=0.2, financing=spec, **_FREE
        )
        report = out["financing"]
        assert set(report["borrow_bps_by_asset"]) <= {"A", "B"}
        for leg, bps in report["borrow_bps_by_asset"].items():
            assert bps == {"A": 100.0, "B": 300.0}[leg]
            assert report["borrow_charged_by_asset"][leg] > 0.0
        # A borrow rate stated per leg leaves nothing to warn about.
        assert not any("borrow_fee_bps=0" in w for w in out["warnings"])


# ── the two paths agree ─────────────────────────────────────────────────


@requires_cpp
class TestTheKernelAndTheLoopAgree:
    """The native kernel and the Python loop read the same resolved arrays:
    on one name they agree bit for bit under a dated benchmark; on many,
    within the tolerance their summation orders leave."""

    def test_one_name_bit_for_bit(self):
        frame, index = _walk(n=120, seed=11)
        spec = FinancingSpec(
            _moving_rates(index),
            "ACT/360",
            credit_spread_bps=20.0,
            debit_spread_bps=60.0,
            borrow_bps={"A": 150.0},
            short_proceeds="segregated",
            rebate_spread_bps=40.0,
        )
        weights = pd.DataFrame(
            {"A": [1.8, -0.6, 0.3, -1.2]}, index=index[[0, 30, 60, 90]]
        )
        native, looped = _both(
            pe.run_portfolio_simulation,
            pe,
            {"A": frame},
            weights,
            max_gross_leverage=2.0,
            max_position_pct=2.0,
            financing=spec,
        )
        for key in ("equity_curve", "cash_curve", "gross_exposure_curve"):
            assert np.array_equal(native[key].to_numpy(), looped[key].to_numpy()), key
        for key in (
            "interest_credited",
            "interest_charged",
            "borrow_charged",
            "rebate_credited",
            "debit_balance_years",
        ):
            assert native["financing"][key] == looped["financing"][key], key
        assert native["financing"]["borrow_charged_by_asset"] == (
            looped["financing"]["borrow_charged_by_asset"]
        )

    @pytest.mark.parametrize("fill", ["close", "next_open", "hl2_exploratory"])
    def test_many_names_within_tolerance(self, fill):
        rng = np.random.default_rng(5)
        index = pd.bdate_range("2023-01-02", periods=150)
        data = {
            f"T{i}": _frame(100 * np.cumprod(1 + rng.normal(0, 0.012, 150)), index)
            for i in range(10)
        }
        reb = index[::10][:-1]
        w = rng.normal(0, 1, (len(reb), 10))
        w = w / np.abs(w).sum(axis=1, keepdims=True) * 1.8
        weights = pd.DataFrame(w, index=reb, columns=list(data))
        spec = FinancingSpec(
            _moving_rates(index, 0.02, 0.0001),
            "ACT/365F",
            benchmark_fill="refuse",
            credit_spread_bps=10.0,
            debit_spread_bps=40.0,
            borrow_bps={f"T{i}": 50.0 + 30.0 * i for i in range(10)},
        )
        native, looped = _both(
            pe.run_portfolio_simulation,
            pe,
            data,
            weights,
            max_gross_leverage=2.0,
            max_position_pct=2.0,
            fill_price=fill,
            financing=spec,
        )
        scale = float(np.nanmax(np.abs(looped["gross_exposure_curve"].to_numpy())))
        for key in ("equity_curve", "cash_curve"):
            np.testing.assert_allclose(
                native[key].to_numpy(),
                looped[key].to_numpy(),
                rtol=1e-13,
                atol=1e-13 * scale,
            )
        for key in ("interest_charged", "borrow_charged", "interest_credited"):
            assert native["financing"][key] == pytest.approx(
                looped["financing"][key], rel=1e-12
            )


# ── the dated benchmark's gaps ──────────────────────────────────────────


class TestADatedBenchmarksGaps:
    """Under 'refuse' a bar a balance is carried from needs its own rate,
    and the refusal names the first date without one; under 'previous' the
    last rate carries forward, and a bar before the first rate is refused."""

    def _run(self, spec, frame):
        weights = pd.DataFrame({"A": [0.5]}, index=frame.index[:1])
        return pe.run_portfolio_simulation({"A": frame}, weights, financing=spec)

    def test_refuse_names_the_gap(self):
        frame, index = _walk(n=20)
        rates = _moving_rates(index)
        gap = str(index[7].date())
        del rates[gap]
        with pytest.raises(ValidationError, match=f"bar dated {gap}"):
            self._run(FinancingSpec(rates, "ACT/360"), frame)

    def test_previous_carries_the_last_rate(self):
        frame, index = _walk(n=20)
        rates = _moving_rates(index)
        del rates[str(index[7].date())]
        out = self._run(
            FinancingSpec(rates, "ACT/360", benchmark_fill="previous"), frame
        )
        accrual = out["benchmark_accrual"]
        # Bar 8 accrues at the rate in force on bar 7 -- bar 6's, carried.
        days = (index[8] - index[7]).days
        assert accrual.iloc[8] == pytest.approx(
            rates[str(index[6].date())] * days / 360, rel=1e-15
        )
        assert out["financing"]["benchmark"]["n_bars_filled"] == 1

    def test_previous_refuses_a_bar_before_the_first_rate(self):
        frame, index = _walk(n=20)
        rates = {
            k: v for k, v in _moving_rates(index).items() if k >= str(index[3].date())
        }
        with pytest.raises(ValidationError, match=f"bar dated {index[0].date()}"):
            self._run(FinancingSpec(rates, "ACT/360", benchmark_fill="previous"), frame)

    def test_the_last_bar_needs_no_rate(self):
        frame, index = _walk(n=20)
        rates = _moving_rates(index)
        del rates[str(index[-1].date())]
        out = self._run(FinancingSpec(rates, "ACT/360"), frame)
        bench = out["financing"]["benchmark"]
        assert bench["last_date"] == str(index[-2].date())
        assert bench["first"] == rates[str(index[0].date())]


# ── the refusals ────────────────────────────────────────────────────────


class TestTheRefusals:
    def test_thirty_360_is_refused_with_its_reason(self):
        for spelling in ("30/360", "30E/360", "30/360 US"):
            with pytest.raises(ValidationError, match="calendar days"):
                FinancingSpec(0.05, spelling)
        with pytest.raises(Exception, match="calendar days"):
            FinancingSpecInput(benchmark_rate=0.05, day_count="30/360")

    def test_the_day_count_is_required(self):
        with pytest.raises(Exception, match="day_count"):
            FinancingSpecInput(benchmark_rate=0.05)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"benchmark_rate": 5.0},
            {"benchmark_rate": float("nan")},
            {"benchmark_rate": {"2024-01-02": 1.5}},
            {"credit_spread_bps": -1.0},
            {"borrow_bps": {"A": float("inf")}},
            {"rebate_spread_bps": 10.0},
        ],
    )
    def test_out_of_range_terms(self, kwargs):
        base = {"benchmark_rate": 0.05, "day_count": "ACT/360"}
        with pytest.raises(Exception):
            FinancingSpecInput(**{**base, **kwargs})
        with pytest.raises(ValidationError):
            FinancingSpec(**{**base, **kwargs})

    def test_a_short_with_no_borrow_rate_is_refused_by_name(self):
        frame, index = _walk(n=20)
        weights = pd.DataFrame({"A": [-0.5]}, index=index[:1])
        spec = FinancingSpec(0.05, "ACT/360", borrow_bps={"B": 50.0})
        with pytest.raises(ValidationError, match="'A'"):
            pe.run_portfolio_simulation({"A": frame}, weights, financing=spec)
        with pytest.raises(ValidationError, match="'A'"):
            engine.run_strategy(
                frame, pd.Series(-1.0, index=index), financing=spec, asset="A"
            )
        # A long-only book needs no borrow rate at all.
        pe.run_portfolio_simulation({"A": frame}, weights.abs(), financing=spec)

    @pytest.mark.parametrize(
        "scalar",
        [
            {"cash_rate": 0.04},
            {"margin_interest_rate": 0.06},
            {"borrow_fee_bps": 30.0},
            {"day_count": "ACT/360"},
            {"short_proceeds": "segregated"},
        ],
    )
    def test_scalars_beside_a_spec_are_refused_by_the_engine(self, scalar):
        frame, index = _walk(n=20)
        weights = pd.DataFrame({"A": [0.5]}, index=index[:1])
        name = next(iter(scalar))
        with pytest.raises(ValidationError, match=name):
            pe.run_portfolio_simulation(
                {"A": frame},
                weights,
                financing=FinancingSpec(0.05, "ACT/360"),
                **scalar,
            )

    @pytest.mark.parametrize(
        "scalar",
        [
            {"risk_free_rate": 0.05},
            {"cash_rate": 0.04},
            {"financing_rate": 0.06},
            {"borrow_fee_bps": 30.0},
        ],
    )
    def test_scalars_beside_a_spec_are_refused_by_run_strategy(self, scalar):
        frame, index = _walk(n=20)
        name = next(iter(scalar))
        with pytest.raises(ValidationError, match=name):
            engine.run_strategy(
                frame,
                pd.Series(1.0, index=index),
                financing=FinancingSpec(0.05, "ACT/360"),
                **scalar,
            )
        with pytest.raises(ValidationError, match=name):
            engine.backtest_grid(
                frame,
                "sma_crossover",
                {"fast_period": [3], "slow_period": [8]},
                financing=FinancingSpec(0.05, "ACT/360"),
                **scalar,
            )

    def test_run_strategy_refuses_segregated_proceeds(self):
        frame, index = _walk(n=20)
        spec = FinancingSpec(0.05, "ACT/360", short_proceeds="segregated")
        with pytest.raises(ValidationError, match="segregated"):
            engine.run_strategy(frame, pd.Series(1.0, index=index), financing=spec)


# ── with no spec and zero rates nothing moves ───────────────────────────


_BOOKS = {
    "long": [0.6, 0.9, 0.3],
    "levered": [1.8, 1.2, 2.0],
    "short": [-0.7, -0.4, -1.0],
    "flat": [0.0, 0.0, 0.0],
}


class TestNothingMovesAtZero:
    """With no terms and every rate at zero the financing block does not
    run: a book's cash changes only where it trades, to the bit, on both
    paths, under every fill and for levered, short and flat books -- and a
    spec whose benchmark, spreads and borrow are all zero is the same run."""

    @pytest.mark.parametrize("fill", ["close", "next_open", "hl2_exploratory"])
    @pytest.mark.parametrize("book", list(_BOOKS))
    def test_the_shared_cash_engine(self, fill, book):
        frame, index = _walk(n=60, seed=2)
        reb = index[[0, 20, 40]]
        weights = pd.DataFrame({"A": _BOOKS[book]}, index=reb)
        kwargs = dict(
            fill_price=fill,
            max_gross_leverage=2.0,
            max_position_pct=2.0,
        )
        zero_spec = FinancingSpec(0.0, "ACT/365F", default_borrow_bps=0.0)
        runs = []
        for financing in (None, zero_spec):
            runs.extend(
                _both(
                    pe.run_portfolio_simulation,
                    pe,
                    {"A": frame},
                    weights,
                    financing=financing,
                    **kwargs,
                )
            )
        traded = set(index.get_indexer(reb))
        if fill == "next_open":
            traded = {b + 1 for b in traded}
        for out in runs:
            cash = out["cash_curve"].to_numpy()
            for bar in range(1, len(cash)):
                if bar not in traded:
                    assert cash[bar] == cash[bar - 1], bar
            for key in (
                "interest_credited",
                "interest_charged",
                "borrow_charged",
                "rebate_credited",
            ):
                assert out["financing"][key] == 0.0
        reference = runs[0]
        for out in runs[1:]:
            for key in ("equity_curve", "cash_curve"):
                np.testing.assert_allclose(
                    out[key].to_numpy(), reference[key].to_numpy(), rtol=1e-13
                )
        # The spec at zero is the no-spec run on the same path, to the bit.
        for a, b in ((runs[0], runs[2]), (runs[1], runs[3])):
            assert np.array_equal(
                a["equity_curve"].to_numpy(), b["equity_curve"].to_numpy()
            )

    @pytest.mark.parametrize("fill", ["close", "next_open", "hl2_exploratory"])
    @pytest.mark.parametrize("level", [1.0, 2.0, -1.0, 0.0])
    def test_run_strategy(self, fill, level):
        frame, index = _walk(n=60, seed=6)
        signal = pd.Series(np.where(np.arange(60) % 12 < 6, level, 0.0), index=index)
        zero_spec = FinancingSpec(0.0, "ACT/365F", default_borrow_bps=0.0)
        for native in (True, False):
            saved = engine.HAS_CPP
            engine.HAS_CPP = native and saved
            try:
                plain = engine.run_strategy(frame, signal, fill_price=fill)
                spec = engine.run_strategy(
                    frame, signal, fill_price=fill, financing=zero_spec
                )
            finally:
                engine.HAS_CPP = saved
            assert np.array_equal(
                plain["equity_curve"].to_numpy(), spec["equity_curve"].to_numpy()
            )
            for key in ("sharpe_ratio", "sortino_ratio", "total_return"):
                a, b = plain[key], spec[key]
                assert (math.isnan(a) and math.isnan(b)) or a == b, key

    def test_the_close_fill_curve_is_the_cash_account_arithmetic(self):
        """At a zero rate a bar returns the lagged position times the
        close-to-close move less the cost of changing it, and nothing else."""
        frame, index = _walk(n=40, seed=9)
        signal = pd.Series(np.where(np.arange(40) % 10 < 5, 1.0, -1.0), index=index)
        close = frame["Close"].to_numpy()
        held = np.concatenate([[0.0], signal.to_numpy()[:-1]])
        change = np.abs(np.diff(np.concatenate([[0.0], held])))
        bar = np.zeros(40)
        bar[1:] = held[1:] * (close[1:] / close[:-1] - 1.0)
        bar -= change * 0.0015
        sequential = [10_000.0]
        for x in bar[1:]:
            sequential.append(sequential[-1] * (1.0 + x))
        native, looped = _both(engine.run_strategy, engine, frame, signal)
        np.testing.assert_allclose(
            native["equity_curve"].to_numpy(), sequential, rtol=1e-14
        )
        np.testing.assert_allclose(
            looped["equity_curve"].to_numpy(),
            10_000.0 * np.cumprod(1.0 + bar),
            rtol=1e-14,
        )


# ── the Sharpe measures against the benchmark ───────────────────────────


class TestTheSharpeMeasuresAgainstTheBenchmark:
    """Under a spec the benchmark is the rate the Sharpe subtracts, bar by
    bar as it accrued, so a book that holds only cash at no credit spread
    has an excess of exactly zero on every bar."""

    def test_run_strategy_flat_book(self):
        frame, index = _walk(n=60)
        spec = FinancingSpec(_moving_rates(index), "ACT/360")
        for out in _both(
            engine.run_strategy,
            engine,
            frame,
            pd.Series(0.0, index=index),
            financing=spec,
        ):
            assert out["total_return"] > 0.0
            assert out["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9) or math.isnan(
                out["sharpe_ratio"]
            )

    def test_the_grid_ranks_on_the_same_sharpe_as_the_single_run(self):
        frame, index = _walk(n=150, seed=12)
        spec = FinancingSpec(_moving_rates(index, 0.02, 0.0001), "ACT/360")
        grid = engine.backtest_grid(
            frame,
            "sma_crossover",
            {"fast_period": [5], "slow_period": [20]},
            financing=spec,
        )
        saved = engine.HAS_CPP
        engine.HAS_CPP = False
        try:
            loop = engine.backtest_grid(
                frame,
                "sma_crossover",
                {"fast_period": [5], "slow_period": [20]},
                financing=spec,
                n_workers=1,
            )
        finally:
            engine.HAS_CPP = saved
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        single = engine.run_strategy(
            frame,
            STRATEGY_REGISTRY["sma_crossover"](frame, fast_period=5, slow_period=20),
            financing=spec,
        )
        for table in (grid, loop):
            assert table["sharpe_ratio"].iloc[0] == pytest.approx(
                single["sharpe_ratio"], abs=1e-4
            )
            assert table["total_return"].iloc[0] == pytest.approx(
                single["total_return"], abs=1e-6
            )


class TestThePanelsCombinedSharpeChargesNoOpeningBar:
    """A one-ticker panel at weight 1 is the ticker: its combined Sharpe is
    the ticker's own. The combined row subtracted rate / periods on the
    opening bar too, which the ticker's row does not; at 5% on this 120-bar
    book it read 0.2206 beside the ticker's 0.2246 (Sortino 0.3058 against
    0.3113)."""

    def test_one_ticker_reads_its_own_sharpe(self):
        frame, index = _walk(n=120, seed=21)
        signal = pd.DataFrame(
            {"A": np.where(np.arange(120) % 20 < 10, 1.0, 0.0)}, index=index
        )
        out = run_signal_panel_backtest(
            {"A": frame}, signal, risk_free_rate=0.05, **_FREE
        )
        assert out["portfolio_metrics"]["sharpe_ratio"] == pytest.approx(
            out["per_ticker"]["A"]["sharpe_ratio"], abs=2e-4
        )
        assert out["portfolio_metrics"]["sortino_ratio"] == pytest.approx(
            out["per_ticker"]["A"]["sortino_ratio"], abs=2e-4
        )

    def test_a_zero_rate_moves_nothing(self):
        from standard_quant_tools.portfolio.portfolio import portfolio_metrics

        frame, index = _walk(n=60, seed=22)
        signal = pd.DataFrame({"A": np.ones(60)}, index=index)
        out = run_signal_panel_backtest({"A": frame}, signal, **_FREE)
        returns = out["per_ticker"]["A"]["equity_curve"].pct_change().fillna(0.0)
        plain = portfolio_metrics(returns.to_frame("A"), [1.0], periods_per_year=252)
        assert out["portfolio_metrics"]["sharpe_ratio"] == plain["sharpe_ratio"]

    def test_each_ticker_pays_its_own_borrow_under_a_spec(self):
        frame, index = _walk(n=60, seed=23)
        signal = pd.DataFrame({"A": -np.ones(60), "B": -np.ones(60)}, index=index)
        spec = FinancingSpec(0.0, "ACT/360", borrow_bps={"A": 100.0, "B": 400.0})
        out = run_signal_panel_backtest(
            {"A": frame, "B": frame}, signal, financing=spec, **_FREE
        )
        a = out["per_ticker"]["A"]["total_return"]
        b = out["per_ticker"]["B"]["total_return"]
        assert b < a  # the same short, four times the borrow


# ── the tools ───────────────────────────────────────────────────────────


@pytest.fixture
def frames_tool(monkeypatch):
    """Serve the frames a test hands over to both kinds of fetch."""
    frames = {}

    class _Stub:
        def get_ohlcv(self, symbol, *_a, **_k):
            return frames[symbol]

    monkeypatch.setattr(
        "standard_quant_tools.agent.runtimes.backtest.tools.fetch_ohlcv_panel_sync",
        lambda tickers, *_a, **_k: {t: frames[t] for t in tickers},
    )
    monkeypatch.setattr(
        "standard_quant_tools.agent.runtimes.backtest.tools.DataFactory.get_provider",
        staticmethod(lambda *a, **k: _Stub()),
    )
    return frames


class TestTheTools:
    def test_the_portfolio_simulation_reports_the_terms(self, frames_tool):
        frame, index = _walk(n=60)
        frames_tool["A"] = frame
        rates = _moving_rates(index)
        out = dispatch(
            "run_portfolio_simulation",
            {
                "tickers": ["A"],
                "start_date": str(index[0].date()),
                "end_date": str(index[-1].date()),
                "target_weights": {"A": {str(index[0].date()): 0.0}},
                "commission_pct": 0.0,
                "slippage_pct": 0.0,
                "fill_price": "close",
                "financing": {"benchmark_rate": rates, "day_count": "ACT/360"},
            },
        )
        report = out["financing"]
        assert report["source"] == "spec"
        assert report["benchmark"]["kind"] == "dated"
        assert report["benchmark"]["first"] == rates[str(index[0].date())]
        assert report["benchmark"]["last"] == rates[str(index[-2].date())]
        assert report["cash_rate"] is None
        assert any("cash_rate" in note for note in report["notes"])
        # A book of nothing but cash at no credit spread earns the
        # benchmark exactly: an excess of zero.
        assert out["sharpe_ratio"] in (None, 0.0) or abs(out["sharpe_ratio"]) < 1e-9

    def test_the_portfolio_simulation_refuses_scalars_beside_it(self, frames_tool):
        frame, index = _walk(n=20)
        frames_tool["A"] = frame
        with pytest.raises(Exception, match="risk_free_rate"):
            dispatch(
                "run_portfolio_simulation",
                {
                    "tickers": ["A"],
                    "start_date": str(index[0].date()),
                    "end_date": str(index[-1].date()),
                    "target_weights": {"A": {str(index[0].date()): 0.5}},
                    "risk_free_rate": 0.05,
                    "financing": {"benchmark_rate": 0.05, "day_count": "ACT/360"},
                },
            )

    def test_a_strategy_tool_reaches_the_engine_with_the_three_rates(self, frames_tool):
        """At constant prices a bar returns only its financing: a flat book
        earns cash_rate / 252, a 1x short earns cash_rate on its equity and
        its proceeds (2 x cash_rate / 252) and pays borrow_fee_bps on its
        size, and a 2x book pays financing_rate on the borrowed unit. Each
        accrues on the 29 bars after the opening one."""
        index = pd.bdate_range("2024-01-02", periods=30)
        frames_tool["A"] = _frame(np.full(30, 100.0), index)

        def run(level, **extra):
            payload = {
                "symbol": "A",
                "start_date": str(index[0].date()),
                "end_date": str(index[-1].date()),
                "signals": {str(index[0].date()): level},
                "commission_pct": 0.0,
                "slippage_pct": 0.0,
                **extra,
            }
            return dispatch("run_custom_signal_backtest", payload)

        assert run(0.0)["total_return"] == 0.0
        flat = run(0.0, cash_rate=0.05)
        assert flat["total_return"] == pytest.approx(
            (1 + 0.05 / 252) ** 29 - 1, abs=1e-6
        )
        short = run(-1.0, cash_rate=0.02, borrow_fee_bps=300.0)
        assert short["total_return"] == pytest.approx(
            (1 + 2 * 0.02 / 252 - 0.03 / 252) ** 29 - 1, abs=1e-6
        )
        levered = run(
            2.0, signal_type="target_weight", max_abs_weight=2.0, financing_rate=0.06
        )
        assert levered["total_return"] == pytest.approx(
            (1 - 0.06 / 252) ** 29 - 1, abs=1e-6
        )

    def test_the_run_strategy_tools_take_the_rates_and_not_the_terms(self, frames_tool):
        """The run_strategy-based inputs take the three scalar rates at the
        engine's defaults and do not take the full terms, whose schema every
        tool would carry; the shared-cash tools take both. Buy-and-hold takes
        none: a fully long book never borrows or shorts, and its one idle bar
        earns risk_free_rate, which is cash_rate's default."""
        from standard_quant_tools.agent import models

        held = models.BuyAndHoldInput.model_fields
        assert not {"cash_rate", "financing_rate", "borrow_fee_bps", "financing"} & set(
            held
        )
        for name in (
            "BacktestInput",
            "CompareStrategiesInput",
            "RegimeAdaptiveInput",
            "RegimeAdaptiveWalkForwardInput",
            "WalkForwardInput",
            "BacktestOptInput",
            "CustomSignalBacktestInput",
            "SignalPanelBacktestInput",
            "RobustnessDiagnosticsInput",
        ):
            fields = getattr(models, name).model_fields
            assert fields["cash_rate"].default is None, name
            assert fields["financing_rate"].default is None, name
            assert fields["borrow_fee_bps"].default == 0.0, name
            assert "financing" not in fields, name
        # The tools that run only registry strategies take cash_rate alone:
        # see the next test for why the other two cannot accrue there.
        for name in (
            "BacktestDiagnosticsInput",
            "BacktestCompactInput",
            "CompareCostModelsInput",
            "StrategyMatrixInput",
        ):
            fields = getattr(models, name).model_fields
            assert fields["cash_rate"].default is None, name
            assert not {"financing_rate", "borrow_fee_bps", "financing"} & set(
                fields
            ), name
        for name in ("PortfolioSimulationInput", "PairTradeBacktestInput"):
            assert "financing" in getattr(models, name).model_fields, name
        index = pd.bdate_range("2024-01-02", periods=20)
        frames_tool["A"] = _frame(np.full(20, 100.0), index)
        with pytest.raises(Exception, match="financing"):
            dispatch(
                "run_custom_signal_backtest",
                {
                    "symbol": "A",
                    "start_date": str(index[0].date()),
                    "end_date": str(index[-1].date()),
                    "signals": {str(index[0].date()): 1.0},
                    "financing": {"benchmark_rate": 0.05, "day_count": "ACT/360"},
                },
            )

    def test_every_registry_strategy_is_long_or_flat(self):
        """Why the registry-only tools advertise no financing_rate or
        borrow_fee_bps: every registry strategy's signal is 0 or 1, so a book
        it drives is never short and never above 1x, and neither charge can
        accrue. A strategy added to the registry that shorts or levers fails
        here, and those tools need the two fields back."""
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        rng = np.random.default_rng(7)
        n = 600
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
        index = pd.bdate_range("2020-01-02", periods=n)
        frame = pd.DataFrame(
            {
                "Open": close * (1 + rng.normal(0.0, 0.002, n)),
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": rng.integers(1_000, 10_000, n).astype(float),
            },
            index=index,
        )
        for name, strategy in STRATEGY_REGISTRY.items():
            values = set(np.unique(strategy(frame).dropna().to_numpy()))
            assert values <= {0.0, 1.0}, (name, sorted(values))

    def test_estimate_trade_cost_declares_its_day_count(self):
        payload = {
            "notional": 1_000_000,
            "side": "sell",
            "commission_model": "pct",
            "commission_pct": 0.0,
            "spread_model": "fixed_bps",
            "spread_bps": 0.0,
            "short_borrow_bps": 300,
            "holding_days": 90,
        }
        default = dispatch("estimate_trade_cost", payload)
        act365 = dispatch("estimate_trade_cost", {**payload, "day_count": "ACT/365F"})
        act360 = dispatch("estimate_trade_cost", {**payload, "day_count": "ACT/360"})
        assert default == act365  # the default is the convention it always used
        assert default["holding_bps"] == pytest.approx(300 * 90 / 365, abs=1e-4)
        assert act360["holding_bps"] == pytest.approx(300 * 90 / 360, abs=1e-4)
        with pytest.raises(Exception, match="calendar days"):
            dispatch("estimate_trade_cost", {**payload, "day_count": "30/360"})


class TestTheModelPortfolioSpecTakesTheTerms:
    def test_the_evaluation_resolves_and_measures_against_it(self):
        from standard_quant_tools.modeling.portfolio_eval import (
            _spec_financing,
            _summarize_simulation,
        )
        from standard_quant_tools.modeling.specs import PortfolioSimSpec

        frame, index = _walk(n=80)
        rates = _moving_rates(index)
        spec = PortfolioSimSpec(
            financing={"benchmark_rate": rates, "day_count": "ACT/360"}
        )
        kwargs = _spec_financing(spec)
        assert set(kwargs) == {"financing"}
        weights = pd.DataFrame({"A": [0.0]}, index=index[:1])
        simulation = pe.run_portfolio_simulation(
            {"A": frame}, weights, initial_capital=100_000.0, **kwargs
        )
        metrics = _summarize_simulation(
            simulation, 252, 0.0, 0.0, 0.0, initial_capital=100_000.0
        )
        assert abs(metrics["sharpe_ratio"]) < 1e-9
        assert metrics["cumulative_return"] > 0.0

    def test_the_scalars_beside_it_are_refused(self):
        from standard_quant_tools.modeling.portfolio_eval import _spec_financing
        from standard_quant_tools.modeling.specs import PortfolioSimSpec

        spec = PortfolioSimSpec(
            borrow_fee_bps=20.0,
            financing={"benchmark_rate": 0.05, "day_count": "ACT/360"},
        )
        with pytest.raises(ValidationError, match="borrow_fee_bps"):
            _spec_financing(spec)
        with pytest.raises(Exception, match="calendar days"):
            PortfolioSimSpec(financing={"benchmark_rate": 0.05, "day_count": "30/360"})


@requires_cpp
class TestTheNativeBindingsRefuseTwoStatementsOfARate:
    """Each per-bar array replaces a scalar; a direct call that sets both
    would be financed two ways, and a short or non-finite array would be
    read past its end or turn the account NaN. All refused by name."""

    def test_the_portfolio_kernel(self):
        core = pe._cpp_core
        close = np.full((3, 1), 100.0)
        args = (close, close, np.ones((1, 1)), np.array([0]), np.array([0.0, 1.0, 1.0]))
        with pytest.raises(ValueError, match="cash_rate"):
            core.run_portfolio_simulation(
                *args, cash_rate=0.05, credit_rates=np.full(3, 0.05)
            )
        with pytest.raises(ValueError, match="credit_rates"):
            core.run_portfolio_simulation(*args, credit_rates=np.full(2, 0.05))
        with pytest.raises(ValueError, match="borrow_bps"):
            core.run_portfolio_simulation(*args, borrow_bps=np.array([-1.0]))
        with pytest.raises(ValueError, match="debit_rates"):
            core.run_portfolio_simulation(
                *args, debit_rates=np.array([0.0, np.nan, 0.0])
            )

    def test_the_strategy_kernels(self):
        core = engine._cpp_core
        prices = np.linspace(100.0, 110.0, 5)
        signals = np.ones(5)
        arrays = dict(
            cash_per_bar=np.zeros(5),
            debit_per_bar=np.zeros(5),
            borrow_per_bar=np.zeros(5),
        )
        with pytest.raises(ValueError, match="cash_rate"):
            core.run_strategy(prices, signals, cash_rate=0.05, **arrays)
        with pytest.raises(ValueError, match="risk_free_rate"):
            core.run_strategy(
                prices, signals, risk_free_rate=0.05, benchmark_per_bar=np.zeros(5)
            )
        with pytest.raises(ValueError, match="all three"):
            core.run_strategy(prices, signals, cash_per_bar=np.zeros(5))
        with pytest.raises(ValueError, match="benchmark_per_bar"):
            core.batch_run_strategy(
                prices, signals[None, :], benchmark_per_bar=np.zeros(4)
            )
