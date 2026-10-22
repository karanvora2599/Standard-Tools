"""
Every level metric reads its curve from the level BEFORE the first return.

`(1 + r).cumprod()` is the curve after the first return: it never contains
the 1.0 the first return was earned on. cumulative_return and cagr divide by
a curve's first point and max_drawdown seeds its running peak there, so
every caller that built its curve that way dropped the first period. The
planted series below -- a -20% first day, forty flat days, +30%, forty days
at +0.1% -- has a total return of +8.24%, a CAGR of 27.6% a year (82 daily
returns at 252 a year) and a maximum drawdown of -20%. Before the CHANGELOG
entry of 2026-10-22 the doors reported a CAGR of 156%, no drawdown at all and
a Calmar ratio of infinity, several of them beside a correct total return.

`metrics.equity_from_returns` prepends the start, and every door builds its
curve with it. The sweep runs the one planted series through every door.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.analysis.inference import bootstrap_statistic
from standard_quant_tools.backtest.walk_forward import compute_stitched_metrics
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.delta_one.hedging import hedge_effectiveness
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics import (
    cagr,
    calmar_ratio,
    cumulative_return,
    equity_from_returns,
    max_drawdown,
)

RETURNS = [-0.20] + [0.0] * 40 + [0.30] + [0.001] * 40
N = len(RETURNS)
TOTAL = 0.8 * 1.3 * 1.001**40 - 1.0
# 82 daily returns span 82 periods, at 252 a year.
CAGR = (1.0 + TOTAL) ** (252 / N) - 1.0
MDD = -0.20
CALMAR = CAGR / 0.20


def test_the_planted_answers():
    """The numbers every door is held to, worked by hand: +8.24% total,
    27.6% a year, -20% deep."""
    assert N == 82
    assert TOTAL == pytest.approx(0.082422, abs=1e-6)
    assert CAGR == pytest.approx(0.275575, abs=1e-6)
    assert CALMAR == pytest.approx(1.377877, abs=1e-6)


def _ohlcv(returns, start="2022-01-03"):
    close = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(returns, dtype=float)])
    index = pd.date_range(start, periods=len(close), freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": np.full(len(close), 1e6),
        },
        index=index,
    )


@pytest.fixture
def planted_market(monkeypatch):
    """X and Y each carry the planted path plus and minus an alternating
    0.1%, so their 50/50 portfolio IS the planted path; the benchmark
    wiggles so a beta exists."""
    wiggle = 0.001 * np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    planted = np.asarray(RETURNS)
    frames = {
        "X": _ohlcv(planted + wiggle),
        "Y": _ohlcv(planted - wiggle),
        "SOLO": _ohlcv(planted),
        "SPY": _ohlcv(0.0005 + 0.004 * np.sin(np.arange(N))),
    }

    def _by_symbol(symbol, *args, **kwargs):
        return frames[str(symbol)].copy()

    provider = MagicMock()
    provider.get_ohlcv.side_effect = _by_symbol
    provider.get_ohlcv_async = AsyncMock(side_effect=_by_symbol)
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)
    return frames


DATES = {"start_date": "2022-01-01", "end_date": "2023-01-01"}
BOOK = {"tickers": ["X", "Y"], "weights": [0.5, 0.5], **DATES}


def _series_metrics():
    out = dispatch(
        "calculate_series_metrics",
        {
            "series": {"values": RETURNS},
            "metrics": [
                "cumulative_return",
                "cagr",
                "max_drawdown",
                "calmar_ratio",
                "drawdown_series",
            ],
            "run_id": "first-period",
            "name": "drawdown",
        },
    )
    drawdown = handoff.resolve(out["drawdown_ref"])
    if isinstance(drawdown, pd.DataFrame):
        drawdown = drawdown.iloc[:, 0]
    return [
        ("cumulative_return", out["values"]["cumulative_return"], TOTAL, 1e-9),
        ("cagr", out["values"]["cagr"], CAGR, 1e-9),
        ("max_drawdown", out["values"]["max_drawdown"], MDD, 1e-9),
        ("calmar_ratio", out["values"]["calmar_ratio"], CALMAR, 1e-9),
        # On the returns' bars: the first one is the -20% day.
        ("drawdown_series[0]", float(drawdown.iloc[0]), MDD, 1e-9),
        ("len(drawdown_series)", len(drawdown), N, 0),
    ]


def _drawdown_profile():
    out = dispatch("get_drawdown_profile", {"returns": RETURNS})
    return [
        ("max_drawdown", out["max_drawdown"], MDD, 1e-9),
        ("n_drawdowns", out["n_drawdowns"], 1, 0),
        ("length_days", out["worst_drawdowns"][0]["length_days"], 41, 0),
    ]


def _stock_risk():
    out = dispatch(
        "analyze_stock_risk", {"symbol": "SOLO", "benchmark": "SPY", "period": "1y"}
    )
    return [("max_drawdown", out["max_drawdown"], MDD, 1e-6)]


def _extended_risk():
    out = dispatch(
        "get_extended_risk_metrics", {"symbol": "SOLO", "benchmark": "SPY", **DATES}
    )
    return [
        ("annualized_return", out["annualized_return"], CAGR, 1e-6),
        ("calmar_ratio", out["calmar_ratio"], CALMAR, 1e-4),
    ]


def _risk_attribution():
    out = dispatch(
        "get_portfolio_risk_attribution",
        {**BOOK, "benchmark": "SPY", "n_components": 1},
    )
    return [
        ("annualized_return", out["annualized_return"], CAGR, 1e-4),
        ("max_drawdown", out["max_drawdown"], MDD, 1e-6),
    ]


def _portfolio_analysis():
    out = dispatch("get_portfolio_analysis", {**BOOK, "benchmark": "SPY"})
    return [
        ("annualized_return", out["annualized_return"], CAGR, 1e-6),
        ("max_drawdown", out["max_drawdown"], MDD, 1e-6),
        ("calmar_ratio", out["calmar_ratio"], CALMAR, 1e-4),
        ("total_return", out["total_return"], TOTAL, 1e-6),
    ]


def _stitched():
    dated = pd.Series(RETURNS, index=pd.date_range("2022-01-04", periods=N, freq="B"))
    out = compute_stitched_metrics(
        dated, initial_capital=10_000.0, periods_per_year=252
    )
    return [
        ("total_return", out["total_return"], TOTAL, 1e-9),
        ("max_drawdown", out["max_drawdown"], MDD, 1e-9),
        ("calmar_ratio", out["calmar_ratio"], CALMAR, 1e-9),
    ]


def _bootstrap():
    out = bootstrap_statistic(RETURNS, statistic="max_drawdown", n_bootstrap=200)
    return [("point_estimate", out["point_estimate"], MDD, 1e-9)]


def _hedge():
    # A zero ratio leaves the book as it was, so both sides are the planted
    # path.
    out = hedge_effectiveness(
        portfolio_returns=RETURNS,
        hedge_returns=list(0.004 * np.sin(np.arange(N))),
        hedge_ratio=0.0,
        window=None,
    )
    return [
        ("max_drawdown_before", out["max_drawdown_before"], MDD, 1e-9),
        ("max_drawdown_after", out["max_drawdown_after"], MDD, 1e-9),
    ]


DOORS = {
    "calculate_series_metrics": _series_metrics,
    "get_drawdown_profile": _drawdown_profile,
    "analyze_stock_risk": _stock_risk,
    "get_extended_risk_metrics": _extended_risk,
    "get_portfolio_risk_attribution": _risk_attribution,
    "get_portfolio_analysis": _portfolio_analysis,
    "compute_stitched_metrics": _stitched,
    "bootstrap_statistic": _bootstrap,
    "hedge_effectiveness": _hedge,
}


def test_every_door_counts_the_first_period(planted_market):
    """One planted series through every door that reports a growth rate, a
    drawdown or a Calmar ratio. Each used to read its curve from the first
    bar's close, and reported a 156% CAGR with no drawdown; each must now
    report +8.24%, 27.6% a year and -20%. Every door is read, and every
    wrong one is listed, before the test fails."""
    wrong = []
    for door, read in DOORS.items():
        try:
            rows = read()
        except Exception as exc:  # noqa: BLE001 -- listed, not swallowed
            wrong.append(f"{door}: raised {type(exc).__name__}: {exc}")
            continue
        wrong += [
            f"{door}.{field}: reported {got!r}, expected {want!r}"
            for field, got, want, tol in rows
            if got is None or abs(got - want) > tol
        ]
    assert not wrong, "\n".join(wrong)


class TestTheCurveStartsAtTheStart:
    def test_n_returns_give_n_plus_one_levels(self):
        """The start, then the curve the returns compound into."""
        curve = equity_from_returns(pd.Series([0.10, -0.50, 1.0]))
        assert list(curve) == pytest.approx([1.0, 1.1, 0.55, 1.1])

    def test_the_start_scales_the_curve(self):
        curve = equity_from_returns(pd.Series([0.10, -0.50]), start=10_000.0)
        assert list(curve) == pytest.approx([10_000.0, 11_000.0, 5_500.0])

    def test_the_level_metrics_are_right_on_it_unchanged(self):
        """cumulative_return, cagr, max_drawdown and calmar_ratio read the
        start as the denominator and the first peak, and the elapsed time as
        n periods -- with no change of their own."""
        curve = equity_from_returns(pd.Series(RETURNS))
        assert cumulative_return(curve) == pytest.approx(TOTAL, abs=1e-12)
        assert cagr(curve, 252) == pytest.approx(CAGR, abs=1e-12)
        assert max_drawdown(curve) == pytest.approx(MDD, abs=1e-12)
        assert calmar_ratio(curve, 252) == pytest.approx(CALMAR, abs=1e-12)

    def test_a_business_day_curve_starts_on_the_bar_before(self):
        """Monday's return was earned on Friday's level."""
        index = pd.date_range("2024-01-08", periods=3, freq="B")
        curve = equity_from_returns(pd.Series([0.01, 0.02, 0.03], index=index))
        assert curve.index[0] == pd.Timestamp("2024-01-05")
        assert list(curve.index[1:]) == list(index)

    def test_an_irregular_dated_curve_steps_back_one_gap(self):
        index = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"])
        curve = equity_from_returns(pd.Series([0.01, 0.02, 0.03], index=index))
        assert curve.index[0] == pd.Timestamp("2024-01-02")

    def test_a_positional_curve_starts_at_minus_one(self):
        curve = equity_from_returns(pd.Series([0.01, 0.02]))
        assert list(curve.index) == [-1, 0, 1]

    def test_labels_with_no_before_become_positions(self):
        curve = equity_from_returns(pd.Series([0.01, 0.02], index=["a", "b"]))
        assert list(curve.index) == [0, 1, 2]
        assert list(curve) == pytest.approx([1.0, 1.01, 1.0302])

    def test_one_start_per_row_of_a_matrix(self):
        """The bootstrap's resamples are rows; each gets its own start."""
        paths = equity_from_returns(np.array([[-0.5, 0.0], [0.1, 0.1]]))
        assert paths.shape == (2, 3)
        assert paths[:, 0] == pytest.approx([1.0, 1.0])
        assert paths[0] == pytest.approx([1.0, 0.5, 0.5])

    def test_a_missing_return_is_refused(self):
        """Compounding over a gap would carry the previous level, silently
        assuming the period earned nothing."""
        with pytest.raises(ValidationError, match="missing value"):
            equity_from_returns(pd.Series([0.01, np.nan, 0.02]))

    @pytest.mark.parametrize("start", [0.0, -1.0, float("nan")])
    def test_a_start_that_cannot_be_a_denominator_is_refused(self, start):
        with pytest.raises(ValidationError, match="start"):
            equity_from_returns(pd.Series([0.01]), start=start)


class TestSeriesMetricsReadsLevelsAsLevels:
    def test_cumulative_return_compounds_the_returns(self):
        """Registered as a return metric, it divided the last return by the
        first: [0.01, 0, 0.02] reported 1.0 against a true 0.0302."""
        out = dispatch(
            "calculate_series_metrics",
            {"series": {"values": [0.01, 0.0, 0.02]}, "metrics": ["cumulative_return"]},
        )
        assert out["values"]["cumulative_return"] == pytest.approx(0.0302, abs=1e-12)

    def test_a_losing_first_return_is_not_refused(self):
        """A first return of -20% was refused as a series that "starts at
        -0.2" -- the guard against a non-positive opening LEVEL, applied to
        a return."""
        out = dispatch(
            "calculate_series_metrics",
            {"series": {"values": [-0.20, 0.10]}, "metrics": ["cumulative_return"]},
        )
        assert out["values"]["cumulative_return"] == pytest.approx(-0.12, abs=1e-12)

    def test_cagr_and_calmar_annualize_at_the_periods_asked_for(self):
        """The branch passing periods_per_year sat after the one that did
        not, so both were annualized at 252 whatever was asked: twelve
        monthly returns of 1% at 12 a year reported a CAGR of 11.27."""
        out = dispatch(
            "calculate_series_metrics",
            {
                "series": {"values": [0.01, -0.02] * 6},
                "metrics": ["cagr", "calmar_ratio"],
                "periods_per_year": 12,
            },
        )
        # Twelve months is one year. The peak is the first month's 1.01 and
        # the trough the last month's close.
        one_year = (1.01 * 0.98) ** 6 - 1.0
        deepest = (1.01 * 0.98) ** 6 / 1.01 - 1.0
        assert out["values"]["cagr"] == pytest.approx(one_year, abs=1e-12)
        assert out["values"]["calmar_ratio"] == pytest.approx(
            one_year / abs(deepest), abs=1e-12
        )
