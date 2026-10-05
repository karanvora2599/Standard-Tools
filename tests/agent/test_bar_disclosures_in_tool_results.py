"""
What the provider dropped from or flagged on the bars reaches the result of
every tool that fetched them.

The tools a polling consumer calls every few minutes -- the risk profile,
the technical snapshot, the Hurst exponent and the tail-risk fit -- and the
three fetch tools are run against the real yfinance provider with only
`yfinance.Ticker` stubbed. Each condition is run beside the clean series:

  (a) the placeholder yfinance lists outside market hours for the next
      session: the tools answer, say so, and their numbers equal those of
      the series without that row;
  (b) a hole inside the window: dropped as a missing bar, the same;
  (c) a last bar whose session is still trading: kept, and every result
      says it is not a complete session;
  (d) no Close anywhere: refused once, by type, without a retry.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.data._cache as cache_module
import standard_quant_tools.data.bar_hygiene as hygiene
import standard_quant_tools.data.yfinance_provider as yf_module
from standard_quant_tools.agent.models import (
    AnalysisInput,
    HurstInput,
    TailRiskInput,
    TechnicalInput,
)
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes._shared import HAS_CPP
from standard_quant_tools.agent.runtimes.research.tools import (
    analyze_stock_risk,
    get_tail_risk_metrics,
    get_technical_analysis,
    run_hurst_analysis,
)
from standard_quant_tools.error import NonRetryableAPIError

#: About two and a half years of sessions ending a week ago, so the last bar
#: is settled on the real clock, a one-year lookback lands inside it, and
#: the tail-risk fit has the exceedances it needs.
_LAST_SETTLED = pd.bdate_range(
    pd.Timestamp.today().normalize() - pd.Timedelta(days=13),
    pd.Timestamp.today().normalize() - pd.Timedelta(days=7),
)[-1]
SESSIONS = pd.bdate_range(end=_LAST_SETTLED, periods=600)
START, END = str(SESSIONS[0].date()), str(SESSIONS[-1].date())
NEXT_SESSION = SESSIONS[-1] + pd.offsets.BDay(1)
HOLE = 450

ALL_INDICATORS = [
    "sma",
    "ema",
    "macd",
    "rsi",
    "stochastic",
    "bollinger",
    "atr",
    "obv",
    "vwap",
    "adx",
    "williams_r",
]


def _bars(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(SESSIONS))))
    spread = np.abs(rng.normal(0.0, 0.006, len(SESSIONS)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0.0, 0.002, len(SESSIONS))),
            "High": close * (1 + spread + 0.002),
            "Low": close * (1 - spread - 0.002),
            "Close": close,
            "Volume": rng.uniform(1e6, 5e6, len(SESSIONS)),
        },
        index=SESSIONS,
    )


CLEAN = {"AAPL": _bars(1), "SPY": _bars(2), "NVDA": _bars(3)}


def _with_placeholder(frame: pd.DataFrame) -> pd.DataFrame:
    row = pd.DataFrame(
        {c: [np.nan] for c in ("Open", "High", "Low", "Close")} | {"Volume": [0.0]},
        index=pd.DatetimeIndex([NEXT_SESSION]),
    )
    return pd.concat([frame, row])


def _with_hole(frame: pd.DataFrame) -> pd.DataFrame:
    holed = frame.copy()
    holed.iloc[HOLE, holed.columns.get_loc("Close")] = np.nan
    return holed


class _Market:
    """`yfinance` answering from frames, counting the requests."""

    def __init__(self, frames: Dict[str, pd.DataFrame]) -> None:
        self.calls: List[str] = []
        market = self

        class _Ticker:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

            def history(self, start=None, end=None, interval="1d", **_kw):
                market.calls.append(self.symbol)
                frame = frames[self.symbol].copy()
                index = pd.DatetimeIndex(frame.index)
                keep = np.ones(len(frame), dtype=bool)
                if start is not None:
                    keep &= index >= pd.Timestamp(start).normalize()
                if end is not None:
                    keep &= index < pd.Timestamp(end)
                frame = frame[keep]
                frame.index = pd.DatetimeIndex(frame.index).tz_localize(
                    "America/New_York"
                )
                return frame

        self.Ticker = _Ticker


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No disk tier, so every run below computes from the frames it was
    given rather than from what an earlier run in the test cached."""
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(yf_module, "_is_historical", lambda *_a, **_k: False)
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("standard_quant_tools.data._retry.time.sleep", lambda s: None)
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()
    yield
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()


def _market(monkeypatch, **overrides: pd.DataFrame) -> _Market:
    market = _Market({**CLEAN, **overrides})
    monkeypatch.setattr(yf_module, "yf", market)
    return market


def _run_polled(end: str = END) -> Dict[str, object]:
    return {
        "analyze_stock_risk": analyze_stock_risk(
            AnalysisInput(symbol="AAPL", benchmark="SPY", period="1y")
        ),
        "get_technical_analysis": get_technical_analysis(
            TechnicalInput(
                symbol="AAPL", start_date=START, end_date=end, indicators=ALL_INDICATORS
            )
        ),
        "run_hurst_analysis": run_hurst_analysis(
            HurstInput(symbol="AAPL", start_date=START, end_date=end)
        ),
        "get_tail_risk_metrics": get_tail_risk_metrics(
            TailRiskInput(symbol="AAPL", start_date=START, end_date=end)
        ),
    }


def _numbers(result) -> dict:
    return {k: v for k, v in result.model_dump().items() if k != "warnings"}


def _about(result, symbol: str) -> List[str]:
    return [w for w in result.warnings if w.startswith(f"{symbol}: ")]


# ── (a) the overnight placeholder ────────────────────────────────────────────


class TestTheOvernightPlaceholder:
    def test_every_polled_tool_answers_and_says_what_was_dropped(self, monkeypatch):
        _market(monkeypatch, AAPL=_with_placeholder(CLEAN["AAPL"]))
        results = _run_polled(end=str(NEXT_SESSION.date()))
        for tool, result in results.items():
            (note,) = _about(result, "AAPL")
            assert "at the end of the window had no Close" in note, tool
            assert str(NEXT_SESSION.date()) in note, tool
            assert _about(result, "SPY") == [], tool

    def test_the_numbers_equal_those_of_the_series_without_the_row(self, monkeypatch):
        _market(monkeypatch, AAPL=_with_placeholder(CLEAN["AAPL"]))
        dropped = _run_polled(end=str(NEXT_SESSION.date()))
        _market(monkeypatch)
        clean = _run_polled(end=str(NEXT_SESSION.date()))
        for tool in clean:
            assert _numbers(dropped[tool]) == _numbers(clean[tool]), tool

    def test_a_clean_closed_series_carries_no_warning(self, monkeypatch):
        _market(monkeypatch)
        results = _run_polled()
        for tool, result in results.items():
            assert _about(result, "AAPL") == [], tool
            assert _about(result, "SPY") == [], tool
        for tool in ("analyze_stock_risk", "get_technical_analysis"):
            assert results[tool].warnings == [], tool
        assert results["get_tail_risk_metrics"].warnings == []


# ── (b) a hole inside the window ─────────────────────────────────────────────


class TestAHoleInsideTheWindow:
    def test_every_polled_tool_answers_and_names_the_missing_bar(self, monkeypatch):
        _market(monkeypatch, AAPL=_with_hole(CLEAN["AAPL"]))
        day = str(SESSIONS[HOLE].date())
        for tool, result in _run_polled().items():
            (note,) = _about(result, "AAPL")
            assert "dropped as missing bars" in note and day in note, tool

    def test_the_indicators_equal_those_of_the_series_without_the_row(
        self, monkeypatch
    ):
        _market(monkeypatch, AAPL=_with_hole(CLEAN["AAPL"]))
        holed = _run_polled()
        _market(monkeypatch, AAPL=CLEAN["AAPL"].drop(SESSIONS[HOLE]))
        removed = _run_polled()
        for tool in removed:
            assert _numbers(holed[tool]) == _numbers(removed[tool]), tool

    @pytest.mark.skipif(not HAS_CPP, reason="the gap rule is the native kernels'")
    def test_dropping_the_row_is_what_the_native_recursions_do_with_a_gap(self):
        from standard_quant_tools.indicators.momentum import rsi

        gapped = _with_hole(CLEAN["AAPL"])["Close"]
        dropped = gapped.dropna()
        assert rsi(gapped, 14).dropna().iloc[-1] == pytest.approx(
            rsi(dropped, 14).dropna().iloc[-1], rel=1e-12
        )


# ── (c) a last bar still trading ─────────────────────────────────────────────


class TestALastBarStillTrading:
    def test_every_polled_tool_says_the_last_bar_is_not_a_full_session(
        self, monkeypatch
    ):
        _market(monkeypatch)
        # 09:41 in New York on the last session.
        session = str(SESSIONS[-1].date())
        monkeypatch.setattr(
            hygiene, "_utc_now", lambda: pd.Timestamp(f"{session} 13:41", tz="UTC")
        )
        results = _run_polled()
        for tool, result in results.items():
            (note,) = _about(result, "AAPL")
            assert f"the last bar ({session}) is a session that has not closed" in (
                note
            ), tool
        assert len(_about(results["analyze_stock_risk"], "SPY")) == 1

    def test_after_the_close_nothing_is_said(self, monkeypatch):
        _market(monkeypatch)
        session = str(SESSIONS[-1].date())
        monkeypatch.setattr(
            hygiene, "_utc_now", lambda: pd.Timestamp(f"{session} 21:30", tz="UTC")
        )
        for tool, result in _run_polled().items():
            assert _about(result, "AAPL") == [], tool


# ── (d) no Close anywhere ────────────────────────────────────────────────────


class TestNoCloseAnywhere:
    def test_the_tool_is_refused_once_by_type(self, monkeypatch):
        empty = CLEAN["AAPL"].copy()
        empty["Close"] = np.nan
        market = _market(monkeypatch, AAPL=empty)
        with pytest.raises(NonRetryableAPIError, match="none of them is fully priced"):
            get_technical_analysis(
                TechnicalInput(symbol="AAPL", start_date=START, end_date=END)
            )
        assert market.calls == ["AAPL"]

    def test_a_clean_series_is_fetched_once_too(self, monkeypatch):
        market = _market(monkeypatch)
        get_technical_analysis(
            TechnicalInput(symbol="AAPL", start_date=START, end_date=END)
        )
        assert market.calls == ["AAPL"]


# ── the fetch tools ──────────────────────────────────────────────────────────


def _data(tool: str, **arguments) -> dict:
    return resolve("data").dispatch(tool, arguments)


class TestTheFetchToolsSayItToo:
    def test_fetch_ohlcv(self, monkeypatch):
        _market(monkeypatch, AAPL=_with_placeholder(CLEAN["AAPL"]))
        out = _data(
            "fetch_ohlcv",
            symbol="AAPL",
            start_date=START,
            end_date=str(NEXT_SESSION.date()),
            run_id="r",
            name="bars",
        )
        assert out["rows"] == len(SESSIONS)
        assert any(
            "at the end of the window had no Close" in w for w in out["warnings"]
        )

    def test_fetch_ohlcv_panel_names_each_symbol(self, monkeypatch):
        _market(
            monkeypatch,
            AAPL=_with_placeholder(CLEAN["AAPL"]),
            NVDA=_with_hole(CLEAN["NVDA"]),
        )
        out = _data(
            "fetch_ohlcv_panel",
            tickers=["AAPL", "NVDA", "SPY"],
            start_date=START,
            end_date=str(NEXT_SESSION.date()),
            run_id="r",
            name="panel",
        )
        assert any(w.startswith("AAPL: 1 bar(s) at the end") for w in out["warnings"])
        assert any(
            w.startswith("NVDA: 1 bar(s) inside the window") for w in out["warnings"]
        )
        assert not any(w.startswith("SPY: ") for w in out["warnings"])

    def test_fetch_returns_panel_names_each_symbol(self, monkeypatch):
        _market(
            monkeypatch,
            AAPL=_with_placeholder(CLEAN["AAPL"]),
            NVDA=_with_hole(CLEAN["NVDA"]),
        )
        out = _data(
            "fetch_returns_panel",
            tickers=["AAPL", "NVDA", "SPY"],
            start_date=START,
            end_date=str(NEXT_SESSION.date()),
            run_id="r",
            name="rets",
        )
        assert any(w.startswith("AAPL: 1 bar(s) at the end") for w in out["warnings"])
        assert any(
            w.startswith("NVDA: 1 bar(s) inside the window") for w in out["warnings"]
        )

    def test_a_clean_universe_carries_no_bar_warning(self, monkeypatch):
        _market(monkeypatch)
        for tool, name in (
            ("fetch_ohlcv_panel", "panel"),
            ("fetch_returns_panel", "rets"),
        ):
            out = _data(
                tool,
                tickers=["AAPL", "NVDA", "SPY"],
                start_date=START,
                end_date=END,
                run_id="r",
                name=name,
            )
            assert not any(
                w.split(":")[0] in ("AAPL", "NVDA", "SPY") for w in out["warnings"]
            ), tool
        out = _data(
            "fetch_ohlcv",
            symbol="AAPL",
            start_date=START,
            end_date=END,
            run_id="r",
            name="bars",
        )
        assert out["warnings"] == []

    def test_describe_data_capabilities_fetches_no_bars(self, monkeypatch):
        """The fifth polled tool reads what a provider can serve, not bars,
        so there is nothing for it to disclose."""
        market = _market(monkeypatch)
        resolve("meta").dispatch("describe_data_capabilities", {"source": "yfinance"})
        assert market.calls == []
