"""
A market the surface layer can call without the network, and baselines that
reach the computation.

WHY A FAKE MARKET. The surface tests used to fetch from live yfinance. That
made them slow, nondeterministic and dependent on a connection -- and
offline, every fetch became a typed `DataProviderError` that the
adversarial layer counted as a clean refusal, so everything behind the fetch
went untested without a single failure to say so. `FakeTicker` replaces only
`yfinance.Ticker`; the real provider still runs its interval checks, its
inclusive-end trim, its session cache and its audit data-access records.

WHY PUBLISHED FIXTURES. A synthesized input names no reference, dataset,
model or request id that exists, so about two tools in five refused at the
baseline and every mutation of them re-tested the argument lookup and
nothing else. `publish_fixtures` builds each once per session -- price,
returns, score, signal and weight panels, a covariance, an equity curve and a
trade log, a tick tape, quotes, registered order-book and order-event files,
an indicator panel, a data bundle, a dataset, two ridge models and their
predictions with outcomes -- and `published_baseline` substitutes them into a
tool's
synthesized input by field name, with a short table for the inputs whose
shape the schema cannot describe.
"""

from __future__ import annotations

import copy
import os
import re
import zlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pydantic

from . import synth

SYMBOLS = list(synth.SYMBOLS)

#: The fake market's last day. Fixed, so a run on any date sees the same
#: history.
_LAST_DAY = "2026-09-25"
_TICKER_OK = re.compile(r"^[A-Za-z0-9.^=\-]{1,12}$")
_MASTER: Dict[str, pd.DataFrame] = {}
_INTRADAY = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "90m": "90min",
    "1h": "60min",
}


def _master(symbol: str) -> pd.DataFrame:
    """Daily bars for one symbol: a common market factor plus its own noise,
    seeded by the symbol so every run and every process agrees."""
    key = symbol.upper()
    if key in _MASTER:
        return _MASTER[key]
    days = pd.bdate_range("1995-01-02", _LAST_DAY)
    market = np.random.default_rng(7).normal(0.0003, 0.010, len(days))
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    beta = 0.6 + (zlib.crc32(key.encode()) % 100) / 100.0
    close = 100.0 * np.exp(
        np.cumsum(beta * market + rng.normal(0.0001, 0.009, len(days)))
    )
    open_ = np.concatenate([[close[0]], close[:-1]]) * (
        1 + rng.normal(0, 0.002, len(days))
    )
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(days))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(days))))
    frame = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.uniform(1e6, 5e6, len(days)),
        },
        index=days,
    )
    _MASTER[key] = frame
    return frame


class FakeTicker:
    """`yfinance.Ticker`, answered from `_master`."""

    def __init__(self, symbol: str, *_args: Any, **_kwargs: Any) -> None:
        self.symbol = symbol

    def history(
        self, start=None, end=None, interval="1d", **_kwargs: Any
    ) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        if not _TICKER_OK.match(str(self.symbol)):
            return empty
        master = _master(self.symbol)
        begin = pd.Timestamp(start) if start is not None else master.index[0]
        stop = (
            pd.Timestamp(end)
            if end is not None
            else master.index[-1] + pd.Timedelta(days=1)
        )
        begin = begin.tz_localize(None) if begin.tzinfo is not None else begin
        stop = stop.tz_localize(None) if stop.tzinfo is not None else stop
        daily = master[(master.index >= begin.normalize()) & (master.index < stop)]
        if daily.empty:
            return empty
        if interval in _INTRADAY:
            daily = daily.iloc[-30:]  # yfinance serves only recent intraday bars
            pieces = []
            for day, bar in daily.iterrows():
                index = pd.date_range(
                    day + pd.Timedelta(hours=9, minutes=30),
                    day + pd.Timedelta(hours=15, minutes=59),
                    freq=_INTRADAY[interval],
                )
                rng = np.random.default_rng(zlib.crc32(f"{self.symbol}{day}".encode()))
                path = bar["Open"] * np.exp(np.cumsum(rng.normal(0, 0.001, len(index))))
                pieces.append(
                    pd.DataFrame(
                        {
                            "Open": path,
                            "High": path * 1.0005,
                            "Low": path * 0.9995,
                            "Close": path,
                            "Volume": rng.uniform(1e3, 5e4, len(index)),
                        },
                        index=index,
                    )
                )
            out = pd.concat(pieces)
        elif interval in ("1wk", "5d", "1mo", "3mo"):
            rule = "W-FRI" if interval in ("1wk", "5d") else "ME"
            out = (
                daily.resample(rule)
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )
        else:
            out = daily.copy()
        out.index = out.index.tz_localize("America/New_York")
        return out

    @property
    def info(self) -> Dict[str, Any]:
        if not _TICKER_OK.match(str(self.symbol)):
            return {}
        h = zlib.crc32(self.symbol.upper().encode())
        return {
            "longName": f"{self.symbol.upper()} Synthetic Inc.",
            "sector": ["Technology", "Energy", "Financials", "Consumer Staples"][h % 4],
            "industry": "Synthetic",
            "fullTimeEmployees": 1000 + h % 100000,
            "city": "Nowhere",
            "country": "United States",
            "website": "https://example.invalid",
            "forwardPE": 10 + h % 30,
            "trailingPE": 12 + h % 30,
            "priceToBook": 1 + (h % 50) / 10,
            "debtToEquity": 20 + h % 200,
            "returnOnEquity": 0.05 + (h % 30) / 100,
            "profitMargins": 0.02 + (h % 25) / 100,
            "dividendYield": (h % 40) / 1000,
            "marketCap": 1e9 * (1 + h % 2000),
        }


class FakeYFinance:
    """Only `Ticker` exists; anything else the provider reaches for fails by
    name rather than quietly going to the network."""

    Ticker = FakeTicker

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"yfinance.{name} is not served by the surface tests' market"
        )


class _NoSleep:
    """`time` as the retry decorator sees it: everything but the wait."""

    def sleep(self, _seconds: float) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        import time

        return getattr(time, name)


def install(monkeypatch: Any) -> None:
    """Route yfinance to `FakeTicker` for as long as `monkeypatch` lives.

    The Parquet tier is switched off for the same span, so fake bars are never
    written where a later test outside this layer would read them as real
    ones, and the session tier is emptied on the way in and out so no real
    bars fetched earlier are mixed in. A retry sleeps for nothing: an
    impossible date or a frame that cannot be repaired is refused once, but
    a probe that provokes a transient-looking failure is still retried, and
    three seconds per ticker of backoff would dominate the run.
    """
    import standard_quant_tools.data._cache as cache
    import standard_quant_tools.data._retry as retry
    import standard_quant_tools.data.yfinance_provider as provider

    monkeypatch.setattr(provider, "yf", FakeYFinance())
    monkeypatch.setattr(provider, "_is_historical", lambda *_a, **_k: False)
    # The retry module's own reference to `time`, not the module itself:
    # every other sleep in the process keeps its meaning.
    monkeypatch.setattr(retry, "time", _NoSleep())
    with cache._session_cache_lock:
        cache._session_cache.clear()


def uninstall() -> None:
    import standard_quant_tools.data._cache as cache

    with cache._session_cache_lock:
        cache._session_cache.clear()


# ── published fixtures ──────────────────────────────────────────────────


def _call(runtime: str, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    from standard_quant_tools.agent.runtimes import resolve

    return resolve(runtime).dispatch(tool, arguments)


def publish_fixtures(directory: str) -> Dict[str, Any]:
    """Every reference, dataset, model and record id a baseline can name.

    Each step that fails is recorded under `errors` rather than raised: the
    baselines that needed it then refuse, and `EXPECTED_BASELINE_REFUSAL`
    in the adversarial layer says which those are allowed to be.
    """
    from standard_quant_tools.agent.runtimes import handoff
    from standard_quant_tools.audit.dispatch import last_request_id

    fx: Dict[str, Any] = {"refs": {}, "errors": {}, "paths": {}, "uris": {}}
    start, end = "2019-01-02", "2020-07-31"
    rng = np.random.default_rng(1)

    def step(label: str, fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            fx["errors"][label] = f"{type(exc).__name__}: {exc}"[:400]
            return None

    refs = fx["refs"]
    for kind, tool, name in (
        ("price_panel", "fetch_ohlcv_panel", "bars"),
        ("returns_panel", "fetch_returns_panel", "rets"),
    ):
        out = step(
            kind,
            lambda tool=tool, name=name: _call(
                "data",
                tool,
                dict(
                    tickers=SYMBOLS,
                    start_date=start,
                    end_date=end,
                    run_id="fx",
                    name=name,
                ),
            ),
        )
        if out:
            refs[kind] = out["ref"]

    # A covariance, published by the tool that estimates one, from the
    # returns panel above -- so the allocators' covariance_ref baselines read
    # what estimate_covariance actually writes.
    if refs.get("returns_panel"):
        out = step(
            "covariance",
            lambda: _call(
                "portfolio",
                "estimate_covariance",
                dict(returns_ref=refs["returns_panel"], run_id="fx"),
            ),
        )
        if out:
            refs["covariance"] = out["covariance_ref"]
        # One-step-ahead VaR forecasts of the first column, published by the
        # tool that makes them, for the tester's forecasts_ref.
        out = step(
            "var_forecasts",
            lambda: _call(
                "research",
                "forecast_var_series",
                dict(
                    returns_ref=refs["returns_panel"],
                    column=SYMBOLS[0],
                    run_id="fx",
                    name="var_forecasts",
                ),
            ),
        )
        if out:
            refs["var_forecasts"] = out["forecasts_ref"]

    dates = pd.bdate_range(start, end)
    scores = pd.DataFrame(
        rng.normal(size=(len(dates), len(SYMBOLS))), index=dates, columns=SYMBOLS
    )

    def panel(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        return {
            t: {d.strftime("%Y-%m-%d"): float(frame.loc[d, t]) for d in dates}
            for t in SYMBOLS
        }

    refs["score_panel"] = step(
        "score_panel",
        lambda: handoff.publish(
            panel(scores), kind="score_panel", run_id="fx", name="scores"
        ),
    )
    refs["signal_panel"] = step(
        "signal_panel",
        lambda: handoff.publish(
            panel(np.sign(scores).astype(float)),
            kind="signal_panel",
            run_id="fx",
            name="signals",
        ),
    )
    out = step(
        "weight_panel",
        lambda: _call(
            "portfolio",
            "construct_weights_from_scores",
            dict(scores_ref=refs.get("score_panel"), run_id="fx", name="weights"),
        ),
    )
    if out:
        refs["weight_panel"] = out["ref"]
    equity = pd.Series(
        100_000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates)))),
        index=dates,
        name="equity",
    )
    refs["equity_curve"] = step(
        "equity_curve",
        lambda: handoff.publish(
            equity, kind="equity_curve", run_id="fx", name="equity"
        ),
    )
    trades = pd.DataFrame(
        {
            "entry_date": dates[:40:2],
            "exit_date": dates[1:41:2],
            "entry_price": 100.0 + np.arange(20),
            "exit_price": 101.0 + np.arange(20) * 1.01,
            "pnl": rng.normal(10, 50, 20),
            "return_pct": rng.normal(0.5, 2, 20),
        }
    )
    refs["trade_log"] = step(
        "trade_log",
        lambda: handoff.publish(trades, kind="trade_log", run_id="fx", name="trades"),
    )

    # One session of trades and quotes.
    stamps = pd.date_range("2020-07-30 14:30", periods=2000, freq="3s")
    mid = 100 + np.cumsum(rng.normal(0, 0.01, len(stamps)))
    quotes = pd.DataFrame(
        {
            "bid_price": mid - 0.01,
            "ask_price": mid + 0.01,
            "bid_size": rng.integers(1, 20, len(stamps)) * 100.0,
            "ask_size": rng.integers(1, 20, len(stamps)) * 100.0,
        },
        index=stamps,
    )
    quotes.index.name = "timestamp"
    side = rng.choice([-1, 1], len(stamps))
    tape = pd.DataFrame(
        {"price": mid + side * 0.01, "size": rng.integers(1, 10, len(stamps)) * 100.0},
        index=stamps + pd.Timedelta(milliseconds=500),
    )
    tape.index.name = "timestamp"
    refs["tick_tape"] = step(
        "tick_tape",
        lambda: handoff.publish(tape, kind="tick_tape", run_id="fx", name="tape"),
    )
    refs["quote_panel"] = step(
        "quote_panel",
        lambda: handoff.publish(quotes, kind="quote_panel", run_id="fx", name="quotes"),
    )

    # External kinds: files on disk, registered where they lie.
    os.makedirs(directory, exist_ok=True)
    book: Dict[str, Any] = {"timestamp": stamps}
    for level in range(3):
        book[f"bid_price_{level}"] = mid - 0.01 - 0.01 * level
        book[f"ask_price_{level}"] = mid + 0.01 + 0.01 * level
        book[f"bid_size_{level}"] = rng.integers(1, 20, len(stamps)) * 100.0
        book[f"ask_size_{level}"] = rng.integers(1, 20, len(stamps)) * 100.0
    paths = fx["paths"]
    paths["book"] = os.path.join(directory, "book.parquet")
    pd.DataFrame(book).to_parquet(paths["book"])
    n_events = 3000
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-07-30 14:30", periods=n_events, freq="1s"),
            "order_id": np.arange(n_events) // 3,
            "action": np.tile(["A", "M", "C"], n_events // 3),
            "side": np.where(np.arange(n_events) % 2, "B", "A"),
            "price": 100 + rng.normal(0, 0.05, n_events).round(2),
            "size": rng.integers(1, 10, n_events) * 100.0,
        }
    )
    paths["events"] = os.path.join(directory, "events.parquet")
    events.to_parquet(paths["events"])
    paths["tape"] = os.path.join(directory, "tape.parquet")
    tape.reset_index().to_parquet(paths["tape"])
    paths["tape_csv"] = os.path.join(directory, "tape.csv")
    tape.reset_index().to_csv(paths["tape_csv"], index=False)
    for kind, path, name in (
        ("order_book_panel", paths["book"], "book"),
        ("order_event_panel", paths["events"], "events"),
    ):
        out = step(
            kind,
            lambda kind=kind, path=path, name=name: _call(
                "data",
                "register_external_dataset",
                dict(path=path, kind=kind, run_id="fx", name=name),
            ),
        )
        if out:
            refs[kind] = out["ref"]

    if refs.get("price_panel"):
        out = step(
            "indicator_panel",
            lambda: _call(
                "research",
                "compute_indicator_panel",
                dict(
                    tickers=SYMBOLS,
                    start_date=start,
                    end_date=end,
                    indicators=["rsi"],
                    run_id="fx",
                    name="ind",
                    price_panel_ref=refs["price_panel"],
                ),
            ),
        )
        if out:
            refs["indicator_panel"] = (out.get("refs") or {}).get("rsi")
        out = step(
            "data_bundle",
            lambda: _call(
                "data",
                "build_data_bundle",
                dict(
                    frames=[
                        {
                            "frame_kind": "bars",
                            "ref": refs["price_panel"],
                            "source": "yfinance",
                        }
                    ],
                    run_id="fx",
                    name="bundle",
                ),
            ),
        )
        if out:
            refs["data_bundle"] = out.get("ref")

    # A backtest's own persisted artifacts, for the tools that take a URI.
    from standard_quant_tools.agent.runtimes import resolve

    compact_model = resolve("backtest").dispatch_table["run_backtest_compact"][1]
    compact_args, _reason = synth.synthesize(compact_model)
    out = step(
        "run_backtest_compact",
        lambda: _call("backtest", "run_backtest_compact", compact_args),
    )
    if out:
        fx["uris"] = {
            k: v
            for k, v in out.items()
            if isinstance(v, str) and ("uri" in k or v.startswith("sqt://"))
        }

    # Two decision records, for the tools that explain, replay or compare one.
    step("request_ids", lambda: _call("meta", "list_strategies", {}))
    fx["request_id_1"] = last_request_id()
    step("request_ids_2", lambda: _call("meta", "list_stress_scenarios", {}))
    fx["request_id_2"] = last_request_id()

    # A dataset and two models: the route every modeling tool needs. One
    # column is aliased, as a real multi-horizon spec's are, so every tool
    # that takes a column name is probed with a name the catalog does not
    # know -- the shape `check_leakage` once reported as unsafe.
    fx["dataset_spec"] = {
        "universe": SYMBOLS[:4],
        "start": "2019-01-02",
        "end": "2021-06-30",
        "features": [
            {"id": "technical.rsi"},
            {"id": "risk.rolling_beta"},
            {
                "id": "risk.realized_volatility",
                "params": {"period": 20},
                "alias": "rvol_20",
            },
        ],
        "target": {"horizon": 5},
        "benchmark": "SPY",
    }
    dataset = step(
        "dataset",
        lambda: _call("modeling", "build_model_dataset", {"spec": fx["dataset_spec"]}),
    )
    if dataset:
        fx["dataset_id"] = dataset["dataset_id"]
        fx["feature_ids"] = dataset.get("feature_ids")
        spec = {
            "task": "regression",
            "estimator": {"type": "ridge", "params": {"alpha": 1.0}},
            "validation": {"train_window": 150, "test_window": 30, "embargo": 5},
            "random_seed": 11,
        }
        fx["model_spec"] = spec
        first = step(
            "model_1",
            lambda: _call(
                "modeling",
                "run_model_experiment",
                {"dataset_id": dataset["dataset_id"], "spec": spec},
            ),
        )
        second_spec = dict(
            spec, random_seed=12, estimator={"type": "ridge", "params": {"alpha": 5.0}}
        )
        second = step(
            "model_2",
            lambda: _call(
                "modeling",
                "run_model_experiment",
                {"dataset_id": dataset["dataset_id"], "spec": second_spec},
            ),
        )
        if first:
            fx["model_id"] = first["model_id"]
            fx["model_1"] = {
                k: v for k, v in first.items() if isinstance(v, (str, int))
            }
            for value in first.values():
                if isinstance(value, str) and value.startswith("sqt://predictions/"):
                    refs["predictions"] = value
        if second:
            fx["model_id_2"] = second["model_id"]
    for key, model_key, name in (
        ("predictions_with_outcomes", "model_id", "outcomes"),
        ("predictions_with_outcomes_2", "model_id_2", "outcomes2"),
    ):
        if fx.get(model_key):
            out = step(
                key,
                lambda model_key=model_key, name=name: _call(
                    "modeling",
                    "attach_model_outcomes",
                    {"model_id": fx[model_key], "run_id": "fx", "name": name},
                ),
            )
            for value in (out or {}).values():
                if isinstance(value, str) and value.startswith("sqt://"):
                    refs.setdefault(key, value)

    panel_dates = pd.bdate_range("2019-01-02", periods=300)
    external_panel = pd.DataFrame(
        [
            {
                "date": d,
                "entity": s,
                "target": float(rng.normal(0, 0.01)),
                "f1": float(rng.normal()),
                "f2": float(rng.normal()),
            }
            for d in panel_dates
            for s in SYMBOLS[:4]
        ]
    )
    paths["panel"] = os.path.join(directory, "panel.parquet")
    external_panel.to_parquet(paths["panel"])
    return fx


# ── baselines built on the fixtures ─────────────────────────────────────

_DELETE = object()

#: Counts of resampled draws, paths and permutations, by name, and what the
#: baseline lowers them to. A draw BUDGET (max_draws) is left alone: it is the
#: ceiling the count is checked against, and lowering it refuses the call.
_RESAMPLING = re.compile(
    r"^(n_bootstrap|n_bootstrap_iterations|n_simulations|n_paths|n_permutations)$"
)
_CHEAP_COUNT = 50

#: Reference fields, by name, and the fixture each takes.
_REF_BY_FIELD = {
    "scores_ref": "score_panel",
    "returns_ref": "returns_panel",
    "signal_panel_ref": "signal_panel",
    "price_panel_ref": "price_panel",
    "target_weights_ref": "weight_panel",
    "tick_tape_ref": "tick_tape",
    "trades_ref": "tick_tape",
    "quote_panel_ref": "quote_panel",
    "quotes_ref": "quote_panel",
    "predictions_ref": "predictions",
    "predictions_ref_a": "predictions",
    "predictions_ref_b": "predictions",
    "equity_curve_ref": "equity_curve",
    "covariance_ref": "covariance",
    "weights_ref": "weight_panel",
    "forecasts_ref": "var_forecasts",
}

#: Tools whose one `ref` field wants a particular kind.
_REF_BY_TOOL = {
    "infer_temporal_contract": "price_panel",
    "describe_external_dataset": "order_book_panel",
    "validate_external_dataset": "order_book_panel",
    "describe_data_bundle": "data_bundle",
    "validate_data_bundle": "data_bundle",
    "describe_reference": "returns_panel",
    "read_reference": "returns_panel",
    "convert_reference": "predictions",
    "get_order_book_metrics": "order_book_panel",
    "get_order_event_metrics": "order_event_panel",
}


def _fetched_returns() -> Dict[str, str]:
    """A `DataSource` that fetches: a symbol needs its window, and the
    synthesizer, trying one optional field at a time, settles on a `ref`
    that names nothing."""
    return {"symbol": SYMBOLS[0], "start_date": "2019-01-02", "end_date": "2020-12-31"}


def _date_series(
    n: int = 120, start: float = 100.0, step: float = 0.3
) -> Dict[str, float]:
    days = pd.bdate_range("2020-01-02", periods=n)
    return {
        d.strftime("%Y-%m-%d"): round(start + step * i + (i % 7) * 0.1, 4)
        for i, d in enumerate(days)
    }


def _position(
    instrument: str, quantity: float, price: float, currency: str, **terms: Any
) -> Dict[str, Any]:
    """One line of holdings for a finance tool's baseline."""
    return {
        "instrument": instrument,
        "quantity": quantity,
        "price": price,
        "currency": currency,
        **terms,
    }


def _hand_written(
    tool: str, fx: Dict[str, Any], base: Dict[str, Any]
) -> Dict[str, Any]:
    """The inputs whose meaningful shape the schema cannot describe."""
    refs, paths = fx.get("refs", {}), fx.get("paths", {})
    outcomes = refs.get("predictions_with_outcomes")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if tool == "convert_reference":
        return {"to_kind": "score_panel"}
    if tool in ("describe_tool", "validate_tool_call"):
        return {"tool_name": "get_rolling_beta"}
    if tool == "describe_artifact":
        return {"uri": fx.get("uris", {}).get("equity_curve_uri")}
    if tool == "export_audit_bundle":
        return {"start_date": today, "end_date": today}
    if tool == "register_external_dataset":
        return {"path": paths.get("tape"), "kind": "tick_tape"}
    if tool == "prepare_vendor_extract":
        return {"path": paths.get("tape_csv"), "kind": "tick_tape"}
    if tool == "detect_liquidity_events":
        return {
            "channels": ["book_imbalance", "bid_depth"],
            "ref": refs.get("order_book_panel"),
        }
    if tool == "run_screener":
        return {"filters": {"pe_ratio_max": 40.0}}
    if tool == "calculate_series_metrics":
        return {
            "series": _fetched_returns(),
            "metrics": ["sharpe_ratio", "annualized_volatility"],
        }
    if tool == "run_terminal_monte_carlo":
        return {"returns": _fetched_returns()}
    if tool == "construct_weights_from_scores":
        return {"scores_ref": refs.get("score_panel")}
    if tool == "run_futures_backtest":
        prices = _date_series()
        return {"prices": prices, "target_contracts": {d: 1.0 for d in prices}}
    if tool == "run_futures_hedge_backtest":
        return {
            "portfolio_values": _date_series(start=1e6, step=500.0),
            "future_prices": _date_series(start=4000.0, step=2.0),
        }
    if tool in (
        "optimize_risk_parity",
        "optimize_max_diversification",
        "get_marginal_risk_contribution",
    ):
        # The published covariance stands in for the synthesized matrix, and
        # names its own assets; the synthesized input keeps the inline form.
        if refs.get("covariance"):
            return {"covariance": _DELETE, "assets": _DELETE}
        names = base.get("asset_names") or base.get("tickers") or SYMBOLS
        return {"covariance": synth._covariance(len(names))}
    if tool == "optimize_portfolio":
        # A utility with a forecast per asset, a cap and a group limit, so the
        # probes reach the rows and the bounds, not only the covariance. The
        # published covariance names the six symbols.
        limits = {
            "objective": "max_utility",
            "expected_returns": {s: 0.04 + 0.01 * i for i, s in enumerate(SYMBOLS)},
            "risk_aversion": 2.0,
            "max_weight": 0.5,
            "groups": {"first": SYMBOLS[:3]},
            "group_max": {"first": 0.6},
        }
        if refs.get("covariance"):
            return {"covariance": _DELETE, "assets": _DELETE, **limits}
        return {
            "assets": list(SYMBOLS),
            "covariance": synth._covariance(len(SYMBOLS)),
            **limits,
        }
    # The returns panel's columns are the six symbols. Each tool given it in
    # place of its inline returns reads them in its own roles: the assets,
    # the strategy, the configurations, the candidates beside a benchmark.
    if refs.get("returns_panel"):
        if tool == "estimate_covariance":
            return {"tickers": _DELETE, "start_date": _DELETE, "end_date": _DELETE}
        if tool == "optimize_hierarchical_risk_parity":
            return {"returns": _DELETE}
        if tool == "get_deflated_sharpe_ratio":
            return {"returns": _DELETE, "returns_column": SYMBOLS[0]}
        if tool == "estimate_backtest_overfitting":
            return {"trial_returns": _DELETE}
        if tool == "run_reality_check":
            return {
                "candidate_returns": _DELETE,
                "benchmark_returns": _DELETE,
                "benchmark_column": SYMBOLS[0],
            }
        if tool == "decompose_tail_risk":
            return {"scenarios": _DELETE}
        # The allocation tools read the panel's columns as strategy streams:
        # every column held together, or one as the stream overlaid or sized.
        if tool == "get_kelly_report":
            return {"returns": _DELETE}
        if tool == "apply_risk_overlay":
            return {"returns": _DELETE, "column": SYMBOLS[0]}
        if tool == "get_strategy_capacity_curve" and refs.get("weight_panel"):
            return {
                "returns": _DELETE,
                "column": SYMBOLS[0],
                "weights": _DELETE,
                "adv": {s: 5e7 * (i + 1) for i, s in enumerate(SYMBOLS)},
                "daily_volatility": {s: 0.02 for s in SYMBOLS},
                "spread_bps": {s: 2.0 + i for i, s in enumerate(SYMBOLS)},
            }
        if tool == "estimate_factor_model":
            # One panel holds both sides: two of its columns are the
            # factors and the other four the assets.
            return {
                "returns": _DELETE,
                "factors": _DELETE,
                "columns": list(SYMBOLS[2:]),
                "factors_ref": refs["returns_panel"],
                "factor_columns": list(SYMBOLS[:2]),
            }
        if tool == "forecast_var_series":
            return {"returns": _DELETE, "column": SYMBOLS[0]}
        # The sleeve tools: the panel's columns are the sleeves, in place of
        # the synthesized streams (whose references name nothing) or the
        # synthesized inline returns (keyed by tickers, not dates).
        if tool == "combine_return_streams":
            return {"streams": _DELETE}
        if tool in ("allocate_across_sleeves", "evaluate_allocation_rule"):
            return {"returns": _DELETE}
    if tool == "backtest_var_forecasts" and refs.get("var_forecasts"):
        # The published forecasts carry their own level.
        return {"forecasts": _DELETE, "confidence": _DELETE}
    if tool == "build_continuous_futures_series":
        near = pd.bdate_range("2020-01-02", "2020-03-20")
        far = pd.bdate_range("2020-01-02", "2020-06-19")

        def prices(days: Any, level: float) -> Dict[str, float]:
            return {
                x.strftime("%Y-%m-%d"): round(level + 0.1 * i, 3)
                for i, x in enumerate(days)
            }

        def volume(days: Any, rising: bool) -> Dict[str, float]:
            return {
                x.strftime("%Y-%m-%d"): float(
                    1000 + (i * 50 if rising else 4000 - i * 50)
                )
                for i, x in enumerate(days)
            }

        return {
            "contracts": [
                {
                    "symbol": "ESH0",
                    "expiry": "2020-03-20",
                    "prices": prices(near, 3000.0),
                    "volume": volume(near, False),
                },
                {
                    "symbol": "ESM0",
                    "expiry": "2020-06-19",
                    "prices": prices(far, 3005.0),
                    "volume": volume(far, True),
                },
            ]
        }
    if tool == "build_data_bundle":
        return {
            "frames": [
                {
                    "frame_kind": "bars",
                    "ref": refs.get("price_panel"),
                    "source": "yfinance",
                }
            ]
        }
    if tool == "analyze_dividend_points":
        return {
            "constituents": [
                {
                    "symbol": s,
                    "shares": 1000.0,
                    "dividend_per_share": 0.5,
                    "ex_date": "2020-03-02",
                }
                for s in SYMBOLS
            ],
            "divisor": 100.0,
            "as_of": "2020-01-02",
            "expiry": "2020-06-19",
            "spot": 3000.0,
            "future_price": 2995.0,
            "financing_rate": 0.02,
            "time_to_expiry": 0.46,
        }
    if tool == "analyze_futures_curve":
        return {
            "contracts": [
                {
                    "price": 100.5 + 0.5 * i,
                    "time_to_expiry": 0.25 * (i + 1),
                    "label": f"M{i + 1}",
                }
                for i in range(4)
            ],
            "spot": 100.0,
        }
    if tool == "analyze_index_basket":
        return {
            "constituents": [
                {
                    "symbol": s,
                    "price": 100.0 + i,
                    "weight": 1 / 6,
                    "reference_price": 99.0 + i,
                }
                for i, s in enumerate(SYMBOLS)
            ],
            "index_level": 1000.0,
        }
    if tool == "analyze_total_return_future":
        return {
            "quote": 50.0,
            "quote_convention": "spread_bps",
            "underlying_price": 4000.0,
            "time_to_expiry": 0.5,
            "reference_rate": 0.03,
        }
    if tool == "price_total_return_swap":
        return {
            "notional": 1e6,
            "initial_price": 100.0,
            "current_price": 105.0,
            "financing_rate": 0.03,
            "start_date": "2020-01-02",
            "valuation_date": "2020-07-01",
        }
    if tool == "solve_forward_carry":
        return {
            "spot": 100.0,
            "forward": 101.0,
            "time_to_expiry": 0.5,
            "solve_for": "financing_rate",
            "dividend_yield": 0.01,
            "borrow_rate": 0.0,
        }
    if tool == "analyze_option_strategy":
        return {
            "legs": [
                {
                    "option_type": "call",
                    "quantity": 1.0,
                    "strike": 100.0,
                    "volatility": 0.2,
                    "time_to_expiry": 0.5,
                },
                {
                    "option_type": "call",
                    "quantity": -1.0,
                    "strike": 110.0,
                    "volatility": 0.2,
                    "time_to_expiry": 0.5,
                },
            ],
            "spot": 100.0,
        }
    if tool == "get_implied_volatility":
        return {
            "option_price": 10.4506,
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "risk_free_rate": 0.05,
        }
    if tool == "compare_feature_sets":
        return {
            "left": ["technical.rsi"],
            "right": ["risk.rolling_beta", "rvol_20"],
        }
    if tool == "estimate_corwin_schultz_spread":
        px = [
            float(x)
            for x in 100
            * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.01, 300)))
        ]
        return {"high": [x * 1.01 for x in px], "low": [x * 0.99 for x in px]}
    if tool == "get_implementation_shortfall":
        return {
            "decision_price": 100.0,
            "arrival_price": 100.1,
            "fills": [
                {"quantity": 100.0, "price": 100.2},
                {"quantity": 100.0, "price": 100.3},
            ],
            "target_quantity": 300.0,
            "final_price": 100.5,
        }
    if tool == "get_intraday_volume_profile":
        stamps = [
            t
            for d in pd.bdate_range("2020-07-27", periods=5)
            for t in pd.date_range(
                d + pd.Timedelta(hours=9, minutes=30),
                d + pd.Timedelta(hours=15, minutes=55),
                freq="5min",
            )
        ]
        return {
            "timestamps": [t.strftime("%Y-%m-%d %H:%M:%S") for t in stamps],
            "volume": [float(1000 + (i % 78) * 10) for i in range(len(stamps))],
        }
    if tool == "get_order_event_metrics":
        return {"events": _DELETE, "ref": refs.get("order_event_panel")}
    if tool == "attach_model_outcomes":
        return {
            "model_id": fx.get("model_id"),
            "predictions_ref": _DELETE,
            "dataset_id": _DELETE,
        }
    if tool == "compare_signals" and outcomes:
        return {
            "mode": "paired",
            "alpha": _DELETE,
            "method": _DELETE,
            "p_values": _DELETE,
            "ic_a": _DELETE,
            "ic_b": _DELETE,
            "predictions_ref_a": outcomes,
            "predictions_ref_b": refs.get("predictions_with_outcomes_2") or outcomes,
            "task": "regression",
        }
    if tool == "estimate_feature_warmup":
        # Priced on the published dataset's interval; a model_id beside the
        # dataset_id is refused, so only one of the two is kept.
        return {
            "features": [{"id": "technical.rsi"}, {"id": "risk.rolling_beta"}],
            "model_id": _DELETE,
        }
    if tool == "monitor_model":
        return {
            "predictions_uri": (fx.get("model_1") or {}).get("oos_predictions_uri"),
            "outcomes_ref": outcomes or _DELETE,
            "features_uri": _DELETE,
        }
    if tool == "promote_model":
        return {"to_stage": "validated", "public_key_path": _DELETE}
    if tool == "register_external_panel" and paths.get("panel"):
        return {
            "path": paths["panel"],
            "horizon": 5,
            "targets": _DELETE,
            "label_end_column": _DELETE,
            "event_column": _DELETE,
            "feature_columns": _DELETE,
        }
    if tool == "get_factor_exposure_budget":
        return {
            "weights": {s: 1 / 6 for s in SYMBOLS},
            "factor_loadings": {
                s: {"mkt": 0.8 + 0.1 * i, "size": 0.1 * i - 0.2}
                for i, s in enumerate(SYMBOLS)
            },
            "factors": ["mkt", "size"],
            "factor_covariance": [[0.04, 0.005], [0.005, 0.02]],
        }
    if tool == "decompose_factor_risk":
        # A benchmark beside the synthesized model, so every probe reaches
        # the active decomposition and the tracking error too.
        return {"benchmark_weights": {s: (i + 1) / 21 for i, s in enumerate(SYMBOLS)}}
    if tool == "attribute_returns_brinson":
        # Three quarters of a three-segment book against its benchmark, so
        # every probe reaches the effects and the linking.
        quarters = ("2024-03-31", "2024-06-30", "2024-09-30")
        segments = ("tech", "energy", "cash")

        def by_quarter(values: Any) -> Dict[str, Dict[str, float]]:
            return {
                q: {seg: values[i][j] for j, seg in enumerate(segments)}
                for i, q in enumerate(quarters)
            }

        return {
            "portfolio_weights": by_quarter([[0.6, 0.3, 0.1]] * 3),
            "benchmark_weights": by_quarter([[0.5, 0.4, 0.1]] * 3),
            "portfolio_returns": by_quarter(
                [[0.05, 0.02, 0.01], [-0.02, 0.03, 0.01], [0.04, -0.01, 0.01]]
            ),
            "benchmark_returns": by_quarter(
                [[0.04, 0.02, 0.01], [-0.01, 0.02, 0.01], [0.03, 0.0, 0.01]]
            ),
        }
    if tool == "analyze_cash_flows":
        # A project: 100 out, then 12 a year and the 100 back. One IRR
        # (12%), a present value at 6% for the durations, and both MIRR
        # rates, so every probe reaches each number.
        return {
            "flows": [-100.0] + [12.0] * 9 + [112.0],
            "rate": 0.06,
            "finance_rate": 0.05,
            "reinvest_rate": 0.06,
        }
    if tool == "estimate_cost_of_capital":
        # With debt, a standard error and a target, so the relevered and
        # the sensitivity blocks are reached too.
        return {
            "risk_free_rate": 0.04,
            "beta": 1.2,
            "equity_risk_premium": 0.055,
            "equity_value": 800.0,
            "debt_value": 200.0,
            "cost_of_debt": 0.06,
            "tax_rate": 0.21,
            "beta_standard_error": 0.15,
            "target_debt_to_equity": 0.5,
        }
    if tool == "value_cash_flows":
        return {
            "cash_flows": [10.0, 11.0, 12.0, 12.5, 13.0],
            "discount_rate": 0.09,
            "terminal_growth": 0.025,
        }
    if tool == "measure_book_returns":
        # Quarterly values over two years, each flow on a valuation date, so
        # the true TWR, Modified Dietz and the MWR are all defined.
        return {
            "valuations": {
                "2020-01-02": 1000.0,
                "2020-04-01": 1040.0,
                "2020-07-01": 1100.0,
                "2020-10-01": 1080.0,
                "2021-01-04": 1150.0,
                "2021-04-01": 1210.0,
                "2021-07-01": 1190.0,
                "2021-10-01": 1260.0,
                "2022-01-03": 1300.0,
            },
            "flows": {"2020-04-01": 50.0, "2021-01-04": -80.0, "2021-07-01": 30.0},
        }
    if tool == "get_currency_exposure":
        # Four currencies, a partial forward hedge and a cross hedge, and a
        # return on every line, so a probe reaches the hedge ratios and the
        # return split rather than the exposure alone.
        return {
            "positions": [
                _position("SAP", 1000.0, 120.0, "EUR", local_return=0.03),
                _position("7203", -2000.0, 2900.0, "JPY", local_return=-0.02),
                _position("VOD", 5000.0, 0.72, "GBP", local_return=0.01),
                _position("AAPL", 300.0, 190.0, "USD", local_return=0.015),
            ],
            "base_currency": "USD",
            "fx_rates": {"EUR": 1.08, "JPY": 0.0067, "GBP": 1.27},
            "hedges": {"EURUSD": -60000.0, "EURGBP": -20000.0},
            "fx_returns": {"EUR": 0.01, "JPY": -0.015, "GBP": 0.004},
        }
    if tool == "analyze_book_carry":
        # A future, an FX forward, a funded long and a funded short, so a
        # probe reaches both kinds of line and a short's borrow.
        return {
            "positions": [
                _position(
                    "ESZ6",
                    2.0,
                    5050.0,
                    "USD",
                    multiplier=50.0,
                    spot=5000.0,
                    years_to_expiry=0.25,
                ),
                _position(
                    "EURUSD 3M",
                    -1e6,
                    1.1038,
                    "USD",
                    spot=1.1,
                    years_to_expiry=0.25,
                ),
                _position(
                    "SAP",
                    1000.0,
                    120.0,
                    "EUR",
                    income_rate=0.025,
                    financing_rate=0.039,
                ),
                _position(
                    "7203",
                    -2000.0,
                    2900.0,
                    "JPY",
                    borrow_fee_bps=40.0,
                    income_rate=0.028,
                    financing_rate=0.001,
                ),
            ],
            "base_currency": "USD",
            "fx_rates": {"EUR": 1.08, "JPY": 0.0067},
        }
    if tool == "price_fx_forward_hedge":
        # A basis and a period, so a probe reaches the basis leg and the
        # hedged and unhedged returns.
        return {
            "pair": "EURUSD",
            "spot": 1.1,
            "days": 90,
            "base_currency": "USD",
            "base_rate": 0.053,
            "foreign_rate": 0.039,
            "base_day_count": "ACT/360",
            "foreign_day_count": "ACT/360",
            "basis_bps": -15.0,
            "asset_return": 0.02,
            "spot_at_maturity": 1.08,
        }
    if tool == "aggregate_option_book":
        # Every line's underlying is a key the spots, rates and betas price,
        # with shares on one of them, so the probes reach the greeks, the
        # beta weighting and the revaluation grid.
        a, b = SYMBOLS[:2]
        line = {"option_type": "call", "time_to_expiry": 0.5, "multiplier": 100}
        return {
            "options": [
                {
                    **line,
                    "underlying": a,
                    "strike": 100.0,
                    "volatility": 0.25,
                    "quantity": 3,
                },
                {
                    **line,
                    "underlying": b,
                    "option_type": "put",
                    "strike": 90.0,
                    "volatility": 0.3,
                    "quantity": -2,
                },
            ],
            "spots": {a: 101.0, b: 95.0},
            "rates": {a: 0.04, b: 0.04},
            "dividend_yields": {a: 0.01},
            "underlying_units": {b: 200.0},
            "betas": {a: 1.1, b: 0.8},
            "valuation_date": _DELETE,
        }
    if tool == "get_sharpe_stability":
        return {
            "returns": [
                float(x) for x in np.random.default_rng(9).normal(0.0004, 0.01, 1000)
            ]
        }
    if tool == "run_cointegration_test":
        return {"symbol_b": "MSFT"}
    if tool in ("score_predictions", "score_prediction_intervals") and outcomes:
        return {"predictions_ref": outcomes}
    if tool == "score_model":
        return {"as_of": "2021-09-30", "universe": SYMBOLS[:4]}
    if tool == "estimate_financing_cost":
        # A horizon, and terms that give every synthesized short a borrow.
        return {
            "days": 30,
            "financing": {
                "benchmark_rate": 0.053,
                "day_count": "ACT/360",
                "debit_spread_bps": 50.0,
                "default_borrow_bps": 30.0,
            },
        }
    if tool == "compute_margin_requirement":
        # The synthesized house rates are keyed by symbols no line holds.
        return {"house_rates": _DELETE, "equity": 1.0e5}
    if tool == "analyze_margin_runway":
        # A path losing 100 a day from 9,000 against 10,000 / 8,000 calls
        # on day 11, so the probes reach the walk and the buffer.
        return {
            "equity": 9_000.0,
            "equity_changes": [-100.0] * 30,
            "initial_margin": 10_000.0,
            "maintenance_margin": 8_000.0,
            "cure_days": 5,
        }
    return {}


def synth_kind(annotation: Any) -> str:
    """'str', 'list[str]', ... for the fields the substitution keys on."""
    import typing

    ann = annotation
    while True:
        origin = typing.get_origin(ann)
        if origin is typing.Annotated:
            ann = typing.get_args(ann)[0]
            continue
        if origin is typing.Union:
            options = [a for a in typing.get_args(ann) if a is not type(None)]
            if len(options) == 1:
                ann = options[0]
                continue
        break
    if ann is str:
        return "str"
    if typing.get_origin(ann) in (list, typing.List):
        args = typing.get_args(ann)
        return "list[str]" if args and synth_kind(args[0]) == "str" else "list"
    return "other"


def published_baseline(
    tool: str, model: type, synthesized: Dict[str, Any], fx: Dict[str, Any]
) -> Dict[str, Any]:
    """The synthesized input with every fixture it can name substituted in."""
    refs = fx.get("refs", {})
    args = copy.deepcopy(synthesized)
    fields = model.model_fields
    for name, info in fields.items():
        if name in _REF_BY_FIELD and refs.get(_REF_BY_FIELD[name]):
            args[name] = refs[_REF_BY_FIELD[name]]
        elif name == "ref" and tool in _REF_BY_TOOL:
            args[name] = refs.get(_REF_BY_TOOL[tool])
        elif name == "dataset_id" and fx.get("dataset_id"):
            args[name] = fx["dataset_id"]
        elif name == "model_id" and fx.get("model_id"):
            args[name] = fx["model_id"]
        elif (
            name == "reference_model_id"
            and fx.get("model_id_2")
            and name in synthesized
        ):
            args[name] = fx["model_id_2"]
        elif name == "model_ids" and fx.get("model_id"):
            args[name] = [fx["model_id"], fx["model_id_2"]]
        elif name in ("request_id", "request_id_a") and fx.get("request_id_1"):
            args[name] = fx["request_id_1"]
        elif name == "request_id_b" and fx.get("request_id_2"):
            args[name] = fx["request_id_2"]
        elif (
            name in ("feature_ids", "features")
            and fx.get("feature_ids")
            and synth_kind(info.annotation) == "list[str]"
        ):
            args[name] = list(fx["feature_ids"])
        elif (
            name in ("feature_id", "feature")
            and fx.get("feature_ids")
            and synth_kind(info.annotation) == "str"
        ):
            args[name] = fx["feature_ids"][0]
        elif name == "equity_curve_uri" and fx.get("uris", {}).get("equity_curve_uri"):
            args[name] = fx["uris"]["equity_curve_uri"]
        annotation_name = getattr(_inner_model(info.annotation), "__name__", "")
        if annotation_name == "ModelSpec" and fx.get("model_spec"):
            args[name] = copy.deepcopy(fx["model_spec"])
        elif annotation_name == "DatasetSpec" and fx.get("dataset_spec"):
            args[name] = copy.deepcopy(fx["dataset_spec"])
    for name, value in _hand_written(tool, fx, args).items():
        if name in fields and value is not None:
            args[name] = value
    args = {k: v for k, v in args.items() if v is not None and v is not _DELETE}
    # A resampling count at its default (2,000 bootstrap draws, 10,000
    # simulated paths) makes every probe of that tool cost seconds, and what
    # the probes test does not depend on it. The field's own lower bound is
    # respected, so the baseline stays one its model accepts.
    for name, info in fields.items():
        value = args.get(name)
        if (
            _RESAMPLING.search(name)
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            low, _high = synth._numeric_bounds(info)
            args[name] = min(value, max(int(low or 1), _CHEAP_COUNT))
    # A reference beside the inline value it replaces is refused by every
    # "exactly one of" rule (`signal_panel` / `signal_panel_ref`, `snapshots` /
    # `ref`); the synthesized inline value is then the baseline instead.
    try:
        model(**copy.deepcopy(args))
    except pydantic.ValidationError:
        return copy.deepcopy(synthesized)
    return args


def _inner_model(annotation: Any) -> Optional[type]:
    import typing

    ann = annotation
    while True:
        origin = typing.get_origin(ann)
        if origin is typing.Annotated:
            ann = typing.get_args(ann)[0]
            continue
        if origin is typing.Union:
            options = [a for a in typing.get_args(ann) if a is not type(None)]
            if len(options) == 1:
                ann = options[0]
                continue
        return ann if isinstance(ann, type) else None
