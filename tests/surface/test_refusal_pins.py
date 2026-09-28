"""
The inputs that once crashed a tool, each pinned to a refusal.

`test_adversarial_inputs.py` generates its probes from the schemas and is
marked `slow`, so `-m "not slow"` skips it. These are the cases those
probes found, written out by hand so they stay pinned in the fast run. Each
used to escape as something other than a refusal -- an `IndexError` from a
pandas indexer, numpy's bare "expected non-negative integer", pandas'
`DateParseError`, sklearn's "Input contains NaN", a `ZeroDivisionError`, a
pybind11 cast error -- and each must now be refused by name: the library's
`ValidationError` (or another `QuantError`), or the input model's own
refusal.

Four more pin a wrong ANSWER rather than a crash: an empty leakage check
reported safe, an inverted Hurst window reported 0.0, factor names that
did not match their tickers were zipped short, and a weight panel was
accepted as scores and transformed twice.

`BAD_VALUES` pins inputs that can only be a mistake -- a rate of 1e308, a
price past anything a market quotes, a NaN holding. Each was answered with
null numbers and a reason, which is the right answer to a legal input with
no defined result and the wrong one to an input that should never have run.

Every call runs against the hermetic market and the published fixtures in
conftest, so a pin reaches the computation that failed.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import pydantic
import pytest

from standard_quant_tools.error import QuantError

from . import hermetic, synth

Modifier = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def _set(**values: Any) -> Modifier:
    return lambda args, _fx: {**args, **values}


def _duplicate(field: str) -> Modifier:
    return lambda args, _fx: {
        **args,
        field: [args[field][0]] * max(2, len(args[field])),
    }


def _first_value(field: str, value: Any) -> Modifier:
    def modify(args: Dict[str, Any], _fx: Dict[str, Any]) -> Dict[str, Any]:
        mapping = dict(args[field])
        mapping[next(iter(mapping))] = value
        return {**args, field: mapping}

    return modify


def _first_key(field: str, key: str) -> Modifier:
    def modify(args: Dict[str, Any], _fx: Dict[str, Any]) -> Dict[str, Any]:
        mapping = dict(args[field])
        mapping[key] = mapping.pop(next(iter(mapping)))
        return {**args, field: mapping}

    return modify


def _keyed_by_tickers(field: str) -> Modifier:
    return lambda args, _fx: {
        **args,
        field: dict(zip(hermetic.SYMBOLS, args[field].values())),
    }


def _scaled(field: str, factor: float) -> Modifier:
    return lambda args, _fx: {**args, field: [v * factor for v in args[field]]}


def _nested_first(field: str, name: str, value: Any) -> Modifier:
    def modify(args: Dict[str, Any], _fx: Dict[str, Any]) -> Dict[str, Any]:
        items = copy.deepcopy(args[field])
        items[0][name] = value
        return {**args, field: items}

    return modify


def _ref(field: str, kind: str) -> Modifier:
    return lambda args, fx: {**args, field: fx["refs"][kind]}


def _swap(low: str, high: str) -> Modifier:
    return lambda args, _fx: {**args, low: args[high], high: args[low]}


#: Values past any plausible magnitude, or a non-finite holding. The bounds
#: are the library's own: rates are decimals within +/-10, prices within
#: the 1e12 the option pricers use, a spread at most the whole notional.
BAD_VALUES: List[Tuple[str, str, str, Modifier]] = [
    (
        "backtest",
        "run_portfolio_simulation",
        "margin_interest_rate=1e308",
        _set(margin_interest_rate=1e308),
    ),
    (
        "delta_one",
        "analyze_cash_futures_basis",
        "a future quoted at 1e308",
        _set(future_price=1e308),
    ),
    (
        "delta_one",
        "analyze_etf_fair_value",
        "a fund priced at 1e308",
        _set(etf_price=1e308),
    ),
    (
        "delta_one",
        "monitor_spread_stream",
        "primary prices scaled by 1e300",
        _scaled("primary_prices", 1e300),
    ),
    (
        "portfolio",
        "plan_rebalance",
        "a NaN target weight",
        _first_value("target_weights", float("nan")),
    ),
    ("portfolio", "estimate_trade_cost", "spread_bps=1e308", _set(spread_bps=1e308)),
    (
        "portfolio",
        "get_efficient_frontier",
        "risk_free_rate=1e308",
        _set(risk_free_rate=1e308),
    ),
]

#: (runtime, tool, what is wrong, how the baseline is changed). Each must be
#: REFUSED: every one of these inputs names something that cannot be done.
REFUSED: List[Tuple[str, str, str, Modifier]] = BAD_VALUES + [
    ("backtest", "compare_against_random", "seed=-1", _set(seed=-1)),
    ("backtest", "get_robustness_diagnostics", "random_seed=-1", _set(random_seed=-1)),
    ("backtest", "run_backtest_optimization", "top_n=0", _set(top_n=0)),
    (
        "backtest",
        "run_custom_signal_backtest",
        "a signal key that is not a date",
        _first_key("signals", "not-a-date"),
    ),
    (
        "backtest",
        "run_custom_signal_backtest",
        "signals keyed by tickers",
        _keyed_by_tickers("signals"),
    ),
    (
        "backtest",
        "run_futures_backtest",
        "a price key that is not a date",
        _first_key("prices", "not-a-date"),
    ),
    (
        "backtest",
        "run_futures_backtest",
        "a target of +inf contracts",
        _first_value("target_contracts", float("inf")),
    ),
    ("backtest", "run_monte_carlo_simulation", "tickers=[]", _set(tickers=[])),
    ("backtest", "run_monte_carlo_simulation", "random_seed=-1", _set(random_seed=-1)),
    ("backtest", "run_monte_carlo_trade_paths", "seed=-1", _set(seed=-1)),
    ("backtest", "run_pair_trade_backtest", "zscore_window=-1", _set(zscore_window=-1)),
    ("backtest", "run_reality_check", "seed=-1", _set(seed=-1)),
    (
        "backtest",
        "run_terminal_monte_carlo",
        "initial_capital=+inf",
        _set(initial_capital=float("inf")),
    ),
    ("backtest", "run_terminal_monte_carlo", "seed=-1", _set(seed=-1)),
    (
        "backtest",
        "run_walk_forward_backtest",
        "train_bars longer than the data",
        _set(train_bars=5000),
    ),
    (
        "backtest",
        "run_regime_adaptive_walkforward_backtest",
        "train_bars longer than the data",
        _set(train_bars=5000),
    ),
    (
        "data",
        "build_continuous_futures_series",
        "an expiry that is not a date",
        _nested_first("contracts", "expiry", "not-a-date"),
    ),
    ("data", "fetch_ohlcv", "an unknown source", _set(source="zz_not_valid")),
    (
        "delta_one",
        "optimize_replication_basket",
        "benchmark returns scaled by 1e300",
        _scaled("benchmark_returns", 1e300),
    ),
    ("derivatives", "simulate_delta_hedge", "seed=-1", _set(seed=-1)),
    (
        "feature_lab",
        "compare_feature_sets",
        "a duplicated left set",
        _duplicate("left"),
    ),
    ("feature_lab", "compare_feature_sets", "selection_end=''", _set(selection_end="")),
    (
        "feature_lab",
        "get_feature_drift",
        "split_date='not-a-date'",
        _set(split_date="not-a-date"),
    ),
    (
        "feature_lab",
        "get_feature_drift",
        "split_date='2019-13-45'",
        _set(split_date="2019-13-45"),
    ),
    (
        "feature_lab",
        "get_feature_redundancy",
        "duplicated features",
        _duplicate("features"),
    ),
    (
        "feature_lab",
        "run_feature_ablation",
        "duplicated features",
        _duplicate("features"),
    ),
    (
        "feature_lab",
        "screen_feature_stability",
        "split_date='not-a-date'",
        _set(split_date="not-a-date"),
    ),
    ("feature_lab", "select_features", "duplicated features", _duplicate("features")),
    ("feature_lab", "select_features", "selection_end=''", _set(selection_end="")),
    ("modeling", "analyze_features", "duplicated features", _duplicate("features")),
    ("modeling", "compare_signals", "seed=-1", _set(seed=-1)),
    ("modeling", "score_model", "as_of=''", _set(as_of="")),
    ("modeling", "score_predictions", "train_mean=NaN", _set(train_mean=float("nan"))),
    ("modeling", "score_predictions", "train_mean=+inf", _set(train_mean=float("inf"))),
    ("modeling", "check_leakage", "feature_ids=[]", _set(feature_ids=[])),
    (
        "portfolio",
        "analyze_concentration",
        "a weight of +inf",
        _first_value("weights", float("inf")),
    ),
    ("portfolio", "get_capacity_report", "adv_lookback=0", _set(adv_lookback=0)),
    ("portfolio", "get_liquidity_metrics", "tickers=[]", _set(tickers=[])),
    (
        "portfolio",
        "get_portfolio_risk_attribution",
        "duplicated tickers",
        _duplicate("tickers"),
    ),
    (
        "portfolio",
        "get_position_size",
        "account_equity=NaN",
        _set(account_equity=float("nan")),
    ),
    (
        "portfolio",
        "get_position_size",
        "account_equity=+inf",
        _set(account_equity=float("inf")),
    ),
    (
        "portfolio",
        "construct_weights_from_scores",
        "a weight panel as scores",
        _ref("scores_ref", "weight_panel"),
    ),
    ("research", "get_bootstrap_interval", "seed=-1", _set(seed=-1)),
    (
        "research",
        "get_rolling_beta",
        "a window longer than the data",
        _set(window=5000),
    ),
    (
        "research",
        "run_pca_analysis",
        "a single ticker",
        lambda a, _fx: {**a, "tickers": a["tickers"][:1]},
    ),
    ("research", "run_pca_analysis", "n_components=5000", _set(n_components=5000)),
    (
        "research",
        "run_hurst_analysis",
        "min_window above max_window",
        _set(min_window=200, max_window=50),
    ),
    (
        "research",
        "run_factor_regression",
        "fewer factor names than factor tickers",
        _set(factor_tickers=["SPY", "IWM", "IWD"], factor_names=["mkt"]),
    ),
]

#: Inputs that must be HANDLED -- refused by name or answered -- where the
#: honest outcome depends on the fix: a predictions reference can be pivoted
#: into scores or refused with a pointer to convert_reference; a futures
#: hedge given ticker-keyed maps is refused. Crashing is the only wrong one.
HANDLED: List[Tuple[str, str, str, Modifier]] = [
    (
        "portfolio",
        "construct_weights_from_scores",
        "a predictions reference as scores",
        _ref("scores_ref", "predictions"),
    ),
]


def _baseline(
    runtime: str, tool: str, published: Dict[str, Any]
) -> Tuple[type, Dict[str, Any]]:
    from standard_quant_tools.agent.runtimes import resolve

    _fn, model = resolve(runtime).dispatch_table[tool]
    arguments, reason = synth.synthesize(model)
    assert arguments is not None, f"{tool} cannot be synthesized: {reason}"
    return model, hermetic.published_baseline(tool, model, arguments, published)


def _outcome(
    runtime: str, tool: str, arguments: Dict[str, Any]
) -> Optional[BaseException]:
    """The refusal raised, or None when the tool returned; anything that is
    not a refusal fails here, with its type."""
    from standard_quant_tools.agent.runtimes import resolve

    try:
        resolve(runtime).dispatch(tool, arguments)
    except (QuantError, pydantic.ValidationError) as refusal:
        assert str(refusal).strip(), f"{tool} refused with an empty message"
        return refusal
    except Exception as exc:  # noqa: BLE001 - the defect this file pins
        pytest.fail(
            f"{tool} raised {type(exc).__module__}.{type(exc).__name__}: "
            f"{str(exc)[:300]} -- not a refusal. It names no argument and "
            "`except QuantError` misses it."
        )
    return None


@pytest.mark.parametrize(
    "runtime,tool,what,modify",
    REFUSED,
    ids=[f"{tool}: {what}" for _r, tool, what, _m in REFUSED],
)
def test_the_input_is_refused_by_name(runtime, tool, what, modify, published):
    _model, base = _baseline(runtime, tool, published)
    refusal = _outcome(runtime, tool, modify(base, published))
    assert refusal is not None, f"{tool} answered {what} instead of refusing it"


@pytest.mark.parametrize(
    "runtime,tool,what,modify",
    HANDLED,
    ids=[f"{tool}: {what}" for _r, tool, what, _m in HANDLED],
)
def test_the_input_is_handled(runtime, tool, what, modify, published):
    _model, base = _baseline(runtime, tool, published)
    _outcome(runtime, tool, modify(base, published))


def test_a_futures_hedge_on_its_synthesized_input_is_handled():
    """The synthesized input keys both maps by ticker, not by date; it used
    to raise pandas' DateParseError from the rehedge schedule."""
    from standard_quant_tools.agent.runtimes import resolve

    _fn, model = resolve("backtest").dispatch_table["run_futures_hedge_backtest"]
    arguments, _reason = synth.synthesize(model)
    _outcome("backtest", "run_futures_hedge_backtest", arguments)


@pytest.mark.parametrize(
    "runtime,tool",
    [(r, t) for r, t, _w, _m in BAD_VALUES]
    + [(r, t) for r, t, _w, _m in REFUSED[len(BAD_VALUES) : len(BAD_VALUES) + 3]],
)
def test_the_unmodified_baseline_still_returns(runtime, tool, published):
    """The null case: the same baseline, unchanged, is answered -- so each
    refusal above is about the one thing the pin changed."""
    _model, base = _baseline(runtime, tool, published)
    assert _outcome(runtime, tool, base) is None
