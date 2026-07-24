"""
Two-leg pair-trade backtest: reuses run_portfolio_simulation
(backtest/portfolio_engine.py) so both legs enter and exit on the same
rebalance event, share one cash account, and get synchronized costs — a
pair trade is just a 2-asset portfolio with a dollar-neutral weight vector
at each transition, not a new state machine or execution engine.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from standard_quant_tools.analysis.cointegration import compute_spread, spread_zscore
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)


def _execution_prices_for_weights(
    d: Any,
    common_idx: pd.Index,
    close_a: pd.Series,
    close_b: pd.Series,
    price_data: Dict[str, pd.DataFrame],
    symbol_a: str,
    symbol_b: str,
    fill_price: str,
) -> tuple:
    """
    hedge_ratio -> dollar-weight conversion must use whatever price
    run_portfolio_simulation will ACTUALLY execute this transition at, not
    always that date's own Close: with fill_price="next_open" (the
    default), the trade fills at the FOLLOWING bar's Open, so sizing off
    today's Close and then executing at tomorrow's Open silently breaks the
    hedge share ratio (shares_b/shares_a stops equaling hedge_ratio) unless
    A and B happen to have identical overnight gaps -- exactly the case
    the pre-existing hedge_ratio=1.0/near-equal-price test fixture could
    never expose.
    """
    if fill_price == "next_open":
        pos = int(common_idx.get_loc(d))
        if pos + 1 < len(common_idx):
            exec_date = common_idx[pos + 1]
            return (
                float(price_data[symbol_a]["Open"].loc[exec_date]),
                float(price_data[symbol_b]["Open"].loc[exec_date]),
            )
        # Last bar with no following bar to execute at -- run_portfolio_simulation
        # raises a clear ValidationError for this case; fall back to Close here
        # so weight construction doesn't crash before that error surfaces.
        return float(close_a.loc[d]), float(close_b.loc[d])
    if fill_price == "hl2_exploratory":
        return (
            float(
                (price_data[symbol_a]["High"].loc[d] + price_data[symbol_a]["Low"].loc[d])
                / 2.0
            ),
            float(
                (price_data[symbol_b]["High"].loc[d] + price_data[symbol_b]["Low"].loc[d])
                / 2.0
            ),
        )
    return float(close_a.loc[d]), float(close_b.loc[d])


def _spread_state(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """
    Stateful entry/exit machine on the z-scored spread: +1 = long the
    spread (long symbol_a, short symbol_b), entered when z crosses at or
    below -entry_z, held until z crosses back at or above -exit_z; -1 =
    short the spread (mirror condition); 0 = flat. NaN z-scores (e.g. before
    a rolling window fills) hold the previous state rather than forcing
    flat — the same "hold until exit condition" pattern the mean-reversion
    signal generators in strategies.py use.
    """
    values = []
    state = 0.0
    for zi in z.to_numpy(dtype=float):
        if pd.isna(zi):
            values.append(state)
            continue
        if state == 0.0:
            if zi <= -entry_z:
                state = 1.0
            elif zi >= entry_z:
                state = -1.0
        elif state == 1.0 and zi >= -exit_z:
            state = 0.0
        elif state == -1.0 and zi <= exit_z:
            state = 0.0
        values.append(state)
    return pd.Series(values, index=z.index, dtype=float)


def run_pair_backtest(
    price_data: Dict[str, pd.DataFrame],
    symbol_a: str,
    symbol_b: str,
    hedge_ratio: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    zscore_window: Optional[int] = 30,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    gross_leverage: float = 1.0,
    fill_price: str = "next_open",
) -> Dict[str, Any]:
    """
    Backtest a cointegrated pair as one synchronized two-leg trade: long the
    spread (long symbol_a, short symbol_b) when the z-scored spread falls to
    or below -entry_z, short the spread on the mirror condition, exit to
    flat once the z-score reverts inside exit_z. Both legs enter/exit on the
    same rebalance event by construction — they're two columns of one
    target_weights row passed to run_portfolio_simulation, so there is no
    separate per-leg state machine that could fall out of sync.

    Args:
        price_data: {symbol_a: OHLCV df, symbol_b: OHLCV df}, each with a
            'Close' column. Aligned to the intersection of both indices
            before computing the spread, so every date used for a rebalance
            is guaranteed valid for run_portfolio_simulation.
        hedge_ratio: spread = Close_a - hedge_ratio * Close_b (same
            convention as analysis/cointegration.py's compute_spread — the
            hedge_ratio from cointegration_test's Engle-Granger regression
            is the typical source).
        entry_z, exit_z: z-score thresholds — see _spread_state.
        zscore_window: rolling window for spread_zscore; defaults to 30 so
            every signal only uses spread history available up to that bar.
            Passing None reverts to spread_zscore's full-sample static
            z-score, which computes mean/std over the ENTIRE series
            (including bars after the signal date) — this leaks future
            spread statistics into historical signals and will produce an
            optimistically-biased backtest. Only pass None for exploratory
            analysis, never to evaluate real strategy performance.
        gross_leverage: sum(|weight|) while in a position, split between the
            two legs so the *share* ratio matches hedge_ratio (1 share of
            symbol_a per hedge_ratio shares of symbol_b — the same
            convention compute_spread uses), not a dollar ratio. Converting
            a share ratio to dollar weights requires the price the trade
            will actually EXECUTE at (Close on the trigger date itself for
            fill_price="close"; the FOLLOWING bar's Open for the default
            "next_open" — sizing off the trigger date's Close and then
            executing a bar later would silently break the share ratio
            unless both legs happened to gap overnight by the same
            percentage): weight_a = gross_leverage * exec_price_a /
            (exec_price_a + |hedge_ratio| * exec_price_b), weight_b =
            sign(hedge_ratio) * gross_leverage * |hedge_ratio| *
            exec_price_b / (exec_price_a + |hedge_ratio| * exec_price_b)
            (sign also flips with the position direction). This is only
            dollar-neutral when |hedge_ratio| * exec_price_b ~= exec_price_a;
            recomputed at every transition since the two legs' prices drift
            apart over time.
        initial_capital, commission_pct, slippage_pct: passed through to
            run_portfolio_simulation.
        fill_price: passed through to run_portfolio_simulation; defaults to
            "next_open" (not "close") because the z-score signal used to
            decide a transition is itself computed from that same bar's
            Close — executing at that bar's own Close would be look-ahead
            (trading at the exact price the signal was computed from).
            Pass "close" only for explicit same-bar/exploratory analysis.

    Returns:
        run_portfolio_simulation's result dict, plus: hedge_ratio,
        entry_spread (spread value at the most recent entry, None if the
        spread never crossed either entry threshold), current_spread (last
        spread value), n_round_trips (completed entry -> exit cycles),
        state (pd.Series — the daily long/short/flat spread position).

    Raises:
        ValidationError: missing symbol in price_data, or the spread never
        crosses entry_z (nothing to backtest).
    """
    missing = [s for s in (symbol_a, symbol_b) if s not in price_data]
    if missing:
        raise ValidationError(f"price_data is missing OHLCV for: {missing}")

    common_idx = (
        price_data[symbol_a]
        .index.intersection(price_data[symbol_b].index)
        .sort_values()
    )
    close_a = price_data[symbol_a]["Close"].loc[common_idx]
    close_b = price_data[symbol_b]["Close"].loc[common_idx]

    spread = compute_spread(close_a, close_b, hedge_ratio=hedge_ratio)
    z = spread_zscore(spread, window=zscore_window)
    state = _spread_state(z, entry_z=entry_z, exit_z=exit_z)

    pos_diff = state.diff()
    pos_diff.iloc[0] = state.iloc[0]
    transition_dates = pos_diff[pos_diff != 0].index

    if len(transition_dates) == 0:
        raise ValidationError(
            "spread z-score never crossed entry_z — no trade to backtest "
            f"(entry_z={entry_z}, z range=[{float(z.min()):.2f}, {float(z.max()):.2f}])"
        )

    # hedge_ratio is a SHARE ratio (1 share of A per hedge_ratio shares of
    # B), not a dollar-weight ratio — converting to dollar weights requires
    # the price the trade will ACTUALLY execute at (see
    # _execution_prices_for_weights), recomputed at every transition since
    # Close_a/Close_b (and Open_a/Open_b) drift apart over time (a static
    # split, as if computed once from a single date's prices, would
    # silently size the hedge leg wrong as soon as prices move).
    sign_hedge = 1.0 if hedge_ratio >= 0 else -1.0
    abs_hedge = abs(hedge_ratio)
    weight_a_vals = []
    weight_b_vals = []
    for d in transition_dates:
        price_a, price_b = _execution_prices_for_weights(
            d, common_idx, close_a, close_b, price_data, symbol_a, symbol_b, fill_price,
        )
        denom = price_a + abs_hedge * price_b
        weight_a_d = gross_leverage * price_a / denom
        weight_b_d = sign_hedge * gross_leverage * abs_hedge * price_b / denom
        weight_a_vals.append(float(state.loc[d]) * weight_a_d)
        weight_b_vals.append(-float(state.loc[d]) * weight_b_d)

    # A large |hedge_ratio| (or gross_leverage > 1) can put one leg's own
    # weight above run_portfolio_simulation's default max_position_pct=1.0,
    # which would reject an otherwise-valid gross_leverage. Derive the bound
    # from the actual largest realized leg weight instead of hardcoding 1.0.
    max_leg_weight = max((abs(w) for w in weight_a_vals + weight_b_vals), default=0.0)

    target_weights = pd.DataFrame(
        {symbol_a: weight_a_vals, symbol_b: weight_b_vals},
        index=transition_dates,
    )

    result = run_portfolio_simulation(
        price_data,
        target_weights,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        max_gross_leverage=gross_leverage + 1e-6,
        max_position_pct=max_leg_weight + 1e-6,
        fill_price=fill_price,
    )

    n_round_trips = int(sum(1 for d in transition_dates if state.loc[d] == 0.0))
    entries = [d for d in transition_dates if state.loc[d] != 0.0]
    entry_spread = float(spread.loc[entries[-1]]) if entries else None

    logger.debug(
        "[pair_backtest] %s/%s  hedge_ratio=%.4f  transitions=%d  round_trips=%d  final_equity=%.2f",
        symbol_a,
        symbol_b,
        hedge_ratio,
        len(transition_dates),
        n_round_trips,
        result["final_equity"],
    )

    result["hedge_ratio"] = hedge_ratio
    result["entry_spread"] = entry_spread
    result["current_spread"] = float(spread.iloc[-1])
    result["n_round_trips"] = n_round_trips
    result["state"] = state
    return result
