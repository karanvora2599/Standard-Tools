"""
A cash book carried through time with a futures hedge on top of it.

WHY THIS IS NOT `run_futures_simulation`. That function simulates a futures
POSITION: margin, variation, rolls, expiry. This one simulates a HEDGED
BOOK -- a cash portfolio whose exposure is being neutralised, where the
hedge is re-sized as the book's value drifts and as its beta is
re-estimated. The two questions are different:

    run_futures_simulation   what did this futures position do?
    run_futures_hedge_backtest   did hedging this book actually work?

The second is the one a Delta One desk asks, and answering it needs both
halves at once: the cash P&L, the futures P&L, and the residual that the
hedge failed to remove.

WHAT IT DOES NOT DO. It does not re-estimate beta from the data unless
asked. A rolling beta is a choice with a lookback and a bias, and
`analysis.correlation.rolling_beta` already makes that choice explicitly --
so a caller who wants it passes the series and gets a beta path, and a
caller who has a beta from a factor model passes that instead. Inventing a
lookback here would bury the most consequential parameter in the
simulation.

NOTHING HERE PRICES A CONTRACT. Sizing is `delta_one.hedging.futures_hedge`
and the futures account is `backtest.futures_engine.run_futures_simulation`.
This module decides WHEN to re-hedge and assembles the two P&L streams; the
economics live where they already lived.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.backtest.futures_engine import run_futures_simulation
from standard_quant_tools.delta_one.hedging import futures_hedge, hedge_effectiveness
from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_scalar

logger = logging.getLogger(__name__)

__all__ = ["run_futures_hedge_backtest"]

#: Re-hedging schedules. `drift` is the one a desk actually runs: re-size
#: only when the residual exposure has grown past a band, because every
#: re-hedge costs two spreads and a commission.
REHEDGE_RULES = ("daily", "weekly", "monthly", "drift")


def _rehedge_dates(
    index: pd.Index, rule: str, residual_fraction: Sequence[float], band: float
) -> np.ndarray:
    """Which bars re-size the hedge. True on the first bar always."""
    flags = np.zeros(len(index), dtype=bool)
    flags[0] = True
    if rule == "daily":
        flags[:] = True
    elif rule in ("weekly", "monthly"):
        stamps = pd.DatetimeIndex(index)
        period = stamps.isocalendar().week if rule == "weekly" else stamps.month
        changed = np.asarray(period)[1:] != np.asarray(period)[:-1]
        flags[1:] |= changed
    else:  # drift
        # Re-hedge when the residual has drifted outside the band. This is
        # the only rule that reacts to the book rather than to the calendar,
        # and it is why `band` exists.
        flags |= np.abs(np.asarray(residual_fraction, dtype=float)) > band
    return flags


def run_futures_hedge_backtest(
    *,
    portfolio_values: Mapping[Any, float],
    future_prices: Mapping[Any, float],
    multiplier: float,
    portfolio_beta: float | Sequence[float] = 1.0,
    future_beta: float = 1.0,
    initial_margin: float = 0.0,
    commission_per_contract: float = 0.0,
    slippage_points: float = 0.0,
    collateral_rate: float = 0.0,
    contract_map: Optional[Mapping[Any, str]] = None,
    rehedge: str = "monthly",
    drift_band: float = 0.05,
    allow_fractional: bool = False,
) -> Dict[str, Any]:
    """
    Carry a cash book and its futures hedge together, bar by bar.

    `portfolio_values` is the book's mark, NOT its returns: a hedge is
    sized off notional, and a return series has thrown that away. The book
    is taken as given -- this simulates the hedge, not the strategy.

    `portfolio_beta` accepts a scalar or a per-bar sequence. A sequence is
    how a caller supplies a rolling or factor-model beta; nothing is
    estimated here, because the lookback is the most consequential choice in
    the simulation and it belongs to whoever makes it.

    Returns the two P&L streams separately -- that is the entire point.
    A hedged book that made money because the hedge lost less than the cash
    leg is a different outcome from one where the hedge worked, and a single
    net number cannot tell them apart.

    Raises:
        ValidationError: fewer than two bars, a rehedge rule that is not one
        of REHEDGE_RULES, a non-positive multiplier, or portfolio_beta given
        as a sequence of the wrong length.
    """
    if rehedge not in REHEDGE_RULES:
        raise ValidationError(
            f"run_futures_hedge_backtest: rehedge={rehedge!r} is not one of "
            f"{list(REHEDGE_RULES)}."
        )
    multiplier = require_finite_scalar(
        multiplier, "multiplier", "run_futures_hedge_backtest", minimum=1e-12
    )
    drift_band = require_finite_scalar(
        drift_band, "drift_band", "run_futures_hedge_backtest", minimum=0.0
    )

    book = pd.Series(dict(portfolio_values), dtype=float).sort_index()
    futures = pd.Series(dict(future_prices), dtype=float).sort_index()
    index = book.index.intersection(futures.index)
    if len(index) < 2:
        raise ValidationError(
            f"run_futures_hedge_backtest: {len(index)} overlapping dates "
            "between portfolio_values and future_prices. A hedge needs at "
            "least two bars to have done anything."
        )
    book = book.loc[index]
    futures = futures.loc[index]

    if isinstance(portfolio_beta, (int, float)) and not isinstance(
        portfolio_beta, bool
    ):
        betas = np.full(len(index), float(portfolio_beta))
    else:
        betas = np.asarray(list(portfolio_beta), dtype=float)
        if betas.size != len(index):
            raise ValidationError(
                f"run_futures_hedge_backtest: portfolio_beta has "
                f"{betas.size} entries for {len(index)} overlapping dates."
            )

    # ── pass one: how many contracts on each bar ─────────────────────────
    #
    # Sized through `futures_hedge`, which is where the dollar-beta ->
    # contracts arithmetic and the rounding live. Repeating it here would
    # be a second implementation of the one thing this whole runtime is for.
    residual_fraction: List[float] = []
    exact: List[float] = []
    for i in range(len(index)):
        sized = futures_hedge(
            portfolio_value=float(book.iloc[i]),
            portfolio_beta=float(betas[i]),
            future_price=float(futures.iloc[i]),
            multiplier=multiplier,
            future_beta=future_beta,
        )
        exact.append(float(sized["contracts_exact"]))
        residual_fraction.append(
            float(sized["residual_dollar_beta"]) / float(book.iloc[i])
            if book.iloc[i]
            else 0.0
        )

    flags = _rehedge_dates(index, rehedge, residual_fraction, drift_band)

    # Held contracts step only on a re-hedge bar; between them the position
    # is stale, which is the whole reason a calendar rule leaves residual.
    held: List[float] = []
    current = 0.0
    for i in range(len(index)):
        if flags[i]:
            current = exact[i] if allow_fractional else float(round(exact[i]))
        held.append(current)

    targets = {index[i]: held[i] for i in range(len(index)) if flags[i]}

    # ── pass two: the futures account ────────────────────────────────────
    # COLLATERAL SIZED SO MARGIN NEVER BINDS, on purpose.
    #
    # This measures the hedge's P&L CONTRIBUTION to a book that is already
    # funded, not the solvency of a separately capitalised futures account.
    # An arbitrary collateral figure -- 10% of the book was my first guess
    # -- lets `run_futures_simulation`'s maintenance logic liquidate the
    # hedge mid-simulation, so the thing being measured would be a margin
    # call rather than the hedge. A caller who wants that question should
    # run `run_futures_backtest` directly with their real margin terms.
    #
    # `total_variation_margin` and `n_margin_calls` are still reported, so a
    # hedge whose funding requirement is implausible is visible rather than
    # hidden by this choice.
    peak_notional = float(
        (np.abs(np.asarray(held, dtype=float)) * futures.to_numpy() * multiplier).max()
    )
    hedge = run_futures_simulation(
        prices={k: float(v) for k, v in futures.items()},
        target_contracts=targets,
        multiplier=multiplier,
        initial_capital=max(peak_notional * 2.0, 1.0),
        initial_margin=initial_margin,
        commission_per_contract=commission_per_contract,
        slippage_points=slippage_points,
        collateral_rate=collateral_rate,
        contract_map=contract_map,
        allow_fractional=allow_fractional,
    )

    # ── the two streams, kept apart ──────────────────────────────────────
    cash_pnl = book.diff().fillna(0.0)
    hedge_equity = pd.Series(hedge["equity_curve"], index=index, dtype=float)
    hedge_pnl = hedge_equity.diff().fillna(0.0)
    combined_pnl = cash_pnl + hedge_pnl

    unhedged_returns = (cash_pnl / book.shift(1)).fillna(0.0)
    hedged_returns = (combined_pnl / book.shift(1)).fillna(0.0)
    future_returns = futures.pct_change(fill_method=None).fillna(0.0)

    # THE BETA BEING NEUTRALISED, not a contract count.
    #
    # `hedge_effectiveness` computes `portfolio - hedge_ratio * hedge`, so
    # it wants the fraction of the book's return the hedge should remove --
    # for a beta-neutral hedge, the portfolio beta. My first version passed
    # `mean(held) * multiplier / mean(book)`, a contract-notional ratio of
    # -0.18, and `beta_after` came back 1.1122 against a `beta_before` of
    # 1.1124: the hedge looked like it did nothing, because the number
    # describing it was the wrong quantity.
    #
    # NOT negated. `hedge_effectiveness` computes `portfolio + ratio *
    # hedge` and its docstring says a short hedge must therefore carry a
    # NEGATIVE ratio -- it even warns when one does not. `held` is already
    # negative for a short hedge, so flipping it (which I did first) made
    # `beta_after` come back 2.2324 against a `beta_before` of 1.1124: the
    # hedge was being added to the book instead of removed from it, exactly
    # doubling the exposure it was meant to cancel.
    #
    # Averaged over the realised path rather than taken from the input, so
    # rounding and a stale calendar hedge both show up.
    hedged_notional = np.asarray(held, dtype=float) * futures.to_numpy() * multiplier
    effective_ratio = (
        float(np.mean(hedged_notional / book.to_numpy())) * future_beta
        if float(book.abs().min()) > 0
        else 0.0
    )
    effectiveness = hedge_effectiveness(
        portfolio_returns=unhedged_returns.tolist(),
        hedge_returns=future_returns.tolist(),
        hedge_ratio=effective_ratio,
    )

    warnings: List[str] = list(hedge.get("warnings", []))
    n_rehedges = int(flags.sum())
    if n_rehedges <= 1:
        warnings.append(
            "The hedge was sized once and never revisited, so every bit of "
            "drift in the book's value or beta is sitting in the residual. "
            "That is a valid baseline and it is not a hedging programme."
        )
    if rehedge == "drift" and n_rehedges > len(index) * 0.5:
        warnings.append(
            f"The drift band of {drift_band:.1%} triggered {n_rehedges} "
            f"re-hedges over {len(index)} bars. A band that fires most days "
            "is a daily rule paying for the pretence of being a band."
        )

    logger.debug(
        "[futures_hedge_backtest] bars=%d  rehedges=%d  rule=%s",
        len(index),
        n_rehedges,
        rehedge,
    )

    return {
        "n_bars": int(len(index)),
        "rehedge_rule": rehedge,
        "n_rehedges": n_rehedges,
        "contracts_held": {str(k): float(v) for k, v in zip(index, held)},
        # THE TWO LEGS, SEPARATELY. A net number cannot distinguish a hedge
        # that worked from a cash leg that happened to fall less.
        "cash_pnl": float(cash_pnl.sum()),
        "hedge_pnl": float(hedge_pnl.sum()),
        "combined_pnl": float(combined_pnl.sum()),
        "unhedged_volatility": float(unhedged_returns.std(ddof=1)),
        "hedged_volatility": float(hedged_returns.std(ddof=1)),
        "volatility_reduction": (
            1.0
            - float(hedged_returns.std(ddof=1)) / float(unhedged_returns.std(ddof=1))
            if float(unhedged_returns.std(ddof=1)) > 0
            else None
        ),
        #  is what hedge_effectiveness calls the residual.
        "residual_beta": effectiveness.get("beta_after"),
        "effective_hedge_ratio": effective_ratio,
        "hedge_effectiveness": effectiveness,
        "hedge_variation_margin": float(hedge.get("total_variation_margin", 0.0)),
        "hedge_margin_calls": int(len(hedge.get("margin_calls", []) or [])),
        "peak_hedge_notional": peak_notional,
        "total_commission": float(hedge.get("total_commission", 0.0)),
        "total_slippage": float(hedge.get("total_slippage", 0.0)),
        "n_rolls": int(hedge.get("n_rolls", 0)),
        "warnings": warnings,
    }
