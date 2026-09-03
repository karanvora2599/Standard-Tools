"""
modeling.portfolio_eval: evaluate a registered model as a PORTFOLIO,
through the shared-cash simulator, rather than as a per-ticker signal
panel.

Why this exists alongside `bridge.oos_predictions_to_signal_panel`:

    bridge  -> {-1, 0, +1} per ticker -> run_signal_panel_backtest
               (backtest/panel.py: every ticker gets its OWN
               initial_capital, and the per-ticker return streams are
               blended afterwards)

    here    -> target WEIGHTS per date  -> run_portfolio_simulation
               (backtest/portfolio_engine.py: one shared cash balance,
               positions sized against current account equity, weights
               drifting between rebalances)

The bridge is not wrong — sign is the only defensible conversion for an
engine that multiplies a SCORE straight into `strategy_return` as a raw
leverage multiplier. But it answers a narrower question than a
cross-sectional model is trying to answer. A model that ranks 50 names
predicts an ORDERING; reducing that ordering to three values, and then
giving every name its own independent capital, discards both the rank and
the fact that the names compete for the same dollars. The economically
meaningful question — "what would this model have done to an account" —
needs weights and one balance.

Leakage: this consumes `run_model_experiment`'s walk-forward OOS
predictions, exactly as the bridge does, and never `score_model`. Every
prediction here came from a fold-local model that did not see that fold's
dates in training, so the concatenation across folds is a genuine
out-of-sample track record. `score_model`'s estimator is the final
full-panel refit; asking it to "predict" historical dates would be
in-sample, and the resulting equity curve would be fiction. There is no
parameter on this module that can select that path.

What is NOT reimplemented here: the score -> weight math. `backtest.sizing`
already builds gross-normalized weight panels from a score panel, and is
called as-is (`_SIZERS` below). This module contributes the parts sizing.py
has no concept of — a per-position cap, an exact gross AND net exposure
target, a rebalance schedule, sparse-cross-section handling, and the
artifact/provenance plumbing that makes the result auditable.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.audit.hashing import hash_dataframe
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.backtest.sizing import (
    equal_weight_top_bottom,
    rank_weighted,
    vol_scaled,
    zscore_normalized,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.return_metrics import (
    annualized_volatility,
    cagr,
    cumulative_return,
)
from standard_quant_tools.metrics.risk_metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from standard_quant_tools.modeling.specs import TASKS

from . import artifacts as _artifacts
from .bridge import _assert_continuous_calendar, _validate_predictions_frame
from .dataset.fetch import fetch_universe_ohlcv
from .features.base import periods_per_year_for_interval
from .registry.model_registry import load_dataset_spec, load_manifest
from .specs import PortfolioSimSpec, PredictionTransformSpec

logger = logging.getLogger(__name__)

__all__ = [
    "apply_exposure_targets",
    "evaluate_model_portfolio",
    "predictions_to_score_panel",
    "select_rebalance_dates",
    "transform_predictions_to_weights",
]

# Below this, a float is treated as zero rather than as a direction. Shared
# by every book-splitting and normalization step so "is this name long" has
# one answer throughout, instead of each step picking its own epsilon.
_EPS = 1e-12


# ── Predictions -> score panel ──────────────────────────────────────────


def predictions_to_score_panel(
    predictions_df: pd.DataFrame, task: str, source: str = "<predictions>"
) -> pd.DataFrame:
    """
    Reshape a long (date, entity, prediction) OOS artifact into the wide
    date x entity panel every sizing function expects.

    Classification predictions are recentered to `proba - 0.5` rather than
    left as a raw positive-class probability. It makes `method="sign"` mean
    the same thing for every task: positive score = the model is bullish. A
    raw probability is in [0, 1], so its sign is +1 for every name and
    every date — a "long everything" portfolio that looks like a signal.

    A RANKING model's predictions pass through unchanged, like a
    regressor's, because a ranker emits a relative score and not a
    probability. This docstring used to say ranking was "unaffected" while
    the guard below refused it outright, so a ranker could be trained and
    never evaluated as a portfolio.

    Missing (entity, date) pairs stay NaN here rather than being filled
    with 0.0. Zero is a real score — it is the middle of the cross-section
    for a centered panel — so filling with it would rank an entity the
    model said nothing about ahead of every name it was bearish on. The
    weighting step treats NaN as "not in the cross-section on that date"
    and gives it zero WEIGHT, which is the honest reading.
    """
    if task not in TASKS:
        raise ValidationError(f"task must be one of {list(TASKS)}, got {task!r}.")
    _validate_predictions_frame(predictions_df, source)
    panel = predictions_df.pivot(index="date", columns="entity", values="prediction")
    panel = panel.sort_index()
    panel.columns.name = None
    if task == "classification":
        panel = panel - 0.5
    return panel


# ── Rebalance schedule ──────────────────────────────────────────────────


def select_rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """
    Pick the rebalance dates out of the available prediction calendar.

    Takes the FIRST available date within each period, never the last.
    Both are lookahead-free in the sense that the WEIGHT is computed from
    that date's own prediction — but "last date in the month" is only
    knowable once the month has ended, so a schedule built that way cannot
    be reproduced live without waiting. First-of-period can.
    """
    if frequency == "daily":
        return pd.DatetimeIndex(dates)
    freq_code = {"weekly": "W", "monthly": "M"}.get(frequency)
    if freq_code is None:
        raise ValidationError(
            f"rebalance_frequency must be one of daily/weekly/monthly, got "
            f"{frequency!r}"
        )
    frame = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    periods = frame["date"].dt.to_period(freq_code)
    firsts = frame.groupby(periods, sort=True)["date"].min()
    return pd.DatetimeIndex(firsts.to_numpy())


# ── Exposure targeting and position capping ─────────────────────────────


def _cap_book(
    values: np.ndarray, target_gross: float, cap: float
) -> Tuple[np.ndarray, float]:
    """
    Scale one book (all-positive magnitudes) to `target_gross` with no
    single element exceeding `cap`, redistributing capped excess onto the
    names that still have headroom.

    Iterative because redistribution is not one-shot: pushing excess onto
    the uncapped names can lift one of THEM over the cap, which then has
    to be capped and redistributed in turn. Each pass strictly grows the
    capped set, so the loop terminates in at most len(values) passes.

    Returns (weights, realized_gross). realized_gross < target_gross only
    when the book is genuinely infeasible — n_names * cap < target_gross,
    i.e. there are not enough names to hold that much exposure under the
    cap. Reporting the shortfall is the point: silently exceeding the cap
    would break the risk limit, and silently rescaling the OTHER book
    would break the net-exposure target.
    """
    n = values.size
    if n == 0 or target_gross <= _EPS:
        return np.zeros(n, dtype=float), 0.0

    magnitude = np.abs(values)
    total = magnitude.sum()
    if total <= _EPS:
        # Every selected name scored exactly zero — proportional allocation
        # is undefined, so fall back to equal weight across the book.
        weights = np.full(n, target_gross / n, dtype=float)
    else:
        weights = magnitude * (target_gross / total)

    feasible_max = n * cap
    if feasible_max <= target_gross + _EPS:
        # Not enough names to reach target_gross under the cap. Everything
        # goes to the cap; the shortfall is reported, never hidden.
        return np.full(n, cap, dtype=float), float(feasible_max)

    capped = np.zeros(n, dtype=bool)
    for _ in range(n + 1):
        over = (weights > cap + _EPS) & ~capped
        if not over.any():
            break
        capped |= over
        weights[capped] = cap
        remaining = target_gross - cap * capped.sum()
        free = ~capped
        free_total = weights[free].sum()
        if free_total <= _EPS:
            weights[free] = remaining / max(int(free.sum()), 1)
        else:
            weights[free] *= remaining / free_total
    return weights, float(weights.sum())


def apply_exposure_targets(
    scores_row: np.ndarray,
    gross_exposure: float,
    net_exposure: float,
    max_position_weight: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Turn one date's signed raw weights into final weights hitting BOTH
    sum(|w|) == gross_exposure and sum(w) == net_exposure exactly.

    A single rescale cannot control two targets, so the vector is split
    into its long and short books and each is sized independently:

        long book gross  L = (gross + net) / 2
        short book gross S = (gross - net) / 2

    which gives L + S = gross and L - S = net by construction, for any
    |net| <= gross. Within each book, allocation is proportional to the
    incoming magnitude (so the sizing function's ranking survives) and
    then capped.

    NaN entries are treated as "not in this cross-section" and receive
    exactly 0.0.
    """
    n = scores_row.size
    weights = np.zeros(n, dtype=float)
    finite = np.isfinite(scores_row)
    long_mask = finite & (scores_row > _EPS)
    short_mask = finite & (scores_row < -_EPS)

    target_long = (gross_exposure + net_exposure) / 2.0
    target_short = (gross_exposure - net_exposure) / 2.0

    long_w, long_realized = _cap_book(
        scores_row[long_mask], target_long, max_position_weight
    )
    short_w, short_realized = _cap_book(
        np.abs(scores_row[short_mask]), target_short, max_position_weight
    )
    weights[long_mask] = long_w
    weights[short_mask] = -short_w

    diagnostics = {
        "n_long": int(long_mask.sum()),
        "n_short": int(short_mask.sum()),
        "target_long_gross": float(target_long),
        "target_short_gross": float(target_short),
        "realized_long_gross": long_realized,
        "realized_short_gross": short_realized,
        "realized_gross": long_realized + short_realized,
        "realized_net": long_realized - short_realized,
    }
    return weights, diagnostics


# ── Score panel -> target weights ───────────────────────────────────────

# The sizing functions this module delegates the ranking math to. Each
# returns a gross-normalized (date x ticker) panel; apply_exposure_targets
# then re-splits that panel into books to hit the net target and the cap,
# which sizing.py has no concept of. gross_leverage is passed as 1.0
# because that intermediate normalization is discarded by the re-split —
# only the RELATIVE magnitudes carry through.
_SIZERS = {
    "cross_sectional_rank": rank_weighted,
    "cross_sectional_zscore": zscore_normalized,
}


def _quantile_counts(n_names: int, long_q: float, short_q: float) -> Tuple[int, int]:
    """
    Names per book for a quantile portfolio, given how many names that
    date's cross-section actually has.

    Rounds each side to at least one name whenever its quantile is
    nonzero: with 8 names and long_quantile=0.1, int(0.8) == 0 would leave
    the long book empty and hand the whole gross target to a book that
    cannot absorb it. Then shrinks the pair, longs first, if rounding up
    made the two books overlap.
    """
    n_long = int(round(n_names * long_q)) if long_q > 0 else 0
    n_short = int(round(n_names * short_q)) if short_q > 0 else 0
    if long_q > 0:
        n_long = max(n_long, 1)
    if short_q > 0:
        n_short = max(n_short, 1)
    while n_long + n_short > n_names and (n_long > 0 or n_short > 0):
        if n_long >= n_short and n_long > 0:
            n_long -= 1
        elif n_short > 0:
            n_short -= 1
    return n_long, n_short


def _raw_weights_for_group(
    scores: pd.DataFrame,
    spec: PredictionTransformSpec,
    returns_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Run the chosen sizing function over a dense sub-panel (one group of
    dates sharing the same set of available entities)."""
    if spec.method == "sign":
        return np.sign(scores)

    if spec.method == "top_bottom_quantile":
        n_long, n_short = _quantile_counts(
            scores.shape[1], spec.long_quantile, spec.short_quantile
        )
        if n_long == 0 and n_short == 0:
            return pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        return equal_weight_top_bottom(
            scores, n_long=n_long, n_short=n_short, gross_leverage=1.0
        )

    if spec.volatility_scale:
        if returns_df is None:
            raise ValidationError(
                "volatility_scale=True requires returns; none were supplied."
            )
        return vol_scaled(
            scores,
            returns_df=returns_df,
            lookback=spec.volatility_lookback,
            gross_leverage=1.0,
        )
    return _SIZERS[spec.method](scores, gross_leverage=1.0)


def transform_predictions_to_weights(
    score_panel: pd.DataFrame,
    spec: PredictionTransformSpec,
    returns_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert a wide (date x entity) score panel into target weights the
    portfolio simulator can consume.

    Every date is transformed INDEPENDENTLY of every other date. That is
    what makes the result point-in-time: no rescaling, ranking or
    standardization step ever looks across dates, so a weight for date t
    could have been computed on date t with nothing but that day's
    cross-section. (The one exception is `volatility_scale`, which uses a
    TRAILING return window — backward-looking by construction.)

    Sparse cross-sections are handled by grouping dates that share the same
    set of available entities and running the sizing function once per
    group. Dropping every date with a missing entity would silently shorten
    the evaluation window, and forward-filling a stale prediction would
    make a model look like it had a view it did not have.

    Returns (weights, diagnostics).
    """
    if score_panel.empty:
        raise ValidationError("score panel is empty — nothing to transform")

    weights = pd.DataFrame(
        0.0, index=score_panel.index, columns=score_panel.columns, dtype=float
    )
    availability = score_panel.notna()
    # Group by the availability PATTERN so each sizing call sees a dense
    # sub-panel. In the common case every date has the same entities and
    # this is a single group.
    # Keyed by a bitstring rather than a tuple: pandas groupby treats a
    # tuple value as a potential multi-key, so grouping on a Series of
    # tuples is ambiguous in a way a plain string key is not.
    pattern = availability.apply(
        lambda row: "".join("1" if v else "0" for v in row.to_numpy()), axis=1
    )
    for _, dates in pattern.groupby(pattern, sort=False).groups.items():
        idx = pd.DatetimeIndex(dates)
        present = [c for c in score_panel.columns if availability.loc[idx[0], c]]
        if len(present) < 2:
            # A one-name "cross-section" has no ordering to exploit; every
            # cross-sectional method would either divide by a zero spread
            # or hand 100% to that single name on the strength of no
            # comparison at all. Left flat, and counted in diagnostics.
            continue
        sub_scores = score_panel.loc[idx, present]
        sub_returns = None
        if returns_df is not None:
            missing = [c for c in present if c not in returns_df.columns]
            if missing:
                raise ValidationError(
                    f"returns are missing column(s) {missing} needed for "
                    "volatility_scale."
                )
            sub_returns = returns_df[present]
        raw = _raw_weights_for_group(sub_scores, spec, sub_returns)
        weights.loc[idx, present] = raw.to_numpy(dtype=float)

    # Re-split every row into books to hit gross AND net exactly, and cap.
    # Done here rather than inside the group loop so the exposure contract
    # is enforced in exactly one place for every method.
    raw_matrix = weights.to_numpy(dtype=float).copy()
    # Entities absent on a date must not be eligible for a book: their raw
    # value is 0.0 (never assigned), which is already excluded by the
    # strict > / < comparisons in apply_exposure_targets, but NaN makes
    # the intent explicit for rows skipped entirely above.
    raw_matrix[~availability.to_numpy()] = np.nan

    per_date: List[Dict[str, float]] = []
    final = np.zeros_like(raw_matrix)
    for i in range(raw_matrix.shape[0]):
        final[i], diag = apply_exposure_targets(
            raw_matrix[i],
            spec.gross_exposure,
            spec.net_exposure,
            spec.max_position_weight,
        )
        per_date.append(diag)

    out = pd.DataFrame(final, index=score_panel.index, columns=score_panel.columns)

    realized_gross = np.array([d["realized_gross"] for d in per_date])
    shortfall = realized_gross < spec.gross_exposure - 1e-6
    empty_dates = np.array([d["n_long"] + d["n_short"] == 0 for d in per_date])
    diagnostics: Dict[str, Any] = {
        "n_dates": int(out.shape[0]),
        "n_entities": int(out.shape[1]),
        "mean_names_per_date": float(availability.sum(axis=1).mean()),
        "min_names_per_date": int(availability.sum(axis=1).min()),
        "mean_n_long": float(np.mean([d["n_long"] for d in per_date])),
        "mean_n_short": float(np.mean([d["n_short"] for d in per_date])),
        "mean_realized_gross": float(realized_gross.mean()),
        "mean_realized_net": float(np.mean([d["realized_net"] for d in per_date])),
        "n_dates_below_target_gross": int(shortfall.sum()),
        "n_dates_with_no_position": int(empty_dates.sum()),
        "max_abs_weight": float(np.nanmax(np.abs(final))) if final.size else 0.0,
    }

    # THE SHORTFALL HAS TO SAY SOMETHING. `n_dates_below_target_gross` was
    # counted and then left sitting in the dict, so a caller who asked for
    # gross 1.0 / net 0.0 and received a 100%-LONG book at gross 0.5 had
    # to notice a number nothing pointed at.
    #
    # It happens whenever the sizer does not centre its scores and the
    # scores are one-sided: `vol_scaled` divides by volatility and
    # normalizes gross without recentring, so all-positive predictions
    # produce no short book at all, and `apply_exposure_targets` can only
    # fill the long half. Measured on 40 names x 60 dates of positive
    # predictions: 60 of 60 dates short of target, mean net +0.34 against
    # a requested 0.0, and `warnings` was None.
    notes: List[str] = []
    if int(shortfall.sum()):
        mean_net = float(np.mean([d["realized_net"] for d in per_date]))
        notes.append(
            f"{int(shortfall.sum())} of {out.shape[0]} dates came in below "
            f"the requested gross exposure of {spec.gross_exposure:.2f} "
            f"(mean realized {realized_gross.mean():.4f}), and mean net was "
            f"{mean_net:+.4f} against a target of {spec.net_exposure:+.2f}. "
            "One book could not be filled. The usual cause is a sizing "
            "method that does not recentre -- volatility_scale=True divides "
            "by volatility and normalizes gross without recentring, so "
            "one-sided scores yield no short book."
        )
    if int(empty_dates.sum()):
        notes.append(f"{int(empty_dates.sum())} dates hold no position at all.")
    diagnostics["warnings"] = notes

    return out, diagnostics


# ── Full evaluation ─────────────────────────────────────────────────────


def _summarize_simulation(
    result: Dict[str, Any],
    periods_per_year: int,
    risk_free_rate: float,
    commission_pct: float,
    slippage_pct: float,
) -> Dict[str, float]:
    equity = result["equity_curve"].dropna()
    if equity.empty:
        raise ValidationError("simulation produced an empty equity curve")
    returns = equity.pct_change(fill_method=None).dropna()

    rebalance_log = result["rebalance_log"]
    turnover = (
        rebalance_log["turnover_pct"].astype(float)
        if not rebalance_log.empty
        else pd.Series(dtype=float)
    )
    years = max(len(equity) / periods_per_year, 1e-9)

    metrics: Dict[str, float] = {
        "cumulative_return": float(cumulative_return(equity)),
        "cagr": float(cagr(equity, periods_per_year)),
        "annualized_volatility": (
            float(annualized_volatility(returns, periods_per_year))
            if not returns.empty
            else 0.0
        ),
        "sharpe_ratio": (
            float(sharpe_ratio(returns, risk_free_rate, periods_per_year))
            if not returns.empty
            else 0.0
        ),
        "sortino_ratio": (
            float(sortino_ratio(returns, risk_free_rate, periods_per_year))
            if not returns.empty
            else 0.0
        ),
        "max_drawdown": float(max_drawdown(equity)),
        "calmar_ratio": float(calmar_ratio(equity, periods_per_year)),
        "final_equity": float(result["final_equity"]),
        "n_rebalances": int(len(rebalance_log)),
        "mean_turnover_pct": float(turnover.mean()) if not turnover.empty else 0.0,
        "annualized_turnover": (
            float(turnover.sum() / years) if not turnover.empty else 0.0
        ),
        "mean_gross_exposure": float(result["gross_exposure_curve"].mean()),
        "mean_net_exposure": float(result["net_exposure_curve"].mean()),
        "mean_n_positions": (
            float(rebalance_log["n_positions"].mean())
            if not rebalance_log.empty
            else 0.0
        ),
        # Derived, not measured. The simulator deducts costs from cash but
        # does not report a total, so this reconstructs the commission +
        # spread component from realized turnover. It EXCLUDES borrow fees,
        # margin interest and any impact model, so it is a floor on total
        # cost drag, not the whole of it.
        "estimated_cost_drag_pct": (
            float(turnover.sum() * (commission_pct + slippage_pct))
            if not turnover.empty
            else 0.0
        ),
    }
    return metrics


def evaluate_model_portfolio(
    model_id: str,
    transform: Optional[PredictionTransformSpec] = None,
    portfolio: Optional[PortfolioSimSpec] = None,
) -> Dict[str, Any]:
    """
    Evaluate a registered model's out-of-sample predictions as a
    shared-cash portfolio, and persist the target weights that produced
    the result.

    Pipeline, all of it recorded in the returned provenance block:

        manifest -> verified OOS predictions artifact
                 -> wide score panel (per-date cross-sections)
                 -> rebalance-date selection
                 -> target weights (gross/net/cap enforced per date)
                 -> persisted, content-addressed weights artifact
                 -> run_portfolio_simulation
                 -> economic metrics

    Returns a dict — the tool layer wraps it in a pydantic result model.
    """
    transform = transform or PredictionTransformSpec()
    portfolio = portfolio or PortfolioSimSpec()
    warnings: List[str] = []

    if transform.gross_exposure > portfolio.max_gross_leverage + 1e-9:
        raise ValidationError(
            f"transform.gross_exposure ({transform.gross_exposure}) exceeds "
            f"portfolio.max_gross_leverage ({portfolio.max_gross_leverage}) — the "
            "simulator would reject every rebalance date. Raise the leverage "
            "limit deliberately, or lower the target gross."
        )
    if transform.max_position_weight > portfolio.max_position_pct + 1e-9:
        raise ValidationError(
            f"transform.max_position_weight ({transform.max_position_weight}) "
            f"exceeds portfolio.max_position_pct ({portfolio.max_position_pct}), "
            "which the simulator enforces — every capped position would be "
            "rejected."
        )

    manifest = load_manifest(model_id)
    dataset_spec = load_dataset_spec(model_id)
    interval = str(dataset_spec.get("interval", "1d"))
    provider_name = str(dataset_spec.get("provider", "yfinance"))

    periods_per_year = periods_per_year_for_interval(interval)
    if periods_per_year is None:
        periods_per_year = 252
        warnings.append(
            f"interval={interval!r} has no defined bars-per-year without an "
            "exchange calendar; annualized metrics (Sharpe, CAGR, volatility, "
            "Calmar, annualized turnover) were computed with 252 and are "
            "wrong by a fixed factor for this interval."
        )

    # Verified before loading, for the same reason bridge.py verifies:
    # structural validation passes on an edited file that kept its shape,
    # so an altered prediction column would otherwise produce a clean and
    # entirely fictional equity curve.
    predictions_uri = str(manifest.oos_predictions_uri)
    _artifacts.verify_file(
        Path(predictions_uri),
        manifest.content_hashes.get("oos_predictions"),
        "oos_predictions",
    )
    predictions_df = _artifacts.load_artifact(predictions_uri)
    score_panel = predictions_to_score_panel(
        predictions_df, manifest.task, predictions_uri
    )

    skipped_folds = (manifest.validation_report or {}).get("skipped_folds") or None
    # The same continuity contract the bridge enforces, and for a related
    # reason: a hole in the OOS calendar means the model produced nothing
    # over a span, and the portfolio would sit in a stale position across
    # it while the equity curve keeps marking to market. Here the gap does
    # NOT compress the price axis (the simulator runs on its own master
    # calendar, not the signal index), so the distortion is different in
    # kind — but a track record with an unexplained hold-through is still
    # not the model's out-of-sample performance.
    _assert_continuous_calendar(
        pd.DatetimeIndex(score_panel.index), predictions_uri, skipped_folds
    )

    entities = [str(c) for c in score_panel.columns]
    if len(entities) < 2:
        raise ValidationError(
            f"model {model_id!r} has OOS predictions for {len(entities)} "
            "entity(ies). A portfolio evaluation needs a cross-section to "
            "allocate across — with one name there is no allocation decision, "
            "only a timing one. Use bridge.oos_predictions_to_signal_panel + "
            "run_signal_panel_backtest for single-name timing."
        )

    # Prices over the DATASET's full window, not just the prediction span:
    # volatility_scale needs trailing history before the first rebalance,
    # and the simulator needs a bar after the last rebalance for next_open
    # fills. Re-fetching the training window costs nothing beyond the
    # provider cache and guarantees both.
    provider = DataFactory.get_provider(provider_name)
    price_data = fetch_universe_ohlcv(
        provider,
        entities,
        str(dataset_spec["start"]),
        str(dataset_spec["end"]),
        interval,
    )
    missing_prices = [e for e in entities if e not in price_data]
    if missing_prices:
        raise ValidationError(
            f"no price data returned for {missing_prices} — every entity with "
            "OOS predictions needs prices to simulate against."
        )

    # The simulator's own master calendar is the intersection of every
    # ticker's index; a rebalance date outside it is rejected there. Doing
    # the same intersection here means a date that cannot be traded is
    # dropped with an explanation instead of failing the whole run.
    master_index = price_data[entities[0]].index
    for entity in entities[1:]:
        master_index = master_index.intersection(price_data[entity].index)
    master_index = master_index.sort_values()

    rebalance_dates = select_rebalance_dates(
        pd.DatetimeIndex(score_panel.index), transform.rebalance_frequency
    )
    tradable = rebalance_dates.intersection(master_index)
    n_untradable = len(rebalance_dates) - len(tradable)
    if n_untradable:
        warnings.append(
            f"{n_untradable} of {len(rebalance_dates)} rebalance date(s) are not "
            "in the master trading calendar (the intersection of every entity's "
            "price index) and were dropped."
        )
    rebalance_dates = pd.DatetimeIndex(tradable)

    if portfolio.fill_price == "next_open" and len(rebalance_dates):
        last_bar = master_index[-1]
        if rebalance_dates[-1] == last_bar:
            rebalance_dates = rebalance_dates[:-1]
            warnings.append(
                "the final rebalance date is the last bar in the trading "
                "calendar, so there is no following open to fill against; it "
                "was dropped (fill_price='next_open')."
            )
    if len(rebalance_dates) < 2:
        raise ValidationError(
            f"only {len(rebalance_dates)} tradable rebalance date(s) remain after "
            f"applying rebalance_frequency={transform.rebalance_frequency!r} to "
            f"{len(score_panel.index)} prediction date(s). Use a higher frequency, "
            "or train over a longer window."
        )

    returns_df = None
    if transform.volatility_scale:
        # Built on the price data's own bar frequency and only THEN sampled
        # at the rebalance dates — the same ordering vol_scaled documents,
        # because reindexing first would turn an N-bar volatility window
        # into N rebalance-period observations.
        returns_df = pd.DataFrame(
            {e: price_data[e]["Close"].pct_change(fill_method=None) for e in entities}
        )

    weights, transform_diagnostics = transform_predictions_to_weights(
        score_panel.loc[rebalance_dates], transform, returns_df
    )

    if transform_diagnostics["n_dates_below_target_gross"]:
        warnings.append(
            f"{transform_diagnostics['n_dates_below_target_gross']} rebalance "
            f"date(s) could not reach gross_exposure={transform.gross_exposure} "
            f"under max_position_weight={transform.max_position_weight} — too few "
            "names in a book. Realized gross on those dates is lower than "
            "requested (mean realized gross "
            f"{transform_diagnostics['mean_realized_gross']:.4f}); the residual "
            "sat in cash."
        )
    if transform_diagnostics["n_dates_with_no_position"]:
        warnings.append(
            f"{transform_diagnostics['n_dates_with_no_position']} rebalance "
            "date(s) produced no position at all (no entity scored on either "
            "side); the portfolio was fully in cash across them."
        )
    if portfolio.fill_price == "close":
        warnings.append(
            "fill_price='close' executes each weight at the same bar's close, "
            "the bar its own features were computed from — look-ahead. The "
            "resulting metrics are optimistic and are not a live-tradable "
            "track record."
        )
    if transform.volatility_scale and transform.method in (
        "top_bottom_quantile",
        "sign",
    ):
        warnings.append(
            f"volatility_scale=True has no effect with method="
            f"{transform.method!r}: its weights come from book membership, not "
            "from score magnitude, so scaling the score cannot change them."
        )
    warnings.extend(manifest.dataset_warnings)

    weights_hash = hash_dataframe(weights)
    weights_uri = _artifacts.save_artifact(
        weights,
        run_id=model_id,
        name=f"target_weights_{weights_hash}",
        overwrite=True,
    )

    logger.debug(
        "[evaluate_model_portfolio] model=%s  entities=%d  rebalances=%d  "
        "method=%s  freq=%s  gross=%.3f  net=%.3f  fill=%s",
        model_id,
        len(entities),
        len(rebalance_dates),
        transform.method,
        transform.rebalance_frequency,
        transform.gross_exposure,
        transform.net_exposure,
        portfolio.fill_price,
    )

    simulation = run_portfolio_simulation(
        price_data=price_data,
        target_weights=weights,
        initial_capital=portfolio.initial_capital,
        commission_pct=portfolio.commission_pct,
        slippage_pct=portfolio.slippage_pct,
        max_gross_leverage=portfolio.max_gross_leverage,
        max_position_pct=portfolio.max_position_pct,
        fill_price=portfolio.fill_price,
        borrow_fee_bps=portfolio.borrow_fee_bps,
        margin_interest_rate=portfolio.margin_interest_rate,
        max_adv_participation=portfolio.max_adv_participation,
    )
    warnings.extend(simulation.get("warnings", []))

    metrics = _summarize_simulation(
        simulation,
        periods_per_year,
        portfolio.risk_free_rate,
        portfolio.commission_pct,
        portfolio.slippage_pct,
    )

    equity_curve = simulation["equity_curve"]
    equity_hash = hash_dataframe(equity_curve.to_frame(name="equity"))
    equity_uri = _artifacts.save_artifact(
        equity_curve,
        run_id=model_id,
        name=f"portfolio_equity_{equity_hash}",
        overwrite=True,
    )

    return {
        "model_id": model_id,
        "metrics": metrics,
        "transform_diagnostics": transform_diagnostics,
        "coverage": {
            "n_entities": len(entities),
            "n_prediction_dates": int(len(score_panel.index)),
            "n_rebalance_dates": int(len(rebalance_dates)),
            "first_rebalance_date": str(rebalance_dates[0].date()),
            "last_rebalance_date": str(rebalance_dates[-1].date()),
            "n_simulated_bars": int(len(equity_curve)),
            "entities": entities,
        },
        "target_weights_uri": weights_uri,
        "equity_curve_uri": equity_uri,
        # Every input that determined the numbers above, so a reported
        # Sharpe can be traced back to the exact predictions and weights
        # that produced it. The predictions hash is the manifest's own
        # recorded digest (already verified above), so all three identify
        # bytes on disk rather than a path that may have been rewritten.
        "provenance": {
            "oos_predictions_uri": predictions_uri,
            "oos_predictions_hash": manifest.content_hashes.get("oos_predictions", ""),
            "target_weights_hash": weights_hash,
            "equity_curve_hash": equity_hash,
            "dataset_id": manifest.dataset_id,
            "dataset_spec_hash": manifest.dataset_spec_hash,
            "task": manifest.task,
            "estimator_type": manifest.estimator_type,
            "interval": interval,
            "provider": provider_name,
            "periods_per_year": int(periods_per_year),
            "transform_spec": transform.model_dump(),
            "portfolio_spec": portfolio.model_dump(),
        },
        "warnings": warnings,
    }
