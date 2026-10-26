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
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.audit.hashing import canonical_frame_hash
from standard_quant_tools.backtest.costs import DEFAULT_DAY_COUNT
from standard_quant_tools.backtest.financing import (
    accrued_rate_per_bar,
    ratios_over_accrued_rate,
    ratios_over_benchmark,
    resolve_debit_rate,
)
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.backtest.sizing import (
    equal_weight_top_bottom,
    rank_weighted,
    vol_adjusted_scores,
    zscore_normalized,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.return_metrics import (
    annualized_volatility,
    cagr,
    cumulative_return,
)
from standard_quant_tools.metrics.risk_metrics import max_drawdown
from standard_quant_tools.modeling.specs import TASKS

from . import artifacts as _artifacts
from .assets import fetch_plan
from .bridge import (
    _assert_continuous_calendar,
    _refuse_cpcv,
    _validate_predictions_frame,
)
from .dataset.fetch import fetch_universe_ohlcv
from .features.base import periods_per_year_for_interval
from .registry.model_registry import (
    load_dataset_spec,
    load_manifest,
    resolve_model_artifact,
)
from .specs import MeanVarianceSpec, PortfolioSimSpec, PredictionTransformSpec

logger = logging.getLogger(__name__)

__all__ = [
    "apply_exposure_targets",
    "cross_sectional_z",
    "evaluate_model_portfolio",
    "evaluate_predictions_portfolio",
    "holding_period_bars",
    "mean_variance_weights",
    "predictions_to_score_panel",
    "realized_rank_ic",
    "select_rebalance_dates",
    "trailing_ic",
    "transform_predictions_to_weights",
]

#: The form of `target_weights_hash` and `equity_curve_hash`, recorded in a
#: result's provenance as `frame_hash_version`. 2 is
#: `audit.canonical_frame_hash`, which does not depend on the pandas
#: version. A result without the key carries `audit.hash_dataframe`
#: (version 1), which covers the resolution the frames' dates are stored
#: at. Both frames are indexed by dates read from the predictions and the
#: prices, which pandas 3 stores at `[us]` where pandas 2 stores `[ns]`, so
#: identical weights were hashed and named differently under the two.
FRAME_HASH_VERSION = 2

# Below this, a float is treated as zero rather than as a direction. Shared
# by every book-splitting and normalization step so "is this name long" has
# one answer throughout, instead of each step picking its own epsilon.
_EPS = 1e-12


# ── Predictions -> score panel ──────────────────────────────────────────


def predictions_to_score_panel(
    predictions_df: pd.DataFrame,
    task: Optional[str],
    source: str = "<predictions>",
    proba_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Reshape a long (date, entity, prediction) OOS artifact into the wide
    date x entity panel every sizing function expects.

    THE one implementation of this reshape. `meta/convert.py` used to carry
    a second one that checked three column names and nothing else, so a
    duplicate (entity, date) pair silently overwrote itself there while the
    same frame was refused here — two answers to one question, and the
    quieter one was the one an agent reached through `convert_reference`.
    That module now calls this function.

    Classification predictions are recentered to `proba - proba_threshold`
    rather than left as a raw positive-class probability. It makes
    `method="sign"` mean the same thing for every task: positive score =
    the model is bullish. A raw probability is in [0, 1], so its sign is +1
    for every name and every date — a "long everything" portfolio that
    looks like a signal. The threshold is a parameter, defaulting to the
    0.5 this recentring was written with, because a classifier calibrated
    to a rarer event has its decision boundary somewhere else and
    subtracting 0.5 from it would encode a different claim than the model
    makes.

    `task=None` passes the predictions through unchanged and is a
    deliberate option, not a missing argument: a caller holding a score of
    unstated provenance is better served by a panel of exactly what it
    holds than by an offset picked on its behalf. A task that IS given must
    be a real one.

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
    if task is not None and task not in TASKS:
        raise ValidationError(f"task must be one of {list(TASKS)}, got {task!r}.")
    _validate_predictions_frame(predictions_df, source)
    panel = predictions_df.pivot(index="date", columns="entity", values="prediction")
    panel = panel.sort_index()
    panel.columns.name = None
    if task == "classification":
        panel = panel - float(proba_threshold)
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

    # BEFORE WEIGHTING, which is what the field says: the scores are
    # adjusted and the chosen method then weights them. This returned
    # `vol_scaled(...)` instead, which does its own gross-leverage
    # normalization and so never ran the method at all -- a
    # cross_sectional_zscore spec lost its z-scoring, a rank spec its
    # ranking, and nothing reported the substitution.
    #
    # The two membership methods above return before this on purpose and
    # the field documents it: scaling a score cannot change an equal
    # weight. Adjusting earlier would reorder names and so change quantile
    # MEMBERSHIP, which is not what the flag claims to do.
    if spec.volatility_scale:
        if returns_df is None:
            raise ValidationError(
                "volatility_scale=True requires returns; none were supplied."
            )
        scores = vol_adjusted_scores(
            scores,
            returns_df=returns_df,
            lookback=spec.volatility_lookback,
        )
    if spec.method == "uncertainty_scaled":
        # The scores reaching here are already prediction / interval width
        # (see `scale_by_uncertainty`); what remains is to size them like
        # any magnitude-carrying score, standardized within the date.
        return zscore_normalized(scores, gross_leverage=1.0)
    return _SIZERS[spec.method](scores, gross_leverage=1.0)


def scale_by_uncertainty(predictions_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    The prediction divided by the width of its conformal interval.

    A forecast of +2% with a +/-1% band and one of +2% with a +/-10% band
    are not the same view, and every other transform here sizes them the
    same. Dividing by the width -- the model's own statement of how sure
    it is, read off the interval it was validated with -- is where the
    portfolio layer starts consuming the distribution. A model without
    intervals is refused by name rather than sized as though every band
    were equal, which is what silently falling back would mean.
    """
    missing = [c for c in ("lower", "upper") if c not in predictions_df.columns]
    if missing:
        raise ValidationError(
            f"transform.method='uncertainty_scaled' needs an interval per "
            f"prediction, and {source} carries no {missing} column(s). Train "
            "the model with ModelSpec.intervals (a split-conformal band) and "
            "evaluate that one, or use a transform that sizes on the point "
            "prediction alone."
        )
    width = (predictions_df["upper"] - predictions_df["lower"]).to_numpy(dtype=float)
    if not np.all(np.isfinite(width)) or (width <= 0).any():
        raise ValidationError(
            f"transform.method='uncertainty_scaled': {source} carries an interval "
            "of zero or negative width, which is not a statement of uncertainty."
        )
    out = predictions_df.copy()
    out["prediction"] = out["prediction"].to_numpy(dtype=float) / width
    return out


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
    if spec.method == "mean_variance":
        # Not a per-date function of the scores: the trade depends on the
        # book already held, the IC on realized returns, the risk on prices.
        raise ValidationError(
            "transform.method='mean_variance' needs prices, the previous book "
            "and the simulator's cost rates, which this function does not "
            "take. Call mean_variance_weights(score_panel, rebalance_dates, "
            "closes, spec, linear_cost=..., periods_per_year=...), or evaluate "
            "through evaluate_model_portfolio / evaluate_predictions_portfolio, "
            "which supply them."
        )
    if score_panel.empty:
        raise ValidationError("score panel is empty — nothing to transform")

    availability = score_panel.notna()
    mask = availability.to_numpy()
    values = score_panel.to_numpy(dtype=float)
    columns = score_panel.columns
    # Filled positionally and returned as the matrix the exposure step
    # already wanted -- the intermediate DataFrame existed only to be
    # converted back, and its label-based `.loc` take and assign were two of
    # the three costs here.
    raw_matrix = np.zeros(mask.shape, dtype=float)

    # Group dates by their availability PATTERN so each sizing call sees a
    # dense sub-panel. In the common case every date has the same entities
    # and this is a single group; the count approaches n_dates as soon as
    # names enter and leave the universe, which is why this loop is worth
    # keeping cheap.
    #
    # The key is the packed bit-pattern viewed as an opaque record, which
    # `factorize` groups in one pass. It replaces a per-ROW Python string
    # build (`availability.apply(..., axis=1)`); on a 500x1000 panel with
    # staggered listing dates that build, the scalar `.loc[date, col]` per
    # column per group below it, and the label-based takes were together
    # most of a 57-second call.
    packed = np.ascontiguousarray(np.packbits(mask, axis=1))
    keys = packed.view([("", packed.dtype)] * packed.shape[1]).ravel()
    codes = pd.factorize(keys)[0]

    for code in range(int(codes.max()) + 1 if codes.size else 0):
        rows = np.flatnonzero(codes == code)
        present_positions = np.flatnonzero(mask[rows[0]])
        if present_positions.size < 2:
            # A one-name "cross-section" has no ordering to exploit; every
            # cross-sectional method would either divide by a zero spread
            # or hand 100% to that single name on the strength of no
            # comparison at all. Left flat, and counted in diagnostics.
            continue
        present = columns[present_positions]
        sub_scores = pd.DataFrame(
            values[np.ix_(rows, present_positions)],
            index=score_panel.index[rows],
            columns=present,
        )
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
        raw_matrix[np.ix_(rows, present_positions)] = raw.to_numpy(dtype=float)

    # Re-split every row into books to hit gross AND net exactly, and cap.
    # Done here rather than inside the group loop so the exposure contract
    # is enforced in exactly one place for every method.
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

    # TURNOVER DAMPING, after the books are built and before anything is
    # measured off them. A blend toward the previous row: every name moves
    # the same fraction of its own distance, so the target's ordering
    # survives exactly and what is given up is the speed of adjustment.
    #
    # It is applied here, sequentially, because each row's starting point
    # is the row before it AFTER damping -- not the undamped target, which
    # would let the book drift arbitrarily far from where it actually is.
    n_damped = 0
    realized_turnover: List[float] = []
    if spec.max_turnover is not None:
        cap = float(spec.max_turnover)
        previous = np.zeros(final.shape[1])
        for i in range(final.shape[0]):
            target = np.nan_to_num(final[i])
            move = float(np.abs(target - previous).sum())
            if i > 0 and move > cap > 0.0:
                final[i] = previous + (cap / move) * (target - previous)
                n_damped += 1
                move = cap
            else:
                final[i] = target
            realized_turnover.append(move)
            previous = final[i]

    out = pd.DataFrame(final, index=score_panel.index, columns=score_panel.columns)

    if spec.max_turnover is not None:
        # The books moved, so the exposures must be re-measured from the
        # rows that will actually be traded rather than from the targets
        # they were damped away from.
        damped = out.to_numpy(dtype=float)
        per_date = [
            {
                **diag,
                "realized_gross": float(np.abs(row).sum()),
                "realized_net": float(row.sum()),
            }
            for diag, row in zip(per_date, damped)
        ]

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
        # What the damping did, when it was asked for. None is not zero:
        # a run that damped nothing and a run that could not damp are
        # different, and only one of them was asked.
        "max_turnover": spec.max_turnover,
        "n_dates_damped": n_damped if spec.max_turnover is not None else None,
        "mean_rebalance_turnover": (
            float(np.mean(realized_turnover)) if realized_turnover else None
        ),
    }

    # THE SHORTFALL HAS TO SAY SOMETHING. `n_dates_below_target_gross` was
    # counted and then left sitting in the dict, so a caller who asked for
    # gross 1.0 / net 0.0 and received a 100%-LONG book at gross 0.5 had
    # to notice a number nothing pointed at.
    #
    # It happens whenever the sizer does not centre its scores and the
    # scores are one-sided. `method='sign'` is the plain case: all-positive
    # predictions are all +1, so there is no short book at all and
    # `apply_exposure_targets` can only fill the long half. Measured on 40
    # names x 60 dates of positive predictions, through the uncentred
    # divide-by-volatility sizer this transform used to call: 60 of 60
    # dates short of target, mean net +0.34 against a requested 0.0, and
    # `warnings` was None. That route is gone -- volatility_scale now
    # composes with the method, and both cross-sectional methods recentre
    # -- but the shortfall is a property of any uncentred one-sided book,
    # which is why it is counted here and not at one sizer.
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


# ── Predictions into the exact optimizer ────────────────────────────────
#
# The five transforms above size a name from its own score, one date at a
# time. Two names that move together and score alike each get a full
# position, nothing holds a volatility, and every rebalance pays to reach
# its target however little the view moved. `method='mean_variance'`
# replaces the sizing rule with a decision: the date's predictions become
# expected returns, a trailing covariance is the risk model, and
# `optimize_rebalance` chooses the trade from the book already held, net of
# the costs the simulation will charge for it.

#: Rebalances a year under each schedule `select_rebalance_dates` builds
#: other than daily (which rebalances every bar). With the interval's bars
#: a year this gives the holding period in bars, from the configuration
#: alone, so it is the same number at every date of a run.
_REBALANCES_PER_YEAR = {"weekly": 52.0, "monthly": 12.0}

#: The risk aversion `sizing='target_volatility'` starts from before the
#: volatility cap's search raises it: 1, the growth-optimal (full Kelly)
#: investor's. Below it a book takes more risk than maximizes growth, so an
#: alpha too weak to justify the target's risk is held at full Kelly rather
#: than levered up to the target; any alpha that does justify it reaches
#: the cap, where the book is the most expected return at that risk.
_TARGET_VOLATILITY_RISK_AVERSION = 1.0

#: Fewest names a date's cross-section needs for a rank correlation, and
#: for a book to allocate across.
_MIN_IC_NAMES = 3

#: The largest certificate residual (the optimizer's own, relative) an
#: uncertified answer may carry and still be traded. The certificate's own
#: tolerance is 1e-12; a miss below this is rounding in an answer that is
#: the optimum, and one above it is a point the method stopped at, not an
#: optimum of anything.
_NEAR_CERTIFIED = 1e-6


def holding_period_bars(frequency: str, periods_per_year: float) -> float:
    """
    Bars between two rebalances under a schedule: 1 for daily, and the
    interval's bars a year over 52 or 12 rebalances for weekly or monthly.

    Read from the configuration, not measured off the schedule, so a
    weight dated t never depends on how many dates came after it.
    """
    if frequency == "daily":
        return 1.0
    per_year = _REBALANCES_PER_YEAR.get(frequency)
    if per_year is None:
        raise ValidationError(
            f"rebalance_frequency must be one of daily/weekly/monthly, got "
            f"{frequency!r}"
        )
    if not (math.isfinite(periods_per_year) and periods_per_year > 0):
        raise ValidationError(
            f"periods_per_year={periods_per_year!r} must be finite and positive."
        )
    return max(1.0, float(periods_per_year) / per_year)


def cross_sectional_z(values: np.ndarray, score: str) -> np.ndarray:
    """
    One date's predictions as z-scores across the names that have one:
    mean 0 and standard deviation 1 over the finite entries, NaN elsewhere.

    'zscore' standardizes the prediction itself, so its magnitude carries.
    'rank_normal' sends the ranks through the inverse normal and then
    standardizes, so a single outlier is one rank and no more: a model
    whose scale is wrong on a few names cannot hand those names the book.
    A cross-section with no spread (every finite value equal) has no view
    and scores 0 throughout.
    """
    from scipy.special import ndtri
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return out
    present = values[finite]
    if score == "rank_normal":
        present = ndtri((rankdata(present) - 0.5) / n)
    elif score != "zscore":
        raise ValidationError(f"score must be 'rank_normal' or 'zscore', got {score!r}")
    centred = present - present.mean()
    spread = float(np.sqrt(np.mean(centred**2)))
    out[finite] = centred / spread if spread > _EPS else 0.0
    return out


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman's rho with average ranks for ties; NaN with no spread."""
    from scipy.stats import rankdata

    a = rankdata(left)
    b = rankdata(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denominator <= 0.0:
        return math.nan
    return float(np.sum(a * b)) / denominator


def realized_rank_ic(
    score_panel: pd.DataFrame, closes: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """
    The model's realized cross-sectional rank IC, one row per prediction
    date s: Spearman's rho between the predictions dated s and the
    close-to-close return from s over the next `horizon` bars of `closes`.

    Each row carries `known_at`, the calendar position at which its forward
    window closed (s's position + horizon). An IC is usable at a rebalance
    only once that position has been reached, which is what keeps the
    trailing estimate point in time: the OOS predictions carry no labels,
    and the returns they are scored against are read off the same prices
    the simulation runs on.

    A prediction date off the price calendar, a window that runs past its
    end, or fewer than three names with both a prediction and a return
    yields no row.
    """
    if horizon < 1:
        raise ValidationError(f"horizon must be at least one bar, got {horizon}.")
    missing = [c for c in score_panel.columns if c not in closes.columns]
    if missing:
        raise ValidationError(
            f"closes have no column for {missing}; the realized IC needs a price "
            "for every name the predictions score."
        )
    calendar = pd.DatetimeIndex(closes.index)
    positions = calendar.get_indexer(pd.DatetimeIndex(score_panel.index))
    prices = closes[list(score_panel.columns)].to_numpy(dtype=float)
    predictions = score_panel.to_numpy(dtype=float)
    rows: List[Tuple[Any, int, int, float, int]] = []
    for k, position in enumerate(positions):
        if position < 0 or position + horizon >= len(calendar):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            forward = prices[position + horizon] / prices[position] - 1.0
        usable = np.isfinite(forward) & np.isfinite(predictions[k])
        if int(usable.sum()) < _MIN_IC_NAMES:
            continue
        ic = _rank_correlation(predictions[k][usable], forward[usable])
        if not math.isfinite(ic):
            continue
        rows.append(
            (
                score_panel.index[k],
                int(position),
                int(position + horizon),
                ic,
                int(usable.sum()),
            )
        )
    return pd.DataFrame(
        rows, columns=["date", "position", "known_at", "ic", "n_names"]
    ).astype({"position": int, "known_at": int, "ic": float, "n_names": int})


def trailing_ic(
    ic_frame: pd.DataFrame,
    position: int,
    *,
    lookback: int,
    min_dates: int,
    horizon: int,
    shrink: bool,
) -> Dict[str, Any]:
    """
    The IC a rebalance at calendar `position` may scale its alphas by.

    The mean of the last `lookback` realized ICs whose forward windows had
    closed by `position`, floored at 0 and, with `shrink`, multiplied by
    max(0, 1 - 1/t^2): a James-Stein-style pull toward no view by the
    estimate's own noise, so an IC the sample cannot tell from zero sizes
    nothing and one it measures well is barely touched.

    THE T-STATISTIC COUNTS INDEPENDENT WINDOWS, NOT DATES. Daily ICs over a
    five-bar horizon share four of their five bars with the next date's, so
    the mean of 250 of them carries about 50 observations' worth of
    information; dividing by sqrt(250) would overstate t by sqrt(5) and
    undo the shrink exactly where it matters. The effective count is the
    number of non-overlapping windows the dates span, capped at the count.

    Fewer than `min_dates` closed dates is the warm-up: no IC at all.
    """
    known = ic_frame["known_at"].to_numpy()
    available = int(np.searchsorted(known, position, side="right"))
    window = ic_frame.iloc[max(0, available - lookback) : available]
    n = int(len(window))
    out: Dict[str, Any] = {
        "ic_raw": None,
        "ic_used": 0.0,
        "ic_t_stat": None,
        "n_ic_dates": n,
        "n_ic_effective": None,
        "ic_warmup": n < min_dates,
    }
    if n < min_dates:
        return out
    values = window["ic"].to_numpy(dtype=float)
    spots = window["position"].to_numpy()
    mean = float(values.mean())
    spread = float(values.std(ddof=1)) if n > 1 else 0.0
    effective = min(n, int((int(spots[-1]) - int(spots[0])) // horizon) + 1)
    if spread > 0.0:
        t_stat = mean / (spread / math.sqrt(effective))
    else:
        t_stat = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
    used = max(mean, 0.0)
    if shrink:
        used *= max(0.0, 1.0 - 1.0 / (t_stat * t_stat)) if t_stat > 0 else 0.0
    out.update(
        ic_raw=mean,
        ic_used=float(used),
        ic_t_stat=float(t_stat) if math.isfinite(t_stat) else None,
        n_ic_effective=int(effective),
    )
    return out


def _closes_on_calendar(
    closes: pd.DataFrame, columns: Iterable[Any], what: str
) -> pd.DataFrame:
    """The closes for `columns`, refused by name when one is missing or the
    calendar is not a sorted set of dates."""
    columns = list(columns)
    missing = [c for c in columns if c not in closes.columns]
    if missing:
        raise ValidationError(
            f"{what}: closes have no column for {missing}. Every name the "
            "predictions score needs a price series."
        )
    index = pd.DatetimeIndex(closes.index)
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValidationError(
            f"{what}: the closes' dates must be unique and increasing."
        )
    return closes[columns]


def mean_variance_weights(
    score_panel: pd.DataFrame,
    rebalance_dates: Iterable[Any],
    closes: pd.DataFrame,
    spec: PredictionTransformSpec,
    *,
    linear_cost: float,
    periods_per_year: float,
    fill_price: str,
    opens: Optional[pd.DataFrame] = None,
    target_horizon: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Target weights from the exact cost-aware optimizer, one trade per
    rebalance from the book actually held: the previous rebalance's target,
    drifted by the prices since it was filled.

    At each rebalance date t, with nothing dated after t:

    - ALPHA by Grinold's rule, per holding period: alpha_i = IC * sigma_i *
      z_i * min(h, H) / sqrt(H), with z the date's cross-sectional score,
      sigma_i the name's per-bar volatility over the risk window, h the
      bars a rebalance holds and H the bars the IC was measured over. At
      h == H this is IC * (sigma_i sqrt(h)) * z_i, the textbook form; the
      factor keeps an IC measured at one horizon from being spent at
      another as if the two were the same (a five-bar IC traded daily
      earns a fifth of its five-bar alpha a day, and nothing after H).
      The IC is the caller's (`optimizer.ic`) or the model's trailing
      realized one (`trailing_ic`).
    - RISK: `estimate_covariance` over the last `risk_lookback` bar
      returns through t, scaled to the holding period.
    - COSTS: `linear_cost` per unit of weight traded (the simulator's
      commission + spread, so the optimizer pays what the simulation will
      charge) times `cost_multiplier`, plus quadratic `impact`.
    - CONSTRAINTS: the budget sum(w) = net_exposure, sum|w| <=
      gross_exposure, |w| <= max_position_weight (w >= 0 when net equals
      gross), and max_turnover as an exact cap on the trade, except the
      book's first build from cash, which no cap could otherwise allow.
    - SIZING: a volatility cap at `target_volatility` (annualized), a
      fixed risk aversion, or risk aversion 1/f for a fractional Kelly
      book.

    THE STARTING BOOK IS THE ONE HELD, NOT THE ONE LAST ASKED FOR. The
    simulator holds share counts between rebalances and resizes every
    position from equity at the next, so a starting book of the previous
    TARGET put the no-trade band somewhere the account was not: a name the
    optimizer left alone was still traded back to its old weight, paying
    for the drift and not the signal. The book held at t is the previous
    target w filled at prices F, marked at t's close C, in the simulator's
    own arithmetic: each position w_i * C_i / F_i, the cash the remainder
    1 - sum(w) less the cost the fill paid (linear_cost on its turnover,
    charged to cash as the simulator charges it), and every weight that
    position over the equity they add up to,

        w0_i = w_i g_i / (1 - cost + sum_j w_j (g_j - 1)),  g = C / F.

    Shorts drift by the same expression (a negative weight's value grows
    more negative as its price rises), so a dollar-neutral book needs no
    separate leg arithmetic. Under fill_price='close' F is the previous
    rebalance's close and the held book is the simulator's to rounding, so
    a name inside its no-trade band comes back at exactly the weight the
    account holds and the simulator trades none of it. Under 'next_open'
    F is the open after the previous rebalance, and the trade fills at the
    open after t, which is not known at t's close: the book is marked to
    t's close, the last price a decision at t can read, and the simulator
    trades the overnight move from that close to the fill on every name,
    the untouched ones included. Financing (cash credit, margin, borrow)
    is not marked, so under a financing spec the held book is off by the
    interest accrued since the last rebalance.

    The first rebalance starts from cash. A name with no prediction at t,
    or no complete return window, is not in the problem and is sold to
    zero. A date with fewer than two such names, or before `risk_lookback`
    bars of returns exist, holds the book as it stands.

    `closes` is the close price per date and name on one calendar (the
    simulation's master calendar), covering the score panel's names and
    every rebalance date; `opens` the open on the same calendar, required
    under fill_price='next_open'. `linear_cost` is the simulator's own
    commission + spread rate, which is both what the optimizer is charged
    (times `cost_multiplier`) and what marks the held book. Returns
    (weights, diagnostics, per_date): the weights indexed by the rebalance
    dates with one column per name, a JSON-safe summary, and one row of
    what each date's solve saw.
    """
    if spec.method != "mean_variance":
        raise ValidationError(
            f"mean_variance_weights needs transform.method='mean_variance', got "
            f"{spec.method!r}; transform_predictions_to_weights sizes the others."
        )
    if score_panel.empty:
        raise ValidationError("score panel is empty — nothing to transform")
    if not (math.isfinite(linear_cost) and linear_cost >= 0.0):
        raise ValidationError(
            f"linear_cost={linear_cost!r} must be a finite, non-negative "
            "decimal per unit of weight traded (0.0015 is 15 bps)."
        )
    if fill_price not in ("close", "next_open"):
        raise ValidationError(
            f"fill_price must be 'close' or 'next_open', got {fill_price!r}: the "
            "convention the book will be simulated under, which sets the "
            "price the previous rebalance bought at."
        )
    optimizer = spec.optimizer or MeanVarianceSpec()
    holding = holding_period_bars(spec.rebalance_frequency, periods_per_year)

    if optimizer.ic_horizon is not None:
        horizon, horizon_source = int(optimizer.ic_horizon), "optimizer.ic_horizon"
    elif target_horizon is not None and int(target_horizon) >= 1:
        horizon, horizon_source = int(target_horizon), "target_horizon"
    else:
        horizon, horizon_source = max(1, int(round(holding))), "rebalance_interval"

    names = list(score_panel.columns)
    prices = _closes_on_calendar(closes, names, "mean_variance_weights")
    calendar = pd.DatetimeIndex(prices.index)
    schedule = pd.DatetimeIndex(rebalance_dates)
    rebalance_positions = calendar.get_indexer(schedule)
    if (rebalance_positions < 0).any():
        absent = [str(d.date()) for d in schedule[rebalance_positions < 0][:3]]
        raise ValidationError(
            f"rebalance date(s) {absent} are not in the closes' calendar; a "
            "trade dated t is priced from t's close."
        )
    if not schedule.isin(pd.DatetimeIndex(score_panel.index)).all():
        raise ValidationError(
            "every rebalance date must be a prediction date of the score panel."
        )

    price_values = prices.to_numpy(dtype=float)
    if fill_price == "next_open":
        if opens is None:
            raise ValidationError(
                "fill_price='next_open' fills each rebalance at the next bar's "
                "open, so marking the held book needs `opens` (the open per "
                "date and name on the closes' calendar). Pass it, or "
                "fill_price='close'."
            )
        open_values = (
            _closes_on_calendar(opens, names, "mean_variance_weights (opens)")
            .reindex(calendar)
            .to_numpy(dtype=float)
        )
    else:
        open_values = price_values
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_returns = np.full(price_values.shape, np.nan)
        bar_returns[1:] = price_values[1:] / price_values[:-1] - 1.0

    ic_frame = (
        realized_rank_ic(score_panel, prices, horizon) if optimizer.ic is None else None
    )
    horizon_scale = min(holding, float(horizon)) / math.sqrt(float(horizon))
    lookback = int(optimizer.risk_lookback)
    cap = float(spec.max_position_weight)
    gross = float(spec.gross_exposure)
    net = float(spec.net_exposure)
    lower = 0.0 if abs(net - gross) <= 1e-12 else -cap
    upper = 0.0 if abs(net + gross) <= 1e-12 else cap
    # A one-sided book's gross IS its net, so the budget already holds it
    # at gross_exposure; passing the cap as well would hand the optimizer
    # the same row twice.
    gross_cap: Optional[float] = None if lower == 0.0 or upper == 0.0 else gross
    kappa = float(linear_cost) * float(optimizer.cost_multiplier)
    eta = float(optimizer.impact)
    if not optimizer.cost_aware:
        kappa, eta = 0.0, 0.0
    if optimizer.sizing == "target_volatility":
        risk_aversion = _TARGET_VOLATILITY_RISK_AVERSION
        volatility_cap: Optional[float] = float(optimizer.target_volatility) * (
            math.sqrt(holding / float(periods_per_year))
        )
    elif optimizer.sizing == "kelly":
        risk_aversion, volatility_cap = 1.0 / float(optimizer.kelly_fraction), None
    else:
        risk_aversion, volatility_cap = float(optimizer.risk_aversion), None
    # Per-period volatility to annualized: sqrt(rebalances a year).
    annualize = math.sqrt(float(periods_per_year) / holding)

    from standard_quant_tools.portfolio.covariance import estimate_covariance
    from standard_quant_tools.portfolio.rebalance_optimizer import optimize_rebalance

    scores_on = score_panel.loc[schedule].to_numpy(dtype=float)
    weights = np.zeros((len(schedule), len(names)))
    # The last target, the prices it filled at, and the cost the fill
    # paid as a fraction of the equity it was sized from.
    target = np.zeros(len(names))
    filled_at = np.ones(len(names))
    fill_cost = 0.0
    records: List[Dict[str, Any]] = []
    solve_seconds = 0.0
    covariance_warnings: List[str] = []
    for row, (date, position) in enumerate(zip(schedule, rebalance_positions)):
        record: Dict[str, Any] = {
            "status": "solved",
            "n_names": 0,
            "ic_raw": None,
            "ic_used": None,
            "ic_t_stat": None,
            "n_ic_dates": None,
            "ic_warmup": False,
            "ex_ante_volatility": None,
            "turnover": 0.0,
            "share_held": None,
            "n_forced_exits": 0,
            "certified": None,
            "effective_risk_aversion": None,
            "volatility_cap_binding": False,
            "turnover_cap_binding": False,
            "shrinkage": None,
            "n_covariance_warnings": 0,
        }
        previous = _held_book(
            target, filled_at, price_values[position], fill_cost, names, date
        )
        window = (
            bar_returns[position - lookback + 1 : position + 1]
            if position >= lookback
            else None
        )
        prediction = scores_on[row]
        tradable = np.isfinite(prediction) & np.isfinite(price_values[position])
        if window is not None:
            tradable &= np.isfinite(window).all(axis=0)
        else:
            tradable[:] = False
        present = np.flatnonzero(tradable)
        exits = (~tradable) & (previous != 0.0)
        record["n_forced_exits"] = int(exits.sum())
        record["n_names"] = int(present.size)

        if present.size < 2:
            record["status"] = "no_risk_model" if window is None else "too_few_names"
            new = previous.copy()
            new[exits] = 0.0
            record["turnover"] = float(np.abs(new - previous).sum())
            weights[row] = new
            target, fill_cost = new, float(linear_cost) * record["turnover"]
            filled_at = _fill_prices(fill_price, price_values, open_values, position)
            records.append(record)
            continue

        if ic_frame is None:
            ic_used = float(optimizer.ic)
            record.update(ic_raw=ic_used, ic_used=ic_used)
        else:
            estimate = trailing_ic(
                ic_frame,
                int(position),
                lookback=int(optimizer.ic_lookback),
                min_dates=int(optimizer.ic_min_dates),
                horizon=horizon,
                shrink=bool(optimizer.ic_shrink),
            )
            ic_used = float(estimate["ic_used"])
            record.update(
                ic_raw=estimate["ic_raw"],
                ic_used=ic_used,
                ic_t_stat=estimate["ic_t_stat"],
                n_ic_dates=estimate["n_ic_dates"],
                ic_warmup=estimate["ic_warmup"],
            )

        names_now = [names[i] for i in present]
        sample = window[:, present]
        volatility = sample.std(axis=0, ddof=1)
        covariance = estimate_covariance(
            pd.DataFrame(sample, columns=names_now),
            method=optimizer.covariance,
            periods_per_year=1,
        )
        matrix = covariance["matrix"]
        record["shrinkage"] = covariance["shrinkage_intensity"]
        record["n_covariance_warnings"] = len(covariance["warnings"])
        if covariance["warnings"]:
            covariance_warnings = [
                f"{pd.Timestamp(date).date()}: {w}" for w in covariance["warnings"]
            ]
        per_bar = np.array(
            [[matrix[a][b] for b in names_now] for a in names_now], dtype=float
        )
        z = cross_sectional_z(prediction[present], optimizer.score)
        alpha = ic_used * volatility * z * horizon_scale
        start = previous[present]
        turnover_cap = spec.max_turnover if np.any(previous != 0.0) else None

        clock = time.perf_counter()
        try:
            solved = optimize_rebalance(
                pd.DataFrame(per_bar * holding, index=names_now, columns=names_now),
                expected_returns=alpha,
                current_weights=start,
                risk_aversion=risk_aversion,
                linear_cost=kappa,
                quadratic_cost=eta,
                lower_bounds=lower,
                upper_bounds=upper,
                budget=net,
                budget_type="equal",
                max_turnover=turnover_cap,
                max_gross=gross_cap,
                max_volatility=volatility_cap,
            )
        except ValidationError as exc:
            raise ValidationError(
                f"method='mean_variance' could not solve the rebalance on "
                f"{pd.Timestamp(date).date()} ({present.size} names): {exc} "
                + (
                    f"A target_volatility of {optimizer.target_volatility} may "
                    "sit below the least volatile book the budget allows (a "
                    "long-only book carries the market's volatility): raise "
                    "it, or use sizing='risk_aversion'."
                    if volatility_cap is not None
                    else ""
                )
            ) from exc
        finally:
            solve_seconds += time.perf_counter() - clock

        # AN ANSWER THAT IS NOT AN OPTIMUM IS REFUSED; ONE THAT MISSES ITS
        # CERTIFICATE BY ROUNDING IS TRADED AND COUNTED. A solve that
        # reached no point returns no weights, and one that stopped short
        # (a singular system, a cycle) returns the last feasible point it
        # held: a book the constraints allow but no rule chose. Simulating
        # either would report a rule nobody ran, so the date is named and
        # the run refused. Holding the previous book instead would be the
        # same fiction (on the first date it is cash, which a long-only
        # budget forbids). An answer whose residuals are rounding above the
        # certificate's tolerance is the optimum to that rounding.
        certificate = solved.get("certificate") or {}
        residual = max(
            (
                float(value)
                for key, value in certificate.items()
                if key != "tolerance" and value is not None
            ),
            default=math.inf,
        )
        if not solved.get("weights") or (
            not solved["certified"] and not residual <= _NEAR_CERTIFIED
        ):
            reason = (solved.get("warnings") or [str(solved.get("method"))])[-1]
            raise ValidationError(
                f"method='mean_variance': the optimizer returned no optimum on "
                f"{pd.Timestamp(date).date()} ({present.size} names, largest "
                f"certificate residual {residual:.3g}): {reason} The run is "
                "refused rather than simulated on a book no rule chose. "
                "Loosen the limits that bind there (max_turnover, "
                "max_position_weight, gross_exposure, target_volatility), or "
                "evaluate with another method."
            )
        chosen = np.array([solved["weights"][a] for a in names_now], dtype=float)
        new = np.zeros(len(names))
        new[present] = chosen
        kinds = {str(c.get("kind")) for c in solved.get("binding_constraints") or []}
        effective = solved.get("effective_risk_aversion")
        record.update(
            ex_ante_volatility=float(solved["volatility"]) * annualize,
            turnover=float(np.abs(new - previous).sum()),
            share_held=len(solved.get("no_trade") or []) / float(present.size),
            certified=bool(solved["certified"]),
            effective_risk_aversion=None if effective is None else float(effective),
            volatility_cap_binding=any("volatil" in k for k in kinds),
            turnover_cap_binding=any("turnover" in k for k in kinds),
        )
        weights[row] = new
        target, fill_cost = new, float(linear_cost) * record["turnover"]
        filled_at = _fill_prices(fill_price, price_values, open_values, position)
        records.append(record)

    index = score_panel.loc[schedule].rename_axis(index=None).index
    out = pd.DataFrame(weights, index=index, columns=score_panel.columns)
    per_date = pd.DataFrame(records, index=index)
    diagnostics = _mean_variance_diagnostics(
        out,
        per_date,
        spec,
        optimizer,
        horizon=horizon,
        horizon_source=horizon_source,
        holding=holding,
        linear_cost=kappa,
        solve_seconds=solve_seconds,
        covariance_warnings=covariance_warnings,
    )
    return out, diagnostics, per_date


def _fill_prices(
    fill_price: str, closes: np.ndarray, opens: np.ndarray, position: int
) -> np.ndarray:
    """The prices a rebalance decided at calendar `position` fills at, as
    the simulator fills it: that bar's close, or the next bar's open. A
    decision on the last bar has no next open and no later rebalance to
    mark it, so its row is all NaN."""
    if fill_price == "close":
        return closes[position]
    if position + 1 < len(opens):
        return opens[position + 1]
    return np.full(closes.shape[1], np.nan)


def _held_book(
    target: np.ndarray,
    filled_at: np.ndarray,
    marks: np.ndarray,
    fill_cost: float,
    names: List[Any],
    date: Any,
) -> np.ndarray:
    """
    The previous target as weights of the equity it has become: each
    position grown by mark / fill price, the cash the remainder less the
    fill's cost, everything over the new equity. All cash stays all cash.
    """
    held = target != 0.0
    if not held.any():
        return np.zeros_like(target)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(held, marks / filled_at, 1.0)
    unmarked = held & ~(np.isfinite(growth) & (growth > 0.0))
    if unmarked.any():
        missing = [names[i] for i in np.flatnonzero(unmarked)][:5]
        raise ValidationError(
            f"mean_variance_weights: on {pd.Timestamp(date).date()} the held "
            f"book cannot be marked for {missing}: a fill or close price there "
            "is missing or not positive. Every name held needs a price at "
            "its fill and at each later rebalance."
        )
    equity = 1.0 - fill_cost + float(np.sum(target * (growth - 1.0)))
    if not equity > 0.0:
        raise ValidationError(
            f"mean_variance_weights: on {pd.Timestamp(date).date()} the held "
            f"book's equity is {equity:.4g} of what it was at the last "
            "rebalance -- insolvent, as the simulator would refuse it."
        )
    return np.where(held, target * growth / equity, 0.0)


def _mean_or_none(values: pd.Series) -> Optional[float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _mean_variance_diagnostics(
    weights: pd.DataFrame,
    per_date: pd.DataFrame,
    spec: PredictionTransformSpec,
    optimizer: MeanVarianceSpec,
    *,
    horizon: int,
    horizon_source: str,
    holding: float,
    linear_cost: float,
    solve_seconds: float,
    covariance_warnings: List[str],
) -> Dict[str, Any]:
    """The run's summary: the keys every transform reports, read off the
    weights, plus what the optimizer saw and did under `mean_variance`."""
    values = weights.to_numpy(dtype=float)
    solved = per_date["status"] == "solved"
    no_alpha = solved & (pd.to_numeric(per_date["ic_used"], errors="coerce") <= 0.0)
    warmup = solved & per_date["ic_warmup"].astype(bool)
    uncertified = solved & (per_date["certified"] == False)  # noqa: E712
    gross = np.abs(values).sum(axis=1)
    summary: Dict[str, Any] = {
        "sizing": optimizer.sizing,
        "score": optimizer.score,
        "ic_source": "given" if optimizer.ic is not None else "trailing_realized",
        "ic_horizon": int(horizon),
        "ic_horizon_source": horizon_source,
        "holding_period_bars": float(holding),
        "linear_cost": float(linear_cost),
        "n_dates_solved": int(solved.sum()),
        "n_dates_without_risk_model": int(
            (per_date["status"] == "no_risk_model").sum()
        ),
        "n_dates_too_few_names": int((per_date["status"] == "too_few_names").sum()),
        "n_dates_without_alpha": int(no_alpha.sum()),
        "n_dates_ic_warmup": int(warmup.sum()),
        "mean_ic_raw": _mean_or_none(per_date.loc[solved, "ic_raw"]),
        "mean_ic_used": _mean_or_none(per_date.loc[solved, "ic_used"]),
        "mean_ex_ante_volatility": _mean_or_none(
            per_date.loc[solved, "ex_ante_volatility"]
        ),
        "target_volatility": (
            float(optimizer.target_volatility)
            if optimizer.sizing == "target_volatility"
            else None
        ),
        "n_dates_volatility_cap_binding": int(
            per_date.loc[solved, "volatility_cap_binding"].astype(bool).sum()
        ),
        "n_dates_turnover_cap_binding": int(
            per_date.loc[solved, "turnover_cap_binding"].astype(bool).sum()
        ),
        "mean_turnover": _mean_or_none(per_date["turnover"]),
        "mean_share_held_at_previous": _mean_or_none(
            per_date.loc[solved, "share_held"]
        ),
        "n_forced_exits": int(per_date["n_forced_exits"].sum()),
        "mean_effective_risk_aversion": _mean_or_none(
            per_date.loc[solved, "effective_risk_aversion"]
        ),
        "mean_shrinkage_intensity": _mean_or_none(per_date.loc[solved, "shrinkage"]),
        "n_dates_covariance_warned": int(
            (per_date.loc[solved, "n_covariance_warnings"] > 0).sum()
        ),
        "n_uncertified": int(uncertified.sum()),
        # Wall-clock seconds in the solver: the one number here that is not
        # a function of the inputs.
        "solve_seconds": float(solve_seconds),
    }
    notes: List[str] = []
    if summary["n_dates_without_risk_model"]:
        notes.append(
            f"{summary['n_dates_without_risk_model']} rebalance date(s) came "
            f"before {optimizer.risk_lookback} bars of returns existed and held "
            "the previous book (cash at the start). Lower optimizer."
            "risk_lookback or fetch a longer price history to trade them."
        )
    if summary["n_dates_too_few_names"]:
        notes.append(
            f"{summary['n_dates_too_few_names']} rebalance date(s) had fewer "
            "than two names with a prediction and a complete return window, "
            "and held the previous book."
        )
    if summary["n_dates_without_alpha"]:
        notes.append(
            f"{summary['n_dates_without_alpha']} of {int(solved.sum())} solved "
            f"date(s) had no alpha ({summary['n_dates_ic_warmup']} before "
            f"{optimizer.ic_min_dates} realized ICs had closed, the rest with "
            "a trailing IC at or below zero"
            + (" or shrunk to zero by its t-statistic" if optimizer.ic_shrink else "")
            + "): on those the book only managed risk and costs."
        )
    if summary["n_dates_covariance_warned"]:
        notes.append(
            f"The covariance estimate warned on "
            f"{summary['n_dates_covariance_warned']} solved date(s); the last: "
            + " ".join(covariance_warnings)
        )
    if summary["n_uncertified"]:
        notes.append(
            f"{summary['n_uncertified']} solve(s) missed the optimizer's "
            "certificate tolerance by rounding (every residual under "
            f"{_NEAR_CERTIFIED:g}); their weights were traded."
        )
    if optimizer.ic is not None:
        notes.append(
            f"optimizer.ic={optimizer.ic} is the caller's number and was used "
            "on every date. An IC measured over these same out-of-sample dates "
            "is hindsight, and the result is then a hindsight result."
        )
    if not optimizer.cost_aware:
        notes.append(
            "optimizer.cost_aware=False: the optimizer was charged no costs, "
            "while the simulation charges them. A comparison, not a book to run."
        )
    return {
        "n_dates": int(values.shape[0]),
        "n_entities": int(values.shape[1]),
        "mean_names_per_date": float(per_date["n_names"].mean()),
        "min_names_per_date": int(per_date["n_names"].min()),
        "mean_n_long": float((values > _EPS).sum(axis=1).mean()),
        "mean_n_short": float((values < -_EPS).sum(axis=1).mean()),
        "mean_realized_gross": float(gross.mean()),
        "mean_realized_net": float(values.sum(axis=1).mean()),
        # gross_exposure is a ceiling under this method, not a target, so a
        # book below it is the optimizer's choice and not a shortfall.
        "n_dates_below_target_gross": None,
        "n_dates_with_no_position": int((gross <= _EPS).sum()),
        "max_abs_weight": float(np.abs(values).max()) if values.size else 0.0,
        "max_turnover": spec.max_turnover,
        "n_dates_damped": None,
        "mean_rebalance_turnover": summary["mean_turnover"],
        "mean_variance": summary,
        "warnings": notes,
    }


# ── Full evaluation ─────────────────────────────────────────────────────


def _summarize_simulation(
    result: Dict[str, Any],
    periods_per_year: int,
    risk_free_rate: float,
    commission_pct: float,
    slippage_pct: float,
    day_count: str = DEFAULT_DAY_COUNT,
    *,
    initial_capital: float,
) -> Dict[str, float]:
    equity = result["equity_curve"].dropna()
    if equity.empty:
        raise ValidationError("simulation produced an empty equity curve")
    if not (math.isfinite(initial_capital) and initial_capital > 0):
        raise ValidationError(
            f"_summarize_simulation: initial_capital={initial_capital!r}. It is "
            "the level every return and drawdown here is measured from, so "
            "it must be finite and positive: the capital the simulation "
            "opened with."
        )
    # The account opened at initial_capital on the first bar, and the
    # curve's first point is that bar AFTER a rebalance on it, net of its
    # costs: measured from there, a book that paid 10 bps to put itself on
    # read 10 bps richer, and its drawdown started below the peak it fell
    # from. Growth is read from the capital, as run_portfolio_simulation's
    # `_metrics_with_day0_cost` reads it, over the same elapsed bars as
    # before: the capital and the opening fill are one bar, so the opening
    # point is replaced, not prepended. The drawdown path keeps both, the
    # capital first, so an opening cost is a fall from the peak. Where the
    # first bar trades nothing, its equity IS the capital, and all four
    # numbers are the curve's own to the bit. The per-bar returns keep the
    # curve's bars too: their benchmark accrues from the second bar on, as
    # the engine credits cash.
    capital = float(initial_capital)
    from_capital = pd.Series(
        np.concatenate(([capital], equity.to_numpy(dtype=float)[1:]))
    )
    path = pd.Series(np.concatenate(([capital], equity.to_numpy(dtype=float))))
    growth = float(cagr(from_capital, periods_per_year))
    drawdown = float(max_drawdown(path))
    # calmar_ratio's rule, on the two series above.
    if drawdown == 0.0:
        calmar = math.inf if growth > 0 else math.nan
    else:
        calmar = growth / abs(drawdown)
    returns = equity.pct_change(fill_method=None).dropna()
    # The excess is over the rate as the simulation's cash accrued it: the
    # same rate, day count and calendar days, so a book sitting in cash
    # scores zero rather than the rate it was charged and never credited.
    # Built on the equity's own index, so each return is benchmarked on the
    # days that passed before it.
    sharpe, sortino = (0.0, 0.0)
    if not returns.empty:
        # Under a financing spec the rate is its benchmark as each bar
        # accrued it, which the simulation returns.
        spec_benchmark = result.get("benchmark_accrual")
        benchmark = (
            accrued_rate_per_bar(equity.index, risk_free_rate, day_count)
            if risk_free_rate != 0.0 and spec_benchmark is None
            else None
        )
        sharpe, sortino = (
            ratios_over_benchmark(returns, spec_benchmark, periods_per_year)
            if spec_benchmark is not None
            else ratios_over_accrued_rate(
                returns,
                risk_free_rate,
                day_count,
                periods_per_year,
                benchmark=benchmark,
            )
        )

    rebalance_log = result["rebalance_log"]
    turnover = (
        rebalance_log["turnover_pct"].astype(float)
        if not rebalance_log.empty
        else pd.Series(dtype=float)
    )
    years = max(len(equity) / periods_per_year, 1e-9)

    metrics: Dict[str, float] = {
        "cumulative_return": float(cumulative_return(from_capital)),
        "cagr": growth,
        "annualized_volatility": (
            float(annualized_volatility(returns, periods_per_year))
            if not returns.empty
            else 0.0
        ),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": drawdown,
        "calmar_ratio": float(calmar),
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
        # cost drag, not the whole of it; the financing the simulator does
        # total is in provenance["financing"].
        "estimated_cost_drag_pct": (
            float(turnover.sum() * (commission_pct + slippage_pct))
            if not turnover.empty
            else 0.0
        ),
    }
    return metrics


def _spec_financing(portfolio: PortfolioSimSpec) -> Dict[str, Any]:
    """
    The simulator's financing terms from a spec, resolved: cash earns the
    risk-free rate unless cash_rate says otherwise, and negative cash pays
    it plus margin_spread_bps unless margin_interest_rate says otherwise.

    The Sharpe this evaluation reports subtracts that rate, and the cash it
    charged it on earned nothing, so a model whose book sat partly in cash
    scored below the same model fully invested. A rate past 100% a year
    cannot be a cash rate, so one left to default to such a risk-free rate
    is refused under that name.
    """
    if portfolio.financing is not None:
        # The full terms, refused beside any scalar term set away from its
        # default, with a dated benchmark's keys read as every dated input
        # on the tool surface reads them.
        from standard_quant_tools.agent.runtimes._financing import (
            account_financing_kwargs,
        )

        return account_financing_kwargs(portfolio, "evaluate_model_portfolio")
    rf = float(portfolio.risk_free_rate)
    if portfolio.cash_rate is None and not (math.isfinite(rf) and abs(rf) <= 1.0):
        raise ValidationError(
            f"risk_free_rate={portfolio.risk_free_rate!r} is the default "
            "cash_rate, and a cash rate must be a finite decimal no larger "
            "than 1 (100%) in magnitude. Pass 0.05 for 5%, or set cash_rate."
        )
    rebate = portfolio.short_rebate_rate
    return {
        "cash_rate": rf if portfolio.cash_rate is None else float(portfolio.cash_rate),
        "margin_interest_rate": resolve_debit_rate(
            portfolio.margin_interest_rate, rf, portfolio.margin_spread_bps
        ),
        "borrow_fee_bps": float(portfolio.borrow_fee_bps),
        "short_proceeds": portfolio.short_proceeds,
        "short_rebate_rate": 0.0 if rebate is None else float(rebate),
        "day_count": portfolio.day_count,
    }


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
    _assert_spec_limits(transform, portfolio)

    manifest = load_manifest(model_id)
    _refuse_cpcv(manifest, "evaluate_model_portfolio")
    dataset_spec = load_dataset_spec(model_id)
    interval = str(dataset_spec.get("interval", "1d"))
    provider_name = str(dataset_spec.get("provider", "yfinance"))

    # The calendar the dataset was built under, so an intraday model's
    # Sharpe is annualized by the bars its venue actually has.
    calendar = dataset_spec.get("calendar")

    # Verified before loading, for the same reason bridge.py verifies:
    # structural validation passes on an edited file that kept its shape,
    # so an altered prediction column would otherwise produce a clean and
    # entirely fictional equity curve.
    #
    # Resolved through the registry rather than opened as a path: the
    # manifest names this artifact by its filename in the model's own
    # directory, so a package pulled into another runs root reads its own
    # predictions instead of the registering machine's.
    predictions_uri = str(
        resolve_model_artifact(model_id, str(manifest.oos_predictions_uri))
    )
    _artifacts.verify_file(
        Path(predictions_uri),
        manifest.content_hashes.get("oos_predictions"),
        "oos_predictions",
    )
    predictions_df = _artifacts.load_artifact(predictions_uri)
    skipped_folds = (manifest.validation_report or {}).get("skipped_folds") or None
    # The horizon the model was trained to forecast, which is the horizon
    # its realized IC is measured over under method='mean_variance'.
    target_horizon = (manifest.validation_report or {}).get("target_horizon")

    result = _simulate_predictions_portfolio(
        predictions_df,
        manifest.task,
        interval=interval,
        provider_name=provider_name,
        calendar=str(calendar) if calendar else None,
        start=str(dataset_spec["start"]),
        end=str(dataset_spec["end"]),
        transform=transform,
        portfolio=portfolio,
        run_id=model_id,
        source=predictions_uri,
        skipped_folds=skipped_folds,
        extra_warnings=manifest.dataset_warnings,
        target_horizon=target_horizon,
    )
    # Every input that determined the numbers above, so a reported Sharpe
    # can be traced back to the exact predictions and weights that produced
    # it. The predictions hash is the manifest's own recorded digest
    # (already verified above), so all three identify bytes on disk rather
    # than a path that may have been rewritten.
    result["provenance"] = {
        "oos_predictions_uri": predictions_uri,
        "oos_predictions_hash": manifest.content_hashes.get("oos_predictions", ""),
        "dataset_id": manifest.dataset_id,
        "dataset_spec_hash": manifest.dataset_spec_hash,
        "estimator_type": manifest.estimator_type,
        **result["provenance"],
    }
    return {"model_id": model_id, **result}


def evaluate_predictions_portfolio(
    predictions_df: pd.DataFrame,
    task: Optional[str],
    *,
    interval: str,
    provider_name: str,
    calendar: Optional[str],
    start: str,
    end: str,
    transform: Optional[PredictionTransformSpec] = None,
    portfolio: Optional[PortfolioSimSpec] = None,
    run_id: str,
    source: str = "<predictions>",
    target_horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """
    The same evaluation for predictions that are not a registered model's:
    an ensemble, an externally computed alpha, a converted panel.

    What the model path reads from a manifest, this one is told: the task,
    and the five dataset fields (interval, provider, calendar, start, end)
    that say which bars to price the book against and how to annualize it.
    Nothing else in the simulation ever looked at the manifest.

    There is no content-hash check here because there is no registered
    digest to check against — the caller's store is the root of trust, and
    the reference it resolved is recorded in the result's provenance
    instead. That is a weaker claim than `evaluate_model_portfolio`'s and
    is spelled differently on purpose.

    `target_horizon` is the bars the predictions forecast, when the caller
    knows it (a dataset's target): method='mean_variance' measures the
    realized IC over it, and over the rebalance interval without it.

    Returns the same dict shape, with `source_ref` in place of `model_id`.
    """
    transform = transform or PredictionTransformSpec()
    portfolio = portfolio or PortfolioSimSpec()

    # A combinatorial-purged frame carries one prediction per (entity,
    # date, PATH), so the duplicate check below would reject it with a
    # message about overwriting rows — true, but not the remedy. The model
    # path refuses cpcv by name off the manifest; a bare frame has no
    # manifest, and the `path` column is the shape that gives it away.
    if "path" in predictions_df.columns:
        raise ValidationError(
            f"{source} carries a 'path' column, which is the shape of "
            "combinatorial purged cross-validation: one prediction per "
            "(entity, date, path), so there is no single out-of-sample "
            "track record to simulate — the paths are alternative "
            "histories, not a sequence. Evaluate a model validated with "
            "walk_forward, or collapse the paths to one series deliberately "
            "before publishing the reference."
        )

    result = _simulate_predictions_portfolio(
        predictions_df,
        task,
        interval=interval,
        provider_name=provider_name,
        calendar=calendar,
        start=start,
        end=end,
        transform=transform,
        portfolio=portfolio,
        run_id=run_id,
        source=source,
        target_horizon=target_horizon,
    )
    return {"source_ref": source, **result}


def _assert_spec_limits(
    transform: PredictionTransformSpec, portfolio: PortfolioSimSpec
) -> None:
    """The two spec limits the simulator would reject every rebalance date
    under, refused before any data is fetched."""
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


def _simulate_predictions_portfolio(
    predictions_df: pd.DataFrame,
    task: Optional[str],
    *,
    interval: str,
    provider_name: str,
    calendar: Optional[str],
    start: str,
    end: str,
    transform: PredictionTransformSpec,
    portfolio: PortfolioSimSpec,
    run_id: str,
    source: str,
    skipped_folds: Optional[List[Any]] = None,
    extra_warnings: Iterable[str] = (),
    target_horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Predictions -> score panel -> weights -> shared-cash simulation ->
    metrics, for a frame from anywhere.

    Everything below the manifest lookup in `evaluate_model_portfolio` was
    already frame-only; this is that half, named, so a reference and a
    registered model reach the identical arithmetic instead of one of them
    getting a second implementation that drifts.

    Returns the result dict WITHOUT the identity field: the caller names
    what it evaluated (`model_id` or `source_ref`) and adds whatever
    lineage it can vouch for to `provenance`.
    """
    _assert_spec_limits(transform, portfolio)
    warnings: List[str] = []

    periods_per_year = periods_per_year_for_interval(interval, calendar)
    if periods_per_year is None:
        periods_per_year = 252
        warnings.append(
            f"interval={interval!r} has no defined bars-per-year without an "
            "exchange calendar (the dataset names none); annualized metrics "
            "(Sharpe, CAGR, volatility, Calmar, annualized turnover) were "
            "computed with 252 and are wrong by a fixed factor for this "
            "interval. Build the dataset with DatasetSpec.calendar set."
        )

    # Whatever names these predictions to a reader of the refusals below:
    # a registered model's artifact URI, or a published reference.
    predictions_uri = source
    if transform.method == "uncertainty_scaled":
        # FROM THE TASK, not from the absent columns. A classification
        # model can never carry an interval -- ModelSpec refuses it,
        # "intervals are calibrated for task='regression' only;
        # task='classification' has no residual to place an interval
        # around" -- so the column-level refusal below told the caller to
        # "train the model with ModelSpec.intervals", which is the one
        # thing the spec forbids. A closed loop is worse than a refusal.
        if task == "classification":
            raise ValidationError(
                "transform.method='uncertainty_scaled' sizes on the width of a "
                "conformal interval, and a classification model has none: "
                "ModelSpec refuses `intervals` for task='classification' "
                "because there is no residual to place an interval around. A "
                "class probability is already a statement of confidence, so "
                "size on it directly — method='sign' recentres it on "
                "`proba_threshold`, and 'cross_sectional_zscore' or "
                "'cross_sectional_rank' size on how far above the threshold "
                "each name sits."
            )
        # Before the pivot, where the interval columns still stand beside
        # the prediction: the score the sizer sees is prediction / width.
        predictions_df = scale_by_uncertainty(predictions_df, predictions_uri)
    # A SURVIVAL SCORE IS A RISK, NOT A RETURN FORECAST. `survival.py` sets
    # the convention -- "higher means sooner" -- and the score panel passes
    # it through unchanged, like a regressor's, so the largest LONG goes to
    # the name whose event is expected soonest. Classification is
    # recentred precisely so that "positive score = the model is bullish"
    # holds for every task; survival was left out of that, and the sign it
    # needs depends on whether the event is good or bad, which only the
    # caller knows.
    if task == "survival":
        warnings.append(
            "task='survival': the predictions are RISK scores — higher means "
            "the event comes sooner — and they are sized as given, so this "
            "book is LONG the names whose event is expected soonest. If the "
            "event is adverse (a default, a delisting, a halt), that is the "
            "portfolio inverted: negate the predictions before evaluating, or "
            "read every sign in this result backwards. Nothing here can tell "
            "which, because the polarity of the event is not recorded in the "
            "model."
        )
    score_panel = predictions_to_score_panel(predictions_df, task, predictions_uri)

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
            f"{predictions_uri} has predictions for {len(entities)} "
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
    # Prices are fetched by each entity's symbol and keyed back by the
    # entity, so a venue-qualified universe simulates under its own names.
    plan = fetch_plan(entities)
    fetched = fetch_universe_ohlcv(
        provider,
        list(plan.values()),
        start,
        end,
        interval,
    )
    price_data = {
        entity: fetched[symbol] for entity, symbol in plan.items() if symbol in fetched
    }
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
        # WHICH OF THE TWO CAUSES. This blamed the frequency in both
        # cases, so a run whose predictions simply do not overlap the price
        # calendar was told to "use a higher frequency" -- and found daily
        # failing identically, because daily is already the highest and
        # the frequency was never the problem. Measured live: 504
        # prediction dates, 0 tradable at weekly AND at daily.
        selected = select_rebalance_dates(
            pd.DatetimeIndex(score_panel.index), transform.rebalance_frequency
        )
        if len(selected) >= 2:
            first, last = score_panel.index[0], score_panel.index[-1]
            raise ValidationError(
                f"{len(selected)} rebalance date(s) were selected from "
                f"{len(score_panel.index)} prediction date(s), and "
                f"{len(rebalance_dates)} of them are in the master trading "
                "calendar — the intersection of every entity's price index. "
                "The frequency is not the problem and raising it will not "
                f"help. The predictions span {pd.Timestamp(first).date()} to "
                f"{pd.Timestamp(last).date()}; the prices "
                + (
                    f"span {pd.Timestamp(master_index[0]).date()} to "
                    f"{pd.Timestamp(master_index[-1]).date()}"
                    if len(master_index)
                    else "are empty for this universe"
                )
                + ". Fetch prices covering the prediction window, or check "
                "that the entities are the ones the model was fitted on."
            )
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

    # THE ROWS ARE NAMED HERE, NOT BY PANDAS. `.loc` with an Index of
    # labels named its rows after that Index up to pandas 3.0 -- here the
    # schedule, which carries no name -- and pandas 3.1 names them after
    # the panel's own index, `date`. The weights keep these rows, and
    # `canonical_frame_hash` covers the index's name, so identical weights
    # were hashed and their artifact named differently under 3.1. The
    # index is left unnamed, as every earlier pandas left it.
    if transform.method == "mean_variance":
        # The whole score panel, not only the rebalance rows: the realized
        # IC is measured on every prediction date whose forward window has
        # closed. The closes are the master calendar's, the bars the
        # simulation prices the book on.
        closes = pd.DataFrame(
            {column: price_data[str(column)]["Close"] for column in score_panel.columns}
        ).reindex(master_index)
        # The opens the simulator fills a next_open rebalance at, which is
        # what the held book is marked from.
        opens = (
            pd.DataFrame(
                {
                    column: price_data[str(column)]["Open"]
                    for column in score_panel.columns
                }
            ).reindex(master_index)
            if portfolio.fill_price == "next_open"
            else None
        )
        weights, transform_diagnostics, _ = mean_variance_weights(
            score_panel,
            rebalance_dates,
            closes,
            transform,
            linear_cost=portfolio.commission_pct + portfolio.slippage_pct,
            periods_per_year=periods_per_year,
            fill_price=portfolio.fill_price,
            opens=opens,
            target_horizon=target_horizon,
        )
        warnings.extend(transform_diagnostics["warnings"])
    else:
        rebalance_scores = score_panel.loc[rebalance_dates].rename_axis(index=None)
        weights, transform_diagnostics = transform_predictions_to_weights(
            rebalance_scores, transform, returns_df
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
    if (
        transform_diagnostics["n_dates_with_no_position"]
        and transform.method != "mean_variance"
    ):
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
    warnings.extend(extra_warnings)

    # Names the artifact as well as identifying it, so it is the hash that
    # does not depend on the pandas version (see FRAME_HASH_VERSION). A
    # file written under the earlier digest keeps its name and loads as
    # before; evaluating again writes the same weights once more under the
    # new name.
    weights_hash = canonical_frame_hash(weights)
    weights_uri = _artifacts.save_artifact(
        weights,
        run_id=run_id,
        name=f"target_weights_{weights_hash}",
        overwrite=True,
    )

    logger.debug(
        "[evaluate_portfolio] source=%s  entities=%d  rebalances=%d  "
        "method=%s  freq=%s  gross=%.3f  net=%.3f  fill=%s",
        source,
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
        max_adv_participation=portfolio.max_adv_participation,
        **_spec_financing(portfolio),
    )
    warnings.extend(simulation.get("warnings", []))

    metrics = _summarize_simulation(
        simulation,
        periods_per_year,
        portfolio.risk_free_rate,
        portfolio.commission_pct,
        portfolio.slippage_pct,
        day_count=portfolio.day_count,
        initial_capital=portfolio.initial_capital,
    )

    equity_curve = simulation["equity_curve"]
    equity_hash = canonical_frame_hash(equity_curve.to_frame(name="equity"))
    equity_uri = _artifacts.save_artifact(
        equity_curve,
        run_id=run_id,
        name=f"portfolio_equity_{equity_hash}",
        overwrite=True,
    )

    # THE SAME BYTES, ADDRESSED BY KIND. The URIs above stay for existing
    # callers and the MCP resource layer; these are what every
    # reference-taking tool in every runtime accepts. Without them this
    # function was a terminal node: `handoff.register` refuses to adopt a
    # path for a non-external kind, so an agent holding the URI could not
    # mint a reference. The equity curve converts to a returns reference
    # (`convert_reference` to `returns_panel`); the deflated Sharpe, PBO
    # and reality-check tools read returns inline and take no reference.
    #
    # `overwrite=True` for the same reason the artifacts above use it: the
    # name carries the frame's own digest, so re-evaluating the same
    # portfolio rewrites identical bytes under an identical name, and a
    # reference that resolved once resolves to the same value.
    from standard_quant_tools.agent.runtimes import handoff

    weights_ref = handoff.publish(
        weights,
        "weight_panel",
        run_id,
        f"target_weights_{weights_hash}_ref",
        producer="modeling.evaluate_model_portfolio",
        overwrite=True,
    )
    equity_curve_ref = handoff.publish(
        equity_curve,
        "equity_curve",
        run_id,
        f"portfolio_equity_{equity_hash}_ref",
        producer="modeling.evaluate_model_portfolio",
        overwrite=True,
    )

    return {
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
        "target_weights_ref": weights_ref,
        "equity_curve_ref": equity_curve_ref,
        # The inputs that determined the numbers above and that this
        # function can vouch for on its own. A caller holding a manifest
        # adds the lineage only it knows (the predictions digest, the
        # dataset and the estimator) on top.
        "provenance": {
            "target_weights_hash": weights_hash,
            "equity_curve_hash": equity_hash,
            "frame_hash_version": FRAME_HASH_VERSION,
            "task": task,
            "interval": interval,
            "provider": provider_name,
            "calendar": calendar,
            "periods_per_year": int(periods_per_year),
            "transform_spec": transform.model_dump(),
            "portfolio_spec": portfolio.model_dump(),
            # The rates the account was financed at once None was resolved
            # to the risk-free rate, and the currency each came to.
            "financing": dict(simulation.get("financing") or {}),
        },
        "warnings": warnings,
    }
