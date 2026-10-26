"""
modeling.model_sleeves: registered models' out-of-sample portfolios as
return streams on one calendar -- the sleeves capital is allocated across.

WHY THIS EXISTS. Allocating across models took three tools per model:
`evaluate_model_portfolio` (an equity curve), `convert_reference` (that
curve as returns), and `combine_return_streams` once for all of them. On the
way nothing said how many models had been tried before these were kept,
which is the count `evaluate_allocation_rule` needs for its deflated Sharpe
ratio, nor that the first stretch of every curve is the account holding
cash before its model's first prediction. Here each model is evaluated by
the function `evaluate_model_portfolio` calls, its stream is read off that
evaluation's own equity curve, the streams meet on the dates they share, and
the registry is asked how many models were trained on the same datasets.

THE STREAM IS MEASURED FROM THE CAPITAL, as `portfolio_eval.
_summarize_simulation` measures growth, so it compounds to the evaluation's
`cumulative_return` whatever the opening bar did: the curve's first point is
the opening bar after any trade on it, net of its costs, and it is replaced
by the capital the account opened with rather than kept as the base. A
model's first prediction comes long after its curve opens, so its opening
bar trades nothing and reading from that bar would agree; the rule is the
evaluation's all the same, so the stream and the evaluation cannot drift
apart on a curve whose opening bar does trade.

THE IDLE BARS ARE NOT THE MODEL'S. The evaluation prices the book over the
dataset's whole window, and the first prediction comes only after the
features' warm-up and the first training window, so every curve opens with
a stretch -- often a third of it -- in which the account held cash. Those
returns are the cash rate's (exactly zero at the default rate). Left in a
panel, they are periods in which every sleeve looks riskless: an allocation
fitted over them reads a sleeve's volatility low, and a walk-forward scores
periods in which no model had traded. So each stream starts, by default, at
its model's first live bar -- the first rebalance date under a close fill,
the bar after it under the next open -- and the bars before it are counted
and compounded (`n_idle_returns`, `idle_return`) rather than dropped
silently: the live stream times the idle growth is the evaluation's total,
and at a zero cash rate the live stream alone is. `keep_idle=True` keeps
them, for a panel that is the evaluations' curves bar for bar.

THE TRIAL COUNT is every model registered on the datasets these models were
trained on, read off the registry the way `list_models` reads it, and never
fewer than the models passed. It is a lower bound on the configurations
behind the panel: a transform tried and discarded, or a model deleted, leaves
no trace in the registry. A model's own hyperparameter search runs inside
each fold of its walk-forward, so its out-of-sample record already pays for
that search, and it is not counted again.

REFUSED, each by name: fewer than two models (one stream is nothing to
allocate across) or more than twenty; a model or a sleeve name given twice;
a model validated by combinatorial purged cross-validation, which has no
single track record; models whose bars are not the same period (a daily
return beside a weekly one on one row is not one period, and a panel has
one `periods_per_year`); and streams sharing fewer dates than `min_overlap`.
Models whose tasks or labels differ are NOT refused: each stream is an
account's return per bar whatever the model predicted, so the panel means
the same thing either way. Models trained on different target horizons are
said to be, because one shared rebalance schedule suits one horizon at most.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.audit.hashing import canonical_frame_hash
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.portfolio import correlation_matrix
from standard_quant_tools.portfolio.streams import (
    align_return_streams,
    alignment_notes,
)

from . import artifacts as _artifacts
from .bridge import _refuse_cpcv
from .engine import _target_horizon
from .features.base import periods_per_year_for_interval
from .portfolio_eval import _assert_spec_limits, evaluate_model_portfolio
from .registry.model_registry import load_dataset_spec, load_manifest
from .specs import PortfolioSimSpec, PredictionTransformSpec

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_MODELS",
    "MIN_MODELS",
    "SleeveRequest",
    "build_model_sleeves",
    "check_model_count",
    "check_sleeve_requests",
    "opening_capital_returns",
    "registered_trials",
]

#: The fewest models one call takes: one stream is nothing to allocate across.
MIN_MODELS = 2

#: The most. Each is a full evaluation with its own price fetch, and the
#: allocation tools fit a covariance of the sleeves, which twenty already
#: asks a lot of a few years of daily returns.
MAX_MODELS = 20

#: The bars-per-year an evaluation falls back to when its interval has none
#: without a calendar, as `_simulate_predictions_portfolio` does.
_FALLBACK_PERIODS_PER_YEAR = 252


@dataclass(frozen=True)
class SleeveRequest:
    """One model to evaluate as a sleeve: its id, the panel column it is
    published under (the id when None), and its own transform (the shared
    one when None)."""

    model_id: str
    name: Optional[str] = None
    transform: Optional[PredictionTransformSpec] = None

    @property
    def sleeve(self) -> str:
        return self.model_id if self.name is None else self.name


def _requests(models: Sequence[Any], what: str) -> List[SleeveRequest]:
    """The models as `SleeveRequest`s, refused by name when they are too
    few or too many or name a model or a sleeve twice."""
    out: List[SleeveRequest] = []
    for entry in models:
        if isinstance(entry, SleeveRequest):
            out.append(entry)
        elif isinstance(entry, Mapping):
            out.append(SleeveRequest(**dict(entry)))
        else:
            # The agent layer's input model, or anything carrying the fields.
            out.append(
                SleeveRequest(
                    model_id=entry.model_id,
                    name=getattr(entry, "name", None),
                    transform=getattr(entry, "transform", None),
                )
            )
    check_sleeve_requests(out, what)
    return out


def check_model_count(count: int, what: str) -> None:
    """Refuse fewer than `MIN_MODELS` or more than `MAX_MODELS` models."""
    if count < MIN_MODELS:
        raise ValidationError(
            f"{what}: models has {count} entry(ies), and a panel to allocate "
            f"across needs at least {MIN_MODELS}: one model is one stream, and "
            "there is no split of capital to make. Evaluate a single model "
            "with evaluate_model_portfolio, or add a second model_id."
        )
    if count > MAX_MODELS:
        raise ValidationError(
            f"{what}: models has {count} entries, more than the {MAX_MODELS} "
            "one call evaluates. Each is a full evaluation, and a covariance "
            "of more sleeves than that is mostly noise on a few years of "
            "returns; pass the ones you mean to allocate across."
        )


def check_sleeve_requests(requests: Sequence[SleeveRequest], what: str) -> None:
    """The refusals that need no registry: how many, and nothing twice."""
    check_model_count(len(requests), what)
    ids = [r.model_id for r in requests]
    repeated = sorted({m for m in ids if ids.count(m) > 1})
    if repeated:
        raise ValidationError(
            f"{what}: models names {repeated} more than once. The same "
            "predictions under two transforms are one model's bet sized two "
            "ways, and an allocator reads the two as independent sleeves. "
            "Compare transforms with evaluate_model_portfolio, pass the model "
            "once with the transform you chose, and count the ones you tried "
            "in n_trials."
        )
    names = [r.sleeve for r in requests]
    clashing = sorted({n for n in names if names.count(n) > 1})
    if clashing:
        raise ValidationError(
            f"{what}: the sleeve name(s) {clashing} are given to more than one "
            "model (a name defaults to its model_id). Each name is a column "
            "of the published panel, so give each model a distinct name."
        )


def opening_capital_returns(
    equity: pd.Series, initial_capital: float, *, what: str = "model sleeve"
) -> pd.Series:
    """
    An evaluated account's return per bar, measured from the capital it
    opened with.

    The curve's first point is the opening bar after any rebalance on it,
    net of its costs, so it is replaced by the capital -- exactly as
    `portfolio_eval._summarize_simulation` reads growth -- and the returns
    are dated on the curve's bars from the second on: n points give n - 1
    returns, and they compound to `curve[-1] / capital - 1`, the
    evaluation's `cumulative_return`.
    """
    curve = equity.dropna()
    if len(curve) < 2:
        raise ValidationError(
            f"{what}: the equity curve has {len(curve)} bar(s), and a return "
            "needs two."
        )
    capital = float(initial_capital)
    if not (math.isfinite(capital) and capital > 0):
        raise ValidationError(
            f"{what}: initial_capital={initial_capital!r} is the level every "
            "return is measured from, so it must be finite and positive."
        )
    levels = np.concatenate(([capital], curve.to_numpy(dtype=float)[1:]))
    return pd.Series(
        levels[1:] / levels[:-1] - 1.0,
        index=curve.index[1:],
        dtype="float64",
    )


def _first_live_position(
    equity_index: pd.Index, first_rebalance: Any, fill_price: str, what: str
) -> int:
    """
    The position, in the curve's bars, of the first return the model's book
    can move: the first rebalance date itself under a close fill (the trade
    and its cost land on that close), the bar after it under the next open.
    Never 0 -- the opening bar's return is the next bar's, from the capital.
    """
    try:
        position = equity_index.get_loc(pd.Timestamp(first_rebalance))
    except KeyError as exc:
        raise ValidationError(
            f"{what}: the first rebalance date {first_rebalance} is not a bar "
            "of the evaluation's own equity curve, so its first live return "
            "cannot be placed. Evaluate the model again; if this persists, "
            "the evaluation's artifacts are not the ones it published."
        ) from exc
    if not isinstance(position, (int, np.integer)):
        raise ValidationError(
            f"{what}: the equity curve repeats the bar {first_rebalance}."
        )
    return max(int(position) + (0 if fill_price == "close" else 1), 1)


def registered_trials(dataset_ids: Iterable[str]) -> Dict[str, int]:
    """
    How many models the registry holds on each of these datasets.

    Read the way `list_models` reads the registry -- every `mdl_*` manifest
    under the runs directory, a manifest that does not load skipped rather
    than failing the count -- so the two never disagree about what is
    registered. A dataset with nothing on it counts 0.
    """
    wanted = {str(d) for d in dataset_ids}
    counts = {d: 0 for d in sorted(wanted)}
    for path in sorted(_artifacts._runs_dir().glob("mdl_*/manifest.json")):
        try:
            manifest = load_manifest(path.parent.name)
        except Exception:  # a partial or corrupt run, as list_models skips it
            continue
        if manifest.dataset_id in counts:
            counts[manifest.dataset_id] += 1
    return counts


def _bars_a_year(model_id: str) -> Tuple[str, Optional[str], int]:
    """(interval, calendar, periods a year) as the evaluation will read them."""
    spec = load_dataset_spec(model_id)
    interval = str(spec.get("interval", "1d"))
    calendar = spec.get("calendar")
    calendar = str(calendar) if calendar else None
    per_year = periods_per_year_for_interval(interval, calendar)
    return interval, calendar, int(per_year or _FALLBACK_PERIODS_PER_YEAR)


def _merged_warnings(by_sleeve: Mapping[str, Sequence[str]]) -> List[str]:
    """Each sleeve's warnings once, naming every sleeve that raised it: a
    dataset's coverage warning is the same text on every model trained on it."""
    order: List[str] = []
    raised: Dict[str, List[str]] = {}
    for sleeve, notes in by_sleeve.items():
        for note in notes:
            if note not in raised:
                raised[note] = []
                order.append(note)
            raised[note].append(sleeve)
    return [f"sleeve(s) {raised[note]}: {note}" for note in order]


def build_model_sleeves(
    models: Sequence[Any],
    *,
    run_id: str,
    transform: Optional[PredictionTransformSpec] = None,
    portfolio: Optional[PortfolioSimSpec] = None,
    min_overlap: int = 20,
    keep_idle: bool = False,
    what: str = "build_model_sleeves",
) -> Dict[str, Any]:
    """
    Evaluate each model as a portfolio, put the streams on the dates they
    share, and publish the panel as a `returns_panel` at `run_id`.

    `models` holds `SleeveRequest`s (or mappings, or objects, with
    `model_id`, `name` and `transform`); `transform` and `portfolio` are
    shared, with `evaluate_model_portfolio`'s defaults, and a model's own
    transform replaces the shared one for it. Each evaluation publishes its
    own weights and equity curve under the model's id, as
    `evaluate_model_portfolio` always does.

    The panel is named by its own content hash and written with overwrite,
    as the evaluation's artifacts are: the same models and settings under
    the same `run_id` publish the same bytes under the same reference, so a
    repeated call answers the same and refuses nothing.

    Returns the reference (`returns_ref`), the `columns` in the order given,
    the alignment report, `periods_per_year`, per sleeve the evaluation's
    numbers and lineage (`sleeves`), the aligned streams' `correlation`, the
    registry's count per dataset (`trials_by_dataset`) and
    `n_trials_lower_bound`, and warnings.
    """
    from standard_quant_tools.agent.runtimes import handoff

    requests = _requests(models, what)
    shared_transform = transform or PredictionTransformSpec()
    portfolio = portfolio or PortfolioSimSpec()

    # Everything that can be refused without a price is refused before the
    # first fetch: the models exist, none is combinatorial, the bars are one
    # period, and every transform fits the portfolio's limits.
    manifests = {}
    periods: Dict[str, Tuple[str, Optional[str], int]] = {}
    for index, request in enumerate(requests):
        manifest = load_manifest(request.model_id)
        _refuse_cpcv(manifest, what)
        manifests[request.sleeve] = manifest
        periods[request.sleeve] = _bars_a_year(request.model_id)
        try:
            _assert_spec_limits(request.transform or shared_transform, portfolio)
        except ValidationError as exc:
            raise ValidationError(
                f"{what}: models[{index}] ({request.model_id!r}): {exc}"
            ) from exc
    bars = {sleeve: (p[0], p[2]) for sleeve, p in periods.items()}
    if len(set(bars.values())) > 1:
        raise ValidationError(
            f"{what}: the models' bars are not the same period -- (interval, "
            f"bars a year) per sleeve: {bars}. A return over one bar beside a "
            "return over another on one row is not one period, and the panel "
            "is read with one periods_per_year. Pass models trained on the "
            "same interval and calendar."
        )
    per_year = next(iter(bars.values()))[1]

    warnings: List[str] = []
    horizons = {
        sleeve: _target_horizon(manifest.target_id)
        for sleeve, manifest in manifests.items()
    }
    if len({h for h in horizons.values()}) > 1:
        warnings.append(
            f"{what}: the models were trained on different target horizons, "
            f"in bars: {horizons}. Each is evaluated on its own transform's "
            "rebalance schedule, and one shared schedule suits one horizon at "
            "most; give a model whose horizon differs its own transform with "
            "a rebalance_frequency to match, if that is what you mean. The "
            "panel itself is unaffected: every column is an account's return "
            "per bar."
        )

    trials = registered_trials(m.dataset_id for m in manifests.values())

    streams: Dict[str, pd.Series] = {}
    sleeves: Dict[str, Dict[str, Any]] = {}
    raised: Dict[str, List[str]] = {}
    live_from: Dict[str, pd.Timestamp] = {}
    for request in requests:
        sleeve = request.sleeve
        manifest = manifests[sleeve]
        evaluation = evaluate_model_portfolio(
            request.model_id,
            transform=request.transform or shared_transform,
            portfolio=portfolio,
        )
        equity = handoff.resolve(evaluation["equity_curve_ref"], expect="equity_curve")
        returns = opening_capital_returns(
            equity, portfolio.initial_capital, what=f"{what}: sleeve {sleeve!r}"
        )
        # The schedule's first date, read off the weights that drove the
        # simulation rather than off coverage's date string, which drops the
        # time of an intraday bar.
        weights = _artifacts.load_artifact(evaluation["target_weights_uri"])
        first_rebalance = pd.Timestamp(weights.index[0])
        live_at = _first_live_position(
            equity.dropna().index,
            first_rebalance,
            portfolio.fill_price,
            f"{what}: sleeve {sleeve!r}",
        )
        # Return i is bar i + 1's, so the first live bar's is at live_at - 1.
        idle = returns.iloc[: live_at - 1]
        live = returns.iloc[live_at - 1 :]
        if live.empty:
            raise ValidationError(
                f"{what}: sleeve {sleeve!r} first trades on the last bar of its "
                "evaluation, so it has no live return to allocate on."
            )
        idle_growth = float(np.prod(1.0 + idle.to_numpy())) - 1.0 if len(idle) else 0.0
        live_from[sleeve] = pd.Timestamp(live.index[0])
        streams[sleeve] = (returns if keep_idle else live).rename(sleeve)
        metrics = evaluation["metrics"]
        sleeves[sleeve] = {
            "model_id": request.model_id,
            "estimator_type": manifest.estimator_type,
            "dataset_id": manifest.dataset_id,
            "task": manifest.task,
            "target_horizon": horizons[sleeve],
            "sharpe": metrics["sharpe_ratio"],
            "annual_return": metrics["cagr"],
            "volatility": metrics["annualized_volatility"],
            "max_drawdown": metrics["max_drawdown"],
            "total_return": metrics["cumulative_return"],
            "annual_turnover": metrics["annualized_turnover"],
            "n_rebalances": int(metrics["n_rebalances"]),
            "first_rebalance_date": _iso(first_rebalance),
            "n_idle_returns": int(len(idle)),
            "idle_return": idle_growth,
            "n_returns": int(len(streams[sleeve])),
            "equity_curve_ref": evaluation["equity_curve_ref"],
            "target_weights_ref": evaluation["target_weights_ref"],
        }
        raised[sleeve] = list(evaluation.get("warnings") or [])

    frame, report = align_return_streams(streams, min_overlap=min_overlap, what=what)
    warnings += alignment_notes(report, what)
    if keep_idle:
        warnings += _idle_rows_note(frame, live_from, what)
    warnings += _merged_warnings(raised)

    ref = handoff.publish(
        frame,
        kind="returns_panel",
        run_id=run_id,
        name=f"model_sleeves_{canonical_frame_hash(frame)}",
        producer=what,
        overwrite=True,
    )
    with single_threaded_blas():
        correlation = correlation_matrix(frame)
    total = sum(trials.values())
    logger.debug(
        "[%s] sleeves=%d  dates=%d  trials=%s  ref=%s",
        what,
        len(streams),
        len(frame),
        trials,
        ref,
    )
    return {
        "returns_ref": ref,
        "columns": [str(c) for c in frame.columns],
        "alignment": report,
        "periods_per_year": per_year,
        "keep_idle": bool(keep_idle),
        "sleeves": sleeves,
        "correlation": {
            str(row): {str(col): float(v) for col, v in values.items()}
            for row, values in correlation.iterrows()
        },
        "trials_by_dataset": trials,
        "n_trials_lower_bound": max(len(requests), int(total)),
        "warnings": warnings,
    }


def _iso(stamp: pd.Timestamp) -> str:
    """A date as a caller writes it: 'YYYY-MM-DD' unless it has a time."""
    return str(stamp.date()) if stamp == stamp.normalize() else stamp.isoformat()


def _idle_rows_note(
    frame: pd.DataFrame, live_from: Mapping[str, pd.Timestamp], what: str
) -> List[str]:
    """With the idle bars kept: how many of the panel's rows come before a
    sleeve's first live bar, and what that means for the tools reading it."""
    idle = {name: int((frame.index < start).sum()) for name, start in live_from.items()}
    worst = max(idle.values()) if idle else 0
    if not worst:
        return []
    return [
        f"{what}: keep_idle=True keeps each account's bars before its first "
        "live bar, and the panel's rows before it hold that sleeve in cash -- "
        f"{idle} row(s) of {len(frame)}. An allocation fitted over them reads "
        "those sleeves as riskless; keep evaluate_allocation_rule's "
        f"estimation_window at {worst} or more so no scored period precedes a "
        "sleeve's first trade, or leave keep_idle off."
    ]
