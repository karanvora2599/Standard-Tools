"""Tree-based estimator allowlist, both tasks — from scikit-learn>=1.3.0.

`n_estimators`, `max_iter` and `max_depth` carry explicit ceilings. These
are the parameters where an unbounded value is not merely a bad
hyperparameter but a resource-exhaustion path: an agent could request
n_estimators=10_000_000 in a single tool call and pin CPU and memory for as
long as sklearn kept fitting. The ceilings are generous enough that any
realistic research request passes (see estimators/bounds.py).

HISTOGRAM BOOSTING'S EARLY STOPPING IS TIME-ORDERED. scikit-learn's own
rule, under its default `early_stopping='auto'`, sets aside a SHUFFLED
`validation_fraction` of the training rows above 10,000 of them and stops
when the loss on those rows has not improved for `n_iter_no_change`
iterations. On a panel those rows are dated inside the training window,
among rows whose overlapping labels share their outcomes, so the stopping
point was chosen on rows that are not out of sample in time. Here the
validation rows are the training window's LAST dates instead, with the
rows whose labels reach into them dropped from the fit, and scikit-learn
is handed them as `X_val` (see `prepare_early_stopping`). The MLPs of
`neural.py` stop on the same block: their fit takes `X_val` too.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from standard_quant_tools.error import ValidationError

from .bounds import (
    LEARNING_RATE,
    MAX_DEPTH,
    MAX_ITER,
    N_ESTIMATORS,
    EstimatorParamSchema,
    FlagBound,
    FractionBound,
    ParamBound,
)
from .registry import EstimatorCost, register_estimator

#: scikit-learn's own threshold under `early_stopping='auto'`: a fit of
#: more than this many training rows stops early, a smaller one does not.
AUTO_EARLY_STOPPING_ROWS = 10_000

#: The first scikit-learn release whose histogram-boosting `fit` takes a
#: validation set (`X_val`, `y_val`, `sample_weight_val`).
VALIDATION_SET_RELEASE = "1.7"

_HIST_GB_CLASSES = (HistGradientBoostingRegressor, HistGradientBoostingClassifier)


def _early_stopping_compatibility(params: Dict[str, Any]) -> None:
    """validation_fraction and n_iter_no_change describe the stopping rule,
    so they are refused beside early_stopping=False, where no rule runs."""
    if params.get("early_stopping", "auto") is not False:
        return
    given = sorted(
        p for p in ("validation_fraction", "n_iter_no_change") if p in params
    )
    if given:
        raise ValidationError(
            "estimator 'hist_gradient_boosting': early_stopping=False fits "
            f"without a stopping rule, so {given} would be ignored. Drop "
            f"{'it' if len(given) == 1 else 'them'}, or set early_stopping to "
            "True or 'auto'."
        )


_HIST_GB = EstimatorParamSchema(
    bounds={
        "max_iter": MAX_ITER,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "early_stopping": FlagBound(
            "bool",
            choices=("auto",),
            note=(
                "'auto' (the default) stops early when the training window "
                "has more than 10,000 rows, scikit-learn's own threshold; "
                "True always, False never. The validation rows are the "
                "window's last validation_fraction of dates, and the rows "
                "whose labels end on or after the first of them are left "
                "out of the fit (and so is every row on the label horizon's "
                "dates before them). Needs scikit-learn 1.7 or later: under "
                "an older one True is refused and 'auto' fits without early "
                "stopping and says so."
            ),
        ),
        "validation_fraction": FractionBound(
            "float",
            0.01,
            0.5,
            note=(
                "The share of the training window's DATES, its last ones, "
                "that early stopping is scored on; not a share of rows, and "
                "not drawn at random."
            ),
        ),
        "n_iter_no_change": ParamBound(
            "int",
            1,
            100_000,
            note=(
                "Boosting stops when the loss on the validation dates has "
                "not improved for this many iterations; max_iter still caps "
                "the total."
            ),
        ),
    },
    compatibility=(_early_stopping_compatibility,),
)
# The three that decide what a forest costs besides its size. A forest's
# fit time is in sorting each candidate feature at each node: scikit-learn
# considers every feature at every split for a regressor (max_features=1.0)
# and bootstraps as many rows as it has. Measured on one walk-forward fold
# of a 30-name daily equity panel, 200 trees of depth 6: max_features=0.33
# fitted 2.58x faster and max_samples=0.5 1.67x. Each defaults to
# scikit-learn's own value when absent, so a spec that names none of them
# fits the forest it always did.
_RANDOM_FOREST = EstimatorParamSchema(
    bounds={
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "max_features": FractionBound(
            "float",
            1e-3,
            1.0,
            choices=("sqrt", "log2"),
            allow_none=True,
            note=(
                "The share of features each split considers, as a fraction, "
                "or 'sqrt' / 'log2' of their count; None is all of them."
            ),
        ),
        "max_samples": FractionBound(
            "float",
            1e-3,
            1.0,
            allow_none=True,
            note=(
                "The share of training rows each tree's bootstrap sample "
                "draws; None is as many as there are."
            ),
        ),
        "min_samples_leaf": ParamBound(
            "int",
            1,
            100_000,
            note="The fewest training rows a leaf may hold.",
        ),
    }
)
_GRADIENT_BOOSTING = EstimatorParamSchema(
    bounds={
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
    }
)


# ── Early stopping on a time-ordered validation block ──────────────────


def fit_takes_validation_set() -> bool:
    """Whether this scikit-learn's histogram boosting accepts `X_val` in
    `fit` (from 1.7). Read off the signature rather than the version
    string, so a backport or a patched build answers for itself."""
    return "X_val" in inspect.signature(HistGradientBoostingRegressor.fit).parameters


def is_hist_gradient_boosting(estimator_cls: Any) -> bool:
    """Whether a registered class is scikit-learn's histogram boosting."""
    return isinstance(estimator_cls, type) and issubclass(
        estimator_cls, _HIST_GB_CLASSES
    )


def stops_on_validation_set(estimator_cls: Any) -> bool:
    """
    Whether a registered class early-stops on a validation set its fit is
    handed as `X_val`, `y_val` and `sample_weight_val`: scikit-learn's
    histogram boosting, or a class that says so by carrying the protocol
    `neural._TimeOrderedEarlyStopping` gives the MLPs -- `estimator_name`
    (its registry name), `min_validation_rows`, and a static
    `takes_validation_set()` answering for this scikit-learn.
    """
    return is_hist_gradient_boosting(estimator_cls) or (
        isinstance(estimator_cls, type)
        and callable(getattr(estimator_cls, "takes_validation_set", None))
    )


def _profile(estimator_cls: type) -> "Tuple[str, bool, int]":
    """A validation-set fitter's registry name, whether this scikit-learn
    lets its fit take a validation set, and the fewest validation rows its
    rule scores (scikit-learn's histogram boosting takes one)."""
    if issubclass(estimator_cls, _HIST_GB_CLASSES):
        return "hist_gradient_boosting", fit_takes_validation_set(), 1
    return (
        str(estimator_cls.estimator_name),
        bool(estimator_cls.takes_validation_set()),
        int(estimator_cls.min_validation_rows),
    )


def _validation_fitter(estimator: Any) -> "Tuple[Any, bool]":
    """The estimator a fit trains that stops on a validation set -- the
    estimator itself, or the one a probability calibration wraps -- and
    whether it is wrapped; None for any other estimator."""
    if stops_on_validation_set(type(estimator)):
        return estimator, False
    inner = getattr(estimator, "estimator", None)
    if inner is not None and stops_on_validation_set(type(inner)):
        from sklearn.calibration import CalibratedClassifierCV

        if isinstance(estimator, CalibratedClassifierCV):
            return inner, True
    return None, False


def _sklearn_version() -> str:
    import sklearn

    return str(sklearn.__version__)


def _old_sklearn_refusal(where: str) -> str:
    return (
        f"{where}: early_stopping=True needs scikit-learn {VALIDATION_SET_RELEASE} "
        "or later, whose HistGradientBoosting fit takes a validation set "
        f"(X_val); this is scikit-learn {_sklearn_version()}. Without one, "
        "scikit-learn stops on a shuffled share of the training rows, dated "
        "among the rows it fits and sharing their overlapping labels' "
        "outcomes, and the library does not fit it that way. Upgrade "
        "scikit-learn, or set early_stopping=False ('auto' then fits without "
        "early stopping and says so)."
    )


def _no_validation_set_refusal(estimator_cls: type, where: str) -> str:
    """Why early_stopping=True cannot run under this scikit-learn."""
    if issubclass(estimator_cls, _HIST_GB_CLASSES):
        return _old_sklearn_refusal(where)
    return (
        f"{where}: early_stopping=True needs a time-ordered validation set, "
        f"and this scikit-learn's ({_sklearn_version()}) "
        f"{estimator_cls.__name__} cannot be handed one. Without one, "
        "scikit-learn stops on a shuffled share of the training rows, dated "
        "among the rows it fits and sharing their overlapping labels' "
        "outcomes, and the library does not fit it that way. Set "
        "early_stopping=False."
    )


def refuse_early_stopping_without_validation_set(
    estimator_cls: Any, params: Dict[str, Any], search: Any = None
) -> None:
    """
    Refuse, before any data is read, a spec that asks for early stopping
    -- in its params or on a search axis -- from an estimator that stops on
    a validation set, under a scikit-learn that cannot hand it one.

    'auto' is not refused: it was never chosen, and it fits without early
    stopping under such a scikit-learn, with a warning (see
    `prepare_early_stopping`).
    """
    if not stops_on_validation_set(estimator_cls) or _profile(estimator_cls)[1]:
        return
    asked = params.get("early_stopping") is True
    grid = getattr(search, "param_grid", None) or {}
    asked = asked or any(v is True for v in grid.get("early_stopping", ()))
    if asked:
        raise ValidationError(
            _no_validation_set_refusal(estimator_cls, "run_model_experiment")
        )


def _date_label(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def time_ordered_validation(
    dates: np.ndarray,
    label_end: "np.ndarray | None",
    horizon: "int | None",
    fraction: float,
) -> "Tuple[Optional[Dict[str, Any]], Optional[str]]":
    """
    The fitted rows and the validation rows of one training window, or the
    reason the window cannot give them.

    THE RULE `select_features` HOLDS OUT BY. The validation rows are those
    on the window's last `fraction` of its distinct dates, counted as
    `_selection_cutoff` counts a holdout: the first floor(n * (1 -
    fraction)) dates, at least one and at most n - 1, are before it.
    Between them and the fitted rows is the embargo `_split_at_holdout`
    puts before a holdout, with the label horizon h as its length: a row
    dated on the last h dates before the validation block is dropped, and
    so is an earlier one whose recorded `label_end_date` falls on or after
    the block's first date. With no horizon the second rule alone applies,
    which is the walk-forward purge's (`label_overlap_mask`), so either way
    no fitted label ends inside the block. A row with no label end (NaT) has
    no resolved label to leak and is kept, as the purge keeps it.

    Written here on the arrays rather than called: `_split_at_holdout`
    reads a frame, and pandas' `to_datetime` on its date columns took 15 to
    19 ms a call on windows of the live panel's sizes (14,970 and 31,680
    rows), where the fit itself takes 33 to 65 ms on one thread; this takes
    0.2 to 0.3 ms. The masks are the same, and a test holds them to it.

    Returns ({"fit_rows", "validation_rows" (boolean masks), "embargo_dates",
    "n_embargoed_rows", "validation_start", "validation_end",
    "n_validation_dates"}, None), or (None, reason).
    """
    dates = np.asarray(dates)
    n_rows = int(dates.shape[0])
    window = np.unique(dates)
    n_dates = int(window.size)
    h = int(horizon) if horizon is not None and horizon > 0 else 0
    if n_dates < 2:
        return None, (
            f"its {n_rows:,} rows span {n_dates} date(s), and a validation "
            "block after the fitted rows needs at least two"
        )
    n_before = int(np.floor(n_dates * (1.0 - float(fraction))))
    n_before = min(max(n_before, 1), n_dates - 1)
    n_validation_dates = n_dates - n_before
    if h >= n_before:
        return None, (
            f"its {n_rows:,} rows span {n_dates} date(s): validation takes "
            f"the last {n_validation_dates} and the label horizon's embargo "
            f"the {h} before them, leaving no date to fit on"
        )
    first = window[n_before]
    validation_rows = dates >= first
    fit_rows = dates <= window[n_before - 1 - h]
    if label_end is not None:
        fit_rows &= ~(np.asarray(label_end) >= first)
    if not fit_rows.any():
        return None, (
            f"every row before the validation block's first date, "
            f"{_date_label(first)}, has a label_end_date on or after it, "
            "leaving no row to fit on"
        )
    return {
        "fit_rows": fit_rows,
        "validation_rows": validation_rows,
        "embargo_dates": h,
        "n_embargoed_rows": int(n_rows - fit_rows.sum() - validation_rows.sum()),
        "validation_start": _date_label(first),
        "validation_end": _date_label(window[-1]),
        "n_validation_dates": int(n_validation_dates),
    }, None


@dataclass
class EarlyStoppingFit:
    """
    What the library did to one fit's stopping rule (histogram boosting or
    an MLP).

    `block` is the split from `time_ordered_validation` when the fit early-
    stops on it; None when early stopping was turned off for this fit, and
    then `off_kind` ('scikit-learn' or 'window') and `reason` say why.
    """

    block: Optional[Dict[str, Any]] = None
    off_kind: Optional[str] = None
    reason: Optional[str] = None
    n_window_dates: Optional[int] = None

    def apply(
        self, X: np.ndarray, y: np.ndarray, weights: "np.ndarray | None"
    ) -> "Tuple[np.ndarray, np.ndarray, np.ndarray | None, Dict[str, Any]]":
        """The rows `fit` is handed, and the validation arguments beside
        them; the window unchanged when early stopping is off."""
        if self.block is None:
            return X, y, weights, {}
        fit, val = self.block["fit_rows"], self.block["validation_rows"]
        extra: Dict[str, Any] = {"X_val": X[val], "y_val": y[val]}
        if weights is not None:
            extra["sample_weight_val"] = weights[val]
        return X[fit], y[fit], (None if weights is None else weights[fit]), extra

    def report(self, estimator: Any) -> Dict[str, Any]:
        """The record a fold or the refit carries: the validation block and
        the iterations boosting ran (an MLP's epochs), or why early stopping
        was off. An MLP also reports `best_iter`, the epoch whose weights
        it kept, which scikit-learn restores at the end of its epochs."""
        fitter, wrapped = _validation_fitter(estimator)
        if wrapped:
            fitted = [
                c.estimator for c in getattr(estimator, "calibrated_classifiers_", [])
            ]
        else:
            fitted = [] if fitter is None else [fitter]
        n_iter: Any = [int(f.n_iter_) for f in fitted]
        best: Any = [f.best_iteration() for f in fitted if hasattr(f, "best_iteration")]
        if not wrapped:
            n_iter = n_iter[0] if n_iter else None
            best = best[0] if best else None
        if self.block is None:
            return {"applied": False, "reason": self.reason, "n_iter": n_iter}
        block = self.block
        record = {
            "applied": True,
            "validation_start": block["validation_start"],
            "validation_end": block["validation_end"],
            "n_validation_dates": block["n_validation_dates"],
            "n_validation_rows": int(block["validation_rows"].sum()),
            "embargo_dates": block["embargo_dates"],
            "n_embargoed_rows": block["n_embargoed_rows"],
            "n_fit_rows": int(block["fit_rows"].sum()),
            "n_iter": n_iter,
        }
        if fitter is not None and hasattr(fitter, "best_iteration"):
            record["best_iter"] = best
        return record


def prepare_early_stopping(
    estimator: Any,
    y: np.ndarray,
    index: Any,
    horizon: "int | None",
) -> Optional[EarlyStoppingFit]:
    """
    Decide one fit's stopping rule, for histogram boosting or an MLP, and
    set the estimator's `early_stopping` to match. None -- the fit left
    exactly as scikit-learn would run it -- for any other estimator, for
    `early_stopping=False`, and for 'auto' on 10,000 rows or fewer.

    Otherwise the fit early-stops on `time_ordered_validation`'s block,
    with `early_stopping` set to True: scikit-learn's 'auto' decides on
    the rows `fit` is handed, which are now fewer than the window's, and
    would turn the rule off for a window just above the threshold. The
    window, not the fitted rows, is what 'auto' is decided on. An MLP's
    `early_stopping` is True or False; it has no 'auto'.

    When the rule cannot run on a time-ordered block -- a scikit-learn
    whose fit cannot take a validation set (histogram boosting before
    1.7), or a window whose dates cannot hold a validation block and the
    embargo before it with a row left to fit (scikit-learn's own minimum
    is one row each side, two validation rows for an MLP, and for a
    classifier both classes among the fitted rows and no class in the
    validation rows that the fitted rows lack) -- 'auto' turns early
    stopping off for this fit and says why, and True is refused by name.
    A window under calibration is split the same way and every
    calibration fit stops on its block.
    """
    fitter, wrapped = _validation_fitter(estimator)
    if fitter is None:
        return None
    setting = fitter.early_stopping
    n_rows = int(len(y))
    if setting is False or (setting == "auto" and n_rows <= AUTO_EARLY_STOPPING_ROWS):
        return None
    explicit = setting is True
    name, takes_validation_set, min_validation_rows = _profile(type(fitter))
    where = f"estimator '{name}'"

    def off(kind: str, reason: str, n_dates: Optional[int]) -> EarlyStoppingFit:
        fitter.set_params(early_stopping=False)
        return EarlyStoppingFit(off_kind=kind, reason=reason, n_window_dates=n_dates)

    if not takes_validation_set:
        if explicit:
            raise ValidationError(_no_validation_set_refusal(type(fitter), where))
        return off(
            "scikit-learn",
            f"scikit-learn {_sklearn_version()} takes no validation set "
            f"(X_val needs {VALIDATION_SET_RELEASE}), so this fit used every "
            "row of its window without early stopping",
            None,
        )
    if index is None:
        reason = "the fit carried no sample index to order its rows by date"
        if explicit:
            raise ValidationError(
                f"{where}: early_stopping=True needs a time-ordered validation "
                f"block, and {reason}."
            )
        return off("window", reason, None)

    fraction = fitter.validation_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, float):
        raise ValidationError(
            f"{where}: validation_fraction={fraction!r} must be a share of the "
            "training window's dates, such as 0.1."
        )
    dates = np.asarray(index.dates)
    n_dates = int(np.unique(dates).size)
    block, reason = time_ordered_validation(
        dates, getattr(index, "label_end", None), horizon, fraction
    )
    if block is not None and is_classifier(fitter):
        fitted = np.unique(np.asarray(y)[block["fit_rows"]])
        validated = np.unique(np.asarray(y)[block["validation_rows"]])
        if fitted.size < 2:
            block, reason = None, "the rows left to fit on hold one class"
        elif np.setdiff1d(validated, fitted).size:
            block, reason = None, (
                "the validation rows hold a class the rows left to fit on do not"
            )
    if block is not None and int(block["validation_rows"].sum()) < min_validation_rows:
        block, reason = None, (
            f"its last {block['n_validation_dates']} date(s) hold "
            f"{int(block['validation_rows'].sum())} row(s), and {name}'s "
            f"validation score needs at least {min_validation_rows}"
        )
    if block is not None and wrapped:
        if int(block["fit_rows"].sum()) == int(block["validation_rows"].sum()):
            # The calibration hands each of its fits the arguments whose
            # length matches X cut to that fit's rows, which would cut the
            # validation rows too.
            block, reason = None, (
                "the fitted and validation rows number the same "
                f"({int(block['fit_rows'].sum()):,}), and the calibration would "
                "then cut the validation rows as if they were fitted ones"
            )
    if block is None:
        if explicit:
            advice = (
                "Set early_stopping='auto' or False, lower validation_fraction, "
                "or widen the training window."
                if is_hist_gradient_boosting(type(fitter))
                else "Set early_stopping=False, or widen the training window."
            )
            raise ValidationError(
                f"{where}: early_stopping=True needs a time-ordered validation "
                f"block, and this training window cannot give one: {reason}. "
                f"{advice}"
            )
        return off("window", str(reason), n_dates)
    fitter.set_params(early_stopping=True)
    return EarlyStoppingFit(block=block, n_window_dates=n_dates)


def early_stopping_warnings(notes: List[EarlyStoppingFit]) -> List[str]:
    """
    One warning per reason early stopping was turned off under 'auto',
    counted over every fit of a run -- folds, search candidates, interval
    fits and the refit. Built from counts and minimums, so the text does
    not depend on the order fits running side by side finished in.
    """
    warnings: List[str] = []
    by_kind: Dict[str, List[EarlyStoppingFit]] = {}
    for note in notes:
        if note.off_kind is not None:
            by_kind.setdefault(note.off_kind, []).append(note)
    old = by_kind.get("scikit-learn")
    if old:
        warnings.append(
            f"hist_gradient_boosting: early_stopping='auto' stops early above "
            f"{AUTO_EARLY_STOPPING_ROWS:,} training rows on the window's last "
            "dates, and this scikit-learn "
            f"({_sklearn_version()}) cannot take a validation set (X_val needs "
            f"{VALIDATION_SET_RELEASE}), so {len(old)} fit(s) of that size used "
            "every row of their window and ran up to max_iter iterations "
            "without early stopping, rather than stop on a shuffled share of "
            "their own rows. Upgrade scikit-learn for the time-ordered rule, or "
            "set early_stopping=False to fit this way without this warning."
        )
    windows = by_kind.get("window")
    if windows:
        known = [n.n_window_dates for n in windows if n.n_window_dates is not None]
        fewest = f" (the fewest dates in one: {min(known)})" if known else ""
        warnings.append(
            f"hist_gradient_boosting: early_stopping='auto' stops early above "
            f"{AUTO_EARLY_STOPPING_ROWS:,} training rows, and {len(windows)} "
            "fit(s) of that size could not hold a time-ordered validation "
            "block -- too few dates for it and the label horizon's embargo "
            "before it with rows left to fit, or, for a classifier, one class "
            f"left to fit on{fewest} -- so each used every row of its window "
            "without early stopping. validation_report.folds[i].early_stopping"
            ".reason says why for a fold."
        )
    return warnings


# One default fit on a 15,030-row, 8-feature window of a 30-name daily
# equity panel, 16 logical cores (see EstimatorCost): hist_gradient_boosting
# 0.16 s regression / 0.24 s classification on one OpenMP thread when it
# runs its 100 iterations (1.75 s / 1.27 s at the runtime's default of every
# core, under the PASSIVE wait policy) -- a window of 10,000 rows or fewer,
# or early_stopping=False. Stopping early on the window's last dates, a
# 14,880-row fold stops after 10 to 16 iterations and fits in 0.039-0.045 s
# / 0.054-0.063 s; the class stays medium, the cost of a full fit;
# random_forest 11.1 s regression / 2.5 s classification (the regressor
# considers every feature at every split, the classifier the square root
# of them); gradient_boosting 3.1 s / 2.8 s.
register_estimator(
    "regression",
    "hist_gradient_boosting",
    HistGradientBoostingRegressor,
    _HIST_GB,
    cost=EstimatorCost("medium", "openmp"),
)
register_estimator(
    "classification",
    "hist_gradient_boosting",
    HistGradientBoostingClassifier,
    _HIST_GB,
    cost=EstimatorCost("medium", "openmp"),
)
register_estimator(
    "classification",
    "random_forest",
    RandomForestClassifier,
    _RANDOM_FOREST,
    cost=EstimatorCost("high", "budget"),
)
register_estimator(
    "regression",
    "random_forest",
    RandomForestRegressor,
    _RANDOM_FOREST,
    cost=EstimatorCost("high", "budget"),
)
register_estimator(
    "regression",
    "gradient_boosting",
    GradientBoostingRegressor,
    _GRADIENT_BOOSTING,
    cost=EstimatorCost("high", "one"),
)
register_estimator(
    "classification",
    "gradient_boosting",
    GradientBoostingClassifier,
    _GRADIENT_BOOSTING,
    cost=EstimatorCost("high", "one"),
)
