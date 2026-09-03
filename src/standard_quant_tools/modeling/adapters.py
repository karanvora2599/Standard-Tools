"""
One place that knows how a task's models are fitted, scored and judged.

WHY THIS EXISTS NOW, AND NOT EARLIER. The engine's `X` was a flat
`(n_rows, n_features)` matrix and every estimator took `fit(X, y)` /
`predict(X)`, so a single code path served everything. Adding rankers broke
that: they need the rows sorted by query group, the target converted to
integer grades, a `group` argument at fit time, and a metric set that
excludes R2 because their output has no scale. That arrived as `if task ==
"ranking"` branches in four places — the fold fit, the final refit, the
prediction step and the metric selection — and a fifth would have been due
the moment anything else non-sklearn-shaped showed up.

So the branching is collected here instead of spread through
`run_experiment`. Each adapter answers four questions about its task:

    prepare()      what the estimator actually needs to be handed
    score()        how to get a continuous score out of a fitted estimator
    metrics()      which numbers mean anything for this kind of model
    fold_ic()      what this fold contributes to the pooled OOS statistics

WHAT THIS IS NOT. It is not a plugin system, and it deliberately does not
try to abstract over models that need a different `X` altogether — a
sequence model wanting `(entity, time, feature)` tensors, or a graph model
wanting an adjacency structure. Those need the DATASET to emit a different
shape, which is a change to `build_dataset`, not to this file. Pretending
otherwise would produce an interface shaped by speculation rather than by
two real cases; when a third arrives, the honest move is to extend this from
what it actually needs.

`capabilities()` exists so an agent can ask what a model can do before
choosing it, rather than the registry growing a tool per model.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .validation.metrics import (
    classification_metrics,
    cross_sectional_ic,
    positive_class_proba,
    regression_metrics,
)
from .validation.ranking import (
    fold_ic_series,
    group_sizes,
    ranking_metrics,
    relevance_grades,
)


@dataclass(frozen=True)
class FitArrays:
    """Exactly what gets handed to `estimator.fit`, after any reshaping."""

    X: np.ndarray
    y: np.ndarray
    sample_weight: Optional[np.ndarray] = None
    group: Optional[np.ndarray] = None

    def fit_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.sample_weight is not None:
            kwargs["sample_weight"] = self.sample_weight
        if self.group is not None:
            kwargs["group"] = self.group
        return kwargs


def _exposes_coefficients(estimator_cls: type) -> bool:
    """
    Whether a fitted instance will have `coef_`.

    `hasattr(cls, "coef_")` is the obvious check and is WRONG: sklearn sets
    `coef_` during fit, so it does not exist on the class and every linear
    model reported as having no coefficients. Class membership is what is
    actually knowable before fitting, and it is what `fold_feature_importance`
    will find at fit time -- the two must agree, or the capability report
    promises a diagnostic the run then does not produce.
    """
    try:
        from sklearn.linear_model._base import LinearClassifierMixin, LinearModel

        if issubclass(estimator_cls, (LinearModel, LinearClassifierMixin)):
            return True
    except ImportError:  # pragma: no cover - sklearn is a core dependency
        pass
    # Non-sklearn estimators may declare it as a property or class attribute.
    return hasattr(estimator_cls, "coef_")


def _probes_true(estimator_cls: type, attribute: str) -> bool:
    """
    Whether a DEFAULT instance really exposes `attribute`.

    `hasattr` on the class is not the same question when sklearn guards a
    method with `available_if`: the descriptor is present on the class and
    only raises when accessed on an instance whose configuration does not
    support it. Falls back to the class check when the estimator cannot be
    constructed without arguments, which is the best answer available then.

    A conditional capability is reported for the DEFAULT configuration --
    an estimator can still be configured out of it, and the registry's
    allowlist is what keeps that from happening silently.
    """
    try:
        return hasattr(estimator_cls(), attribute)
    except Exception:  # pragma: no cover - estimator needs constructor args
        return hasattr(estimator_cls, attribute)


class ModelAdapter:
    """Base adapter: the flat-matrix, `fit(X, y)` case every sklearn
    estimator satisfies. Subclasses override only what differs."""

    task: ClassVar[str] = ""
    #: Whether this task needs query groups at fit time.
    needs_groups: ClassVar[bool] = False
    #: Whether the score this adapter produces has meaningful units. False
    #: for rankers, whose output is invariant to any monotone rescale — the
    #: reason R2 and MAE are not reported for them.
    score_has_scale: ClassVar[bool] = True

    def prepare(
        self,
        model_spec: Any,
        frame: pd.DataFrame,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> FitArrays:
        return FitArrays(X=X.to_numpy(), y=y, sample_weight=weights)

    def score(self, estimator: Any, X: pd.DataFrame) -> np.ndarray:
        """
        The continuous score downstream code consumes (see modeling.bridge).

        Always continuous, never a discrete label — a portfolio needs an
        ordering, and `predict` on a classifier throws that away.
        """
        return np.asarray(estimator.predict(X.to_numpy()))

    def metrics(
        self,
        model_spec: Any,
        estimator: Any,
        X: pd.DataFrame,
        y_true: np.ndarray,
        score: np.ndarray,
        dates: Optional[np.ndarray],
        train_y: Optional[np.ndarray],
    ) -> Dict[str, float]:
        raise NotImplementedError

    def fold_ic(
        self, y_true: np.ndarray, score: np.ndarray, dates: Optional[np.ndarray]
    ) -> Dict[str, pd.Series]:
        """Per-date IC series this fold contributes to the pooled OOS
        dispersion statistics — see aggregate_cross_sectional_ic for why the
        pooling cannot be done by averaging per-fold values."""
        if dates is None:
            return {}
        return {
            "cs_ic": cross_sectional_ic(y_true, score, dates, "pearson"),
            "cs_rank_ic": cross_sectional_ic(y_true, score, dates, "spearman"),
        }

    def capabilities(self, estimator_cls: type) -> Dict[str, Any]:
        """
        What this (task, estimator) pair can do, read off the class rather
        than hardcoded — so a newly registered estimator describes itself
        correctly without anyone remembering to update a table.
        """
        try:
            fit_params = set(inspect.signature(estimator_cls.fit).parameters)
        except (TypeError, ValueError):  # pragma: no cover - builtins/C types
            fit_params = set()
        return {
            "task": self.task,
            "input_kind": "tabular",
            "needs_groups": self.needs_groups,
            "score_has_scale": self.score_has_scale,
            "supports_sample_weight": "sample_weight" in fit_params,
            "supports_partial_fit": hasattr(estimator_cls, "partial_fit"),
            # NOT hasattr on the class. sklearn guards some methods with
            # `available_if`, a descriptor that EXISTS on the class and
            # raises only on instance access -- so SGDClassifier reported
            # supports_probability=True while an instance carrying its
            # default hinge loss has no predict_proba at all. Probing a
            # default instance describes the estimator as it would actually
            # be built, which is what this dict claims to do.
            "supports_probability": _probes_true(estimator_cls, "predict_proba"),
            "exposes_coefficients": _exposes_coefficients(estimator_cls),
            # A property on the class, so hasattr sees it before fitting.
            # Correctly False for HistGradientBoosting, which genuinely has
            # none -- that is why fold_feature_importance reports NaN for it.
            "exposes_feature_importance": hasattr(
                estimator_cls, "feature_importances_"
            ),
        }


class RegressionAdapter(ModelAdapter):
    task = "regression"

    def metrics(self, model_spec, estimator, X, y_true, score, dates, train_y):
        return regression_metrics(y_true, score, dates=dates, train_y=train_y)


class ClassificationAdapter(ModelAdapter):
    task = "classification"

    def score(self, estimator: Any, X: pd.DataFrame) -> np.ndarray:
        """
        The positive-class PROBABILITY, not the 0/1 predicted label.

        A discrete label carries no ordering, and everything downstream —
        the portfolio bridge, the cross-sectional IC — needs one. See
        positive_class_proba for why the column is looked up by class rather
        than hardcoded to index 1.
        """
        return positive_class_proba(estimator, X.to_numpy())

    def metrics(self, model_spec, estimator, X, y_true, score, dates, train_y):
        # The discrete label is needed for accuracy, which the probability
        # cannot give; both come from the same fitted estimator.
        predicted = estimator.predict(X.to_numpy())
        return classification_metrics(y_true, predicted, score)

    def fold_ic(self, y_true, score, dates):
        # Cross-sectional IC against a 0/1 target is not the quantity the
        # classification report is built around, and pooling it would put a
        # number in the summary that nothing else references.
        return {}


class RankingAdapter(ModelAdapter):
    task = "ranking"
    needs_groups = True
    score_has_scale = False

    def prepare(self, model_spec, frame, X, y, weights) -> FitArrays:
        """
        Reorder and re-label a training fold for a learning-to-rank fit.

        Three things must be true before either ranker trains correctly, and
        only the first raises if you get it wrong:

          1. the LABEL must be integer relevance grades. Both libraries
             reject a continuous target outright, and shifting it to be
             non-negative does not help.
          2. the ROWS must be ordered by query group. Both take `group` as
             consecutive counts and neither verifies the ordering, so
             unsorted rows train silently on the wrong groupings.
          3. the GROUP counts must match that ordering, which group_sizes()
             checks rather than assumes.

        Sorted by (date, entity), not date alone. Date alone is enough for
        the grouping to be CORRECT, but leaves the within-date row order
        equal to whatever order the caller's panel happened to be in — and
        both libraries break histogram ties by row order, so the same data
        arriving differently produced a slightly different model. Measured at
        0.5849 vs 0.5862 rank IC on a shuffled panel: close enough to prove
        the grouping was right, different enough that a run was not
        reproducible from its inputs alone.

        Grading is per date because that is what a query group is here: the
        ranker learns "AAPL should rank above MSFT *today*". Grades pooled
        across dates would be asking it to rank today's names against last
        year's.
        """
        dates_raw = frame["date"].to_numpy()
        order = np.lexsort((frame["entity"].to_numpy(), dates_raw))
        dates = dates_raw[order]
        return FitArrays(
            X=X.to_numpy()[order],
            y=relevance_grades(y[order], dates, model_spec.ranking.n_grades),
            sample_weight=None if weights is None else weights[order],
            group=group_sizes(dates),
        )

    def metrics(self, model_spec, estimator, X, y_true, score, dates, train_y):
        return ranking_metrics(
            y_true,
            score,
            dates,
            n_grades=model_spec.ranking.n_grades,
            ks=tuple(model_spec.ranking.ndcg_at or (5, 10)),
        )

    def fold_ic(self, y_true, score, dates):
        return fold_ic_series(y_true, score, dates) if dates is not None else {}


_ADAPTERS: Dict[str, ModelAdapter] = {
    RegressionAdapter.task: RegressionAdapter(),
    ClassificationAdapter.task: ClassificationAdapter(),
    RankingAdapter.task: RankingAdapter(),
}


def get_adapter(task: str) -> ModelAdapter:
    adapter = _ADAPTERS.get(task)
    if adapter is None:
        raise ValidationError(
            f"no model adapter for task={task!r} — known tasks: " f"{sorted(_ADAPTERS)}"
        )
    return adapter


def available_tasks() -> List[str]:
    return sorted(_ADAPTERS)
