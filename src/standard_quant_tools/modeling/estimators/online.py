"""
Incremental learners, and an honest account of what "online" already is
here.

WHAT WAS ALREADY BUILT. Walk-forward validation refits the estimator on
every fold, so a model in this library is already re-estimated as time
advances -- that is the part of online learning that matters for
correctness. And `validation/weights.py` already implements exponential
TIME DECAY, so "weight recent evidence more" is a spec field, not a missing
capability. Adding a second mechanism for either would be the same feature
twice.

WHAT WAS ACTUALLY MISSING is an estimator whose fit is cheap enough to
repeat often and whose step size is under the caller's control. SGD is
that: a fold refit costs a pass or two rather than a full re-solve, and
`eta0` / `learning_rate` decide how hard recent rows move the coefficients
-- a second, optimizer-level expression of recency that composes with the
sample weights rather than replacing them.

WHAT IS DELIBERATELY NOT EXPOSED. `partial_fit` -- updating the fitted
coefficients with new rows instead of refitting -- is not reachable through
a ModelSpec, and that is a decision rather than an omission. The engine's
guarantee is that every out-of-sample prediction comes from an estimator
that never saw that row or anything overlapping it; a warm-started
estimator carries state from every previous fold, including rows inside the
current fold's purge and embargo window. The speed it buys is real and the
guarantee it costs is the one every number downstream depends on. An
estimator that cannot state where its coefficients came from cannot support
the lineage this library records.

SCALE SENSITIVITY. SGD is more scale-sensitive than any other estimator
registered here -- an unscaled column dominates the gradient outright. The
engine winsorizes and z-scores per fold on training rows only, so that is
handled; it is worth knowing because an SGD model fitted on a panel
assembled ANY other way will behave badly for reasons that look like the
model and are not.
"""

from __future__ import annotations

from sklearn.linear_model import SGDClassifier, SGDRegressor

from standard_quant_tools.error import ValidationError

from .bounds import ALPHA, FIT_INTERCEPT, MAX_ITER, EstimatorParamSchema, ParamBound
from .registry import register_estimator

#: How the step size shrinks as fitting proceeds.
_LEARNING_RATE = ParamBound(
    "str",
    choices=("constant", "optimal", "invscaling", "adaptive"),
    note=(
        "'invscaling' (sklearn's default for regression) decays the step "
        "over the fit. 'constant' keeps it fixed, which is what makes later "
        "rows move the coefficients as much as early ones -- the "
        "optimizer-level version of recency weighting."
    ),
)

_ETA0 = ParamBound(
    "float",
    1e-9,
    10.0,
    note="Initial step size. Read by every schedule except 'optimal'.",
)

_LOSS_REGRESSION = ParamBound(
    "str",
    choices=(
        "squared_error",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ),
    note=(
        "'huber' is the one to reach for on financial targets: squared "
        "error lets a single 8-sigma day contribute 64 times what a "
        "1-sigma day does, and those days exist."
    ),
)

#: ONLY the losses that yield a probability. sklearn offers 'hinge',
#: 'squared_hinge' and 'perceptron' as well, and they are absent on
#: purpose: a classifier registered here is asked for probabilities
#: unconditionally -- `adapters.score` calls `positive_class_proba` on
#: every fold -- so a hinge SGD raises a raw sklearn AttributeError from
#: inside walk-forward, having already trained. Offering a choice that
#: cannot work is worse than not offering it.
_LOSS_CLASSIFICATION = ParamBound(
    "str",
    choices=("log_loss", "modified_huber"),
    note=(
        "'hinge', 'squared_hinge' and 'perceptron' are deliberately not "
        "offered: they have no predict_proba, and this library asks every "
        "classifier for one."
    ),
)

_PENALTY = ParamBound("str", choices=("l2", "l1", "elasticnet"), allow_none=True)

_L1_RATIO = ParamBound("float", 0.0, 1.0)

_TOL = ParamBound("float", 1e-12, 1.0, allow_none=True)

_RANDOM_STATE = ParamBound(
    "int",
    0,
    2**31 - 1,
    allow_none=True,
    note="Row shuffling is random, so two runs of the same spec differ "
    "without this and nothing in the manifest explains why.",
)

_SHARED = {
    "alpha": ALPHA,
    "penalty": _PENALTY,
    "l1_ratio": _L1_RATIO,
    "fit_intercept": FIT_INTERCEPT,
    "max_iter": MAX_ITER,
    "tol": _TOL,
    "learning_rate": _LEARNING_RATE,
    "eta0": _ETA0,
    "random_state": _RANDOM_STATE,
}


def _elasticnet_needs_a_ratio(params) -> None:
    """`l1_ratio` is silently ignored unless the penalty is elasticnet,
    and an elasticnet with the default 0.15 is a choice nobody made."""
    if params.get("penalty") == "elasticnet" and "l1_ratio" not in params:
        raise ValidationError(
            "estimator 'sgd': penalty='elasticnet' mixes L1 and L2 in a "
            "proportion set by l1_ratio, and leaving it out takes sklearn's "
            "0.15 -- a weighting nobody chose. Pass l1_ratio explicitly."
        )


register_estimator(
    "regression",
    "sgd",
    SGDRegressor,
    EstimatorParamSchema(
        bounds={**_SHARED, "loss": _LOSS_REGRESSION},
        compatibility=(_elasticnet_needs_a_ratio,),
    ),
)


class ProbabilisticSGDClassifier(SGDClassifier):
    """
    SGDClassifier defaulting to a loss that can produce a probability.

    sklearn's default is 'hinge', which has none. Every other classifier
    in this registry works with no params at all, and this one did not: a
    bare EstimatorSpec(type="sgd") trained for several folds and then died
    inside `positive_class_proba` with an AttributeError from sklearn's
    internals. The default is changed rather than the caller being asked
    to know this, because a default that violates the contract the engine
    enforces is not a default, it is a trap.
    """

    def __init__(
        self,
        loss: str = "log_loss",
        penalty="l2",
        alpha: float = 1e-4,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol=1e-3,
        learning_rate: str = "optimal",
        eta0: float = 0.0,
        random_state=None,
    ):
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate,
            eta0=eta0,
            random_state=random_state,
        )


register_estimator(
    "classification",
    "sgd",
    ProbabilisticSGDClassifier,
    EstimatorParamSchema(
        bounds={**_SHARED, "loss": _LOSS_CLASSIFICATION},
        compatibility=(_elasticnet_needs_a_ratio,),
    ),
)

__all__ = ["ProbabilisticSGDClassifier", "SGDRegressor"]
