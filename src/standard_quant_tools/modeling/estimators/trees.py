"""Tree-based estimator allowlist, both tasks — from scikit-learn>=1.3.0.

`n_estimators`, `max_iter` and `max_depth` carry explicit ceilings. These
are the parameters where an unbounded value is not merely a bad
hyperparameter but a resource-exhaustion path: an agent could request
n_estimators=10_000_000 in a single tool call and pin CPU and memory for as
long as sklearn kept fitting. The ceilings are generous enough that any
realistic research request passes (see estimators/bounds.py)."""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from .bounds import (
    LEARNING_RATE,
    MAX_DEPTH,
    MAX_ITER,
    N_ESTIMATORS,
    EstimatorParamSchema,
    FractionBound,
    ParamBound,
)
from .registry import EstimatorCost, register_estimator

_HIST_GB = EstimatorParamSchema(
    bounds={
        "max_iter": MAX_ITER,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
    }
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

# One default fit on a 15,030-row, 8-feature window of a 30-name daily
# equity panel, 16 logical cores (see EstimatorCost): hist_gradient_boosting
# 0.16 s regression / 0.24 s classification on one OpenMP thread (1.75 s /
# 1.27 s at the runtime's default of every core, under the PASSIVE wait
# policy);
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
