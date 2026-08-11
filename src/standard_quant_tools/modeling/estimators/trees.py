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
)
from .registry import register_estimator

_HIST_GB = EstimatorParamSchema(
    bounds={
        "max_iter": MAX_ITER,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
    }
)
_RANDOM_FOREST = EstimatorParamSchema(
    bounds={"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH}
)
_GRADIENT_BOOSTING = EstimatorParamSchema(
    bounds={
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
    }
)

register_estimator(
    "regression", "hist_gradient_boosting", HistGradientBoostingRegressor, _HIST_GB
)
register_estimator(
    "classification", "hist_gradient_boosting", HistGradientBoostingClassifier, _HIST_GB
)
register_estimator(
    "classification", "random_forest", RandomForestClassifier, _RANDOM_FOREST
)
register_estimator("regression", "random_forest", RandomForestRegressor, _RANDOM_FOREST)
register_estimator(
    "regression", "gradient_boosting", GradientBoostingRegressor, _GRADIENT_BOOSTING
)
register_estimator(
    "classification",
    "gradient_boosting",
    GradientBoostingClassifier,
    _GRADIENT_BOOSTING,
)
