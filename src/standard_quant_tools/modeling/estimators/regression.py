"""Regression estimator allowlist — from scikit-learn>=1.3.0, already a
core dependency of this package (no new install).

Each entry declares parameter BOUNDS, not just names: an allowlist of
names is not a compute budget, and an unbounded max_iter/alpha is an
agent-triggerable resource exhaustion path (see estimators/bounds.py)."""

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .bounds import (
    ALPHA,
    FIT_INTERCEPT,
    L1_RATIO,
    LEARNING_RATE,
    MAX_DEPTH,
    MAX_ITER,
    N_ESTIMATORS,
    EstimatorParamSchema,
)
from .registry import register_estimator

register_estimator(
    "regression",
    "linear",
    LinearRegression,
    EstimatorParamSchema(bounds={"fit_intercept": FIT_INTERCEPT}),
)
register_estimator(
    "regression",
    "ridge",
    Ridge,
    EstimatorParamSchema(
        bounds={"alpha": ALPHA, "fit_intercept": FIT_INTERCEPT, "max_iter": MAX_ITER}
    ),
)
register_estimator(
    "regression",
    "lasso",
    Lasso,
    EstimatorParamSchema(
        bounds={"alpha": ALPHA, "fit_intercept": FIT_INTERCEPT, "max_iter": MAX_ITER}
    ),
)
register_estimator(
    "regression",
    "elastic_net",
    ElasticNet,
    EstimatorParamSchema(
        bounds={
            "alpha": ALPHA,
            "l1_ratio": L1_RATIO,
            "fit_intercept": FIT_INTERCEPT,
            "max_iter": MAX_ITER,
        }
    ),
)

__all__ = ["LEARNING_RATE", "MAX_DEPTH", "N_ESTIMATORS"]
