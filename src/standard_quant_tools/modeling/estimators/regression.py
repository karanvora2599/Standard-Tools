"""Regression estimator allowlist — from scikit-learn>=1.3.0, already a
core dependency of this package (no new install).

Each entry declares parameter BOUNDS, not just names: an allowlist of
names is not a compute budget, and an unbounded max_iter/alpha is an
agent-triggerable resource exhaustion path (see estimators/bounds.py)."""

from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)

from .bounds import (
    ALPHA,
    FIT_INTERCEPT,
    L1_RATIO,
    LEARNING_RATE,
    MAX_DEPTH,
    MAX_ITER,
    N_ESTIMATORS,
    EstimatorParamSchema,
    ParamBound,
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


# ── robust regression ───────────────────────────────────────────────────
#
# Every squared-loss fit above is steered by its worst week: an 8-sigma day
# contributes 64 times what a 1-sigma day does, and financial targets have
# those. Huber is quadratic near zero and linear in the tail, so a single
# outlier moves the fit by a bounded amount rather than by its square.
#
# `epsilon` is where the loss switches, in units of the residual scale.
# 1.35 is the conventional default and is not arbitrary -- it gives about
# 95% of the efficiency of ordinary least squares when the errors really are
# Gaussian, which is the price paid for the robustness when they are not.
register_estimator(
    "regression",
    "huber",
    HuberRegressor,
    EstimatorParamSchema(
        bounds={
            "epsilon": ParamBound(
                "float",
                1.0,
                10.0,
                note=(
                    "Where the loss switches from quadratic to linear, in "
                    "residual scales. Lower is more robust and less "
                    "efficient; sklearn's default of 1.35 gives about 95% of "
                    "OLS efficiency when the errors really are Gaussian, "
                    "which is the price of the robustness when they are not."
                ),
            ),
            "alpha": ALPHA,
            "fit_intercept": FIT_INTERCEPT,
            "max_iter": MAX_ITER,
        }
    ),
)
