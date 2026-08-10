"""Regression estimator allowlist — from scikit-learn>=1.3.0, already a
core dependency of this package (no new install)."""

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .registry import register_estimator

register_estimator("regression", "linear", LinearRegression, {"fit_intercept"})
register_estimator("regression", "ridge", Ridge, {"alpha", "fit_intercept"})
register_estimator("regression", "lasso", Lasso, {"alpha", "fit_intercept"})
register_estimator(
    "regression", "elastic_net", ElasticNet, {"alpha", "l1_ratio", "fit_intercept"}
)
