"""Tree-based estimator allowlist, both tasks — from scikit-learn>=1.3.0."""

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)

from .registry import register_estimator

register_estimator(
    "regression",
    "hist_gradient_boosting",
    HistGradientBoostingRegressor,
    {"max_iter", "max_depth", "learning_rate"},
)
register_estimator(
    "classification",
    "hist_gradient_boosting",
    HistGradientBoostingClassifier,
    {"max_iter", "max_depth", "learning_rate"},
)
register_estimator(
    "classification",
    "random_forest",
    RandomForestClassifier,
    {"n_estimators", "max_depth"},
)
