"""Tree-based estimator allowlist, both tasks — from scikit-learn>=1.3.0."""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
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
register_estimator(
    "regression",
    "random_forest",
    RandomForestRegressor,
    {"n_estimators", "max_depth"},
)
register_estimator(
    "regression",
    "gradient_boosting",
    GradientBoostingRegressor,
    {"n_estimators", "max_depth", "learning_rate"},
)
register_estimator(
    "classification",
    "gradient_boosting",
    GradientBoostingClassifier,
    {"n_estimators", "max_depth", "learning_rate"},
)
