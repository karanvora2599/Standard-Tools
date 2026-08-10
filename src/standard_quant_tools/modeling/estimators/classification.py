"""Classification estimator allowlist — from scikit-learn>=1.3.0."""

from sklearn.linear_model import LogisticRegression

from .registry import register_estimator

register_estimator(
    "classification", "logistic", LogisticRegression, {"C", "penalty", "fit_intercept"}
)
