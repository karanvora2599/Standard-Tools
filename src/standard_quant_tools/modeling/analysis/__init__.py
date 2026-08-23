"""
Feature research, as opposed to feature computation.

`features/` builds columns. This package asks whether those columns are worth
building: how well each one is populated, whether it predicts anything, how
much of it is already expressed by its neighbours, and whether its apparent
predictive power survives a causality check.

Kept separate from `validation/` on purpose. `validation/` measures a fitted
MODEL; everything here is model-independent and runs on the dataset alone,
before an estimator has been chosen.
"""

from .feature_report import (
    build_feature_report,
    feature_distribution_stats,
    feature_predictive_stats,
    lead_lag_ic_curve,
    redundancy_report,
)

__all__ = [
    "build_feature_report",
    "feature_distribution_stats",
    "feature_predictive_stats",
    "lead_lag_ic_curve",
    "redundancy_report",
]
