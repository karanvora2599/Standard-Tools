"""Classification estimator allowlist — from scikit-learn>=1.3.0.

`solver` is exposed alongside `penalty` and the pair is validated together
(see bounds.LOGISTIC_SCHEMA). Previously `penalty` was exposed on its own,
so `penalty="l1"` reached LogisticRegression's default lbfgs solver — which
does not support it — and failed deep inside sklearn instead of at this
boundary with a message naming the compatible solvers."""

from sklearn.linear_model import LogisticRegression

from .bounds import LOGISTIC_SCHEMA
from .registry import EstimatorCost, register_estimator

# One default fit on a 15,030-row, 8-feature window, 16 logical cores:
# 0.0065 s. See EstimatorCost.
register_estimator(
    "classification",
    "logistic",
    LogisticRegression,
    LOGISTIC_SCHEMA,
    cost=EstimatorCost("low", "one"),
)
