"""
Gradient-boosting estimators from LightGBM and XGBoost, plus quantile
regression — all registered only when their library is importable.

WHY THESE EXIST. Measured on this pipeline, `random_forest` takes 174
seconds for one walk-forward run at 50 entities. That is not a slow run at
universe scale, it is an unusable one, and no amount of tuning the
surrounding Python changes it — the cost is in sklearn's tree building.
LightGBM and XGBoost fit the same shape of model with histogram-based
split finding and native multithreading, which is the difference between
minutes and seconds on a panel of this size. sklearn's own
`hist_gradient_boosting` closes much of that gap and remains available with
no extra dependency; these are here for the cases where it does not.

WHY THEY ARE OPTIONAL. Neither is a declared dependency of this package. A
missing library must not break an import of `standard_quant_tools.modeling`,
so registration is guarded and a caller who asks for an unavailable
estimator gets the registry's ordinary "unknown estimator, allowed: [...]"
error listing what IS installed. The alternative — a hard dependency on two
compiled libraries for an optional capability — is a worse trade.

QUANTILE REGRESSION. `quantile` predicts a chosen quantile of the target
rather than its mean, which is what makes an uncertainty band possible: fit
the 10th, 50th and 90th and the spread between them is the model's own
statement about how confident it is. It also has a property that matters
for return data specifically — the median is far less sensitive to a fat
tail than the mean, so a single extreme move does not drag the whole fit.
"""

import logging

from sklearn.ensemble import GradientBoostingRegressor as _GradientBoostingRegressor

from .bounds import (
    ALPHA,
    LEARNING_RATE,
    MAX_DEPTH,
    N_ESTIMATORS,
    EstimatorParamSchema,
    ParamBound,
)
from .registry import register_estimator

logger = logging.getLogger(__name__)

# Bounds specific to the boosting libraries. Ceilings are resource budgets
# on the same reasoning as trees.py: an unbounded value here is not a bad
# hyperparameter, it is a way for one tool call to exhaust the machine.
NUM_LEAVES = ParamBound(
    "int", 2, 4_096, note="Leaf counts above a few thousand are a memory concern."
)
SUBSAMPLE = ParamBound("float", 1e-3, 1.0)
COLSAMPLE = ParamBound("float", 1e-3, 1.0)
MIN_CHILD_SAMPLES = ParamBound("int", 1, 1_000_000)
MIN_CHILD_WEIGHT = ParamBound("float", 0.0, 1e9)
REG_TERM = ParamBound("float", 0.0, 1e9)
QUANTILE = ParamBound(
    "float",
    0.0,
    1.0,
    note="The quantile to predict; 0.5 is the median.",
)

_LIGHTGBM = EstimatorParamSchema(
    bounds={
        "n_estimators": N_ESTIMATORS,
        "num_leaves": NUM_LEAVES,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "min_child_samples": MIN_CHILD_SAMPLES,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
        "reg_alpha": REG_TERM,
        "reg_lambda": REG_TERM,
    }
)

_XGBOOST = EstimatorParamSchema(
    bounds={
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
        "reg_alpha": REG_TERM,
        "reg_lambda": REG_TERM,
    }
)

_QUANTILE_LINEAR = EstimatorParamSchema(
    bounds={
        "quantile": QUANTILE,
        "alpha": ALPHA,
        "fit_intercept": ParamBound("bool"),
        "solver": ParamBound("str", choices=("highs", "highs-ds", "highs-ipm")),
    }
)

_QUANTILE_GB = EstimatorParamSchema(
    bounds={
        "alpha": QUANTILE,  # GradientBoostingRegressor names the quantile 'alpha'
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
    }
)


def _register_lightgbm() -> bool:
    try:
        import lightgbm
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError:
        return False

    # LightGBM prints a banner per fit, and a 20-fold run with an inner
    # hyperparameter search would bury the actual output under hundreds of
    # lines of library chatter. Routed through Python logging (quiet by
    # default, and still reachable if someone wants it) rather than by
    # subclassing to pin verbose=-1: an sklearn estimator's __init__ must
    # name every parameter explicitly, because get_params() reads them off
    # the signature. A subclass taking **kwargs silently loses random_state
    # from get_params and then fails inside fit -- which is exactly what
    # the first version of this file did.
    try:
        lightgbm.register_logger(logging.getLogger("lightgbm"))
    except Exception:  # noqa: BLE001 - older versions lack register_logger
        logger.debug("[modeling] lightgbm.register_logger unavailable")

    register_estimator("regression", "lightgbm", LGBMRegressor, _LIGHTGBM)
    register_estimator("classification", "lightgbm", LGBMClassifier, _LIGHTGBM)
    return True


def _register_xgboost() -> bool:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        return False
    register_estimator("regression", "xgboost", XGBRegressor, _XGBOOST)
    register_estimator("classification", "xgboost", XGBClassifier, _XGBOOST)
    return True


class QuantileGradientBoostingRegressor(_GradientBoostingRegressor):
    """
    GradientBoostingRegressor with the quantile loss pinned on.

    Every constructor parameter is named explicitly and none is forwarded
    through **kwargs. That is not style: sklearn's get_params() reads the
    parameter names off this signature, and an estimator that hides them
    behind **kwargs loses random_state from get_params and then fails
    inside fit with an error that points at the library rather than at the
    subclass. `loss` is deliberately absent -- this entry exists to BE
    quantile regression, and letting the loss be overridden would make the
    estimator's registered name a lie.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state=None,
    ):
        super().__init__(
            loss="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )


def _register_quantile() -> None:
    """
    Quantile regression, always available (both come from sklearn).

    Two shapes because they answer differently: `quantile` is a linear
    program solved exactly, so it is interpretable and cheap on a small
    feature set; `quantile_gradient_boosting` captures interactions but
    fits one model per quantile and costs accordingly.
    """
    from sklearn.linear_model import QuantileRegressor

    register_estimator("regression", "quantile", QuantileRegressor, _QUANTILE_LINEAR)
    register_estimator(
        "regression",
        "quantile_gradient_boosting",
        QuantileGradientBoostingRegressor,
        _QUANTILE_GB,
    )


_register_quantile()
HAS_LIGHTGBM = _register_lightgbm()
HAS_XGBOOST = _register_xgboost()

if not (HAS_LIGHTGBM or HAS_XGBOOST):
    logger.debug(
        "[modeling] neither lightgbm nor xgboost is installed; "
        "hist_gradient_boosting remains the fastest available booster"
    )
