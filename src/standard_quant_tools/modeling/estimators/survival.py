"""
Survival estimators: a risk score from a duration that may be censored.

WHAT THEY FIT. `y` is (n, 2): the duration until the event, and 1 where
the event was observed or 0 where the window ended first. Every estimator
here reads both columns and emits a RISK -- higher means the event sooner
-- so the engine, the bridge and the portfolio path consume it like any
other score. A regression on the duration alone, which is what the label
was fitted as before this module, reads a censored row as an event at the
horizon; the registry now refuses that pairing by name.

THREE ESTIMATORS, ONE ALWAYS. `cox_ph` is Cox's proportional-hazards model
on its partial likelihood, written here in a hundred lines of numpy with
Breslow's handling of ties, so the survival task exists on every install
and has a linear, coefficient-bearing baseline. `xgboost_cox` and
`xgboost_aft` are two objectives on a library already in the registry:
Cox regression, which XGBoost reads from a signed label (negative means
censored), and the accelerated-failure-time model, which needs the
censoring passed as label bounds through the native API and so is wrapped
rather than reached through `XGBRegressor`. Both register only when
xgboost imports, and are declared in `boosting.OPTIONAL_ESTIMATORS` so the
generated reference lists them either way.

WHY EVERY CONSTRUCTOR NAMES ITS PARAMETERS. `_instantiate` reads the
signature to know whether to pass `random_state`, and the capability
report reads it to know what the estimator accepts. A `**kwargs` wrapper
would hide both, which is the same failure `QuantileGradientBoostingRegressor`
documents.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from standard_quant_tools.error import ValidationError

from .bounds import (
    LEARNING_RATE,
    MAX_DEPTH,
    N_ESTIMATORS,
    EstimatorParamSchema,
    ParamBound,
)
from .registry import register_estimator

#: Bounds for the parameters these estimators take, on the same reasoning
#: as the boosters': a ceiling is a resource budget, not an opinion.
_L2 = ParamBound("float", 0.0, 1e9, note="L2 penalty on the coefficients.")
_MAX_ITER = ParamBound("int", 1, 10_000)
_TOL = ParamBound("float", 1e-12, 1.0)
_FRACTION = ParamBound("float", 1e-3, 1.0)
_REG_TERM = ParamBound("float", 0.0, 1e9)
_MIN_CHILD_WEIGHT = ParamBound("float", 0.0, 1e9)
_AFT_DISTRIBUTION = ParamBound("str", choices=("normal", "logistic", "extreme"))
_AFT_SCALE = ParamBound("float", 1e-3, 100.0)

_COX_PH = EstimatorParamSchema(
    bounds={"alpha": _L2, "max_iter": _MAX_ITER, "tol": _TOL}
)
_XGB_COX = EstimatorParamSchema(
    bounds={
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "min_child_weight": _MIN_CHILD_WEIGHT,
        "subsample": _FRACTION,
        "colsample_bytree": _FRACTION,
        "reg_alpha": _REG_TERM,
        "reg_lambda": _REG_TERM,
    }
)
_XGB_AFT = EstimatorParamSchema(
    bounds={
        **_XGB_COX.bounds,
        "aft_loss_distribution": _AFT_DISTRIBUTION,
        "aft_loss_distribution_scale": _AFT_SCALE,
    }
)


def _split_labels(y: Any) -> "tuple[np.ndarray, np.ndarray]":
    """(duration, event) from the (n, 2) survival label, checked."""
    labels = np.asarray(y, dtype=float)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValidationError(
            "a survival estimator fits a (n, 2) label of [duration, event]; "
            f"got shape {labels.shape}. The survival adapter builds it from "
            "the panel's `target` and `event` columns."
        )
    duration, event = labels[:, 0], labels[:, 1]
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValidationError(
            "every survival duration must be finite and positive; a "
            "non-positive duration has no order among the others and XGBoost "
            "reads its sign as the censoring flag."
        )
    if not np.isin(event, (0.0, 1.0)).all():
        raise ValidationError("the event indicator must be 0 or 1 on every row.")
    if event.sum() == 0:
        raise ValidationError(
            "no observed event in the training rows: with every row censored "
            "there is no ordering to learn."
        )
    return duration, event


class CoxPHRegressor:
    """
    Cox proportional hazards on the partial likelihood, by Newton's method.

    Fits `beta` maximizing the Breslow partial likelihood with an L2
    penalty `alpha`, on features standardized internally for conditioning
    and reported back in the original scale as `coef_`. `predict` returns
    the log hazard ratio `X beta`: the risk score, higher meaning sooner.
    Suited to a modest feature count -- the Hessian is accumulated as an
    (n, p, p) cumulative sum, which is the cost of exact ties handling.
    """

    #: Set at fit; declared on the class so the capability report can see
    #: that a fitted instance carries coefficients.
    coef_: Optional[np.ndarray] = None

    def __init__(self, alpha: float = 0.0, max_iter: int = 100, tol: float = 1e-7):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        duration, event = _split_labels(y)
        n, p = X.shape
        weights = (
            np.ones(n)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        Z = (X - self.mean_) / self.scale_

        # Descending time, so the risk set of row i -- everyone still at
        # risk when its event happens, durations >= t_i -- is a prefix of
        # the sorted rows. Ties are contiguous, and a tie group's risk set
        # is the prefix through the group's LAST member (Breslow).
        order = np.argsort(-duration, kind="stable")
        t, e, w, Zs = duration[order], event[order], weights[order], Z[order]
        _unique, inverse = np.unique(-t, return_inverse=True)
        last_in_group = np.zeros(inverse.max() + 1, dtype=int)
        np.maximum.at(last_in_group, inverse, np.arange(n))
        group_end = last_in_group[inverse]

        beta = np.zeros(p)
        eye = np.eye(p)
        for _iteration in range(self.max_iter):
            eta = np.clip(Zs @ beta, -30.0, 30.0)
            wr = w * np.exp(eta)
            s0 = np.cumsum(wr)[group_end]
            s1 = np.cumsum(wr[:, None] * Zs, axis=0)[group_end]
            s2 = np.cumsum(
                wr[:, None, None] * (Zs[:, :, None] * Zs[:, None, :]), axis=0
            )[group_end]
            mean = s1 / s0[:, None]
            gradient = ((w * e)[:, None] * (Zs - mean)).sum(axis=0) - self.alpha * beta
            curvature = s2 / s0[:, None, None] - mean[:, :, None] * mean[:, None, :]
            hessian = (
                -((w * e)[:, None, None] * curvature).sum(axis=0) - self.alpha * eye
            )
            try:
                step = np.linalg.solve(-hessian + 1e-10 * eye, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(-hessian + 1e-6 * eye, gradient, rcond=None)[0]
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta / self.scale_
        self.n_iter_ = _iteration + 1
        self.n_features_in_ = p
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise ValidationError("CoxPHRegressor.predict called before fit.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.coef_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"alpha": self.alpha, "max_iter": self.max_iter, "tol": self.tol}

    def set_params(self, **params: Any) -> "CoxPHRegressor":
        for key, value in params.items():
            setattr(self, key, value)
        return self


class _XGBSurvivalBase:
    """What the two XGBoost survival objectives share."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state=None,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.random_state = random_state

    def _tree_params(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            **self._tree_params(),
            "random_state": self.random_state,
        }


class XGBCoxSurvival(_XGBSurvivalBase):
    """
    XGBoost's `survival:cox` objective through `XGBRegressor`, which reads
    a NEGATIVE label as a right-censored duration. `predict` returns the
    hazard ratio: higher means sooner.
    """

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None):
        from xgboost import XGBRegressor

        duration, event = _split_labels(y)
        signed = np.where(event == 1.0, duration, -duration)
        self._model = XGBRegressor(
            objective="survival:cox",
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbosity=0,
            **self._tree_params(),
        )
        self._model.fit(np.asarray(X, dtype=float), signed, sample_weight=sample_weight)
        self.feature_importances_ = np.asarray(
            self._model.feature_importances_, dtype=float
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(X, dtype=float)), dtype=float)


class XGBAFTSurvival(_XGBSurvivalBase):
    """
    XGBoost's accelerated-failure-time objective, through the native API
    because the censoring is passed as label BOUNDS: an observed event has
    lower = upper = duration, a censored row lower = duration and upper =
    infinity. The booster predicts a time; `predict` returns its negative
    log, so higher means sooner like every other risk here.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        aft_loss_distribution: str = "normal",
        aft_loss_distribution_scale: float = 1.0,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )
        self.aft_loss_distribution = str(aft_loss_distribution)
        self.aft_loss_distribution_scale = float(aft_loss_distribution_scale)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            **super().get_params(deep),
            "aft_loss_distribution": self.aft_loss_distribution,
            "aft_loss_distribution_scale": self.aft_loss_distribution_scale,
        }

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None):
        import xgboost as xgb

        X = np.asarray(X, dtype=float)
        duration, event = _split_labels(y)
        matrix = xgb.DMatrix(X, weight=sample_weight)
        matrix.set_float_info("label_lower_bound", duration)
        matrix.set_float_info(
            "label_upper_bound", np.where(event == 1.0, duration, np.inf)
        )
        tree = self._tree_params()
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": self.aft_loss_distribution,
            "aft_loss_distribution_scale": self.aft_loss_distribution_scale,
            "max_depth": tree["max_depth"],
            "eta": tree["learning_rate"],
            "min_child_weight": tree["min_child_weight"],
            "subsample": tree["subsample"],
            "colsample_bytree": tree["colsample_bytree"],
            "alpha": tree["reg_alpha"],
            "lambda": tree["reg_lambda"],
            "seed": int(self.random_state or 0),
            "verbosity": 0,
        }
        self._booster = xgb.train(params, matrix, num_boost_round=self.n_estimators)
        gains = self._booster.get_score(importance_type="gain")
        importances = np.array(
            [float(gains.get(f"f{i}", 0.0)) for i in range(X.shape[1])], dtype=float
        )
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, X: Any) -> np.ndarray:
        import xgboost as xgb

        time = np.asarray(
            self._booster.predict(xgb.DMatrix(np.asarray(X, dtype=float))), dtype=float
        )
        return -np.log(np.maximum(time, 1e-12))


register_estimator("survival", "cox_ph", CoxPHRegressor, _COX_PH)


def _register_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    register_estimator("survival", "xgboost_cox", XGBCoxSurvival, _XGB_COX)
    register_estimator("survival", "xgboost_aft", XGBAFTSurvival, _XGB_AFT)
    return True


HAS_XGBOOST_SURVIVAL = _register_xgboost()

__all__ = [
    "HAS_XGBOOST_SURVIVAL",
    "CoxPHRegressor",
    "XGBAFTSurvival",
    "XGBCoxSurvival",
]
