"""
The non-linear-over-a-window estimator, which is what "sequence model"
means once the window is in the columns.

WHY THIS AND NOT A TCN. `engine.py` hands an estimator a 2-D X whose rows
are (date, entity) observations carrying no entity identity -- the contract
that lets ridge, LightGBM and this be interchangeable. A recurrent or
convolutional model cannot reconstruct per-entity sequences from that, so
the window has to arrive as columns either way (`FeatureSpec.lags`). Once
it has, what remains for a sequence architecture to add is weight sharing
ACROSS lag positions, and that pays at hundreds of timesteps and thousands
of series -- not at the depth a daily panel supports. An MLP over the lag
columns is the same hypothesis class without a dependency, and it fits in
the walk-forward loop that already exists rather than beside it.

WHAT THIS NEEDS THAT TREES DO NOT. Gradient descent on unscaled inputs is
dominated by whichever column happens to be largest. The engine already
winsorizes at the 1st/99th percentile and z-scores, fitting those statistics
on each fold's TRAINING rows only, so the scaling this needs is present and
is not refit on test. That is why an MLP can be registered here at all
without carrying its own preprocessing.

THE ARCHITECTURE IS TWO SCALARS, NOT A TUPLE. sklearn takes
`hidden_layer_sizes` as a tuple, which the parameter allowlist cannot bound
-- and an unbounded tuple is exactly the resource-exhaustion path
`bounds.py` exists to close. Width and depth are exposed as bounded
integers instead, and the tuple is built from them.

EARLY STOPPING IS TIME-ORDERED. scikit-learn's `early_stopping=True` sets
aside a SHUFFLED `validation_fraction` of the training rows (stratified for
the classifier), dated among the rows it fits, whose overlapping labels
share their outcomes. The engine hands these estimators the training
window's last dates instead, with the label horizon embargoed before them,
as `fit(X_val=, y_val=, sample_weight_val=)` -- the block histogram
boosting stops on (see `trees.prepare_early_stopping`). scikit-learn's MLP
takes no validation set, so `_TimeOrderedEarlyStopping` hands it one.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils import check_array

from .bounds import EstimatorParamSchema, ParamBound
from .registry import EstimatorCost, register_estimator

#: Neurons per hidden layer. The ceiling is a compute budget: this fits
#: once per fold, and a walk-forward run does that many times over.
N_HIDDEN_UNITS = ParamBound(
    "int",
    1,
    512,
    note="Neurons per hidden layer; the tuple sklearn wants is built from "
    "this and n_hidden_layers.",
)

#: Depth. Past three, a panel of this size is fitting noise, and the extra
#: layers cost fold time that walk-forward multiplies.
N_HIDDEN_LAYERS = ParamBound("int", 1, 3)

_LEARNING_RATE_INIT = ParamBound(
    "float",
    1e-6,
    1.0,
    note="Adam's initial step. Too large and the fit diverges differently "
    "on every fold, which reads as instability in the model rather than in "
    "the optimizer.",
)

_ALPHA = ParamBound(
    "float",
    0.0,
    1e6,
    note="L2 penalty. The main defence against a network memorising a "
    "training window, which on overlapping labels it can do convincingly.",
)

_MAX_ITER = ParamBound("int", 1, 10_000)

_EARLY_STOPPING = ParamBound(
    "bool",
    note=(
        "Off by default. True stops training once the score on the training "
        "window's last 10% of dates (R2 for regression, accuracy for "
        "classification) has not beaten its best by 1e-4 for more than 10 "
        "epochs in a row, and keeps the weights of the best epoch. The rows "
        "whose labels end on or after the first of those dates are left out "
        "of the fit, and so is every row on the label horizon's dates before "
        "them. A window too short to hold that block is refused. "
        "scikit-learn's own rule, a shuffled 10% of the rows dated among the "
        "rows fitted, is not used."
    ),
)

_RANDOM_STATE = ParamBound(
    "int",
    0,
    2**31 - 1,
    allow_none=True,
    note="Weight initialisation is random, so without this two runs of the "
    "same spec differ and nothing in the manifest explains why.",
)


def _sizes(n_hidden_units: int, n_hidden_layers: int):
    return tuple([int(n_hidden_units)] * int(n_hidden_layers))


#: The methods of scikit-learn's MLP a validation set is handed through,
#: each present with the same arguments from 1.3 to 1.9: `_initialize` sets
#: up a fit's state, `_fit_stochastic` runs its epochs, and
#: `_update_no_improvement_count` scores each epoch.
_HOOKS = ("_initialize", "_fit_stochastic", "_update_no_improvement_count")

_STOCHASTIC_SOLVERS = ("adam", "sgd")


def _scoring_arguments() -> "tuple | None":
    """The arguments after `self` of scikit-learn's per-epoch scoring --
    (early_stopping, X_val, y_val) before 1.7, (early_stopping, X, y,
    sample_weight) from 1.7 -- or None when this scikit-learn's MLP lacks
    one of the three methods or takes something else."""
    if not all(callable(getattr(MLPRegressor, name, None)) for name in _HOOKS):
        return None
    try:
        signature = inspect.signature(MLPRegressor._update_no_improvement_count)
    except (TypeError, ValueError):
        return None
    names = tuple(signature.parameters)[1:]
    if len(names) not in (3, 4) or names[0] != "early_stopping":
        return None
    return names


def takes_validation_set() -> bool:
    """Whether this scikit-learn's MLP can be handed a validation set
    through `_TimeOrderedEarlyStopping`. Read off the methods rather than
    the version string, as `trees.fit_takes_validation_set` is."""
    return _scoring_arguments() is not None


class _TimeOrderedEarlyStopping:
    """
    `fit(X, y, X_val=, y_val=, sample_weight_val=)` for scikit-learn's MLP:
    with `early_stopping=True` it stops on the rows handed in, by
    scikit-learn's own rule, instead of on a shuffled share of `X`.

    HOW. scikit-learn's fit reads `early_stopping` in three places: to set
    the state the scoring starts from (`_initialize`), to cut a shuffled
    `validation_fraction` from X before the epochs (`_fit_stochastic`), and
    to put the best epoch's weights back after them. The rule itself -- the
    score after each epoch (R2 or accuracy, weighted from 1.7), `tol`, the
    `n_iter_no_change` patience, and the copy of the best epoch's weights
    -- is the early-stopping branch of `_update_no_improvement_count`,
    handed the validation rows. So the fit runs with `early_stopping` off
    for scikit-learn (no split: every row of X is trained on), each epoch's
    scoring is sent to that branch with the rows handed in, and the
    starting state and the restore are done as scikit-learn does them. The
    epochs, the optimizer, the random stream (less the split's draws) and
    the rule are scikit-learn's; which rows are scored is not.
    `validation_scores_`, `best_validation_score_`, `n_iter_`,
    `loss_curve_` and the weights come out as scikit-learn's early
    stopping leaves them when given the same split, bit for bit (a test
    holds the two to it on the split scikit-learn draws).

    WHY NOT `partial_fit`, one epoch a call. It re-seeds an integer
    `random_state` on every call, so each epoch would shuffle the rows with
    a fresh stream where `fit` continues one; it refuses
    `early_stopping=True`; and the rule would have to be written a second
    time here.

    Without a validation set the fit is scikit-learn's, untouched: the rows
    handed in live on the estimator only while it fits, so nothing is left
    behind that a plain fit does not leave.
    """

    #: The registry's name for these estimators, for messages.
    estimator_name = "mlp"
    #: scikit-learn refuses an early-stopping split of fewer validation
    #: rows (from 1.7; an R2 on one row is not defined).
    min_validation_rows = 2

    @staticmethod
    def takes_validation_set() -> bool:
        return takes_validation_set()

    def fit(self, X, y, *, X_val=None, y_val=None, sample_weight_val=None, **kwargs):
        # `clone` rebuilds from get_params, which carries the two scalars
        # and not the tuple, so the tuple is rederived here rather than
        # trusted from __init__ -- a cloned estimator would otherwise fit
        # sklearn's default width regardless of what the spec asked for.
        self.hidden_layer_sizes = _sizes(self.n_hidden_units, self.n_hidden_layers)
        if X_val is None and y_val is None and sample_weight_val is None:
            return super().fit(X, y, **kwargs)
        held_out = self._validation_set(X_val, y_val, sample_weight_val)
        setting = self.early_stopping
        self._held_out = held_out
        self.early_stopping = False
        try:
            super().fit(X, y, **kwargs)
        finally:
            self.early_stopping = setting
            del self._held_out
        scores = getattr(self, "validation_scores_", None)
        if scores is None or len(scores) != self.n_iter_:
            raise RuntimeError(
                f"{type(self).__name__}: scikit-learn's MLP ran {self.n_iter_} "
                f"epoch(s) and scored {0 if scores is None else len(scores)} on "
                "the validation set; this scikit-learn no longer scores each "
                "epoch through the methods the validation set is handed "
                "through. Set early_stopping=False."
            )
        return self

    def _validation_set(self, X_val, y_val, sample_weight_val):
        """The validation arguments as the epochs score them, refused with
        a reason where scikit-learn's own early stopping could not run."""
        name = type(self).__name__
        if X_val is None or y_val is None:
            raise ValueError(f"{name}: X_val and y_val are given together.")
        setting = self.early_stopping
        if not (isinstance(setting, (bool, np.bool_)) and setting):
            raise ValueError(
                f"{name}: X_val is the validation set early stopping is scored "
                f"on, and early_stopping is {setting!r}. Set "
                "early_stopping=True, or drop X_val."
            )
        if self.solver not in _STOCHASTIC_SOLVERS or self.warm_start:
            raise ValueError(
                f"{name}: a validation set stops a fresh fit by one of the "
                f"stochastic solvers {list(_STOCHASTIC_SOLVERS)}; this one has "
                f"solver={self.solver!r} and warm_start={self.warm_start!r}."
            )
        arguments = _scoring_arguments()
        if arguments is None:
            raise TypeError(
                f"{name}: this scikit-learn's MLP lacks the methods a "
                f"validation set is handed through ({', '.join(_HOOKS)}), so "
                "it cannot stop on one. Set early_stopping=False."
            )
        weighted = len(arguments) == 4
        X_val = check_array(
            X_val, accept_sparse=["csr", "csc"], dtype=(np.float64, np.float32)
        )
        y_val = check_array(y_val, ensure_2d=False, dtype=None)
        if y_val.ndim == 2 and y_val.shape[1] == 1:
            y_val = y_val.ravel()
        if y_val.shape[0] != X_val.shape[0]:
            raise ValueError(
                f"{name}: X_val has {X_val.shape[0]} rows and y_val "
                f"{y_val.shape[0]}."
            )
        if X_val.shape[0] < self.min_validation_rows:
            raise ValueError(
                f"{name}: the validation set has {X_val.shape[0]} row(s); "
                f"early stopping scores at least {self.min_validation_rows}."
            )
        if not isinstance(self, MLPClassifier):
            # The column scikit-learn scores its own split's targets as.
            if y_val.dtype.kind == "O":
                y_val = y_val.astype(np.float64)
            if y_val.ndim == 1:
                y_val = y_val.reshape((-1, 1))
        if sample_weight_val is not None:
            if not weighted:
                raise TypeError(
                    f"{name}: sample_weight_val needs scikit-learn 1.7 or "
                    "later, whose MLP takes sample weights."
                )
            from sklearn.utils.validation import _check_sample_weight

            sample_weight_val = _check_sample_weight(sample_weight_val, X_val)
        return X_val, y_val, sample_weight_val, weighted

    def _initialize(self, y, layer_units, dtype):
        super()._initialize(y, layer_units, dtype)
        held_out = self.__dict__.get("_held_out")
        if held_out is None:
            return
        X_val, y_val, sample_weight_val, weighted = held_out
        if X_val.shape[1] != layer_units[0]:
            raise ValueError(
                f"{type(self).__name__}: X_val has {X_val.shape[1]} features "
                f"and X has {layer_units[0]}."
            )
        # Scored in the dtype X was validated to, as a split of X would be.
        self._held_out = (
            X_val.astype(dtype, copy=False),
            y_val,
            sample_weight_val,
            weighted,
        )
        # The state scikit-learn's own early stopping starts from.
        self.validation_scores_ = []
        self.best_validation_score_ = -np.inf
        self.best_loss_ = None

    def _update_no_improvement_count(self, early_stopping, *args, **kwargs):
        held_out = self.__dict__.get("_held_out")
        if held_out is None:
            return super()._update_no_improvement_count(early_stopping, *args, **kwargs)
        X_val, y_val, sample_weight_val, weighted = held_out
        if weighted:
            return super()._update_no_improvement_count(
                True, X_val, y_val, sample_weight_val
            )
        return super()._update_no_improvement_count(True, X_val, y_val)

    def _fit_stochastic(self, *args, **kwargs):
        super()._fit_stochastic(*args, **kwargs)
        if self.__dict__.get("_held_out") is not None:
            # Restore the best epoch's weights, as scikit-learn's early
            # stopping does at the end of its epochs.
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def best_iteration(self) -> "int | None":
        """The epoch whose weights an early-stopped fit kept: the first to
        reach the best validation score, since scikit-learn replaces them
        only on a strictly higher one; 0, the initial weights, when no
        epoch scored a number. None for a fit that did not stop early."""
        scores = getattr(self, "validation_scores_", None)
        if not scores:
            return None
        best, epoch = -np.inf, 0
        for i, score in enumerate(scores, start=1):
            if score > best:
                best, epoch = score, i
        return epoch


class PanelMLPRegressor(_TimeOrderedEarlyStopping, MLPRegressor):
    """MLPRegressor whose architecture is two bounded integers, and whose
    early stopping can be handed its validation rows."""

    def __init__(
        self,
        n_hidden_units: int = 64,
        n_hidden_layers: int = 1,
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        early_stopping: bool = False,
        random_state=None,
    ):
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        super().__init__(
            hidden_layer_sizes=_sizes(n_hidden_units, n_hidden_layers),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )


class PanelMLPClassifier(_TimeOrderedEarlyStopping, MLPClassifier):
    """MLPClassifier whose architecture is two bounded integers, and whose
    early stopping can be handed its validation rows."""

    def __init__(
        self,
        n_hidden_units: int = 64,
        n_hidden_layers: int = 1,
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        early_stopping: bool = False,
        random_state=None,
    ):
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        super().__init__(
            hidden_layer_sizes=_sizes(n_hidden_units, n_hidden_layers),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )


_MLP_SCHEMA = EstimatorParamSchema(
    bounds={
        "n_hidden_units": N_HIDDEN_UNITS,
        "n_hidden_layers": N_HIDDEN_LAYERS,
        "alpha": _ALPHA,
        "learning_rate_init": _LEARNING_RATE_INIT,
        "max_iter": _MAX_ITER,
        "early_stopping": _EARLY_STOPPING,
        "random_state": _RANDOM_STATE,
    }
)

# One default fit on a 15,030-row, 8-feature window, 16 logical cores:
# 0.82 s regression, 4.6 s classification (log loss converges in more
# iterations than squared error on this label). See EstimatorCost. An
# epoch costs the same with early_stopping=True, so the fit time follows
# the epoch count: on the live panel's folds of 14,880 to 14,970 rows the
# window's last dates stopped the regressor after 13 to 50 epochs, where
# scikit-learn's shuffled split ran 31 to 96 (seeds 7 and 42).
register_estimator(
    "regression",
    "mlp",
    PanelMLPRegressor,
    _MLP_SCHEMA,
    cost=EstimatorCost("medium", "one"),
)
register_estimator(
    "classification",
    "mlp",
    PanelMLPClassifier,
    _MLP_SCHEMA,
    cost=EstimatorCost("high", "one"),
)

__all__ = [
    "N_HIDDEN_LAYERS",
    "N_HIDDEN_UNITS",
    "PanelMLPClassifier",
    "PanelMLPRegressor",
]
