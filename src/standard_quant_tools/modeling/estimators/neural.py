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
"""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier, MLPRegressor

from .bounds import EstimatorParamSchema, ParamBound
from .registry import register_estimator

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
        "OFF by default here, unlike sklearn. Early stopping holds out a "
        "RANDOM slice of the training window, and on an overlapping label a "
        "held-out row shares most of its bars with rows still being trained "
        "on -- so the stopping signal is optimistic and stops late. It is "
        "not lookahead against the test fold, which the engine's purge and "
        "embargo still protect, but it is not the honest validation curve "
        "it looks like."
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


class PanelMLPRegressor(MLPRegressor):
    """MLPRegressor whose architecture is two bounded integers."""

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

    def fit(self, X, y, **kwargs):
        # `clone` rebuilds from get_params, which carries the two scalars
        # and not the tuple, so the tuple is rederived here rather than
        # trusted from __init__ -- a cloned estimator would otherwise fit
        # sklearn's default width regardless of what the spec asked for.
        self.hidden_layer_sizes = _sizes(self.n_hidden_units, self.n_hidden_layers)
        return super().fit(X, y, **kwargs)


class PanelMLPClassifier(MLPClassifier):
    """MLPClassifier whose architecture is two bounded integers."""

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

    def fit(self, X, y, **kwargs):
        self.hidden_layer_sizes = _sizes(self.n_hidden_units, self.n_hidden_layers)
        return super().fit(X, y, **kwargs)


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

register_estimator("regression", "mlp", PanelMLPRegressor, _MLP_SCHEMA)
register_estimator("classification", "mlp", PanelMLPClassifier, _MLP_SCHEMA)

__all__ = [
    "N_HIDDEN_LAYERS",
    "N_HIDDEN_UNITS",
    "PanelMLPClassifier",
    "PanelMLPRegressor",
]
