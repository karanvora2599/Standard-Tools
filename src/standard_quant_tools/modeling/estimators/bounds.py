"""
Value bounds and cross-parameter compatibility for the estimator allowlist.

The registry already refused unknown estimator NAMES and unknown parameter
NAMES, but placed no constraint on parameter VALUES. That left two distinct
problems:

1. **An agent-triggerable resource exhaustion path.** `n_estimators`,
   `max_iter` and `max_depth` were unbounded, so a single tool call could
   request `n_estimators=10_000_000` and pin CPU and memory for as long as
   sklearn kept trying. An allowlist of names is not a compute budget.

2. **Incompatible combinations failing deep inside sklearn.** `penalty` was
   exposed for logistic regression without `solver`, so `penalty="l1"`
   reached LogisticRegression's default lbfgs solver — which does not
   support it — and raised from inside sklearn rather than at the modeling
   boundary with an actionable message.

Bounds are deliberately generous: they exist to keep a runaway request from
taking the process down, not to express an opinion about good
hyperparameters. Anything a real research workflow would ask for fits
comfortably inside them.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from standard_quant_tools.error import ValidationError


@dataclass(frozen=True)
class ParamBound:
    """One parameter's accepted type and range."""

    kind: str  # "int" | "float" | "bool" | "str"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    allow_none: bool = False
    note: str = ""

    def validate(self, estimator: str, name: str, value: Any) -> None:
        if value is None:
            if self.allow_none:
                return
            raise ValidationError(
                f"estimator {estimator!r}: parameter {name!r} may not be None."
            )

        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValidationError(
                    f"estimator {estimator!r}: parameter {name!r} must be a bool, "
                    f"got {value!r} ({type(value).__name__})."
                )
            return

        if self.kind == "str":
            if not isinstance(value, str):
                raise ValidationError(
                    f"estimator {estimator!r}: parameter {name!r} must be a string, "
                    f"got {value!r} ({type(value).__name__})."
                )
            if self.choices is not None and value not in self.choices:
                raise ValidationError(
                    f"estimator {estimator!r}: parameter {name!r}={value!r} is not one of "
                    f"{sorted(c for c in self.choices if c is not None)}."
                )
            return

        # Numeric. bool is a subclass of int, so exclude it explicitly --
        # True would otherwise pass as a valid n_estimators of 1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                f"estimator {estimator!r}: parameter {name!r} must be a number, "
                f"got {value!r} ({type(value).__name__})."
            )
        if not math.isfinite(float(value)):
            raise ValidationError(
                f"estimator {estimator!r}: parameter {name!r} must be finite, got {value!r}."
            )
        if self.kind == "int" and not float(value).is_integer():
            raise ValidationError(
                f"estimator {estimator!r}: parameter {name!r} must be a whole number, "
                f"got {value!r}."
            )
        numeric = float(value)
        if self.minimum is not None and numeric < self.minimum:
            raise ValidationError(
                f"estimator {estimator!r}: parameter {name!r}={value!r} is below the "
                f"minimum {self.minimum}." + (f" {self.note}" if self.note else "")
            )
        if self.maximum is not None and numeric > self.maximum:
            raise ValidationError(
                f"estimator {estimator!r}: parameter {name!r}={value!r} exceeds the "
                f"maximum {self.maximum}."
                + (f" {self.note}" if self.note else "")
                + " This ceiling is a resource budget, not a modelling opinion — an "
                "unbounded value here lets a single call exhaust CPU and memory."
            )


# ── Shared bounds ───────────────────────────────────────────────────────
# Ceilings sized so any realistic research request passes, while a runaway
# one is rejected before sklearn allocates anything.

N_ESTIMATORS = ParamBound(
    "int", 1, 2_000, note="Ensembles above ~2000 trees are a compute budget concern."
)
MAX_ITER = ParamBound("int", 1, 100_000)
MAX_DEPTH = ParamBound(
    "int",
    1,
    64,
    allow_none=True,
    note="None means unlimited depth (sklearn's default).",
)
LEARNING_RATE = ParamBound("float", 1e-6, 10.0)
ALPHA = ParamBound("float", 0.0, 1e9)
L1_RATIO = ParamBound("float", 0.0, 1.0)
C_BOUND = ParamBound("float", 1e-9, 1e9)
FIT_INTERCEPT = ParamBound("bool")

# LogisticRegression solver/penalty compatibility, straight from sklearn's
# own matrix. Exposed so a caller CAN use l1/elasticnet, rather than being
# silently restricted -- but validated here instead of failing inside .fit().
_LOGISTIC_SOLVER_PENALTIES: Dict[str, Tuple[Optional[str], ...]] = {
    "lbfgs": ("l2", None),
    "newton-cg": ("l2", None),
    "sag": ("l2", None),
    "liblinear": ("l1", "l2"),
    "saga": ("l1", "l2", "elasticnet", None),
}


def _logistic_compatibility(params: Dict[str, Any]) -> None:
    solver = params.get("solver", "lbfgs")
    penalty = params.get("penalty", "l2")
    supported = _LOGISTIC_SOLVER_PENALTIES.get(solver)
    if supported is None:
        raise ValidationError(
            f"estimator 'logistic': solver={solver!r} is not supported — allowed: "
            f"{sorted(_LOGISTIC_SOLVER_PENALTIES)}."
        )
    if penalty not in supported:
        raise ValidationError(
            f"estimator 'logistic': penalty={penalty!r} is not supported by "
            f"solver={solver!r}. Compatible penalties for this solver: "
            f"{sorted(str(p) for p in supported)}. "
            f"(penalty='l1' needs solver='liblinear' or 'saga'; "
            f"penalty='elasticnet' needs solver='saga'.)"
        )
    if penalty == "elasticnet" and "l1_ratio" not in params:
        raise ValidationError(
            "estimator 'logistic': penalty='elasticnet' also requires l1_ratio."
        )


@dataclass(frozen=True)
class EstimatorParamSchema:
    """Per-parameter bounds plus optional cross-parameter checks."""

    bounds: Dict[str, ParamBound] = field(default_factory=dict)
    compatibility: Tuple[Callable[[Dict[str, Any]], None], ...] = ()

    def validate(self, estimator: str, params: Dict[str, Any]) -> None:
        unknown = sorted(set(params) - set(self.bounds))
        if unknown:
            raise ValidationError(
                f"estimator {estimator!r} does not accept params {unknown} — "
                f"allowed: {sorted(self.bounds)}"
            )
        for name, value in params.items():
            self.bounds[name].validate(estimator, name, value)
        for check in self.compatibility:
            check(params)

    @property
    def allowed_names(self) -> List[str]:
        return sorted(self.bounds)


LOGISTIC_SCHEMA = EstimatorParamSchema(
    bounds={
        "C": C_BOUND,
        "penalty": ParamBound(
            "str", choices=("l1", "l2", "elasticnet"), allow_none=True
        ),
        "solver": ParamBound("str", choices=tuple(_LOGISTIC_SOLVER_PENALTIES)),
        "l1_ratio": L1_RATIO,
        "max_iter": MAX_ITER,
        "fit_intercept": FIT_INTERCEPT,
    },
    compatibility=(_logistic_compatibility,),
)
