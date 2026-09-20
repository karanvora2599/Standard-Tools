"""
The supervised tasks this library fits, declared ONCE, in a leaf module.

It was written five times, in two widths: three copies said
regression/classification/ranking and two said regression/classification.
The narrow pair was not a different opinion, it was a place ranking had
been forgotten -- which is exactly the drift a repeated literal produces
and the reason this name exists.

A leaf, with no imports of its own, because both `specs` and the target
registry need it and the registry must not import `specs`: the built-in
targets register at import, `specs` reads the registry to validate a
`TargetSpec`, and a module that both sides import cannot depend on either.
"""

from typing import Literal

#: `survival` fits a DURATION that may be right-censored: the label is
#: how long until an event -- a fill, a default, a barrier touch -- with
#: an indicator saying whether the event was observed or the window ended
#: first. A regression on that label reads every censored row as an event
#: at the horizon, which is the bias the task exists to remove.
TASKS = ("regression", "classification", "ranking", "survival")
Task = Literal["regression", "classification", "ranking", "survival"]

#: Tasks whose prediction is a CONTINUOUS SCORE rather than a probability.
#: A ranker emits a relative score exactly as a regressor emits a
#: magnitude, so everything downstream that asks "which side is this" reads
#: the sign of both the same way. A survival model emits a RISK score --
#: higher means the event sooner -- which is a continuous ordering too,
#: though of a different question, which is why an ensemble refuses to
#: average it with a return forecast. Classification is the odd one out,
#: being bounded in [0, 1] with a decision boundary in the middle.
SCORE_TASKS = ("regression", "ranking", "survival")

__all__ = ["SCORE_TASKS", "TASKS", "Task"]
