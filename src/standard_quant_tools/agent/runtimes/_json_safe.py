"""
The one place a non-finite number is turned into a null.

This helper was written fourteen times, once per runtime module, each copy
private and each used exactly once. Nothing was dead and nothing was wrong,
which is why it survived: fourteen correct three-line functions look like
tidy local style rather than a problem.

They had already drifted into four spellings. One tested `isinstance(value,
float)`, another added an `int`/`bool` branch that cannot change the answer
because an integer is always finite, and two carried the docstring
explaining WHY any of it is necessary while twelve did not. That is the
whole cost of the duplication: the reasoning lived in two copies out of
fourteen, so twelve readers met a bare `isinstance` check with no way to
know what it was defending against, and the next person to touch one of them
had a one-in-seven chance of finding the explanation.

`modeling.agent.feature_models` keeps its own, deliberately. It looks like a
fifteenth copy and is not the same function: it coerces with `float()` and
swallows `TypeError`/`ValueError`, so a string reaches it and leaves as
`None`, where this one passes it straight through. Merging them would have
been a silent behaviour change in whichever direction it went.
"""

from __future__ import annotations

import math
from typing import Any


def finite_or_none(value: Any) -> Any:
    """
    Non-finite in, null out.

    Applied before validation so a NaN never reaches the serializer. The
    alternative -- letting it through and relying on the JSON encoder --
    produces `NaN` in the payload, which is not valid JSON and which several
    MCP clients reject at the transport layer rather than at the tool. A
    rejection there is much worse than a null: it fails the whole response
    rather than the one field that could not be computed, and it fails it
    somewhere the agent cannot see or act on.

    Anything that is not a float passes through untouched. Integers are
    always finite, and a value this does not recognise is not this
    function's to reinterpret.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
