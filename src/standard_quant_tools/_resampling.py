"""
Moving-block resampling, built once.

WHY THIS FILE EXISTS. Four places drew moving-block bootstrap samples, all
by the same scheme -- pick `ceil(target / block_size)` starts uniformly from
`[0, n - block_size]`, lay the blocks end to end, truncate to `target` --
and three of them built the result the slow way:

    np.concatenate([np.arange(s, s + block_size) for s in starts])[:target]

`analysis/inference.py` had already measured that. Its docstring records the
finding: for one default 2,000-draw call on 2,500 observations, the
comprehension ran 358,000 separate `np.arange` calls and profiled at 87% of
`bootstrap_statistic`'s total runtime, against 11% for the statistic
actually being bootstrapped. It replaced its own copy with a broadcast and
left the other three alone, because nothing connected them.

Re-measured across the shapes these callers use, the broadcast is 9x to 17x
faster and returns BIT-IDENTICAL indices from the same seed -- it makes the
same `rng.integers` call with the same arguments, so every existing seeded
result is unchanged. That is what makes this worth doing rather than
interesting: there is no tradeoff to weigh.

    n=2500 block=13 draws=2000    343 ms -> 22 ms   15.6x
    n=1000 block=10 draws=2000    179 ms -> 20 ms    9.0x
    n=5000 block=20 draws=1000    212 ms -> 12 ms   17.1x

WHY BLOCKS AT ALL. An iid bootstrap destroys serial correlation, and every
statistic these callers bootstrap -- Sharpe ratios, drawdowns, reality-check
maxima -- depends on it. Resampling blocks keeps the local dependence
structure inside each block, at the cost of breaking it at the seams.
"""

from __future__ import annotations

import math

import numpy as np

from standard_quant_tools.numeric_contract import require_positive_int

__all__ = ["block_indices"]


def block_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
    target: int | None = None,
) -> np.ndarray:
    """
    Indices for one moving-block resample of length `target`.

    `n` is the length of the source series; `target` defaults to `n`, and is
    given separately for the simulation callers that resample a historical
    series of one length into a forward horizon of another.

    BUILT BY BROADCAST, not by concatenating a per-block list. See the module
    docstring for the measurement; the short version is that this is called
    once per draw and the index construction, which nobody profiles, was
    costing more than the statistic.

    `span` is clamped because `block_size` may exceed `n`. In that case the
    concatenating version drew every start at 0 and truncated, so the clamp
    reproduces it rather than changing it; below the clamp every start
    satisfies `s + span <= n` by construction, so no block runs off the end
    and the truncation is the only thing trimming.

    `n`, `block_size` and `target` must all be positive integers. A
    `block_size` larger than `n` is fine and is clamped to `n` -- see
    `span` above -- but one below 1 is refused rather than clamped up,
    because silently returning an IID resample under this function's name
    destroys exactly the serial correlation a caller chose it to preserve.

    A `block_size` of 1 needs no special case: `span` is 1, `n_blocks` is
    `target`, and the call becomes `rng.integers(0, n, target)` -- exactly
    the iid draw, from exactly the same call, so the random stream matches
    an explicit iid branch draw for draw.
    """
    if target is None:
        target = n

    # Both of these used to be silent. `n = 0` divided by zero and raised a
    # bare ZeroDivisionError from inside the ceil, and `block_size` of 0 or
    # -3 was clamped up to 1 -- which is an IID resample, returned under the
    # name of a block bootstrap. The docstring below reasons carefully about
    # why a block_size of exactly 1 is safe and then said nothing about the
    # values that are not.
    n = require_positive_int(n, "n", "block_indices")
    block_size = require_positive_int(block_size, "block_size", "block_indices")
    target = require_positive_int(target, "target", "block_indices")

    span = min(block_size, n)
    n_blocks = int(math.ceil(target / span))
    starts = rng.integers(0, n - span + 1, n_blocks)
    return (starts[:, None] + np.arange(span)).ravel()[:target]
