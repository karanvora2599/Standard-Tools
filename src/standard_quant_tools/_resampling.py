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

TWO SCHEMES, AND WHY THE SECOND IS NOT THE FIRST. `block_indices` is the
moving-block draw: a block starts anywhere in `[0, n - block_size]`, so the
first and last `block_size - 1` observations are reached through fewer
starts than the middle ones -- the first observation only by a block that
starts on it. `circular_block_indices` (Politis and Romano 1992) wraps the
series into a circle, so every start is legal and every observation is
drawn equally often. That is the property a test that CENTRES its null on
the sample mean needs, and `reality_check` is that test: on 120 days with
20-day blocks the moving draw left its null too narrow and it rejected a
true null 9-12% of the time at a nominal 5%.

The two confidence intervals took the circular draw next, for the same
reason: `bootstrap_statistic` and `block_bootstrap_ci` read a percentile
interval off draws that should be centred on the sample's own statistic,
and on short series their nominal 95% intervals covered the truth about
92% of the time with the default blocks and 76-87% with 20-day ones (see
`circular_block_scale`). Each asks for the circular draw itself. The
shared `block_indices` is unchanged, deliberately: its remaining callers
draw forward paths, which have no centre to keep, and the native
forward-path kernel in `monte_carlo.cpp` reproduces it bit for bit, so
changing it would move every seeded simulation and part it from its twin.

A BLOCK BOOTSTRAP ALSO UNDERSTATES THE VARIANCE OF A MEAN, whichever way
it draws; `circular_block_scale` is the factor that corrects it.
"""

from __future__ import annotations

import math

import numpy as np

from standard_quant_tools.numeric_contract import require_positive_int

__all__ = ["block_indices", "circular_block_indices", "circular_block_scale"]


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


def circular_block_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
    target: int | None = None,
) -> np.ndarray:
    """
    Indices for one circular-block resample of length `target`.

    The moving-block draw of `block_indices` with the series bent into a
    circle: a start is drawn from all `n` positions and a block that runs
    past the last observation continues from the first. Every observation
    is then reached by exactly `block_size` starts, so each is drawn equally
    often and the expected resampled mean IS the sample mean -- which the
    moving draw's is not. See the module docstring for who uses which.

    The seam the wrap adds, last observation to first, is one more seam of
    the kind every block bootstrap already has between its blocks.

    Arguments are refused and clamped exactly as `block_indices` refuses
    and clamps them.
    """
    if target is None:
        target = n
    n = require_positive_int(n, "n", "circular_block_indices")
    block_size = require_positive_int(
        block_size, "block_size", "circular_block_indices"
    )
    target = require_positive_int(target, "target", "circular_block_indices")

    span = min(block_size, n)
    n_blocks = int(math.ceil(target / span))
    starts = rng.integers(0, n, n_blocks)
    return ((starts[:, None] + np.arange(span)) % n).ravel()[:target]


def circular_block_scale(n: int, block_size: int) -> float:
    """
    sqrt(n / (n - block_size)): what a circular block bootstrap's draws are
    widened by, about their own centre, so that their spread is the
    sampling spread they estimate.

    A block bootstrap's variance of a mean is the Bartlett sum of the
    autocovariances taken about the SAMPLE mean, and with no dependence at
    all its expectation is sigma^2 (1 - b/n) / n: 17% short at n = 120 and
    b = 20, 4% at the n^(1/3) default block of 5. The factor is Bessel's
    correction at b = 1. It is derived for a mean and holds to first order
    for a smooth function of means -- a Sharpe ratio, a standard deviation,
    a quantile through its influence function.

    Measured on nominal 95% percentile intervals of a mean, 1,000 normal
    series per cell, IID / AR(0.3), with the n^(1/3) default block:

        n                       60           120          250
        moving draw          92.9 / 89.6  92.4 / 90.1  93.3 / 91.4  (%)
        circular, widened    94.0 / 91.2  93.4 / 91.6  94.2 / 92.2

    and with 20-day blocks, a mean or a Sharpe ratio alike:

        moving draw          76 / 73      87 / 86      90 / 90      (%)
        circular, widened    90 / 88      91 / 90      93 / 93

    What is left is a variance read off n / b blocks, which has few degrees
    of freedom when the blocks are long, and under AR(0.3) the n^(1/3)
    block, which is too short to hold all of the dependence.

    1.0 when the block is the whole series: there is nothing to widen.
    """
    return math.sqrt(n / (n - block_size)) if block_size < n else 1.0
