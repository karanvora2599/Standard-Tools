"""
Delta One: the economics that connect one instrument to another.

WHAT THIS IS FOR. Pricing, relative value, hedging, replication, carry,
financing and execution of instruments whose value moves approximately
one-for-one with an underlying equity or index -- cash, ETFs, baskets,
futures, forwards, synthetic forwards, total return swaps.

WHY IT IS NOT `derivatives`. That package answers "what is this contract
worth and what does holding it do to me", and its subject is one convex
instrument. This one answers "which instrument is the cheapest way to own
or hedge this exposure", and its subject is the relationship between
several linear ones. The two meet at exactly one place, and it is a
feature: put-call parity produces a synthetic forward, and that forward is
one row in a comparison of ways to hold the position.

NOTHING HERE FETCHES. Every function takes its quotes, its contract
specifications and its dividend schedule as arguments. This library has no
futures data provider, no index-constituent source and no dividend
calendar, and a module that pretended otherwise would compute a curve that
does not exist. It is the same call `analysis/derivatives.py` made about
option chains, and it has the same side benefit: the functions work on a
hypothetical curve, which is most of what they are used for.

THE MODULES ARE IMPORTED BY PATH. `analysis/__init__.py` stopped
re-exporting its newer modules and this package follows that, so import
`standard_quant_tools.delta_one.carry` rather than expecting names here.
"""

from __future__ import annotations

__all__: list[str] = []
