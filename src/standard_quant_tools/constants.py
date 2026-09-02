"""
The numbers that were the same in nine places.

WHY THIS FILE EXISTS. `252` was defined nine times across `analysis`,
`backtesting`, `delta_one` and `portfolio`, under two spellings --
`TRADING_DAYS` eight times and `TRADING_DAYS_PER_YEAR` once -- and the one
that named itself canonically had no importers at all. One of the eight, in
`analysis/microstructure_estimators.py`, was never read after being defined.

Nothing had gone wrong yet. Every copy said 252 and they all still say 252,
so this changes no result anywhere. The point is that nine independent
definitions of an annualization factor is nine places to edit when a caller
wants 365 for crypto or 260 for a five-day calendar, and eight of them would
be missed. A convention this load-bearing -- it is the square root of this
number that turns a daily volatility into an annual one, in every Sharpe
ratio in the library -- should have exactly one definition.

THE OLD NAMES SURVIVE. Five of the eight were in their module's `__all__`,
so `from standard_quant_tools.analysis.inference import TRADING_DAYS` is
public API and still resolves. Each module keeps its name as an alias of
this one.

THIS IS A DEFAULT, NOT A LAW. It is the count of US equity trading days in
a year, and it is wrong for anything else. Functions that annualize take a
`periods_per_year` argument for exactly that reason, and this is only what
they fall back to. `numeric_contract.require_periods_per_year` validates
what a caller passes instead.
"""

from __future__ import annotations

__all__ = ["TRADING_DAYS_PER_YEAR"]

#: US equity trading days in a calendar year: 365 days, less weekends, less
#: the nine NYSE holidays. The standard convention, and the default
#: annualization factor throughout this library.
TRADING_DAYS_PER_YEAR = 252
