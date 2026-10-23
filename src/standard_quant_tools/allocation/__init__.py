"""
Capital allocation across return streams: how much each strategy, sleeve or
model deserves, whether a rule for deciding that survives out of sample,
how hard to lever it, and what a book with external flows returned.

Every function here takes dated return streams (aligned with
`portfolio.streams.align_return_streams`) or plain arrays, never tickers,
and fetches nothing.
"""

from .attribution import brinson_fachler
from .book_returns import measure_book_returns
from .capacity import capacity_curve
from .kelly import drawdown_probability, empirical_kelly, kelly_report
from .overlays import apply_overlay
from .sleeves import (
    SleeveRule,
    allocate_across_sleeves,
    fit_sleeve_rule,
    shrunk_sharpe_ratios,
    water_fill,
)
from .walk_forward import evaluate_allocation_rule, walk_rule_forward

__all__: list = [
    "SleeveRule",
    "allocate_across_sleeves",
    "apply_overlay",
    "brinson_fachler",
    "capacity_curve",
    "drawdown_probability",
    "empirical_kelly",
    "evaluate_allocation_rule",
    "fit_sleeve_rule",
    "kelly_report",
    "measure_book_returns",
    "shrunk_sharpe_ratios",
    "walk_rule_forward",
    "water_fill",
]
