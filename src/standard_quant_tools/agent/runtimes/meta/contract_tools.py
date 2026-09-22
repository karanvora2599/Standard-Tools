"""
The numerical input contract, stated instead of triggered.

Every public boundary in this library runs the same rules on the numbers it
is handed: an infinity is refused, an all-NaN series is refused, a price
must be strictly positive, an annualization factor has a ceiling, a
covariance must be symmetric. The rules exist because the same invalid
input used to produce a `ValidationError` in one function and a
plausible-looking number in another -- a single infinity in a return series
gave a max drawdown of -1.703437775179145, which reads as measured.

WHAT WAS MISSING WAS THE STATEMENT. The rules were enforced on every call
and reported by no tool, so a caller learned them one refusal at a time,
after paying for a fetch and a run to find out. That is the same shape as a
bound that lives only in a validator: correct, and discoverable only by
being wrong.

So this reports the table. Each row names the rule, what it applies to, the
threshold where it bites, an excerpt of the message it raises, and why the
line is drawn there rather than somewhere more permissive. The excerpts are
substrings of the real refusals, which is what makes them worth printing:
an agent that reads one here recognises it when it arrives.

The companion is `validate_tool_call`, which now RUNS these rules against
any numbers already present in a proposed call, so an all-NaN series is a
refusal before the fetch rather than after it.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


class NumericContractInput(BaseModel):
    """No arguments: the contract is the same for every caller.

    Extras are still forbidden. An argument this takes none of is a
    hallucinated one, and reporting it is more useful than accepting it
    and answering as though it had been understood.
    """

    model_config = ConfigDict(extra="forbid")


class ContractRule(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule: str = ""
    applies_to: str = Field("", description="The kind of argument this is checked on.")
    threshold: Optional[str] = Field(
        None,
        description=(
            "Where the rule bites, as a value or a condition. Null when the "
            "rule is about a shape rather than a number."
        ),
    )
    message_excerpt: str = Field(
        "",
        description=(
            "A substring of the refusal this raises. Matching it against an "
            "error you received tells you which rule you hit."
        ),
    )
    why: str = Field("", description="What goes wrong when the rule is not enforced.")


class NumericContractResult(_Result):
    n_rules: int = 0
    rules: List[ContractRule] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


#: The contract as the library enforces it. Written out rather than
#: introspected because the useful half of each row -- the reason the line
#: is where it is -- lives in prose that no signature carries, and a table
#: derived from signatures would report the rules without the reasoning
#: that makes them actionable. Each `message_excerpt` is pinned by a test
#: that triggers the rule and matches the excerpt against what was raised,
#: so a rewritten message cannot leave this table quietly wrong.
_RULES = (
    {
        "rule": "series_rejects_infinity",
        "applies_to": "any numeric series argument",
        "threshold": "any +inf or -inf value",
        "message_excerpt": "non-finite (infinite) value(s) at",
        "why": (
            "An infinity is not a measurement -- it is a division that "
            "should not have happened upstream, commonly a zero or negative "
            "price reaching a percentage change -- and it does not stay "
            "visible: one infinity produced a max drawdown of -1.70, which "
            "reads as a measured number."
        ),
    },
    {
        "rule": "series_rejects_all_nan",
        "applies_to": "any numeric series argument",
        "threshold": "every observation missing",
        "message_excerpt": "contains no observations (every value is NaN)",
        "why": (
            "This is the absence of data rather than data with gaps, and "
            "functions disagreed wildly about it: one returned NaN, one "
            "returned +inf (indistinguishable from a strategy with no "
            "losing bars) and one raised an IndexError."
        ),
    },
    {
        "rule": "series_allows_partial_nan",
        "applies_to": "any numeric series argument",
        "threshold": "refused only where the computation sets allow_nan=False",
        "message_excerpt": "cannot tolerate gaps",
        "why": (
            "The deliberate limit of the contract. Indicator warm-ups, a "
            "ticker that lists mid-sample and a benchmark on another "
            "holiday calendar all legitimately produce gaps, so making them "
            "fatal everywhere would break correct code to catch a problem "
            "those callers already handle."
        ),
    },
    {
        "rule": "series_rejects_empty",
        "applies_to": "any numeric series argument",
        "threshold": "zero observations, unless the computation allows it",
        "message_excerpt": "is empty",
        "why": (
            "An empty series has no answer to give, and the arithmetic "
            "below produces one anyway -- a mean of nothing, a ratio of two "
            "nothings -- rather than saying so."
        ),
    },
    {
        "rule": "price_series_strictly_positive",
        "applies_to": "a price series",
        "threshold": "> 0",
        "message_excerpt": "A price must be > 0",
        "why": (
            "Finiteness is not enough, which is why this is a separate "
            "rule: 0.0 and -5.0 are perfectly finite and are not prices. A "
            "zero divides by zero in a return, and a negative denominator "
            "flips the sign of every return derived from it -- a single "
            "-5.00 close produced a total return of +0.397914."
        ),
    },
    {
        "rule": "level_series_starts_positive",
        "applies_to": "an equity curve or other level series",
        "threshold": "the FIRST observation > 0",
        "message_excerpt": "the opening level is the denominator",
        "why": (
            "Deliberately weaker than the price rule, and the difference "
            "matters: a leveraged position can be wiped out, so a curve "
            "legitimately reaches zero or goes negative at its tail. What "
            "must hold is that the denominator every derived quantity "
            "divides by -- fixed by the opening level -- is positive."
        ),
    },
    {
        "rule": "paired_series_share_a_length",
        "applies_to": "two series compared or regressed against each other",
        "threshold": "equal row counts",
        "message_excerpt": "rows but",
        "why": (
            "Two series of different lengths cannot be paired at all, and "
            "the arithmetic below would pair as many as it could and drop "
            "the rest without saying which."
        ),
    },
    {
        "rule": "paired_series_share_an_index",
        "applies_to": "two series compared or regressed against each other",
        "threshold": "identical index labels, not merely equal length",
        "message_excerpt": "Equal length is not alignment",
        "why": (
            "Three days starting Monday and three starting Tuesday have the "
            "same length and describe different days. pandas label-aligns "
            "them while a NumPy or native path pairs them positionally, so "
            "the same two inputs mean different things depending on which "
            "execution path happened to run."
        ),
    },
    {
        "rule": "count_rejects_a_bool",
        "applies_to": "a window, period or count argument",
        "threshold": "a bool is not a count",
        "message_excerpt": "must be a positive whole number, got bool",
        "why": (
            "True is 1 to every arithmetic operation in Python, so a bool "
            "passed where a window belongs runs a one-period window and "
            "returns a result rather than an error."
        ),
    },
    {
        "rule": "count_is_a_whole_number",
        "applies_to": "a window, period or count argument",
        "threshold": "integral",
        "message_excerpt": "counts whole periods and must be an integer",
        "why": (
            "A fractional window is truncated somewhere below, silently, "
            "and the number that comes back is for a different window than "
            "the one that was asked for."
        ),
    },
    {
        "rule": "count_is_at_least_one",
        "applies_to": "a window, period or count argument",
        "threshold": ">= 1",
        "message_excerpt": "must be >= 1",
        "why": (
            "A negative period is not merely invalid: pandas reads a "
            "negative shift or percentage-change period as a FORWARD "
            "window, which reads future data and reports a backtest that "
            "knew tomorrow."
        ),
    },
    {
        "rule": "periods_per_year_ceiling",
        "applies_to": "the periods_per_year annualization factor",
        "threshold": "31536000",
        "message_excerpt": "exceeds the maximum 31536000",
        "why": (
            "One period per second for a year is the finest sampling that "
            "means anything. Left unchecked the factor produces confidently "
            "wrong numbers rather than errors -- -252 returned a CAGR of "
            "-0.535, which reads as an ordinary annual loss."
        ),
    },
    {
        "rule": "scalar_is_finite_before_any_range_check",
        "applies_to": "any scalar numeric argument",
        "threshold": "finite, checked BEFORE the bounds",
        "message_excerpt": "NaN compares False against every bound",
        "why": (
            "Order is the whole point. Range guards are comparisons and "
            "every comparison against NaN is False, so a check written as "
            "`if rate <= 0: raise` never fires for NaN and the NaN flows on "
            "into a result that carries a success flag and no numbers."
        ),
    },
    {
        "rule": "scalar_within_its_declared_range",
        "applies_to": "any scalar numeric argument with bounds",
        "threshold": "the minimum and maximum the computation declares",
        "message_excerpt": "must be >= ",
        "why": (
            "The bound is checked once, at the boundary, with the argument "
            "named -- rather than in the middle of the arithmetic, where "
            "the message names an internal variable instead."
        ),
    },
    {
        "rule": "covariance_is_square",
        "applies_to": "a covariance matrix",
        "threshold": "2-D with equal dimensions",
        "message_excerpt": "must be a square 2-D matrix",
        "why": (
            "A non-square matrix reaches an eigenvalue routine and fails "
            "there, with a message that names neither the argument nor the "
            "tool that was called."
        ),
    },
    {
        "rule": "covariance_is_finite",
        "applies_to": "a covariance matrix",
        "threshold": "every entry finite",
        "message_excerpt": "non-finite entr(ies)",
        "why": (
            "A NaN covariance is especially quiet: the usual degeneracy "
            "guard is `if variance <= 0`, which NaN does not satisfy, so it "
            "passes the very check meant to catch a degenerate matrix and "
            "emerges as NaN weights."
        ),
    },
    {
        "rule": "covariance_is_symmetric",
        "applies_to": "a covariance matrix",
        "threshold": "rtol 1e-9 against its own transpose",
        "message_excerpt": "is not symmetric (largest |A - A'|",
        "why": (
            "A covariance matrix is symmetric by definition, so an "
            "asymmetric one was not built as a covariance -- and "
            "eigenvalue-based code downstream would silently use one "
            "triangle and answer as though the other agreed."
        ),
    },
    {
        "rule": "frame_rejects_infinity",
        "applies_to": "a numeric frame or panel",
        "threshold": "any +inf or -inf entry, reported per column",
        "message_excerpt": "non-finite (infinite) value(s) in column(s)",
        "why": (
            "Matrix routines fail differently from scalar ones: an infinity "
            "reaching a decomposition raises a bare 'SVD did not converge', "
            "which names neither the input nor the offending column, and a "
            "caller reads it as an algorithmic failure rather than bad data."
        ),
    },
    {
        "rule": "frame_rejects_all_nan",
        "applies_to": "a numeric frame or panel",
        "threshold": "every entry missing",
        "message_excerpt": "contains no observations",
        "why": (
            "The frame-level counterpart of the all-NaN series rule, for "
            "the same reason: a panel with nothing in it is the absence of "
            "data and not a panel with gaps."
        ),
    },
)


def describe_numeric_contract(
    input_data: NumericContractInput,
) -> NumericContractResult:
    """
    Every numerical rule this library enforces at its public boundary.

    These run on every call and were reported by no tool, so the only way
    to learn one was to trigger it -- after a fetch and a run had already
    been paid for. Each row names the rule, the threshold where it bites,
    an excerpt of the message it raises so a refusal you already have can
    be matched to it, and why the line is drawn there.

    The rules are about what a NUMBER may be. Each tool adds its own bounds
    on top in its schema, which `describe_tool` reports and
    `validate_tool_call` checks; this is the layer underneath, shared by
    every boundary so the same input cannot be an error in one function and
    a plausible answer in another.

    Offline, static, and fetches nothing.
    """
    rules = [ContractRule(**row) for row in _RULES]
    logger.debug("[describe_numeric_contract] %d rules", len(rules))
    return NumericContractResult(
        n_rules=len(rules),
        rules=rules,
        notes=[
            "This is the shared contract, not the whole of validation. A "
            "tool's own bounds live in its schema -- describe_tool reports "
            "them and validate_tool_call checks both layers without "
            "calling anything.",
            "Partial missing data is ALLOWED by default. That is a "
            "deliberate limit rather than an oversight: warm-up windows and "
            "mismatched holiday calendars produce legitimate gaps, and a "
            "computation that genuinely cannot tolerate one refuses it by "
            "name.",
        ],
        warnings=[
            "An excerpt is a substring of the refusal, not the whole "
            "message. The real one names the offending positions or "
            "columns, which is the part that tells you where to look."
        ],
    )


__all__ = [
    "ContractRule",
    "NumericContractInput",
    "NumericContractResult",
    "describe_numeric_contract",
]
