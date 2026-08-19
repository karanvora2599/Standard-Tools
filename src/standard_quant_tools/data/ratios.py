"""
The canonical definition and unit of every `FinancialRatios` field, and the
per-provider adaptation to it.

`FinancialRatios` is populated by yfinance, Polygon and Bloomberg, and the
shared field names implied an interchangeability that did not exist. Two
separate problems hid behind them.

UNITS. Vendors disagree about whether a rate is a fraction or a percentage,
and the same number means two different things depending on who served it:

    yfinance  debtToEquity          150.5   (a PERCENTAGE)
    Polygon   liabilities / equity    1.505 (a plain RATIO)

A screen written as `debt_equity_max=2.0` therefore admitted almost every
company on one provider and almost none on another, with nothing in either
result indicating which convention was in force.

DEFINITIONS. `debt_to_equity` is the clearest case. Bloomberg's
`TOT_DEBT_TO_TOT_EQY` is total DEBT over equity; Polygon's is computed from
total LIABILITIES over equity, and liabilities include payables, deferred
revenue and lease obligations that are not debt. The Polygon figure is
therefore systematically higher for the same company — not by a scale factor
that could be corrected, but because it answers a different question.

The response to those two problems is deliberately different. A unit
difference is mechanical, so it is CONVERTED. A definition difference is not,
so it is DECLARED: `FinancialRatios.definition_notes` names any field whose
basis departs from the canonical one, and the value is still returned rather
than discarded — a liabilities-to-equity ratio is useful when you know that is
what it is.

There is no plausibility-based auto-correction here on purpose. Inferring
"15.0 must be a percentage" would silently rewrite a genuine 1500% return on
equity, which small-equity companies really do post. Each provider declares
its own vendor's units, and `implausible_value_warnings` reports a value that
looks like an unconverted percentage instead of quietly changing it — so a
vendor changing convention surfaces as a warning rather than as a wrong
number.
"""

from typing import Dict, List, Optional

__all__ = [
    "CANONICAL_UNITS",
    "FIELD_DEFINITIONS",
    "percent_to_fraction",
    "implausible_value_warnings",
]

# What each field means once it leaves a provider, regardless of which one.
CANONICAL_UNITS: Dict[str, str] = {
    "forward_pe": "plain ratio (price / forward earnings per share)",
    "trailing_pe": "plain ratio (price / trailing 12-month earnings per share)",
    "price_to_book": "plain ratio (market cap / book value of equity)",
    "debt_to_equity": "plain ratio (total DEBT / total shareholder equity)",
    "return_on_equity": "decimal fraction (0.15 == 15%)",
    "profit_margins": "decimal fraction (0.25 == 25%)",
    "dividend_yield": "decimal fraction (0.015 == 1.5%)",
    "market_cap": "absolute units of the reporting currency (not millions)",
}

FIELD_DEFINITIONS: Dict[str, str] = {
    "forward_pe": "Price divided by consensus forward EPS.",
    "trailing_pe": "Price divided by trailing twelve-month EPS.",
    "price_to_book": "Market capitalisation divided by book value of equity.",
    "debt_to_equity": (
        "Total interest-bearing DEBT divided by total shareholder equity. "
        "Not total liabilities: payables, deferred revenue and lease "
        "obligations are liabilities but not debt, and including them raises "
        "the ratio for reasons unrelated to leverage."
    ),
    "return_on_equity": "Net income divided by shareholder equity.",
    "profit_margins": "Net income divided by revenue.",
    "dividend_yield": "Indicated annual dividend divided by price.",
    "market_cap": "Shares outstanding times price, in the reporting currency.",
}

# Fields that are decimal fractions rather than plain ratios. Only these are
# candidates for a percentage-to-fraction conversion.
_FRACTION_FIELDS = frozenset({"return_on_equity", "profit_margins", "dividend_yield"})

# Above this, a "decimal fraction" is far more likely to be an unconverted
# percentage than a real value. 3.0 == 300%: real but rare for ROE, and
# essentially unheard of for a dividend yield or a net margin. Chosen to sit
# above genuine outliers so the warning stays worth reading.
_IMPLAUSIBLE_FRACTION = 3.0


def percent_to_fraction(value: Optional[float]) -> Optional[float]:
    """
    Convert a vendor percentage (15.0) to this package's decimal fraction
    (0.15).

    Applied only where the PROVIDER's own documented unit is a percentage —
    never conditionally on the magnitude of the value. A magnitude test would
    silently rewrite a genuine 1500% return on equity, which companies with
    very small equity really do report.
    """
    if value is None:
        return None
    return float(value) / 100.0


def implausible_value_warnings(ratios: object) -> List[str]:
    """
    Flag values that look like an unconverted percentage.

    A detector, not a corrector. If a vendor changes its convention — as
    yfinance did with `dividendYield`, which moved from a fraction to a
    percentage between releases — this surfaces the change as a warning
    rather than letting every downstream screen shift by a factor of 100 in
    silence. The value itself is left alone, because the alternative is
    guessing on the caller's behalf about data they can see and we cannot.
    """
    warnings: List[str] = []
    for field in sorted(_FRACTION_FIELDS):
        value = getattr(ratios, field, None)
        if value is None:
            continue
        if abs(float(value)) > _IMPLAUSIBLE_FRACTION:
            warnings.append(
                f"{field}={value} is implausible as a decimal fraction "
                f"({float(value) * 100:.0f}%). This package's canonical unit for "
                f"{field} is {CANONICAL_UNITS[field]}, so a value this large "
                "usually means the provider reported a percentage that was not "
                "converted — check the provider adapter before trusting any "
                "screen built on this field. The value is reported unchanged "
                "rather than auto-corrected, since a genuine outlier is "
                "possible."
            )
    return warnings
