"""
Which candidate factors earn a place in a risk model, judged out of sample.

`estimate_factor_model` fits B, F and D for whatever factors it is handed,
and every number it returns is in-sample, where one more factor never makes
the fit worse. `screen_factor_candidates` takes the same inputs -- the
assets and the factors as dated values inline or as `returns_panel`
references with their columns -- plus the factors always in the model
(`base_factors`), and asks four things of every other factor:

- does it explain returns it was not fitted to (walk-forward incremental
  R-squared, against circular shifts of itself, Benjamini-Hochberg across
  the candidates)?
- are its loadings stable (the sign each asset's loading keeps across the
  windows, the two halves' loadings)?
- is it already in the model (VIF against the base and the other
  candidates, and the factor it correlates with most)?
- does it make the risk model's minimum-variance book less volatile out of
  sample, against the base model and against Ledoit-Wolf?

and returns the ones that pass all four, with the reasons every other one
failed. REFERENCES OR INLINE VALUES, NEVER TICKERS: nothing is fetched, and
the shifts are seeded, so the same input gives the same bits. The
arithmetic is `analysis.factor_screen.screen_factor_candidates`.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.models import Seed
from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
    finite_or_none,
)
from standard_quant_tools.agent.runtimes.streams import (
    StreamAlignment,
    aligned_streams,
    columns_only_with_ref,
    dated_streams,
    exactly_one_source,
    finite_dated_streams,
    returns_ref_streams,
)
from standard_quant_tools.analysis import factor_screen as screen_lib
from standard_quant_tools.error import ValidationError

Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]

#: The most assets and factors one call takes, as `estimate_factor_model`
#: takes them. Held in the validators, not advertised.
MAX_ASSETS = 1000
MAX_FACTORS = 50

_WHO = "screen_factor_candidates"


class FactorScreenInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="{asset: {ISO date: return}}, two or more assets."
    )
    returns_ref: Optional[str] = Field(
        None, description="A returns_panel reference in place of returns."
    )
    columns: Optional[List[str]] = Field(
        None, min_length=1, description="returns_ref's assets; default all."
    )
    factors: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="{factor: {ISO date: return}}: base and candidates."
    )
    factors_ref: Optional[str] = Field(
        None, description="A returns_panel reference in place of factors."
    )
    factor_columns: Optional[List[str]] = Field(
        None, min_length=1, description="factors_ref's factors; default all."
    )
    base_factors: List[str] = Field(
        default_factory=list,
        max_length=MAX_FACTORS,
        description="Factors always in the model (a market factor); every "
        "other factor is a candidate.",
    )
    estimation_window: int = Field(
        252, ge=10, le=10_000, description="Rows each window's loadings use."
    )
    step: int = Field(21, ge=1, le=10_000, description="Rows between refits.")
    periods_per_year: int = Field(
        252, ge=1, le=100_000, description="Annualizes the volatilities."
    )
    n_shifts: int = Field(
        500,
        ge=20,
        le=5000,
        description="Circular shifts per candidate; the smallest p is "
        "1 / (n_shifts + 1).",
    )
    random_seed: Seed = Field(0, description="Seeds the shifts.")
    fdr: float = Field(
        0.1, gt=0, le=0.5, description="Benjamini-Hochberg level to pass."
    )

    @field_validator("returns", "factors")
    @classmethod
    def _finite_values(cls, v: Optional[Dict], info) -> Optional[Dict]:
        most = MAX_ASSETS if info.field_name == "returns" else MAX_FACTORS
        return finite_dated_streams(v, info.field_name, most=most)

    @model_validator(mode="after")
    def _one_source_each(self) -> "FactorScreenInput":
        for inline, ref, columns in (
            ("returns", "returns_ref", "columns"),
            ("factors", "factors_ref", "factor_columns"),
        ):
            exactly_one_source(self, inline, ref, _WHO)
            columns_only_with_ref(self, columns, ref)
        repeated = sorted(
            {b for b in self.base_factors if self.base_factors.count(b) > 1}
        )
        if repeated:
            raise ValueError(f"base_factors names {repeated} more than once.")
        return self


class FactorCandidateScreen(ExplainsNulls):
    """One candidate's four tests, and whether it passed them."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "sign_stability": "every full-sample loading is zero, so there is no "
        "sign to keep",
        "half_sample_correlation": "one half's loadings are all zero",
        "vif": "the candidate is an exact linear combination of the other "
        "factors, so its VIF is infinite",
        "most_correlated": "there is no other factor to correlate with",
        "min_variance_volatility": "a window's covariance could not be solved "
        "for the minimum-variance weights",
    }

    incremental_r_squared: Stat = Field(
        None,
        description="OUT OF SAMPLE: (SSE base - SSE base + candidate) / SST, "
        "over every asset and test row.",
    )
    null_mean: Stat = Field(
        None, description="The same over the candidate's circular shifts, mean."
    )
    p_value: Stat = Field(
        None, description="Share of shifts at least as good, +1 corrected."
    )
    p_value_bh: Stat = Field(
        None, description="Benjamini-Hochberg adjusted across the candidates."
    )
    sign_stability: Stat = Field(
        None,
        description="Share of windows each asset's loading keeps its "
        "full-sample sign, weighted by |loading|.",
    )
    half_sample_correlation: Stat = Field(
        None,
        description="Uncentred correlation of the loadings fitted on each "
        "half of the sample: -1 is every loading flipped.",
    )
    vif: Stat = Field(
        None, description="IN-SAMPLE: against the base and the other candidates."
    )
    most_correlated_factor: Optional[str] = None
    most_correlated: Stat = Field(None, description="Its correlation with that one.")
    min_variance_volatility: Stat = Field(
        None,
        description="OUT OF SAMPLE, annualized: the minimum-variance book of "
        "the base + candidate model's B F B' + D, refit every step.",
    )
    passed: bool = False
    failed: List[str] = Field(
        default_factory=list, description="Why it was not selected, per test."
    )


class FactorScreenResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "base_oos_r_squared": "the test rows' returns never vary",
        "min_variance_volatility": "that model has more coefficients than a "
        "window has rows, or a window's covariance could not be solved (the "
        "warnings say which)",
    }

    assets: List[str] = Field(default_factory=list)
    base_factors: List[str] = Field(default_factory=list)
    selected: List[str] = Field(
        default_factory=list,
        description="Candidates passing all four: BH p <= fdr, stable "
        "loadings, VIF under 10, a lower minimum-variance volatility.",
    )
    n_candidates: int = Field(0, description="Candidates screened: the trials.")
    candidates: Dict[str, FactorCandidateScreen] = Field(default_factory=dict)
    min_variance_volatility: Dict[str, Stat] = Field(
        default_factory=dict,
        description="OUT OF SAMPLE, annualized, on the same steps: base, "
        "ledoit_wolf, all_candidates, and selected (base + selected, "
        "HINDSIGHT: scored on the steps that selected it; the base alone "
        "when none passed).",
    )
    base_oos_r_squared: Stat = Field(
        None, description="The base model's out-of-sample R-squared."
    )
    n_observations: int = Field(0, description="Dates every stream shares.")
    estimation_window: int = 0
    step: int = 0
    n_steps: int = Field(0, description="Walk-forward refits.")
    n_test_rows: int = 0
    first_test_date: Optional[str] = None
    last_test_date: Optional[str] = None
    streams: StreamAlignment = Field(
        default_factory=StreamAlignment,
        description="How the asset and factor streams were put on the dates "
        "they all share, and what each lost.",
    )
    warnings: List[str] = Field(default_factory=list)


def _side(
    inline: Optional[Dict[str, Dict[str, float]]],
    ref: Optional[str],
    columns: Optional[List[str]],
    inline_field: str,
    ref_field: str,
) -> Dict[str, Any]:
    """One side's series, as dated Series by name."""
    if ref is None:
        assert inline is not None  # one of the two, by the model
        return dated_streams(inline, what=_WHO, field=inline_field)
    return returns_ref_streams(ref, what=_WHO, columns=columns, field=ref_field)


def _iso(stamp: Any) -> str:
    import pandas as pd

    stamp = pd.Timestamp(stamp)
    return str(stamp.date()) if stamp == stamp.normalize() else stamp.isoformat()


def screen_factor_candidates(input_data: FactorScreenInput) -> FactorScreenResult:
    """
    The candidates that pass significance, stability, redundancy and the
    risk model's out-of-sample minimum-variance test, and why the rest fail.
    """
    assets = _side(
        input_data.returns,
        input_data.returns_ref,
        input_data.columns,
        "returns",
        "returns_ref",
    )
    factors = _side(
        input_data.factors,
        input_data.factors_ref,
        input_data.factor_columns,
        "factors",
        "factors_ref",
    )
    for side, values, most in (
        ("asset", assets, MAX_ASSETS),
        ("factor", factors, MAX_FACTORS),
    ):
        if len(values) > most:
            raise ValidationError(
                f"{_WHO}: {len(values)} {side} series, more than the {most} one "
                "call takes. Name the ones you want in columns or "
                "factor_columns."
            )
    both = [name for name in assets if name in factors]
    if both:
        raise ValidationError(
            f"{_WHO}: {both[:5]} named both as an asset and as a factor. Each "
            "name is one series on the common calendar, and an asset regressed "
            "on itself is an identity; rename one side's series or leave it out."
        )
    frame, alignment, notes = aligned_streams(
        {**assets, **factors},
        what=_WHO,
        min_overlap=input_data.estimation_window + 2 * input_data.step,
    )
    out = screen_lib.screen_factor_candidates(
        frame[list(assets)],
        frame[list(factors)],
        base_factors=input_data.base_factors,
        estimation_window=input_data.estimation_window,
        step=input_data.step,
        periods_per_year=input_data.periods_per_year,
        n_shifts=input_data.n_shifts,
        random_seed=input_data.random_seed,
        fdr=input_data.fdr,
    )
    candidates = {
        name: FactorCandidateScreen(
            **{
                key: row[key]
                for key in FactorCandidateScreen.model_fields
                if key in row
            }
        )
        for name, row in out["candidates_screened"].items()
    }
    return FactorScreenResult(
        assets=out["assets"],
        base_factors=out["base_factors"],
        selected=out["selected"],
        n_candidates=len(out["candidates"]),
        candidates=candidates,
        min_variance_volatility=out["min_variance_volatility"],
        base_oos_r_squared=out["base_oos_r_squared"],
        n_observations=out["n_observations"],
        estimation_window=out["estimation_window"],
        step=out["step"],
        n_steps=out["n_steps"],
        n_test_rows=out["n_test_rows"],
        first_test_date=_iso(out["first_test_date"]),
        last_test_date=_iso(out["last_test_date"]),
        streams=alignment,
        warnings=notes + list(out["warnings"]),
    )


FACTOR_SCREEN_TOOL_DEFS = [
    (
        "screen_factor_candidates",
        "Which candidate factors earn a place in a risk model, out of sample: "
        "incremental R-squared against circular shifts, loading stability, "
        "VIF and minimum-variance risk. Walked forward over base_factors "
        "(loadings fitted on the trailing estimation_window, applied to the "
        "next step), the p-values Benjamini-Hochberg adjusted across the "
        "candidates; the minimum-variance book from B F B' + D must be less "
        "volatile with the factor than without, Ledoit-Wolf beside them. "
        "selected passes all four and every other candidate says why not. "
        "Takes estimate_factor_model's inputs.",
        FactorScreenInput,
    ),
]

FACTOR_SCREEN_TOOL_DISPATCH = {
    "screen_factor_candidates": (screen_factor_candidates, FactorScreenInput),
}

__all__ = [
    "FACTOR_SCREEN_TOOL_DEFS",
    "FACTOR_SCREEN_TOOL_DISPATCH",
    "FactorCandidateScreen",
    "FactorScreenInput",
    "FactorScreenResult",
    "screen_factor_candidates",
]
