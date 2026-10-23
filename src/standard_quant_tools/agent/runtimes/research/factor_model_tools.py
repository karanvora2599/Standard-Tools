"""
A factor model estimated from return series, for the risk tools to read.

`run_factor_regression` regresses ONE ticker on factor tickers and reports
its loadings as findings. `estimate_factor_model` estimates the model a
risk decomposition needs -- loadings B for every asset, the factor
covariance F and the specific variances D -- from returns already in hand:
the assets as dated values inline or a published `returns_panel`, and the
factors the same way, put on the dates they all share with the alignment
report. What it returns is `decompose_factor_risk`'s input as it stands,
and with a `run_id` the implied covariance Sigma = B F B' + D is published
as a `covariance` reference, which the allocators and `optimize_portfolio`
take as `covariance_ref`.

REFERENCES OR INLINE VALUES, NEVER TICKERS. Nothing is fetched, so the same
input gives the same bits. Inline values are read by the stream helpers
every `ReturnStream` is read by (`agent.runtimes.streams`), and checked by
the same `align_return_streams`; a reference is read by
`returns_ref_streams`. The `ReturnStream` model itself is not an input
here because it also takes a `symbol`, which would make this tool a
fetcher.

The arithmetic is `analysis.factor_model.estimate_factor_model`, whose
regression is `multi_factor_regression`'s.
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

from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
)
from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
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
from standard_quant_tools.analysis import factor_model as factor_lib
from standard_quant_tools.error import ValidationError

Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]

#: The most assets and factors one call takes. Held in the validators
#: rather than advertised: a guard against a runaway payload, not something
#: a caller plans around.
MAX_ASSETS = 1000
MAX_FACTORS = 50

_WHO = "estimate_factor_model"


class FactorModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="{asset: {ISO date: return}}; aligned with the factors on "
        "the dates they share.",
    )
    returns_ref: Optional[str] = Field(
        None, description="A returns_panel reference in place of returns."
    )
    columns: Optional[List[str]] = Field(
        None, min_length=1, description="returns_ref's assets; default all."
    )
    factors: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="{factor: {ISO date: return}}."
    )
    factors_ref: Optional[str] = Field(
        None, description="A returns_panel reference in place of factors."
    )
    factor_columns: Optional[List[str]] = Field(
        None, min_length=1, description="factors_ref's factors; default all."
    )
    intercept: bool = Field(True, description="Fit an intercept (alpha).")
    periods_per_year: int = Field(
        252, ge=1, description="Annualizes F, D and the covariance; 12 monthly."
    )
    run_id: Optional[str] = Field(
        None,
        description="Publish B F B' + D as a covariance reference "
        "(covariance_ref). A used run_id is refused.",
    )

    @field_validator("returns", "factors")
    @classmethod
    def _finite_values(cls, v: Optional[Dict], info) -> Optional[Dict]:
        most = MAX_ASSETS if info.field_name == "returns" else MAX_FACTORS
        return finite_dated_streams(v, info.field_name, most=most)

    @model_validator(mode="after")
    def _one_source_each(self) -> "FactorModelInput":
        for inline, ref, columns in (
            ("returns", "returns_ref", "columns"),
            ("factors", "factors_ref", "factor_columns"),
        ):
            exactly_one_source(self, inline, ref, _WHO)
            columns_only_with_ref(self, columns, ref)
        return self


_EXACT_FIT = (
    "the fit is exact (no residual variance), so the standard error is zero "
    "and the t-statistic is undefined"
)


class FactorModelResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "loading_t_stats": _EXACT_FIT,
        "alpha_t_stats": _EXACT_FIT,
        "loading_standard_errors": "X'X could not be inverted",
        "alpha_standard_errors": "X'X could not be inverted",
        "r_squared": "the asset's return never varied over the dates used, so "
        "there is no variance to explain",
        "largest_residual_correlation": "there are not two assets whose "
        "residuals both vary, so no residual correlation to report",
    }

    assets: List[str] = Field(default_factory=list)
    factors: List[str] = Field(default_factory=list)
    intercept: bool = True
    n_observations: int = Field(0, description="Dates every stream shares.")
    n_coefficients: int = Field(0, description="p: the factors, plus 1 for alpha.")
    degrees_of_freedom: int = Field(0, description="n_observations - p.")
    periods_per_year: int = 252
    factor_loadings: Dict[str, Dict[str, Stat]] = Field(
        default_factory=dict,
        description="B: asset -> {factor: OLS loading}. decompose_factor_risk's "
        "factor_loadings.",
    )
    loading_standard_errors: Dict[str, Dict[str, Stat]] = Field(
        default_factory=dict,
        description="sqrt(s_i^2 [(X'X)^-1]_jj): independent, homoskedastic "
        "residuals assumed.",
    )
    loading_t_stats: Dict[str, Dict[str, Stat]] = Field(default_factory=dict)
    alpha: Optional[Dict[str, Stat]] = Field(
        None,
        description="Asset -> intercept, PER PERIOD (not annualized). Null "
        "with intercept=false.",
    )
    alpha_standard_errors: Optional[Dict[str, Stat]] = None
    alpha_t_stats: Optional[Dict[str, Stat]] = None
    r_squared: Dict[str, Stat] = Field(
        default_factory=dict,
        description="IN-SAMPLE: the share of each asset's variance its fitted "
        "loadings explain on the dates they were fitted to.",
    )
    factor_covariance: Dict[str, Dict[str, Stat]] = Field(
        default_factory=dict,
        description="F: the factors' sample covariance (n - 1), annualized. "
        "decompose_factor_risk's factor_covariance.",
    )
    factor_covariance_standard_errors: Dict[str, Dict[str, Stat]] = Field(
        default_factory=dict,
        description="Normal theory: sqrt((F_jk^2 + F_jj F_kk) / (n - 1)).",
    )
    specific_variance: Dict[str, Stat] = Field(
        default_factory=dict,
        description="D: each asset's residual variance SSR / (n - p), "
        "annualized. decompose_factor_risk's specific_variance.",
    )
    specific_variance_standard_errors: Dict[str, Stat] = Field(
        default_factory=dict, description="Normal theory: D sqrt(2 / (n - p))."
    )
    specific_variance_basis: str = Field(
        "", description="The degrees-of-freedom correction D was computed with."
    )
    largest_residual_correlation: Stat = Field(
        None,
        description="The largest |correlation| between two assets' residuals, "
        "signed. The model's D sets every one of them to zero.",
    )
    largest_residual_correlation_pair: Optional[List[str]] = None
    covariance_ref: Optional[str] = Field(
        None,
        description="Sigma = B F B' + D, annualized, as an `sqt://covariance/...` "
        "reference: what the allocators and optimize_portfolio take as "
        "covariance_ref. Null without a run_id.",
    )
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
    """One side's series, as dated Series by name: the inline values, or the
    reference's columns (those named in `columns`, in that order)."""
    if ref is None:
        assert inline is not None  # one of the two, by the model
        return dated_streams(inline, what=_WHO, field=inline_field)
    return returns_ref_streams(ref, what=_WHO, columns=columns, field=ref_field)


def estimate_factor_model(input_data: FactorModelInput) -> FactorModelResult:
    """
    B, F and D of a time-series factor model from return series, and the
    covariance they imply, published as a reference when given a run_id.
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
            "on itself is an identity, not an estimate; rename one side's "
            "series or leave it out."
        )
    p = len(factors) + (1 if input_data.intercept else 0)
    frame, alignment, notes = aligned_streams(
        {**assets, **factors}, what=_WHO, min_overlap=p + 1
    )
    out = factor_lib.estimate_factor_model(
        frame[list(assets)],
        frame[list(factors)],
        intercept=input_data.intercept,
        periods_per_year=input_data.periods_per_year,
    )
    covariance_ref = None
    if input_data.run_id is not None:
        from standard_quant_tools.agent.runtimes import handoff

        covariance_ref = handoff.publish(
            {
                "assets": list(out["assets"]),
                "matrix": out["covariance"],
                "annualized": True,
                "periods_per_year": int(out["periods_per_year"]),
                "method": "factor_model",
                "n_observations": int(out["n_observations"]),
            },
            kind="covariance",
            run_id=input_data.run_id,
            name="factor_model",
            producer=_WHO,
        )
    return FactorModelResult(
        assets=out["assets"],
        factors=out["factors"],
        intercept=out["intercept"],
        n_observations=out["n_observations"],
        n_coefficients=out["n_coefficients"],
        degrees_of_freedom=out["degrees_of_freedom"],
        periods_per_year=out["periods_per_year"],
        factor_loadings=out["loadings"],
        loading_standard_errors=out["loading_standard_errors"],
        loading_t_stats=out["loading_t_stats"],
        alpha=out["alpha"],
        alpha_standard_errors=out["alpha_standard_error"],
        alpha_t_stats=out["alpha_t_stat"],
        r_squared=out["r_squared"],
        factor_covariance=out["factor_covariance"],
        factor_covariance_standard_errors=out["factor_covariance_standard_errors"],
        specific_variance=out["specific_variance"],
        specific_variance_standard_errors=out["specific_variance_standard_errors"],
        specific_variance_basis=out["specific_variance_basis"],
        largest_residual_correlation=out["largest_residual_correlation"],
        largest_residual_correlation_pair=out["largest_residual_correlation_pair"],
        covariance_ref=covariance_ref,
        streams=alignment,
        warnings=notes + list(out["warnings"]),
    )


FACTOR_MODEL_TOOL_DEFS = [
    (
        "estimate_factor_model",
        "A factor model from return series: each asset's OLS loadings on "
        "the caller's factor returns, the factor covariance F and the "
        "specific variances D, with standard errors. They come back as "
        "decompose_factor_risk's inputs, and with a run_id the implied "
        "covariance B F B' + D is published as a covariance reference for "
        "the allocators and optimize_portfolio. D is SSR / (n - p), the "
        "residual variance with the degrees-of-freedom correction; F and D "
        "are annualized by periods_per_year. Unlike run_factor_regression "
        "(one ticker on factor tickers) this takes dated values or a "
        "returns_panel reference for many assets, and the factors the same "
        "way, on the dates they share. Every number is in-sample, and D "
        "sets residual covariances to zero: read "
        "largest_residual_correlation.",
        FactorModelInput,
    ),
]

FACTOR_MODEL_TOOL_DISPATCH = {
    "estimate_factor_model": (estimate_factor_model, FactorModelInput),
}

__all__ = [
    "FACTOR_MODEL_TOOL_DEFS",
    "FACTOR_MODEL_TOOL_DISPATCH",
    "FactorModelInput",
    "FactorModelResult",
    "estimate_factor_model",
]
