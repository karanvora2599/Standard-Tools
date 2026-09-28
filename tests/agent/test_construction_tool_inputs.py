"""
What the concentration and factor-exposure tools accept.

Both take inline weights and neither checked them: an infinite weight made
`analyze_concentration` divide by zero, a NaN weight was dropped by it
without a word, and in `get_factor_exposure_budget` a NaN or infinite weight
or loading -- or a held asset with no loading on some factor -- came out as
a NaN or infinite exposure, reported as if it had been measured. See the
CHANGELOG entry of 2026-09-28.
"""

import math

import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.portfolio.construction_tools import (
    ConcentrationInput,
    FactorExposureInput,
    FactorExposureResult,
    analyze_concentration,
    get_factor_exposure_budget,
)
from standard_quant_tools.error import ValidationError

LOADINGS = {
    "A": {"mkt": 1.0, "size": 0.2},
    "B": {"mkt": 0.8, "size": -0.4},
    "C": {"mkt": 1.2, "size": 0.0},
}
WEIGHTS = {"A": 0.5, "B": 0.3, "C": 0.2}


class TestConcentration:
    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_a_non_finite_weight_is_refused_by_the_schema(self, bad):
        with pytest.raises(pydantic.ValidationError, match="finite"):
            ConcentrationInput(weights={"A": bad, "B": 0.5})

    def test_through_dispatch_it_is_a_schema_refusal_not_a_zero_division(self):
        with pytest.raises(pydantic.ValidationError):
            resolve("portfolio").dispatch(
                "analyze_concentration", {"weights": {"A": float("inf"), "B": 1.0}}
            )

    def test_a_gross_exposure_past_the_float_range_is_refused(self):
        """Each weight finite, their sum not."""
        with pytest.raises(ValidationError, match="float range"):
            analyze_concentration(ConcentrationInput(weights={"A": 1e308, "B": 1e308}))

    def test_an_ordinary_book_is_measured(self):
        result = analyze_concentration(ConcentrationInput(weights={"A": 0.5, "B": 0.5}))
        assert result.effective_n == pytest.approx(2.0)
        assert result.herfindahl == pytest.approx(0.5)


class TestFactorExposure:
    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_a_non_finite_weight_is_refused_by_the_schema(self, bad):
        with pytest.raises(pydantic.ValidationError, match="finite"):
            FactorExposureInput(weights={**WEIGHTS, "A": bad}, factor_loadings=LOADINGS)

    def test_a_non_finite_loading_is_refused_by_the_schema(self):
        loadings = {**LOADINGS, "B": {"mkt": float("nan"), "size": 0.1}}
        with pytest.raises(pydantic.ValidationError, match="finite"):
            FactorExposureInput(weights=WEIGHTS, factor_loadings=loadings)

    def test_a_held_asset_missing_a_loading_is_refused_by_name(self):
        """Absent is not zero: the gap used to make the factor's exposure
        NaN for the whole portfolio."""
        loadings = {**LOADINGS, "C": {"mkt": 1.2}}
        with pytest.raises(ValidationError, match=r"\['C/size'\]"):
            get_factor_exposure_budget(
                FactorExposureInput(weights=WEIGHTS, factor_loadings=loadings)
            )

    def test_a_gap_outside_the_held_assets_does_not_matter(self):
        loadings = {**LOADINGS, "D": {"mkt": 0.9}}
        result = get_factor_exposure_budget(
            FactorExposureInput(weights=WEIGHTS, factor_loadings=loadings)
        )
        assert set(result.exposures) == {"mkt", "size"}

    def test_narrowing_factors_to_the_shared_ones_is_the_remedy(self):
        loadings = {**LOADINGS, "C": {"mkt": 1.2}}
        result = get_factor_exposure_budget(
            FactorExposureInput(
                weights=WEIGHTS, factor_loadings=loadings, factors=["mkt"]
            )
        )
        assert result.exposures["mkt"] == pytest.approx(0.5 + 0.24 + 0.24)

    def test_an_exposure_past_the_float_range_is_refused(self):
        with pytest.raises(ValidationError, match="float range"):
            get_factor_exposure_budget(
                FactorExposureInput(
                    weights={"A": 1e300, "B": 1e300},
                    factor_loadings={"A": {"mkt": 1e300}, "B": {"mkt": 1e300}},
                )
            )

    def test_an_empty_factor_list_is_refused(self):
        with pytest.raises(pydantic.ValidationError, match="names no factor"):
            FactorExposureInput(weights=WEIGHTS, factor_loadings=LOADINGS, factors=[])

    def test_a_repeated_factor_is_refused(self):
        with pytest.raises(pydantic.ValidationError, match=r"repeats \['mkt'\]"):
            FactorExposureInput(
                weights=WEIGHTS, factor_loadings=LOADINGS, factors=["mkt", "mkt"]
            )

    def test_the_exposures_are_the_weighted_loadings(self):
        """The null case."""
        result = get_factor_exposure_budget(
            FactorExposureInput(weights=WEIGHTS, factor_loadings=LOADINGS)
        )
        assert result.exposures["mkt"] == pytest.approx(0.5 + 0.24 + 0.24)
        assert result.exposures["size"] == pytest.approx(0.1 - 0.12)
        assert all(math.isfinite(v) for v in result.exposures.values())

    def test_a_non_finite_exposure_could_only_ever_be_null(self):
        """The result field is a `Stat` map: whatever reaches it, a NaN
        cannot be serialized as one."""
        result = FactorExposureResult(exposures={"mkt": float("nan"), "size": 0.1})
        assert result.exposures == {"mkt": None, "size": 0.1}
