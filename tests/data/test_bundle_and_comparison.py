"""
The bundle, the source comparison, and the point-in-time tools.

THE COMPARISON IS THE ONE WORTH READING. `FinancialRatios` already documents
that `debt_to_equity` means different things depending on where it came
from: Polygon derives it from total LIABILITIES, yfinance reports it as a
percentage. That is a docstring somebody has to read, and nothing checked
it. A screen ranking a universe on a mix of the two orders it partly by
which provider answered, with no error anywhere.

The hard part is not spotting a difference — a diff does that. It is
separating three cases that look identical and need opposite responses:

  scale       a CONSTANT ratio. A missed unit conversion; fix it with
              arithmetic.
  definition  systematic, ratio NOT constant. The two are computing
              different quantities; no conversion exists.
  agree       within rounding. Vendors differ at the margin about
              everything and it is not a finding.

The tests below use data constructed so the right answer is known: an exact
100x for the unit error, a wandering ratio for the definition difference.
"""

import pandas as pd
import pytest

from standard_quant_tools.data.base import FinancialRatios
from standard_quant_tools.data.bundle import DataBundle, validate_bundle
from standard_quant_tools.data.comparison import (
    classify_divergence,
    compare_ratio_sources,
)
from standard_quant_tools.data.temporal import TemporalContract, price_contract
from standard_quant_tools.error import ValidationError


def _unsafe(kind="fundamentals"):
    return TemporalContract(
        source="yf",
        frame_kind=kind,
        has_event_time=True,
        has_available_time=False,
    )


class TestTellingTheThreeCasesApart:
    def test_a_pure_unit_error_is_scale(self):
        """Percent versus fraction: exactly 100x on every entity."""
        pairs = [("A", 0.15, 15.0), ("B", 0.09, 9.0), ("C", -0.04, -4.0)]
        result = classify_divergence(pairs)
        assert result["verdict"] == "scale"
        assert result["ratio"] == pytest.approx(100.0)

    def test_a_wandering_ratio_is_a_definition_difference(self):
        """The discriminator is whether the ratio is CONSTANT, not how big
        the difference is. These differ by a similar magnitude to the unit
        error above and mean something entirely different."""
        pairs = [("A", 0.15, 0.30), ("B", 0.09, 0.11), ("C", 0.22, 0.51)]
        result = classify_divergence(pairs)
        assert result["verdict"] == "definition"
        assert "no conversion exists" in result["detail"].lower()

    def test_rounding_is_not_a_finding(self):
        pairs = [("A", 18.00, 18.02), ("B", 22.00, 21.98)]
        assert classify_divergence(pairs)["verdict"] == "agree"

    def test_no_overlap_is_reported_as_such(self):
        """Not 'agree'. Two sources that never answered about the same
        entity have not agreed about anything."""
        assert classify_divergence([])["verdict"] == "no_overlap"

    def test_zeros_do_not_masquerade_as_a_definition_difference(self):
        """x/0 says nothing about scale. Letting those into the ratio makes
        it wander for arithmetic reasons and mislabels a clean unit error."""
        pairs = [("A", 0.0, 0.0), ("B", 0.15, 15.0), ("C", 0.09, 9.0)]
        assert classify_divergence(pairs)["verdict"] == "scale"

    def test_a_single_pair_is_not_evidence_of_a_constant_ratio(self):
        """One observation is consistent with every ratio, so calling it a
        unit conversion would be a guess wearing a verdict's clothes."""
        assert classify_divergence([("A", 0.15, 15.0)])["verdict"] == "definition"


class TestTheDebtToEquityCase:
    """The specific bug the expansion plan named."""

    @staticmethod
    def _sources():
        yf = {
            e: FinancialRatios(debt_to_equity=v * 100, trailing_pe=p)
            for e, (v, p) in {
                "AAA": (1.2, 18.0),
                "BBB": (0.4, 22.0),
                "CCC": (2.1, 11.0),
            }.items()
        }
        # Liabilities-based: higher, and by a different factor each time
        # because payables and deferred revenue are not proportional to debt.
        pg = {
            "AAA": FinancialRatios(debt_to_equity=2.9, trailing_pe=18.02),
            "BBB": FinancialRatios(debt_to_equity=0.7, trailing_pe=21.98),
            "CCC": FinancialRatios(debt_to_equity=6.4, trailing_pe=11.01),
        }
        return yf, pg

    def test_it_is_caught_as_a_definition_difference(self):
        yf, pg = self._sources()
        report = compare_ratio_sources(yf, pg, left_name="yf", right_name="pg")
        d2e = next(f for f in report["fields"] if f["field"] == "debt_to_equity")
        assert d2e["verdict"] == "definition"

    def test_the_fields_that_agree_are_not_flagged(self):
        """A report that flagged everything would be ignored."""
        yf, pg = self._sources()
        report = compare_ratio_sources(yf, pg, left_name="yf", right_name="pg")
        pe = next(f for f in report["fields"] if f["field"] == "trailing_pe")
        assert pe["verdict"] == "agree"
        assert not any("trailing_pe" in w for w in report["warnings"])

    def test_the_warning_says_no_conversion_exists(self):
        """The actionable difference from a scale verdict: one of these can
        be fixed with arithmetic and the other cannot."""
        yf, pg = self._sources()
        report = compare_ratio_sources(yf, pg, left_name="yf", right_name="pg")
        warning = next(w for w in report["warnings"] if "debt_to_equity" in w)
        assert "no conversion exists" in warning.lower()

    def test_it_names_the_worst_offenders(self):
        yf, pg = self._sources()
        report = compare_ratio_sources(yf, pg, left_name="yf", right_name="pg")
        d2e = next(f for f in report["fields"] if f["field"] == "debt_to_equity")
        assert d2e["examples"]
        assert set(d2e["examples"][0]) == {"entity", "yf", "pg"}

    def test_a_declared_note_is_surfaced_separately(self):
        """A difference the provider declares about itself is not a bug, and
        is not convertible either -- it belongs beside the measured ones
        rather than mixed in."""
        yf, pg = self._sources()
        pg["AAA"] = FinancialRatios(
            debt_to_equity=2.9,
            definition_notes={"debt_to_equity": "derived from total liabilities"},
        )
        report = compare_ratio_sources(yf, pg, left_name="yf", right_name="pg")
        assert report["declared_definition_notes"]
        assert any("declare a definition difference" in w for w in report["warnings"])


class TestTheBundle:
    def test_it_pairs_every_frame_with_its_contract(self):
        bundle = DataBundle("b")
        bundle.add("bars", pd.DataFrame({"date": [1]}), price_contract("yf"))
        assert bundle.kinds == ["bars"]
        assert bundle.contract("bars").pit_safe

    def test_the_weakest_frame_decides(self):
        """A bundle is used as a unit. Reporting 'mostly safe' would be
        reporting a number nobody can act on."""
        bundle = DataBundle("b")
        bundle.add("bars", pd.DataFrame({"date": [1]}), price_contract("yf"))
        bundle.add("fundamentals", pd.DataFrame({"event_time": [1]}), _unsafe())
        assert bundle.contract("bars").pit_safe
        assert not bundle.pit_safe

    def test_validation_blocks_and_says_which_frame(self):
        bundle = DataBundle("b")
        bundle.add("bars", pd.DataFrame({"date": [1]}), price_contract("yf"))
        bundle.add("fundamentals", pd.DataFrame({"event_time": [1]}), _unsafe())
        verdict = validate_bundle(bundle)
        assert not verdict["usable"]
        assert verdict["blocking"][0].startswith("[fundamentals]")

    def test_a_caller_who_does_not_need_pit_can_say_so(self):
        """Deliberately a decision made in writing rather than a default."""
        bundle = DataBundle("b")
        bundle.add("fundamentals", pd.DataFrame({"event_time": [1]}), _unsafe())
        assert validate_bundle(bundle, require_pit=False)["usable"]

    def test_asking_for_an_unsafe_frame_pit_gated_is_refused(self):
        bundle = DataBundle("b")
        bundle.add("fundamentals", pd.DataFrame({"event_time": [1]}), _unsafe())
        bundle.frame("fundamentals")  # ungated is fine
        with pytest.raises(ValidationError, match="available_time"):
            bundle.frame("fundamentals", require_pit=True)

    def test_replacing_a_frame_is_refused(self):
        """Two callers disagreeing about which frame was the validated one
        is worse than an error."""
        bundle = DataBundle("b")
        bundle.add("bars", pd.DataFrame({"date": [1]}), price_contract("yf"))
        with pytest.raises(ValidationError, match="already in this bundle"):
            bundle.add("bars", pd.DataFrame({"date": [2]}), price_contract("yf"))

    def test_an_unknown_frame_kind_is_refused(self):
        with pytest.raises(ValidationError, match="unknown frame kind"):
            DataBundle("b").add("vibes", pd.DataFrame())

    def test_a_missing_frame_says_what_it_does_hold(self):
        bundle = DataBundle("b")
        bundle.add("bars", pd.DataFrame({"date": [1]}), price_contract("yf"))
        with pytest.raises(ValidationError, match="bars"):
            bundle.frame("macro")

    def test_an_empty_bundle_is_not_silently_valid(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_bundle(DataBundle("b"))


class TestThePitTools:
    """The error worth catching is the two timestamps the wrong way round."""

    def test_swapped_timestamps_are_rejected(self):
        from standard_quant_tools.modeling.agent.models import PitRecordsInput
        from standard_quant_tools.modeling.agent.tools import validate_pit_records

        result = validate_pit_records(
            PitRecordsInput(
                records=[
                    {
                        "entity": "AAA",
                        # available BEFORE the period it describes ended
                        "event_time": "2024-10-25",
                        "available_time": "2024-09-30",
                        "eps": 1.2,
                    }
                ]
            )
        )
        assert not result.valid
        assert "the wrong way round" in (result.problem or "")

    def test_it_reports_the_hindsight_a_naive_join_would_give(self):
        """The number to read even when everything passes."""
        from standard_quant_tools.modeling.agent.models import PitRecordsInput
        from standard_quant_tools.modeling.agent.tools import validate_pit_records

        result = validate_pit_records(
            PitRecordsInput(
                records=[
                    {
                        "entity": "AAA",
                        "event_time": "2024-09-30",
                        "available_time": "2024-10-25",
                        "eps": 1.2,
                    },
                    {
                        "entity": "BBB",
                        "event_time": "2024-09-30",
                        "available_time": "2024-10-28",
                        "eps": 0.8,
                    },
                ]
            )
        )
        assert result.valid
        assert result.median_publication_lag_days == pytest.approx(26.5, abs=1.0)

    def test_a_restatement_is_recognised_as_versioned(self):
        from standard_quant_tools.modeling.agent.models import PitRecordsInput
        from standard_quant_tools.modeling.agent.tools import validate_pit_records

        result = validate_pit_records(
            PitRecordsInput(
                records=[
                    {
                        "entity": "AAA",
                        "event_time": "2024-09-30",
                        "available_time": "2024-10-25",
                        "eps": 1.20,
                    },
                    {
                        "entity": "AAA",
                        "event_time": "2024-09-30",
                        "available_time": "2025-02-10",
                        "eps": 1.05,
                    },
                ]
            )
        )
        assert result.revisions == "versioned"
        assert result.reproduces_history

    def test_one_version_of_everything_stays_unknown(self):
        from standard_quant_tools.modeling.agent.models import PitRecordsInput
        from standard_quant_tools.modeling.agent.tools import validate_pit_records

        result = validate_pit_records(
            PitRecordsInput(
                records=[
                    {
                        "entity": "AAA",
                        "event_time": "2024-09-30",
                        "available_time": "2024-10-25",
                        "eps": 1.2,
                    }
                ]
            )
        )
        assert result.revisions == "unknown"
        assert not result.reproduces_history
        assert any("cannot show whether" in w for w in result.warnings)

    def test_zero_lag_is_flagged_for_a_reported_figure(self):
        """Right for a market bar, wrong for anything published -- usually
        event_time copied across into both columns."""
        from standard_quant_tools.modeling.agent.models import PitRecordsInput
        from standard_quant_tools.modeling.agent.tools import validate_pit_records

        result = validate_pit_records(
            PitRecordsInput(
                records=[
                    {
                        "entity": "AAA",
                        "event_time": "2024-09-30",
                        "available_time": "2024-09-30",
                        "eps": 1.2,
                    }
                ]
            )
        )
        assert result.valid
        assert any("available at the instant" in w for w in result.warnings)

    def test_the_inline_record_cap_is_in_the_schema(self):
        """Bounded where a caller can see it, rather than by whatever the
        JSON layer happens to tolerate."""
        from standard_quant_tools.modeling.agent.models import (
            MAX_INLINE_PIT_RECORDS,
            PitRecordsInput,
        )

        with pytest.raises(Exception):
            PitRecordsInput(
                records=[{"event_time": "2024-01-01"}] * (MAX_INLINE_PIT_RECORDS + 1)
            )

    def test_an_unknown_argument_is_rejected(self):
        from standard_quant_tools.modeling.agent.models import PitRecordsInput

        with pytest.raises(Exception):
            PitRecordsInput(records=[{"event_time": "x"}], entity_scope=True)
