"""
The temporal contract: what a source can say about WHEN, asked before
anything is fetched.

WHY IT EXISTS SEPARATELY FROM THE JOIN. `asof_join` already refuses a frame
with no `available_time`, and that refusal is correct. It is also late: by
the time it fires, a caller has chosen a universe, fetched a history and
written a cache. The contract answers the same question first.

THE PLAN ASKED FOR THREE TIMESTAMPS. It got two, and the third became a
declaration instead. The reasoning is measured rather than asserted —
`test_a_restatement_is_a_row_not_a_column` below reproduces a restated
earnings figure through the existing two-timestamp join and shows the
as-of-then value coming back correctly. A `revision_time` column would have
carried the same value as that row's `available_time`, and would have
invited the one-row-per-fact encoding that cannot reproduce history at all.

So what is declared instead is HOW REVISIONS ARE ENCODED, which is the thing
that actually separates a source you can backtest on from one that can only
describe the present.
"""

import pandas as pd
import pytest

from standard_quant_tools.data.temporal import (
    FRAME_KINDS,
    REVISION_ENCODINGS,
    TemporalContract,
    contract_for_frame,
    price_contract,
    require_pit,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.point_in_time import asof_join


def _contract(**kwargs):
    base = dict(
        source="test",
        frame_kind="fundamentals",
        has_event_time=True,
        has_available_time=True,
    )
    base.update(kwargs)
    return TemporalContract(**base)


class TestTwoTimestampsAreEnough:
    def test_a_restatement_is_a_row_not_a_column(self):
        """
        The measurement behind dropping the plan's third timestamp.

        Q3 EPS reported at 1.20 and later restated to 1.05, encoded as two
        rows sharing an event_time. The join returns the version that was
        current at each date, which is exactly what a `revision_time` column
        was supposed to provide.
        """
        records = pd.DataFrame(
            [
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
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-11-01", "2024-12-01", "2025-03-01"]),
                "entity": ["AAA"] * 3,
            }
        )
        joined = asof_join(panel, records, fields=["eps"])
        assert list(joined["eps"]) == [1.20, 1.20, 1.05], (
            "the two-timestamp encoding failed to reproduce the as-of-then "
            "value, which is the entire premise for not adding a third"
        )

    def test_before_anything_was_published_the_answer_is_nothing(self):
        """Not zero, and not the eventual value. Nobody knew yet."""
        records = pd.DataFrame(
            [
                {
                    "entity": "AAA",
                    "event_time": "2024-09-30",
                    "available_time": "2024-10-25",
                    "eps": 1.20,
                }
            ]
        )
        panel = pd.DataFrame(
            {"date": pd.to_datetime(["2024-10-01"]), "entity": ["AAA"]}
        )
        joined = asof_join(panel, records, fields=["eps"])
        assert pd.isna(joined["eps"].iloc[0])


class TestPitSafety:
    def test_missing_availability_is_not_pit_safe(self):
        assert not _contract(has_available_time=False).pit_safe

    def test_missing_event_time_is_still_pit_safe(self):
        """A labelling problem, not a leak. You cannot say what period the
        value describes; you can still place it in time correctly."""
        contract = _contract(has_event_time=False)
        assert contract.pit_safe
        assert any("cannot be attributed" in c for c in contract.caveats())

    def test_the_refusal_names_the_leak_it_prevents(self):
        contract = _contract(has_available_time=False)
        with pytest.raises(ValidationError) as exc:
            require_pit(contract, "build_fundamental_panel")
        message = str(exc.value)
        assert "build_fundamental_panel" in message
        assert "available_time" in message
        # It must say WHY, not just "unsupported" -- the caller has to be
        # able to relay this to a human who will ask.
        assert "hindsight" in message

    def test_a_safe_contract_passes_silently(self):
        require_pit(_contract(), "anything")


class TestPitSafeAndReproducesHistoryComeApart:
    """The distinction the whole type exists to carry."""

    def test_a_snapshot_source_joins_safely_but_rewrites_history(self):
        contract = _contract(revisions="snapshot")
        assert contract.pit_safe, "a snapshot source does not leak the future"
        assert not contract.reproduces_history
        caveats = contract.caveats()
        assert any("restated" in c for c in caveats)
        assert any("indicative, not reproducible" in c for c in caveats)

    def test_a_versioned_source_does_both(self):
        contract = _contract(revisions="versioned")
        assert contract.pit_safe and contract.reproduces_history
        assert contract.caveats() == []

    def test_a_never_restated_source_does_both(self):
        assert _contract(revisions="none").reproduces_history

    def test_unknown_is_treated_as_the_unsafe_case(self):
        """A provider that does not say must not be given the benefit of the
        doubt: the cost of assuming `versioned` wrongly is a backtest whose
        numbers nobody can reproduce."""
        assert not _contract(revisions="unknown").reproduces_history

    def test_an_unsafe_source_never_reproduces_history(self):
        assert not _contract(
            has_available_time=False, revisions="versioned"
        ).reproduces_history


class TestReadingTheContractOffAFrame:
    """What a file claims about itself is a hope; what its columns contain
    is a fact."""

    def test_it_detects_versioning_from_duplicate_facts(self):
        frame = pd.DataFrame(
            [
                {
                    "entity": "AAA",
                    "event_time": "2024-09-30",
                    "available_time": "2024-10-25",
                    "eps": 1.2,
                },
                {
                    "entity": "AAA",
                    "event_time": "2024-09-30",
                    "available_time": "2025-02-10",
                    "eps": 1.05,
                },
            ]
        )
        contract = contract_for_frame(frame, source="t", frame_kind="fundamentals")
        assert contract.revisions == "versioned"
        assert contract.reproduces_history

    def test_one_version_of_everything_proves_nothing(self):
        """Seeing no restatement is not evidence that restatements would be
        kept, so the encoding stays unknown rather than being upgraded."""
        frame = pd.DataFrame(
            [
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
        contract = contract_for_frame(frame, source="t", frame_kind="fundamentals")
        assert contract.revisions == "unknown"
        assert contract.pit_safe
        assert not contract.reproduces_history

    def test_a_frame_without_availability_is_read_as_unsafe(self):
        frame = pd.DataFrame(
            [{"entity": "AAA", "event_time": "2024-09-30", "eps": 1.2}]
        )
        contract = contract_for_frame(frame, source="t", frame_kind="fundamentals")
        assert not contract.pit_safe


class TestPriceContract:
    def test_bars_are_safe_by_construction(self):
        contract = price_contract("yfinance")
        assert contract.pit_safe and contract.reproduces_history
        assert contract.revisions == "none"

    def test_it_says_not_to_carry_the_assumption_elsewhere(self):
        """The whole reason this is stated rather than assumed: the
        assumption is true for bars and false for every filing."""
        assert any(
            "do not carry the assumption" in note.lower()
            for note in price_contract("yfinance").notes
        )


class TestTheTypeRefusesNonsense:
    def test_an_unknown_frame_kind_is_rejected(self):
        with pytest.raises(Exception, match="frame_kind"):
            _contract(frame_kind="vibes")

    def test_an_unknown_revision_encoding_is_rejected(self):
        with pytest.raises(Exception, match="revisions"):
            _contract(revisions="sometimes")

    def test_an_unknown_field_is_rejected(self):
        with pytest.raises(Exception):
            _contract(has_revision_time=True)

    @pytest.mark.parametrize("kind", FRAME_KINDS)
    def test_every_declared_frame_kind_constructs(self, kind):
        assert _contract(frame_kind=kind).frame_kind == kind

    @pytest.mark.parametrize("encoding", REVISION_ENCODINGS)
    def test_every_declared_encoding_constructs(self, encoding):
        assert _contract(revisions=encoding).revisions == encoding


class TestTheProviderHook:
    def test_bars_are_safe_and_everything_else_is_declared_unsupported(self):
        from standard_quant_tools.data.factory import DataFactory

        provider = DataFactory.get_provider("yfinance")
        assert provider.get_temporal_contract("bars").pit_safe
        for kind in ("fundamentals", "estimates", "macro", "events"):
            contract = provider.get_temporal_contract(kind)
            assert not contract.pit_safe, kind
            assert contract.caveats(), f"{kind} refused without saying why"

    def test_it_is_not_abstract(self):
        """A default that every provider must implement would make adding a
        provider harder for no gain -- the honest default is knowable."""
        from standard_quant_tools.data.base import DataProvider

        assert not getattr(
            DataProvider.get_temporal_contract, "__isabstractmethod__", False
        )

    def test_the_refusal_blames_the_provider_not_the_library(self):
        """The join is built and tested; what is missing is a source. An
        agent told 'point-in-time is unsupported' would reasonably stop
        looking for one."""
        from standard_quant_tools.data.factory import DataFactory

        contract = DataFactory.get_provider("yfinance").get_temporal_contract(
            "fundamentals"
        )
        assert any("about the provider" in note for note in contract.notes)


class TestTheTool:
    def test_it_answers_without_fetching(self):
        from standard_quant_tools.agent.runtimes import resolve

        result = resolve("meta").dispatch(
            "describe_temporal_contract", {"frame_kind": "fundamentals"}
        )
        assert result["pit_safe"] is False
        assert result["caveats"]

    def test_bars_come_back_safe(self):
        from standard_quant_tools.agent.runtimes import resolve

        result = resolve("meta").dispatch(
            "describe_temporal_contract", {"frame_kind": "bars"}
        )
        assert result["pit_safe"] is True
        assert result["reproduces_history"] is True

    def test_an_unknown_argument_is_rejected(self):
        from standard_quant_tools.agent.models import TemporalContractInput

        with pytest.raises(Exception):
            TemporalContractInput(frame_kind="bars", provider="yfinance")

    def test_it_defaults_to_the_same_provider_as_its_sibling(self):
        """Two discovery tools describing different providers by default
        would be a trap: an agent would read one and act on the other."""
        from standard_quant_tools.agent.models import (
            DataCapabilitiesInput,
            TemporalContractInput,
        )

        assert (
            TemporalContractInput.model_fields["source"].default
            == DataCapabilitiesInput.model_fields["source"].default
        )
