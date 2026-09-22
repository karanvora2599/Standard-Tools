"""
The delta one surface's silent parameters: a convention, five fields and a
threshold.

`analyze_roll` divided by a hard-coded 365 and named the convention
nowhere, so the two rates it returns could not be compared with a USD repo
quoted ACT/360. `daycount.CONVENTIONS` held the one-line reason to choose
each of the four and only one of them reached a schema. `monitor_spread_
stream` accepted `channel`, `label`, `warmup`, `threshold` and `slack` on
every resumed call and discarded all five, so a monitor opened on one
formula went on computing it after a caller asked for another -- and its
default threshold was the BATCH detector's 9.0, which fires on pure noise
about 45% of the time over 5,000 streamed observations. `scan_basis_
dislocations` fixed the detector's four settings at their defaults, so a
scan could not be made stricter than a single-pair call. And
`delta_one.contracts` described futures contracts no shipped source
serves.

Every number below is planted and every detector has its null case.

See the CHANGELOG entry of 2026-09-22.
"""

import importlib

import numpy as np
import pytest

from standard_quant_tools.agent.runtimes.delta_one.models import (
    BasisScanInput,
    BasisScanPair,
    RollAnalysisInput,
    SpreadMonitorInput,
)
from standard_quant_tools.agent.runtimes.delta_one.tools import (
    analyze_roll,
    monitor_spread_stream,
    scan_basis_dislocations,
)
from standard_quant_tools.delta_one.daycount import CONVENTIONS
from standard_quant_tools.delta_one.futures import roll_analysis
from standard_quant_tools.delta_one.streaming import STREAMING_THRESHOLD
from standard_quant_tools.error import ValidationError
from standard_quant_tools.mcp.catalog import build_catalog

#: A quarterly roll: ten ES-sized contracts, 25 points of calendar spread,
#: 91 calendar days between the two expiries and five left on the front.
_ROLL = dict(
    front_price=6240.0,
    next_price=6265.0,
    contracts_held=10.0,
    multiplier=50.0,
    days_to_front_expiry=5.0,
    days_between_expiries=91.0,
    cost_per_contract=2.0,
    spread_ticks=1.0,
    tick_value=12.5,
)


class TestTheRollSaysWhichDayCountItUsed:
    def test_the_same_roll_is_two_rates_under_two_conventions(self):
        """A 25-point step over 91 days, annualized two ways.

        ACT/360 reports the LOWER rate: 91 days is a larger slice of a
        360-day year, so the same step spread over more of a year is a
        smaller rate. (The familiar "ACT/360 accrues 1.4% more" is about an
        accrual at a given rate and runs the other way.)
        """
        fixed365 = analyze_roll(RollAnalysisInput(**_ROLL))
        money_market = analyze_roll(RollAnalysisInput(day_count="ACT/360", **_ROLL))

        assert fixed365.roll_yield_bps == pytest.approx(160.376, abs=1e-3)
        assert money_market.roll_yield_bps == pytest.approx(158.179, abs=1e-3)
        # The convention is on the answer, not only in the request.
        assert fixed365.day_count == "ACT/365F"
        assert money_market.day_count == "ACT/360"

    def test_the_break_even_moves_with_it_and_the_cash_does_not(self):
        fixed365 = analyze_roll(RollAnalysisInput(**_ROLL))
        money_market = analyze_roll(RollAnalysisInput(day_count="ACT/360", **_ROLL))

        # Both annualized numbers are day counts turned into rates.
        assert money_market.breakeven_annualized_rate < (
            fixed365.breakeven_annualized_rate
        )
        assert money_market.breakeven_annualized_rate / (
            fixed365.breakeven_annualized_rate
        ) == pytest.approx(360.0 / 365.0, rel=1e-12)
        # Everything in currency is a cash flow and has no year in it.
        assert money_market.net_roll_cost == fixed365.net_roll_cost
        assert money_market.cash_impact == fixed365.cash_impact
        assert money_market.execution_cost == fixed365.execution_cost

    def test_the_default_reproduces_todays_numbers(self):
        """The null case: the default is the 365 that was hard-coded."""
        default = analyze_roll(RollAnalysisInput(**_ROLL))
        spelled_out = analyze_roll(RollAnalysisInput(day_count="ACT/365F", **_ROLL))

        assert default.roll_yield_bps == 160.37562393888237
        assert default.roll_yield_rate == 0.016037562393888236
        assert default.breakeven_annualized_rate == 0.2992396671066364
        assert default.model_dump() == spelled_out.model_dump()

    def test_a_360_day_year_is_a_360_day_year_whichever_way_it_is_counted(self):
        """30/360 and ACT/360 share a denominator, and a roll is given in

        DAYS -- so here they agree, while over real dates they need not."""
        thirty = analyze_roll(RollAnalysisInput(day_count="30/360", **_ROLL))
        actual = analyze_roll(RollAnalysisInput(day_count="ACT/360", **_ROLL))
        assert thirty.roll_yield_bps == actual.roll_yield_bps

    def test_a_convention_this_library_does_not_count_is_refused_by_name(self):
        with pytest.raises(ValidationError, match="30E/360"):
            roll_analysis(day_count="30E/360", **_ROLL)

    def test_the_library_normalizes_the_spelling_it_accepts(self):
        assert roll_analysis(day_count="act/365", **_ROLL)["day_count"] == "ACT/365F"


class TestTheConventionRationalesReachTheSchema:
    """Four sentences that decide which convention a desk picks lived in

    `daycount.CONVENTIONS` with no importer outside their own module. Only
    the ACT/360 one had ever been copied into a schema, into one tool."""

    @pytest.fixture(scope="class")
    def descriptions(self):
        catalog = build_catalog()
        return {
            tool: catalog[tool].input_schema["properties"]["day_count"]["description"]
            for tool in ("analyze_roll", "price_total_return_swap")
        }

    @pytest.mark.parametrize("convention", sorted(CONVENTIONS))
    def test_both_day_count_fields_carry_every_rationale(
        self, descriptions, convention
    ):
        rationale = CONVENTIONS[convention]
        for tool, description in descriptions.items():
            assert rationale in description, f"{tool} drops the {convention} reason"

    def test_both_carry_the_no_business_day_caveat(self, descriptions):
        """There is no holiday calendar in this library, so none of the four

        adjusts a date that lands on a weekend. Saying so beats shipping a
        Modified Following that assumes weekends are the only holidays."""
        for description in descriptions.values():
            assert "CALENDAR days" in description
            assert "holiday calendar" in description


class TestAResumedMonitorCannotBeRebuilt:
    @staticmethod
    def _open(**overrides):
        rng = np.random.default_rng(5)
        reference = np.full(80, 100.0)
        primary = reference * (1 + rng.normal(30, 4, 80) / 10_000.0)
        opened = monitor_spread_stream(
            SpreadMonitorInput(
                primary_prices=list(primary),
                reference_prices=list(reference),
                channel="relative_bps",
                label="ES basis",
                **overrides,
            )
        )
        return opened.state

    def test_asking_for_a_different_formula_is_refused_with_the_remedy(self):
        """A ratio in basis points and a difference in points are different

        questions. This was accepted and ignored: the answer was still a
        ratio, with no warning that the request had been dropped."""
        state = self._open()
        with pytest.raises(ValidationError) as excinfo:
            monitor_spread_stream(
                SpreadMonitorInput(
                    primary_prices=[100.3],
                    reference_prices=[100.0],
                    state=state,
                    channel="absolute_points",
                )
            )
        message = str(excinfo.value)
        assert "absolute_points" in message and "relative_bps" in message
        # The remedy, not just the refusal.
        assert "WITHOUT `state`" in message

    @pytest.mark.parametrize(
        "field, value",
        [
            ("threshold", 9.0),
            ("slack", 2.0),
            ("warmup", 120),
            ("label", "NQ basis"),
        ],
    )
    def test_every_construction_field_is_checked_not_just_the_channel(
        self, field, value
    ):
        state = self._open()
        with pytest.raises(ValidationError, match=field):
            monitor_spread_stream(
                SpreadMonitorInput(
                    primary_prices=[100.3],
                    reference_prices=[100.0],
                    state=state,
                    **{field: value},
                )
            )

    def test_a_resume_repeating_the_same_values_is_accepted(self):
        """The null case. Restating what the monitor already is is not a

        request to change it, and a caller that echoes its own arguments
        back on every call is the ordinary way to drive this tool."""
        state = self._open()
        resumed = monitor_spread_stream(
            SpreadMonitorInput(
                primary_prices=[100.3],
                reference_prices=[100.0],
                state=state,
                channel="relative_bps",
                label="ES basis",
                warmup=60,
                threshold=STREAMING_THRESHOLD,
                slack=0.5,
            )
        )
        assert resumed.n_observations == 81

    def test_a_resume_that_sends_none_of_them_is_accepted(self):
        """The other null case: silence means "carry on", not "rebuild at

        the defaults" -- otherwise a monitor opened at a custom threshold
        could never be fed again."""
        state = self._open(threshold=20.0)
        resumed = monitor_spread_stream(
            SpreadMonitorInput(
                primary_prices=[100.3], reference_prices=[100.0], state=state
            )
        )
        assert resumed.state["threshold"] == 20.0
        assert resumed.n_observations == 81


class TestTheStreamThresholdIsTheStreamingOne:
    @staticmethod
    def _noise_alarm_rate(threshold, trials=40, n=5_000):
        """Pure noise, no shift anywhere: every alarm is a false one."""
        fired = 0
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            reference = np.full(n, 100.0)
            primary = reference * (1 + rng.normal(30, 4, n) / 10_000.0)
            arguments = dict(
                primary_prices=list(primary), reference_prices=list(reference)
            )
            if threshold is not None:
                arguments["threshold"] = threshold
            fired += bool(
                monitor_spread_stream(SpreadMonitorInput(**arguments)).triggered
            )
        return fired / trials

    def test_the_default_is_the_librarys_streaming_constant(self):
        assert SpreadMonitorInput.model_fields["threshold"].default == (
            STREAMING_THRESHOLD
        )
        assert STREAMING_THRESHOLD == 15.0

    def test_a_long_feed_of_noise_rarely_alarms_at_the_default(self):
        """Bounded, not pinned: the rate is a measurement over 40 trials and

        moves with the seeds. The batch threshold is not merely worse here,
        it is not a detector -- its baseline sharpens as its series grows
        and a stream's is frozen at `warmup`, so the statistic accumulates
        against a fixed scale forever."""
        at_default = self._noise_alarm_rate(None)
        at_batch = self._noise_alarm_rate(9.0)

        assert at_default <= 0.20, at_default
        assert at_batch >= 0.25, at_batch
        assert at_batch > 3 * at_default


class TestTheScanExposesItsDetector:
    @staticmethod
    def _pair(label, shift, seed, at=200, n=300):
        rng = np.random.default_rng(seed)
        basis_bps = np.concatenate(
            [rng.normal(25, 3, at), rng.normal(25 + shift, 3, n - at)]
        )
        spot = 6000 * np.exp(np.cumsum(rng.normal(0.0002, 0.007, n)))
        return BasisScanPair(
            label=label,
            spot=list(spot),
            futures=list(spot * (1 + basis_bps / 10_000.0)),
        )

    @staticmethod
    def _pairs():
        return [
            # Planted: a large early shift, a marginal late one whose CUSUM
            # peak is 12.3, and a pair that never moves.
            TestTheScanExposesItsDetector._pair("wide", 40.0, 1),
            TestTheScanExposesItsDetector._pair("marginal", 3.0, 7, at=290),
            TestTheScanExposesItsDetector._pair("quiet", 0.0, 3),
        ]

    def test_a_stricter_threshold_finds_no_more_shifts_and_the_same_order(self):
        pairs = self._pairs()
        calibrated = scan_basis_dislocations(BasisScanInput(pairs=pairs, threshold=9.0))
        strict = scan_basis_dislocations(BasisScanInput(pairs=pairs, threshold=25.0))

        fired_at_9 = {row.label for row in calibrated.ranked if row.shift_detected}
        fired_at_25 = {row.label for row in strict.ranked if row.shift_detected}
        assert fired_at_25 < fired_at_9, "the threshold reached nothing"
        assert "wide" in fired_at_25
        assert "marginal" in fired_at_9 and "marginal" not in fired_at_25
        assert "quiet" not in fired_at_9

        # The ranking is on |z| from the basis history and the detector
        # cannot touch it -- a level and an event are different findings.
        assert [row.label for row in strict.ranked] == [
            row.label for row in calibrated.ranked
        ]
        assert [row.zscore for row in strict.ranked] == [
            row.zscore for row in calibrated.ranked
        ]

    def test_the_detector_settings_are_echoed(self):
        scanned = scan_basis_dislocations(
            BasisScanInput(
                pairs=self._pairs(),
                reference_fraction=0.4,
                threshold=25.0,
                slack=0.75,
                max_breaks=2,
            )
        )
        assert scanned.reference_fraction == 0.4
        assert scanned.threshold == 25.0
        assert scanned.slack == 0.75
        assert scanned.max_breaks == 2
        assert scanned.detect_shifts is True

    def test_the_defaults_are_the_single_pair_detectors(self):
        """The null case: a scan asked for nothing still runs the detector

        the single-pair tool would have run."""
        scanned = scan_basis_dislocations(BasisScanInput(pairs=self._pairs()))
        assert (
            scanned.reference_fraction,
            scanned.threshold,
            scanned.slack,
            scanned.max_breaks,
        ) == (0.3, 9.0, 0.5, 3)
        assert {row.label for row in scanned.ranked if row.shift_detected} == {
            "wide",
            "marginal",
        }


class TestTheContractSpecificationModuleIsGone:
    def test_it_no_longer_imports(self):
        """177 lines with no production caller. Every method was correct and

        every one needed a multiplier and a tick size the caller already
        had: no shipped provider serves contract metadata, so there was
        nothing to read one from. Each tool on this surface takes the
        multiplier as an argument instead."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("standard_quant_tools.delta_one.contracts")
