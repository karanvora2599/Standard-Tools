"""
The two statistics the library computed on every call and threw away:
whether a prediction band covered, and whether one signal beat another
once the family of tests it belongs to is accounted for.

Every oracle here is planted rather than recomputed by the test. Eighteen
of twenty rows inside the band is a coverage of exactly 0.9 and a pinball
loss worked out by hand from the definition; one crossed row in four is a
crossing rate of exactly 0.25; the three multiple-testing corrections on
`[0.01, 0.04, 0.03]` are the hand-worked triples `[0.03, 0.06, 0.06]`,
`[0.03, 0.12, 0.09]` and `[0.03, 0.04, 0.04]`; a signal compared against
itself has a difference of exactly zero on every date. Each detector also
gets a null case -- a band whose coverage is identical on every date must
raise no regime warning, a white-noise difference must block like an IID
resample, and a family of pure noise must reject nothing -- because a
tool that warns about everything and a tool that warns about nothing are
equally useless.

One thing is deliberately NOT a single-seed assertion: how often Holm
rejects under the complete null. That is a 5% event, so counting it over
forty draws has a real chance of landing on six rather than two whatever
the code does. The seed base is fixed and named, and the claim that
actually holds is checked over four hundred draws beside it.
"""

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.statistics_models import (
    CompareSignalsInput,
    ScorePredictionIntervalsInput,
)
from standard_quant_tools.modeling.agent.statistics_tools import (
    compare_signals,
    score_prediction_intervals,
)
from standard_quant_tools.modeling.validation.comparison import (
    bh_adjust,
    bonferroni_adjust,
    holm_adjust,
)

#: Forty draws of a ~5% event; the base is stated because the count over
#: forty is itself noisy (two is the expectation, six is a one-in-thirty
#: draw), and a test that hid the base would be reporting a statistic it
#: had chosen the sample for. The stable claim is checked over 400 draws.
_NULL_SEED_BASE = 100


def _publish(frame: pd.DataFrame, name: str) -> str:
    return handoff.publish(
        frame, kind="predictions", run_id="interval_and_signal_checks", name=name
    )


def _panel(n_dates: int, n_entities: int, start: str = "2023-01-02") -> pd.DataFrame:
    """A (date, entity) skeleton with nothing in it but the keys."""
    dates = np.repeat(pd.bdate_range(start, periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    return pd.DataFrame({"date": dates, "entity": entities})


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + rng.normal(0.0, sigma * math.sqrt(1 - phi**2))
    return out


def _ic_map(values: np.ndarray, start: str = "2021-01-04") -> dict:
    dates = pd.bdate_range(start, periods=len(values))
    return {
        date.strftime("%Y-%m-%d"): float(value) for date, value in zip(dates, values)
    }


def _ci_width(result) -> float:
    """How wide the interval on the mean difference came back."""
    return result.comparison["ci_upper"] - result.comparison["ci_lower"]


def _step_up_rejections(p_values, alpha: float):
    """
    The Benjamini-Hochberg procedure written out as a rejection rule.

    Deliberately NOT the adjusted-p-value route the library takes: find
    the largest rank `k` whose `p_(k) <= k / m * alpha` and reject every
    test at rank `k` or below. The library returns a monotone adjusted
    p-value instead, and the claim worth testing is that comparing that
    number to alpha lands on the same set this does -- a second,
    independent spelling of the procedure is the only way to check it.
    """
    ordered = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    cutoff = 0
    for rank, index in enumerate(ordered, start=1):
        if p_values[index] <= rank / m * alpha:
            cutoff = rank
    rejected = [False] * m
    for rank, index in enumerate(ordered, start=1):
        rejected[index] = rank <= cutoff
    return rejected


class TestWhatTheBandCovered:
    def test_eighteen_of_twenty_rows_inside_the_band_is_a_coverage_of_point_nine(self):
        """Coverage and pinball both against a hand computation.

        Eighteen outcomes at +-0.5 sit inside [-1, 1]; 2.0 and -3.0 do
        not. The median quantile is 0 everywhere, so pinball at 0.5 is
        half the mean absolute outcome: 0.5 * 14/20 = 0.35. At 0.05
        against a prediction of -1, an outcome above it costs
        0.05 * (y + 1) and the one below costs 0.95 * 2.0, giving
        (9*0.075 + 9*0.025 + 0.15 + 1.9) / 20 = 0.1475.
        """
        outcomes = [0.5, -0.5] * 9 + [2.0, -3.0]
        frame = _panel(4, 5).assign(
            target=outcomes, q05=-1.0, q50=0.0, q95=1.0, lower=-1.0, upper=1.0
        )
        result = score_prediction_intervals(
            ScorePredictionIntervalsInput(
                predictions_ref=_publish(frame, "band_covering_18_of_20")
            )
        )
        assert result.n_rows == 20
        assert result.interval_coverage == pytest.approx(0.9)
        assert result.interval_nominal_coverage == pytest.approx(0.9)
        assert result.interval_width == pytest.approx(2.0)
        assert result.pinball["pinball_q50"] == pytest.approx(0.35)
        assert result.pinball["pinball_q05"] == pytest.approx(0.1475)
        assert result.quantile_levels == {"q05": 0.05, "q50": 0.5, "q95": 0.95}
        assert result.quantile_crossing_rate == pytest.approx(0.0)
        # Exactly on the claim, so no band is reported as broken.
        assert not any("below the claimed" in note for note in result.warnings)

    def test_a_band_that_covers_everything_is_exposed_by_its_width(self):
        """Coverage alone cannot distinguish a calibrated band from a
        useless one, which is why the width is a field and why nothing
        here returns a verdict computed from coverage."""
        rng = np.random.default_rng(11)
        outcomes = rng.normal(size=40)
        spread = float(np.percentile(outcomes, 75) - np.percentile(outcomes, 25))
        half = 5.0 * spread  # a band ten interquartile ranges wide
        frame = _panel(8, 5).assign(target=outcomes, lower=-half, upper=half)
        result = score_prediction_intervals(
            ScorePredictionIntervalsInput(
                predictions_ref=_publish(frame, "band_ten_iqrs_wide")
            )
        )
        assert result.interval_coverage == pytest.approx(1.0)
        assert result.interval_width / spread == pytest.approx(10.0)
        # Nothing in the result reads as an endorsement: there is no
        # verdict field, and the only signal that this is not a good band
        # is the width beside the coverage.
        assert not hasattr(result, "verdict")
        assert any(
            "bought with width" in note for note in result.warnings
        ), result.warnings

    def test_identical_coverage_on_every_date_has_no_spread_and_no_regime_warning(self):
        """The null case for the by= axis: ten dates that each cover nine
        of ten rows have nothing to separate, and the tool must say
        nothing rather than find a regime in rounding."""
        frame = _panel(10, 10).assign(lower=-1.0, upper=1.0)
        frame["target"] = np.tile([0.5] * 9 + [5.0], 10)
        ref = _publish(frame, "identical_coverage_every_date")

        per_date = score_prediction_intervals(
            ScorePredictionIntervalsInput(
                predictions_ref=ref, by="date", min_group_rows=10
            )
        )
        assert per_date.n_groups == 10
        coverages = [group.interval_coverage for group in per_date.groups]
        assert max(coverages) - min(coverages) == pytest.approx(0.0)
        assert per_date.interval_coverage == pytest.approx(0.9)
        assert not any("VARIES ACROSS" in note for note in per_date.warnings)
        assert not any("exchangeab" in note for note in per_date.warnings)

        pooled = score_prediction_intervals(
            ScorePredictionIntervalsInput(predictions_ref=ref, by="all")
        )
        assert pooled.n_groups == 0
        assert any("exchangeable" in note for note in pooled.warnings)
        assert any("97%" in note and "62%" in note for note in pooled.warnings)

    def test_a_regime_the_band_did_not_survive_shows_up_per_date(self):
        """The other side of the null: one date where the band covers
        two of ten must be findable, and the pooled number hides it."""
        frame = _panel(10, 10).assign(lower=-1.0, upper=1.0)
        outcomes = np.tile(np.array([0.5] * 9 + [5.0]), 10).astype(float)
        outcomes[-10:] = [0.5, 0.5] + [7.0] * 8  # the last date covers 2 of 10
        frame["target"] = outcomes
        ref = _publish(frame, "one_date_the_band_missed")
        result = score_prediction_intervals(
            ScorePredictionIntervalsInput(
                predictions_ref=ref, by="date", min_group_rows=10
            )
        )
        assert result.worst_group == max(
            group.key for group in result.groups
        )  # the last date
        assert result.groups[-1].interval_coverage == pytest.approx(0.2)
        assert any("VARIES ACROSS" in note for note in result.warnings)

    def test_one_crossed_row_in_four_is_a_quarter_and_says_the_width_is_negative(self):
        """The 5th percentile predicted above the 95th on row three: the
        pair is not a distribution there, and the mean width is dragged
        down by a negative contribution."""
        frame = _panel(4, 1).assign(
            target=[0.0, 0.0, 0.0, 10.0],
            q05=[-1.0, -1.0, 1.0, -1.0],
            q50=[0.0, 0.0, 0.0, 0.0],
            q95=[1.0, 1.0, 0.5, 1.0],
        )
        result = score_prediction_intervals(
            ScorePredictionIntervalsInput(
                predictions_ref=_publish(frame, "quantiles_crossed_on_one_row")
            )
        )
        assert result.quantile_crossing_rate == pytest.approx(0.25)
        assert result.quantile_coverage["quantile_coverage_90"] == pytest.approx(0.5)
        assert result.quantile_width["quantile_width_90"] == pytest.approx(
            float(np.mean([2.0, 2.0, -0.5, 2.0]))
        )
        assert result.interval_coverage is None
        assert any("out of order" in note for note in result.warnings)
        assert any("NEGATIVE" in note for note in result.warnings)

    def test_short_groups_are_reported_and_flagged_rather_than_ranked(self):
        frame = _panel(6, 3).assign(target=0.0, lower=-1.0, upper=1.0)
        result = score_prediction_intervals(
            ScorePredictionIntervalsInput(
                predictions_ref=_publish(frame, "three_rows_a_date"),
                by="date",
                min_group_rows=20,
            )
        )
        assert result.n_groups == 6
        assert all(group.n_rows == 3 for group in result.groups)
        assert result.worst_group is None and result.best_group is None
        assert any("fewer than 20 rows" in note for note in result.warnings)


class TestTheIntervalRefusals:
    def test_a_frame_with_nothing_distributional_names_what_was_looked_for(self):
        frame = _panel(4, 2).assign(target=0.0, prediction=0.0)
        with pytest.raises(ValidationError) as excinfo:
            score_prediction_intervals(
                ScorePredictionIntervalsInput(
                    predictions_ref=_publish(frame, "point_predictions_only")
                )
            )
        message = str(excinfo.value)
        assert "'lower'" in message and "'upper'" in message
        assert "q05" in message and "q95" in message
        assert "score_predictions" in message

    def test_a_lower_edge_without_an_upper_one_is_refused(self):
        frame = _panel(4, 2).assign(target=0.0, lower=-1.0)
        with pytest.raises(ValidationError, match="upper_column"):
            score_prediction_intervals(
                ScorePredictionIntervalsInput(
                    predictions_ref=_publish(frame, "one_sided_band")
                )
            )

    def test_more_groups_than_the_cap_is_refused_naming_the_count(self):
        frame = _panel(30, 2).assign(target=0.0, lower=-1.0, upper=1.0)
        with pytest.raises(ValidationError) as excinfo:
            score_prediction_intervals(
                ScorePredictionIntervalsInput(
                    predictions_ref=_publish(frame, "thirty_dates_five_allowed"),
                    by="date",
                    max_groups=5,
                )
            )
        assert "30 groups" in str(excinfo.value)
        assert "max_groups" in str(excinfo.value)

    def test_grouping_by_entity_without_an_entity_column_is_refused(self):
        frame = (
            _panel(4, 2)
            .drop(columns=["entity"])
            .assign(target=0.0, lower=-1.0, upper=1.0)
        )
        with pytest.raises(ValidationError, match="entity"):
            score_prediction_intervals(
                ScorePredictionIntervalsInput(
                    predictions_ref=_publish(frame, "no_entity_column"), by="entity"
                )
            )

    def test_a_missing_outcome_column_is_refused(self):
        frame = _panel(4, 2).assign(lower=-1.0, upper=1.0)
        with pytest.raises(ValidationError, match="target"):
            score_prediction_intervals(
                ScorePredictionIntervalsInput(
                    predictions_ref=_publish(frame, "band_without_outcomes")
                )
            )

    def test_every_row_missing_its_outcome_is_refused(self):
        frame = _panel(4, 2).assign(target=np.nan, lower=-1.0, upper=1.0)
        with pytest.raises(ValidationError, match="coverage"):
            score_prediction_intervals(
                ScorePredictionIntervalsInput(
                    predictions_ref=_publish(frame, "outcomes_not_realized_yet")
                )
            )


class TestTheAdjustments:
    def test_the_three_corrections_on_a_hand_worked_triple(self):
        p_values = {"momentum": 0.01, "value": 0.04, "carry": 0.03}
        expected = {
            "holm": {"momentum": 0.03, "value": 0.06, "carry": 0.06},
            "bonferroni": {"momentum": 0.03, "value": 0.12, "carry": 0.09},
            "bh": {"momentum": 0.03, "value": 0.04, "carry": 0.04},
        }
        for method, oracle in expected.items():
            result = compare_signals(
                CompareSignalsInput(mode="adjust", p_values=p_values, method=method)
            )
            assert result.method == method
            assert result.n_tests == 3
            adjusted = {row.label: row.p_adjusted for row in result.adjusted}
            for label, value in oracle.items():
                assert adjusted[label] == pytest.approx(value), (method, label)
            assert {row.label for row in result.adjusted} == set(p_values)
        # The library functions and the tool agree, because the tool is a
        # door onto them and computes nothing of its own.
        raw = [0.01, 0.04, 0.03]
        assert holm_adjust(raw) == pytest.approx([0.03, 0.06, 0.06])
        assert bonferroni_adjust(raw) == pytest.approx([0.03, 0.12, 0.09])
        assert bh_adjust(raw) == pytest.approx([0.03, 0.04, 0.04])
        assert holm_adjust([]) == [] and bonferroni_adjust([]) == []
        assert bh_adjust([]) == []

    def test_an_empty_family_is_refused_by_the_schema(self):
        with pytest.raises(PydanticValidationError, match="at least 1"):
            CompareSignalsInput(mode="adjust", p_values={})

    def test_a_p_value_outside_the_unit_interval_is_refused(self):
        with pytest.raises(PydanticValidationError):
            CompareSignalsInput(mode="adjust", p_values={"a": 1.4})

    def test_twelve_noise_p_values_reject_nothing_after_holm(self):
        """Under the complete null Holm rejects at about 5%, so the count
        over forty draws is itself a random variable -- 400 draws is the
        claim, and the forty is the planted case at a stated base."""
        rejecting = []
        for seed in range(_NULL_SEED_BASE, _NULL_SEED_BASE + 40):
            draws = np.random.default_rng(seed).uniform(size=12)
            result = compare_signals(
                CompareSignalsInput(
                    mode="adjust",
                    p_values={f"signal_{i:02d}": float(p) for i, p in enumerate(draws)},
                )
            )
            assert result.n_tests == 12
            if result.n_rejected:
                rejecting.append(seed)
        assert len(rejecting) <= 2, f"{len(rejecting)} of 40 null families rejected"

        wide = sum(
            1
            for seed in range(400)
            if any(
                value <= 0.05
                for value in holm_adjust(np.random.default_rng(seed).uniform(size=12))
            )
        )
        assert 0 < wide / 400 <= 0.10, wide

    def test_a_planted_signal_survives_the_correction_that_noise_does_not(self):
        """The null case for the null case: a detector that never rejects
        is as useless as one that always does."""
        p_values = {f"noise_{i:02d}": 0.4 + 0.01 * i for i in range(11)}
        p_values["planted"] = 0.0001
        result = compare_signals(CompareSignalsInput(mode="adjust", p_values=p_values))
        assert result.n_rejected == 1
        assert result.adjusted[0].label == "planted"
        assert result.adjusted[0].reject_at_alpha is True

    def test_only_the_false_discovery_rate_method_says_it_is_one(self):
        p_values = {"a": 0.001, "b": 0.02, "c": 0.3}
        fdr = compare_signals(
            CompareSignalsInput(mode="adjust", p_values=p_values, method="bh")
        )
        fwer = compare_signals(
            CompareSignalsInput(mode="adjust", p_values=p_values, method="holm")
        )
        assert any("FALSE DISCOVERY RATE" in note for note in fdr.warnings)
        assert any("DIFFERENT" in note for note in fdr.warnings)
        assert not any("FALSE DISCOVERY RATE" in note for note in fwer.warnings)
        # Both carry the selection caveat, whichever quantity they control.
        for result in (fdr, fwer):
            assert any("run_reality_check" in note for note in result.warnings)
            assert any("SELECTED" in note for note in result.warnings)

    def test_the_monotone_pass_is_what_makes_the_adjusted_value_decidable(self):
        """The rank-scaled values are not monotone on their own: on two
        tests the smaller p-value 0.03 scales to 2/1 * 0.03 = 0.06 while
        the larger 0.04 scales to 2/2 * 0.04 = 0.04. The step-up sweep
        replaces the larger with the smaller. Without it, both are still
        rejected at 0.05 -- the procedure rejects a prefix of the
        ordering -- and `p_adjusted <= alpha` would say otherwise for the
        first, which is the disagreement the sweep removes."""
        result = compare_signals(
            CompareSignalsInput(
                mode="adjust", p_values={"a": 0.03, "b": 0.04}, method="bh"
            )
        )
        by_label = {row.label: row for row in result.adjusted}
        # Raw, 'a' would be 2/1 * 0.03 = 0.06 and would miss alpha while
        # the larger p-value cleared it; the sweep brings it to 0.04.
        assert by_label["a"].p_adjusted == pytest.approx(0.04)
        assert by_label["b"].p_adjusted == pytest.approx(0.04)
        assert by_label["a"].reject_at_alpha is True
        assert result.n_rejected == 2
        ordered = sorted(result.adjusted, key=lambda row: row.p_value)
        values = [row.p_adjusted for row in ordered]
        assert values == sorted(values)

    def test_the_adjusted_value_and_the_step_up_procedure_never_disagree(self):
        """`reject_at_alpha` is `p_adjusted <= alpha` for every method, so
        the one claim worth checking is that this equals what the
        Benjamini-Hochberg procedure itself rejects -- on random families,
        not one hand-picked one."""
        rng = np.random.default_rng(42)
        for trial in range(200):
            size = int(rng.integers(1, 15))
            raw = np.round(rng.uniform(size=size), 4)
            alpha = float(rng.choice([0.01, 0.05, 0.1, 0.2]))
            result = compare_signals(
                CompareSignalsInput(
                    mode="adjust",
                    p_values={f"t{i:02d}": float(p) for i, p in enumerate(raw)},
                    method="bh",
                    alpha=alpha,
                )
            )
            decided = {row.label: row.reject_at_alpha for row in result.adjusted}
            expected = _step_up_rejections(raw, alpha)
            assert decided == {f"t{i:02d}": expected[i] for i in range(size)}, (
                trial,
                raw,
                alpha,
            )
            # And monotone, which is what allows the comparison at all.
            ordered = sorted(result.adjusted, key=lambda row: row.p_value)
            values = [row.p_adjusted for row in ordered]
            assert values == sorted(values), (trial, raw)


class TestThePairedComparison:
    def _frame(self, seed: int = 0, n_entities: int = 12, n_dates: int = 40):
        rng = np.random.default_rng(seed)
        frame = _panel(n_dates, n_entities)
        signal = rng.normal(size=len(frame))
        frame["prediction"] = signal
        frame["target"] = signal + rng.normal(scale=0.5, size=signal.size)
        return frame

    def test_a_signal_against_itself_is_indistinguishable(self):
        ref = _publish(self._frame(), "signal_compared_to_itself")
        result = compare_signals(
            CompareSignalsInput(
                mode="paired",
                predictions_ref_a=ref,
                predictions_ref_b=ref,
                task="regression",
            )
        )
        assert result.mode == "paired"
        assert result.comparison["mean_difference"] == 0.0
        assert result.comparison["verdict"] == "indistinguishable"
        # NaN when every date tied, which is what a twin produces and
        # what a share of ALL dates would read as 'b lost every day'.
        assert result.comparison["hit_rate"] is None
        assert result.comparison["n_ties"] == result.comparison["n_dates"]
        assert result.n_tests == 1 and result.hac is None
        assert any("run_reality_check" in note for note in result.warnings)
        assert any("ONE test" in note for note in result.warnings)

    def test_a_better_signal_is_found_and_the_difference_is_paired(self):
        base = self._frame(seed=5)
        better = base.copy()
        better["prediction"] = base["target"] + np.random.default_rng(6).normal(
            scale=0.05, size=len(base)
        )
        result = compare_signals(
            CompareSignalsInput(
                mode="paired",
                predictions_ref_a=_publish(base, "baseline_signal"),
                predictions_ref_b=_publish(better, "near_oracle_signal"),
                task="regression",
                n_bootstrap=400,
            )
        )
        assert result.comparison["verdict"] == "b_better"
        assert result.comparison["mean_difference"] > 0
        assert result.comparison["ci_lower"] > 0
        assert result.comparison["diebold_mariano"]["loss"] == "squared_error"


class TestTheIcSeries:
    def test_the_block_interval_is_wider_on_an_autocorrelated_difference(self):
        rng = np.random.default_rng(3)
        flat = _ic_map(np.zeros(400))
        persistent = _ic_map(_ar1(400, 0.8, 0.02, rng))
        blocked = compare_signals(
            CompareSignalsInput(mode="ic_series", ic_a=flat, ic_b=persistent, seed=1)
        )
        iid = compare_signals(
            CompareSignalsInput(
                mode="ic_series", ic_a=flat, ic_b=persistent, seed=1, block_size=1
            )
        )
        assert _ci_width(blocked) > _ci_width(iid)
        assert blocked.comparison["n_dates"] == 400
        assert blocked.comparison["block_size"] > 1

    def test_white_noise_blocks_like_an_iid_resample(self):
        """The null case: with no serial correlation there is nothing for
        a block to preserve, so the two intervals agree."""
        rng = np.random.default_rng(7)
        flat = _ic_map(np.zeros(400))
        noise = _ic_map(rng.normal(0.0, 0.02, 400))
        blocked = compare_signals(
            CompareSignalsInput(mode="ic_series", ic_a=flat, ic_b=noise, seed=1)
        )
        iid = compare_signals(
            CompareSignalsInput(
                mode="ic_series", ic_a=flat, ic_b=noise, seed=1, block_size=1
            )
        )
        assert _ci_width(blocked) == pytest.approx(_ci_width(iid), rel=0.2)

    def test_the_hac_ratio_rises_on_a_persistent_series_and_not_on_noise(self):
        rng = np.random.default_rng(3)
        flat = _ic_map(np.zeros(400))
        persistent = compare_signals(
            CompareSignalsInput(
                mode="ic_series",
                ic_a=flat,
                ic_b=_ic_map(_ar1(400, 0.8, 0.02, rng)),
                seed=1,
                n_bootstrap=200,
            )
        )
        noise = compare_signals(
            CompareSignalsInput(
                mode="ic_series",
                ic_a=flat,
                ic_b=_ic_map(np.random.default_rng(7).normal(0.0, 0.02, 400)),
                seed=1,
                n_bootstrap=200,
            )
        )
        assert persistent.hac["hac_ratio"] > 1.0
        assert persistent.hac["hac_variance"] > persistent.hac["hac_variance_lag0"]
        # floor(4 * (400/100)^(2/9)) = 5 on four hundred dates.
        assert persistent.hac["hac_lag"] == pytest.approx(5.0)
        assert noise.hac["hac_ratio"] == pytest.approx(1.0, abs=0.25)
        assert any("autocorrelated" in note for note in persistent.warnings)
        assert not any("autocorrelated" in note for note in noise.warnings)
        # lag 0 is the ordinary variance of the mean, by construction.
        at_zero = compare_signals(
            CompareSignalsInput(
                mode="ic_series",
                ic_a=flat,
                ic_b=_ic_map(_ar1(400, 0.8, 0.02, np.random.default_rng(3))),
                seed=1,
                n_bootstrap=200,
                hac_lag=0,
            )
        )
        assert at_zero.hac["hac_ratio"] == pytest.approx(1.0)

    def test_a_short_series_carries_the_caveat_the_library_states(self):
        rng = np.random.default_rng(9)
        a = _ic_map(rng.normal(0.02, 0.05, 20))
        b = _ic_map(rng.normal(0.02, 0.05, 20))
        result = compare_signals(
            CompareSignalsInput(mode="ic_series", ic_a=a, ic_b=b, n_bootstrap=200)
        )
        assert result.comparison["n_dates"] == 20
        assert any("not enough dates to tell" in note for note in result.warnings)

    def test_fewer_than_ten_shared_dates_is_refused(self):
        rng = np.random.default_rng(10)
        a = _ic_map(rng.normal(size=8))
        b = _ic_map(rng.normal(size=8))
        with pytest.raises(ValidationError, match="ten dates"):
            compare_signals(CompareSignalsInput(mode="ic_series", ic_a=a, ic_b=b))


class TestTheModesStaySeparate:
    def test_a_field_from_another_mode_is_refused_by_name(self):
        with pytest.raises(ValidationError) as excinfo:
            compare_signals(
                CompareSignalsInput(
                    mode="adjust",
                    p_values={"a": 0.01},
                    ic_a={"2023-01-02": 0.1},
                    ic_b={"2023-01-02": 0.2},
                )
            )
        message = str(excinfo.value)
        assert "'ic_a'" in message and "'ic_b'" in message
        assert "ic_series" in message
        assert "p_values" not in message.split(":")[0]

    def test_a_bootstrap_knob_on_an_adjust_call_is_refused(self):
        with pytest.raises(ValidationError, match="n_bootstrap"):
            compare_signals(
                CompareSignalsInput(
                    mode="adjust", p_values={"a": 0.01}, n_bootstrap=500
                )
            )

    def test_a_mode_missing_its_own_fields_is_refused(self):
        with pytest.raises(ValidationError) as excinfo:
            compare_signals(CompareSignalsInput(mode="paired"))
        message = str(excinfo.value)
        assert "predictions_ref_a" in message and "task" in message

    def test_an_unknown_argument_is_rejected_rather_than_ignored(self):
        with pytest.raises(PydanticValidationError):
            CompareSignalsInput(mode="adjust", p_values={"a": 0.1}, alpha_level=0.1)
        with pytest.raises(PydanticValidationError):
            ScorePredictionIntervalsInput(predictions_ref="sqt://x/y/z", coverage=0.9)
