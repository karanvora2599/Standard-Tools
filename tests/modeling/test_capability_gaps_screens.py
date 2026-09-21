"""
The capability-gaps fixes of 2026-09-21 (CHANGELOG): the feature lab's two
panel-wide screens.

`screen_feature_significance` closes a loop the library left open. The
permutation test's own docstring calls `null_p95_abs` "the honest floor for
`select_features(min_abs_rank_ic=...)`" and answers for ONE feature, while
`select_features` takes that floor as a number the caller invents. The
screen runs the test across the whole candidate set and reports the largest
null as `honest_floor`, which usually inverts the guess: a floor picked
because 0.02 sounds small keeps whatever the panel's noise happens to
produce.

`screen_feature_stability` closes a different one. `get_feature_drift` and
`get_feature_regime_stability` are single-feature and `analyze_features` is
silent about time, so a feature that stopped being the same measurement is
invisible unless somebody already suspected it -- and a single split says
"it moved" without saying when or how fast, which is why the drift CURVE
(`psi_by_block`) is folded in here rather than made a tool of its own.

WHAT THESE TESTS PIN. Panels whose answer is known before the tool runs: a
feature built as `strength * target + noise` among nine pure-noise columns,
a column shifted +3 sigma in its second half, a feature whose distribution
never moves while its IC flips sign, a feature that slides a little in every
block. The assertions are RELATIONS between the planted quantity and the
reported one -- the floor clears every noise |IC|, the curve rises against a
fixed reference and does not against a rolling one -- rather than constants,
because a constant would pin this machine's draw of the same null.

Every detector has a null case beside it: a stationary panel for the drift
warning, a steady feature for the decay warning, `circular_shift` for the
`within_date` caveat.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.feature_models import (
    ScreenFeatureSignificanceInput,
    ScreenFeatureStabilityInput,
)
from standard_quant_tools.modeling.agent.feature_tools import (
    feature_dispatch,
    screen_feature_significance,
    screen_feature_stability,
)
from standard_quant_tools.modeling.agent.models import RegisterExternalPanelInput
from standard_quant_tools.modeling.agent.tools import register_external_panel
from standard_quant_tools.modeling.analysis.feature_stability import PSI_SIGNIFICANT

# ── planted panels ───────────────────────────────────────────────────────


def _register(frame: pd.DataFrame, tmp_path, name: str = "panel") -> str:
    """Register a planted frame as a dataset.

    By reference, because these panels are built here rather than fetched:
    registration synthesizes a real DatasetSpec from the columns and the
    screens then reach them through the same `dataset_id` path every other
    feature tool uses.
    """
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return register_external_panel(
        RegisterExternalPanelInput(path=str(path), horizon=5)
    ).dataset_id


def _dates(n_dates: int, n_entities: int):
    index = pd.bdate_range("2022-01-03", periods=n_dates)
    return (
        np.repeat(index.to_numpy(), n_entities),
        np.tile([f"E{j}" for j in range(n_entities)], n_dates),
    )


def _one_signal_among_noise(
    *, n_noise: int = 9, n_dates: int = 120, n_entities: int = 20, seed: int = 7
) -> pd.DataFrame:
    """One real feature and nine that are not.

    `planted` is `0.35 * target + noise`, so its cross-sectional IC is real
    and the same on every date. The nine `noise_*` columns are independent
    of the target everywhere, so any IC they show is the panel's noise
    level -- which is exactly the quantity `honest_floor` claims to measure.
    """
    rng = np.random.default_rng(seed)
    dates, entities = _dates(n_dates, n_entities)
    n = n_dates * n_entities
    target = rng.normal(size=n)
    data = {
        "date": dates,
        "entity": entities,
        "target": target,
        "planted": 0.35 * target + rng.normal(size=n),
    }
    for k in range(n_noise):
        data[f"noise_{k}"] = rng.normal(size=n)
    return pd.DataFrame(data)


def _only_noise(
    *, n_noise: int = 10, n_dates: int = 120, n_entities: int = 12, seed: int = 19
) -> pd.DataFrame:
    """Ten columns with nothing in them. The whole screen's null case."""
    rng = np.random.default_rng(seed)
    dates, entities = _dates(n_dates, n_entities)
    n = n_dates * n_entities
    data = {"date": dates, "entity": entities, "target": rng.normal(size=n)}
    for k in range(n_noise):
        data[f"noise_{k}"] = rng.normal(size=n)
    return pd.DataFrame(data)


def _autocorrelated_ic(
    *, n_dates: int = 120, n_entities: int = 25, seed: int = 5
) -> pd.DataFrame:
    """A feature whose PER-DATE IC is autocorrelated.

    The strength of the relationship follows an AR(1) in time rather than
    being constant, so consecutive dates share it and the observed IC
    series has real lag-1 autocorrelation. That is the regime where the
    two nulls disagree: shuffling within a date destroys the serial
    correlation that the observed ICs have, and the resulting p-values are
    too small.
    """
    rng = np.random.default_rng(seed)
    dates, entities = _dates(n_dates, n_entities)
    n = n_dates * n_entities
    strength = np.empty(n_dates)
    strength[0] = 0.0
    for t in range(1, n_dates):
        strength[t] = 0.95 * strength[t - 1] + rng.normal(0, 0.35)
    feature = rng.normal(size=n)
    target = np.repeat(strength, n_entities) * feature + rng.normal(size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "target": target,
            "drifting_strength": feature,
        }
    )


def _drifting_panel(
    *, n_dates: int = 160, n_entities: int = 12, seed: int = 3
) -> pd.DataFrame:
    """Four columns, four different behaviours, all known in advance.

    - `shifted` jumps +3 sigma halfway through: a break.
    - `creeping` slides by the same amount in every block: a drift.
    - `steady` predicts the target and keeps doing it.
    - `plain` is noise that never moves.
    """
    rng = np.random.default_rng(seed)
    dates, entities = _dates(n_dates, n_entities)
    n = n_dates * n_entities
    second_half = np.repeat(np.arange(n_dates) >= n_dates // 2, n_entities)
    block = np.repeat(np.arange(n_dates) // (n_dates // 4), n_entities)
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "target": 0.4 * signal + rng.normal(size=n),
            "steady": signal,
            "plain": rng.normal(size=n),
            "shifted": rng.normal(size=n) + 3.0 * second_half,
            "creeping": rng.normal(size=n) + 1.2 * block,
        }
    )


def _edge_dies_panel(
    *, n_dates: int = 160, n_entities: int = 12, seed: int = 8
) -> pd.DataFrame:
    """A feature whose DISTRIBUTION never moves and whose IC flips sign.

    The two failures have to stay separable: this one is invisible to PSI
    and fatal to the model, and a screen that reported only drift would
    call it healthy.
    """
    rng = np.random.default_rng(seed)
    dates, entities = _dates(n_dates, n_entities)
    n = n_dates * n_entities
    second_half = np.repeat(np.arange(n_dates) >= n_dates // 2, n_entities)
    feature = rng.normal(size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "target": np.where(second_half, -0.45, 0.45) * feature + rng.normal(size=n),
            "inverting": feature,
        }
    )


def _fading_panel(
    *, n_dates: int = 160, n_entities: int = 12, seed: int = 11
) -> pd.DataFrame:
    """A feature that keeps its sign in every block and loses its size.

    Sign consistency stays high all the way down, which is the number an
    agent reads first and the one that misses this.
    """
    rng = np.random.default_rng(seed)
    dates, entities = _dates(n_dates, n_entities)
    n = n_dates * n_entities
    block = np.repeat(np.arange(n_dates) // (n_dates // 4), n_entities)
    feature = rng.normal(size=n)
    strength = np.choose(block, [0.6, 0.55, 0.02, 0.01])
    return pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "target": strength * feature + rng.normal(size=n),
            "fading": feature,
        }
    )


# ── the floor is measured, not guessed ───────────────────────────────────


class TestTheHonestFloorIsAPropertyOfThePanel:
    @pytest.fixture
    def planted(self, tmp_path):
        dataset_id = _register(_one_signal_among_noise(), tmp_path, "planted")
        return screen_feature_significance(
            ScreenFeatureSignificanceInput(dataset_id=dataset_id, n_permutations=200)
        )

    def test_the_planted_feature_is_the_only_one_that_clears_the_floor(self, planted):
        """PLANTED: one feature is `0.35 * target + noise` and nine are
        noise. The only defensible answers are that the first is
        significant, the other nine are not, and the floor sits above every
        |IC| the nine produced -- which is what makes it a floor rather
        than a number."""
        significant = [r.feature for r in planted.features if r.significant_at_05]
        assert significant == ["planted"]
        assert planted.n_significant == 1

        noise = [r for r in planted.features if r.feature != "planted"]
        assert len(noise) == 9
        assert planted.honest_floor is not None
        for row in noise:
            assert abs(row.rank_ic) < planted.honest_floor, (
                f"{row.feature} has |IC| {abs(row.rank_ic):.4f} at or above the "
                f"floor {planted.honest_floor:.4f}, which would make the floor "
                "one that keeps noise"
            )
        assert planted.n_kept_at_floor == 1
        assert planted.floor_feature in {r.feature for r in planted.features}

    def test_the_result_says_what_it_cost_and_which_null_it_asked(self, planted):
        assert planted.n_features == 10
        assert planted.n_draws == 10 * 200
        assert planted.null == "circular_shift"
        assert planted.random_seed == 0
        assert all(r.n_usable_permutations > 0 for r in planted.features)

    def test_ten_noise_features_keep_nothing_and_the_warning_says_so(self, tmp_path):
        """The inversion the screen exists for. A floor of 0.02 keeps
        several of ten columns that are noise by construction; the measured
        floor keeps none of them. Asserted as the RELATION between the two
        counts, because the exact number is this panel's draw of the same
        null."""
        dataset_id = _register(_only_noise(), tmp_path, "noise_only")
        result = screen_feature_significance(
            ScreenFeatureSignificanceInput(dataset_id=dataset_id, n_permutations=200)
        )
        assert result.n_significant <= 1, (
            "ten independent noise columns at alpha 0.05 should deliver about "
            f"half a significant result, not {result.n_significant}"
        )
        naive = [r for r in result.features if abs(r.rank_ic) >= 0.02]
        assert naive, "a 0.02 floor that keeps nothing here would not be the trap"
        assert result.n_kept_at_floor == 0
        assert result.honest_floor > 0.02

        floor_sentence = next(w for w in result.warnings if "min_abs_rank_ic=" in w)
        assert f"A floor of 0.02 keeps {len(naive)} of 10 features" in floor_sentence
        assert "the measured floor keeps 0" in floor_sentence
        assert "select_features(min_abs_rank_ic=" in floor_sentence

    def test_the_family_wise_sentence_counts_the_questions_asked(self, tmp_path):
        dataset_id = _register(_only_noise(), tmp_path, "family_wise")
        result = screen_feature_significance(
            ScreenFeatureSignificanceInput(dataset_id=dataset_id, n_permutations=20)
        )
        sentence = next(w for w in result.warnings if "alpha 0.05" in w)
        assert "10 features were tested" in sentence
        assert "0.5 significant results are expected" in sentence
        assert "compare_signals(mode='adjust')" in sentence

    def test_a_constant_column_is_reported_with_no_ic_and_never_significant(
        self, tmp_path
    ):
        """A column with no cross-sectional variation has no rank
        correlation, so there is nothing to test and nothing to compare a
        null against. The screen reports that as an absence rather than
        failing the other features' answers with it."""
        frame = _only_noise(n_noise=2)
        frame["flat"] = 1.0
        dataset_id = _register(frame, tmp_path, "constant")
        result = screen_feature_significance(
            ScreenFeatureSignificanceInput(dataset_id=dataset_id, n_permutations=20)
        )
        flat = next(r for r in result.features if r.feature == "flat")
        assert flat.rank_ic is None
        assert flat.p_value is None
        assert flat.null_p95_abs is None
        assert flat.significant_at_05 is False
        assert flat.n_usable_permutations == 0
        assert any("'flat'" in w and "not testable" in w for w in result.warnings)
        # And the other two were still answered.
        assert result.n_features == 3
        assert sum(1 for r in result.features if r.rank_ic is not None) == 2

    def test_the_draw_budget_is_refused_before_the_first_shuffle(self, tmp_path):
        """Fifty features at five thousand permutations is a quarter of a
        million passes over the panel. The product is named, and so is the
        ceiling past which no max_draws will buy it."""
        frame = _only_noise(n_noise=50, n_dates=20, n_entities=5)
        dataset_id = _register(frame, tmp_path, "fifty")
        with pytest.raises(ValidationError) as excinfo:
            screen_feature_significance(
                ScreenFeatureSignificanceInput(
                    dataset_id=dataset_id, n_permutations=5000
                )
            )
        message = str(excinfo.value)
        assert "250,000" in message
        assert "50 features x 5000 permutations" in message
        assert "200,000" in message
        assert "narrow `features`" in message

    def test_a_budget_that_fits_is_not_refused(self, tmp_path):
        """The null case for the budget: the same panel, a product under
        the default ceiling, and the screen runs."""
        frame = _only_noise(n_noise=50, n_dates=20, n_entities=5)
        dataset_id = _register(frame, tmp_path, "fifty_cheap")
        result = screen_feature_significance(
            ScreenFeatureSignificanceInput(
                dataset_id=dataset_id, n_permutations=20, max_draws=1_000
            )
        )
        assert result.n_draws == 1_000
        assert result.n_features == 50

    def test_the_within_date_null_carries_its_caveat_on_autocorrelated_ics(
        self, tmp_path
    ):
        """PLANTED: the relationship's strength follows an AR(1), so the
        per-date ICs are serially correlated. Shuffling inside a date
        destroys that correlation and the p-values come out too small --
        the number the module measured, quoted where the caller will read
        it."""
        dataset_id = _register(_autocorrelated_ic(), tmp_path, "autocorrelated")
        result = screen_feature_significance(
            ScreenFeatureSignificanceInput(
                dataset_id=dataset_id, n_permutations=200, null="within_date"
            )
        )
        row = result.features[0]
        assert row.ic_autocorrelation_lag1 > 0.3, (
            "the planted panel is supposed to have autocorrelated per-date "
            f"ICs; it measured {row.ic_autocorrelation_lag1}"
        )
        caveat = [w for w in result.warnings if "27-35%" in w]
        assert len(caveat) == 1
        assert "null='within_date'" in caveat[0]
        assert "null='circular_shift'" in caveat[0]

    def test_circular_shift_on_the_same_panel_carries_no_caveat(self, tmp_path):
        """The null case. The default null keeps the serial correlation, so
        there is nothing to warn about -- and a warning that fired under
        both nulls would be noise rather than a reading."""
        dataset_id = _register(_autocorrelated_ic(), tmp_path, "autocorrelated")
        result = screen_feature_significance(
            ScreenFeatureSignificanceInput(
                dataset_id=dataset_id, n_permutations=200, null="circular_shift"
            )
        )
        assert result.features[0].ic_autocorrelation_lag1 > 0.3
        assert not [w for w in result.warnings if "27-35%" in w]

    def test_an_unknown_feature_is_refused_by_name(self, tmp_path):
        dataset_id = _register(_only_noise(n_noise=3), tmp_path, "unknown")
        with pytest.raises(ValidationError, match="has no feature 'noise_99'"):
            screen_feature_significance(
                ScreenFeatureSignificanceInput(
                    dataset_id=dataset_id, features=["noise_99"], n_permutations=20
                )
            )

    def test_an_empty_feature_list_is_refused_rather_than_read_as_all(self, tmp_path):
        """A caller that filtered its candidates down to nothing asked for
        nothing. Screening all forty instead would answer a question nobody
        asked, expensively."""
        dataset_id = _register(_only_noise(n_noise=3), tmp_path, "empty_list")
        with pytest.raises(ValidationError, match="names nothing to screen"):
            screen_feature_significance(
                ScreenFeatureSignificanceInput(
                    dataset_id=dataset_id, features=[], n_permutations=20
                )
            )

    def test_the_input_rejects_what_it_does_not_declare(self, tmp_path):
        with pytest.raises(Exception):
            ScreenFeatureSignificanceInput(dataset_id="ds_x", n_permutation=200)

    def test_the_draw_ceiling_is_in_the_schema(self):
        """The bound an LLM reads before calling, not one it discovers by
        being refused."""
        schema = ScreenFeatureSignificanceInput.model_json_schema()
        assert schema["properties"]["max_draws"]["maximum"] == 200_000
        assert schema["properties"]["n_permutations"]["maximum"] == 5000


# ── the drift screen, with the curve folded in ───────────────────────────


class TestTheDriftScreenNamesTheFeatureThatMoved:
    def test_the_shifted_feature_is_significant_and_the_rest_are_stable(self, tmp_path):
        """PLANTED: one column jumps +3 sigma halfway through and two do
        not move at all. On a live panel exactly one feature of ten had
        drifted, which is the shape -- a screen that flagged everything or
        nothing would be useless at it."""
        frame = _drifting_panel().drop(columns=["creeping"])
        dataset_id = _register(frame, tmp_path, "shifted")
        result = screen_feature_stability(
            ScreenFeatureStabilityInput(dataset_id=dataset_id)
        )
        assert result.n_features == 3
        assert result.most_drifted == "shifted"
        assert result.n_significant == 1

        by_name = {r.feature: r for r in result.features}
        assert by_name["shifted"].psi_verdict == "significant"
        assert by_name["shifted"].psi >= PSI_SIGNIFICANT
        assert by_name["shifted"].ks_statistic > 0.5
        for name in ("steady", "plain"):
            assert by_name[name].psi_verdict == "stable"
        # Ordered most drifted first, so the answer is the first row.
        assert result.features[0].feature == "shifted"

        named = [w for w in result.warnings if "no longer the same measurement" in w]
        assert len(named) == 1
        assert "'shifted'" in named[0]
        assert "describes neither side" in named[0]

    def test_a_stationary_panel_is_reported_as_one(self, tmp_path):
        """The null case for the drift detector. The conventions caveat
        still comes back -- it qualifies the thresholds, which are always
        reported -- and nothing claims a feature moved."""
        frame = _drifting_panel().drop(columns=["shifted", "creeping"])
        dataset_id = _register(frame, tmp_path, "stationary")
        result = screen_feature_stability(
            ScreenFeatureStabilityInput(dataset_id=dataset_id)
        )
        assert result.n_significant == 0
        assert result.n_moderate == 0
        assert result.n_stable == result.n_features == 2
        assert not [w for w in result.warnings if "no longer the same" in w]
        assert any("conventions rather than tests" in w for w in result.warnings)
        assert result.psi_thresholds == {"moderate": 0.10, "significant": 0.25}

    def test_a_dead_edge_behind_a_stable_distribution_stays_visible(self, tmp_path):
        """PLANTED: the feature's distribution is identical on both sides
        and its IC flips sign. The two failures need different fixes --
        rescaling fixes drift and does nothing here -- so they have to stay
        separable in the result."""
        dataset_id = _register(_edge_dies_panel(), tmp_path, "edge_dies")
        result = screen_feature_stability(
            ScreenFeatureStabilityInput(dataset_id=dataset_id)
        )
        row = next(r for r in result.features if r.feature == "inverting")
        assert row.psi_verdict == "stable"
        assert row.ic_flipped is True
        assert row.ic_before > 0.1 and row.ic_after < -0.1
        # The full-sample number is the one that hides it.
        assert abs(row.ic_overall) < abs(row.ic_before)
        assert result.n_significant == 0

    def test_a_steady_feature_is_not_flagged_as_flipped(self, tmp_path):
        """The null case for `ic_flipped`: an edge that survives."""
        frame = _drifting_panel().drop(columns=["shifted", "creeping"])
        dataset_id = _register(frame, tmp_path, "steady_only")
        result = screen_feature_stability(
            ScreenFeatureStabilityInput(dataset_id=dataset_id)
        )
        steady = next(r for r in result.features if r.feature == "steady")
        assert steady.ic_flipped is False
        assert steady.sign_consistency == 1.0

    def test_a_monotone_drift_rises_against_the_first_block_and_not_the_previous(
        self, tmp_path
    ):
        """This contrast IS the curve's point. The same feature, the same
        blocks: measured against the first block a steady slide
        accumulates, and measured against its predecessor it reads flat.
        One split reports a number and cannot tell a break from a decay."""
        frame = _drifting_panel()[["date", "entity", "target", "creeping"]]
        dataset_id = _register(frame, tmp_path, "creeping")

        against_first = screen_feature_stability(
            ScreenFeatureStabilityInput(
                dataset_id=dataset_id, n_blocks=4, reference="first"
            )
        ).features[0]
        first_curve = [b.psi for b in against_first.psi_by_block]
        assert first_curve[0] is None
        assert all(
            later > earlier for earlier, later in zip(first_curve[1:], first_curve[2:])
        ), f"drift against a fixed reference must accumulate; got {first_curve}"

        against_previous = screen_feature_stability(
            ScreenFeatureStabilityInput(
                dataset_id=dataset_id, n_blocks=4, reference="previous"
            )
        ).features[0]
        rolling = [b.psi for b in against_previous.psi_by_block][1:]
        assert rolling[-1] < first_curve[-1] / 2, (
            "the same slide measured block against block must NOT accumulate; "
            f"got {rolling} against {first_curve}"
        )
        assert (
            max(rolling) / min(rolling) < 1.5
        ), f"a constant slide should read flat against its predecessor; got {rolling}"

    def test_the_first_block_reports_no_psi_at_all(self, tmp_path):
        """It is the reference under 'first' and has no predecessor under
        'previous'. Reporting 0.0 would put a measurement where there is
        none."""
        dataset_id = _register(_drifting_panel(), tmp_path, "blocks")
        for reference in ("first", "previous"):
            result = screen_feature_stability(
                ScreenFeatureStabilityInput(
                    dataset_id=dataset_id, n_blocks=5, reference=reference
                )
            )
            for row in result.features:
                assert len(row.psi_by_block) == 5
                assert [b.block for b in row.psi_by_block] == [0, 1, 2, 3, 4]
                assert row.psi_by_block[0].psi is None
                assert row.psi_by_block[0].psi_verdict is None
                assert all(b.psi is not None for b in row.psi_by_block[1:])
                # The curve's blocks are contiguous and in order.
                ends = [b.end for b in row.psi_by_block]
                starts = [b.start for b in row.psi_by_block]
                assert starts == sorted(starts) and ends == sorted(ends)

    def test_a_decaying_edge_is_named_despite_a_high_sign_consistency(self, tmp_path):
        """PLANTED: the IC is 0.6 in the first two blocks and 0.02 in the
        last two. Sign consistency stays high through it, which is the
        number an agent reads first."""
        dataset_id = _register(_fading_panel(), tmp_path, "fading")
        result = screen_feature_stability(
            ScreenFeatureStabilityInput(dataset_id=dataset_id)
        )
        row = result.features[0]
        assert row.feature == "fading"
        assert row.psi_verdict == "stable"
        assert row.sign_consistency >= 0.75
        decay = [w for w in result.warnings if "decay" in w]
        assert len(decay) == 1
        assert "'fading'" in decay[0]
        assert "sign consistency" in decay[0]

    def test_a_steady_feature_raises_no_decay_warning(self, tmp_path):
        """The null case for the decay detector: an IC that does not fall."""
        frame = _drifting_panel()[["date", "entity", "target", "steady"]]
        dataset_id = _register(frame, tmp_path, "no_decay")
        result = screen_feature_stability(
            ScreenFeatureStabilityInput(dataset_id=dataset_id)
        )
        assert not [w for w in result.warnings if "decay" in w]

    def test_fewer_dates_than_blocks_is_refused(self, tmp_path):
        frame = _only_noise(n_noise=2, n_dates=3, n_entities=6)
        dataset_id = _register(frame, tmp_path, "three_dates")
        with pytest.raises(ValidationError, match="cannot be split into 4 blocks"):
            screen_feature_stability(
                ScreenFeatureStabilityInput(dataset_id=dataset_id, n_blocks=4)
            )

    def test_a_split_outside_the_panel_is_refused_with_the_remedy(self, tmp_path):
        dataset_id = _register(_drifting_panel(), tmp_path, "split_outside")
        with pytest.raises(ValidationError, match="inside the panel's range"):
            screen_feature_stability(
                ScreenFeatureStabilityInput(
                    dataset_id=dataset_id, split_date="2031-01-01"
                )
            )

    def test_more_features_than_the_ceiling_is_refused_naming_both(self, tmp_path):
        frame = _only_noise(n_noise=12, n_dates=40, n_entities=6)
        dataset_id = _register(frame, tmp_path, "too_many")
        with pytest.raises(ValidationError) as excinfo:
            screen_feature_stability(
                ScreenFeatureStabilityInput(dataset_id=dataset_id, max_features=5)
            )
        message = str(excinfo.value)
        assert "12 features" in message
        assert "max_features=5" in message
        assert "max_features=12" in message

    def test_an_unknown_feature_and_an_empty_list_are_both_refused(self, tmp_path):
        dataset_id = _register(_drifting_panel(), tmp_path, "refusals")
        with pytest.raises(ValidationError, match="has no feature 'stedy'"):
            screen_feature_stability(
                ScreenFeatureStabilityInput(dataset_id=dataset_id, features=["stedy"])
            )
        with pytest.raises(ValidationError, match="names nothing to screen"):
            screen_feature_stability(
                ScreenFeatureStabilityInput(dataset_id=dataset_id, features=[])
            )

    def test_a_split_date_is_echoed_per_feature(self, tmp_path):
        """The before/after numbers mean nothing without the boundary they
        were measured across, and the default is per-feature."""
        dataset_id = _register(_drifting_panel(), tmp_path, "split_echo")
        result = screen_feature_stability(
            ScreenFeatureStabilityInput(dataset_id=dataset_id, split_date="2022-05-02")
        )
        assert {r.split_date for r in result.features} == {"2022-05-02"}


# ── both tools through the runtime boundary ──────────────────────────────


class TestBothScreensCrossTheProtocolBoundary:
    """Every result leaves as JSON with `allow_nan=False`. A bare `NaN`
    token is not valid JSON and a strict client rejects the whole message,
    so a non-finite statistic has to arrive as `null` -- which these
    screens produce routinely: block 0's PSI, a constant column's IC, a
    sign consistency on a feature with one usable block."""

    def test_the_significance_screen_survives_dispatch_and_strict_json(self, tmp_path):
        frame = _only_noise(n_noise=3)
        frame["flat"] = 2.5
        dataset_id = _register(frame, tmp_path, "dispatch_significance")
        payload = feature_dispatch(
            "screen_feature_significance",
            {"dataset_id": dataset_id, "n_permutations": 20},
        )
        encoded = json.dumps(payload, allow_nan=False, default=str)
        assert '"honest_floor"' in encoded
        flat = next(r for r in payload["features"] if r["feature"] == "flat")
        assert flat["rank_ic"] is None
        assert payload["n_draws"] == 4 * 20

    def test_the_stability_screen_survives_dispatch_and_strict_json(self, tmp_path):
        dataset_id = _register(_drifting_panel(), tmp_path, "dispatch_stability")
        payload = feature_dispatch(
            "screen_feature_stability",
            {"dataset_id": dataset_id, "n_blocks": 3, "reference": "previous"},
        )
        encoded = json.dumps(payload, allow_nan=False, default=str)
        assert '"psi_by_block"' in encoded
        assert payload["features"][0]["psi_by_block"][0]["psi"] is None
        assert payload["psi_thresholds"] == {"moderate": 0.10, "significant": 0.25}

    def test_both_are_advertised_by_the_runtime_that_owns_them(self):
        from standard_quant_tools.agent.runtimes import owner_of
        from standard_quant_tools.modeling.agent.feature_tools import (
            FEATURE_TOOL_DISPATCH,
            get_feature_tools,
        )

        advertised = {d["function"]["name"] for d in get_feature_tools()}
        for name in ("screen_feature_significance", "screen_feature_stability"):
            assert name in FEATURE_TOOL_DISPATCH
            assert name in advertised
            assert owner_of(name) == "feature_lab"

    def test_the_descriptions_say_what_gets_refused(self):
        """A description is what the model reads before it calls. Both of
        these refuse, and a refusal nobody was warned about reads as a
        broken tool."""
        from standard_quant_tools.modeling.agent.feature_tools import FEATURE_TOOL_DEFS

        by_name = {name: text for name, text, _ in FEATURE_TOOL_DEFS}
        assert "REFUSED past max_draws" in by_name["screen_feature_significance"]
        assert "honest_floor" in by_name["screen_feature_significance"]
        assert "Refuses" in by_name["screen_feature_stability"]
        assert "psi_by_block" in by_name["screen_feature_stability"]
