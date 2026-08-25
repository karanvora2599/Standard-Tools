"""
Drift, stability and significance.

These three answer questions a full-sample feature report structurally
cannot, and each of them has talked somebody out of a strategy:

- a mean IC computed across a regime break describes neither side of it;
- a mean IC of 0.04 that is 0.30 in one year and -0.05 in the rest is not a
  weaker version of a steady 0.04, it is a different claim;
- an IC of 0.03 on a small panel is a number noise produces routinely.

WHAT THESE TESTS PIN. The statistics are standard, so the tests are built
around data whose ANSWER IS KNOWN BY CONSTRUCTION -- a feature whose edge is
switched off halfway, a feature that is pure noise, two samples drawn from
the same distribution. A test that only asserts "returns a float" would pass
for an implementation that returned the wrong float.

The permutation test gets the most attention because its null is the part
that is easy to get subtly wrong, and the failures are silent: a null that
is too weak calls noise significant, and a one-sided test called two-sided
reports p near 1 for a strongly negative IC. Both are checked below against
data whose answer is known.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.analysis.feature_stability import (
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    feature_drift,
    feature_stability,
    ks_statistic,
    permutation_test_ic,
    population_stability_index,
)


def _panel(seed=1, n_dates=200, n_entities=25, break_at=None):
    """A panel with a known answer.

    `break_at` switches the feature's predictive power OFF from that date
    index onward, without changing its distribution -- which is exactly the
    case where PSI stays quiet and the IC collapses.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, date in enumerate(pd.bdate_range("2022-01-03", periods=n_dates)):
        live = 1.0 if (break_at is None or i < break_at) else 0.0
        for entity in (f"E{j}" for j in range(n_entities)):
            signal = rng.normal()
            rows.append(
                {
                    "date": date,
                    "entity": entity,
                    "real": signal,
                    "noise": rng.normal(),
                    "target": live * 0.5 * signal + rng.normal(),
                }
            )
    return pd.DataFrame(rows)


class TestPSI:
    def test_the_same_distribution_scores_near_zero(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=5000), rng.normal(size=5000)
        assert population_stability_index(a, b) < PSI_MODERATE

    def test_a_shifted_distribution_scores_significant(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 5000)
        b = rng.normal(2, 1, 5000)
        assert population_stability_index(a, b) > PSI_SIGNIFICANT

    def test_it_is_not_symmetric_about_which_window_is_reference(self):
        """Bin edges come from the REFERENCE window on purpose. Pooling
        would let the current window move the edges it is measured against,
        muting the drift the statistic exists to find. Asymmetry is the
        evidence that the edges are not pooled."""
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 4000)
        b = rng.normal(0, 3, 4000)
        assert population_stability_index(a, b) != pytest.approx(
            population_stability_index(b, a), rel=0.05
        )

    def test_an_empty_bucket_does_not_produce_infinity(self):
        """Without the floor, one bucket empty in a window sends PSI to
        infinity -- which reads as 'infinitely drifted' when it means 'no
        observations here'."""
        a = np.concatenate([np.zeros(500), np.ones(500)])
        b = np.zeros(500)
        value = population_stability_index(a, b)
        assert np.isfinite(value)

    def test_an_empty_sample_is_nan_not_zero(self):
        assert np.isnan(population_stability_index(np.array([]), np.ones(10)))


class TestKS:
    def test_identical_samples_score_zero(self):
        a = np.arange(100, dtype=float)
        assert ks_statistic(a, a) == pytest.approx(0.0)

    def test_disjoint_samples_score_one(self):
        assert ks_statistic(np.zeros(50), np.ones(50)) == pytest.approx(1.0)

    def test_it_is_symmetric(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=300), rng.normal(1, size=300)
        assert ks_statistic(a, b) == pytest.approx(ks_statistic(b, a))


class TestDrift:
    def test_a_dead_edge_shows_as_ic_collapse_not_distribution_drift(self):
        """The case the tool exists for, and the reason both halves are
        reported. The feature is drawn from the same distribution
        throughout; only its relationship to the target stops."""
        result = feature_drift(_panel(break_at=100), "real")
        assert result["psi"] < PSI_MODERATE, "distribution did not drift"
        assert abs(result["ic_before"]) > 0.2
        assert abs(result["ic_after"]) < 0.1

    def test_a_stable_feature_is_stable_on_both_measures(self):
        result = feature_drift(_panel(), "real")
        assert result["psi"] < PSI_MODERATE
        assert result["psi_verdict"] == "stable"
        assert not result["ic_flipped"]

    def test_the_split_defaults_to_the_median_date(self):
        result = feature_drift(_panel(n_dates=100), "real")
        assert result["n_before"] > 0 and result["n_after"] > 0
        # Split by TIME, so the halves are near-equal on a balanced panel.
        assert abs(result["n_before"] - result["n_after"]) < result["n_before"] * 0.1

    def test_a_split_outside_the_range_is_refused(self):
        with pytest.raises(ValidationError, match="leaves one side empty"):
            feature_drift(_panel(), "real", split_date="2030-01-01")

    def test_an_unknown_feature_is_refused(self):
        with pytest.raises(ValidationError, match="no feature"):
            feature_drift(_panel(), "nope")


class TestStability:
    def test_it_finds_the_block_where_the_edge_died(self):
        result = feature_stability(_panel(break_at=100), "real", n_blocks=4)
        ics = [b["ic_mean"] for b in result["blocks"]]
        assert ics[0] > 0.2 and ics[1] > 0.2
        assert abs(ics[2]) < 0.1 and abs(ics[3]) < 0.1

    def test_blocks_are_contiguous_and_ordered(self):
        """Never shuffled -- interleaving would average away the regime
        structure this exists to expose."""
        result = feature_stability(_panel(), "real", n_blocks=5)
        starts = [b["start"] for b in result["blocks"]]
        assert starts == sorted(starts)
        for earlier, later in zip(result["blocks"], result["blocks"][1:]):
            assert earlier["end"] < later["start"]

    def test_sign_consistency_alone_misses_decay(self):
        """Documented in the tool description, so pinned here. A feature
        that decays without flipping keeps perfect sign consistency -- which
        is why the block ICs have to be read too."""
        result = feature_stability(_panel(break_at=100), "real", n_blocks=4)
        assert result["sign_consistency"] == pytest.approx(1.0)
        assert result["ic_block_max"] - result["ic_block_min"] > 0.2

    def test_too_few_dates_for_the_blocks_is_refused(self):
        with pytest.raises(ValidationError, match="cannot be split"):
            feature_stability(_panel(n_dates=3), "real", n_blocks=10)

    def test_one_block_is_refused(self):
        with pytest.raises(ValidationError, match="at least 2"):
            feature_stability(_panel(), "real", n_blocks=1)


class TestPermutationTest:
    def test_a_real_signal_is_significant(self):
        result = permutation_test_ic(
            _panel(), "real", n_permutations=100, random_seed=3
        )
        assert result["significant_at_05"]
        assert result["p_value"] < 0.05

    def test_the_null_is_calibrated_on_pure_noise(self):
        """
        A correctly calibrated test rejects a true null about 5% of the
        time. This checks that the implementation is somewhere near that,
        which catches a null that has been broken outright -- not shuffling,
        or shuffling the wrong array.

        It does NOT distinguish within-date from global shuffling. Measured,
        those two produce nulls within 2% of each other, because the IC is
        computed within each date and averaged, so both deliver a random
        assignment inside each date. An earlier version of this docstring
        claimed otherwise; mutating the code to shuffle globally passed every
        test here, which is how the claim was found to be wrong.

        Asserted across ten independent noise features rather than one,
        because one is a coin flip: a correctly calibrated test rejects a
        true null 5% of the time by definition, and a single seed that
        happens to land there says nothing. (The first draft of this test
        asserted a single seed, picked one of the 5%, and failed -- which is
        the behaviour it was written to confirm.)

        The threshold is loose on purpose. Correct is ~0-1 of ten; broken is
        ~10 of ten. Three separates those without being sensitive to which
        seeds were chosen.
        """
        significant = sum(
            permutation_test_ic(
                _panel(seed=seed), "noise", n_permutations=100, random_seed=seed
            )["significant_at_05"]
            for seed in range(10)
        )
        assert significant <= 3, (
            f"{significant}/10 pure-noise features were called significant. "
            "A correctly calibrated test gives about 0-1; this many means the "
            "null is too weak, which is what shuffling across dates rather "
            "than within them produces."
        )

    def test_a_strong_negative_ic_is_significant(self):
        """
        The p-value is two-sided, and this is the test that says so.

        Mutating the implementation from `|null| >= |observed|` to
        `null >= observed` passed every other test in this file. Under that
        one-sided version a feature with an IC of -0.25 -- a strong feature
        with a sign -- reports p near 1 and is discarded as noise.
        """
        panel = _panel()
        panel["inverted"] = -panel["real"]
        result = permutation_test_ic(
            panel, "inverted", n_permutations=100, random_seed=2
        )
        assert result["observed_ic"] < -0.1, "expected a strong negative IC"
        assert result["significant_at_05"], (
            f"a strongly negative IC of {result['observed_ic']:.3f} came back "
            f"at p={result['p_value']:.3f}; the test is one-sided"
        )

    def test_the_null_is_centred_near_zero(self):
        """Shuffling within date destroys the link and nothing else, so the
        null IC should sit at zero. A biased null means the shuffle is
        disturbing something it should not."""
        result = permutation_test_ic(
            _panel(), "real", n_permutations=150, random_seed=5
        )
        assert abs(result["null_mean"]) < 3 * result["null_std"]

    def test_a_p_value_of_exactly_zero_is_never_reported(self):
        """200 shuffles cannot distinguish 'p < 0.005' from 'p = 0', and
        printing 0.0 claims a precision the sample size does not have."""
        result = permutation_test_ic(_panel(), "real", n_permutations=50, random_seed=1)
        assert result["p_value"] > 0.0
        assert result["p_value"] == pytest.approx(1 / 51, rel=0.01)

    def test_it_is_reproducible_for_a_seed(self):
        kwargs = dict(n_permutations=60, random_seed=7)
        first = permutation_test_ic(_panel(), "real", **kwargs)
        second = permutation_test_ic(_panel(), "real", **kwargs)
        assert first["p_value"] == second["p_value"]
        assert first["null_std"] == second["null_std"]

    def test_a_different_seed_gives_a_different_null(self):
        a = permutation_test_ic(_panel(), "noise", n_permutations=60, random_seed=1)
        b = permutation_test_ic(_panel(), "noise", n_permutations=60, random_seed=2)
        assert a["null_std"] != b["null_std"]

    def test_the_p95_floor_is_above_the_null_mean(self):
        result = permutation_test_ic(
            _panel(), "noise", n_permutations=120, random_seed=4
        )
        assert result["null_p95_abs"] > abs(result["null_mean"])

    def test_a_constant_feature_is_never_significant(self):
        """
        A feature with no variance predicts nothing, and the test must say
        so rather than finding structure in a straight line.

        This test originally expected a refusal, on the assumption that a
        constant feature yields an undefined IC. It does not:
        `cross_sectional_ic` returns 0.0 for zero-variance input, which is a
        defensible choice and is the library's. So the guarantee worth
        pinning is the one that actually protects a caller -- the p-value
        comes back high and the feature is not called significant.
        """
        frame = _panel(n_dates=60)
        frame["constant"] = 1.0
        result = permutation_test_ic(
            frame, "constant", n_permutations=50, random_seed=0
        )
        assert result["observed_ic"] == 0.0
        assert not result["significant_at_05"]
        assert result["p_value"] > 0.5, (
            "a constant feature should be at least as unremarkable as its " "own null"
        )
