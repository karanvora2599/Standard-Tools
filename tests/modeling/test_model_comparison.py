"""
Paired model comparison: the interval on the per-date IC difference, the
Diebold-Mariano loss test, and the Holm adjustment.

Every oracle is planted. Two IC series that differ by a known constant
plus AR(1) noise must recover the constant inside the interval; two with
identical skill and independent noise must produce an interval that
covers zero (the null); the block interval must be wider than the IID one
on autocorrelated differences and about the same on white noise; a loss
differential with a known mean must produce a DM statistic of the right
sign; and Holm on a hand-worked triple must give the hand-worked answer.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.validation.comparison import (
    compare_ic_series,
    diebold_mariano,
    holm_adjust,
    newey_west_variance,
    paired_comparison,
)

DATES = pd.bdate_range("2021-01-04", periods=400)


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + rng.normal(0.0, sigma * np.sqrt(1 - phi**2))
    return out


class TestTheDifferenceInterval:
    def test_recovers_a_planted_improvement(self):
        rng = np.random.default_rng(0)
        base = pd.Series(0.03 + _ar1(400, 0.5, 0.05, rng), index=DATES)
        better = base + 0.02 + _ar1(400, 0.5, 0.01, rng)
        result = compare_ic_series(base, better, seed=1)
        assert result["ci_lower"] < 0.02 < result["ci_upper"]
        assert result["mean_difference"] == pytest.approx(
            float((better - base).mean()), abs=1e-12
        )
        assert result["verdict"] == "b_better"
        assert result["p_value"] < 0.05
        assert result["hit_rate"] > 0.9

    def test_the_null_covers_zero_at_about_the_nominal_rate(self):
        """Same discipline as the DM null: twenty independent nulls, and
        the interval may miss zero about one time in twenty."""
        misses = 0
        for seed in range(20):
            rng = np.random.default_rng(200 + seed)
            base = pd.Series(0.03 + _ar1(400, 0.5, 0.05, rng), index=DATES)
            same_skill = base + _ar1(400, 0.5, 0.01, rng)
            result = compare_ic_series(base, same_skill, seed=1, n_bootstrap=400)
            if result["verdict"] != "indistinguishable":
                misses += 1
        assert misses <= 3, f"{misses} of 20 nulls called a difference"

    def test_the_block_interval_is_wider_on_autocorrelated_differences(self):
        rng = np.random.default_rng(3)
        base = pd.Series(np.zeros(400), index=DATES)
        persistent = base + _ar1(400, 0.8, 0.02, rng)
        blocked = compare_ic_series(base, persistent, seed=1)
        iid = compare_ic_series(base, persistent, seed=1, block_size=1)
        width = lambda r: r["ci_upper"] - r["ci_lower"]  # noqa: E731
        assert blocked["block_size"] > 1
        assert width(blocked) > 1.3 * width(iid)

    def test_and_about_the_same_on_white_noise(self):
        rng = np.random.default_rng(4)
        base = pd.Series(np.zeros(400), index=DATES)
        white = base + rng.normal(0.0, 0.02, 400)
        blocked = compare_ic_series(base, white, seed=1)
        iid = compare_ic_series(base, white, seed=1, block_size=1)
        width = lambda r: r["ci_upper"] - r["ci_lower"]  # noqa: E731
        assert width(blocked) == pytest.approx(width(iid), rel=0.25)

    def test_aligns_on_common_dates_and_needs_ten(self):
        a = pd.Series(np.arange(20, dtype=float) / 100, index=DATES[:20])
        b = pd.Series(np.arange(20, dtype=float) / 100 + 0.01, index=DATES[5:25])
        result = compare_ic_series(a, b, n_bootstrap=100)
        assert result["n_dates"] == 15
        with pytest.raises(ValidationError, match="fewer than ten"):
            compare_ic_series(a.iloc[:5], b.iloc[:5])

    def test_the_seed_makes_it_reproducible(self):
        rng = np.random.default_rng(5)
        a = pd.Series(rng.normal(size=100), index=DATES[:100])
        b = a + rng.normal(size=100) * 0.1
        assert compare_ic_series(a, b, seed=7) == compare_ic_series(a, b, seed=7)


class TestDieboldMariano:
    def test_a_planted_loss_gap_has_the_right_sign_and_is_significant(self):
        rng = np.random.default_rng(6)
        loss_b = pd.Series(1.0 + rng.normal(0, 0.1, 400), index=DATES)
        loss_a = loss_b + 0.1  # A's loss is larger, so B is better
        result = diebold_mariano(loss_a, loss_b, lag=0)
        assert result["statistic"] > 0
        assert result["p_value"] < 1e-6
        assert result["mean_differential"] == pytest.approx(0.1, abs=1e-12)

    def test_equal_loss_rejects_at_about_the_nominal_rate(self):
        """
        The null, as a RATE rather than one draw: a single seed rejects at
        the 5% level one time in twenty by design, so the honest check is
        that forty independent nulls reject about that often -- not that a
        particular one happens not to.
        """
        rejections = 0
        for seed in range(40):
            rng = np.random.default_rng(100 + seed)
            loss_b = pd.Series(1.0 + rng.normal(0, 0.1, 400), index=DATES)
            loss_a = loss_b + rng.normal(0, 0.1, 400)
            if diebold_mariano(loss_a, loss_b, lag=0)["p_value"] < 0.05:
                rejections += 1
        assert rejections <= 6, f"{rejections} of 40 nulls rejected at 5%"

    def test_identical_models_have_no_variance_and_no_verdict(self):
        loss = pd.Series(np.ones(50), index=DATES[:50])
        result = diebold_mariano(loss, loss, lag=2)
        assert np.isnan(result["statistic"]) and np.isnan(result["p_value"])

    def test_newey_west_grows_with_positive_autocorrelation(self):
        rng = np.random.default_rng(8)
        persistent = _ar1(2000, 0.7, 1.0, rng)
        assert newey_west_variance(persistent, 10) > 2.0 * newey_west_variance(persistent, 0)
        white = rng.normal(size=2000)
        assert newey_west_variance(white, 10) == pytest.approx(newey_west_variance(white, 0), rel=0.2)

    def test_lag_zero_is_the_variance_of_the_mean(self):
        x = np.array([1.0, 2.0, 4.0, 7.0])
        assert newey_west_variance(x, 0) == pytest.approx(x.var(ddof=0) / 4)


class TestHolm:
    def test_the_hand_worked_triple(self):
        # Sorted: 0.01, 0.03, 0.04 -> 0.03, max(0.03, 0.06)=0.06, max(0.06, 0.04)=0.06
        assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_monotone_and_capped(self):
        adjusted = holm_adjust([0.5, 0.9, 0.2])
        assert max(adjusted) <= 1.0
        assert adjusted[1] >= adjusted[0] >= adjusted[2]
        assert holm_adjust([]) == []


def _panel(n_entities: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for date in DATES[:120]:
        signal = rng.normal(size=n_entities)
        for i in range(n_entities):
            rows.append({"date": date, "entity": f"E{i}", "target": signal[i] + rng.normal(0, 0.5)})
    return pd.DataFrame(rows), rng


class TestPairedComparison:
    def test_a_perfect_model_beats_a_shuffled_one(self):
        panel, rng = _panel()
        perfect = panel.assign(prediction=panel["target"])
        shuffled = panel.assign(prediction=rng.permutation(panel["target"].to_numpy()))
        result = paired_comparison(shuffled, perfect, task="regression", horizon=5, seed=1)
        assert result["mean_b"] == pytest.approx(1.0)
        assert abs(result["mean_a"]) < 0.15
        assert result["verdict"] == "b_better"
        assert result["diebold_mariano"]["statistic"] > 0
        assert result["diebold_mariano"]["lag"] == 4
        assert result["diebold_mariano"]["loss"] == "squared_error"
        assert result["n_entities"] == 12

    def test_the_intersection_is_what_is_compared(self):
        panel, _rng = _panel()
        a = panel.assign(prediction=panel["target"])
        b = panel.assign(prediction=panel["target"]).iloc[: len(panel) // 2]
        result = paired_comparison(a, b, task="ranking", n_bootstrap=100)
        assert result["n_rows"] == len(b)
        assert result["diebold_mariano"] is None  # a ranker has no loss with units
        assert result["verdict"] == "indistinguishable"

    def test_disagreeing_outcomes_are_refused(self):
        panel, _rng = _panel()
        a = panel.assign(prediction=panel["target"])
        b = panel.assign(prediction=panel["target"], target=panel["target"] + 1.0)
        with pytest.raises(ValidationError, match="realized outcomes disagree"):
            paired_comparison(a, b, task="regression")

    def test_no_shared_rows_and_missing_columns_are_refused(self):
        panel, _rng = _panel()
        a = panel.assign(prediction=panel["target"])
        b = a.assign(entity="Z" + a["entity"])
        with pytest.raises(ValidationError, match="share no"):
            paired_comparison(a, b, task="regression")
        with pytest.raises(ValidationError, match="missing column"):
            paired_comparison(a.drop(columns=["target"]), a, task="regression")

    def test_an_unknown_metric_is_refused(self):
        panel, _rng = _panel()
        a = panel.assign(prediction=panel["target"])
        with pytest.raises(ValidationError, match="metric="):
            paired_comparison(a, a, task="regression", metric="r2")

    def test_a_classifier_gets_a_brier_test(self):
        panel, rng = _panel()
        binary = panel.assign(target=(panel["target"] > 0).astype(float))
        good = binary.assign(prediction=np.clip(0.5 + 0.4 * np.sign(panel["target"]), 0, 1))
        coin = binary.assign(prediction=rng.uniform(size=len(panel)))
        result = paired_comparison(coin, good, task="classification", n_bootstrap=200)
        assert result["diebold_mariano"]["loss"] == "brier"
        assert result["diebold_mariano"]["statistic"] > 0
        assert result["verdict"] == "b_better"
