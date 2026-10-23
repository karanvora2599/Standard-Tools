"""
Whether a series of VaR and ES forecasts was right.

`backtest_var_series` lines each forecast up against the return that
followed it and tests the exceptions. Each test is held to an answer worked
out apart from the code:

- Kupiec's LR is its closed form on a planted count: 8 exceptions in 250 at
  99% is 7.7336, and none at all is -2T ln(1 - p).
- The Basel traffic light at its boundaries: 4 green, 5 yellow (+0.40),
  9 yellow (+0.85), 10 red (+1.00), on the LAST 250 forecasts.
- Christoffersen's independence test rejects exceptions planted in runs and
  not the same number scattered at random, and is null with the reason
  where it is undefined.
- On i.i.d. normal returns with `normal` VaR the exception count stays
  inside its binomial band and Kupiec rejects about as often as its level
  says, over repeated samples.
- Z2 has expected value 0 when the forecasts are the true distribution's,
  its simulated p-value rejects a correct model about 5% of the time, and an
  understated tail drives it negative and rejects.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from standard_quant_tools.metrics.var_backtest import (
    Z2_NULL_DISTRIBUTION,
    acerbi_szekely_z2,
    backtest_var_series,
    basel_traffic_light,
    christoffersen_independence,
    kupiec_pof,
)
from standard_quant_tools.metrics.var_forecast import forecast_var_series


def _planted(exception_positions, n=250, var=0.02, es=None):
    """Returns of 0.001 except a loss of 0.05 on each planted position,
    against a constant VaR: the exceptions are exactly the plant."""
    realized = np.full(n, 0.001)
    realized[list(exception_positions)] = -0.05
    frame = pd.DataFrame(
        {"realized": realized, "var": np.full(n, var)},
        index=pd.bdate_range("2021-01-04", periods=n),
    )
    if es is not None:
        frame["es"] = es
    return frame


class TestKupiec:
    def test_the_planted_count_gives_the_closed_form(self):
        """T = 250, p = 0.01, x = 8: -2 ln[(0.99^242)(0.01^8)] +
        2 ln[(242/250)^242 (8/250)^8] = 7.7336, p = 0.0054."""
        out = backtest_var_series(
            _planted(range(5, 250, 31)[:8]), 0.99, n_simulations=100
        )
        assert out["n_exceptions"] == 8
        T, x, p = 250, 8, 0.01
        closed = -2 * ((T - x) * math.log(1 - p) + x * math.log(p)) + 2 * (
            (T - x) * math.log(1 - x / T) + x * math.log(x / T)
        )
        assert closed == pytest.approx(7.7336, abs=5e-5)
        assert out["kupiec"]["lr"] == pytest.approx(closed, rel=1e-12)
        assert out["kupiec"]["p_value"] == pytest.approx(
            stats.chi2.sf(closed, 1), rel=1e-12
        )

    def test_no_exception_is_defined_too(self):
        """With 0 ln 0 = 0 the LR is -2 T ln(1 - p): a 99% VaR never
        breached in 500 days is itself evidence it is too wide."""
        out = kupiec_pof(0, 500, 0.01)
        assert out["lr"] == pytest.approx(-2 * 500 * math.log(0.99), rel=1e-12)
        assert out["p_value"] < 0.01

    def test_the_expected_count_gives_zero(self):
        assert kupiec_pof(5, 500, 0.01)["lr"] == pytest.approx(0.0, abs=1e-12)


class TestBaselTrafficLight:
    @pytest.mark.parametrize(
        "count,zone,plus",
        [
            (0, "green", 0.0),
            (4, "green", 0.0),
            (5, "yellow", 0.40),
            (6, "yellow", 0.50),
            (7, "yellow", 0.65),
            (8, "yellow", 0.75),
            (9, "yellow", 0.85),
            (10, "red", 1.0),
            (14, "red", 1.0),
        ],
    )
    def test_the_zones_and_plus_factors(self, count, zone, plus):
        out = basel_traffic_light(count)
        assert out["zone"] == zone
        assert out["plus_factor"] == plus
        assert out["multiplier"] == pytest.approx(3.0 + plus)

    @pytest.mark.parametrize(
        "count,zone", [(4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")]
    )
    def test_the_boundaries_on_a_planted_series(self, count, zone):
        positions = range(3, 3 + 20 * count, 20)
        out = backtest_var_series(_planted(positions), 0.99, n_simulations=100)
        assert out["basel"]["exceptions"] == count
        assert out["basel"]["zone"] == zone
        assert out["basel"]["null_reason"] is None

    def test_the_light_reads_the_last_250(self):
        """Ten exceptions in the first 50 of 300 forecasts are outside the
        last 250: green, though the whole series has ten."""
        out = backtest_var_series(
            _planted(range(0, 50, 5), n=300), 0.99, n_simulations=100
        )
        assert out["n_exceptions"] == 10
        assert out["basel"]["exceptions"] == 0
        assert out["basel"]["zone"] == "green"
        assert (
            out["basel"]["first_date"] == pd.bdate_range("2021-01-04", periods=300)[50]
        )

    def test_off_99_or_short_it_is_null_with_the_reason(self):
        at_95 = backtest_var_series(_planted([3]), 0.95, n_simulations=100)
        assert at_95["basel"]["zone"] is None
        assert "defined on 99% VaR" in at_95["basel"]["null_reason"]
        short = backtest_var_series(_planted([3], n=200), 0.99, n_simulations=100)
        assert short["basel"]["zone"] is None
        assert "there are 200" in short["basel"]["null_reason"]


class TestChristoffersen:
    def test_clustered_exceptions_are_rejected(self):
        """Ten exceptions in two runs of five: an exception is followed by
        another eight times in nine, where independence says 4%."""
        hits = np.zeros(250, dtype=bool)
        hits[40:45] = True
        hits[160:165] = True
        out = christoffersen_independence(hits)
        assert (out["n11"], out["n10"]) == (8, 2)
        assert out["p_value"] < 1e-6

    def test_the_same_count_scattered_is_not(self):
        """Ten exceptions none of which follows another (n11 = 0): the
        chance of one after one is 0, against 4% overall -- no evidence of
        clustering."""
        hits = np.zeros(250, dtype=bool)
        hits[np.arange(10, 250, 24)] = True
        out = christoffersen_independence(hits)
        assert out["n11"] == 0
        assert out["p_value"] > 0.3

    def test_iid_exceptions_reject_at_about_the_level(self):
        """Independent 5% exceptions over 1,000 dates, 300 times. These
        seeds reject 24 times (8%): the chi-squared reading of the LR is
        known to run a little over its level when n11 is a handful, and the
        bound allows that while catching a test that rejects at random."""
        rng = np.random.default_rng(8)
        rejects = sum(
            christoffersen_independence(rng.random(1000) < 0.05)["p_value"] < 0.05
            for _ in range(300)
        )
        assert 3 <= rejects <= 36

    def test_the_lr_is_the_markov_likelihood_ratio(self):
        hits = np.zeros(100, dtype=bool)
        hits[[3, 4, 20, 50, 51, 52, 80]] = True
        out = christoffersen_independence(hits)
        n00, n01, n10, n11 = out["n00"], out["n01"], out["n10"], out["n11"]
        assert n00 + n01 + n10 + n11 == 99
        pi01, pi11 = n01 / (n00 + n01), n11 / (n10 + n11)
        pi = (n01 + n11) / 99
        lr = -2 * (
            (n00 + n10) * math.log(1 - pi)
            + (n01 + n11) * math.log(pi)
            - n00 * math.log(1 - pi01)
            - n01 * math.log(pi01)
            - n10 * math.log(1 - pi11)
            - n11 * math.log(pi11)
        )
        assert out["lr"] == pytest.approx(lr, rel=1e-12)

    def test_undefined_cases_are_null_with_the_reason(self):
        none = christoffersen_independence(np.zeros(50, dtype=bool))
        assert none["lr"] is None and "no exception" in none["null_reason"]
        every = christoffersen_independence(np.ones(50, dtype=bool))
        assert every["lr"] is None and "every date" in every["null_reason"]
        last = np.zeros(50, dtype=bool)
        last[-1] = True
        only_last = christoffersen_independence(last)
        assert only_last["lr"] is None and "0/0" in only_last["null_reason"]

    def test_conditional_coverage_adds_the_two(self):
        frame = _planted([10, 11, 100, 200])
        out = backtest_var_series(frame, 0.99, n_simulations=100)
        cc = out["kupiec"]["lr"] + out["christoffersen"]["lr"]
        assert out["conditional_coverage"]["lr"] == pytest.approx(cc, rel=1e-12)
        assert out["conditional_coverage"]["p_value"] == pytest.approx(
            stats.chi2.sf(cc, 2), rel=1e-12
        )


class TestCoverageOnACorrectModel:
    def test_normal_var_on_normal_returns_has_the_nominal_size(self):
        """300 samples of 1,500 i.i.d. normal returns, a 95% normal VaR from
        a 1,000-day window, 500 forecasts each. Every exception count sits in
        its 99.9% binomial band, the mean rate is 5%, and Kupiec rejects at
        5% in about 5% of samples (estimation in the window costs a little).
        """
        dates = pd.bdate_range("2000-01-03", periods=1500)
        lo, hi = stats.binom.ppf([0.0005, 0.9995], 500, 0.05)
        rejects, rates = 0, []
        for rep in range(300):
            rng = np.random.default_rng(1000 + rep)
            returns = pd.Series(rng.normal(0.0002, 0.01, 1500), index=dates)
            frame = forecast_var_series(
                returns, method="normal", confidence=0.95, window=1000
            )["forecasts"]
            out = backtest_var_series(frame, 0.95, n_simulations=100)
            assert lo <= out["n_exceptions"] <= hi
            rates.append(out["exception_rate"])
            rejects += out["kupiec"]["p_value"] < 0.05
        assert np.mean(rates) == pytest.approx(0.05, abs=0.003)
        # Binomial(300, 0.05) has a 99.9% band of about [4, 28].
        assert 4 <= rejects <= 28


def _true_forecasts(rng, T, confidence):
    """Returns from N(mu_t, s_t) with the true VaR and ES beside them."""
    p = 1 - confidence
    z = stats.norm.ppf(confidence)
    scale = 0.01 * np.exp(rng.normal(0, 0.3, T))
    loc = rng.normal(0, 0.001, T)
    realized = loc + scale * rng.standard_normal(T)
    var = -loc + scale * z
    es = -loc + scale * stats.norm.pdf(z) / p
    return realized, var, es


class TestAcerbiSzekely:
    def test_z2_is_the_formula(self):
        """Two exceptions of -0.05 and -0.04 against ES 0.03 and 0.025 in
        200 dates at 97.5%: Z2 = (-0.05/0.03 - 0.04/0.025) / (200 x 0.025) + 1."""
        realized = np.full(200, 0.001)
        realized[[10, 90]] = [-0.05, -0.04]
        es = np.full(200, 0.03)
        es[90] = 0.025
        out = acerbi_szekely_z2(realized, np.full(200, 0.02), es, 0.025)
        expected = (-0.05 / 0.03 - 0.04 / 0.025) / (200 * 0.025) + 1
        assert out["z2"] == pytest.approx(expected, rel=1e-12)
        assert out["null_distribution"] == Z2_NULL_DISTRIBUTION

    def test_its_expected_value_is_zero_under_the_true_model(self):
        rng = np.random.default_rng(5)
        values = []
        for _ in range(3000):
            realized, var, es = _true_forecasts(rng, 250, 0.975)
            values.append(
                acerbi_szekely_z2(realized, var, es, 0.025, n_simulations=1)["z2"]
            )
        mean, se = np.mean(values), np.std(values) / math.sqrt(len(values))
        assert abs(mean) < 4 * se
        assert se < 0.01

    def test_its_p_value_rejects_a_correct_model_at_about_the_level(self):
        rng = np.random.default_rng(6)
        rejects = 0
        for rep in range(200):
            realized, var, es = _true_forecasts(rng, 250, 0.975)
            p_value = acerbi_szekely_z2(
                realized, var, es, 0.025, random_seed=rep, n_simulations=1000
            )["p_value"]
            rejects += p_value < 0.05
        assert 2 <= rejects <= 22  # Binomial(200, 0.05): 10 +- 3.1 sd

    def test_an_understated_tail_is_rejected(self):
        """Student-t(4) returns with sd 0.015 forecast as if normal with sd
        0.01: more losses go through the VaR and each is larger than the ES
        says. In expectation Z2 is about -3."""
        rng = np.random.default_rng(7)
        T, c = 500, 0.975
        realized = rng.standard_t(4, T) * 0.015 / math.sqrt(2)
        z = stats.norm.ppf(c)
        var = np.full(T, 0.01 * z)
        es = np.full(T, 0.01 * stats.norm.pdf(z) / 0.025)
        out = acerbi_szekely_z2(realized, var, es, 0.025, random_seed=1)
        assert out["z2"] < -0.5
        assert out["p_value"] < 0.01

    def test_the_seed_fixes_the_p_value_and_moves_it(self):
        rng = np.random.default_rng(9)
        realized, var, es = _true_forecasts(rng, 250, 0.975)
        a = acerbi_szekely_z2(realized, var, es, 0.025, random_seed=3)
        b = acerbi_szekely_z2(realized, var, es, 0.025, random_seed=3)
        c = acerbi_szekely_z2(realized, var, es, 0.025, random_seed=4)
        assert a == b
        assert a["p_value"] != c["p_value"]
        assert a["z2"] == c["z2"]

    def test_drawing_in_smaller_blocks_changes_no_draw(self, monkeypatch):
        """The simulated series are drawn a block of rows at a time to bound
        memory; the generator fills rows in order, so the p-value is the
        same bits whatever the block."""
        from standard_quant_tools.metrics import var_backtest

        rng = np.random.default_rng(10)
        realized, var, es = _true_forecasts(rng, 250, 0.975)
        whole = acerbi_szekely_z2(realized, var, es, 0.025, n_simulations=3000)
        monkeypatch.setattr(var_backtest, "_Z2_DRAWS_PER_BLOCK", 250 * 7)
        blocks = acerbi_szekely_z2(realized, var, es, 0.025, n_simulations=3000)
        assert blocks == whole

    def test_es_not_beyond_var_is_null_with_the_reason(self):
        realized = np.full(100, 0.001)
        out = acerbi_szekely_z2(realized, np.full(100, 0.02), np.full(100, 0.02), 0.01)
        assert out["z2"] is None and "not above VaR" in out["null_reason"]
        missing = np.full(100, 0.03)
        missing[5] = np.nan
        out = acerbi_szekely_z2(realized, np.full(100, 0.02), missing, 0.01)
        assert out["z2"] is None and "no ES forecast" in out["null_reason"]

    def test_without_es_the_series_test_is_null(self):
        out = backtest_var_series(_planted([3]), 0.99, n_simulations=100)
        assert out["z2"]["z2"] is None
        assert "no ES forecasts" in out["z2"]["null_reason"]


class TestRefusals:
    def test_one_forecast_is_refused(self):
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError, match="at least two"):
            backtest_var_series(_planted([], n=1), 0.99)

    def test_a_missing_var_is_refused(self):
        from standard_quant_tools.error import ValidationError

        frame = _planted([3])
        frame.iloc[4, 1] = np.nan
        with pytest.raises(ValidationError, match="var is missing"):
            backtest_var_series(frame, 0.99)
