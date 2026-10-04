"""
One rule for every mean over dates whose labels overlap (the CHANGELOG
entries of 2026-10-04).

The run's headline, `score_predictions`' headline, the Diebold-Mariano test
of `compare_models(method='paired')` and `compare_signals(mode='paired')`,
and `compare_signals(mode='ic_series')`'s long-run variance read one rule.
It was a Newey-West variance at max(2h, the Andrews bandwidth) read against
the normal, which on simulated per-date rank ICs of a 5-bar label over 504
dates rejected a true zero 7.2% of the time at a nominal 5% (20-bar over 126
dates: 21.9%). It is now the variance of the series' lowest cosine
frequencies, min(floor(0.4 n^(2/3)), floor(n / 3h)) of them, read against
Student's t with as many degrees of freedom: 4.6% to 5.9% over twenty
cells of horizon and length. What these tests hold:

- the rule's frequency count, its floor of one and its cap of n - 1;
- `cosine_variance` is the average squared projection on the orthonormal
  DCT-II cosines, does not read the mean, and on an independent normal
  series makes the t Student's;
- `paired_comparison` and `diebold_mariano` use the rule for the dates and
  horizon they are given, with no small-sample factor; a named lag is used
  as named, with the Harvey-Leybourne-Newbold factor at the horizon, and
  reproduces both earlier statistics to the bit;
- `compare_signals(mode='ic_series')` reads the rule, the horizon
  included, and `hac_lag` wins over it; its autocorrelation sentence is
  said of persistence, not of the cosine variance's own noise;
- on a planted 5-bar overlap the first lag over-rejects, the previous rule
  less, and this one near its nominal 5%.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.fft import dct

from standard_quant_tools.modeling.agent.statistics_models import (
    CompareSignalsInput,
)
from standard_quant_tools.modeling.agent.statistics_tools import (
    _autocorrelation_warning,
    _hac_block,
    compare_signals,
)
from standard_quant_tools.modeling.validation.comparison import (
    cosine_variance,
    diebold_mariano,
    headline_degrees_of_freedom,
    headline_lag,
    mean_vs_null_test,
    newey_west_variance,
    paired_comparison,
)

DATES = pd.bdate_range("2021-01-04", periods=504)


def _overlap(n: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """The mean of h consecutive iid shocks: what an h-bar label sampled
    every bar makes of a loss differential, autocorrelation falling
    linearly to zero at lag h."""
    shocks = rng.normal(size=n + h - 1)
    return np.convolve(shocks, np.ones(h) / h, mode="valid")


def _named_lag_statistic(differential: np.ndarray, lag: int, h: int) -> float:
    """The statistic for a named lag, written out: Newey-West at `lag`,
    Harvey-Leybourne-Newbold at forecast horizon `h`."""
    n = differential.size
    correction = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    variance = newey_west_variance(differential, lag)
    return float(differential.mean() / math.sqrt(variance) * correction)


def _frames(n_dates=120, n_entities=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = []
    for date in dates:
        signal = rng.normal(size=n_entities)
        for i in range(n_entities):
            rows.append(
                {
                    "date": date,
                    "entity": f"E{i}",
                    "target": signal[i] + rng.normal(0, 0.5),
                    "a": rng.normal(),
                    "b": signal[i] + rng.normal(0, 1.0),
                }
            )
    panel = pd.DataFrame(rows)
    a = panel[["date", "entity", "target"]].assign(prediction=panel["a"])
    b = panel[["date", "entity", "target"]].assign(prediction=panel["b"])
    return a, b


class TestTheRule:
    @pytest.mark.parametrize(
        "n, horizon, expected",
        [
            (504, None, 25),
            (504, 1, 25),
            (504, 5, 25),
            (504, 10, 16),
            (504, 20, 8),
            (126, 1, 10),
            (126, 20, 2),
            (2000, 20, 33),
            (120, 5, 8),
            (10, 5, 1),
            (2, 1, 1),
            (1, 5, 0),
            (0, None, 0),
        ],
    )
    def test_the_count(self, n, horizon, expected):
        """min(floor(0.4 n^(2/3)), floor(n / 3h)), at least one."""
        assert headline_degrees_of_freedom(n, horizon) == expected

    def test_it_stays_inside_the_cosines_a_series_has(self):
        """A series of n dates has n - 1 cosines orthogonal to its mean."""
        for n in range(2, 400):
            for horizon in (1, 5, 20):
                assert 1 <= headline_degrees_of_freedom(n, horizon) <= n - 1


class TestTheCosineVariance:
    def test_it_is_the_average_squared_dct_projection(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=504)
        projections = dct(x - x.mean(), type=2, norm="ortho")[1:26]
        assert cosine_variance(x, 25) == pytest.approx(
            float(np.mean(projections**2)) / 504, rel=1e-12
        )

    def test_the_mean_does_not_enter(self):
        rng = np.random.default_rng(1)
        x = _overlap(300, 5, rng)
        assert cosine_variance(x + 7.0, 20) == pytest.approx(
            cosine_variance(x, 20), rel=1e-9
        )

    def test_degenerate_series(self):
        """A constant is not tested. Its mean is rounded, so its deviations
        from it are a few ulps rather than zero, and fifty values of 0.2
        made a t of about 1e31 under either variance."""
        assert math.isnan(cosine_variance(np.array([0.3]), 1))
        assert math.isnan(cosine_variance(np.arange(10.0), 0))
        assert cosine_variance(np.full(50, 0.2), 5) == pytest.approx(0.0, abs=1e-30)
        for lag in (None, 3):
            out = mean_vs_null_test(np.full(50, 0.2), lag=lag)
            assert math.isnan(out["t_stat"]) and math.isnan(out["p_value"])
            assert math.isnan(out["t_stat_uncorrected"])

    def test_an_independent_series_t_is_students(self):
        """4,000 independent normal series of 60 dates, six frequencies:
        the test rejects a true zero 4.3% of the time at a nominal 5%."""
        rng = np.random.default_rng(5)
        rejected = 0
        for _ in range(4000):
            out = mean_vs_null_test(rng.normal(size=60))
            assert out["degrees_of_freedom"] == 6
            rejected += out["p_value"] < 0.05
        assert 0.035 < rejected / 4000 < 0.065


class TestTheDieboldMariano:
    def test_a_paired_comparison_uses_the_rule(self):
        a, b = _frames()
        for horizon, degrees in ((1, 9), (5, 8), (20, 2)):
            dm = paired_comparison(
                a, b, task="regression", horizon=horizon, n_bootstrap=100
            )["diebold_mariano"]
            assert dm["lag"] is None
            assert dm["degrees_of_freedom"] == degrees
            assert degrees == headline_degrees_of_freedom(dm["n_dates"], horizon)

    def test_without_a_horizon_the_rule_for_the_dates(self):
        a, b = _frames()
        dm = paired_comparison(a, b, task="regression", n_bootstrap=100)[
            "diebold_mariano"
        ]
        assert dm["degrees_of_freedom"] == headline_degrees_of_freedom(120, None) == 9

    def test_the_statistic_is_the_headline_s_t_without_a_small_sample_factor(self):
        """Student's t with the frequencies' degrees of freedom is the
        small-sample reference: the Harvey-Leybourne-Newbold factor, built
        for a Newey-West variance, is not applied on top of it."""
        rng = np.random.default_rng(2)
        differential = 0.05 + _overlap(504, 5, rng)
        loss_a = pd.Series(differential, index=DATES)
        loss_b = pd.Series(np.zeros(504), index=DATES)
        ruled = diebold_mariano(loss_a, loss_b, horizon=5)
        expected = differential.mean() / math.sqrt(cosine_variance(differential, 25))
        assert ruled["statistic"] == pytest.approx(expected, rel=1e-12)
        assert ruled["p_value"] == pytest.approx(
            2 * stats.t.sf(abs(expected), 25), rel=1e-9
        )
        same = mean_vs_null_test(differential, horizon=5)
        assert ruled["statistic"] == pytest.approx(same["t_stat"], rel=1e-12)

    def test_a_named_lag_is_used_as_named_and_reproduces_the_first_statistic(self):
        """`lag=h - 1` without a horizon is what this function returned
        first: Newey-West at h - 1, the small-sample factor at lag + 1."""
        rng = np.random.default_rng(1)
        differential = 0.05 + _overlap(504, 5, rng)
        loss_a = pd.Series(differential, index=DATES)
        loss_b = pd.Series(np.zeros(504), index=DATES)
        for lag in (0, 2, 4):
            named = diebold_mariano(loss_a, loss_b, lag=lag)
            assert named["lag"] == lag and named["degrees_of_freedom"] is None
            assert named["statistic"] == _named_lag_statistic(
                differential, lag, lag + 1
            )

    def test_the_previous_rule_is_a_named_lag_away(self):
        """`lag=headline_lag(n, h), horizon=h` is the statistic the
        function returned before the cosine rule, to the bit: Newey-West at
        lag 10 for a 5-bar label, the small-sample factor at h = 5."""
        rng = np.random.default_rng(2)
        differential = 0.05 + _overlap(504, 5, rng)
        loss_a = pd.Series(differential, index=DATES)
        loss_b = pd.Series(np.zeros(504), index=DATES)
        lag = headline_lag(504, 5)
        named = diebold_mariano(loss_a, loss_b, lag=lag, horizon=5)
        assert lag == 10 and named["lag"] == 10
        assert named["statistic"] == _named_lag_statistic(differential, 10, 5)
        assert named["p_value"] == 2 * stats.norm.sf(abs(named["statistic"]))

    def test_on_a_planted_overlap_the_rule_is_near_its_size(self):
        """400 true-zero differentials of a 5-bar overlap on 504 dates: at
        this seed lag h - 1 rejects 11.3%, the previous rule 7.0% and the
        cosine rule 5.3% at a nominal 5% (20,000 draws: 7.6% for the
        previous rule and 5.2% for this one, the CHANGELOG entry of
        2026-10-04)."""
        rng = np.random.default_rng(20261004)
        first = previous = ruled = 0
        for _ in range(400):
            differential = _overlap(504, 5, rng)
            loss_a = pd.Series(differential, index=DATES)
            loss_b = pd.Series(np.zeros(504), index=DATES)
            first += diebold_mariano(loss_a, loss_b, lag=4)["p_value"] < 0.05
            previous += (
                diebold_mariano(loss_a, loss_b, lag=10, horizon=5)["p_value"] < 0.05
            )
            ruled += diebold_mariano(loss_a, loss_b, horizon=5)["p_value"] < 0.05
        assert first > previous > ruled
        assert first / 400 > 0.09
        assert 0.035 < ruled / 400 < 0.065


class TestCompareSignals:
    @staticmethod
    def _maps(n=400, seed=3):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2021-01-04", periods=n)
        values = _overlap(n, 5, rng) * 0.02
        flat = {d.strftime("%Y-%m-%d"): 0.0 for d in dates}
        moved = {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(dates, values)}
        return flat, moved

    def test_ic_series_reads_the_rule_and_the_horizon(self):
        """400 dates: 21 frequencies at horizon 1, at most 400 / 60 = 6 for
        a 20-bar label."""
        flat, moved = self._maps()
        difference = np.array(list(moved.values()))
        for horizon, degrees in ((1, 21), (20, 6)):
            result = compare_signals(
                CompareSignalsInput(
                    mode="ic_series",
                    ic_a=flat,
                    ic_b=moved,
                    horizon=horizon,
                    n_bootstrap=200,
                )
            )
            assert result.hac["hac_degrees_of_freedom"] == pytest.approx(degrees)
            assert result.hac["hac_lag"] is None
            assert result.hac["hac_variance"] == cosine_variance(difference, degrees)

    def test_a_named_hac_lag_wins_over_the_rule(self):
        flat, moved = self._maps()
        result = compare_signals(
            CompareSignalsInput(
                mode="ic_series",
                ic_a=flat,
                ic_b=moved,
                horizon=5,
                hac_lag=3,
                n_bootstrap=200,
            )
        )
        difference = np.array(list(moved.values()))
        assert result.hac["hac_lag"] == pytest.approx(3.0)
        assert result.hac["hac_degrees_of_freedom"] is None
        assert result.hac["hac_variance"] == newey_west_variance(difference, 3)

    def test_the_autocorrelation_sentence_is_not_the_variance_s_noise(self):
        """On an independent series the cosine variance's ratio to the
        ordinary one is about chi-squared over its degrees of freedom: over
        1.25 for 19.9% of 2,000 white-noise series of 400 dates, so the
        sentence waits for that distribution's 95th percentile (1.51 at 25
        frequencies) and was said of 4.2% of them. A named lag keeps the
        threshold of 1.25."""
        assert (
            _autocorrelation_warning(
                {"hac_ratio": 1.4, "hac_lag": None, "hac_degrees_of_freedom": 25.0}
            )
            is None
        )
        said = _autocorrelation_warning(
            {"hac_ratio": 1.6, "hac_lag": None, "hac_degrees_of_freedom": 25.0}
        )
        assert said is not None and "25 lowest cosine frequencies" in said
        named = _autocorrelation_warning(
            {"hac_ratio": 1.3, "hac_lag": 5.0, "hac_degrees_of_freedom": None}
        )
        assert named is not None and "at lag 5" in named
        rng = np.random.default_rng(8)
        warned = sum(
            _autocorrelation_warning(_hac_block(rng.normal(size=400), None, 1))
            is not None
            for _ in range(2000)
        )
        assert warned / 2000 < 0.06

    def test_the_horizon_is_read_by_both_modes_and_refused_by_adjust(self):
        with pytest.raises(Exception, match="'paired' or 'ic_series'"):
            compare_signals(
                CompareSignalsInput(mode="adjust", p_values={"a": 0.1}, horizon=5)
            )
