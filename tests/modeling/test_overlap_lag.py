"""
One Newey-West lag for every mean over dates whose labels overlap (the
CHANGELOG entry of 2026-10-04).

The run's headline was tested at max(2h, the Andrews bandwidth) while
`compare_models(method='paired')` and `compare_signals(mode='paired')` ran
their Diebold-Mariano test at h - 1. Bartlett weights cut at h - 1 recover
68% of an h-day overlap's long-run variance (1 + 2 * sum_{k<h} (1-k/h)^2 of
h), so that test was too confident. What these tests hold:

- `paired_comparison` and `diebold_mariano` use `headline_lag` for the
  dates and horizon they are given, and a lag the caller names as named;
- the Harvey-Leybourne-Newbold factor reads the forecast horizon, so a
  named lag of h - 1 reproduces the old statistic to the bit;
- `compare_signals(mode='ic_series')` reads a horizon, its default
  (horizon 1) gives the lag it gave before, and `hac_lag` wins over it;
- on a planted 5-bar overlap the old lag rejects a true zero more often
  than its nominal 5% and the new one less often than the old.
"""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.agent.statistics_models import (
    CompareSignalsInput,
)
from standard_quant_tools.modeling.agent.statistics_tools import compare_signals
from standard_quant_tools.modeling.validation.comparison import (
    andrews_lag,
    diebold_mariano,
    headline_lag,
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


def _old_statistic(differential: np.ndarray, lag: int) -> float:
    """The statistic as `diebold_mariano` computed it before, written out:
    Newey-West at `lag`, Harvey-Leybourne-Newbold at h = lag + 1."""
    n = differential.size
    h = lag + 1
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


class TestTheDieboldMarianoLag:
    def test_a_paired_comparison_uses_the_headline_lag(self):
        a, b = _frames()
        for horizon in (1, 5, 20):
            dm = paired_comparison(
                a, b, task="regression", horizon=horizon, n_bootstrap=100
            )["diebold_mariano"]
            assert dm["lag"] == headline_lag(dm["n_dates"], horizon)
        # 120 dates: Andrews gives 4, so horizon 1 reads 4, 5 reads 10.
        assert andrews_lag(120) == 4

    def test_without_a_horizon_the_andrews_bandwidth(self):
        a, b = _frames()
        dm = paired_comparison(a, b, task="regression", n_bootstrap=100)[
            "diebold_mariano"
        ]
        assert dm["lag"] == andrews_lag(120)

    def test_a_named_lag_is_used_as_named_and_reproduces_the_old_statistic(self):
        rng = np.random.default_rng(1)
        differential = 0.05 + _overlap(504, 5, rng)
        loss_a = pd.Series(differential, index=DATES)
        loss_b = pd.Series(np.zeros(504), index=DATES)
        for lag in (0, 2, 4):
            named = diebold_mariano(loss_a, loss_b, lag=lag)
            assert named["lag"] == lag
            assert named["statistic"] == _old_statistic(differential, lag)

    def test_the_small_sample_factor_reads_the_horizon(self):
        """Lag 10 for a 5-bar label is not a 11-bar forecast: the
        Harvey-Leybourne-Newbold factor is computed at h = 5."""
        rng = np.random.default_rng(2)
        differential = 0.05 + _overlap(504, 5, rng)
        loss_a = pd.Series(differential, index=DATES)
        loss_b = pd.Series(np.zeros(504), index=DATES)
        ruled = diebold_mariano(loss_a, loss_b, horizon=5)
        assert ruled["lag"] == 10
        n, h = 504, 5
        factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        expected = differential.mean() / math.sqrt(
            newey_west_variance(differential, 10)
        )
        assert ruled["statistic"] == pytest.approx(expected * factor, rel=1e-12)
        # Naming the same lag with the horizon gives the same number.
        assert diebold_mariano(loss_a, loss_b, lag=10, horizon=5)[
            "statistic"
        ] == pytest.approx(ruled["statistic"], rel=1e-15)

    def test_on_a_planted_overlap_the_old_lag_over_rejects_and_the_rule_less(self):
        """400 true-zero differentials of a 5-bar overlap on 504 dates: at
        this seed the old lag rejects 11.2% and the rule 7.0% at a nominal
        5% (4,000 draws: 11.3% and 7.9%, the CHANGELOG entry of
        2026-10-04). The rule is closer to its size, not at it: Bartlett
        weights at 2h still recover only 85% of the long-run variance."""
        rng = np.random.default_rng(20261004)
        old = new = 0
        for _ in range(400):
            differential = _overlap(504, 5, rng)
            loss_a = pd.Series(differential, index=DATES)
            loss_b = pd.Series(np.zeros(504), index=DATES)
            old += diebold_mariano(loss_a, loss_b, lag=4)["p_value"] < 0.05
            new += diebold_mariano(loss_a, loss_b, horizon=5)["p_value"] < 0.05
        assert old > new
        assert old / 400 > 0.09
        assert new / 400 < 0.09


class TestCompareSignals:
    @staticmethod
    def _maps(n=400, seed=3):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2021-01-04", periods=n)
        values = _overlap(n, 5, rng) * 0.02
        flat = {d.strftime("%Y-%m-%d"): 0.0 for d in dates}
        moved = {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(dates, values)}
        return flat, moved

    def test_ic_series_reads_the_horizon(self):
        flat, moved = self._maps()
        result = compare_signals(
            CompareSignalsInput(
                mode="ic_series", ic_a=flat, ic_b=moved, horizon=5, n_bootstrap=200
            )
        )
        assert result.hac["hac_lag"] == pytest.approx(headline_lag(400, 5))
        assert result.hac["hac_lag"] == pytest.approx(10.0)

    def test_its_default_is_the_lag_it_used_before(self):
        """Without a horizon the lag was floor(4 (n/100)^(2/9)); at the
        default horizon of 1 it still is, and so is every number."""
        flat, moved = self._maps()
        result = compare_signals(
            CompareSignalsInput(
                mode="ic_series", ic_a=flat, ic_b=moved, n_bootstrap=200
            )
        )
        lag = int(math.floor(4.0 * (400 / 100.0) ** (2.0 / 9.0)))
        difference = np.array(list(moved.values()))
        assert result.hac["hac_lag"] == lag == 5
        assert result.hac["hac_variance"] == newey_west_variance(difference, lag)

    def test_a_named_hac_lag_wins_over_the_horizon(self):
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
        assert result.hac["hac_lag"] == pytest.approx(3.0)

    def test_the_horizon_is_read_by_both_modes_and_refused_by_adjust(self):
        with pytest.raises(Exception, match="'paired' or 'ic_series'"):
            compare_signals(
                CompareSignalsInput(mode="adjust", p_values={"a": 0.1}, horizon=5)
            )
