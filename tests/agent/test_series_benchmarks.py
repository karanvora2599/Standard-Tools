"""
The four risk metrics that had no series door, and the alignment guard.

`calculate_series_metrics` is the only tool in the library that accepts an
arbitrary return series, and its registry held ten names. Four live
`risk_metrics` functions were outside it and outside every other door:

    information_ratio   needs a benchmark
    treynor_ratio       needs a benchmark; beta is estimated inside it
    drawdown_series     answers with a per-bar SERIES
    evt_tail_risk       answers with a dict of scalars

The consequence was specific rather than cosmetic. A STRATEGY's own return
series -- a backtest output, a synthetic path, anything an agent produced
itself -- could not be scored against a benchmark at all, because every
other route into those functions starts from a listed ticker and fetches
it. The same series could not be run through an extreme-value fit either,
so the only tail number available for it was the empirical quantile, which
by construction cannot see past the worst loss in the sample.

WHY THE ALIGNMENT CHECK IS HERE AND NOT INSIDE THE RATIOS. Both ratios
intersect the two indexes themselves, so a strategy and a benchmark
labelled with different dates do not raise: they are scored over whatever
overlap happens to exist, which may be a handful of days or none. Equal
length is not alignment, and the refusal has to happen before the
intersection silently answers a different question.

See the CHANGELOG entry of 2026-09-22.

THE ANSWERS ARE PLANTED, NOT SHAPE-CHECKED. A strategy set equal to its
benchmark has an information ratio of exactly zero and nothing else; a
series built as `2 * benchmark + noise` has a beta of two, so its Treynor
ratio is a number that can be computed by hand; a Pareto tail is heavier
than the empirical sample can show and a Gaussian one is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.research.reference_tools import (
    METRIC_NAMES,
    SeriesMetricsInput,
    calculate_series_metrics,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import (
    drawdown_series,
    var_historical,
)

#: Long enough that the 5% tail of the default POT fit holds well over the
#: twenty exceedances `evt_tail_risk` insists on.
N_BARS = 2000


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A private artifact store, so a published ref belongs to one test."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _metrics(values, metrics, benchmark=None, **kw):
    payload = {
        "series": {"values": [float(v) for v in values]},
        "metrics": metrics,
    }
    if benchmark is not None:
        payload["benchmark"] = {"values": [float(v) for v in benchmark]}
    return calculate_series_metrics(SeriesMetricsInput(**payload, **kw))


def _benchmark(seed: int = 11, n: int = 600) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0003, 0.010, n)


class TestTheRegistryCarriesTheFourThatHadNoDoor:
    @pytest.mark.parametrize(
        "metric",
        ["information_ratio", "treynor_ratio", "drawdown_series", "evt_tail_risk"],
    )
    def test_the_metric_is_accepted_by_name(self, metric):
        assert metric in METRIC_NAMES

    def test_the_ten_that_were_already_there_are_still_there(self):
        """An addition, not a replacement: the closed set grew and nothing
        left it."""
        assert set(METRIC_NAMES) >= {
            "cumulative_return",
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "var_historical",
            "var_parametric",
            "cvar",
            "max_drawdown",
        }

    def test_an_unknown_name_is_still_refused_with_the_list(self):
        with pytest.raises(ValidationError, match="unknown metric"):
            _metrics(_benchmark(), ["sharpe_of_the_vibes"])


class TestTheInformationRatioAgainstAPlantedBenchmark:
    def test_a_strategy_equal_to_its_benchmark_scores_exactly_zero(self):
        """Holding the benchmark IS no active bet, and zero is the honest
        answer -- not NaN, and not a number produced by dividing a
        floating-point crumb by another one."""
        benchmark = _benchmark()
        got = _metrics(benchmark, ["information_ratio"], benchmark=benchmark)
        assert got.values["information_ratio"] == 0.0
        assert got.benchmark_observations == len(benchmark)

    def test_a_constant_positive_active_return_is_undefined_not_zero(self):
        """Beating the benchmark by the same amount EVERY day has no
        tracking error to divide by. Zero would read as 'no skill', which
        is the opposite of what happened."""
        benchmark = _benchmark()
        strategy = benchmark + 0.001
        got = _metrics(strategy, ["information_ratio"], benchmark=benchmark)
        assert got.values["information_ratio"] is None

    def test_a_real_active_bet_scores_something_finite(self):
        """The null case above is only meaningful if the ordinary case is
        an ordinary number."""
        benchmark = _benchmark()
        strategy = benchmark + np.random.default_rng(12).normal(0.0002, 0.004, 600)
        got = _metrics(strategy, ["information_ratio"], benchmark=benchmark)
        assert got.values["information_ratio"] is not None
        assert abs(got.values["information_ratio"]) < 20.0


class TestTheTreynorRatioRecoversAPlantedBeta:
    def test_a_beta_of_two_gives_the_hand_computed_ratio(self):
        """r = 2b + noise has beta 2 by construction, so the Treynor ratio
        is (annualized excess return) / 2 and can be checked without
        calling the function that produced it."""
        benchmark = _benchmark()
        noise = np.random.default_rng(13).normal(0.0, 0.0015, len(benchmark))
        strategy = 2.0 * benchmark + noise
        rate, periods = 0.02, 252

        got = _metrics(
            strategy,
            ["treynor_ratio"],
            benchmark=benchmark,
            risk_free_rate=rate,
            periods_per_year=periods,
        )

        expected = (float(strategy.mean()) - rate / periods) * periods / 2.0
        assert got.values["treynor_ratio"] == pytest.approx(expected, rel=0.02)

    def test_a_market_neutral_series_is_undefined_not_zero(self):
        """Treynor is excess return PER UNIT of systematic risk. With no
        systematic risk the unit is zero and the ratio does not exist --
        0.0 would read as an unremarkable Treynor ratio.

        A return stream that does not vary has zero covariance with any
        benchmark, so its beta is exactly 0.0 rather than a rounding
        residue near it. That is the case the library singles out, and a
        cash position is what it describes."""
        benchmark = _benchmark()
        strategy = np.full(len(benchmark), 0.0004)

        got = _metrics(strategy, ["treynor_ratio"], benchmark=benchmark)
        assert got.values["treynor_ratio"] is None

    def test_the_undefined_case_is_not_simply_every_flat_answer(self):
        """The null above must not be reachable by accident: the same
        constant series scored against a benchmark it DOES track gives a
        number."""
        benchmark = _benchmark()
        strategy = 1.5 * benchmark
        got = _metrics(strategy, ["treynor_ratio"], benchmark=benchmark)
        assert got.values["treynor_ratio"] is not None


class TestABenchmarkNeedingMetricWithoutOneIsRefused:
    @pytest.mark.parametrize("metric", ["information_ratio", "treynor_ratio"])
    def test_the_refusal_names_the_slot_to_fill(self, metric):
        with pytest.raises(ValidationError, match="benchmark"):
            _metrics(_benchmark(), [metric])

    @pytest.mark.parametrize("metric", ["information_ratio", "treynor_ratio"])
    def test_the_refusal_names_the_metric_that_needed_it(self, metric):
        with pytest.raises(ValidationError, match=metric):
            _metrics(_benchmark(), [metric])

    def test_a_benchmark_nobody_reads_is_reported_rather_than_ignored(self):
        got = _metrics(_benchmark(), ["sharpe_ratio"], benchmark=_benchmark(seed=15))
        assert any("benchmark was given" in w for w in got.warnings)


class TestEqualLengthIsNotAlignment:
    """The guard that had zero callers in the package until this tool
    became its one. Two series of the same length describing different days
    are paired positionally by a NumPy path and by label by a pandas one,
    which is the backend-divergence class this suite pins against."""

    @staticmethod
    def _publish(runs_dir, name, values, start):
        index = pd.date_range(start, periods=len(values), freq="B")
        return handoff.publish(
            pd.Series(values, index=index, name=name),
            kind="analytic_series",
            run_id="alignment",
            name=name,
            producer="test",
        )

    def test_same_length_different_dates_is_refused(self, runs_dir):
        values = _benchmark(n=300)
        strategy = self._publish(runs_dir, "strategy", values, "2022-01-03")
        # Same number of bars, shifted a year: an intersection exists and is
        # empty, so nothing downstream would raise.
        benchmark = self._publish(runs_dir, "benchmark", values, "2023-01-02")

        with pytest.raises(ValidationError, match="Equal length is not alignment"):
            calculate_series_metrics(
                SeriesMetricsInput(
                    series={"ref": strategy},
                    benchmark={"ref": benchmark},
                    metrics=["information_ratio"],
                )
            )

    def test_the_refusal_counts_the_labels_in_common(self, runs_dir):
        values = _benchmark(n=300)
        strategy = self._publish(runs_dir, "s2", values, "2022-01-03")
        benchmark = self._publish(runs_dir, "b2", values, "2023-01-02")

        with pytest.raises(ValidationError, match="0 of 300 labels in common"):
            calculate_series_metrics(
                SeriesMetricsInput(
                    series={"ref": strategy},
                    benchmark={"ref": benchmark},
                    metrics=["information_ratio"],
                )
            )

    def test_the_same_dates_are_accepted(self, runs_dir):
        """Otherwise the check above would pass by refusing everything."""
        values = _benchmark(n=300)
        strategy = self._publish(runs_dir, "s3", values + 0.0001, "2022-01-03")
        benchmark = self._publish(runs_dir, "b3", values, "2022-01-03")

        got = calculate_series_metrics(
            SeriesMetricsInput(
                series={"ref": strategy},
                benchmark={"ref": benchmark},
                metrics=["information_ratio"],
            )
        )
        assert got.benchmark_observations == 300

    def test_different_lengths_are_refused_by_row_count(self):
        with pytest.raises(ValidationError, match="300 rows but benchmark has 200"):
            _metrics(
                _benchmark(n=300), ["information_ratio"], benchmark=_benchmark(n=200)
            )


class TestTheDrawdownSeriesIsPublishedRatherThanCollapsed:
    @staticmethod
    def _returns():
        return np.random.default_rng(21).normal(0.0002, 0.013, 400)

    def test_the_reference_resolves_to_the_librarys_own_series(self, runs_dir):
        returns = self._returns()
        got = _metrics(
            returns,
            ["drawdown_series", "max_drawdown"],
            run_id="dd_run",
            name="curve",
        )

        assert got.drawdown_ref == "sqt://analytic_series/dd_run/curve"
        resolved = pd.Series(handoff.resolve(got.drawdown_ref)).astype(float)
        truth = drawdown_series((1.0 + pd.Series(returns)).cumprod())
        np.testing.assert_allclose(resolved.to_numpy(), truth.to_numpy())

    def test_max_drawdown_is_the_minimum_of_the_published_series(self, runs_dir):
        """The scalar and the series are two readings of one curve, so a
        reference pointing at the wrong series would show up here."""
        returns = self._returns()
        got = _metrics(
            returns, ["drawdown_series", "max_drawdown"], run_id="dd2", name="curve"
        )
        resolved = pd.Series(handoff.resolve(got.drawdown_ref)).astype(float)
        assert got.values["max_drawdown"] == pytest.approx(
            float(resolved.min()), rel=1e-12
        )

    @pytest.mark.parametrize(
        "half,missing", [({"run_id": "r"}, "name"), ({"name": "n"}, "run_id")]
    )
    def test_exactly_one_half_of_an_address_is_refused(self, half, missing):
        """Half of `sqt://<kind>/<run_id>/<name>` addresses nothing, and
        publishing under an invented half would hand back a reference the
        caller cannot predict."""
        with pytest.raises(ValidationError, match=missing):
            _metrics(self._returns(), ["drawdown_series"], **half)

    def test_neither_half_is_refused_with_the_remedy(self, runs_dir):
        """Asking for a series and being handed nothing is the collapse
        this tool exists to stop, so it is refused rather than answered
        with an empty field."""
        with pytest.raises(ValidationError, match="run_id"):
            _metrics(self._returns(), ["drawdown_series"])

    def test_no_reference_when_the_series_was_not_asked_for(self, runs_dir):
        got = _metrics(self._returns(), ["max_drawdown"], run_id="dd3", name="unused")
        assert got.drawdown_ref is None


class TestTheExtremeValueTailFit:
    """WHAT THE FIT IS COMPARED AGAINST, AND WHY IT IS NOT THE EMPIRICAL
    QUANTILE. At 99% over 2,000 bars the empirical quantile is estimating
    the same number the POT extrapolation is, from twenty observations that
    are actually there -- measured over forty seeds the two agree to within
    a couple of percent on average in both directions, so "EVT is larger"
    would be a statement about a seed and not about a tail.

    What the fit adds is the SHAPE. `shape_xi` says how fast the tail
    decays, and two scale-free readings of it separate a heavy tail from a
    Gaussian one with no overlap at all across forty seeds: how far the
    extrapolated VaR sits above the threshold it was fitted at, and how much
    worse the average loss BEYOND that VaR is. Neither is recoverable from
    an empirical quantile, and the second is what a capital buffer is sized
    from.
    """

    @staticmethod
    def _pareto(seed: int = 31, n: int = N_BARS) -> np.ndarray:
        """Losses from a Lomax tail -- index 2.0, so the true tail shape is
        0.5 and the fourth moment does not exist."""
        return -(np.random.default_rng(seed).pareto(2.0, n) * 0.01)

    @staticmethod
    def _gaussian(seed: int = 32, n: int = N_BARS) -> np.ndarray:
        return np.random.default_rng(seed).normal(0.0, 0.01, n)

    def test_a_pareto_tail_is_classified_heavy(self):
        got = _metrics(self._pareto(), ["evt_tail_risk"])
        assert got.evt is not None
        assert got.evt["shape_xi"] > 0.1
        assert got.evt["n_obs"] == N_BARS
        assert got.evt["n_exceedances"] == pytest.approx(0.05 * N_BARS, abs=1)

    def test_a_gaussian_tail_is_not(self):
        """The null case for the detector. Without it, "heavy" would be
        satisfied by a fit that says heavy about everything."""
        assert self._gaussian() is not None
        got = _metrics(self._gaussian(), ["evt_tail_risk"])
        assert got.evt["shape_xi"] <= 0.1

    def test_the_heavy_tail_extrapolates_far_above_its_own_threshold(self):
        """Measured over forty seeds: a Lomax tail puts the 99% VaR at 2.2x
        to 3.8x its own 95% threshold, a Gaussian one at 1.3x to 1.5x. The
        gap is the shape, and it is scale-free -- multiplying every return
        by ten moves neither number."""
        heavy = _metrics(self._pareto(), ["evt_tail_risk"]).evt
        light = _metrics(self._gaussian(), ["evt_tail_risk"]).evt
        assert heavy["var_evt"] / heavy["threshold"] > 2.0
        assert light["var_evt"] / light["threshold"] < 1.6
        assert heavy["var_evt"] / heavy["threshold"] > (
            light["var_evt"] / light["threshold"]
        )

    def test_the_loss_beyond_var_is_far_worse_under_the_heavy_tail(self):
        """CVaR over VaR: 1.6x and up for the Lomax tail, under 1.25x for
        the Gaussian. This is the number a buffer is sized from, and an
        empirical quantile does not produce it."""
        heavy = _metrics(self._pareto(), ["evt_tail_risk"]).evt
        light = _metrics(self._gaussian(), ["evt_tail_risk"]).evt
        assert heavy["cvar_evt"] / heavy["var_evt"] > 1.5
        assert light["cvar_evt"] / light["var_evt"] < 1.25

    def test_the_gaussian_fit_agrees_with_the_empirical_quantile(self):
        """The calibration check. Where the sample CAN answer, the
        extrapolation must not disagree with it -- an EVT VaR that ran away
        from the quantile on ordinary data would be a fitting artifact."""
        returns = self._gaussian(n=5000)
        got = _metrics(returns, ["evt_tail_risk"])
        historical = float(var_historical(pd.Series(returns), confidence=0.99))
        assert got.evt["var_evt"] == pytest.approx(historical, rel=0.05)

    def test_the_block_is_absent_unless_the_metric_was_requested(self):
        assert _metrics(_benchmark(), ["sharpe_ratio"]).evt is None

    def test_the_block_carries_every_scalar_key_and_no_strings(self):
        got = _metrics(self._pareto(), ["evt_tail_risk"])
        assert set(got.evt) == {
            "confidence",
            "tail_fraction",
            "threshold",
            "n_exceedances",
            "n_obs",
            "shape_xi",
            "scale_beta",
            "var_evt",
            "cvar_evt",
        }
        assert all(isinstance(v, float) for v in got.evt.values())

    def test_the_fit_is_described_in_the_warnings(self):
        """The two keys that are not numbers travel as a sentence rather
        than being dropped by the numeric coercion."""
        got = _metrics(self._pareto(seed=33), ["evt_tail_risk"])
        assert any("heavy_tailed" in w and "pwm" in w for w in got.warnings)

    def test_too_few_exceedances_is_refused_by_name(self):
        """The library refuses a GPD fit below twenty exceedances, and that
        refusal is the answer -- it states the remedy and the count."""
        rng = np.random.default_rng(34)
        with pytest.raises(ValidationError, match="20 exceedances"):
            _metrics(rng.normal(0.0, 0.01, 120), ["evt_tail_risk"])

    def test_the_refusal_names_the_metric_it_came_from(self):
        rng = np.random.default_rng(35)
        with pytest.raises(ValidationError, match="evt_tail_risk"):
            _metrics(rng.normal(0.0, 0.01, 120), ["evt_tail_risk"])


class TestTheOriginalContractIsUnchanged:
    """Everything above is an addition. The equity-curve dispatch and the
    level refusal are the two things this tool got wrong before, and they
    stay exactly as they were fixed."""

    def test_an_equity_curve_metric_still_receives_the_equity_curve(self):
        from standard_quant_tools.metrics.return_metrics import cagr

        returns = pd.Series(np.random.default_rng(3).normal(0.0002, 0.013, 504))
        got = _metrics(returns, ["cagr"])
        assert got.values["cagr"] == pytest.approx(
            float(cagr((1.0 + returns).cumprod())), rel=1e-9
        )
        assert got.values["cagr"] > 0

    def test_a_level_series_is_still_refused_as_a_benchmark_too(self):
        """The `benchmark` slot resolves through the same function, so it
        inherits the refusal rather than opening a second door around it."""
        levels = 100.0 * np.cumprod(
            1.0 + np.random.default_rng(4).normal(0.0002, 0.011, 400)
        )
        with pytest.raises(ValidationError, match="RETURN series"):
            _metrics(
                _benchmark(n=400), ["information_ratio"], benchmark=levels.tolist()
            )
