"""
A model's lower quantile tested as the VaR forecast it is.

A model fitted with quantiles writes `q01`, `q05` ... into its predictions;
`backtest_var_forecasts(predictions_ref=..., quantile_column='q01')` tests
q01 as a 99% VaR, -q01 the loss and an outcome below q01 an exception.
Every oracle is planted:

- outcomes drawn from N(0, s_t) with the quantile set at exactly
  s_t * z_p: the exception rate lands on p, and over two hundred
  independent entities the Kupiec p-values are distributed as a true
  null's (its exact, discrete distribution from the binomial), with no
  entity rejected after Benjamini-Hochberg;
- the same outcomes against the 10% quantile labelled q01: ten times the
  exceptions a 99% VaR allows, and Kupiec rejects;
- each entity's line of the pooled table is the call made with that
  entity alone, number for number;
- a five-bar forward return sampled every bar: thinned to every fifth date
  (from the frame's label_end_date or from `horizon`, the same numbers
  either way) its exceptions do not cluster, and unthinned the same
  forecasts fail Christoffersen by construction -- which is why the rows
  are thinned and the result says so.

The forecasts paths are untouched: their new fields stay at their defaults,
and `tests/agent/test_var_forecast_tools.py` holds the rest unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest
from scipy import stats

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.agent.runtimes.research.var_tools import LABEL_END_COLUMN
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.alignment import LABEL_END_COL
from standard_quant_tools.modeling.validation.comparison import bh_adjust

ROW_FIELDS = (
    "n_observations",
    "n_exceptions",
    "kupiec_p_value",
    "christoffersen_p_value",
    "basel_zone",
)


@pytest.fixture(scope="module")
def research():
    return resolve("research")


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _publish(frame: pd.DataFrame, name: str = "preds") -> str:
    return handoff.publish(frame, kind="predictions", run_id="var_q", name=name)


def _panel(n_entities, n_dates, *, level=0.01, true_level=None, seed=0):
    """Outcomes N(0, s_t) with s_t varying by entity and date, and the
    column for `level` set at the `true_level` quantile of each row's own
    distribution (the right one unless told otherwise)."""
    rng = np.random.default_rng(seed)
    true_level = level if true_level is None else true_level
    dates = pd.bdate_range("2015-01-02", periods=n_dates)
    frames = []
    for e in range(n_entities):
        scale = 0.01 * np.exp(rng.normal(0.0, 0.4, n_dates))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "entity": f"E{e:03d}",
                    "prediction": 0.0,
                    "target": rng.normal(0.0, scale),
                    f"q{round(level * 100):02d}": scale * stats.norm.ppf(true_level),
                    "q50": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _kupiec_null(n: int, p: float):
    """(Kupiec p-value, probability) for every exception count of n
    forecasts that are right at rate p: the binomial pmf, and the
    proportion-of-failures LR written out with 0 ln 0 = 0."""

    def xlogy(a, b):
        return 0.0 if a == 0 else a * math.log(b)

    out = []
    for x in range(n + 1):
        lr = -2 * (xlogy(n - x, 1 - p) + xlogy(x, p)) + 2 * (
            xlogy(n - x, 1 - x / n) + xlogy(x, x / n)
        )
        out.append((float(stats.chi2.sf(max(lr, 0.0), 1)), stats.binom.pmf(x, n, p)))
    return out


def _five_bar(n_dates=2500, seed=3):
    """A five-bar forward return sampled every bar, its 5% quantile exact,
    the last five rows' outcomes not yet realized, and label_end_date five
    bars on as a dataset records it."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates)
    daily = rng.normal(0.0, 0.01, n_dates + 5)
    forward = np.array([daily[t + 1 : t + 6].sum() for t in range(n_dates)])
    forward[-5:] = np.nan
    ends = pd.Series(dates).shift(-5)
    return pd.DataFrame(
        {
            "date": dates,
            "entity": "X",
            "prediction": 0.0,
            "target": forward,
            "q05": 0.01 * math.sqrt(5) * stats.norm.ppf(0.05),
            LABEL_END_COLUMN: ends,
        }
    )


class TestTheQuantileIsTheVar:
    def test_quantiles_at_their_true_level_are_exceeded_at_that_rate(
        self, research, runs_dir
    ):
        """Two hundred independent entities, 750 dates each, the 5% quantile
        exact. The pooled rate is 5% within its sampling error, the Kupiec
        p-values are distributed as a true null's, and Benjamini-Hochberg
        leaves none rejected.

        The null's distribution is not the uniform: the count of exceptions
        is a binomial, so a Kupiec p-value takes one value per count. It is
        computed here exactly -- the binomial probability of every count
        and the LR written out by hand -- and the share of entities at or
        under each of 0.05, 0.25 and 0.5 must sit within three standard
        errors of it (a KS test against the uniform rejects a discrete
        null however right the forecasts are)."""
        frame = _panel(200, 750, level=0.05, seed=11)
        out = research.dispatch(
            "backtest_var_forecasts",
            {"predictions_ref": _publish(frame), "quantile_column": "q05"},
        )
        assert out["confidence"] == pytest.approx(0.95, abs=1e-12)
        assert out["n_observations"] == 150_000
        rate = out["n_exceptions"] / out["n_observations"]
        assert rate == pytest.approx(0.05, abs=4 * math.sqrt(0.05 * 0.95 / 150_000))
        assert out["expected_exceptions"] == pytest.approx(7500.0)
        p = np.array([row["kupiec_p_value"] for row in out["entities"].values()])
        assert len(p) == 200
        null = _kupiec_null(750, 0.05)
        for level in (0.05, 0.25, 0.5):
            exact = sum(prob for value, prob in null if value <= level + 1e-12)
            assert abs(exact - level) < 0.06  # the discreteness, small at 750
            assert abs(np.mean(p <= level + 1e-12) - exact) < 3 * math.sqrt(
                exact * (1 - exact) / 200
            )
        assert out["entities_rejected_bh"] == 0
        assert out["kupiec_p_value"] is None and out["christoffersen_lr"] is None
        assert sum(out["zone_counts"].values()) == 200
        assert out["zone_counts"] == {"none": 200}  # the light is for 99% only

    def test_a_ninety_percent_quantile_labelled_ninety_nine_is_rejected(
        self, research, runs_dir
    ):
        """The 10% quantile in the q01 column: about 10% exceptions against
        the 1% a 99% VaR allows. Kupiec rejects with p far below any level
        and the traffic light is red."""
        frame = _panel(1, 1000, level=0.01, true_level=0.10, seed=5)
        out = research.dispatch(
            "backtest_var_forecasts",
            {"predictions_ref": _publish(frame), "quantile_column": "q01"},
        )
        assert out["confidence"] == pytest.approx(0.99, abs=1e-12)
        assert 0.08 <= out["exception_rate"] <= 0.12
        assert out["kupiec_p_value"] < 1e-20
        assert out["basel_zone"] == "red"
        assert out["entities"] == {}

    def test_an_exception_is_an_outcome_below_the_quantile(self, research, runs_dir):
        """Hand-counted: outcomes -3, -1, 0.5, -2.5 against a quantile of -2
        are two exceptions, on the first and the fourth date; the VaR is the
        loss 2, and an outcome exactly at the quantile is not one."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        frame = pd.DataFrame(
            {
                "date": dates,
                "target": [-3.0, -1.0, 0.5, -2.5, -2.0],
                "q01": -2.0,
            }
        )
        out = research.dispatch(
            "backtest_var_forecasts",
            {"predictions_ref": _publish(frame), "quantile_column": "q01"},
        )
        assert out["n_observations"] == 5
        assert out["n_exceptions"] == 2
        assert out["exception_dates"] == ["2024-01-02", "2024-01-05"]


class TestEachEntityIsItsOwnSeries:
    def test_a_pooled_line_is_the_single_entity_call(self, research, runs_dir):
        """Four entities of different volatility: each row of `entities` is
        what the call with that entity named returns, and the top-level
        counts are the rows' sums."""
        frame = _panel(4, 600, level=0.01, seed=21)
        ref = _publish(frame)
        pooled = research.dispatch(
            "backtest_var_forecasts", {"predictions_ref": ref, "quantile_column": "q01"}
        )
        assert sorted(pooled["entities"]) == ["E000", "E001", "E002", "E003"]
        for name, row in pooled["entities"].items():
            alone = research.dispatch(
                "backtest_var_forecasts",
                {"predictions_ref": ref, "quantile_column": "q01", "entity": name},
            )
            assert {k: row[k] for k in ROW_FIELDS} == {k: alone[k] for k in ROW_FIELDS}
            assert alone["entities"] == {}
        rows = pooled["entities"].values()
        assert pooled["n_observations"] == sum(r["n_observations"] for r in rows)
        assert pooled["n_exceptions"] == sum(r["n_exceptions"] for r in rows)

    def test_the_kupiec_p_values_are_benjamini_hochberg_adjusted_across_them(
        self, research, runs_dir
    ):
        """The adjusted column is `bh_adjust` of the raw one, and the
        rejected count reads it at 5%."""
        frame = _panel(6, 400, level=0.01, true_level=0.02, seed=8)
        out = research.dispatch(
            "backtest_var_forecasts",
            {"predictions_ref": _publish(frame), "quantile_column": "q01"},
        )
        names = sorted(out["entities"])
        raw = [out["entities"][n]["kupiec_p_value"] for n in names]
        adjusted = [out["entities"][n]["kupiec_p_value_bh"] for n in names]
        assert adjusted == pytest.approx(bh_adjust(raw), rel=1e-12)
        assert out["entities_rejected_bh"] == sum(p <= 0.05 for p in adjusted)
        assert any(
            "No test is run on the pooled exceptions" in w for w in out["warnings"]
        )


class TestOverlappingOutcomesAreThinned:
    def test_a_five_bar_target_is_thinned_to_every_fifth_date(self, research, runs_dir):
        """label_end_date five bars on: horizon 5, every fifth date kept
        (500 of 2,500 rows, the last of them without an outcome yet), and
        the thinned exceptions do not cluster."""
        frame = _five_bar()
        out = research.dispatch(
            "backtest_var_forecasts",
            {"predictions_ref": _publish(frame), "quantile_column": "q05"},
        )
        assert out["horizon"] == 5
        assert out["n_rows_before_thinning"] == 2500
        assert out["n_observations"] == 499
        assert out["first_date"] == "2010-01-04"
        kept = set(frame["date"].iloc[::5].dt.strftime("%Y-%m-%d"))
        assert set(out["exception_dates"]) <= kept
        assert out["christoffersen_p_value"] > 0.05
        assert any("every 5th date" in w for w in out["warnings"])
        assert any("not yet realized" in w for w in out["warnings"])

    def test_unthinned_the_same_forecasts_fail_independence_by_construction(
        self, research, runs_dir
    ):
        """Without label_end_date or horizon nothing says the outcomes
        overlap: Christoffersen rejects forecasts that are exactly right,
        and the warning names the overlap the outcomes' autocorrelation
        shows (about five bars)."""
        frame = _five_bar().drop(columns=[LABEL_END_COLUMN])
        out = research.dispatch(
            "backtest_var_forecasts",
            {"predictions_ref": _publish(frame), "quantile_column": "q05"},
        )
        assert out["horizon"] == 1
        assert out["n_observations"] == 2495
        assert out["christoffersen_p_value"] < 1e-10
        assert any("about 5 bars" in w and "pass horizon" in w for w in out["warnings"])

    def test_a_horizon_given_is_the_horizon_read(self, research, runs_dir):
        """horizon=5 on the frame without label_end_date is the same test,
        number for number, as the frame's own end dates give."""
        frame = _five_bar()
        by_ends = research.dispatch(
            "backtest_var_forecasts",
            {"predictions_ref": _publish(frame, "a"), "quantile_column": "q05"},
        )
        by_input = research.dispatch(
            "backtest_var_forecasts",
            {
                "predictions_ref": _publish(
                    frame.drop(columns=[LABEL_END_COLUMN]), "b"
                ),
                "quantile_column": "q05",
                "horizon": 5,
            },
        )
        assert by_ends == by_input

    def test_a_horizon_the_end_dates_contradict_is_refused(self, research, runs_dir):
        with pytest.raises(ValidationError, match="horizon=3 but"):
            research.dispatch(
                "backtest_var_forecasts",
                {
                    "predictions_ref": _publish(_five_bar()),
                    "quantile_column": "q05",
                    "horizon": 3,
                },
            )

    def test_the_end_date_column_is_the_datasets(self):
        assert LABEL_END_COLUMN == LABEL_END_COL


class TestRefusals:
    def _ref(self, **columns):
        dates = pd.bdate_range("2024-01-02", periods=30)
        base = {"date": dates, "entity": "A", "prediction": 0.0, "target": 0.001}
        base.update(columns)
        return _publish(pd.DataFrame(base))

    @pytest.mark.parametrize(
        "columns,arguments,match",
        [
            ({"q05": -0.02}, {"quantile_column": "q01"}, r"holds \['q05'\]"),
            ({"q95": 0.02}, {"quantile_column": "q95"}, "not a lower tail"),
            (
                {"q05": -0.02},
                {"quantile_column": "q05", "confidence": 0.99},
                "0.95 VaR",
            ),
            (
                {"q05": -0.02},
                {"quantile_column": "q05", "target_column": "realized"},
                "no 'realized' column",
            ),
            ({"q05": -0.02}, {"quantile_column": "q05", "entity": "Z"}, "not in"),
            ({"q05": -np.inf}, {"quantile_column": "q05"}, "infinite"),
        ],
    )
    def test_a_frame_that_cannot_be_tested_is_refused_by_name(
        self, research, runs_dir, columns, arguments, match
    ):
        with pytest.raises(ValidationError, match=match):
            research.dispatch(
                "backtest_var_forecasts",
                {"predictions_ref": self._ref(**columns), **arguments},
            )

    def test_a_classification_frame_is_refused(self, research, runs_dir):
        """Probabilities and 0/1 labels: no quantile column to be a VaR, and
        a class label where the outcome should be."""
        dates = pd.bdate_range("2024-01-02", periods=30)
        labels = np.tile([0.0, 1.0], 15)
        frame = pd.DataFrame(
            {"date": dates, "entity": "A", "prediction": 0.6, "target": labels}
        )
        with pytest.raises(ValidationError, match="classification"):
            research.dispatch(
                "backtest_var_forecasts",
                {"predictions_ref": _publish(frame), "quantile_column": "q05"},
            )
        frame["q05"] = 0.1
        with pytest.raises(ValidationError, match="class label"):
            research.dispatch(
                "backtest_var_forecasts",
                {"predictions_ref": _publish(frame, "q"), "quantile_column": "q05"},
            )

    def test_no_outcome_column_is_refused(self, research, runs_dir):
        dates = pd.bdate_range("2024-01-02", periods=30)
        frame = pd.DataFrame({"date": dates, "entity": "A", "q05": -0.02})
        with pytest.raises(ValidationError, match="no 'target' column"):
            research.dispatch(
                "backtest_var_forecasts",
                {"predictions_ref": _publish(frame), "quantile_column": "q05"},
            )

    def test_a_repeated_entity_and_date_is_refused(self, research, runs_dir):
        dates = pd.bdate_range("2024-01-02", periods=10)
        frame = pd.DataFrame(
            {"date": dates.append(dates[:1]), "target": 0.0, "q05": -0.02}
        )
        with pytest.raises(ValidationError, match="repeated"):
            research.dispatch(
                "backtest_var_forecasts",
                {"predictions_ref": _publish(frame), "quantile_column": "q05"},
            )

    def test_another_kind_of_reference_is_refused(self, research, runs_dir):
        panel = handoff.publish(
            pd.DataFrame(
                {"A": [0.01, 0.02]}, index=pd.bdate_range("2024-01-02", periods=2)
            ),
            kind="returns_panel",
            run_id="var_q",
            name="r",
        )
        with pytest.raises(ValidationError, match="predictions"):
            research.dispatch(
                "backtest_var_forecasts",
                {"predictions_ref": panel, "quantile_column": "q05"},
            )

    @pytest.mark.parametrize(
        "arguments,match",
        [
            (
                {"predictions_ref": "sqt://predictions/x/y"},
                "quantile_column is required",
            ),
            (
                {
                    "predictions_ref": "sqt://predictions/x/y",
                    "quantile_column": "q05",
                    "forecasts_ref": "sqt://analytic_frame/x/y",
                },
                "exactly one of",
            ),
            (
                {"forecasts_ref": "sqt://analytic_frame/x/y", "quantile_column": "q05"},
                "without it",
            ),
            ({"forecasts_ref": "sqt://analytic_frame/x/y", "horizon": 5}, "without it"),
            ({}, "forecasts / forecasts_ref / predictions_ref"),
            (
                {
                    "predictions_ref": "sqt://predictions/x/y",
                    "quantile_column": "q05",
                    "horizon": 0,
                },
                "greater than or equal",
            ),
        ],
    )
    def test_the_inputs_are_refused_at_the_boundary(self, research, arguments, match):
        with pytest.raises(pydantic.ValidationError, match=match):
            research.dispatch("backtest_var_forecasts", arguments)


class TestTheForecastsPathsAreUnchanged:
    def test_their_new_fields_stay_at_their_defaults(self, research):
        rows = {
            d.strftime("%Y-%m-%d"): [0.001 if i % 20 else -0.05, 0.02, 0.03]
            for i, d in enumerate(pd.bdate_range("2023-01-02", periods=300))
        }
        out = research.dispatch(
            "backtest_var_forecasts", {"forecasts": rows, "confidence": 0.95}
        )
        assert out["entities"] == {} and out["zone_counts"] == {}
        assert out["quantile_column"] is None and out["horizon"] is None
        assert out["n_rows_before_thinning"] is None
        assert out["entities_rejected_bh"] is None
        assert out["n_exceptions"] == 15
