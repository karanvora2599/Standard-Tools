"""
The VaR forecast and its backtest as tools: what they take, what they hand
each other, and what they refuse.

`forecast_var_series` publishes its series as an `analytic_frame` --
realized, var, es and confidence by date -- and `backtest_var_forecasts`
reads that reference, or the same numbers inline as
{ISO date: [return, VaR, ES]}. Held here:

- the reference round trip: the tester on the reference and on the same
  forecasts inline gives the same answer, the library's to the bit;
- a published returns panel's column forecasts exactly as its values
  passed inline;
- the forecasts' own level travels with them, and a different one is
  refused rather than tested;
- each statistic that is undefined is null and the warnings say why;
- both tools are `quant_research` tools of the research runtime, on the
  facade, taking no ticker.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.var_backtest import backtest_var_series
from standard_quant_tools.metrics.var_forecast import forecast_var_series

TOOLS = ("forecast_var_series", "backtest_var_forecasts")


@pytest.fixture(scope="module")
def research():
    return resolve("research")


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _returns(n=600, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    return pd.Series(rng.standard_t(5, n) * 0.008, index=dates)


def _inline(series):
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in series.items()}


class TestTheRoundTrip:
    def test_the_published_series_is_the_librarys(self, research, runs_dir):
        returns = _returns()
        out = research.dispatch(
            "forecast_var_series",
            {"returns": _inline(returns), "method": "normal", "run_id": "v1"},
        )
        assert out["forecasts_ref"] == "sqt://analytic_frame/v1/var_forecasts"
        frame = handoff.resolve(out["forecasts_ref"], expect="analytic_frame")
        assert list(frame.columns) == ["realized", "var", "es", "confidence"]
        expected = forecast_var_series(returns, method="normal")
        pd.testing.assert_frame_equal(
            frame, expected["forecasts"], check_freq=False, check_names=False
        )
        assert out["next_var"] == expected["next_var"]
        assert out["n_forecasts"] == 350
        assert out["first_forecast_date"] == str(returns.index[250].date())

    def test_the_tester_reads_the_reference_as_it_reads_inline(
        self, research, runs_dir
    ):
        returns = _returns()
        published = research.dispatch(
            "forecast_var_series",
            {"returns": _inline(returns), "method": "historical", "run_id": "v2"},
        )
        by_ref = research.dispatch(
            "backtest_var_forecasts",
            {"forecasts_ref": published["forecasts_ref"], "random_seed": 4},
        )
        frame = handoff.resolve(published["forecasts_ref"], expect="analytic_frame")
        rows = {
            d.strftime("%Y-%m-%d"): [row["realized"], row["var"], row["es"]]
            for d, row in frame.iterrows()
        }
        inline = research.dispatch(
            "backtest_var_forecasts",
            {"forecasts": rows, "confidence": 0.99, "random_seed": 4},
        )
        assert by_ref == inline
        library = backtest_var_series(
            frame[["realized", "var", "es"]], 0.99, random_seed=4
        )
        assert by_ref["kupiec_lr"] == library["kupiec"]["lr"]
        assert by_ref["z2"] == library["z2"]["z2"]
        assert by_ref["z2_p_value"] == library["z2"]["p_value"]
        assert by_ref["n_exceptions"] == library["n_exceptions"]
        assert len(by_ref["exception_dates"]) == by_ref["n_exceptions"]

    def test_a_panel_column_forecasts_as_its_values_inline(self, research, runs_dir):
        returns = _returns()
        other = _returns(seed=9)
        panel = handoff.publish(
            pd.DataFrame({"A": returns, "B": other}),
            kind="returns_panel",
            run_id="v3",
            name="rets",
        )
        by_ref = research.dispatch(
            "forecast_var_series",
            {"returns_ref": panel, "column": "B", "method": "ewma"},
        )
        inline = research.dispatch(
            "forecast_var_series", {"returns": _inline(other), "method": "ewma"}
        )
        assert by_ref == inline

    def test_without_run_id_nothing_is_published_and_it_says_so(
        self, research, runs_dir
    ):
        out = research.dispatch("forecast_var_series", {"returns": _inline(_returns())})
        assert out["forecasts_ref"] is None
        assert any("pass run_id" in w for w in out["warnings"])
        assert not (runs_dir).exists() or not any(runs_dir.iterdir())


class TestTheLevelTravelsWithTheForecasts:
    def test_a_different_level_is_refused(self, research, runs_dir):
        out = research.dispatch(
            "forecast_var_series",
            {"returns": _inline(_returns()), "confidence": 0.975, "run_id": "v4"},
        )
        with pytest.raises(ValidationError, match="forecast at 0.975"):
            research.dispatch(
                "backtest_var_forecasts",
                {"forecasts_ref": out["forecasts_ref"], "confidence": 0.99},
            )
        same = research.dispatch(
            "backtest_var_forecasts",
            {"forecasts_ref": out["forecasts_ref"], "confidence": 0.975},
        )
        assert same["confidence"] == 0.975

    def test_inline_forecasts_need_their_level(self, research):
        rows = {"2024-01-02": [0.001, 0.02], "2024-01-03": [-0.03, 0.02]}
        with pytest.raises(pydantic.ValidationError, match="confidence is required"):
            research.dispatch("backtest_var_forecasts", {"forecasts": rows})

    def test_a_returns_panel_is_not_a_forecast_series(self, research, runs_dir):
        panel = handoff.publish(
            pd.DataFrame({"A": _returns()}), kind="returns_panel", run_id="v5", name="r"
        )
        with pytest.raises(ValidationError, match="analytic_frame"):
            research.dispatch(
                "backtest_var_forecasts", {"forecasts_ref": panel, "confidence": 0.99}
            )


class TestNullsSayWhy:
    def test_off_99_the_traffic_light_is_null_with_the_reason(self, research):
        rows = {
            d.strftime("%Y-%m-%d"): [0.001 if i % 20 else -0.05, 0.02, 0.03]
            for i, d in enumerate(pd.bdate_range("2023-01-02", periods=300))
        }
        out = research.dispatch(
            "backtest_var_forecasts", {"forecasts": rows, "confidence": 0.95}
        )
        assert out["n_exceptions"] == 15
        assert out["basel_zone"] is None and out["basel_plus_factor"] is None
        assert any(
            "basel_* fields are null" in w and "99% VaR" in w for w in out["warnings"]
        )
        assert out["kupiec_lr"] == pytest.approx(0.0, abs=1e-12)

    def test_no_exception_leaves_independence_untested(self, research):
        rows = {
            d.strftime("%Y-%m-%d"): [0.001, 0.02]
            for d in pd.bdate_range("2023-01-02", periods=260)
        }
        out = research.dispatch(
            "backtest_var_forecasts", {"forecasts": rows, "confidence": 0.99}
        )
        assert out["n_exceptions"] == 0
        assert out["basel_zone"] == "green"
        assert out["christoffersen_lr"] is None
        assert out["conditional_coverage_lr"] is None
        assert out["kupiec_lr"] == pytest.approx(-2 * 260 * math.log(0.99), rel=1e-12)
        assert any("no clustering to test" in w for w in out["warnings"])
        assert out["z2"] is None
        assert any("no ES forecasts" in w for w in out["warnings"])


class TestRefusals:
    @pytest.mark.parametrize(
        "arguments,match",
        [
            ({}, "exactly one of returns / returns_ref"),
            (
                {"returns": {"2024-01-02": 0.01}, "returns_ref": "sqt://x/y/z"},
                "exactly one of",
            ),
            ({"returns": {"2024-01-02": 0.01}, "column": "A"}, "without it"),
            ({"returns": {"AAPL": 0.01}}, "not"),
            ({"returns": {"2024-01-02": 0.01}, "window": 10}, "greater than or equal"),
            (
                {"returns": {"2024-01-02": 0.01}, "symbol": "AAPL"},
                "no argument 'symbol'",
            ),
        ],
    )
    def test_the_forecaster_refuses_by_name(self, research, arguments, match):
        with pytest.raises((pydantic.ValidationError, ValidationError), match=match):
            research.dispatch("forecast_var_series", arguments)

    def test_a_history_shorter_than_the_window_is_refused(self, research):
        with pytest.raises(ValidationError, match="at least 251"):
            research.dispatch(
                "forecast_var_series", {"returns": _inline(_returns(200))}
            )

    def test_a_level_series_is_refused_as_not_returns(self, research):
        prices = 100 + _returns().cumsum()
        with pytest.raises(ValidationError, match="RETURN series"):
            research.dispatch("forecast_var_series", {"returns": _inline(prices)})

    @pytest.mark.parametrize(
        "rows,match",
        [
            (
                {"2024-01-02": [0.0, 0.02], "2024-01-03": [0.0, 0.02, 0.03]},
                "all of one",
            ),
            ({"2024-01-02": [0.0], "2024-01-03": [0.0]}, "all of one"),
            ({"2024-01-02": [0.0, float("nan")], "2024-01-03": [0.0, 0.02]}, "finite"),
        ],
    )
    def test_the_tester_refuses_malformed_rows(self, research, rows, match):
        with pytest.raises(pydantic.ValidationError, match=match):
            research.dispatch(
                "backtest_var_forecasts", {"forecasts": rows, "confidence": 0.99}
            )

    def test_a_key_that_is_not_a_date_is_refused(self, research):
        rows = {"2024-01-02": [0.0, 0.02], "not-a-date": [0.0, 0.02]}
        with pytest.raises(ValidationError, match="not-a-date"):
            research.dispatch(
                "backtest_var_forecasts", {"forecasts": rows, "confidence": 0.99}
            )


class TestRegistration:
    @pytest.mark.parametrize("name", TOOLS)
    def test_each_is_a_quant_research_tool_of_the_research_runtime(self, name):
        from standard_quant_tools.agent import tools as facade

        assert name in resolve("research").dispatch_table
        assert facade.TOOL_CATEGORY[name] == "quant_research"
        assert getattr(facade, name) is resolve("research").dispatch_table[name][0]

    @pytest.mark.parametrize("name", TOOLS)
    def test_neither_takes_a_ticker(self, name):
        """Inline data or a reference only: nothing here fetches, so both are
        inside the determinism layer and the same input gives the same bits."""
        from standard_quant_tools.mcp.schemas import property_names

        model = resolve("research").dispatch_table[name][1]
        names = set(property_names(model.model_json_schema()))
        assert not names & {"symbol", "symbols", "ticker", "tickers", "universe"}
        assert "benchmark" not in names and "book" not in names

    def test_they_are_exported_from_the_package(self):
        import standard_quant_tools.agent as package

        for name in TOOLS:
            assert name in package.__all__
            assert hasattr(package, name)
