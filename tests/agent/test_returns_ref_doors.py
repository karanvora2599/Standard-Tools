"""
Five tools take a published `returns_panel` in place of their inline returns.

`estimate_covariance`, `optimize_hierarchical_risk_parity`, the deflated
Sharpe, PBO and the reality check read returns only inline -- or, for the
covariance, only by fetching tickers -- so a panel another tool had already
published had to be read back and transcribed through a context window,
with nothing to say whether its columns shared their dates. Each now takes
`returns_ref` instead: its columns are the series, put on the dates they all
share as return streams (each column's losses reported in `streams`), and
the answer is the inline answer to the bit. Exactly one of inline returns
and `returns_ref` is taken; both or neither is refused by name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.agent.runtimes.portfolio import tools as portfolio_tools
from standard_quant_tools.error import ValidationError

COLUMNS = ["A", "B", "C", "D"]


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


@pytest.fixture(scope="module")
def portfolio():
    return resolve("portfolio")


@pytest.fixture(scope="module")
def backtest():
    return resolve("backtest")


@pytest.fixture(scope="module")
def planted():
    """300 business days of four return series, two sharing a factor."""
    rng = np.random.default_rng(20261022)
    dates = pd.bdate_range("2023-01-02", periods=300)
    common = rng.normal(0.0003, 0.009, 300)
    return pd.DataFrame(
        {
            "A": common + rng.normal(0.0002, 0.004, 300),
            "B": common + rng.normal(0.0001, 0.006, 300),
            "C": rng.normal(0.0004, 0.012, 300),
            "D": rng.normal(0.0, 0.008, 300),
        },
        index=dates,
    )


@pytest.fixture
def panel_ref(runs_dir, planted):
    return handoff.publish(planted, kind="returns_panel", run_id="doors", name="rets")


def _lists(frame, columns):
    return {c: frame[c].tolist() for c in columns}


def _dated(frame, columns):
    return {c: {str(d.date()): float(v) for d, v in frame[c].items()} for c in columns}


def _without(result, *fields):
    return {k: v for k, v in result.items() if k not in fields}


def _doors(planted, panel_ref):
    """(runtime, tool, inline arguments, reference arguments) per door."""
    return [
        (
            "portfolio",
            "optimize_hierarchical_risk_parity",
            {"returns": _dated(planted, COLUMNS)},
            {"returns_ref": panel_ref},
        ),
        (
            "backtest",
            "get_deflated_sharpe_ratio",
            {"returns": planted["C"].tolist(), "n_trials": 7},
            {"returns_ref": panel_ref, "returns_column": "C", "n_trials": 7},
        ),
        (
            "backtest",
            "estimate_backtest_overfitting",
            {"trial_returns": _lists(planted, COLUMNS), "n_splits": 6},
            {"returns_ref": panel_ref, "n_splits": 6},
        ),
        (
            "backtest",
            "run_reality_check",
            {
                "candidate_returns": _lists(planted, ["B", "C", "D"]),
                "benchmark_returns": planted["A"].tolist(),
                "n_bootstrap": 300,
                "seed": 5,
            },
            {
                "returns_ref": panel_ref,
                "benchmark_column": "A",
                "n_bootstrap": 300,
                "seed": 5,
            },
        ),
        (
            "backtest",
            "run_reality_check",
            {
                "question": "beats_every_alternative",
                "strategy_returns": planted["D"].tolist(),
                "alternative_returns": _lists(planted, ["A", "B", "C"]),
                "n_bootstrap": 300,
                "seed": 5,
            },
            {
                "question": "beats_every_alternative",
                "returns_ref": panel_ref,
                "strategy_column": "D",
                "n_bootstrap": 300,
                "seed": 5,
            },
        ),
    ]


class TestTheReferenceAnswersAsTheInlineReturnsDo:
    def test_every_door_gives_the_inline_answer_to_the_bit(
        self, planted, panel_ref, monkeypatch
    ):
        """
        The sweep: the same returns, inline and as a published panel, give
        the same result at each of the five doors -- every number, every
        warning -- but for the `streams` report the reference adds. The
        covariance's inline path is its fetch, answered with the planted
        frame.
        """
        monkeypatch.setattr(
            portfolio_tools, "fetch_returns_sync", lambda *a, **k: planted
        )
        doors = _doors(planted, panel_ref) + [
            (
                "portfolio",
                "estimate_covariance",
                {
                    "tickers": COLUMNS,
                    "start_date": "2023-01-02",
                    "end_date": "2024-02-23",
                },
                {"returns_ref": panel_ref},
            )
        ]
        for runtime, tool, inline, by_ref in doors:
            runner = resolve(runtime)
            first = runner.dispatch(tool, inline)
            second = runner.dispatch(tool, by_ref)
            assert _without(second, "streams") == _without(first, "streams"), tool
            if "streams" in second:
                assert first["streams"] is None, tool
                assert second["streams"]["n_dates"] == 300, tool

    def test_the_report_says_which_columns_were_read_and_that_none_lost_a_date(
        self, backtest, panel_ref
    ):
        out = backtest.dispatch(
            "run_reality_check",
            {"returns_ref": panel_ref, "benchmark_column": "A", "n_bootstrap": 100},
        )
        report = out["streams"]
        assert sorted(report["by_stream"]) == COLUMNS
        assert out["n_candidates"] == 3 and out["benchmark_given"] is True
        assert all(
            row["n_kept"] == row["n_supplied"] == 300
            for row in report["by_stream"].values()
        )


class TestGapsAreAlignedAndReported:
    @pytest.fixture
    def gapped_ref(self, runs_dir, planted):
        """
        Planted gaps: C is blank on its first 20 dates and D on its last 10,
        and B on 5 dates in the middle. The shared dates are 300 - 20 - 10 -
        5 = 265.
        """
        frame = planted.copy()
        frame.iloc[:20, frame.columns.get_loc("C")] = np.nan
        frame.iloc[-10:, frame.columns.get_loc("D")] = np.nan
        frame.iloc[100:105, frame.columns.get_loc("B")] = np.nan
        return handoff.publish(frame, kind="returns_panel", run_id="gaps", name="r")

    def test_the_covariance_is_the_one_on_the_shared_dates(
        self, portfolio, planted, gapped_ref, monkeypatch
    ):
        """Known answer: the matrix of the 265 rows every column has,
        estimated directly, and a report naming what each column lost."""
        keep = planted.index[20:-10].delete(range(80, 85))
        monkeypatch.setattr(
            portfolio_tools, "fetch_returns_sync", lambda *a, **k: planted.loc[keep]
        )
        by_hand = portfolio.dispatch(
            "estimate_covariance",
            {"tickers": COLUMNS, "start_date": "2023-01-02", "end_date": "2024-02-23"},
        )
        out = portfolio.dispatch("estimate_covariance", {"returns_ref": gapped_ref})
        assert out["matrix"] == by_hand["matrix"]
        assert out["n_observations"] == 265
        assert out["n_rows_dropped"] == 35
        rows = out["streams"]["by_stream"]
        assert rows["A"]["n_dropped_before"] == 20
        assert rows["A"]["n_dropped_after"] == 10
        assert rows["A"]["n_dropped_inside"] == 5
        assert rows["C"]["n_supplied"] == 280 and rows["C"]["n_dropped_before"] == 0
        assert rows["D"]["n_dropped_after"] == 0
        assert out["streams"]["start_set_by"] == ["C"]
        assert out["streams"]["end_set_by"] == ["D"]
        assert any("dropped" in w for w in out["warnings"])

    def test_hrp_exact_refuses_a_gapped_panel_and_intersect_reports_it(
        self, portfolio, gapped_ref
    ):
        with pytest.raises(ValidationError) as refusal:
            portfolio.dispatch(
                "optimize_hierarchical_risk_parity", {"returns_ref": gapped_ref}
            )
        message = str(refusal.value)
        assert "do not carry the same dates" in message
        assert "300 dates appear across them and 265 are in every one" in message
        assert "alignment='intersect'" in message
        out = portfolio.dispatch(
            "optimize_hierarchical_risk_parity",
            {"returns_ref": gapped_ref, "alignment": "intersect"},
        )
        assert out["n_observations"] == 265
        assert out["alignment"]["by_series"]["A"]["n_dropped"] == 35

    def test_a_short_overlap_names_the_column_that_limits_it(
        self, runs_dir, backtest, planted
    ):
        """PBO at 8 chunks needs 40 shared rows; D, blank on all but its
        last 30, leaves 30 and is named as what limits them."""
        frame = planted.copy()
        frame.iloc[:-30, frame.columns.get_loc("D")] = np.nan
        ref = handoff.publish(frame, kind="returns_panel", run_id="short", name="r")
        with pytest.raises(ValidationError) as refusal:
            backtest.dispatch(
                "estimate_backtest_overfitting", {"returns_ref": ref, "n_splits": 8}
            )
        message = str(refusal.value)
        assert "share 30 date(s), fewer than min_overlap=40" in message
        assert "['D'] limit(s) it most: without it 300 dates" in message


class TestWhatIsRefusedAtTheDoors:
    @pytest.mark.parametrize(
        "runtime, tool, fields",
        [
            ("portfolio", "optimize_hierarchical_risk_parity", {}),
            ("backtest", "get_deflated_sharpe_ratio", {"n_trials": 3}),
            ("backtest", "estimate_backtest_overfitting", {}),
        ],
    )
    def test_neither_inline_returns_nor_a_reference_is_refused_by_name(
        self, runtime, tool, fields
    ):
        with pytest.raises(pydantic.ValidationError, match="returns_ref"):
            resolve(runtime).dispatch(tool, fields)

    def test_both_are_refused_at_every_door(self, planted, panel_ref):
        for runtime, tool, inline, by_ref in _doors(planted, panel_ref):
            with pytest.raises(pydantic.ValidationError, match="returns_ref"):
                resolve(runtime).dispatch(tool, {**inline, **by_ref})

    def test_estimate_covariance_takes_tickers_or_a_reference(
        self, portfolio, panel_ref
    ):
        with pytest.raises(pydantic.ValidationError, match="neither"):
            portfolio.dispatch("estimate_covariance", {})
        with pytest.raises(pydantic.ValidationError, match="both"):
            portfolio.dispatch(
                "estimate_covariance",
                {
                    "tickers": COLUMNS,
                    "start_date": "2023-01-02",
                    "end_date": "2024-02-23",
                    "returns_ref": panel_ref,
                },
            )
        with pytest.raises(pydantic.ValidationError, match="only apply to tickers"):
            portfolio.dispatch(
                "estimate_covariance",
                {"returns_ref": panel_ref, "start_date": "2023-01-02"},
            )
        with pytest.raises(pydantic.ValidationError, match="needs end_date"):
            portfolio.dispatch(
                "estimate_covariance",
                {"tickers": COLUMNS, "start_date": "2023-01-02"},
            )

    def test_the_deflated_sharpe_is_of_one_named_column(self, backtest, panel_ref):
        with pytest.raises(ValidationError, match="Name it in returns_column"):
            backtest.dispatch(
                "get_deflated_sharpe_ratio", {"returns_ref": panel_ref, "n_trials": 3}
            )
        with pytest.raises(ValidationError, match="returns_column='Z' is not a column"):
            backtest.dispatch(
                "get_deflated_sharpe_ratio",
                {"returns_ref": panel_ref, "returns_column": "Z", "n_trials": 3},
            )
        with pytest.raises(pydantic.ValidationError, match="without it"):
            backtest.dispatch(
                "get_deflated_sharpe_ratio",
                {"returns": [0.01] * 30, "returns_column": "A", "n_trials": 3},
            )

    @pytest.mark.parametrize(
        "fields, says",
        [
            (
                {"strategy_column": "A"},
                "strategy_column is read under question='beats_every_alternative'",
            ),
            (
                {"question": "beats_every_alternative", "benchmark_column": "A"},
                "benchmark_column is read under question='best_candidate_vs_benchmark'",
            ),
            ({"question": "beats_every_alternative"}, "needs strategy_column"),
            (
                {"candidate_returns": {"x": [0.01] * 40}},
                "candidate_returns and returns_ref were both given",
            ),
            (
                {
                    "question": "beats_every_alternative",
                    "strategy_column": "A",
                    "method": "hansen_spa",
                },
                "Leave method at 'white'",
            ),
        ],
    )
    def test_the_reality_checks_roles_are_refused_when_ambiguous(
        self, backtest, panel_ref, fields, says
    ):
        with pytest.raises(pydantic.ValidationError, match=says):
            backtest.dispatch("run_reality_check", {"returns_ref": panel_ref, **fields})

    def test_a_role_column_the_panel_lacks_is_named(self, backtest, panel_ref):
        with pytest.raises(
            ValidationError, match="benchmark_column='Z' is not a column"
        ):
            backtest.dispatch(
                "run_reality_check", {"returns_ref": panel_ref, "benchmark_column": "Z"}
            )

    def test_a_column_given_without_the_reference_is_refused(self, backtest):
        with pytest.raises(pydantic.ValidationError, match="given without it"):
            backtest.dispatch(
                "run_reality_check",
                {"candidate_returns": {"x": [0.01] * 40}, "benchmark_column": "A"},
            )

    def test_an_equity_curve_is_refused_naming_both_kinds(self, runs_dir, backtest):
        dates = pd.bdate_range("2024-01-02", periods=40)
        curve = handoff.publish(
            pd.Series(np.linspace(100, 120, 40), index=dates, name="equity"),
            kind="equity_curve",
            run_id="curve",
            name="eq",
        )
        with pytest.raises(ValidationError) as refusal:
            backtest.dispatch(
                "get_deflated_sharpe_ratio", {"returns_ref": curve, "n_trials": 2}
            )
        message = str(refusal.value)
        assert "'returns_panel'" in message and "'equity_curve'" in message
        assert "convert_reference" in message

    def test_levels_published_as_returns_are_refused(self, runs_dir, backtest):
        dates = pd.bdate_range("2024-01-02", periods=60)
        levels = pd.DataFrame(
            {"A": np.linspace(100, 130, 60), "B": np.linspace(50, 40, 60)},
            index=dates,
        )
        ref = handoff.publish(levels, kind="returns_panel", run_id="lv", name="r")
        with pytest.raises(ValidationError, match="level series"):
            backtest.dispatch(
                "estimate_backtest_overfitting", {"returns_ref": ref, "n_splits": 4}
            )

    def test_an_undated_panel_is_refused(self, runs_dir, backtest):
        ref = handoff.publish(
            pd.DataFrame({"A": [0.01] * 40, "B": [0.02] * 40}),
            kind="returns_panel",
            run_id="undated",
            name="r",
        )
        with pytest.raises(ValidationError, match="Undated values are not a stream"):
            backtest.dispatch(
                "estimate_backtest_overfitting", {"returns_ref": ref, "n_splits": 4}
            )
