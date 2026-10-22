"""
The portfolio construction tools take what the other tools hand them.

Two hand-offs failed at the door. `estimate_covariance` returns its matrix as
a map of asset to {asset: covariance}, and the four tools that consume a
covariance -- risk parity, maximum diversification, marginal risk
contribution and the scenario sigmas -- took only rows parallel to `assets`
and refused the map as `list_type`: the output of one could not be the input
of the next. And HRP took undated return lists, so two equal-length series
from different calendars were accepted as aligned and correlated each day
against some other day. See the CHANGELOG entry of 2026-10-22.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.portfolio import tools as portfolio_tools
from standard_quant_tools.agent.runtimes.portfolio.inline_inputs import (
    covariance_rows,
)
from standard_quant_tools.portfolio import construction as lib

TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def portfolio():
    return resolve("portfolio")


def _door_arguments(assets):
    """What each covariance-taking tool needs besides the matrix."""
    weights = {a: w for a, w in zip(assets, (0.5, 0.3, 0.2))}
    return {
        "optimize_risk_parity": {},
        "optimize_max_diversification": {},
        "get_marginal_risk_contribution": {"weights": weights},
        "run_portfolio_scenarios": {
            "weights": weights,
            "scenarios": {"selloff": {a: -0.1 for a in assets}},
        },
    }


@pytest.fixture
def estimated(monkeypatch, portfolio):
    """estimate_covariance's own answer, on planted returns rather than a
    fetch: three names, two sharing a common factor."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2024-01-01", periods=250)
    common = rng.normal(0, 0.01, 250)
    returns = pd.DataFrame(
        {
            "AAA": common + rng.normal(0, 0.006, 250),
            "BBB": common + rng.normal(0, 0.009, 250),
            "CCC": rng.normal(0, 0.015, 250),
        },
        index=index,
    )
    monkeypatch.setattr(
        portfolio_tools, "fetch_returns_sync", lambda *args, **kwargs: returns
    )
    return portfolio.dispatch(
        "estimate_covariance",
        {"tickers": TICKERS, "start_date": "2024-01-01", "end_date": "2024-12-31"},
    )


class TestTheCovarianceHandOff:
    def test_estimate_covariance_feeds_every_door_as_it_comes_back(
        self, portfolio, estimated
    ):
        """
        End to end: the `assets` and `matrix` estimate_covariance returned,
        passed on unchanged, give each of the four tools exactly the answer
        the same matrix gives as rows.
        """
        assets, matrix = estimated["assets"], estimated["matrix"]
        assert isinstance(matrix, dict)
        rows = [[matrix[r][c] for c in assets] for r in assets]
        for tool, extra in _door_arguments(assets).items():
            by_name = portfolio.dispatch(
                tool, {"assets": assets, "covariance": matrix, **extra}
            )
            by_rows = portfolio.dispatch(
                tool, {"assets": assets, "covariance": rows, **extra}
            )
            assert by_name == by_rows, tool

    def test_the_map_is_read_by_name_not_by_position(self, portfolio):
        """
        Planted: two uncorrelated assets with volatilities 0.2 and 0.3, so
        equal risk contribution puts weight in inverse proportion to
        volatility -- 0.6 and 0.4 -- whichever order the map and `assets`
        list them in.
        """
        matrix = {"B": {"B": 0.09, "A": 0.0}, "A": {"B": 0.0, "A": 0.04}}
        for assets in (["A", "B"], ["B", "A"]):
            weights = portfolio.dispatch(
                "optimize_risk_parity", {"assets": assets, "covariance": matrix}
            )["weights"]
            assert weights == pytest.approx({"A": 0.6, "B": 0.4}, abs=1e-9)

    @pytest.mark.parametrize(
        "assets, matrix, says",
        [
            (
                ["A", "C"],
                {"A": {"A": 0.04, "B": 0.0}, "B": {"A": 0.0, "B": 0.09}},
                r"no row for \['C'\] and rows for \['B'\]",
            ),
            (
                ["A", "B"],
                {"A": {"A": 0.04}, "B": {"A": 0.0, "B": 0.09}},
                r"covariance\['A'\] lacks \['B'\]",
            ),
            (
                ["A", "B"],
                {"A": {"A": 0.04, "B": 0.01}, "B": {"A": 0.02, "B": 0.09}},
                "not symmetric",
            ),
        ],
    )
    def test_a_map_that_is_not_a_covariance_of_assets_is_refused(
        self, portfolio, assets, matrix, says
    ):
        """Names that disagree with `assets`, a row without a column, and an
        asymmetric map are refused at every door -- the last by the numeric
        contract, exactly as the list form is."""
        for tool, extra in _door_arguments(assets).items():
            with pytest.raises(ValueError, match=says):
                portfolio.dispatch(
                    tool, {"assets": assets, "covariance": matrix, **extra}
                )

    def test_scenarios_still_need_assets_beside_a_map(self, portfolio):
        with pytest.raises(ValueError, match="needs `assets`"):
            portfolio.dispatch(
                "run_portfolio_scenarios",
                {
                    "weights": {"A": 0.5, "B": 0.5},
                    "scenarios": {"s": {"A": -0.1}},
                    "covariance": {
                        "A": {"A": 0.04, "B": 0.0},
                        "B": {"A": 0.0, "B": 0.09},
                    },
                },
            )

    def test_rows_come_out_in_assets_order(self):
        assert covariance_rows(
            {"B": {"A": 1.0, "B": 3.0}, "A": {"A": 2.0, "B": 1.0}}, ["A", "B"]
        ) == [[2.0, 1.0], [1.0, 3.0]]


def _dated(values, dates):
    return {str(d.date()): float(v) for d, v in zip(dates, values)}


class TestHRPSeesTheDates:
    """Three names over 80 business days; each series given 60 of them."""

    @pytest.fixture(scope="class")
    def planted(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2024-01-01", periods=80)
        common = rng.normal(0, 0.01, 80)
        return {
            "dates": dates,
            "a": common + rng.normal(0, 0.002, 80),
            "b": common + rng.normal(0, 0.002, 80),
            "c": rng.normal(0, 0.02, 80),
        }

    def test_equal_length_series_on_different_dates_are_refused(
        self, portfolio, planted
    ):
        """
        The case undated lists could not catch: three 60-day series, one of
        them 20 business days later. Equal lengths, different dates.
        """
        d = planted["dates"]
        returns = {
            "a": _dated(planted["a"][:60], d[:60]),
            "b": _dated(planted["b"][20:], d[20:]),
            "c": _dated(planted["c"][:60], d[:60]),
        }
        with pytest.raises(ValueError) as refusal:
            portfolio.dispatch(
                "optimize_hierarchical_risk_parity", {"returns": returns}
            )
        message = str(refusal.value)
        assert "do not carry the same dates" in message
        assert "80 dates appear across them and 40 are in every one" in message
        assert "all 60 long" in message
        assert "alignment='intersect'" in message

    def test_intersect_uses_the_shared_dates_and_says_what_each_lost(
        self, portfolio, planted
    ):
        """Known answer: the weights HRP gives on the 40 dates all three
        series share, aligned by hand, and 20 returns dropped from each."""
        d = planted["dates"]
        returns = {
            "a": _dated(planted["a"][:60], d[:60]),
            "b": _dated(planted["b"][20:], d[20:]),
            "c": _dated(planted["c"][:60], d[:60]),
        }
        out = portfolio.dispatch(
            "optimize_hierarchical_risk_parity",
            {"returns": returns, "alignment": "intersect"},
        )
        by_hand = lib.hierarchical_risk_parity(
            pd.DataFrame({k: planted[k][20:60] for k in ("a", "b", "c")})
        )
        assert out["weights"] == by_hand["weights"]
        assert out["n_observations"] == 40
        report = out["alignment"]
        assert report["n_dates"] == 40
        assert (report["first_date"], report["last_date"]) == (
            str(d[20].date()),
            str(d[59].date()),
        )
        assert {k: v["n_dropped"] for k, v in report["by_series"].items()} == {
            "a": 20,
            "b": 20,
            "c": 20,
        }
        assert any(
            "alignment='intersect' kept the 40 dates" in w for w in out["warnings"]
        )

    def test_dated_series_on_one_calendar_give_the_aligned_answer(
        self, portfolio, planted
    ):
        d = planted["dates"]
        returns = {k: _dated(planted[k], d) for k in ("a", "b", "c")}
        out = portfolio.dispatch(
            "optimize_hierarchical_risk_parity", {"returns": returns}
        )
        by_hand = lib.hierarchical_risk_parity(
            pd.DataFrame({k: planted[k] for k in ("a", "b", "c")})
        )
        assert out["weights"] == by_hand["weights"]
        assert out["alignment"]["alignment"] == "exact"
        assert not any("undated" in w for w in out["warnings"])

    def test_undated_lists_still_work_and_say_alignment_was_not_checked(
        self, portfolio, planted
    ):
        returns = {k: planted[k][:60].tolist() for k in ("a", "b", "c")}
        out = portfolio.dispatch(
            "optimize_hierarchical_risk_parity", {"returns": returns}
        )
        assert out["alignment"] is None
        assert any("could not be checked" in w for w in out["warnings"])

    @pytest.mark.parametrize(
        "returns, alignment, says",
        [
            (
                {"a": {"2024-01-02": 0.01, "2024-01-03": 0.0}, "b": [0.01, 0.02]},
                "exact",
                "mixes date-keyed maps",
            ),
            ({"a": [0.01, 0.0], "b": [0.01, 0.02]}, "intersect", "needs dated"),
            (
                {"a": {"2024-01-02": 0.01}, "b": {"2024-02-02": 0.02}},
                "intersect",
                "share no date",
            ),
            (
                {"a": {"Jan 2": 0.01, "2024-01-03": 0.0}, "b": {"2024-01-03": 0.0}},
                "intersect",
                "not ISO dates",
            ),
        ],
    )
    def test_what_cannot_be_aligned_is_refused(
        self, portfolio, returns, alignment, says
    ):
        with pytest.raises(ValueError, match=says):
            portfolio.dispatch(
                "optimize_hierarchical_risk_parity",
                {"returns": returns, "alignment": alignment},
            )
