"""
A covariance is a reference kind, so `estimate_covariance`'s matrix reaches
the allocators without being transcribed.

The matrix came back inline only, and the four tools that take a covariance
read it inline only: an agent passed a table of N x N numbers through its
context to move it one tool along, and nothing travelled with it to say
whether it was annualized or what estimated it. `estimate_covariance` now
publishes it given a `run_id`, as a `covariance` reference holding the
assets, the matrix, the annualization, the periods a year, the method and the
observations; publishing checks it is square under unique names, finite and
symmetric. `optimize_risk_parity`, `optimize_max_diversification`,
`get_marginal_risk_contribution` and `run_portfolio_scenarios` take it as
`covariance_ref`, with the inline answer to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.agent.runtimes.portfolio import tools as portfolio_tools
from standard_quant_tools.error import ValidationError

ASSETS = ["AAA", "BBB", "CCC"]


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


@pytest.fixture(scope="module")
def portfolio():
    return resolve("portfolio")


@pytest.fixture
def returns(monkeypatch):
    """Planted returns behind estimate_covariance's fetch: three names, two
    sharing a factor."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2024-01-01", periods=250)
    common = rng.normal(0, 0.01, 250)
    frame = pd.DataFrame(
        {
            "AAA": common + rng.normal(0, 0.006, 250),
            "BBB": common + rng.normal(0, 0.009, 250),
            "CCC": rng.normal(0, 0.015, 250),
        },
        index=index,
    )
    monkeypatch.setattr(portfolio_tools, "fetch_returns_sync", lambda *a, **k: frame)
    return frame


_WINDOW = {"tickers": ASSETS, "start_date": "2024-01-01", "end_date": "2024-12-31"}


def _doors(assets):
    weights = {a: w for a, w in zip(assets, (0.5, 0.3, 0.2))}
    return {
        "optimize_risk_parity": {"budget": [0.5, 0.3, 0.2]},
        "optimize_max_diversification": {},
        "get_marginal_risk_contribution": {"weights": weights},
        "run_portfolio_scenarios": {
            "weights": weights,
            "scenarios": {"selloff": {a: -0.1 for a in assets}},
        },
    }


def _payload(**overrides):
    payload = {
        "assets": ["A", "B"],
        "matrix": [[0.04, 0.01], [0.01, 0.09]],
        "annualized": True,
        "periods_per_year": 252,
        "method": "sample",
        "n_observations": 250,
    }
    payload.update(overrides)
    return payload


class TestEstimateCovariancePublishes:
    def test_without_a_run_id_the_output_is_what_it_was(
        self, portfolio, returns, runs_dir
    ):
        """Nothing is published and every field the result had is unchanged;
        the two new fields are null."""
        out = portfolio.dispatch("estimate_covariance", _WINDOW)
        assert out["covariance_ref"] is None and out["streams"] is None
        from standard_quant_tools.portfolio.covariance import estimate_covariance

        library = estimate_covariance(returns)
        assert {k: out[k] for k in library} == library
        assert not runs_dir.exists() or not any(runs_dir.rglob("*.parquet"))

    def test_with_a_run_id_the_reference_resolves_to_the_matrix_returned(
        self, portfolio, returns, runs_dir
    ):
        out = portfolio.dispatch("estimate_covariance", {**_WINDOW, "run_id": "cov1"})
        assert out["covariance_ref"] == "sqt://covariance/cov1/covariance"
        stored = handoff.resolve(out["covariance_ref"], expect="covariance")
        assert list(stored.columns) == out["assets"] == list(stored.index)
        assert stored.to_numpy().tolist() == [
            [out["matrix"][r][c] for c in out["assets"]] for r in out["assets"]
        ]
        assert stored.attrs == {
            "annualized": True,
            "periods_per_year": 252,
            "method": "ledoit_wolf",
            "n_observations": 250,
        }
        assert handoff.describe(out["covariance_ref"])["kind"] == "covariance"

    def test_publishing_twice_at_one_run_id_is_refused(
        self, portfolio, returns, runs_dir
    ):
        portfolio.dispatch("estimate_covariance", {**_WINDOW, "run_id": "twice"})
        with pytest.raises(ValidationError, match="already published"):
            portfolio.dispatch("estimate_covariance", {**_WINDOW, "run_id": "twice"})

    def test_it_is_not_advertised_as_safe_to_repeat(self):
        from standard_quant_tools.mcp.catalog import build_catalog

        assert build_catalog()["estimate_covariance"].idempotent is False


class TestTheFourDoorsTakeTheReference:
    def test_every_door_gives_the_inline_matrixs_answer_to_the_bit(
        self, portfolio, returns, runs_dir
    ):
        """
        The sweep: estimate_covariance's matrix inline (assets and map, as
        it came back) and the reference it published give each of the four
        tools the same result -- with `assets` naming the order and without.
        """
        out = portfolio.dispatch("estimate_covariance", {**_WINDOW, "run_id": "sweep"})
        for tool, extra in _doors(out["assets"]).items():
            inline = portfolio.dispatch(
                tool, {"assets": out["assets"], "covariance": out["matrix"], **extra}
            )
            by_ref = portfolio.dispatch(
                tool, {"covariance_ref": out["covariance_ref"], **extra}
            )
            named = portfolio.dispatch(
                tool,
                {
                    "covariance_ref": out["covariance_ref"],
                    "assets": out["assets"],
                    **extra,
                },
            )
            assert by_ref == inline, tool
            assert named == inline, tool

    def test_assets_given_with_the_reference_fix_the_order(
        self, portfolio, returns, runs_dir
    ):
        """Read by name: the reversed order gives the same weight to each
        name as the reference's own order -- to the solver's tolerance, since
        a different order is a different sweep that stops within 1e-10 of
        equal risk shares."""
        out = portfolio.dispatch("estimate_covariance", {**_WINDOW, "run_id": "order"})
        own = portfolio.dispatch(
            "optimize_risk_parity", {"covariance_ref": out["covariance_ref"]}
        )["weights"]
        reversed_ = portfolio.dispatch(
            "optimize_risk_parity",
            {"covariance_ref": out["covariance_ref"], "assets": ASSETS[::-1]},
        )["weights"]
        assert list(reversed_) == ASSETS[::-1]
        assert reversed_ == pytest.approx(own, abs=1e-9)

    def test_a_reference_of_the_wrong_kind_is_refused_naming_both_kinds(
        self, portfolio, runs_dir, returns
    ):
        panel = handoff.publish(returns, kind="returns_panel", run_id="wrong", name="r")
        for tool, extra in _doors(ASSETS).items():
            with pytest.raises(ValidationError) as refusal:
                portfolio.dispatch(tool, {"covariance_ref": panel, **extra})
            message = str(refusal.value)
            assert "expected a 'covariance' reference" in message, tool
            assert "is a 'returns_panel'" in message, tool

    def test_assets_that_are_not_the_references_are_refused(
        self, portfolio, returns, runs_dir
    ):
        out = portfolio.dispatch("estimate_covariance", {**_WINDOW, "run_id": "names"})
        with pytest.raises(ValidationError) as refusal:
            portfolio.dispatch(
                "optimize_max_diversification",
                {"covariance_ref": out["covariance_ref"], "assets": ["AAA", "ZZZ"]},
            )
        message = str(refusal.value)
        assert "omits ['BBB', 'CCC']" in message
        assert "['ZZZ'] are not in the reference" in message

    @pytest.mark.parametrize(
        "tool", ["optimize_risk_parity", "optimize_max_diversification"]
    )
    def test_exactly_one_form_of_the_matrix(self, portfolio, tool):
        rows = [[0.04, 0.0], [0.0, 0.09]]
        with pytest.raises(pydantic.ValidationError, match="both given"):
            portfolio.dispatch(
                tool,
                {
                    "assets": ["A", "B"],
                    "covariance": rows,
                    "covariance_ref": "sqt://covariance/r/covariance",
                },
            )
        with pytest.raises(pydantic.ValidationError, match="a covariance is required"):
            portfolio.dispatch(tool, {"assets": ["A", "B"]})
        with pytest.raises(pydantic.ValidationError, match="needs `assets`"):
            portfolio.dispatch(tool, {"covariance": rows})

    def test_scenarios_still_run_without_any_covariance(self, portfolio):
        out = portfolio.dispatch(
            "run_portfolio_scenarios",
            {"weights": {"A": 0.5, "B": 0.5}, "scenarios": {"s": {"A": -0.1}}},
        )
        assert out["portfolio_volatility"] is None


class TestTheKindIsCheckedOnPublish:
    def test_a_good_payload_round_trips_with_its_record(self, runs_dir):
        ref = handoff.publish(_payload(), kind="covariance", run_id="ok", name="c")
        stored = handoff.resolve(ref, expect="covariance")
        assert stored.to_numpy().tolist() == [[0.04, 0.01], [0.01, 0.09]]
        assert stored.attrs["method"] == "sample"

    def test_the_map_estimate_covariance_returns_is_read_by_name(self, runs_dir):
        matrix = {"B": {"A": 0.01, "B": 0.09}, "A": {"B": 0.01, "A": 0.04}}
        ref = handoff.publish(
            _payload(matrix=matrix), kind="covariance", run_id="map", name="c"
        )
        stored = handoff.resolve(ref, expect="covariance")
        assert stored.loc["A", "A"] == 0.04 and stored.loc["B", "B"] == 0.09

    def test_asymmetry_within_the_tolerance_is_kept_bit_for_bit(self, runs_dir):
        """Rounding-level asymmetry, 1e-18 on entries of 0.01, is inside
        the stated tolerance of 1e-10 of the largest entry, and the matrix
        is stored as given rather than symmetrized."""
        rows = [[0.04, 0.01], [0.01 + 1e-18, 0.09]]
        ref = handoff.publish(
            _payload(matrix=rows), kind="covariance", run_id="tol", name="c"
        )
        assert handoff.resolve(ref, expect="covariance").to_numpy().tolist() == rows

    @pytest.mark.parametrize(
        "overrides, says",
        [
            ({"matrix": [[0.04, 0.01], [0.02, 0.09]]}, "not symmetric"),
            ({"matrix": [[0.04, 0.01], [0.01, float("nan")]]}, "not finite"),
            ({"matrix": [[0.04, 0.01], [0.01, float("inf")]]}, "not finite"),
            ({"matrix": [[0.04, 0.01]]}, "not 2 x 2"),
            ({"matrix": [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0]]}, "not 2 x 2"),
            ({"assets": ["A", "A"]}, "more than once"),
            ({"assets": []}, "non-empty list"),
            ({"matrix": {"A": {"A": 0.04, "B": 0.01}}}, "no row for"),
            ({"annualized": "yes"}, "annualized is true or false"),
            ({"periods_per_year": 0}, "periods_per_year"),
            ({"method": ""}, "method names the estimator"),
            ({"n_observations": 1}, "at least 2"),
            ({"extra": 1}, "unexpected"),
        ],
    )
    def test_what_is_not_a_covariance_is_refused_by_name(
        self, runs_dir, overrides, says
    ):
        with pytest.raises(ValidationError, match=says):
            handoff.publish(
                _payload(**overrides), kind="covariance", run_id="bad", name="c"
            )

    def test_a_missing_field_is_named(self, runs_dir):
        payload = _payload()
        del payload["method"]
        with pytest.raises(ValidationError, match=r"missing \['method'\]"):
            handoff.publish(payload, kind="covariance", run_id="bad", name="c")

    def test_the_kind_is_listed_with_what_it_holds(self):
        kinds = resolve("meta").dispatch("list_reference_kinds", {})["kinds"]
        (row,) = [k for k in kinds if k["kind"] == "covariance"]
        assert "annualized" in row["description"]
