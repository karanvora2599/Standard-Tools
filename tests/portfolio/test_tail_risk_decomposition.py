"""
Which positions a portfolio's tail comes from, scenario by scenario.

`decompose_tail_risk` takes a scenario matrix (or a published
`returns_panel`) and weights, and splits the expected shortfall by asset
with Euler's theorem -- exact, since ES over a fixed tail set is linear in
the weights -- and the VaR with a Gaussian kernel around the quantile
(Epperlein and Smillie 2006), whose components sum to the kernel-smoothed
VaR reported beside the empirical one.

THE PLANTED MATRIX. Twenty scenarios of three assets at 90%: numpy's
linear percentile sits 1.9 places into the sorted P&L, so the tail is the
two worst scenarios and ES is minus their mean -- and each asset's
component is minus its weight times its own mean in those two scenarios.

UNDER A NORMAL with zero mean and covariance Sigma, component ES_i tends
to w_i (Sigma w)_i / sigma_p x phi(z) / (1 - c) and component VaR_i to
w_i (Sigma w)_i / sigma_p x z, as the scenarios grow.

VaR and ES are losses, positive, and on the same P&L they are
`var_historical`'s and `cvar`'s own bits.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest
from scipy import stats

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import (
    cvar,
    cvar_tail_observations,
    var_historical,
)
from standard_quant_tools.portfolio.tail_risk import (
    decompose_tail_risk,
    silverman_bandwidth,
)

#: The planted matrix: the worst scenario is 3 and the second worst 11.
PLANTED = pd.DataFrame(
    {
        "A": [0.010, -0.004, 0.006, -0.050, 0.002, 0.008, -0.010, 0.004, 0.000,
              0.003, 0.005, -0.030, 0.007, -0.002, 0.001, 0.009, -0.006, 0.002,
              0.004, 0.006],
        "B": [0.002, 0.003, -0.001, -0.020, 0.004, -0.002, 0.001, 0.000, 0.003,
              -0.003, 0.002, -0.040, 0.001, 0.002, -0.001, 0.000, 0.002, 0.001,
              -0.002, 0.003],
        "C": [-0.001, 0.002, 0.000, 0.030, -0.002, 0.001, 0.000, -0.001, 0.002,
              0.000, -0.001, 0.010, 0.001, 0.000, 0.002, -0.002, 0.001, 0.000,
              0.001, -0.001],
    }
)  # fmt: skip
PLANTED_WEIGHTS = {"A": 0.5, "B": 0.4, "C": 0.1}


@pytest.fixture(scope="module")
def portfolio():
    return resolve("portfolio")


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _rows(out):
    return {r["asset"]: r for r in out["by_asset"]}


def _pnl(frame, weights):
    """The portfolio's P&L as the tool sums it: asset by asset, in the
    weights' order -- so the same bits, which `cvar` is then held to."""
    return sum(w * frame[asset].to_numpy(dtype=float) for asset, w in weights.items())


def _fat_tailed(n=3000, seed=5):
    """Five assets with Student-t shocks and a common factor: a tail that
    is not the normal one, and a short in the book."""
    rng = np.random.default_rng(seed)
    common = rng.standard_t(4, n) * 0.01
    values = {
        name: beta * common + rng.standard_t(3, n) * vol
        for name, beta, vol in (
            ("A", 1.0, 0.008),
            ("B", 0.7, 0.012),
            ("C", 1.3, 0.006),
            ("D", -0.4, 0.010),
            ("E", 0.2, 0.015),
        )
    }
    weights = {"A": 0.35, "B": 0.25, "C": -0.15, "D": 0.30, "E": 0.25}
    return pd.DataFrame(values), weights


class TestThePlantedMatrix:
    def test_the_tail_is_the_two_worst_scenarios(self):
        """P = 0.5 A + 0.4 B + 0.1 C. The worst is scenario 3 (P = -0.030),
        then 11 (P = -0.030 also: -0.015 - 0.016 + 0.001). The 10th
        percentile of twenty values sits 1.9 places in, between the second
        and third worst, so the tail is those two."""
        P = _pnl(PLANTED, PLANTED_WEIGHTS)
        worst = np.argsort(P, kind="stable")[:2]
        assert sorted(worst.tolist()) == [3, 11]
        out = decompose_tail_risk(PLANTED, PLANTED_WEIGHTS, confidence=0.90)
        assert out["n_tail_observations"] == 2
        tail = PLANTED.iloc[[3, 11]]
        expected_es = -float(np.mean(P[[3, 11]]))
        assert out["es"] == pytest.approx(expected_es, rel=1e-14)
        rows = _rows(out)
        for asset, weight in PLANTED_WEIGHTS.items():
            own_mean = float(tail[asset].mean())
            assert rows[asset]["marginal_es"] == pytest.approx(-own_mean, rel=1e-14)
            assert rows[asset]["component_es"] == pytest.approx(
                -weight * own_mean, rel=1e-14
            )
        # The C leg gains in both tail scenarios: a hedge.
        assert rows["C"]["component_es"] < 0
        assert out["n_hedges"] == 1

    def test_var_and_es_are_the_librarys_own_bits(self):
        """On the same P&L, VaR, ES and the tail count are
        var_historical's, cvar's and cvar_tail_observations' own."""
        P = pd.Series(_pnl(PLANTED, PLANTED_WEIGHTS))
        out = decompose_tail_risk(PLANTED, PLANTED_WEIGHTS, confidence=0.90)
        assert out["var"] == var_historical(P, 0.90)
        assert out["es"] == cvar(P, 0.90)
        assert out["n_tail_observations"] == cvar_tail_observations(P, 0.90)


class TestComponentESIsExact:
    @pytest.mark.parametrize("confidence", [0.9, 0.95, 0.975, 0.99])
    def test_the_components_sum_to_the_es(self, confidence):
        """Euler on a fat-tailed book with a short, at four confidences:
        the components sum to the ES to 1e-12, the shares to 1."""
        frame, weights = _fat_tailed()
        out = decompose_tail_risk(frame, weights, confidence=confidence)
        components = math.fsum(r["component_es"] for r in out["by_asset"])
        assert abs(components - out["es"]) <= 1e-12 * abs(out["es"])
        assert out["sum_of_es_components"] == components
        shares = math.fsum(r["es_share"] for r in out["by_asset"])
        assert shares == pytest.approx(1.0, abs=1e-12)
        P = pd.Series(_pnl(frame, weights))
        assert out["es"] == cvar(P, confidence)
        assert out["var"] == var_historical(P, confidence)

    def test_component_var_sums_to_the_kernel_var_it_reports(self):
        """The kernel components add up, exactly, to the kernel-weighted
        mean loss around the quantile -- computed here by the test."""
        frame, weights = _fat_tailed()
        out = decompose_tail_risk(frame, weights, confidence=0.95)
        P = _pnl(frame, weights)
        q = np.percentile(P, 5.0)
        h = out["bandwidth"]
        kernel = np.exp(-0.5 * ((P - q) / h) ** 2)
        expected = -float(np.sum(kernel * P) / np.sum(kernel))
        assert out["var_kernel"] == pytest.approx(expected, rel=1e-12)
        components = math.fsum(r["component_var"] for r in out["by_asset"])
        assert abs(components - out["var_kernel"]) <= 1e-12 * abs(out["var_kernel"])
        assert math.fsum(r["var_share"] for r in out["by_asset"]) == pytest.approx(
            1.0, abs=1e-12
        )
        assert out["kernel_effective_scenarios"] == pytest.approx(
            float(np.sum(kernel)) ** 2 / float(np.sum(kernel**2)), rel=1e-12
        )


class TestTheGaussianClosedForm:
    COV = np.array(
        [
            [0.00040, 0.00012, -0.00008, 0.00010],
            [0.00012, 0.00090, 0.00010, 0.00015],
            [-0.00008, 0.00010, 0.000225, -0.00005],
            [0.00010, 0.00015, -0.00005, 0.00160],
        ]
    )
    WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1])
    NAMES = ["A", "B", "C", "D"]

    def _errors(self, n, confidence):
        rng = np.random.default_rng(7)
        root = np.linalg.cholesky(self.COV)
        X = rng.standard_normal((n, 4)) @ root.T
        out = decompose_tail_risk(
            pd.DataFrame(X, columns=self.NAMES),
            dict(zip(self.NAMES, self.WEIGHTS)),
            confidence=confidence,
        )
        w = self.WEIGHTS
        sigma = math.sqrt(w @ self.COV @ w)
        beta = w * (self.COV @ w) / sigma
        z = stats.norm.ppf(confidence)
        es_multiplier = stats.norm.pdf(z) / (1 - confidence)
        rows = _rows(out)
        es = np.array([rows[a]["component_es"] for a in self.NAMES])
        var = np.array([rows[a]["component_var"] for a in self.NAMES])
        return (
            float(np.max(np.abs(es - beta * es_multiplier)) / (sigma * es_multiplier)),
            float(np.max(np.abs(var - beta * z)) / (sigma * z)),
        )

    @pytest.mark.parametrize("confidence", [0.95, 0.99])
    def test_components_converge_to_the_closed_form(self, confidence):
        """The largest error of any component, as a fraction of the closed
        form's total, falls as the scenarios grow from 2,000 to 2,000,000,
        and at 2,000,000 is a few tenths of a percent."""
        es_small, var_small = self._errors(2_000, confidence)
        es_large, var_large = self._errors(2_000_000, confidence)
        assert es_large < es_small / 3 and var_large < var_small / 3
        assert es_large < 0.005
        assert var_large < 0.01


class TestAHedgeIsNegative:
    def test_a_position_that_gains_in_the_tail_has_a_negative_component(self):
        """H moves against A and B, and the book holds it long: in the
        scenarios where the book loses, H gains, so its component ES and VaR
        are negative and the result names it a hedge."""
        rng = np.random.default_rng(17)
        n = 5000
        market = rng.normal(0.0, 0.01, n)
        frame = pd.DataFrame(
            {
                "A": market + rng.normal(0.0, 0.004, n),
                "B": 0.8 * market + rng.normal(0.0, 0.006, n),
                "H": -0.9 * market + rng.normal(0.0, 0.003, n),
            }
        )
        out = decompose_tail_risk(frame, {"A": 0.5, "B": 0.4, "H": 0.3})
        rows = _rows(out)
        assert rows["H"]["component_es"] < 0 and rows["H"]["marginal_es"] < 0
        assert rows["H"]["component_var"] < 0
        assert rows["A"]["component_es"] > 0 and rows["B"]["component_es"] > 0
        assert out["n_hedges"] == 1
        assert any("NEGATIVE component ES (['H'])" in w for w in out["warnings"])


class TestTheTailCount:
    def test_it_is_reported_and_a_short_tail_is_warned_about(self, portfolio):
        """250 scenarios at 99% leave three: the count is reported and the
        library's shared warning says what it means."""
        frame, weights = _fat_tailed(n=250)
        out = portfolio.dispatch(
            "decompose_tail_risk",
            {
                "weights": weights,
                "scenarios": {c: frame[c].tolist() for c in weights},
                "confidence": 0.99,
            },
        )
        P = pd.Series(_pnl(frame, weights))
        assert out["n_tail_observations"] == cvar_tail_observations(P, 0.99) == 3
        assert any(
            w.startswith("es is the mean of the worst 3 of 250 returns at 99%")
            for w in out["warnings"]
        )

    def test_a_long_tail_carries_no_warning(self):
        """3,000 scenarios at 95% leave 150 in the tail, and no warning."""
        frame, weights = _fat_tailed(n=3000)
        out = decompose_tail_risk(frame, weights, confidence=0.95)
        assert out["n_tail_observations"] == 150
        assert not any("mean of the worst" in w for w in out["warnings"])


class TestTheBandwidth:
    def test_the_default_is_silvermans_rule_on_the_portfolio_pnl(self):
        """0.9 min(sd, IQR/1.34) n^(-1/5) on the portfolio's P&L, computed
        by the test, and reported with its rule."""
        frame, weights = _fat_tailed()
        P = _pnl(frame, weights)
        sd = float(np.std(P, ddof=1))
        iqr = float(np.percentile(P, 75) - np.percentile(P, 25))
        expected = 0.9 * min(sd, iqr / 1.34) * len(P) ** (-0.2)
        out = decompose_tail_risk(frame, weights)
        assert out["bandwidth_rule"] == "silverman"
        assert out["bandwidth"] == pytest.approx(expected, rel=1e-12)
        assert silverman_bandwidth(P) == out["bandwidth"]

    def test_a_given_bandwidth_is_used_and_reported(self, portfolio):
        """A caller's bandwidth is the one used and echoed; ES does not
        depend on it, the kernel VaR does."""
        frame, weights = _fat_tailed()
        out = portfolio.dispatch(
            "decompose_tail_risk",
            {
                "weights": weights,
                "scenarios": {c: frame[c].tolist() for c in weights},
                "bandwidth": 0.004,
            },
        )
        assert out["bandwidth"] == 0.004 and out["bandwidth_rule"] == "given"
        default = decompose_tail_risk(frame, weights)
        assert out["var_kernel"] != default["var_kernel"]
        # ES does not depend on the bandwidth.
        assert out["es"] == default["es"]


class TestAReferenceAnswersAsTheInlineMatrixDoes:
    def _panel(self, n=400, seed=29):
        frame, weights = _fat_tailed(n=n, seed=seed)
        frame.index = pd.bdate_range("2022-01-03", periods=n)
        return frame, weights

    def test_returns_ref_gives_the_inline_answer_to_the_bit(self, portfolio, runs_dir):
        """The same scenarios inline and as a published panel give the same
        result, every number and warning, but for the streams report."""
        frame, weights = self._panel()
        ref = handoff.publish(frame, kind="returns_panel", run_id="tail", name="r")
        inline = portfolio.dispatch(
            "decompose_tail_risk",
            {"weights": weights, "scenarios": {c: frame[c].tolist() for c in weights}},
        )
        by_ref = portfolio.dispatch(
            "decompose_tail_risk", {"weights": weights, "returns_ref": ref}
        )
        assert inline["streams"] is None
        assert by_ref["streams"]["n_dates"] == 400
        assert {k: v for k, v in by_ref.items() if k != "streams"} == {
            k: v for k, v in inline.items() if k != "streams"
        }

    def test_blank_cells_drop_their_dates_and_are_reported(self, portfolio, runs_dir):
        """D is blank on its first 30 dates: the scenarios are the 370 dates
        every weighted column has, and the answer is the inline one on those
        rows, to the bit."""
        frame, weights = self._panel()
        gapped = frame.copy()
        gapped.iloc[:30, gapped.columns.get_loc("D")] = np.nan
        ref = handoff.publish(gapped, kind="returns_panel", run_id="gap", name="r")
        by_ref = portfolio.dispatch(
            "decompose_tail_risk", {"weights": weights, "returns_ref": ref}
        )
        kept = frame.iloc[30:]
        inline = portfolio.dispatch(
            "decompose_tail_risk",
            {"weights": weights, "scenarios": {c: kept[c].tolist() for c in weights}},
        )
        assert by_ref["n_scenarios"] == 370
        assert by_ref["streams"]["by_stream"]["A"]["n_dropped_before"] == 30
        assert any("dropped" in w for w in by_ref["warnings"])
        assert by_ref["by_asset"] == inline["by_asset"]
        assert by_ref["es"] == inline["es"] and by_ref["var"] == inline["var"]

    def test_columns_without_a_weight_are_not_read(self, portfolio, runs_dir):
        """A panel column with no weight is not read, so its blanks drop no
        scenario."""
        frame, weights = self._panel()
        frame["UNHELD"] = np.nan  # blank throughout: would drop every date
        ref = handoff.publish(frame, kind="returns_panel", run_id="wide", name="r")
        out = portfolio.dispatch(
            "decompose_tail_risk", {"weights": weights, "returns_ref": ref}
        )
        assert out["n_assets"] == 5 and out["n_scenarios"] == 400
        assert sorted(out["streams"]["by_stream"]) == sorted(weights)

    def test_a_weight_with_no_column_is_refused_by_name(self, portfolio, runs_dir):
        """A weight on an asset the panel does not have is refused, naming it."""
        frame, weights = self._panel()
        ref = handoff.publish(frame, kind="returns_panel", run_id="miss", name="r")
        with pytest.raises(ValidationError, match=r"the panel has no column \['Z'\]"):
            portfolio.dispatch(
                "decompose_tail_risk",
                {"weights": {**weights, "Z": 0.1}, "returns_ref": ref},
            )


class TestRefusals:
    def _arguments(self, **overrides):
        arguments = {
            "weights": PLANTED_WEIGHTS,
            "scenarios": {c: PLANTED[c].tolist() for c in PLANTED},
        }
        arguments.update(overrides)
        return arguments

    @pytest.mark.parametrize(
        "overrides, phrase",
        [
            (
                {"weights": {"A": 0.5, "B": 0.5}},
                "scenarios has ['C'], which have no weight",
            ),
            (
                {"weights": {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.1}},
                "weights name ['D'], which have no scenarios",
            ),
            (
                {
                    "scenarios": {
                        "A": [0.01] * 20,
                        "B": [0.01] * 19,
                        "C": [0.0] * 20,
                    }
                },
                "different lengths",
            ),
            (
                {"scenarios": {c: PLANTED[c].tolist()[:9] for c in PLANTED}},
                "too few for a tail",
            ),
            (
                {"scenarios": {c: [0.01] * 20 for c in PLANTED}},
                "the same in every scenario",
            ),
        ],
    )
    def test_a_bad_matrix_is_refused_by_name(self, portfolio, overrides, phrase):
        """Names that disagree, ragged lists, too few scenarios and a P&L
        with no tail are each refused with the reason."""
        with pytest.raises(ValidationError) as refusal:
            portfolio.dispatch("decompose_tail_risk", self._arguments(**overrides))
        assert phrase in str(refusal.value)

    @pytest.mark.parametrize(
        "overrides, phrase",
        [
            ({"returns_ref": "sqt://returns_panel/x/y"}, "exactly one of"),
            ({"scenarios": None}, "exactly one of"),
            (
                {
                    "scenarios": {
                        "A": [float("nan")] * 20,
                        "B": [0.0] * 20,
                        "C": [0.0] * 20,
                    }
                },
                "not finite",
            ),
            ({"bandwidth": float("inf")}, "finite positive"),
            ({"bandwidth": 0.0}, "greater than 0"),
            ({"confidence": 0.3}, "greater than or equal to 0.5"),
            ({"weights": {}}, "weights is empty"),
        ],
    )
    def test_a_bad_input_is_refused_at_the_door(self, portfolio, overrides, phrase):
        """The schema's own rules: one source, finite values, a finite
        positive bandwidth, a confidence of at least 0.5, a weight."""
        with pytest.raises(pydantic.ValidationError, match=phrase):
            portfolio.dispatch("decompose_tail_risk", self._arguments(**overrides))


class TestTheSameInputGivesTheSameBits:
    def test_two_calls_agree_to_the_bit(self):
        """The same 20,000 scenarios give the same bits on a second call."""
        frame, weights = _fat_tailed(n=20_000, seed=41)
        first = decompose_tail_risk(frame, weights)
        second = decompose_tail_risk(frame, weights)
        assert first == second
