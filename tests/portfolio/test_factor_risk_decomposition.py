"""
A portfolio's risk under a factor model, split into the factors and the
names' own risk, for the book and against its benchmark.

`decompose_factor_risk` takes the model itself -- loadings B, factor
covariance F and specific variances D, with Sigma = B F B' + D -- and
splits the volatility and the tracking error by factor and by asset with
Euler contributions. The specific variances are required: the factor
budget that took none reported the factors' share of their OWN variance as
their share of the book's.

THE PLANTED BOOK, by hand. One market factor at variance 0.04; two names
with betas 1.2 and 0.8 and specific variances 0.09 and 0.0625; the book is
50/50 and the benchmark 30/70.

    active weights          a = (0.2, -0.2)
    active exposure         x = 1.2 * 0.2 + 0.8 * (-0.2)          = 0.08
    factor TE^2             0.08^2 * 0.04                        = 0.000256
    specific TE^2           0.2^2 * 0.09 + 0.2^2 * 0.0625        = 0.0061
    TE                      sqrt(0.006356)                       = 0.0797245
    factor share of TE^2    0.000256 / 0.006356                  = 4.03%

and for the book itself: exposure 1.0, factor variance 0.04, specific
variance 0.25 * 0.09 + 0.25 * 0.0625 = 0.038125, volatility
sqrt(0.078125) = 0.279508, factor share 0.512.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.factor_risk import decompose_factor_risk

LOADINGS = {"A": {"MKT": 1.2}, "B": {"MKT": 0.8}}
FACTOR_COV = {"MKT": {"MKT": 0.04}}
SPECIFIC = {"A": 0.09, "B": 0.0625}
WEIGHTS = {"A": 0.5, "B": 0.5}
BENCHMARK = {"A": 0.3, "B": 0.7}

#: The hand arithmetic of the module docstring.
ACTIVE_EXPOSURE = 1.2 * 0.2 + 0.8 * -0.2
FACTOR_TE2 = ACTIVE_EXPOSURE**2 * 0.04
SPECIFIC_TE2 = 0.2**2 * 0.09 + 0.2**2 * 0.0625
TRACKING_ERROR = math.sqrt(FACTOR_TE2 + SPECIFIC_TE2)
FACTOR_SHARE_OF_TE = FACTOR_TE2 / (FACTOR_TE2 + SPECIFIC_TE2)
BOOK_VARIANCE = 1.0**2 * 0.04 + 0.25 * 0.09 + 0.25 * 0.0625


@pytest.fixture(scope="module")
def portfolio():
    return resolve("portfolio")


def _planted(**overrides):
    arguments = {
        "weights": WEIGHTS,
        "factor_loadings": LOADINGS,
        "factor_covariance": FACTOR_COV,
        "specific_variance": SPECIFIC,
        "benchmark_weights": BENCHMARK,
    }
    arguments.update(overrides)
    return arguments


def _book(n_assets=40, n_factors=5, seed=11):
    """A random factor model and a long-short book against a long-only
    benchmark: loadings, a positive definite factor covariance built from
    simulated factor returns, and specific variances."""
    rng = np.random.default_rng(seed)
    assets = [f"S{i:02d}" for i in range(n_assets)]
    factors = [f"F{k}" for k in range(n_factors)]
    loadings = rng.normal(0.0, 0.6, (n_assets, n_factors))
    loadings[:, 0] += 1.0
    factor_returns = rng.normal(0.0, 0.02, (500, n_factors))
    factor_cov = np.cov(factor_returns, rowvar=False) * 252
    specific = rng.uniform(0.01, 0.12, n_assets)
    weights = rng.normal(1.0 / n_assets, 0.04, n_assets)
    benchmark = rng.uniform(0.5, 1.5, n_assets)
    benchmark /= benchmark.sum()
    return {
        "weights": dict(zip(assets, weights)),
        "factor_loadings": {
            a: dict(zip(factors, loadings[i])) for i, a in enumerate(assets)
        },
        "factor_covariance": {
            f: dict(zip(factors, factor_cov[k])) for k, f in enumerate(factors)
        },
        "specific_variance": dict(zip(assets, specific)),
        "benchmark_weights": dict(zip(assets, benchmark)),
    }


def _implied_covariance(arguments):
    """Sigma = B F B' + D, assembled by the test from the inputs."""
    assets = list(arguments["factor_loadings"])
    factors = list(arguments["factor_covariance"])
    B = np.array(
        [[arguments["factor_loadings"][a][f] for f in factors] for a in assets]
    )
    F = np.array(
        [[arguments["factor_covariance"][r][c] for c in factors] for r in factors]
    )
    D = np.diag([arguments["specific_variance"][a] for a in assets])
    return assets, B @ F @ B.T + D


def _library(arguments):
    return decompose_factor_risk(
        arguments["weights"],
        arguments["factor_loadings"],
        arguments["factor_covariance"],
        arguments["specific_variance"],
        benchmark_weights=arguments.get("benchmark_weights"),
    )


class TestThePlantedBook:
    def test_the_tracking_error_is_0_0797_and_the_factors_four_percent_of_it(
        self, portfolio
    ):
        """The planted book through the tool: every term of the hand
        arithmetic in the module docstring, to 1e-12."""
        out = portfolio.dispatch("decompose_factor_risk", _planted())
        assert out["tracking_error"] == pytest.approx(TRACKING_ERROR, rel=1e-12)
        assert round(out["tracking_error"], 4) == 0.0797
        active = out["active"]
        assert active["exposures"]["MKT"] == pytest.approx(0.08, rel=1e-12)
        assert active["factor_variance"] == pytest.approx(0.000256, rel=1e-12)
        assert active["specific_variance"] == pytest.approx(0.0061, rel=1e-12)
        assert active["factor_share"] == pytest.approx(FACTOR_SHARE_OF_TE, rel=1e-12)
        assert round(active["factor_share"], 3) == 0.040
        assert active["specific_share"] == pytest.approx(
            1 - FACTOR_SHARE_OF_TE, rel=1e-12
        )

    def test_the_book_is_half_market_and_half_its_names(self, portfolio):
        """The book's own split, and the benchmark's exposure 0.92 and
        volatility, from the same arithmetic."""
        out = portfolio.dispatch("decompose_factor_risk", _planted())
        book = out["portfolio"]
        assert out["volatility"] == pytest.approx(math.sqrt(BOOK_VARIANCE), rel=1e-12)
        assert book["exposures"] == {"MKT": pytest.approx(1.0, rel=1e-12)}
        assert book["factor_variance"] == pytest.approx(0.04, rel=1e-12)
        assert book["specific_variance"] == pytest.approx(0.038125, rel=1e-12)
        assert book["factor_share"] == pytest.approx(0.04 / BOOK_VARIANCE, rel=1e-12)
        assert out["benchmark_exposures"] == {"MKT": pytest.approx(0.92, rel=1e-12)}
        assert out["benchmark_volatility"] == pytest.approx(
            math.sqrt(0.92**2 * 0.04 + 0.09 * 0.09 + 0.49 * 0.0625), rel=1e-12
        )

    def test_a_factors_contribution_is_exposure_times_volatility_times_correlation(
        self,
    ):
        """The x-sigma-rho form, against a correlation the test computes
        itself: Cov(f, w'r) = (F B' w), so rho = (F B' w) / (sigma_f sigma_p)."""
        out = _library(_planted())
        for side, vector in (("portfolio", (0.5, 0.5)), ("active", (0.2, -0.2))):
            (row,) = out[side]["by_factor"]
            sigma = out[side]["volatility"]
            covariance = 0.04 * (1.2 * vector[0] + 0.8 * vector[1])
            assert row["factor_volatility"] == pytest.approx(0.2, rel=1e-15)
            assert row["correlation"] == pytest.approx(
                covariance / (0.2 * sigma), rel=1e-12
            )
            assert row["contribution"] == pytest.approx(
                row["exposure"] * row["factor_volatility"] * row["correlation"],
                rel=1e-14,
            )

    def test_the_factor_specific_cross_term_is_stated_as_zero(self, portfolio):
        """Zero by the model's construction, and the variance is the factor
        part plus the specific part with nothing between them."""
        out = portfolio.dispatch("decompose_factor_risk", _planted())
        for side in ("portfolio", "active"):
            part = out[side]
            assert part["factor_specific_covariance"] == 0.0
            assert part["variance"] == pytest.approx(
                part["factor_variance"] + part["specific_variance"], rel=1e-15
            )


class TestContributionsSumToTheVolatility:
    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_by_asset_and_by_factor_to_1e_12(self, seed):
        """Euler on random books: the asset contributions, the factor
        contributions plus the specific one, and each asset's factor and
        specific parts all sum to the volatility (or tracking error)."""
        arguments = _book(seed=seed)
        out = _library(arguments)
        for side in ("portfolio", "active"):
            part = out[side]
            sigma = part["volatility"]
            assert sigma > 0
            by_asset = math.fsum(r["contribution"] for r in part["by_asset"])
            assert abs(by_asset - sigma) <= 1e-12
            assert abs(part["sum_of_contributions"] - sigma) <= 1e-12
            by_factor = math.fsum(r["contribution"] for r in part["by_factor"])
            assert abs(by_factor + part["specific_contribution"] - sigma) <= 1e-12
            split = math.fsum(
                r["factor_contribution"] + r["specific_contribution"]
                for r in part["by_asset"]
            )
            assert abs(split - sigma) <= 1e-12
        assert out["tracking_error"] == out["active"]["volatility"]

    def test_the_volatility_is_the_implied_covariance_s(self):
        """The volatility and the tracking error are sqrt(w' Sigma w) on the
        covariance the test assembles from B, F and D."""
        arguments = _book()
        assets, sigma = _implied_covariance(arguments)
        w = np.array([arguments["weights"][a] for a in assets])
        b = np.array([arguments["benchmark_weights"][a] for a in assets])
        out = _library(arguments)
        assert out["volatility"] == pytest.approx(math.sqrt(w @ sigma @ w), rel=1e-12)
        assert out["tracking_error"] == pytest.approx(
            math.sqrt((w - b) @ sigma @ (w - b)), rel=1e-12
        )


class TestTheBenchmarkItself:
    def test_weights_equal_to_the_benchmark_have_zero_tracking_error(self, portfolio):
        """w = b: no active exposure and no active specific risk, so the
        tracking error is exactly zero and every number that divides by it
        is null with the reason, not a 0/0 reported as measured."""
        out = portfolio.dispatch(
            "decompose_factor_risk", _planted(benchmark_weights=WEIGHTS)
        )
        assert out["tracking_error"] == 0.0
        active = out["active"]
        assert active["factor_variance"] == 0.0 and active["specific_variance"] == 0.0
        assert active["factor_share"] is None
        assert all(r["contribution"] is None for r in active["by_asset"])
        assert all(r["correlation"] is None for r in active["by_factor"])
        assert any("tracking error is zero" in w for w in out["warnings"])
        assert any(w.startswith("active.factor_share is null") for w in out["warnings"])
        # The book's own decomposition is untouched by the benchmark.
        assert out["volatility"] == pytest.approx(math.sqrt(BOOK_VARIANCE), rel=1e-12)

    def test_the_same_holds_on_a_wide_book(self):
        """Forty names and five factors: w = b is a tracking error of 0.0."""
        arguments = _book()
        arguments["benchmark_weights"] = dict(arguments["weights"])
        assert _library(arguments)["tracking_error"] == 0.0

    def test_without_a_benchmark_there_is_no_active_side(self, portfolio):
        """No benchmark: no active side, and the null tracking error says why."""
        arguments = _planted()
        del arguments["benchmark_weights"]
        out = portfolio.dispatch("decompose_factor_risk", arguments)
        assert out["active"] is None and out["tracking_error"] is None
        assert any(w.startswith("tracking_error is null") for w in out["warnings"])


class TestAModelWithoutSpecificRisk:
    def test_one_factor_and_zero_specific_variance_is_all_factor(self, portfolio):
        """D = 0 leaves nothing but the factor: its share is exactly 1, of
        the book and of the tracking error."""
        out = portfolio.dispatch(
            "decompose_factor_risk",
            _planted(specific_variance={"A": 0.0, "B": 0.0}),
        )
        for side in ("portfolio", "active"):
            assert out[side]["factor_share"] == 1.0
            assert out[side]["specific_share"] == 0.0
            assert out[side]["specific_contribution"] == 0.0
        # One factor: the book's volatility is |x| sigma_f exactly.
        assert out["volatility"] == pytest.approx(0.2, rel=1e-15)
        assert out["tracking_error"] == pytest.approx(0.08 * 0.2, rel=1e-12)

    def test_the_specific_variance_cannot_be_left_out(self):
        """Required at both doors: the tool's schema and the library."""
        arguments = _planted()
        del arguments["specific_variance"]
        with pytest.raises(pydantic.ValidationError, match="specific_variance"):
            resolve("portfolio").dispatch("decompose_factor_risk", arguments)
        with pytest.raises(ValidationError, match="specific_variance is required"):
            decompose_factor_risk(WEIGHTS, LOADINGS, FACTOR_COV, None)


class TestAgreesWithTheMarginalRiskContribution:
    def test_every_marginal_and_contribution_on_the_implied_covariance(self, portfolio):
        """get_marginal_risk_contribution on Sigma = B F B' + D, assembled by
        the test, gives the same volatility, marginals and contributions."""
        arguments = _book(n_assets=25, n_factors=4, seed=21)
        del arguments["benchmark_weights"]
        assets, sigma = _implied_covariance(arguments)
        mrc = portfolio.dispatch(
            "get_marginal_risk_contribution",
            {
                "weights": arguments["weights"],
                "assets": assets,
                "covariance": sigma.tolist(),
            },
        )
        out = portfolio.dispatch("decompose_factor_risk", arguments)
        assert out["volatility"] == pytest.approx(
            mrc["portfolio_volatility"], rel=1e-12
        )
        theirs = {r["asset"]: r for r in mrc["by_asset"]}
        mine = {r["asset"]: r for r in out["portfolio"]["by_asset"]}
        assert set(mine) == set(theirs)
        for asset, row in mine.items():
            assert row["marginal"] == pytest.approx(
                theirs[asset]["marginal_risk"], rel=1e-11, abs=1e-14
            )
            assert row["contribution"] == pytest.approx(
                theirs[asset]["risk_contribution"], rel=1e-11, abs=1e-14
            )
            assert row["share"] == pytest.approx(
                theirs[asset]["risk_share"], rel=1e-11, abs=1e-14
            )


class TestNamesAreCheckedAgainstEachOther:
    @pytest.mark.parametrize(
        "overrides, phrase",
        [
            (
                {"factor_loadings": {"A": {"MKT": 1.2}, "B": {"SIZE": 0.8}}},
                "which factor_covariance does not have",
            ),
            (
                {"factor_loadings": {"A": {"MKT": 1.2}, "B": {}}},
                "asset/factor loading(s) are missing: ['B/MKT']",
            ),
            ({"specific_variance": {"A": 0.09}}, "no entry for ['B']"),
            (
                {"specific_variance": {"A": 0.09, "B": 0.0625, "C": 0.1}},
                "entries for ['C'], which have no loadings",
            ),
            (
                {"weights": {"A": 0.5, "B": 0.4, "C": 0.1}},
                "weights name(s) ['C'], which have no factor loadings",
            ),
            (
                {"benchmark_weights": {"A": 0.5, "Z": 0.5}},
                "benchmark_weights name(s) ['Z'], which have no factor loadings",
            ),
            (
                {
                    "factor_covariance": {
                        "MKT": {"MKT": 0.04, "SIZE": 0.01},
                        "SIZE": {"SIZE": 0.02},
                    },
                    "factor_loadings": {
                        "A": {"MKT": 1.2, "SIZE": 0.1},
                        "B": {"MKT": 0.8, "SIZE": -0.1},
                    },
                },
                "factor_covariance['SIZE'] lacks ['MKT']",
            ),
            (
                {"specific_variance": {"A": 0.09, "B": -0.01}},
                "finite, non-negative variance",
            ),
            (
                {"factor_covariance": {"MKT": {"MKT": 0.0}}},
                "a variance of zero",
            ),
        ],
    )
    def test_a_disagreement_is_refused_by_name(self, portfolio, overrides, phrase):
        """Each way the model's names can disagree is refused, naming the
        assets or factors, rather than read as zero or left out."""
        with pytest.raises(ValidationError) as refusal:
            portfolio.dispatch("decompose_factor_risk", _planted(**overrides))
        assert phrase in str(refusal.value)

    def test_an_asymmetric_factor_covariance_is_refused(self, portfolio):
        """A covariance is symmetric; one that is not was not built as one."""
        cov = {
            "MKT": {"MKT": 0.04, "SIZE": 0.01},
            "SIZE": {"MKT": 0.02, "SIZE": 0.02},
        }
        loadings = {"A": {"MKT": 1.2, "SIZE": 0.1}, "B": {"MKT": 0.8, "SIZE": 0.0}}
        with pytest.raises(ValidationError, match="not symmetric"):
            portfolio.dispatch(
                "decompose_factor_risk",
                _planted(factor_covariance=cov, factor_loadings=loadings),
            )

    def test_the_covariance_is_read_by_name_not_by_position(self):
        """The same matrix with its rows and columns listed in another order
        gives the same answer: names, not positions, place each entry."""
        cov = {
            "MKT": {"MKT": 0.04, "SIZE": 0.006},
            "SIZE": {"MKT": 0.006, "SIZE": 0.02},
        }
        swapped = {
            "SIZE": {"SIZE": 0.02, "MKT": 0.006},
            "MKT": {"SIZE": 0.006, "MKT": 0.04},
        }
        loadings = {"A": {"MKT": 1.2, "SIZE": 0.4}, "B": {"SIZE": -0.3, "MKT": 0.8}}
        first = _library(_planted(factor_covariance=cov, factor_loadings=loadings))
        second = _library(_planted(factor_covariance=swapped, factor_loadings=loadings))
        assert first["volatility"] == pytest.approx(second["volatility"], rel=1e-15)
        assert first["tracking_error"] == pytest.approx(
            second["tracking_error"], rel=1e-15
        )

    def test_a_non_finite_input_is_refused_at_the_door(self, portfolio):
        """A NaN variance is refused by the schema before it reaches a sum."""
        with pytest.raises(pydantic.ValidationError, match="not finite"):
            portfolio.dispatch(
                "decompose_factor_risk",
                _planted(specific_variance={"A": float("nan"), "B": 0.0625}),
            )

    @pytest.mark.parametrize(
        "overrides, phrase",
        [
            ({"weights": {}}, "weights is empty"),
            ({"benchmark_weights": {}}, "benchmark_weights is empty"),
            (
                {
                    "factor_covariance": {
                        f"F{i}": {f"F{j}": float(i == j) for j in range(201)}
                        for i in range(201)
                    }
                },
                "factor_covariance has 201 entries, more than the 200",
            ),
        ],
    )
    def test_the_size_bounds_are_held_at_the_door(self, portfolio, overrides, phrase):
        """The bounds live in the validators, not the advertised schema, and
        are held there: an empty map names nothing, and past the bound the
        payload is refused rather than decomposed."""
        with pytest.raises(pydantic.ValidationError, match=phrase):
            portfolio.dispatch("decompose_factor_risk", _planted(**overrides))


class TestTheSameInputGivesTheSameBits:
    def test_two_calls_agree_to_the_bit(self):
        """The same model and book give the same bits on a second call."""
        arguments = _book(n_assets=120, n_factors=8, seed=31)
        first, second = _library(arguments), _library(arguments)
        assert first == second


def test_a_frame_input_reads_as_the_maps_do():
    """A direct caller may hand the library the model as frames: the
    loadings asset x factor, the covariance named on both axes."""
    loadings = pd.DataFrame(LOADINGS).T
    cov = pd.DataFrame(FACTOR_COV)
    out = decompose_factor_risk(
        WEIGHTS, loadings, cov, pd.Series(SPECIFIC), benchmark_weights=BENCHMARK
    )
    assert out["tracking_error"] == pytest.approx(TRACKING_ERROR, rel=1e-12)
