"""
A factor model estimated from return series: the loadings B, the factor
covariance F and the specific variances D that `decompose_factor_risk`
takes, and the covariance Sigma = B F B' + D they imply.

THE PLANTED MODEL is `decompose_factor_risk`'s own planted book: one market
factor at an annual variance of 0.04, two names with betas 1.2 and 0.8 and
annual specific variances 0.09 and 0.0625. The daily series here are built
so their sample moments ARE that model, not merely close to it: the factor
and the two residuals are orthonormal columns orthogonal to the constant,
scaled so the factor's sample variance (n - 1) is 0.04 / 252 and each
residual's sum of squares over n - p is its specific variance / 252. OLS
then returns the planted betas, F and D to rounding, and the
decomposition of the estimated model reproduces the hand arithmetic of
the planted book -- a tracking error of 0.0797245 of which 4.03% is the
factor -- which it cannot do if D divides by n rather than n - p, if F is
not annualized, or if a loading is off.

The other checks are against a closed form or a simulation: the loadings
are (X'X)^-1 X'y and the bits `multi_factor_regression` gives; on repeated
draws from a known B, F and D, the estimates fall within 1.96 of their
standard errors about 95% of the time, and D averages the true variance
where SSR / n would not; and the published covariance is B F B' + D,
symmetric to the last bit and positive semidefinite, and is what the
optimizers read through covariance_ref.
"""

from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.analysis.factor_model import estimate_factor_model
from standard_quant_tools.analysis.multi_factor import multi_factor_regression
from standard_quant_tools.error import ValidationError

N_DAYS = 504
DATES = pd.bdate_range("2023-01-02", periods=N_DAYS)

#: decompose_factor_risk's planted book (tests/portfolio/
#: test_factor_risk_decomposition.py), in annual units.
BETAS = {"A": 1.2, "B": 0.8}
SPECIFIC = {"A": 0.09, "B": 0.0625}
FACTOR_VARIANCE = 0.04
ALPHAS = {"A": 0.0002, "B": -0.0001}
TRACKING_ERROR = math.sqrt(0.08**2 * 0.04 + 0.2**2 * 0.09 + 0.2**2 * 0.0625)
FACTOR_SHARE_OF_TE = 0.08**2 * 0.04 / TRACKING_ERROR**2
BOOK_VOLATILITY = math.sqrt(0.04 + 0.25 * 0.09 + 0.25 * 0.0625)


def _iso(index):
    return [d.strftime("%Y-%m-%d") for d in index]


def _dated(values, index=DATES):
    return dict(zip(_iso(index), (float(v) for v in values)))


def _planted_series():
    """Daily returns whose sample moments are exactly the planted model."""
    rng = np.random.default_rng(42)
    design = np.column_stack([np.ones(N_DAYS), rng.standard_normal((N_DAYS, 3))])
    q, _ = np.linalg.qr(design)
    factor = q[:, 1] * math.sqrt(FACTOR_VARIANCE / 252 * (N_DAYS - 1))
    dof = N_DAYS - 2
    assets = {}
    for name, column in (("A", q[:, 2]), ("B", q[:, 3])):
        residual = column * math.sqrt(SPECIFIC[name] / 252 * dof)
        assets[name] = ALPHAS[name] + BETAS[name] * factor + residual
    return assets, factor


@pytest.fixture(scope="module")
def research():
    return resolve("research")


@pytest.fixture(scope="module")
def portfolio():
    return resolve("portfolio")


def _planted_arguments(**overrides):
    assets, factor = _planted_series()
    arguments = {
        "returns": {name: _dated(values) for name, values in assets.items()},
        "factors": {"MKT": _dated(factor)},
    }
    arguments.update(overrides)
    return arguments


def _run_id():
    return "fm" + uuid.uuid4().hex[:12]


class TestThePlantedModelComesBack:
    def test_the_betas_f_and_d_are_the_planted_ones(self, research):
        """Series built to have the planted model as their sample moments
        give it back: betas 1.2 and 0.8, F = 0.04 and D = 0.09 and 0.0625,
        annual, to rounding. D = SSR / (n - p) is what makes the specific
        variances exact; SSR / n would read 0.09 x 502 / 504."""
        out = research.dispatch("estimate_factor_model", _planted_arguments())
        for name, beta in BETAS.items():
            assert out["factor_loadings"][name]["MKT"] == pytest.approx(beta, abs=1e-12)
            assert out["specific_variance"][name] == pytest.approx(
                SPECIFIC[name], rel=1e-12
            )
            assert out["alpha"][name] == pytest.approx(ALPHAS[name], abs=1e-15)
        assert out["factor_covariance"]["MKT"]["MKT"] == pytest.approx(
            FACTOR_VARIANCE, rel=1e-12
        )
        assert out["n_observations"] == N_DAYS
        assert out["n_coefficients"] == 2
        assert out["degrees_of_freedom"] == N_DAYS - 2
        assert "SSR / 502" in out["specific_variance_basis"]
        # The two residuals are orthogonal by construction.
        assert abs(out["largest_residual_correlation"]) < 1e-12

    def test_decompose_factor_risk_reproduces_the_planted_book(
        self, research, portfolio
    ):
        """The estimated model passed straight to decompose_factor_risk:
        the planted book's tracking error of 0.0797245, 4.03% of it the
        factor, and its volatility of 0.279508 -- the hand arithmetic of
        that tool's own planted test, from estimated inputs."""
        model = research.dispatch("estimate_factor_model", _planted_arguments())
        out = portfolio.dispatch(
            "decompose_factor_risk",
            {
                "weights": {"A": 0.5, "B": 0.5},
                "benchmark_weights": {"A": 0.3, "B": 0.7},
                "factor_loadings": model["factor_loadings"],
                "factor_covariance": model["factor_covariance"],
                "specific_variance": model["specific_variance"],
            },
        )
        assert out["tracking_error"] == pytest.approx(TRACKING_ERROR, rel=1e-10)
        assert round(out["tracking_error"], 4) == 0.0797
        assert out["active"]["factor_share"] == pytest.approx(
            FACTOR_SHARE_OF_TE, rel=1e-9
        )
        assert round(out["active"]["factor_share"], 3) == 0.040
        assert out["volatility"] == pytest.approx(BOOK_VOLATILITY, rel=1e-10)


def _random_panel(n=300, seed=5, intercept=True):
    rng = np.random.default_rng(seed)
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (n, 3)), index=DATES[:n], columns=["F1", "F2", "F3"]
    )
    loadings = rng.normal(0.5, 0.4, (4, 3))
    alpha = rng.normal(0, 0.0005, 4) if intercept else np.zeros(4)
    values = alpha + factors.to_numpy() @ loadings.T + rng.normal(0, 0.012, (n, 4))
    assets = pd.DataFrame(values, index=DATES[:n], columns=["W", "X", "Y", "Z"])
    return assets, factors


class TestTheLoadingsAreOLS:
    @pytest.mark.parametrize("intercept", [True, False])
    def test_they_equal_the_normal_equations(self, intercept):
        """Each asset's coefficients equal (X'X)^-1 X'y, with and without
        the intercept column, to 1e-12."""
        assets, factors = _random_panel()
        out = estimate_factor_model(assets, factors, intercept=intercept)
        F = factors.to_numpy()
        X = np.column_stack([np.ones(len(F)), F]) if intercept else F
        for name in assets.columns:
            beta = np.linalg.solve(X.T @ X, X.T @ assets[name].to_numpy())
            slopes = beta[1:] if intercept else beta
            got = [out["loadings"][name][f] for f in factors.columns]
            np.testing.assert_allclose(got, slopes, rtol=0, atol=1e-12)
            if intercept:
                assert out["alpha"][name] == pytest.approx(beta[0], abs=1e-12)
        if not intercept:
            assert out["alpha"] is None
            assert any("No intercept" in w for w in out["warnings"])

    def test_they_are_multi_factor_regressions_bits(self):
        """The fit is multi_factor_regression's, so an asset's loadings,
        alpha and R-squared are its bits on the same dates."""
        assets, factors = _random_panel()
        out = estimate_factor_model(assets, factors)
        for name in assets.columns:
            single = multi_factor_regression(assets[name], factors)
            assert out["loadings"][name] == single["loadings"]
            assert out["alpha"][name] == single["alpha"]
            assert out["r_squared"][name] == pytest.approx(
                single["r_squared"], rel=1e-14
            )

    def test_the_standard_errors_are_s2_times_the_gram_inverse(self):
        """SE_j = sqrt(s^2 [(X'X)^-1]_jj) with s^2 = SSR / (n - p), from
        the residuals the test computes itself."""
        assets, factors = _random_panel(n=120)
        out = estimate_factor_model(assets, factors)
        X = np.column_stack([np.ones(120), factors.to_numpy()])
        inverse = np.linalg.inv(X.T @ X)
        for name in assets.columns:
            y = assets[name].to_numpy()
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            s2 = float(np.sum((y - X @ beta) ** 2)) / (120 - 4)
            se = np.sqrt(s2 * np.diag(inverse))
            got = [out["loading_standard_errors"][name][f] for f in factors.columns]
            np.testing.assert_allclose(got, se[1:], rtol=1e-10)
            assert out["specific_variance"][name] == pytest.approx(s2 * 252, rel=1e-10)
            assert out["specific_variance_standard_errors"][name] == pytest.approx(
                s2 * 252 * math.sqrt(2 / 116), rel=1e-10
            )


#: A known model to draw from: three assets on two correlated factors.
TRUE_B = np.array([[1.1, 0.4], [0.7, -0.3], [0.2, 0.9]])
TRUE_F = np.array([[0.04, 0.006], [0.006, 0.0225]])
TRUE_D = np.array([0.09, 0.0625, 0.04])
TRUE_ALPHA = np.array([0.0002, -0.0001, 0.0])


def _draws(n, repeats, seed, n_factors=2):
    rng = np.random.default_rng(seed)
    index = DATES[:n] if n <= N_DAYS else pd.bdate_range("2000-01-03", periods=n)
    root = np.linalg.cholesky(TRUE_F / 252)
    for _ in range(repeats):
        f = rng.standard_normal((n, 2)) @ root.T
        e = rng.standard_normal((n, 3)) * np.sqrt(TRUE_D / 252)
        y = TRUE_ALPHA + f @ TRUE_B.T + e
        yield (
            pd.DataFrame(y, index=index, columns=["A", "B", "C"]),
            pd.DataFrame(f, index=index, columns=["MKT", "VAL"]),
        )


class TestTheStandardErrorsCoverTheTruth:
    def test_b_f_d_and_alpha_fall_within_1_96_se_about_95_percent_of_the_time(
        self,
    ):
        """400 draws of 250 days from a known B, F and D. A 1.96-standard-
        error band around each estimate covers the true value about 95% of
        the time for every piece: 2,400 loadings, 1,200 factor covariances,
        1,200 specific variances and 1,200 alphas (measured 94.7, 94.3,
        94.7 and 94.0%). The bands are wide enough to catch a standard
        error that is off by a fifth: at 0.8 of the right width the
        coverage would be about 88%."""
        hits = {"B": [], "F": [], "D": [], "alpha": []}
        for assets, factors in _draws(250, 400, seed=2026):
            out = estimate_factor_model(assets, factors)
            for i, a in enumerate(["A", "B", "C"]):
                for j, k in enumerate(["MKT", "VAL"]):
                    est = out["loadings"][a][k]
                    se = out["loading_standard_errors"][a][k]
                    hits["B"].append(abs(est - TRUE_B[i, j]) <= 1.96 * se)
                hits["D"].append(
                    abs(out["specific_variance"][a] - TRUE_D[i])
                    <= 1.96 * out["specific_variance_standard_errors"][a]
                )
                hits["alpha"].append(
                    abs(out["alpha"][a] - TRUE_ALPHA[i])
                    <= 1.96 * out["alpha_standard_error"][a]
                )
            for x, r in enumerate(["MKT", "VAL"]):
                for y, c in enumerate(["MKT", "VAL"]):
                    if y < x:
                        continue
                    hits["F"].append(
                        abs(out["factor_covariance"][r][c] - TRUE_F[x, y])
                        <= 1.96 * out["factor_covariance_standard_errors"][r][c]
                    )
        for piece, flags in hits.items():
            coverage = float(np.mean(flags))
            assert 0.925 <= coverage <= 0.975, (piece, coverage)

    def test_d_averages_the_true_variance_where_ssr_over_n_would_not(self):
        """Five factors on 40 days: p = 6, so SSR / n averages 34/40 = 85%
        of the true specific variance. SSR / (n - p) averages the truth --
        over 600 draws, within 2% of it for each asset."""
        rng = np.random.default_rng(7)
        loadings = rng.normal(0.5, 0.3, (3, 5))
        index = DATES[:40]
        ratios = []
        for _ in range(600):
            f = rng.normal(0, 0.01, (40, 5))
            e = rng.standard_normal((40, 3)) * np.sqrt(TRUE_D / 252)
            out = estimate_factor_model(
                pd.DataFrame(f @ loadings.T + e, index=index, columns=list("ABC")),
                pd.DataFrame(f, index=index, columns=[f"F{k}" for k in range(5)]),
            )
            ratios.append([out["specific_variance"][a] for a in "ABC"] / TRUE_D)
        mean = np.mean(ratios, axis=0)
        np.testing.assert_allclose(mean, 1.0, atol=0.02)


class TestThePublishedCovariance:
    def test_it_is_b_f_b_plus_d_symmetric_and_positive_semidefinite(self, research):
        """The covariance_ref resolves to B F B' + D of the reported
        loadings, factor covariance and specific variances: equal to the
        exactly rounded sum over factor pairs to 1e-14 of its scale,
        symmetric to the last bit, with no eigenvalue below rounding, and
        annualized at 252 with the estimator named."""
        assets, factors = _random_panel(n=250)
        arguments = {
            "returns": {a: _dated(assets[a], assets.index) for a in assets},
            "factors": {f: _dated(factors[f], factors.index) for f in factors},
            "run_id": _run_id(),
        }
        out = research.dispatch("estimate_factor_model", arguments)
        stored = handoff.resolve(out["covariance_ref"], expect="covariance")
        names = list(assets.columns)
        fnames = list(factors.columns)
        assert [str(c) for c in stored.columns] == names
        B = out["factor_loadings"]
        F = out["factor_covariance"]
        D = out["specific_variance"]
        oracle = np.array(
            [
                [
                    math.fsum(
                        F[k][m] * (B[i][k] * B[j][m]) for k in fnames for m in fnames
                    )
                    + (D[i] if i == j else 0.0)
                    for j in names
                ]
                for i in names
            ]
        )
        matrix = stored.loc[names, names].to_numpy()
        scale = np.abs(oracle).max()
        assert np.abs(matrix - oracle).max() <= 1e-14 * scale
        assert np.array_equal(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-14 * scale
        assert stored.attrs["annualized"] is True
        assert stored.attrs["periods_per_year"] == 252
        assert stored.attrs["method"] == "factor_model"
        assert stored.attrs["n_observations"] == 250

    def test_optimize_portfolio_reads_it_as_its_covariance(self, research, portfolio):
        """Through covariance_ref the minimum-variance portfolio of the
        model's covariance is the closed form Sigma^-1 1 / 1' Sigma^-1 1."""
        assets, factors = _random_panel(n=250, seed=9)
        out = research.dispatch(
            "estimate_factor_model",
            {
                "returns": {a: _dated(assets[a], assets.index) for a in assets},
                "factors": {f: _dated(factors[f], factors.index) for f in factors},
                "run_id": _run_id(),
            },
        )
        stored = handoff.resolve(out["covariance_ref"], expect="covariance")
        sigma = stored.to_numpy()
        raw = np.linalg.solve(sigma, np.ones(len(sigma)))
        closed = raw / raw.sum()
        weights = portfolio.dispatch(
            "optimize_portfolio",
            {"objective": "min_variance", "covariance_ref": out["covariance_ref"]},
        )["weights"]
        got = [weights[str(a)] for a in stored.columns]
        np.testing.assert_allclose(got, closed, atol=1e-9)

    def test_without_a_run_id_nothing_is_published(self, research):
        out = research.dispatch("estimate_factor_model", _planted_arguments())
        assert out["covariance_ref"] is None

    def test_a_used_run_id_is_refused(self, research):
        run_id = _run_id()
        research.dispatch("estimate_factor_model", _planted_arguments(run_id=run_id))
        with pytest.raises(Exception, match="already"):
            research.dispatch(
                "estimate_factor_model", _planted_arguments(run_id=run_id)
            )


class TestTheSeriesAreAlignedAndReported:
    def test_a_panel_reference_and_dated_values_on_different_calendars(self, research):
        """The assets from a published returns_panel, the factors as dated
        values starting ten days later: the regression uses the dates they
        share, the alignment report says what each lost, and the answer is
        the regression on exactly those dates."""
        assets, factors = _random_panel(n=200, seed=3)
        run_id = _run_id()
        ref = handoff.publish(
            assets, kind="returns_panel", run_id=run_id, name="assets"
        )
        late = factors.iloc[10:]
        out = research.dispatch(
            "estimate_factor_model",
            {
                "returns_ref": ref,
                "factors": {f: _dated(late[f], late.index) for f in late},
            },
        )
        assert out["n_observations"] == 190
        assert out["streams"]["by_stream"]["W"]["n_dropped_before"] == 10
        assert out["streams"]["by_stream"]["F1"]["n_kept"] == 190
        assert any("aligned on the 190 dates" in w for w in out["warnings"])
        direct = estimate_factor_model(assets.iloc[10:], late)
        assert out["factor_loadings"] == direct["loadings"]

    def test_a_factors_reference_takes_the_place_of_factors(self, research):
        assets, factors = _random_panel(n=150, seed=8)
        run_id = _run_id()
        factors_ref = handoff.publish(
            factors, kind="returns_panel", run_id=run_id, name="factors"
        )
        out = research.dispatch(
            "estimate_factor_model",
            {
                "returns": {a: _dated(assets[a], assets.index) for a in assets},
                "factors_ref": factors_ref,
            },
        )
        assert out["factors"] == ["F1", "F2", "F3"]
        assert (
            out["factor_loadings"] == estimate_factor_model(assets, factors)["loadings"]
        )

    def test_columns_pick_the_assets_and_the_factors_out_of_one_panel(self, research):
        """One published panel holding both sides: `columns` names the
        assets and `factor_columns` the factors, in the order given, and
        the answer is the regression of those columns on those."""
        assets, factors = _random_panel(n=150, seed=4)
        both = pd.concat([assets, factors], axis=1)
        ref = handoff.publish(both, kind="returns_panel", run_id=_run_id(), name="all")
        out = research.dispatch(
            "estimate_factor_model",
            {
                "returns_ref": ref,
                "columns": ["Y", "W"],
                "factors_ref": ref,
                "factor_columns": ["F2", "F1"],
            },
        )
        assert out["assets"] == ["Y", "W"]
        assert out["factors"] == ["F2", "F1"]
        direct = estimate_factor_model(assets[["Y", "W"]], factors[["F2", "F1"]])
        assert out["factor_loadings"] == direct["loadings"]

    def test_it_takes_no_ticker_so_it_fetches_nothing(self):
        """References or inline values, never a symbol: no field anywhere
        in the input names a symbol, ticker or universe, so the tool is
        not a fetcher (openWorldHint false) and the determinism layer
        calls it."""
        from standard_quant_tools.agent.runtimes.research.factor_model_tools import (
            FactorModelInput,
        )
        from standard_quant_tools.mcp.catalog import build_catalog

        entry = build_catalog()["estimate_factor_model"]
        assert entry.reads_market_data is False
        names = set(FactorModelInput.model_fields)
        assert not names & {"symbol", "symbols", "ticker", "tickers", "universe"}


class TestRefusalsAndNulls:
    @pytest.mark.parametrize(
        "overrides, phrase",
        [
            (
                {"returns_ref": "sqt://returns_panel/x/y"},
                "exactly one of returns / returns_ref",
            ),
            ({"returns": None}, "exactly one of returns / returns_ref"),
            ({"factors": None}, "exactly one of factors / factors_ref"),
            ({"factors": {}}, "factors is empty"),
            ({"columns": ["A"]}, "columns names columns of returns_ref"),
            (
                {"factor_columns": ["MKT"]},
                "factor_columns names columns of factors_ref",
            ),
            (
                {
                    "returns": None,
                    "returns_ref": "sqt://returns_panel/x/y",
                    "columns": ["A", "A"],
                },
                r"columns names \['A'\] more than once",
            ),
            ({"returns": {"A": {"2023-01-03": float("nan")}}}, "not finite"),
            ({"returns": {"A": [0.01, 0.02]}}, "returns.A"),
        ],
    )
    def test_each_side_comes_from_exactly_one_place(self, research, overrides, phrase):
        with pytest.raises(pydantic.ValidationError, match=phrase):
            research.dispatch("estimate_factor_model", _planted_arguments(**overrides))

    def test_a_name_on_both_sides_is_refused(self, research):
        arguments = _planted_arguments()
        arguments["factors"] = {"A": arguments["factors"]["MKT"]}
        with pytest.raises(ValidationError, match="both as an asset and as a factor"):
            research.dispatch("estimate_factor_model", arguments)

    def test_dependent_factors_are_named(self, research):
        arguments = _planted_arguments()
        market = arguments["factors"]["MKT"]
        arguments["factors"]["TWICE"] = {d: 2.0 * v for d, v in market.items()}
        with pytest.raises(ValidationError, match="'TWICE'"):
            research.dispatch("estimate_factor_model", arguments)

    def test_a_column_the_panel_lacks_is_refused_by_name(self, research):
        assets, factors = _random_panel(n=80, seed=6)
        ref = handoff.publish(
            assets, kind="returns_panel", run_id=_run_id(), name="assets"
        )
        arguments = {
            "returns_ref": ref,
            "columns": ["W", "NOPE"],
            "factors": {f: _dated(factors[f], factors.index) for f in factors},
        }
        with pytest.raises(ValidationError, match=r"has no column \['NOPE'\]"):
            research.dispatch("estimate_factor_model", arguments)

    def test_a_date_that_is_not_a_date_is_refused(self, research):
        arguments = _planted_arguments()
        arguments["factors"]["MKT"]["not-a-date"] = 0.001
        with pytest.raises((ValidationError, pydantic.ValidationError), match="date"):
            research.dispatch("estimate_factor_model", arguments)

    def test_too_few_dates_for_a_residual_variance(self):
        assets, factors = _random_panel(n=4)
        with pytest.raises(ValidationError, match="degrees of freedom"):
            estimate_factor_model(assets, factors)
        # Without the intercept the same four dates leave one.
        out = estimate_factor_model(assets, factors, intercept=False)
        assert out["degrees_of_freedom"] == 1

    def test_frames_on_different_dates_are_refused(self):
        assets, factors = _random_panel(n=50)
        with pytest.raises(ValidationError, match="not on the same dates"):
            estimate_factor_model(assets, factors.iloc[1:])

    def test_a_flat_asset_has_a_null_r_squared_with_its_reason(self, research):
        arguments = _planted_arguments()
        arguments["returns"]["FLAT"] = {d: 0.0001 for d in _iso(DATES)}
        out = research.dispatch("estimate_factor_model", arguments)
        assert out["r_squared"]["FLAT"] is None
        assert any(
            w.startswith("r_squared") and "never varied" in w for w in out["warnings"]
        )

    def test_one_asset_has_no_residual_correlation_and_says_why(self, research):
        arguments = _planted_arguments()
        del arguments["returns"]["B"]
        out = research.dispatch("estimate_factor_model", arguments)
        assert out["largest_residual_correlation"] is None
        assert out["largest_residual_correlation_pair"] is None
        assert any(
            w.startswith("largest_residual_correlation") for w in out["warnings"]
        )

    def test_a_shared_missing_factor_is_flagged_by_the_residual_correlation(self):
        """Two names that share a factor the model was not given keep a
        residual correlation the diagonal D throws away, and the result
        names them."""
        rng = np.random.default_rng(12)
        n = 400
        index = DATES[:n]
        market = rng.normal(0, 0.01, n)
        hidden = rng.normal(0, 0.01, n)
        values = {
            "A": market + hidden + rng.normal(0, 0.004, n),
            "B": 0.8 * market + hidden + rng.normal(0, 0.004, n),
            "C": 1.1 * market + rng.normal(0, 0.01, n),
        }
        out = estimate_factor_model(
            pd.DataFrame(values, index=index),
            pd.DataFrame({"MKT": market}, index=index),
        )
        assert out["largest_residual_correlation_pair"] == ["A", "B"]
        assert out["largest_residual_correlation"] > 0.8
        assert any("correlate at" in w for w in out["warnings"])
