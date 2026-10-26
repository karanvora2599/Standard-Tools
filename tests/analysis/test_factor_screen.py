"""
Which candidate factors earn a place in a risk model, on planted panels.

Every panel here is built from a known factor model, so what each test
should find is known before it runs:

- a factor that drives a planted share s of every asset's variance: its
  out-of-sample incremental R-squared is s less a little estimation error,
  no circular shift does as well (p is the smallest the shifts allow), and
  it is selected;
- a factor of pure noise is not selected, and over forty seeds its
  p-values are spread as a true null's are -- the circular shift of an
  i.i.d. series is exchangeable with it, so p is uniform on the shifts'
  grid;
- a near-copy of the base factor fails on VIF, naming the factor it
  copies;
- a factor whose loadings flip sign halfway explains returns in every
  window (it is significant) and still fails stability;
- a row after t changes nothing computed for a window ending at or before
  t.

And against independent arithmetic: the Frisch-Waugh-Lovell pass the null
runs gives the errors of a direct fit of base + candidate; each step's
covariance is `estimate_factor_model`'s on the same rows; the
minimum-variance weights solve S w = lambda 1 and sum to one; the
adjusted p-values are `bh_adjust` of the raw ones.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.analysis.factor_model import estimate_factor_model
from standard_quant_tools.analysis.factor_screen import (
    MAX_NULL_CELLS,
    _design,
    _min_variance_weights,
    _Walk,
    factor_covariance,
    screen_factor_candidates,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.validation.comparison import bh_adjust

#: Small enough that a seed costs a tenth of a second.
SMALL = dict(estimation_window=120, step=20, n_shifts=99)


def _panel(
    seed: int,
    *,
    T: int = 600,
    N: int = 12,
    share: float = 0.2,
    noise: int = 1,
    copy: bool = False,
    flip: bool = False,
):
    """Assets from a market factor (betas 0.6 to 1.4, the base) and a factor
    `strong` sized so it is `share` of every asset's variance, with i.i.d.
    residuals of the market's size; plus `noise` pure-noise factors, a
    near-copy of the market and a factor whose loadings change sign
    halfway."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=T)
    vol = 0.01
    mkt = rng.normal(0.0, vol, T)
    strong = rng.normal(0.0, vol, T)
    betas = rng.uniform(0.6, 1.4, N)
    # Every series at the same volatility: b^2 = share (beta^2 + b^2 + 1).
    b = np.sqrt(share * (betas**2 + 1) / (1 - share))
    loadings = b * rng.choice([-1.0, 1.0], N)
    returns = (
        np.outer(mkt, betas) + np.outer(strong, loadings) + rng.normal(0.0, vol, (T, N))
    )
    factors = {"mkt": mkt, "strong": strong}
    if flip:
        flipper = rng.normal(0.0, vol, T)
        sign = np.where(np.arange(T) < T // 2, 1.0, -1.0)
        returns = returns + np.outer(flipper * sign, rng.uniform(0.5, 1.0, N))
        factors["flipper"] = flipper
    if copy:
        factors["copy"] = mkt + rng.normal(0.0, vol / 20, T)
    for j in range(noise):
        factors[f"noise{j}"] = rng.normal(0.0, vol, T)
    assets = pd.DataFrame(returns, index=index, columns=[f"A{i:02d}" for i in range(N)])
    return assets, pd.DataFrame(factors, index=index)


def _screen(assets, factors, **kw):
    arguments = dict(base_factors=["mkt"], **SMALL)
    arguments.update(kw)
    return screen_factor_candidates(assets, factors, **arguments)


class TestWhatTheScreenFinds:
    def test_a_factor_driving_a_planted_share_is_selected_with_the_smallest_p(self):
        """strong is 20% of every asset's variance: out of sample it adds
        about 0.2 of R-squared -- give or take the sample's own factor
        variance over 480 test rows (6% of it, one standard error) and less
        the cost of one more loading per window -- every shift does worse,
        and it passes all four tests."""
        assets, factors = _panel(1, share=0.2)
        out = _screen(assets, factors)
        row = out["candidates_screened"]["strong"]
        assert 0.16 < row["incremental_r_squared"] < 0.24
        assert row["p_value"] == pytest.approx(1 / 100)
        assert row["p_value_bh"] <= 0.10
        assert row["sign_stability"] > 0.99
        assert row["half_sample_correlation"] > 0.95
        assert row["vif"] < 1.2
        assert row["passed"] and row["failed"] == []
        assert "strong" in out["selected"]
        vol = out["min_variance_volatility"]
        assert row["min_variance_volatility"] < vol["base"]

    def test_pure_noise_is_not_selected_and_its_p_values_are_a_true_nulls(self):
        """Forty seeds, three noise candidates each and no true one: the
        circular shift of an i.i.d. series is exchangeable with it, so each p
        is uniform on {1/100, ..., 1}, and the shares at or under 0.1 and 0.5
        sit within three standard errors of 0.1 and 0.5. Every candidate is
        null, so Benjamini-Hochberg at 0.10 lets any through in at most a
        tenth of the seeds (here at most 8 of 40, two standard errors over
        4), and the stability and risk-model tests only remove more."""
        p_values = []
        significant = selected = 0
        for seed in range(40):
            assets, factors = _panel(100 + seed, T=400, N=8, noise=3)
            out = _screen(assets, factors.drop(columns=["strong"]), random_seed=seed)
            rows = out["candidates_screened"]
            p_values.extend(rows[n]["p_value"] for n in ("noise0", "noise1", "noise2"))
            significant += any(row["p_value_bh"] <= 0.10 for row in rows.values())
            selected += bool(out["selected"])
        p = np.array(p_values)
        assert abs(np.mean(p <= 0.1) - 0.1) < 3 * math.sqrt(0.09 / len(p))
        assert abs(np.mean(p <= 0.5) - 0.5) < 3 * math.sqrt(0.25 / len(p))
        assert p.min() >= 1 / 100
        assert significant <= 8
        assert selected <= significant

    def test_a_near_copy_of_the_base_factor_fails_on_vif(self):
        """copy is the market plus noise a twentieth its size: correlation
        0.999, VIF in the hundreds, and the reason names the market."""
        assets, factors = _panel(2, copy=True)
        out = _screen(assets, factors)
        row = out["candidates_screened"]["copy"]
        assert row["vif"] > 100
        assert row["most_correlated_factor"] == "mkt"
        assert row["most_correlated"] > 0.99
        assert any("redundant" in r and "'mkt'" in r for r in row["failed"])
        assert "copy" not in out["selected"]

    def test_loadings_that_flip_halfway_fail_stability_though_significant(self):
        """flipper moves every asset in every window -- so it is significant
        -- but with one sign in the first half and the other in the second:
        the halves' loadings correlate near -1 and each loading keeps its
        full-sample sign in about half the windows."""
        assets, factors = _panel(3, flip=True)
        out = _screen(assets, factors)
        row = out["candidates_screened"]["flipper"]
        assert row["p_value_bh"] <= 0.10
        assert row["half_sample_correlation"] < -0.9
        assert row["sign_stability"] < 0.75
        assert any(r.startswith("unstable loadings") for r in row["failed"])
        assert "flipper" not in out["selected"]
        assert "strong" in out["selected"]

    def test_the_count_screened_is_reported_and_the_selected_score_is_hindsight(self):
        assets, factors = _panel(4, noise=2)
        out = _screen(assets, factors)
        assert out["candidates"] == ["strong", "noise0", "noise1"]
        assert any(
            "3 candidate(s) were screened" in w and "hindsight" in w
            for w in out["warnings"]
        )


class TestNothingAfterTReachesAWindowEndingBeforeIt:
    def test_future_rows_change_no_earlier_step(self):
        """Every row after t (row 400 of 600) redrawn, for the assets and
        every factor: each step whose test block ends by t keeps its errors,
        its windows' loadings and its minimum-variance returns to the bit.
        The p-values, the full-sample loadings and the selection read the
        whole sample and may change; nothing else may."""
        assets, factors = _panel(5, noise=1)
        t = 400
        rng = np.random.default_rng(99)
        later_assets, later_factors = assets.copy(), factors.copy()
        later_assets.iloc[t:] = rng.normal(0.0, 0.03, later_assets.iloc[t:].shape)
        later_factors.iloc[t:] = rng.normal(0.0, 0.03, later_factors.iloc[t:].shape)
        before = _screen(assets, factors)
        after = _screen(later_assets, later_factors)
        cut = assets.index[t - 1]
        ended = [
            date
            for date, (_e0, _t0, t1) in zip(before["steps"].index, _blocks(before))
            if t1 <= t
        ]
        assert len(ended) >= 10
        pd.testing.assert_frame_equal(
            before["steps"].loc[ended], after["steps"].loc[ended]
        )
        columns = ["base", "ledoit_wolf", "strong", "noise0", "all_candidates"]
        pd.testing.assert_frame_equal(
            before["portfolio_returns"].loc[:cut, columns],
            after["portfolio_returns"].loc[:cut, columns],
        )
        for name in ("strong", "noise0"):
            windows = before["candidates_screened"][name]["window_loadings"]
            fitted = windows.index[windows.index <= cut]
            pd.testing.assert_frame_equal(
                windows.loc[fitted],
                after["candidates_screened"][name]["window_loadings"].loc[fitted],
            )
        changed = after["portfolio_returns"].loc[cut:, "base"].iloc[1:]
        assert not np.allclose(
            changed, before["portfolio_returns"].loc[cut:, "base"].iloc[1:]
        )


def _blocks(out):
    walk = _Walk(
        np.zeros((out["n_observations"], 1)), out["estimation_window"], out["step"]
    )
    return walk.blocks


class TestAgainstIndependentArithmetic:
    def test_the_null_pass_is_a_direct_fit_of_base_and_candidate(self):
        """Offset 0 of the Frisch-Waugh-Lovell pass against one least-squares
        fit of base + candidate per window; and an offset k against the
        candidate rolled by k with np.roll."""
        assets, factors = _panel(6)
        Y = assets.to_numpy()
        T = len(Y)
        walk = _Walk(Y, 120, 20)
        base = _design([factors["mkt"].to_numpy()], T)
        names = ["intercept", "mkt"]
        fits = walk.base_fits(base, names)
        c = factors["strong"].to_numpy()
        got = walk.shifted_sse(base, names, fits, c, np.array([0, 37]))
        direct = walk.sse(_design([factors["mkt"].to_numpy(), c], T), names + ["c"])
        rolled = walk.sse(
            _design([factors["mkt"].to_numpy(), np.roll(c, 37)], T), names + ["c"]
        )
        np.testing.assert_allclose(got[:, 0], direct, rtol=1e-10)
        np.testing.assert_allclose(got[:, 1], rolled, rtol=1e-10)

    def test_each_steps_covariance_is_estimate_factor_models(self):
        """B F B' + D on a window is estimate_factor_model's covariance on
        those rows, before it is annualized."""
        assets, factors = _panel(7)
        rows = slice(100, 352)
        X = _design([factors[f].to_numpy()[rows] for f in ("mkt", "strong")], 252)
        mine = factor_covariance(
            X, assets.to_numpy()[rows], ["intercept", "mkt", "strong"]
        )
        theirs = estimate_factor_model(
            assets.iloc[rows], factors[["mkt", "strong"]].iloc[rows], periods_per_year=1
        )
        np.testing.assert_allclose(mine, np.array(theirs["covariance"]), rtol=1e-10)

    def test_the_minimum_variance_weights_solve_the_first_order_condition(self):
        """S w = lambda 1 with 1'w = 1: the gradient of w'Sw is the same in
        every direction the budget allows."""
        rng = np.random.default_rng(8)
        A = rng.normal(size=(6, 6))
        S = A @ A.T + 6 * np.eye(6)
        w = _min_variance_weights(S)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        gradient = S @ w
        np.testing.assert_allclose(gradient, gradient.mean(), rtol=1e-10)

    def test_the_reported_volatility_is_the_realized_returns(self):
        assets, factors = _panel(9)
        out = _screen(assets, factors, periods_per_year=252)
        r = out["portfolio_returns"]
        for name in ("base", "ledoit_wolf"):
            assert out["min_variance_volatility"][name] == pytest.approx(
                float(np.std(r[name], ddof=1) * math.sqrt(252)), rel=1e-12
            )
        assert out["candidates_screened"]["strong"][
            "min_variance_volatility"
        ] == pytest.approx(float(np.std(r["strong"], ddof=1) * math.sqrt(252)))

    def test_the_adjusted_p_values_are_bh_adjust_of_the_raw_ones(self):
        assets, factors = _panel(10, noise=3)
        out = _screen(assets, factors)
        rows = out["candidates_screened"]
        names = out["candidates"]
        assert [rows[n]["p_value_bh"] for n in names] == bh_adjust(
            [rows[n]["p_value"] for n in names]
        )

    def test_a_candidates_draws_do_not_depend_on_the_others(self):
        """The shifts are seeded by the candidate's name: screened alone or
        beside two others, noise0 gets the same p-value; and the same call
        twice is the same bits."""
        assets, factors = _panel(11, noise=3)
        together = _screen(assets, factors)
        alone = _screen(assets, factors[["mkt", "noise0"]])
        assert (
            together["candidates_screened"]["noise0"]["p_value"]
            == alone["candidates_screened"]["noise0"]["p_value"]
        )
        again = _screen(assets, factors)
        assert again["min_variance_volatility"] == together["min_variance_volatility"]
        pd.testing.assert_frame_equal(again["steps"], together["steps"])

    def test_with_no_base_the_intercept_alone_is_the_base(self):
        """No base factor: the base model is each asset's window mean -- the
        very value SST is measured from, so its out-of-sample R-squared is 0
        to rounding -- its covariance the diagonal of the variances, and the
        market itself is a candidate that passes."""
        assets, factors = _panel(12)
        out = _screen(assets, factors, base_factors=[])
        assert out["candidates"] == ["mkt", "strong", "noise0"]
        assert "mkt" in out["selected"]
        assert out["base_oos_r_squared"] == pytest.approx(0.0, abs=1e-12)


class TestRefusals:
    @pytest.mark.parametrize(
        "change,match",
        [
            (dict(base_factors=["nope"]), "not among the factors"),
            (dict(base_factors=["mkt", "mkt"]), "twice"),
            (dict(base_factors=["mkt", "strong", "noise0"]), "no candidate"),
            (dict(estimation_window=3), "no residual variance"),
            (dict(estimation_window=590), "fewer than two steps"),
            (dict(fdr=1.5), "false discovery rate"),
            (dict(fdr=0.0), "false discovery rate"),
            (dict(n_shifts=0), "below 1"),
            (dict(step=True), "whole number"),
        ],
    )
    def test_an_argument_is_refused_by_name(self, change, match):
        assets, factors = _panel(13)
        with pytest.raises(ValidationError, match=match):
            _screen(assets, factors, **change)

    def test_one_asset_has_no_minimum_variance_book(self):
        assets, factors = _panel(14)
        with pytest.raises(ValidationError, match="at least two"):
            _screen(assets[["A00"]], factors)

    def test_a_factor_constant_for_a_window_is_refused(self):
        assets, factors = _panel(15)
        factors.loc[factors.index[:150], "noise0"] = 0.0
        with pytest.raises(ValidationError, match="same value for 120 rows"):
            _screen(assets, factors)

    def test_frames_on_different_dates_are_refused(self):
        assets, factors = _panel(16)
        with pytest.raises(ValidationError, match="not on the same dates"):
            _screen(assets.iloc[1:], factors)

    def test_a_missing_value_is_refused(self):
        assets, factors = _panel(17)
        assets.iloc[5, 2] = np.nan
        with pytest.raises(ValidationError, match="not finite"):
            _screen(assets, factors)

    def test_an_asset_named_as_a_factor_is_refused(self):
        assets, factors = _panel(18)
        with pytest.raises(ValidationError, match="both as an asset"):
            _screen(assets.rename(columns={"A00": "noise0"}), factors)

    def test_a_candidate_the_base_spans_is_refused_by_name(self):
        """Twice the market is no candidate: base + it has no unique
        loadings, and the refusal names it before any shift is drawn."""
        assets, factors = _panel(19)
        factors["twin"] = 2.0 * factors["mkt"]
        with pytest.raises(ValidationError, match="linearly dependent.*'twin'"):
            _screen(assets, factors)

    def test_a_null_too_large_for_one_call_is_refused_before_it_runs(self, monkeypatch):
        import standard_quant_tools.analysis.factor_screen as module

        monkeypatch.setattr(module, "MAX_NULL_CELLS", 1000)
        assets, factors = _panel(19)
        with pytest.raises(ValidationError, match="Lower n_shifts"):
            _screen(assets, factors)
        assert MAX_NULL_CELLS > 1000


# ── the tool ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def research():
    return resolve("research")


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _inline(frame):
    return {
        c: {d.strftime("%Y-%m-%d"): float(v) for d, v in frame[c].items()}
        for c in frame.columns
    }


class TestTheTool:
    def test_references_and_inline_values_give_the_same_screen(
        self, research, runs_dir
    ):
        """The library's numbers through the tool, by reference and inline,
        with the selection, the reasons and the hindsight label."""
        assets, factors = _panel(20, copy=True)
        arguments = dict(base_factors=["mkt"], **SMALL)
        inline = research.dispatch(
            "screen_factor_candidates",
            dict(returns=_inline(assets), factors=_inline(factors), **arguments),
        )
        both = handoff.publish(
            pd.concat([assets, factors], axis=1),
            kind="returns_panel",
            run_id="screen",
            name="panel",
        )
        by_ref = research.dispatch(
            "screen_factor_candidates",
            dict(
                returns_ref=both,
                columns=list(assets.columns),
                factors_ref=both,
                factor_columns=list(factors.columns),
                **arguments,
            ),
        )
        assert by_ref == inline
        library = _screen(assets, factors)
        assert inline["selected"] == library["selected"] == ["strong"]
        assert inline["n_candidates"] == 3
        row = inline["candidates"]["copy"]
        assert row["vif"] == library["candidates_screened"]["copy"]["vif"]
        assert not row["passed"] and any("redundant" in r for r in row["failed"])
        assert inline["min_variance_volatility"] == pytest.approx(
            library["min_variance_volatility"]
        )
        assert inline["first_test_date"] == str(assets.index[120].date())

    def test_a_null_volatility_says_why(self, research, runs_dir):
        """Ten candidates and a window of eleven rows: base + every
        candidate is twelve coefficients, so all_candidates is null and the
        warnings say the window leaves no residual variance."""
        assets, factors = _panel(21, T=200, N=4, noise=9)
        out = research.dispatch(
            "screen_factor_candidates",
            dict(
                returns=_inline(assets),
                factors=_inline(factors),
                base_factors=["mkt"],
                estimation_window=11,
                step=20,
                n_shifts=20,
            ),
        )
        assert out["min_variance_volatility"]["all_candidates"] is None
        assert any("all_candidates is null" in w for w in out["warnings"])

    @pytest.mark.parametrize(
        "arguments,match",
        [
            ({}, "exactly one of returns / returns_ref"),
            ({"returns": {"A": {"2024-01-02": 0.01}}, "factors": {}}, "empty"),
            (
                {
                    "returns": {"A": {"2024-01-02": 0.01}},
                    "factors": {"F": {"2024-01-02": 0.01}},
                    "base_factors": ["F", "F"],
                },
                "more than once",
            ),
            (
                {
                    "returns": {"A": {"2024-01-02": 0.01}},
                    "factors": {"F": {"2024-01-02": 0.01}},
                    "n_shifts": 5,
                },
                "greater than or equal",
            ),
            (
                {
                    "returns": {"A": {"2024-01-02": 0.01}},
                    "factors": {"F": {"2024-01-02": 0.01}},
                    "symbol": "AAPL",
                },
                "symbol",
            ),
        ],
    )
    def test_the_inputs_are_refused_at_the_boundary(self, research, arguments, match):
        with pytest.raises(pydantic.ValidationError, match=match):
            research.dispatch("screen_factor_candidates", arguments)

    def test_it_is_a_quant_research_tool_that_takes_no_ticker(self):
        import standard_quant_tools.agent as package
        from standard_quant_tools.agent import tools as facade
        from standard_quant_tools.mcp.schemas import property_names

        table = resolve("research").dispatch_table
        assert "screen_factor_candidates" in table
        assert facade.TOOL_CATEGORY["screen_factor_candidates"] == "quant_research"
        assert facade.screen_factor_candidates is table["screen_factor_candidates"][0]
        assert "screen_factor_candidates" in package.__all__
        names = set(
            property_names(table["screen_factor_candidates"][1].model_json_schema())
        )
        assert not names & {"symbol", "symbols", "ticker", "tickers", "universe"}
        assert "benchmark" not in names and "book" not in names
