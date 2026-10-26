"""
Regime-conditioned allocation rules: a rule whose mean and covariance are
those of the window's returns in the current regime, shrunk toward the
whole window's.

WHAT IS HELD:

- One regime is the unconditioned rule. A state that does not vary labels
  nothing, and the rule's weights, covariance and walk-forward returns are
  the unconditioned rule's bit for bit; an enormous shrinkage n0 converges
  on them too.
- A planted two-regime covariance is recovered. Returns are drawn from one
  covariance after a calm period and another after a volatile one; with the
  state as a column, the mixture labels and no shrinkage, the fitted
  covariance is the current regime's to within four of the sample
  covariance's own standard errors, sqrt((s_ij^2 + s_ii s_jj) / n), and
  further than that from the other regime's.
- The shrinkage is the stated blend: n_r / (n_r + n0) on the regime's
  estimate, the rest on the whole window's, against numpy on rows chosen
  independently.
- A return is paired with the label of the period before it, never its
  own, and the current regime is the window's last label.
- Walked forward, nothing after a date moves what was chosen at or before
  it, and the state column is in no book.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.allocation.sleeves import (
    RegimeSpec,
    SleeveRule,
    allocate_across_sleeves,
    ewma_volatility,
    fit_sleeve_rule,
    label_regimes,
)
from standard_quant_tools.allocation.walk_forward import evaluate_allocation_rule
from standard_quant_tools.error import ValidationError

P = 252
N = 3
NAMES = [f"S{i}" for i in range(N)]
CALM = np.outer([0.10, 0.12, 0.08], [0.10, 0.12, 0.08]) * np.array(
    [[1.0, 0.2, 0.1], [0.2, 1.0, 0.2], [0.1, 0.2, 1.0]]
)
STORM = np.outer([0.30, 0.35, 0.25], [0.30, 0.35, 0.25]) * np.array(
    [[1.0, 0.8, 0.7], [0.8, 1.0, 0.8], [0.7, 0.8, 1.0]]
)


def _markov(rng, n, stay=(0.99, 0.98), last=None):
    state = np.zeros(n, dtype=int)
    u = rng.random(n)
    for t in range(1, n):
        state[t] = state[t - 1] if u[t] < stay[state[t - 1]] else 1 - state[t - 1]
    if last is not None:
        state[-30:] = last
    return state


def _two_regimes(rng, n, last=None):
    """Sleeve returns whose covariance on row s is CALM or STORM as the
    regime of row s - 1 was, and the state column: that regime plus a
    little noise, known at the end of its row."""
    regime = _markov(rng, n, last=last)
    roots = [np.linalg.cholesky(CALM / P), np.linalg.cholesky(STORM / P)]
    z = rng.standard_normal((n, N))
    previous = np.concatenate([[0], regime[:-1]])
    values = np.where(previous[:, None] == 0, z @ roots[0].T, z @ roots[1].T)
    frame = pd.DataFrame(
        values, columns=NAMES, index=pd.bdate_range("2001-01-02", periods=n)
    )
    frame["state"] = regime + 0.02 * rng.standard_normal(n)
    return frame, regime


def _iid(rng, n):
    root = np.linalg.cholesky(CALM / P)
    values = 0.0003 + rng.standard_normal((n, N)) @ root.T
    return pd.DataFrame(
        values, columns=NAMES, index=pd.bdate_range("2001-01-02", periods=n)
    )


class TestOneRegimeIsTheUnconditionedRule:
    @pytest.mark.parametrize(
        "method",
        ["inverse_volatility", "risk_parity", "hrp", "min_variance", "max_sharpe"],
    )
    def test_a_state_that_never_varies_labels_nothing(self, method):
        """A constant state column: there is one regime, the rule is fitted
        unconditioned and says so, and its weights and covariance are the
        unconditioned rule's on the sleeves, bit for bit."""
        frame = _iid(np.random.default_rng(1), 500)
        with_state = frame.assign(state=0.5)
        spec = RegimeSpec(state_column="state")
        a = fit_sleeve_rule(with_state, SleeveRule(method=method, regime=spec))
        b = fit_sleeve_rule(frame, SleeveRule(method=method))
        assert a["weights"].tobytes() == b["weights"].tobytes()
        assert a["covariance"].tobytes() == b["covariance"].tobytes()
        assert a["sleeves"] == NAMES
        assert a["details"]["regime"]["conditioned"] is False
        assert any("does not vary" in note for note in a["notes"])

    def test_walked_forward_it_earns_the_unconditioned_rules_returns(self):
        """The same through evaluate_allocation_rule: the regime rule on a
        constant state and the plain rule on the sleeves give the same
        out-of-sample returns and targets, bit for bit -- the state column
        rides in the windows and in no book."""
        frame = _iid(np.random.default_rng(2), 800)
        settings = dict(n_trials=1, estimation_window=252, n_bootstrap=0, cost_bps=10)
        a = evaluate_allocation_rule(
            frame.assign(state=1.0),
            SleeveRule(method="min_variance", regime=RegimeSpec(state_column="state")),
            **settings,
        )
        b = evaluate_allocation_rule(
            frame, SleeveRule(method="min_variance"), **settings
        )
        assert a["rule_returns"].tobytes() == b["rule_returns"].tobytes()
        assert a["benchmark_returns"].tobytes() == b["benchmark_returns"].tobytes()
        assert a["targets"].equals(b["targets"])
        assert list(a["full_sample_weights"]) == NAMES
        assert a["regime"]["n_conditioned"] == 0

    @pytest.mark.parametrize("method", ["min_variance", "max_sharpe", "shrunk_sharpe"])
    def test_an_enormous_shrinkage_is_the_unconditioned_rule(self, method):
        """n0 = 1e12 puts a weight of n_r / (n_r + n0) ~ 1e-10 on the
        regime: the conditioned weights are the unconditioned ones to 1e-8."""
        rng = np.random.default_rng(3)
        frame = pd.concat(
            [_iid(rng, 600), _iid(rng, 600).add_prefix("T")], axis=1
        )  # six sleeves: shrunk_sharpe needs at least four
        spec = RegimeSpec(labels="quantile", shrinkage=1e12)
        a = fit_sleeve_rule(frame, SleeveRule(method=method, regime=spec))
        b = fit_sleeve_rule(frame, SleeveRule(method=method))
        assert a["details"]["regime"]["conditioned"] is True
        assert a["details"]["regime"]["shrink_weight"] < 1e-9
        np.testing.assert_allclose(a["weights"], b["weights"], atol=1e-8)


class TestAPlantedTwoRegimeCovarianceIsRecovered:
    @pytest.mark.parametrize("last", [0, 1], ids=["calm_now", "storm_now"])
    def test_the_current_regimes_covariance_within_its_sampling_error(self, last):
        """4,000 periods, the regime Markov (stay 0.99 calm, 0.98 stormy),
        the last 30 forced into one regime. With no shrinkage and the sample
        covariance, the fitted (annual) covariance is that regime's to within
        four standard errors element by element, and at least that far from
        the other regime's on the diagonal: the test can tell them apart."""
        frame, regime = _two_regimes(np.random.default_rng(7 + last), 4000, last=last)
        rule = SleeveRule(
            method="min_variance",
            covariance_method="sample",
            regime=RegimeSpec(state_column="state", shrinkage=0.0),
        )
        fit = fit_sleeve_rule(frame, rule)
        report = fit["details"]["regime"]
        assert report["current_regime"] == last
        assert report["shrink_weight"] == 1.0
        n = report["regime_periods"]
        assert n == int(np.sum(regime[:-1] == last))
        truth, other = (CALM, STORM) if last == 0 else (STORM, CALM)
        se = np.sqrt((truth**2 + np.outer(np.diag(truth), np.diag(truth))) / n)
        assert np.all(np.abs(fit["covariance"] - truth) < 4 * se)
        assert np.all(
            np.abs(np.diag(fit["covariance"]) - np.diag(other)) > 4 * np.diag(se)
        )

    def test_the_shrinkage_is_the_stated_blend(self):
        """The rows are chosen here from the planted state -- returns whose
        previous period's state is in the current cluster -- and numpy's
        covariance of them and of the whole window, blended by n/(n + n0),
        is the fitted covariance; the mean max_sharpe reads is the same
        blend of the means."""
        frame, _regime = _two_regimes(np.random.default_rng(9), 1500, last=1)
        n0 = 50.0
        rule = SleeveRule(
            method="min_variance",
            covariance_method="sample",
            regime=RegimeSpec(state_column="state", shrinkage=n0),
        )
        fit = fit_sleeve_rule(frame, rule)
        state = frame["state"].to_numpy()
        values = frame[NAMES].to_numpy()
        rows = np.flatnonzero(state[:-1] > 0.5) + 1
        weight = rows.size / (rows.size + n0)
        expected = P * (
            weight * np.cov(values[rows], rowvar=False)
            + (1 - weight) * np.cov(values, rowvar=False)
        )
        assert fit["details"]["regime"]["shrink_weight"] == pytest.approx(weight)
        np.testing.assert_allclose(fit["covariance"], expected, rtol=1e-12)


class TestTheLabels:
    def test_quantile_labels_cut_at_the_windows_own_quantiles(self):
        """Two regimes cut at the median, three at the terciles; a state on
        a cut takes the lower label; regime 0 holds the lowest states."""
        state = np.arange(1.0, 10.0)
        two, _ = label_regimes(state, RegimeSpec(labels="quantile"))
        assert list(two) == [0, 0, 0, 0, 0, 1, 1, 1, 1]
        three, _ = label_regimes(state, RegimeSpec(labels="quantile", n_regimes=3))
        assert list(three) == [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def test_mixture_regimes_are_sorted_by_the_states_level(self):
        """The mixture sorts its components by their spread; the labels are
        re-sorted by the state's mean, so a tight cluster of HIGH states --
        whose spread is the smaller -- is regime 1, not regime 0."""
        rng = np.random.default_rng(4)
        low = rng.normal(0.0, 1.0, 200)
        high = rng.normal(10.0, 0.1, 200)
        labels, why = label_regimes(np.concatenate([low, high]), RegimeSpec())
        assert why is None
        assert np.all(labels[:200] == 0) and np.all(labels[200:] == 1)

    def test_a_return_is_paired_with_the_label_before_it(self):
        """With the EWMA volatility state, a flat first period (the lowest
        state) and an enormous last return (the highest): the last period
        is labelled volatile and is the current regime, and the counts per
        regime are those of the labels of every period but the last -- the
        labels the returns after them are paired with -- not of every period
        but the first, which is what a return labelled by its own state
        would give."""
        frame = _iid(np.random.default_rng(5), 300)
        frame.iloc[0] = 0.0
        frame.iloc[-1] = 0.5
        values = frame.to_numpy()
        state = ewma_volatility(values.mean(axis=1), RegimeSpec().halflife)
        labels, _ = label_regimes(state, RegimeSpec(labels="quantile"))
        assert labels[0] == 0 and labels[-1] == 1
        fit = fit_sleeve_rule(
            frame, SleeveRule(method="equal", regime=RegimeSpec(labels="quantile"))
        )
        report = fit["details"]["regime"]
        assert report["current_regime"] == 1
        paired = list(np.bincount(labels[:-1], minlength=2))
        assert report["periods_per_regime"] == paired
        assert paired != list(np.bincount(labels[1:], minlength=2))
        switches = np.count_nonzero(np.diff(labels))
        assert report["persistence"] == pytest.approx(1 - switches / (len(labels) - 1))

    def test_the_ewma_state_is_the_weighted_mean_of_squares(self):
        """Against the definition summed directly: on every period, the root
        of the a^(s-j)-weighted mean of the squares up to it."""
        x = np.random.default_rng(6).normal(0, 0.01, 50)
        a = 0.5 ** (1 / 7.0)
        direct = [
            math.sqrt(
                sum(a ** (s - j) * x[j] ** 2 for j in range(s + 1))
                / sum(a ** (s - j) for j in range(s + 1))
            )
            for s in range(50)
        ]
        np.testing.assert_allclose(ewma_volatility(x, 7.0), direct, rtol=1e-12)


class TestFallsBackWithANote:
    def test_too_few_returns_in_the_current_regime(self):
        """The last period alone is stormy: its regime has no returns after
        it in the window, fewer than min_observations, so the rule is fitted
        unconditioned -- the unconditioned weights to the bit -- and says so."""
        frame = _iid(np.random.default_rng(8), 300)
        frame["state"] = 0.0
        frame.iloc[-1, frame.columns.get_loc("state")] = 1.0
        rule = SleeveRule(
            method="risk_parity",
            regime=RegimeSpec(state_column="state", labels="quantile"),
        )
        fit = fit_sleeve_rule(frame, rule)
        plain = fit_sleeve_rule(frame[NAMES], SleeveRule(method="risk_parity"))
        assert fit["weights"].tobytes() == plain["weights"].tobytes()
        assert fit["details"]["regime"]["current_regime"] == 1
        assert fit["details"]["regime"]["conditioned"] is False
        assert any("fewer than min_observations=20" in n for n in fit["notes"])


class TestNothingAfterADateMovesWhatWasChosenBeforeIt:
    @pytest.mark.parametrize("state_column", [None, "state"])
    def test_changing_the_returns_after_t_changes_nothing_at_or_before_t(
        self, state_column
    ):
        """Every row after 700 changed, the state with it. The regime rule's
        targets on or before that date and its and the benchmark's returns
        are the same bits; after it they differ."""
        frame, _regime = _two_regimes(np.random.default_rng(10), 1200)
        frame[NAMES] += np.array([0.0006, 0.0004, 0.0005])  # means above cash
        if state_column is None:
            frame = frame[NAMES]
        cut = 700
        changed = frame.copy()
        noise = np.random.default_rng(11).normal(0, 0.01, changed.iloc[cut + 1 :].shape)
        changed.iloc[cut + 1 :] = changed.iloc[cut + 1 :].to_numpy() * 1.3 + noise
        rule = SleeveRule(
            method="max_sharpe", regime=RegimeSpec(state_column=state_column)
        )
        settings = dict(n_trials=1, n_bootstrap=0, cost_bps=10, cash_rate=0.01)
        a = evaluate_allocation_rule(frame, rule, **settings)
        b = evaluate_allocation_rule(changed, rule, **settings)
        t = frame.index[cut]
        ta, tb = a["targets"].loc[:t], b["targets"].loc[:t]
        assert len(ta) > 10
        assert ta.to_numpy().tobytes() == tb.to_numpy().tobytes()
        before = np.asarray(a["dates"] <= t)
        for key in ("rule_returns", "benchmark_returns"):
            assert a[key][before].tobytes() == b[key][before].tobytes()
            assert not np.array_equal(a[key][~before], b[key][~before])


class TestTheAllocationReportsItsRegime:
    def test_allocate_across_sleeves_names_the_sleeves_and_the_regime(self):
        """The state column is not a sleeve: the weights, risk and bands are
        over the others, and the report carries the regime."""
        frame, _ = _two_regimes(np.random.default_rng(12), 800, last=1)
        out = allocate_across_sleeves(
            frame,
            SleeveRule(method="risk_parity", regime=RegimeSpec(state_column="state")),
            n_bootstrap=20,
        )
        assert list(out["weights"]) == NAMES
        assert list(out["weight_p5"]) == NAMES
        assert out["n_sleeves"] == N
        assert out["regime"]["current_regime"] == 1
        assert out["regime"]["state"] == "state"

    def test_through_the_tool(self):
        """The tool takes the regime inline and returns its report; a
        changed halflife beside a state column is refused, since the column
        replaces the EWMA state the halflife is for."""
        frame, _ = _two_regimes(np.random.default_rng(13), 300, last=0)
        returns = {
            c: {d.strftime("%Y-%m-%d"): float(v) for d, v in frame[c].items()}
            for c in frame.columns
        }
        out = resolve("allocation").dispatch(
            "allocate_across_sleeves",
            {
                "returns": returns,
                "method": "min_variance",
                "regime": {"state_column": "state", "shrinkage": 0},
                "n_bootstrap": 0,
            },
        )
        assert out["regime"]["conditioned"] is True
        assert out["regime"]["current_regime"] == 0
        assert set(out["weights"]) == set(NAMES)
        from standard_quant_tools.agent.runtimes.allocation.sleeve_models import (
            AllocationRule,
        )

        with pytest.raises(pydantic.ValidationError, match="halflife"):
            AllocationRule(regime={"state_column": "state", "halflife": 3})


class TestRefusals:
    def test_by_name(self):
        """A state column the returns lack, two few or many regimes, an
        unknown labelling, a negative shrinkage, and a state column named as
        a sleeve's capacity: each refused, naming it."""
        frame = _iid(np.random.default_rng(14), 200)
        with pytest.raises(ValidationError, match="state_column='vix'"):
            fit_sleeve_rule(frame, SleeveRule(regime=RegimeSpec(state_column="vix")))
        for kwargs, phrase in [
            ({"n_regimes": 4}, "n_regimes=4"),
            ({"labels": "hmm"}, "labels='hmm'"),
            ({"shrinkage": -1.0}, "shrinkage=-1.0"),
            ({"min_observations": 1}, "min_observations=1"),
        ]:
            with pytest.raises(ValidationError, match=phrase):
                RegimeSpec(**kwargs)
        with pytest.raises(ValidationError, match="state_column='S0'"):
            SleeveRule(
                capital=1e6,
                capacity={"S0": 1e5},
                regime=RegimeSpec(state_column="S0"),
            )
