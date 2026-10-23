"""
`evaluate_allocation_rule`: an allocation rule walked forward over sleeve
return streams and scored only on the periods after each fit.

WHAT IS HELD:

- Nothing after a date moves what was chosen at or before it. A rule fitted
  on a trailing window is handed a frame of its own, so changing every
  return after row t leaves every weight chosen at or before t and every
  out-of-sample return on or before t the same bits -- and a rule that tries
  to read past its window finds nothing, so it chooses exactly what a rule
  that never tried would.
- Zero edge is scored as zero edge. Eight sleeves with no expected return:
  a long-only max-Sharpe allocation fitted on the full sample looks good on
  periods it saw (0.71 on 40 histories), walked forward it does not
  (0.08), and the deflated Sharpe ratio at one trial flags it about as often
  as a 5% test should.
- The book by hand: the entry from cash, the drift of the weights, the cost
  per unit of turnover and the cash rate, against an independent recursion
  in currency.
- The paired Sharpe difference's interval covers a known difference near
  its nominal rate, and is exactly [0, 0] for a rule against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.allocation.sleeves import SleeveRule
from standard_quant_tools.allocation.walk_forward import (
    _paired_sharpe_difference,
    evaluate_allocation_rule,
    walk_rule_forward,
)
from standard_quant_tools.backtesting.overfitting import deflated_sharpe_ratio
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import annualized_sharpe

P = 252
N = 8
NAMES = [f"S{i}" for i in range(N)]
VOL = np.array([0.05, 0.08, 0.10, 0.12, 0.06, 0.15, 0.09, 0.20])
SHARPE = np.array([0.6, 0.3, 0.8, 0.4, 0.5, 0.2, 0.7, 0.5])
CORR = np.full((N, N), 0.1)
CORR[:4, :4] = 0.5
CORR[4:, 4:] = 0.5
np.fill_diagonal(CORR, 1.0)
ROOT = np.linalg.cholesky(np.outer(VOL, VOL) * CORR / P)


def _draw(rng, years, sharpe=SHARPE):
    n = int(round(years * P))
    values = (sharpe * VOL) / P + rng.standard_normal((n, N)) @ ROOT.T
    return pd.DataFrame(
        values, columns=NAMES, index=pd.bdate_range("2010-01-04", periods=n)
    )


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _evaluate(frame, rule, **kwargs):
    settings = dict(
        n_trials=1, estimation_window=252, rebalance_every=21, n_bootstrap=0
    )
    settings.update(kwargs)
    return evaluate_allocation_rule(frame, rule, **settings)


class TestNothingAfterADateMovesWhatWasChosenBeforeIt:
    @pytest.mark.parametrize(
        "rule",
        [
            SleeveRule(method="max_sharpe"),
            SleeveRule(method="risk_parity", target_volatility=0.04),
            SleeveRule(
                method="equal",
                capital=1e7,
                capacity={"S0": 1.3e6},
                capacity_excess="redistribute",
            ),
        ],
        ids=["max_sharpe", "target_volatility", "capacity"],
    )
    def test_changing_the_returns_after_t_changes_nothing_at_or_before_t(self, rule):
        """Every return after row 600 is changed. The weights chosen at or
        before that date, and the rule's and the benchmark's out-of-sample
        returns on or before it, are the same bits; after it they differ,
        so the test is not vacuous."""
        frame = _draw(np.random.default_rng(1), 4)
        cut = 600
        changed = frame.copy()
        noise = np.random.default_rng(2).normal(0.0, 0.01, (len(frame) - cut - 1, N))
        changed.iloc[cut + 1 :] = changed.iloc[cut + 1 :].to_numpy() * 1.5 + noise
        settings = dict(cost_bps=10.0, cash_rate=0.02)
        a = _evaluate(frame, rule, **settings)
        b = _evaluate(changed, rule, **settings)
        t = frame.index[cut]
        before_a, before_b = a["targets"].loc[:t], b["targets"].loc[:t]
        assert len(before_a) > 10
        assert list(before_a.index) == list(before_b.index)
        assert before_a.to_numpy().tobytes() == before_b.to_numpy().tobytes()
        on_or_before = np.asarray(a["dates"] <= t)
        for key in ("rule_returns", "benchmark_returns"):
            assert a[key][on_or_before].tobytes() == b[key][on_or_before].tobytes()
            assert not np.array_equal(a[key][~on_or_before], b[key][~on_or_before])

    def test_the_published_weights_and_returns_obey_it_too(self, runs_dir):
        """End to end through the tool: the published weight panel's rows
        and the published out-of-sample returns up to the date are the same
        with and without the change."""
        frame = _draw(np.random.default_rng(3), 3)
        cut = 450
        changed = frame.copy()
        changed.iloc[cut + 1 :] = -changed.iloc[cut + 1 :].to_numpy()
        tool = resolve("allocation")
        out = {}
        for label, data in (("a", frame), ("b", changed)):
            ref = handoff.publish(data, kind="returns_panel", run_id=label, name="p")
            out[label] = tool.dispatch(
                "evaluate_allocation_rule",
                {
                    "returns_ref": ref,
                    "rule": {"method": "risk_parity"},
                    "n_trials": 1,
                    "n_bootstrap": 0,
                    "run_id": f"out_{label}",
                },
            )
        t = frame.index[cut]
        weights = {}
        for k, v in out.items():
            panel = pd.DataFrame(
                handoff.resolve(v["weights_ref"], expect="weight_panel")
            )
            panel.index = pd.to_datetime(panel.index)
            weights[k] = panel.sort_index()
        returns = {
            k: handoff.resolve(v["oos_returns_ref"], expect="returns_panel")
            for k, v in out.items()
        }
        weights_a = weights["a"].loc[weights["a"].index <= t]
        weights_b = weights["b"].loc[weights["b"].index <= t]
        assert len(weights_a) > 5
        pd.testing.assert_frame_equal(weights_a, weights_b)
        pd.testing.assert_frame_equal(
            returns["a"].loc[returns["a"].index <= t],
            returns["b"].loc[returns["b"].index <= t],
        )


def _reach(array):
    """The largest array reachable from `array` through numpy's `.base`
    chain: what a rule could read beyond the rows it was given."""
    base = array
    while getattr(base, "base", None) is not None:
        base = base.base
    return base


class TestARuleCannotPeek:
    @staticmethod
    def _chase(window, _capital):
        """All of the capital in the sleeve with the best mean return in
        the window: the rule that rewards most whatever it can see."""
        weights = np.zeros(window.shape[1])
        weights[int(np.argmax(window.to_numpy().mean(axis=0)))] = 1.0
        return weights

    def test_a_rule_reaching_past_its_window_finds_nothing(self):
        """The peeking rule looks for rows after its window -- through the
        array it was handed and anything that array is a view of -- and
        records whether it saw any before choosing as the honest rule does.
        It never sees one, and its choices are the honest rule's, bit for
        bit. The detector itself works: on a numpy view of the first year it
        reaches the whole sample."""
        frame = _draw(np.random.default_rng(4), 4, np.zeros(N))
        values = frame.to_numpy()
        assert _reach(values[:252]).shape[0] == len(frame)
        found = []

        def peek(window, capital):
            seen = _reach(window.to_numpy())
            rows = seen.shape[0] if seen.ndim == 2 else seen.size // N
            found.append(rows > len(window))
            return self._chase(window, capital)

        settings = dict(estimation_window=252, rebalance_every=63)
        honest = walk_rule_forward(frame, self._chase, **settings)
        peeking = walk_rule_forward(frame, peek, **settings)
        assert found and not any(found)
        assert peeking["returns"].tobytes() == honest["returns"].tobytes()
        assert peeking["targets"].equals(honest["targets"])

    def test_a_rule_handed_the_future_would_have_shown_it(self):
        """The control: the same oracle given the future explicitly -- by
        the test, not by the library -- beats the honest rule by more than
        two Sharpe ratios on zero-edge sleeves, so finding nothing above is
        not a weak oracle."""
        frame = _draw(np.random.default_rng(4), 4, np.zeros(N))
        future = frame.to_numpy()
        position = {d: i for i, d in enumerate(frame.index)}

        def oracle(window, _capital):
            start = position[window.index[-1]] + 1
            weights = np.zeros(N)
            weights[int(np.argmax(future[start : start + 21].sum(axis=0)))] = 1.0
            return weights

        settings = dict(estimation_window=252, rebalance_every=21)
        path = walk_rule_forward(frame, oracle, **settings)
        honest = walk_rule_forward(frame, self._chase, **settings)
        assert (
            annualized_sharpe(path["returns"], P)
            - annualized_sharpe(honest["returns"], P)
            > 2.0
        )

    def test_the_full_sample_fit_of_the_same_rule_shows_the_look_ahead(self):
        """Fitted once on every period, the chasing rule picks the sleeve
        that did best over the very periods it is then scored on. Over 20
        zero-edge histories its Sharpe ratio on the out-of-sample periods
        averages well above the walked-forward one, which averages about
        zero: that difference is the look-ahead gap the tool reports."""
        rng = np.random.default_rng(6)
        gaps, walked = [], []
        settings = dict(estimation_window=504, rebalance_every=63)
        for _ in range(20):
            frame = _draw(rng, 5, np.zeros(N))
            path = walk_rule_forward(frame, self._chase, **settings)
            held_weights = self._chase(frame, None)
            held = walk_rule_forward(frame, lambda w, c: held_weights, **settings)
            sharpe = annualized_sharpe(path["returns"], P)
            gaps.append(annualized_sharpe(held["returns"], P) - sharpe)
            walked.append(sharpe)
        assert np.mean(gaps) > 0.3
        assert abs(np.mean(walked)) < 0.3


class TestZeroEdgeIsScoredAsZeroEdge:
    @staticmethod
    def _calibration(histories, rebalance_every):
        rng = np.random.default_rng(7)
        full, walked, flagged = [], [], 0
        for _ in range(histories):
            frame = _draw(rng, 5, np.zeros(N))
            out = _evaluate(
                frame,
                SleeveRule(method="max_sharpe", covariance_method="sample"),
                estimation_window=2 * P,
                rebalance_every=rebalance_every,
            )
            if out["full_sample"] is None:
                continue  # no sleeve beat cash over the whole sample
            full.append(out["full_sample"]["sharpe"])
            walked.append(out["rule"]["sharpe"])
            flagged += bool(out["deflated"]["significant_at_95"])
        return np.mean(full), np.mean(walked), flagged / len(full), len(full)

    def test_the_full_sample_fit_looks_good_and_the_walk_forward_does_not(self):
        """40 five-year zero-edge histories, a two-year window, quarterly
        refits: the full-sample fit scores 0.71 on the periods it saw and the
        walk forward 0.08 (one Sharpe ratio's standard error over three years
        is 0.58, so 0.09 over 40), and the deflated Sharpe ratio at one trial
        flags 5% of the walks."""
        full, walked, flagged, n = self._calibration(40, 63)
        assert n >= 35
        assert full > 0.4
        assert full - walked > 0.3
        assert abs(walked) < 0.3
        assert flagged <= 0.15

    @pytest.mark.slow
    def test_the_calibration_holds_on_more_histories(self):
        """150 histories with monthly refits:
        measured 0.70 against 0.10, 5% flagged."""
        full, walked, flagged, _n = self._calibration(150, 21)
        assert full - walked > 0.4
        assert abs(walked) < 0.15
        assert flagged <= 0.1


#: Two sleeves' planted returns, 22 periods: two to fit on, twenty scored.
_A = np.array([0.01 * math.sin(t) for t in range(22)])
_B = np.array([-0.004 + 0.003 * (t % 3) for t in range(22)])


def _by_hand(weights, every, cost_bps, cash_rate):
    """The same book kept in currency: holdings h and cash c, the cost paid
    out of the account before each rebalance's trades."""
    rf = cash_rate / P
    kappa = cost_bps / 10_000.0
    holdings = np.zeros(2)
    cash = 1.0
    out = []
    for t in range(2, 22):
        nav = start = holdings.sum() + cash
        if (t - 2) % every == 0:
            cost = kappa * np.abs(weights - holdings / nav).sum()
            nav *= 1.0 - cost
            holdings = weights * nav
            cash = nav - holdings.sum()
        holdings = holdings * (1.0 + np.array([_A[t], _B[t]]))
        cash = cash * (1.0 + rf)
        out.append((holdings.sum() + cash) / start - 1.0)
    return np.array(out)


class TestTheBookByHand:
    def test_the_first_three_periods_in_closed_form(self):
        """Weights 0.6 / 0.3 and 0.1 in cash, refitted every 2 periods, 50
        bps a unit of turnover, cash at 2.52% a year (1e-4 a day). The entry
        from cash trades 0.9: r2 = (1 - 0.0045)(1 + 0.6 a2 + 0.3 b2 + 0.1
        rf) - 1. Period 3 holds the drifted weights 0.6 (1 + a2) / (1 + g2),
        0.3 (1 + b2) / (1 + g2), cash 0.1 (1 + rf) / (1 + g2). Period 4 pays
        for trading back to 0.6 / 0.3 from where period 3 left them."""
        weights = np.array([0.6, 0.3])
        frame = pd.DataFrame(
            {"a": _A, "b": _B}, index=pd.bdate_range("2024-01-02", periods=22)
        )
        path = walk_rule_forward(
            frame,
            lambda window, capital: weights,
            estimation_window=2,
            rebalance_every=2,
            cost_bps=50.0,
            cash_rate=0.0252,
        )
        rf = 1e-4
        g2 = 0.6 * _A[2] + 0.3 * _B[2] + 0.1 * rf
        r2 = (1 - 0.005 * 0.9) * (1 + g2) - 1
        wa, wb, c = (
            0.6 * (1 + _A[2]) / (1 + g2),
            0.3 * (1 + _B[2]) / (1 + g2),
            0.1 * (1 + rf) / (1 + g2),
        )
        r3 = wa * _A[3] + wb * _B[3] + c * rf
        g3 = r3
        wa, wb = wa * (1 + _A[3]) / (1 + g3), wb * (1 + _B[3]) / (1 + g3)
        cost4 = 0.005 * (abs(0.6 - wa) + abs(0.3 - wb))
        r4 = (1 - cost4) * (1 + 0.6 * _A[4] + 0.3 * _B[4] + 0.1 * rf) - 1
        np.testing.assert_allclose(path["returns"][:3], [r2, r3, r4], rtol=1e-13)
        assert path["turnover"][:2] == pytest.approx(
            [0.9, abs(0.6 - wa) + abs(0.3 - wb)]
        )
        assert path["cost"][0] == pytest.approx(0.0045, rel=1e-15)

    def test_the_whole_path_is_the_recursion_in_currency(self):
        weights = np.array([0.6, 0.3])
        frame = pd.DataFrame(
            {"a": _A, "b": _B}, index=pd.bdate_range("2024-01-02", periods=22)
        )
        for every in (1, 2, 5):
            path = walk_rule_forward(
                frame,
                lambda window, capital: weights,
                estimation_window=2,
                rebalance_every=every,
                cost_bps=50.0,
                cash_rate=0.0252,
            )
            np.testing.assert_allclose(
                path["returns"], _by_hand(weights, every, 50.0, 0.0252), rtol=1e-12
            )

    def test_the_tool_keeps_the_same_book(self, runs_dir):
        """Equal weight through the tool: 0.5 / 0.5 fully invested, the
        same recursion."""
        frame = pd.DataFrame(
            {"a": _A, "b": _B}, index=pd.bdate_range("2024-01-02", periods=22)
        )
        ref = handoff.publish(frame, kind="returns_panel", run_id="hand", name="p")
        out = resolve("allocation").dispatch(
            "evaluate_allocation_rule",
            {
                "returns_ref": ref,
                "rule": {"method": "equal"},
                "n_trials": 1,
                "estimation_window": 2,
                "rebalance_every": 2,
                "cost_bps": 50.0,
                "cash_rate": 0.0252,
                "n_bootstrap": 0,
                "run_id": "hand_out",
            },
        )
        published = handoff.resolve(out["oos_returns_ref"], expect="returns_panel")
        np.testing.assert_allclose(
            published["rule"].to_numpy(),
            _by_hand(np.array([0.5, 0.5]), 2, 50.0, 0.0252),
            rtol=1e-12,
        )
        assert out["rule"]["total_cost"] == pytest.approx(
            out["benchmark"]["total_cost"], rel=1e-15
        )


class TestTheComparison:
    def test_a_rule_against_itself_differs_by_exactly_nothing(self):
        frame = _draw(np.random.default_rng(8), 2)
        out = _evaluate(frame, SleeveRule(method="equal"), n_bootstrap=200)
        difference = out["sharpe_difference"]
        assert difference["observed"] == 0.0
        assert (difference["lower"], difference["upper"]) == (0.0, 0.0)
        assert difference["p_value"] == 1.0

    def test_the_interval_covers_a_known_sharpe_difference_near_95_percent(self):
        """Two correlated daily streams 500 long whose true Sharpe ratios
        differ by 0.476 a year: over 200 pairs the 95% interval covers it
        near 95% (measured 93% on 300; one standard error is 1.5 points)."""
        rng = np.random.default_rng(3)
        truth = (0.0004 / 0.01 - 0.0001 / 0.01) * math.sqrt(P)
        hits = 0
        for seed in range(200):
            z = rng.standard_normal((500, 2))
            x = 0.0004 + 0.01 * z[:, 0]
            y = 0.0001 + 0.01 * (0.6 * z[:, 0] + 0.8 * z[:, 1])
            d = _paired_sharpe_difference(
                x,
                y,
                periods_per_year=P,
                n_bootstrap=400,
                block_size=8,
                random_seed=seed,
                confidence=0.95,
            )
            hits += d["lower"] <= truth <= d["upper"]
        assert 0.88 <= hits / 200 <= 0.99

    def test_the_deflated_sharpe_is_the_librarys_on_the_out_of_sample_stream(self):
        """At n_trials the probability is backtesting.overfitting's own on
        the rule's out-of-sample return over cash; more trials, less of it."""
        frame = _draw(np.random.default_rng(9), 3)
        out = _evaluate(
            frame, SleeveRule(method="risk_parity"), cash_rate=0.01, n_trials=7
        )
        expected = deflated_sharpe_ratio(
            pd.Series(out["rule_returns"] - 0.01 / P), n_trials=7
        )
        assert (
            out["deflated"]["deflated_sharpe_probability"]
            == expected["deflated_sharpe_probability"]
        )
        more = _evaluate(
            frame, SleeveRule(method="risk_parity"), cash_rate=0.01, n_trials=70
        )
        assert (
            more["deflated"]["deflated_sharpe_probability"]
            < out["deflated"]["deflated_sharpe_probability"]
        )


class TestARebalanceThatCannotBeFitted:
    def test_it_trades_nothing_and_says_so(self):
        """Every sleeve loses for the first 300 days, so a max-Sharpe fit on
        any window inside them is refused: those rebalances trade nothing,
        the book stays in cash earning exactly the cash rate, and each is
        counted with its date."""
        rng = np.random.default_rng(10)
        frame = _draw(rng, 3, np.full(N, 0.5))
        frame.iloc[:300] = frame.iloc[:300].to_numpy() - 0.004
        out = _evaluate(
            frame,
            SleeveRule(method="max_sharpe"),
            estimation_window=100,
            rebalance_every=50,
            cash_rate=0.0252,
        )
        assert out["rule"]["n_unfitted"] >= 3
        unfitted = [pd.Timestamp(d) for d, _why in out["unfitted"]]
        first_fit = out["targets"].index[0]
        assert all(d < first_fit for d in unfitted[:3])
        before = np.asarray(out["dates"] < first_fit)
        np.testing.assert_allclose(out["rule_returns"][before], 1e-4, rtol=1e-12)
        assert any("could not be fitted at" in w for w in out["warnings"])


class TestRefusals:
    @pytest.fixture
    def ref(self, runs_dir):
        frame = _draw(np.random.default_rng(12), 2)
        return handoff.publish(frame, kind="returns_panel", run_id="refuse", name="p")

    def test_the_trial_count_is_required(self, ref):
        with pytest.raises(pydantic.ValidationError, match="n_trials"):
            resolve("allocation").dispatch(
                "evaluate_allocation_rule",
                {"returns_ref": ref, "rule": {"method": "equal"}},
            )

    def test_a_window_that_leaves_too_little_out_of_sample_is_refused(self, ref):
        with pytest.raises(ValidationError, match="estimation_window of 500 leave 4"):
            resolve("allocation").dispatch(
                "evaluate_allocation_rule",
                {
                    "returns_ref": ref,
                    "rule": {"method": "equal"},
                    "n_trials": 1,
                    "estimation_window": 500,
                },
            )

    @pytest.mark.parametrize(
        "rule",
        [{"method": "kelly"}, {"method": "hrp", "long_only": False}, {"capital": -1.0}],
    )
    def test_a_rule_that_cannot_be_is_refused(self, ref, rule):
        with pytest.raises(pydantic.ValidationError):
            resolve("allocation").dispatch(
                "evaluate_allocation_rule",
                {"returns_ref": ref, "rule": rule, "n_trials": 1},
            )
