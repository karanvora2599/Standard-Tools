"""
`select_allocation_rule`: choosing among allocation rules by their own
out-of-sample record, as you go, tested out of sample.

WHAT IS HELD:

- A selector that always holds one rule earns that rule's walk-forward
  returns to the bit: the selector's book is `walk_rule_forward`'s
  arithmetic, so one candidate (or copies of one) reproduces
  `evaluate_allocation_rule`'s rule stream exactly.
- Nothing after a date moves what was chosen at or before it. The selector
  scores each candidate on that candidate's realized out-of-sample returns
  strictly before the rebalance; changing every row after t leaves every
  choice, target and return at or before t the same bits.
- A dominating rule is found: once the first selection window has passed
  the selector holds it at every rebalance, after exactly one switch.
- The hysteresis, the top-k blend and the warm-up behave as stated, and the
  book is checked by hand against a recursion in currency.
- No edge is not claimed: on exchangeable sleeves, where equal weight is
  the best any rule can do, the selector's interval against equal weight
  rarely clears zero and the probability of backtest overfitting sits near
  one half.
- The hindsight pick, the PBO and the deflated Sharpe ratios are the
  library's own functions applied to the streams the result reports.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.allocation.selection import (
    _choose,
    default_top_k,
    run_selector,
    select_allocation_rule,
    walk_candidates,
)
from standard_quant_tools.allocation.sleeves import RegimeSpec, SleeveRule
from standard_quant_tools.allocation.walk_forward import (
    evaluate_allocation_rule,
    walk_sleeve_rule,
)
from standard_quant_tools.backtesting.overfitting import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import annualized_sharpe

P = 252
N = 4
NAMES = [f"S{i}" for i in range(N)]
W, K, L = 126, 21, 126


def _draw(rng, years, sharpe=(0.6, 0.3, 0.8, 0.4), vol=0.1, corr=0.3):
    n = int(round(years * P))
    c = np.full((N, N), corr)
    np.fill_diagonal(c, 1.0)
    root = np.linalg.cholesky(vol * vol * c / P)
    values = np.asarray(sharpe) * vol / P + rng.standard_normal((n, N)) @ root.T
    return pd.DataFrame(
        values, columns=NAMES, index=pd.bdate_range("2012-01-02", periods=n)
    )


@pytest.fixture(scope="module")
def three_years():
    return _draw(np.random.default_rng(20261026), 3)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


SETTINGS = dict(estimation_window=W, rebalance_every=K, cost_bps=10.0)


def _select(frame, candidates, **kwargs):
    settings = dict(SETTINGS, n_trials=1, selection_window=L, n_bootstrap=0)
    settings.update(kwargs)
    return select_allocation_rule(frame, candidates, **settings)


class TestOneRuleIsThatRule:
    @pytest.mark.parametrize(
        "rule",
        [
            SleeveRule(method="max_sharpe"),
            SleeveRule(method="risk_parity", target_volatility=0.06),
            SleeveRule(method="min_variance", regime=RegimeSpec(labels="quantile")),
        ],
        ids=["max_sharpe", "target_volatility", "regime"],
    )
    def test_a_selector_over_one_candidate_earns_its_stream_to_the_bit(
        self, three_years, rule
    ):
        """The selector's own book, run over a single candidate's walk, is
        that walk's out-of-sample returns bit for bit -- and so is
        evaluate_allocation_rule's rule stream for the same rule. The cash
        rate is high enough that max-Sharpe windows where no sleeve beat
        cash leave the rule unfitted, so the no-trade branch is crossed."""
        settings = dict(SETTINGS, cash_rate=0.06)
        path = walk_sleeve_rule(three_years, rule, **settings)
        run = run_selector(
            three_years.to_numpy(),
            [path],
            selection_window=L,
            switch_margin=0.5,
            **settings,
        )
        assert run["returns"].tobytes() == path["returns"].tobytes()
        evaluated = evaluate_allocation_rule(
            three_years, rule, n_trials=1, n_bootstrap=0, **settings
        )
        assert run["returns"].tobytes() == evaluated["rule_returns"].tobytes()

    def test_a_regime_rule_on_its_state_column_is_its_stream_too(self, three_years):
        """A candidate whose regime reads a state column walks a frame that
        holds the state; the selector's book, over the sleeves alone, is
        still that candidate's returns to the bit."""
        rng = np.random.default_rng(3)
        frame = three_years.copy()
        frame["state"] = np.abs(rng.standard_normal(len(frame))) * 0.1
        rule = SleeveRule(method="risk_parity", regime=RegimeSpec(state_column="state"))
        out = _select(frame, [("a", rule), ("b", rule)], combine="best")
        evaluated = evaluate_allocation_rule(
            frame, rule, n_trials=1, n_bootstrap=0, **SETTINGS
        )
        assert out["streams"]["selector"].to_numpy().tobytes() == (
            evaluated["rule_returns"].tobytes()
        )
        assert list(out["targets"].columns) == NAMES

    @pytest.mark.parametrize("combine,copies", [("best", 2), ("top_k", 3)])
    def test_copies_of_one_rule_are_that_rule(self, three_years, combine, copies):
        """Through the public function: two copies of one rule tie at every
        score, so the incumbent is never left; three copies blended two at a
        time hold (x + x) / 2 = x. Either way the selector's stream is
        evaluate_allocation_rule's rule stream bit for bit, and the scored
        part is the stream's tail. The blend's one switch is the move from
        the warm-up's single candidate to two."""
        rule = SleeveRule(method="min_variance")
        out = _select(
            three_years,
            [(f"c{i}", rule) for i in range(copies)],
            combine=combine,
            **({"top_k": 2} if combine == "top_k" else {}),
        )
        evaluated = evaluate_allocation_rule(
            three_years, rule, n_trials=1, n_bootstrap=0, **SETTINGS
        )
        stream = evaluated["rule_returns"]
        assert out["streams"]["selector"].to_numpy().tobytes() == stream.tobytes()
        start = out["n_warm_up"]
        assert out["selector_returns"].tobytes() == stream[start:].tobytes()
        assert out["n_switches"] == (1 if combine == "top_k" else 0)


class TestNothingAfterADateMovesWhatWasChosenBeforeIt:
    def test_changing_every_row_after_t_changes_nothing_at_or_before_t(self):
        """Every return after row 560 is changed. The selector's choices and
        targets at rebalances on or before that date, and every stream it
        reports -- its own, the benchmark's and each candidate's -- on or
        before it, are the same bits; after it they differ, so the test is
        not vacuous. A regime candidate is among the rules."""
        frame = _draw(np.random.default_rng(11), 3.5)
        cut = 560
        changed = frame.copy()
        noise = np.random.default_rng(12).normal(0.0, 0.01, (len(frame) - cut - 1, N))
        changed.iloc[cut + 1 :] = changed.iloc[cut + 1 :].to_numpy() * 1.5 + noise
        candidates = [
            ("rp", SleeveRule()),
            ("ms", SleeveRule(method="max_sharpe")),
            ("mv_regime", SleeveRule(method="min_variance", regime=RegimeSpec())),
        ]
        a = _select(frame, candidates, cash_rate=0.02)
        b = _select(changed, candidates, cash_rate=0.02)
        t = frame.index[cut]
        assert a["first_scored_date"] < t
        early_a = [c for c in a["choices"] if c[0] <= t]
        early_b = [c for c in b["choices"] if c[0] <= t]
        assert len(early_a) > 10 and early_a == early_b
        ta, tb = a["targets"].loc[:t], b["targets"].loc[:t]
        assert ta.to_numpy().tobytes() == tb.to_numpy().tobytes()
        before = a["streams"].index <= t
        for column in a["streams"].columns:
            x = a["streams"][column].to_numpy()
            y = b["streams"][column].to_numpy()
            assert x[before].tobytes() == y[before].tobytes(), column
            assert not np.array_equal(x[~before], y[~before]), column


class TestADominatingRuleIsFound:
    def test_it_is_held_at_every_rebalance_after_the_selection_window(self):
        """One sleeve earns 100% a year at 10% volatility, the others
        nothing. The first candidate puts almost no risk on it and the
        second almost all: until each has selection_window periods of record
        the selector holds the first; at the first scored rebalance it
        switches to the second -- one switch -- and holds it at every
        rebalance after."""
        frame = _draw(np.random.default_rng(5), 3, sharpe=(10.0, 0.0, 0.0, 0.0))
        small = {"S0": 0.01, "S1": 1.0, "S2": 1.0, "S3": 1.0}
        large = {"S0": 1.0, "S1": 0.01, "S2": 0.01, "S3": 0.01}
        out = _select(
            frame,
            [
                ("diffuse", SleeveRule(risk_budgets=small)),
                ("dominant", SleeveRule(risk_budgets=large)),
            ],
            combine="best",
        )
        first = W + math.ceil(L / K) * K
        assert out["first_scored_date"] == frame.index[first]
        assert out["n_warm_up"] == first - W
        held = [names for _d, names, scored in out["choices"] if not scored]
        picked = [names for _d, names, scored in out["choices"] if scored]
        assert held and all(names == ["diffuse"] for names in held)
        assert picked and all(names == ["dominant"] for names in picked)
        assert out["times_chosen"] == {"diffuse": 0, "dominant": len(picked)}
        assert out["n_switches"] == 1
        assert out["hindsight_best"] == "dominant"
        assert any("held the first candidate" in w for w in out["warnings"])


class TestTheChoice:
    def test_the_incumbent_is_left_only_past_the_margin(self):
        """A challenger 0.4 ahead does not displace the incumbent at a 0.5
        margin and does at 0.3; exactly at the margin it does not: the
        comparison is strict."""
        scores = np.array([1.0, 1.4, 0.2])
        assert _choose(scores, (0,), size=1, margin=0.5) == (0,)
        assert _choose(scores, (0,), size=1, margin=0.3) == (1,)
        assert _choose(np.array([1.0, 1.5]), (0,), size=1, margin=0.5) == (0,)

    def test_ties_go_to_the_earlier_and_an_undefined_score_never_wins(self):
        """Equal scores keep the earlier candidate; a NaN score is never
        brought in, and a held candidate whose score is NaN gives way to
        any defined one; when every score is NaN nothing moves."""
        assert _choose(np.array([0.5, 0.5]), (0,), size=1, margin=0.0) == (0,)
        assert _choose(np.array([np.nan, -3.0]), (0,), size=1, margin=1.0) == (1,)
        assert _choose(np.array([0.1, np.nan]), (0,), size=1, margin=0.0) == (0,)
        assert _choose(np.array([np.nan, np.nan]), (1,), size=1, margin=0.0) == (1,)

    def test_top_k_fills_then_swaps_the_worst_member(self):
        """From a single incumbent the set is filled with the best outsiders
        at no margin; after that its worst member is swapped for the best
        outsider only past the margin, as many times as that holds."""
        scores = np.array([0.0, 2.0, 1.0, 1.9])
        assert _choose(scores, (0,), size=2, margin=5.0) == (0, 1)
        assert _choose(scores, (0,), size=2, margin=0.5) == (1, 3)
        assert _choose(scores, (1, 2), size=2, margin=1.0) == (1, 2)
        assert _choose(scores, (1, 2), size=2, margin=0.5) == (1, 3)

    def test_the_default_blend_is_the_better_half(self):
        """k = n // 2, never below 1: two or three candidates hold the best."""
        assert [default_top_k(n) for n in (2, 3, 4, 6, 7, 12)] == [1, 1, 2, 3, 3, 6]


class TestTheBookByHand:
    def test_the_selector_matches_a_recursion_in_currency(self, three_years):
        """The selector's book kept in currency from the candidates' own
        holdings: at each rebalance the cost (cost_bps of the traded value)
        leaves the account, the rest is put on the chosen candidates'
        average book; between rebalances holdings grow with their sleeves
        and cash with the cash rate. The selector's returns agree to 1e-12."""
        candidates = [
            ("rp", SleeveRule()),
            ("ms", SleeveRule(method="max_sharpe")),
            ("iv", SleeveRule(method="inverse_volatility")),
            ("mv", SleeveRule(method="min_variance")),
        ]
        cash_rate, cost = 0.03, 25.0
        walks = walk_candidates(
            three_years,
            candidates,
            estimation_window=W,
            rebalance_every=K,
            cost_bps=cost,
            cash_rate=cash_rate,
        )
        values = three_years.to_numpy()
        run = run_selector(
            values,
            walks["paths"],
            estimation_window=W,
            rebalance_every=K,
            selection_window=L,
            switch_margin=0.25,
            size=2,
            cost_bps=cost,
            cash_rate=cash_rate,
        )
        assert len(set(run["choices"])) > 1  # it switched: not one rule's book
        rf = cash_rate / P
        value, holdings, cash = 1.0, np.zeros(N), 1.0
        expected = []
        for t in range(W, len(values)):
            before = value
            if (t - W) % K == 0:
                i = (t - W) // K
                chosen = run["choices"][i]
                weights = np.mean(
                    [walks["paths"][j]["held"].iloc[i].to_numpy() for j in chosen], 0
                )
                book_cash = np.mean([walks["paths"][j]["held_cash"][i] for j in chosen])
                value -= cost / 1e4 * np.abs(weights * value - holdings).sum()
                holdings, cash = weights * value, book_cash * value
            holdings = holdings * (1.0 + values[t])
            cash = cash * (1.0 + rf)
            value = holdings.sum() + cash
            expected.append(value / before - 1.0)
        np.testing.assert_allclose(run["returns"], expected, rtol=0, atol=1e-12)


class TestWhatIsReported:
    @pytest.fixture(scope="class")
    def report(self):
        frame = _draw(np.random.default_rng(21), 3.5)
        candidates = [
            ("eq", SleeveRule(method="equal")),
            ("rp", SleeveRule()),
            ("ms", SleeveRule(method="max_sharpe")),
            ("mv", SleeveRule(method="min_variance")),
        ]
        return _select(frame, candidates, n_trials=3, cash_rate=0.01, n_bootstrap=200)

    def test_the_hindsight_pick_is_the_best_scored_stream(self, report):
        """hindsight_best is the candidate whose scored returns have the
        highest Sharpe ratio over the cash rate, computed here from the
        returns the result reports; hindsight_gap is that minus the
        selector's Sharpe ratio."""
        rf = 0.01 / P
        sharpes = {
            name: annualized_sharpe(
                report["candidate_returns"][name].to_numpy() - rf, P
            )
            for name in report["names"]
        }
        best = max(sharpes, key=sharpes.get)
        assert report["hindsight_best"] == best
        selector = annualized_sharpe(report["selector_returns"] - rf, P)
        assert report["hindsight_gap"] == pytest.approx(
            sharpes[best] - selector, abs=1e-12
        )

    def test_the_pbo_and_the_deflated_ratios_are_the_librarys(self, report):
        """The PBO is the CSCV function on the candidates' scored excess
        returns at eight splits; the selector's deflated Sharpe ratio is at
        n_trials, the hindsight winner's at n_trials times the candidates."""
        rf = 0.01 / P
        pbo = probability_of_backtest_overfitting(
            report["candidate_returns"] - rf, n_splits=8
        )
        assert report["pbo"]["pbo"] == pbo["pbo"]
        dsr = deflated_sharpe_ratio(
            pd.Series(report["selector_returns"] - rf), n_trials=3, periods_per_year=P
        )
        assert report["deflated"]["deflated_sharpe_probability"] == (
            dsr["deflated_sharpe_probability"]
        )
        assert report["hindsight_deflated"]["n_trials"] == 3 * 4

    def test_the_counts_add_up(self, report):
        """Every scored rebalance holds top_k candidates (half of four), so
        the times chosen sum to twice the selections."""
        assert report["top_k"] == 2 and report["combine"] == "top_k"
        assert sum(report["times_chosen"].values()) == 2 * report["n_selections"]
        assert report["n_out_of_sample"] == len(report["selector_returns"])
        assert report["selector"]["n_rebalances"] == report["n_selections"]


class TestNoEdgeIsNotClaimed:
    @pytest.mark.slow
    def test_on_exchangeable_sleeves_the_selector_does_not_beat_equal_weight(self):
        """Four sleeves with one volatility, one correlation and one Sharpe
        ratio: equal weight is the best allocation there is, so any edge the
        selector shows is luck. Over 12 histories its 95% interval against
        equal weight clears zero at most twice, and the probability of
        backtest overfitting over the candidates averages near one half."""
        rng = np.random.default_rng(31)
        clears, pbos = 0, []
        candidates = [
            ("eq", SleeveRule(method="equal")),
            ("rp", SleeveRule()),
            ("ms", SleeveRule(method="max_sharpe")),
            ("mv", SleeveRule(method="min_variance")),
        ]
        for _ in range(12):
            frame = _draw(rng, 4, sharpe=(0.5,) * N)
            out = _select(frame, candidates, n_bootstrap=300)
            clears += out["sharpe_difference"]["lower"] > 0
            pbos.append(out["pbo"]["pbo"])
        assert clears <= 2
        assert 0.3 <= np.mean(pbos) <= 0.75


class TestRefusals:
    def test_by_name(self, three_years):
        """One candidate, a repeated name, a reserved name, a selection
        window that leaves fewer than twenty periods to score, a top_k that
        blends everything, a top_k with combine='best', and a regime state
        column that is not in the returns: each refused, naming it."""
        rp = SleeveRule()
        cases = [
            ([("a", rp)], {}, "at least 2"),
            ([("a", rp), ("a", rp)], {}, "repeated"),
            ([("a", rp), ("selector", rp)], {}, "reserved"),
            ([("a", rp), ("b", rp)], {"selection_window": 700}, "selection_window=700"),
            ([("a", rp), ("b", rp)], {"top_k": 2}, "top_k=2"),
            ([("a", rp), ("b", rp)], {"combine": "best", "top_k": 1}, "top_k=1"),
            (
                [("a", rp), ("b", SleeveRule(regime=RegimeSpec(state_column="vix")))],
                {},
                "state_column='vix'",
            ),
        ]
        for candidates, kwargs, phrase in cases:
            with pytest.raises(ValidationError, match=phrase):
                _select(three_years, candidates, **kwargs)

    def test_the_tool_refuses_at_the_boundary(self):
        """The tool's input model refuses a single candidate, two candidates
        given one name, a given name that collides with a numbered default,
        top_k with combine='best', and a candidate's capital, before any
        data is read. Unnamed rules are named by their method, numbered
        from the second of a method on."""
        from standard_quant_tools.agent.runtimes.allocation.selection_tools import (
            SelectAllocationRuleInput,
        )

        base = {"returns_ref": "ref://x", "n_trials": 1}
        named = SelectAllocationRuleInput(
            candidates=[
                {"method": "risk_parity"},
                {"method": "equal"},
                {"method": "risk_parity", "covariance_method": "sample"},
            ],
            **base,
        )
        assert named.names() == ["risk_parity", "equal", "risk_parity_2"]
        for candidates, extra, phrase in [
            ([{"method": "equal"}], {}, "at least 2"),
            ([{"method": "equal", "name": "a"}, {"name": "a"}], {}, "repeated"),
            (
                [{"method": "hrp"}, {"method": "hrp"}, {"name": "hrp_2"}],
                {},
                "repeated",
            ),
            (
                [{"method": "equal"}, {"method": "hrp"}],
                {"combine": "best", "top_k": 1},
                "combine='top_k' only",
            ),
            (
                [{"method": "equal"}, {"method": "hrp", "capital": 1e6}],
                {},
                "capital",
            ),
        ]:
            with pytest.raises(pydantic.ValidationError, match=phrase):
                SelectAllocationRuleInput(candidates=candidates, **base, **extra)


class TestTheTool:
    def test_end_to_end_with_published_streams(self, three_years, runs_dir):
        """Through the runtime: the result names the candidates, the warm-up
        and the scored periods, and run_id publishes the streams -- the
        selector, the benchmark and each candidate from the first fit on --
        and the selector's weights at every rebalance."""
        ref = handoff.publish(three_years, kind="returns_panel", run_id="in", name="p")
        out = resolve("allocation").dispatch(
            "select_allocation_rule",
            {
                "returns_ref": ref,
                "candidates": [
                    {"method": "equal"},
                    {"method": "risk_parity"},
                    {"method": "min_variance", "name": "mv"},
                    {"method": "max_sharpe", "regime": {"labels": "quantile"}},
                ],
                "n_trials": 2,
                "estimation_window": W,
                "selection_window": L,
                "cost_bps": 10,
                "n_bootstrap": 100,
                "run_id": "sel",
            },
        )
        assert out["candidates_tried"] == ["equal", "risk_parity", "mv", "max_sharpe"]
        assert out["top_k"] == 2
        assert out["n_out_of_sample"] + out["n_warm_up"] == len(three_years) - W
        streams = handoff.resolve(out["oos_returns_ref"], expect="returns_panel")
        assert list(streams.columns) == [
            "selector",
            "benchmark",
            *out["candidates_tried"],
        ]
        assert len(streams) == len(three_years) - W
        weights = pd.DataFrame(
            handoff.resolve(out["weights_ref"], expect="weight_panel")
        )
        assert len(weights) == math.ceil((len(three_years) - W) / K)
        assert list(weights.columns) == NAMES
        assert set(out["candidates"]) == set(out["candidates_tried"])
        assert out["hindsight_best"] in out["candidates_tried"]
