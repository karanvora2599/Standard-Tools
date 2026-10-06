"""
The portfolio that was backtested can be built at the live door.

`apply_exposure_targets` — exact gross AND net, a per-name cap with
iterative redistribution — lived only inside the simulation path, which
needs at least two rebalance dates. `construct_weights_from_scores`, the
door a deployment goes through, offered `gross_leverage` and a boolean
`dollar_neutral`: no cap, no net target, no redistribution.

So a model backtested at `max_position_weight=0.05` deployed through a path
that could hold far more than 0.05 in one name. Two different portfolios,
both reported as "the model", both well-formed, and nothing said so.

It was already a per-DATE function. The live door only ever had to call it.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.portfolio.weight_tools import (
    ConstructWeightsInput,
    construct_weights_from_scores,
)
from standard_quant_tools.error import ValidationError

DATES = pd.date_range("2024-01-01", periods=3, freq="B")
NAMES = [f"N{i}" for i in range(6)]


@pytest.fixture
def scores_ref(tmp_path, monkeypatch):
    """One name dominates, so a cap has something to do."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
    panel = pd.DataFrame(
        [[10.0, 2.0, 1.0, -1.0, -2.0, -3.0]] * len(DATES),
        index=DATES,
        columns=NAMES,
    )
    return handoff.publish(
        panel, "score_panel", "weights_test", "scores", overwrite=True
    )


def _build(scores_ref, name, **kwargs):
    return construct_weights_from_scores(
        ConstructWeightsInput(
            scores_ref=scores_ref,
            method="zscore",
            gross_leverage=1.0,
            run_id="weights_test",
            name=name,
            **kwargs,
        )
    )


class TestTheCapIsHonoured:
    def test_without_it_one_name_can_run_away(self, scores_ref):
        """The state of affairs this closes: the door had no parameter
        for it, so a backtest's cap did not survive deployment."""
        built = _build(scores_ref, "uncapped")
        assert built.max_weight > 0.25

    def test_with_it_no_name_exceeds_the_cap(self, scores_ref):
        built = _build(scores_ref, "capped", max_position_weight=0.20)
        assert built.max_weight <= 0.20 + 1e-9
        assert abs(built.min_weight) <= 0.20 + 1e-9

    def test_the_excess_is_redistributed_not_dropped(self, scores_ref):
        """A cap that simply truncated would shrink the book. The names
        still under the cap absorb it."""
        built = _build(scores_ref, "redistributed", max_position_weight=0.25)
        assert built.gross_leverage == pytest.approx(1.0, abs=1e-6)


class TestBothTargetsAtOnce:
    def test_net_and_gross_are_hit_exactly(self, scores_ref):
        """A single rescale cannot control two targets; the long and short
        books are sized independently so both hold."""
        built = _build(scores_ref, "net_zero", net_exposure=0.0)
        assert built.gross_leverage == pytest.approx(1.0, abs=1e-6)
        assert built.net_exposure == pytest.approx(0.0, abs=1e-6)

    def test_a_tilted_book_is_also_exact(self, scores_ref):
        built = _build(scores_ref, "net_tilt", net_exposure=0.2)
        assert built.gross_leverage == pytest.approx(1.0, abs=1e-6)
        assert built.net_exposure == pytest.approx(0.2, abs=1e-6)

    def test_net_beyond_gross_is_refused(self, scores_ref):
        with pytest.raises(ValidationError, match="more net than it is gross"):
            _build(scores_ref, "impossible", net_exposure=2.0)


class TestItSaysWhenItCouldNotGetThere:
    def test_an_unfillable_half_is_reported_as_a_net_miss(self, tmp_path, monkeypatch):
        """A book built for net 0 that cannot fill one half is not
        market-neutral, whatever it was asked for — and the gross number
        alone does not say so."""
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        # All-positive scores: z-scoring centres them, but a tight cap on
        # few names leaves one half unable to reach its target.
        panel = pd.DataFrame(
            [[5.0, 4.9, 4.8, 4.7]] * 2,
            index=DATES[:2],
            columns=NAMES[:4],
        )
        ref = handoff.publish(
            panel, "score_panel", "weights_miss", "scores", overwrite=True
        )
        built = construct_weights_from_scores(
            ConstructWeightsInput(
                scores_ref=ref,
                method="zscore",
                gross_leverage=1.0,
                net_exposure=0.0,
                max_position_weight=0.05,
                run_id="weights_miss",
                name="tight",
            )
        )
        assert any("requested gross" in w for w in built.warnings)
        assert any("NOT market-neutral" in w for w in built.warnings)

    def test_a_book_that_reaches_its_targets_is_not_warned_about(self, scores_ref):
        built = _build(scores_ref, "clean", net_exposure=0.0)
        assert not any("requested gross" in w for w in built.warnings)


class TestTheTwoWaysToSayNeutral:
    def test_they_cannot_be_combined(self, scores_ref):
        """Two ways to set the same thing, where the second would silently
        undo the first."""
        with pytest.raises(ValidationError, match="two ways to set the same"):
            _build(
                scores_ref, "both", dollar_neutral=True, net_exposure=0.0
            )

    def test_dollar_neutral_still_works_and_says_what_it_costs(self, scores_ref):
        built = _build(scores_ref, "shifted", dollar_neutral=True)
        assert any("preserves ordering, not scale" in w for w in built.warnings)
        assert any("net_exposure=0 hits both targets" in w for w in built.warnings)

    def test_net_exposure_zero_is_the_stronger_one(self, scores_ref):
        """The difference, measured: the shift leaves gross adrift and the
        target hits both."""
        shifted = _build(scores_ref, "shift_cmp", dollar_neutral=True)
        targeted = _build(scores_ref, "target_cmp", net_exposure=0.0)
        assert targeted.gross_leverage == pytest.approx(1.0, abs=1e-6)
        assert shifted.net_exposure == pytest.approx(0.0, abs=1e-6)
        assert targeted.net_exposure == pytest.approx(0.0, abs=1e-6)


class TestNothingChangesWhenNothingIsAsked:
    def test_the_old_call_is_untouched(self, scores_ref):
        """Every existing caller passes neither, and must behave as before."""
        plain = _build(scores_ref, "plain")
        assert plain.gross_leverage == pytest.approx(1.0, abs=1e-6)
        assert not any("requested gross" in w for w in plain.warnings)

    def test_the_published_panel_still_resolves(self, scores_ref):
        built = _build(scores_ref, "published", net_exposure=0.0)
        panel = handoff.resolve(built.ref, expect="weight_panel")
        assert set(panel) <= set(NAMES)
        assert np.isfinite(
            [v for series in panel.values() for v in series.values()]
        ).all()
