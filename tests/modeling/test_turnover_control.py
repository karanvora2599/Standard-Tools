"""
Turnover was measurable and not controllable.

`mean_turnover_pct` and `annualized_turnover` are reported AFTER the
simulation, and nothing in the transform moved them. So "this model wins on
IC and loses on turnover-adjusted economics" was a sentence the library
could say and not act on: the only lever was to change the signal and run it
again.

The damping is a blend toward the previous book — every name moves the same
fraction of its own distance, so the target's ORDERING survives exactly and
what is given up is the speed of adjustment, not the signal. The first test
is that ordering property; without it this would be a different portfolio
rather than a slower one.

WHAT IT COSTS IS REPORTED. A damped row lies between two books that each hit
gross and net exactly, so it hits neither. Re-applying the targets would
restore them and change the turnover again — a loop with no fixed point
worth claiming — so the exposures are re-measured from the rows that will
actually be traded and the drift is visible.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.portfolio_eval import (
    transform_predictions_to_weights,
)
from standard_quant_tools.modeling.specs import PredictionTransformSpec

DATES = pd.date_range("2024-01-01", periods=40, freq="B")
NAMES = [f"N{i}" for i in range(10)]


@pytest.fixture(scope="module")
def churning_scores():
    """Signs that flip often, so an undamped book churns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(len(DATES), len(NAMES))), index=DATES, columns=NAMES)


def _weights(scores, cap=None):
    spec = PredictionTransformSpec(
        method="cross_sectional_zscore",
        gross_exposure=1.0,
        net_exposure=0.0,
        max_turnover=cap,
    )
    return transform_predictions_to_weights(scores, spec, None)


def _turnover(weights):
    return np.abs(np.diff(weights.to_numpy(), axis=0)).sum(axis=1)


class TestTheCapHolds:
    def test_no_rebalance_exceeds_it(self, churning_scores):
        damped, _ = _weights(churning_scores, cap=0.5)
        assert _turnover(damped).max() <= 0.5 + 1e-9

    def test_the_undamped_book_does_exceed_it(self, churning_scores):
        """Or the cap would be doing nothing and the test would be
        vacuous."""
        plain, _ = _weights(churning_scores)
        assert _turnover(plain).max() > 0.5

    def test_a_cap_above_the_natural_turnover_changes_nothing(
        self, churning_scores
    ):
        plain, _ = _weights(churning_scores)
        loose, diagnostics = _weights(churning_scores, cap=100.0)
        assert np.allclose(plain.to_numpy(), loose.to_numpy(), equal_nan=True)
        assert diagnostics["n_dates_damped"] == 0


class TestTheSignalSurvives:
    def test_the_ordering_of_a_damped_row_is_the_targets_ordering(
        self, churning_scores
    ):
        """Every name moves the same fraction of its own distance, so a
        damped book is the same book earlier in its journey — not a
        different one.

        The ordering compared is of the MOVE, which is what the blend
        scales; the level is the previous row plus that move.
        """
        plain, _ = _weights(churning_scores)
        damped, _ = _weights(churning_scores, cap=0.5)
        plain_moves = np.diff(plain.to_numpy(), axis=0)
        damped_moves = np.diff(damped.to_numpy(), axis=0)
        # On the first rebalance both start from the same place (flat), so
        # the two moves differ only by a positive scalar.
        first_plain, first_damped = plain_moves[0], damped_moves[0]
        assert np.argsort(first_plain).tolist() == np.argsort(first_damped).tolist()

    def test_a_damped_run_still_holds_both_sides(self, churning_scores):
        damped, _ = _weights(churning_scores, cap=0.5)
        last = damped.iloc[-1]
        assert (last > 0).any() and (last < 0).any()


class TestTheCostIsReported:
    def test_the_damped_dates_are_counted(self, churning_scores):
        _, diagnostics = _weights(churning_scores, cap=0.5)
        assert diagnostics["n_dates_damped"] > 0
        assert diagnostics["max_turnover"] == 0.5

    def test_the_exposures_are_measured_from_the_rows_that_will_trade(
        self, churning_scores
    ):
        """A damped row lies between two books that each hit the targets,
        so it hits neither — and the diagnostics must say what the book
        actually is, not what it was aiming at."""
        damped, diagnostics = _weights(churning_scores, cap=0.5)
        measured = float(np.abs(damped.to_numpy()).sum(axis=1).mean())
        assert diagnostics["mean_realized_gross"] == pytest.approx(
            measured, rel=1e-6
        )

    def test_nothing_is_claimed_when_nothing_was_asked(self, churning_scores):
        """None is not zero: a run that damped nothing and a run that was
        never asked to damp are different."""
        _, diagnostics = _weights(churning_scores)
        assert diagnostics["max_turnover"] is None
        assert diagnostics["n_dates_damped"] is None

    def test_the_first_rebalance_is_never_damped(self, churning_scores):
        """There is nothing to move from — the book starts flat, and
        damping the entry would hold a fraction of the intended book
        forever rather than reaching it."""
        damped, _ = _weights(churning_scores, cap=0.01)
        assert np.abs(damped.iloc[0].to_numpy()).sum() > 0.01
