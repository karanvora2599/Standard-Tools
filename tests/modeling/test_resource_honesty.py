"""
Knowing the cost before paying it.

The performance picture here is the opposite of the usual: this is
well-measured code and the obvious optimisations are already done, several
refused on record. What was missing is not speed.

`ComputeBudgetSpec` counted fits and nothing else. Columns ARE bounded —
`MAX_EXPANDED_COLUMNS = 400`, because an agent "would otherwise discover the
cost as a memory error rather than as a refusal" — and rows, the other half
of the product that determines memory, had no bound at all. At 1-minute
bars, 500 names over 10 years is about 500 million rows.

`FoldCache` never evicted: every completed fold's matrices stayed resident
until the run ended, though the key is content-hashed per fold so nothing
could ever look them up again.

`QuantileTransform.fit` was quadratic in `n_quantiles`.

SECONDS ARE DELIBERATELY NOT CLAIMED, and one test below pins that.
`EstimatorCost.fit_cost` is a class read off one fit, not a duration;
multiplying it by a fit count would manufacture a measurement nobody made.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.cache import FoldCache
from standard_quant_tools.modeling.plan import plan_experiment
from standard_quant_tools.modeling.preprocessing.base import FoldContext
from standard_quant_tools.modeling.preprocessing.steps import QuantileTransform
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)

DATES = pd.Index(pd.bdate_range("2022-01-03", periods=400))


def _panel():
    return pd.DataFrame(
        {
            "date": np.repeat(DATES.to_numpy(), 3),
            "entity": ["A", "B", "C"] * len(DATES),
            "f1": 1.0,
            "f2": 2.0,
            "target": 0.0,
        }
    )


def _spec(estimator="ridge", **budget):
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type=estimator),
        validation=ValidationSpec(
            method="walk_forward", train_window=120, test_window=40, min_folds=1
        ),
        budget=ComputeBudgetSpec(**budget),
        random_seed=0,
    )


def _plan(**budget):
    return plan_experiment(
        _spec(**budget), DATES, panel=_panel(), feature_ids=["f1", "f2"]
    )


class TestThePlanIsPricedInMemory:
    def test_rows_columns_and_bytes_are_reported(self):
        reported = _plan().to_dict()
        assert reported["n_panel_rows"] == 3 * len(DATES)
        assert reported["n_columns"] == 2
        # Arithmetic over numbers already in the plan, not an estimate of
        # anything unmeasured: rows x columns x 8.
        assert reported["max_fold_bytes"] > 0

    def test_a_row_bound_refuses_before_anything_is_built(self):
        with pytest.raises(ValidationError, match="over budget.max_panel_rows"):
            _plan(max_panel_rows=100).refuse_over_budget("demo")

    def test_a_byte_bound_refuses_and_names_the_shape(self):
        with pytest.raises(ValidationError, match="over budget.max_fold_bytes"):
            _plan(max_fold_bytes=1000).refuse_over_budget("demo")

    def test_memory_is_refused_on_its_own_terms(self):
        """Not through a fit count that may be perfectly modest. The two
        ceilings answer different questions, and the wrong message sends
        the caller to shrink the wrong thing."""
        plan = _plan(max_panel_rows=100)
        assert plan.n_fits <= plan.max_fits, "the fit count must be fine here"
        with pytest.raises(ValidationError) as caught:
            plan.refuse_over_budget("demo")
        assert "estimator fits" not in str(caught.value)

    def test_no_bound_is_the_old_behaviour(self):
        plan = _plan()
        assert plan.max_panel_rows is None
        assert plan.within_budget is True


class TestSecondsAreNotClaimed:
    def test_the_cost_class_is_carried_as_a_class(self):
        """`fit_cost` is 'low'/'medium'/'high', read off one fit. Reported
        beside the fit count so a reader sees "8 fits of a high-cost
        estimator" — which is the honest form of pricing a plan."""
        assert _plan().to_dict()["fit_cost"] == "low"
        cheap = _plan().to_dict()
        dear = plan_experiment(
            _spec("random_forest"), DATES, panel=_panel(), feature_ids=["f1", "f2"]
        ).to_dict()
        assert dear["fit_cost"] == "high"
        assert cheap["n_fits"] == dear["n_fits"]

    def test_there_is_no_seconds_field_to_believe(self):
        """If one is ever added it must be measured, not multiplied out of
        a class. This test exists to make that a decision rather than a
        drift."""
        reported = _plan().to_dict()
        assert not [k for k in reported if "second" in k or "duration" in k]
        assert not hasattr(ComputeBudgetSpec, "max_seconds")


class TestTheCacheEvicts:
    def test_dropping_a_key_forgets_its_entries(self):
        cache = FoldCache()
        frame = pd.DataFrame({"a": [1.0]})
        cache.store("k1", ["a"], frame, frame, projectable=True)
        cache.store("k2", ["a"], frame, frame, projectable=True)
        assert len(cache) == 2
        assert cache.drop("k1") == 1
        assert len(cache) == 1
        assert cache.lookup("k1", ["a"]) is None
        assert cache.lookup("k2", ["a"]) is not None

    def test_the_projection_entry_goes_too(self):
        """Or a dropped key would still serve a projection of itself."""
        cache = FoldCache()
        frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
        cache.store("k", ["a", "b"], frame, frame, projectable=True)
        cache.drop("k")
        assert cache.lookup("k", ["a"]) is None

    def test_dropping_an_unknown_key_is_not_an_error(self):
        assert FoldCache().drop("nothing") == 0

    def test_the_counters_describe_the_run_not_the_contents(self):
        """Resetting them on eviction would make a run that evicted look
        like a run that never looked."""
        cache = FoldCache()
        frame = pd.DataFrame({"a": [1.0]})
        cache.store("k", ["a"], frame, frame, projectable=True)
        cache.lookup("k", ["a"])
        cache.drop("k")
        assert cache.stats()["hits"] == 1
        assert cache.stats()["entries"] == 0


class TestTheQuantileKnots:
    def test_the_knots_are_what_they_always_were(self):
        """The fast form must agree exactly: `np.unique(return_inverse)`
        plus `bincount` groups the same values and means the same grid."""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"f": rng.normal(size=5000)})
        step = QuantileTransform(n_quantiles=1000)
        state = step.fit(frame, FoldContext(dates=None))

        values = frame["f"].to_numpy(dtype=float)
        grid = np.linspace(0.0, 1.0, min(1000, values.size))
        raw = np.quantile(values, grid)
        unique = np.unique(raw)
        reference = np.array([grid[raw == v].mean() for v in unique])

        assert state["quantiles"]["f"] == pytest.approx([float(v) for v in unique])
        assert state["probabilities"]["f"] == pytest.approx(
            [float(p) for p in reference]
        )

    def test_ties_still_collapse_to_one_knot(self):
        """The reason the grouping exists: np.interp needs a strictly
        increasing axis."""
        frame = pd.DataFrame({"f": [1.0] * 50 + [2.0] * 50})
        state = QuantileTransform(n_quantiles=100).fit(
            frame, FoldContext(dates=None)
        )
        knots = state["quantiles"]["f"]
        assert len(knots) == len(set(knots))
        assert knots == sorted(knots)
