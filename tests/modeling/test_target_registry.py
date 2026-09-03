"""
What a label IS, said once.

WHY THIS FILE EXISTS. The target set was written four times and the task
set five, and in both cases the copies drifted: `ranking` was missing from
two task literals, so a ranker could be trained and never traded. A
registry does not prevent that by itself — what prevents it is that every
consumer READS the registry and a test pins the hand-written Literal equal
to it.

The second thing pinned here is subtler. A microstructure label is not a
function of closing prices, so `build_target` must refuse it rather than
approximate it. A fill probability derived from daily bars would be a
number with nothing behind it, and it would look exactly like a number with
something behind it.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.target import build_target
from standard_quant_tools.modeling.specs import (
    EXTERNAL_TARGETS,
    TARGET_KINDS,
    TASKS,
    TargetSpec,
    TargetType,
    targets_for_task,
)

CLOSE = pd.Series(
    100.0 + np.cumsum(np.random.default_rng(3).normal(0, 0.5, 200)),
    index=pd.date_range("2024-01-01", periods=200, freq="B"),
)


class TestTheRegistryIsTheSourceOfTruth:
    def test_the_literal_matches_the_registry(self) -> None:
        """
        A Literal cannot be built from a dict at type-check time, so it is
        written out — and this is what stops the two versions of the same
        list from drifting, which is exactly how `ranking` went missing
        from two task literals.
        """
        assert set(typing.get_args(TargetType)) == set(TARGET_KINDS)

    def test_the_spec_field_uses_it(self) -> None:
        members = typing.get_args(TargetSpec.model_fields["type"].annotation)
        assert set(members) == set(TARGET_KINDS)

    def test_every_target_names_at_least_one_task(self) -> None:
        orphans = [name for name, k in TARGET_KINDS.items() if not k.tasks]
        assert not orphans, (
            f"{orphans} can be declared and never fitted: no task consumes "
            "them, so the compatibility check refuses every model"
        )

    def test_every_named_task_is_a_real_task(self) -> None:
        for name, kind in TARGET_KINDS.items():
            unknown = [t for t in kind.tasks if t not in TASKS]
            assert not unknown, f"{name} names task(s) {unknown} that do not exist"

    def test_every_task_can_consume_something(self) -> None:
        for task in TASKS:
            assert targets_for_task(task), f"{task} has no target it can be fitted on"

    def test_every_target_has_a_description(self) -> None:
        thin = [n for n, k in TARGET_KINDS.items() if len(k.description) < 30]
        assert not thin, f"{thin} carry a description too short to be one"


class TestABuildableTargetCannotFallThrough:
    """
    `build_target` used to end in a bare `else` producing a direction
    target. Any type added to the Literal and forgotten there came back
    silently BINARIZED — a continuous label arriving as 1.0/0.0 with
    nothing raising.
    """

    @pytest.mark.parametrize(
        "name", [n for n, k in TARGET_KINDS.items() if k.buildable]
    )
    def test_each_buildable_target_has_a_branch(self, name: str) -> None:
        spec = TargetSpec(type=name, horizon=5)
        series = build_target(CLOSE, spec)
        assert isinstance(series, pd.Series)
        assert len(series) == len(CLOSE)

    def test_a_continuous_target_is_not_binarized(self) -> None:
        series = build_target(
            CLOSE, TargetSpec(type="forward_return", horizon=5)
        ).dropna()
        assert not set(series.unique()) <= {0.0, 1.0}, (
            "a forward return came back as only 0.0 and 1.0, which is what "
            "the removed `else` branch did to anything it did not recognize"
        )


class TestAnExternalLabelIsRefusedRatherThanApproximated:
    @pytest.mark.parametrize("name", EXTERNAL_TARGETS)
    def test_it_cannot_be_built_from_prices(self, name: str) -> None:
        with pytest.raises(ValidationError, match="cannot be built from a price"):
            build_target(CLOSE, TargetSpec(type=name, horizon=5))

    @pytest.mark.parametrize("name", EXTERNAL_TARGETS)
    def test_the_refusal_says_what_to_do_instead(self, name: str) -> None:
        with pytest.raises(ValidationError) as caught:
            build_target(CLOSE, TargetSpec(type=name, horizon=5))
        assert "register_external_panel" in str(caught.value)

    def test_the_execution_labels_are_all_external(self) -> None:
        """A fill probability needs queue position and cancellations. No
        Close column contains either."""
        for name in (
            "fill_probability",
            "time_to_fill",
            "adverse_selection",
            "future_markout",
        ):
            assert name in EXTERNAL_TARGETS
            assert not TARGET_KINDS[name].buildable

    def test_a_price_label_is_still_buildable(self) -> None:
        assert TARGET_KINDS["forward_return"].buildable
        assert "forward_return" not in EXTERNAL_TARGETS


class TestTheCompatibilityMapIsDerived:
    """
    It was three hand-written sets, two of which were copies of each other
    kept in sync by hand. A label added anywhere had to be remembered here
    too, in a different file.
    """

    @pytest.mark.parametrize("name", sorted(TARGET_KINDS))
    def test_each_target_is_accepted_by_the_tasks_it_names(self, name: str) -> None:
        from standard_quant_tools.modeling.engine import (
            _check_task_target_compatibility,
        )

        for task in TARGET_KINDS[name].tasks:
            _check_task_target_compatibility(task, f"{name}:5")

    @pytest.mark.parametrize("name", sorted(TARGET_KINDS))
    def test_each_target_is_refused_by_the_tasks_it_does_not(self, name: str) -> None:
        from standard_quant_tools.modeling.engine import (
            _check_task_target_compatibility,
        )

        for task in TASKS:
            if task in TARGET_KINDS[name].tasks:
                continue
            with pytest.raises(ValidationError, match="expects one of"):
                _check_task_target_compatibility(task, f"{name}:5")

    def test_a_probability_is_not_a_regression_target(self) -> None:
        """The specific confusion the check exists for: a regressor fitted
        on a 0/1 label does not error, it reports a meaningless R2."""
        from standard_quant_tools.modeling.engine import (
            _check_task_target_compatibility,
        )

        with pytest.raises(ValidationError):
            _check_task_target_compatibility("regression", "fill_probability:5")


class TestTheThresholdRuleReadsTheRegistry:
    def test_a_continuous_label_refuses_a_threshold(self) -> None:
        with pytest.raises(Exception, match="continuous"):
            TargetSpec(type="future_markout", horizon=5, threshold=0.01)

    def test_a_discrete_label_accepts_one(self) -> None:
        assert TargetSpec(type="forward_direction", horizon=5, threshold=0.01)

    def test_every_continuous_label_is_covered(self) -> None:
        """The set used to be inlined and listed four of the six types then
        in existence, so a new continuous label was silently allowed a
        threshold that means nothing for it."""
        for name, kind in TARGET_KINDS.items():
            if not kind.continuous:
                continue
            with pytest.raises(Exception):
                TargetSpec(type=name, horizon=5, threshold=0.01)
