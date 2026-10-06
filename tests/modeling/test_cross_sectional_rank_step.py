"""
A feature can be ranked within its date, the way the label already is.

`rank_within_date` existed, kernel-backed, with three callers — the feature
report, the ensemble combiner, and the rank TARGET — and zero callers in
`preprocessing/`. So the library made the argument on the label side,
`forward_return_rank` being "immune to a fat-tailed return distribution",
and the identical argument on the feature side was inexpressible.

The measured failure is on record in `Documentation/15_modeling.md`: a
feature whose cross-sectional standard deviation ran 0.23 to 22.4, where "a
rank of it sorted names by price level as much as by momentum".

The mapping is the LABEL's — `(rank - 1) / (n - 1) - 0.5` — and the first
test is the one that matters: feature and target come out on the same scale,
because a coefficient between two differently-ranked quantities means
nothing.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.preprocessing.base import FoldContext
from standard_quant_tools.modeling.preprocessing.registry import (
    PREPROCESSOR_REGISTRY,
)
from standard_quant_tools.modeling.preprocessing.steps import CrossSectionalRank

STEP = CrossSectionalRank()


def _dates(*counts):
    """One date per count, that many rows each."""
    out = []
    for index, count in enumerate(counts):
        out.extend([np.datetime64(f"2024-01-{index + 1:02d}")] * count)
    return np.array(out, dtype="datetime64[ns]")


def _rank(values, dates):
    return STEP.transform(
        pd.DataFrame({"f": values}), {}, FoldContext(dates=dates)
    )["f"]


class TestItAgreesWithTheLabel:
    def test_the_mapping_is_the_rank_targets_own(self):
        """Not a new convention. `_stage_rank` maps a return's rank the
        same way so the target is "symmetric and scale-free regardless of
        how many entities are present that day"; a feature ranked any
        other way would not line up with the label it is fitted against.
        """
        from standard_quant_tools.modeling.targets.builtin import _stage_rank

        values = [10.0, 20.0, 30.0, 1000.0]
        dates = _dates(4)
        panel = pd.DataFrame(
            {"date": dates, "entity": list("ABCD"), "target": values}
        )
        labelled = _stage_rank(panel, None, "target")["target"]
        assert list(_rank(values, dates)) == pytest.approx(list(labelled))

    def test_it_spans_minus_half_to_half(self):
        assert list(_rank([3.0, 1.0, 2.0], _dates(3))) == pytest.approx(
            [0.5, -0.5, 0.0]
        )


class TestItIsImmuneToTheDistribution:
    def test_one_extreme_name_does_not_move_the_others(self):
        """What `cross_sectional_standardize` cannot do: a mean and a
        standard deviation are both moved by the outlier, so every other
        name's value changes with it."""
        dates = _dates(4)
        mild = _rank([10.0, 20.0, 30.0, 40.0], dates)
        wild = _rank([10.0, 20.0, 30.0, 1e9], dates)
        assert list(mild) == pytest.approx(list(wild))

    def test_the_standardize_step_is_moved_by_it(self):
        """The contrast, so this file states what the step is FOR rather
        than only that it ranks."""
        from standard_quant_tools.modeling.preprocessing.steps import (
            CrossSectionalStandardize,
        )

        dates = _dates(4)
        step = CrossSectionalStandardize(clip_sigma=0.0)
        mild = step.transform(
            pd.DataFrame({"f": [10.0, 20.0, 30.0, 40.0]}), {}, FoldContext(dates=dates)
        )["f"]
        wild = step.transform(
            pd.DataFrame({"f": [10.0, 20.0, 30.0, 1e9]}), {}, FoldContext(dates=dates)
        )["f"]
        assert list(mild)[:3] != pytest.approx(list(wild)[:3])

    def test_a_feature_on_wildly_different_scales_per_date_is_comparable(self):
        """The documented failure: a cross-sectional standard deviation
        running 0.23 to 22.4 across dates. Ranks put both days on one
        scale."""
        dates = _dates(3, 3)
        ranked = _rank([0.1, 0.2, 0.3, 100.0, 200.0, 300.0], dates)
        assert list(ranked[:3]) == pytest.approx(list(ranked[3:]))


class TestTheEdges:
    def test_a_one_entity_date_is_nan_not_zero(self):
        """`_stage_rank` makes the same call: a one-name rank "is not a
        measurement". A fabricated 0.0 would read as the middle of a
        cross-section that does not exist."""
        ranked = _rank([5.0, 1.0, 2.0], _dates(1, 2))
        assert pd.isna(ranked.iloc[0])
        assert not ranked.iloc[1:].isna().any()

    def test_a_missing_value_stays_missing(self):
        """A name with no value is not ranked, and does not shift the
        names that are present."""
        dates = _dates(4)
        ranked = _rank([10.0, np.nan, 30.0, 40.0], dates)
        assert pd.isna(ranked.iloc[1])
        present = _rank([10.0, 30.0, 40.0], _dates(3))
        assert list(ranked.dropna()) == pytest.approx(list(present))

    def test_ties_share_the_mean_rank(self):
        ranked = _rank([5.0, 5.0, 1.0, 9.0], _dates(4))
        assert ranked.iloc[0] == pytest.approx(ranked.iloc[1])

    def test_rows_without_dates_are_refused_by_name(self):
        with pytest.raises(ValidationError, match="one date per row"):
            STEP.transform(pd.DataFrame({"f": [1.0]}), {}, FoldContext(dates=None))

    def test_an_empty_frame_passes_through(self):
        empty = pd.DataFrame({"f": []})
        out = STEP.transform(empty, {}, FoldContext(dates=np.array([], dtype="datetime64[ns]")))
        assert out.empty


class TestItIsStatelessAndRegistered:
    def test_nothing_is_fitted(self):
        """Each date is ranked against its own cross-section, which is
        contemporaneous information a live model also has — so nothing
        crosses the fold boundary."""
        assert CrossSectionalRank.stateless is True
        assert STEP.fit(pd.DataFrame({"f": [1.0, 2.0]}), FoldContext(dates=_dates(2))) == {}

    def test_the_transform_does_not_depend_on_the_training_fold(self):
        """The property behind `stateless`, asserted rather than trusted:
        the same rows rank the same whatever was fitted before them."""
        dates = _dates(3)
        values = [3.0, 1.0, 2.0]
        assert list(_rank(values, dates)) == pytest.approx(
            list(STEP.transform(pd.DataFrame({"f": values}), {"anything": 1}, FoldContext(dates=dates))["f"])
        )

    def test_it_is_in_the_registry_with_no_parameters(self):
        definition = PREPROCESSOR_REGISTRY["cross_sectional_rank"]
        assert definition.cls is CrossSectionalRank
        assert definition.default_params == {}
        assert "rank" in definition.description.lower()
