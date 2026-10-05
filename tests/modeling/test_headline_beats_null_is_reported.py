"""
The tools that RANK models now report whether the headline beat its null.

`_headline_report` computes `beats_null` and `score_predictions` calls it
"THE HEADLINE TEST ... the test a run makes". It is persisted at
`manifest.validation_report["headline"]["beats_null"]`, so reading it costs
nothing.

`list_models` and `compare_models` reported `headline_value` and not that
field, so a leaderboard showed the number without whether it can be told
from luck. `_headline_report`'s own docstring records what that is worth:
"none of sixteen recorded runs beat zero at 5%, and every run had reported
the headline with nothing to say so."

None is a third state, not a synonym for False: a run with no headline
value, a task with no null, or a model registered before the test existed
made no test, and reporting that as "did not beat" would be a claim nobody
made.
"""

import pytest

from standard_quant_tools.modeling.agent.models import (
    CompareModelsInput,
    ListModelsInput,
)
from standard_quant_tools.modeling.agent.tools import (
    compare_models,
    headline_test,
    list_models,
)
from standard_quant_tools.modeling.registry.model_registry import load_manifest

from .test_scoring import _dataset_spec, _train_a_model_with_spec


@pytest.fixture
def model_id(patched_multi_factory):
    return _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_beats_null")


@pytest.fixture
def two_models(patched_multi_factory):
    """`compare_models` ranks, so it requires at least two — which is also
    the case the test is about: a rank among models that may all be draws.
    """
    return [
        _train_a_model_with_spec(_dataset_spec(), dataset_id=f"ds_beats_null_{n}")
        for n in ("a", "b")
    ]


class TestTheManifestCarriesTheTest:
    def test_the_block_is_read_and_not_recomputed(self, model_id):
        """The manifest is immutable and content-hashed, so this is a read.
        That is what lets a caller gate on the test without re-scoring."""
        block = headline_test(load_manifest(model_id))
        assert block, "the run recorded no headline block"
        assert "beats_null" in block
        assert block["metric"]

    def test_a_manifest_without_the_block_reads_as_empty(self):
        """A model registered before the test existed, not a crash."""

        class _Bare:
            validation_report = {}

        assert headline_test(_Bare()) == {}

        class _None:
            validation_report = None

        assert headline_test(_None()) == {}


class TestTheRankersCarryIt:
    def test_list_models_reports_it(self, model_id):
        result = list_models(ListModelsInput())
        mine = [m for m in result.models if m.model_id == model_id]
        assert mine, "the trained model was not listed"
        expected = headline_test(load_manifest(model_id)).get("beats_null")
        assert mine[0].headline_beats_null == (
            None if expected is None else bool(expected)
        )
        # The value is still there; this is an addition, not a replacement.
        assert mine[0].headline_metric

    def test_compare_models_reports_it(self, two_models):
        result = compare_models(CompareModelsInput(model_ids=two_models))
        assert len(result.comparisons) == 2
        for row in result.comparisons:
            expected = headline_test(load_manifest(row.model_id)).get("beats_null")
            assert row.beats_null == (None if expected is None else bool(expected))

    def test_the_rank_and_the_test_are_separate_answers(self, two_models):
        """A rank among models none of which beat their null is a ranking
        of draws. Both have to be readable at once for that to be visible.
        """
        result = compare_models(CompareModelsInput(model_ids=two_models))
        ranked = [r for r in result.comparisons if r.rank is not None]
        assert ranked, "nothing was ranked, so the contrast cannot be read"
        assert all(hasattr(r, "beats_null") for r in ranked)
