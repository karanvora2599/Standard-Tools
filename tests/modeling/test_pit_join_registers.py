"""
A point-in-time join can be fitted on.

`join_point_in_time` wrote the joined panel and returned a path. `joined_uri`
appeared in exactly three places — the field, the write, the return — and
nothing consumed it. The `dataset_id` it returned was the INPUT's: the panel
it joined onto, not the panel it made. So no lab tool and no experiment
could read the result.

That is the whole route this library advertises for event features: "a
caller who has FOMC dates, an earnings calendar or a set of
index-membership changes can join them onto a panel today". They could join
it, and then they were stuck.

The test that matters is the last one: a model fitted on the joined dataset,
using a joined column as a feature. Everything else is a step towards it.
"""

import pytest

from standard_quant_tools.modeling.agent.models import (
    JoinPointInTimeInput,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.tools import (
    inspect_dataset,
    join_point_in_time,
    run_model_experiment,
)
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    InspectDatasetInput,
)
from standard_quant_tools.modeling.agent.tools import build_model_dataset

from .test_scoring import _dataset_spec

#: An event tape with a release date and the date it became known, which is
#: the whole point of the as-of join: `available_time` is what the panel may
#: see, `event_time` is what it is about.
#:
#: The first event PREDATES the panel (which spans 2022-01-01..2023-12-31) on
#: purpose. An as-of join leaves every row before the first record NaN, and
#: `run_model_experiment` refuses a feature missing from every training row of
#: a fold — rightly, since it would train on a constant under the feature's
#: name. That refusal is not what this file is about.
RECORDS = [
    {
        "event_time": "2021-12-20",
        "available_time": "2021-12-21",
        "entity": entity,
        "eps": 1.0 + index,
    }
    for index, entity in enumerate(("AAA", "BBB", "CCC"))
] + [
    {
        "event_time": "2022-06-15",
        "available_time": "2022-06-16",
        "entity": entity,
        "eps": 2.0 + index,
    }
    for index, entity in enumerate(("AAA", "BBB", "CCC"))
]


@pytest.fixture
def dataset_id(patched_multi_factory):
    """Built through the tool, which is the path a caller takes and the
    one that persists dataset_spec.json beside the panel."""
    return build_model_dataset(BuildModelDatasetInput(spec=_dataset_spec())).dataset_id


@pytest.fixture
def joined(dataset_id):
    return join_point_in_time(
        JoinPointInTimeInput(
            dataset_id=dataset_id,
            records=RECORDS,
            entity_scoped=True,
            fields=["eps"],
            prefix="event.",
        )
    )


class TestItComesBackAsADataset:
    def test_a_new_dataset_id_comes_back(self, joined, dataset_id):
        assert joined.joined_dataset_id
        # NOT the one that was joined onto. Returning that was the defect:
        # it reads like an answer and fits the panel without the join.
        assert joined.joined_dataset_id != dataset_id

    def test_the_joined_column_is_a_feature_of_it(self, joined):
        inspected = inspect_dataset(
            InspectDatasetInput(dataset_id=joined.joined_dataset_id)
        )
        assert "event.eps" in {c.name for c in inspected.columns or []}

    def test_the_warnings_say_which_id_to_fit_on(self, joined):
        """A caller holding two ids needs to be told which one carries the
        join, and that registration is by reference."""
        assert any("pass that to run_model_experiment" in w for w in joined.warnings)
        assert any("REGISTERED BY REFERENCE" in w for w in joined.warnings)

    def test_the_source_dataset_is_untouched(self, joined, dataset_id):
        """The join writes beside the source panel; it must not change what
        the source dataset loads."""
        inspected = inspect_dataset(InspectDatasetInput(dataset_id=dataset_id))
        assert "event.eps" not in {c.name for c in inspected.columns or []}


class TestTheArtifactIsContentAddressed:
    def test_two_different_joins_do_not_collide(self, dataset_id):
        """The name was the fixed `pit_joined` inside the SOURCE dataset's
        directory. Harmless while nothing read the file; once a dataset
        loads from it, the second join would change what the first reads.
        """
        first = join_point_in_time(
            JoinPointInTimeInput(
                dataset_id=dataset_id, records=RECORDS, entity_scoped=True,
                fields=["eps"], prefix="a.",
            )
        )
        second = join_point_in_time(
            JoinPointInTimeInput(
                dataset_id=dataset_id, records=RECORDS, entity_scoped=True,
                fields=["eps"], prefix="b.",
            )
        )
        assert first.joined_uri != second.joined_uri
        assert first.joined_dataset_id != second.joined_dataset_id
        # And the first dataset still loads what it was registered against.
        inspected = inspect_dataset(
            InspectDatasetInput(dataset_id=first.joined_dataset_id)
        )
        assert "a.eps" in {c.name for c in inspected.columns or []}

    def test_the_same_join_twice_is_the_same_artifact(self, dataset_id):
        """The digest is in the name, so identical bytes rewrite in place."""
        kwargs = dict(
            dataset_id=dataset_id, records=RECORDS, entity_scoped=True,
            fields=["eps"], prefix="same.",
        )
        first = join_point_in_time(JoinPointInTimeInput(**kwargs))
        second = join_point_in_time(JoinPointInTimeInput(**kwargs))
        assert first.joined_uri == second.joined_uri


class TestAPanelWithNoLabel:
    def test_it_says_why_nothing_was_registered(self, dataset_id, monkeypatch):
        """Registering a dataset no experiment can fit would be a worse
        answer than saying there is nothing to fit."""
        import standard_quant_tools.modeling.agent.tools as tools

        real = tools._load_dataset_panel

        def unlabelled(ds_id):
            panel, meta, directory = real(ds_id)
            meta = {**meta, "target_id": None, "targets": []}
            return panel, meta, directory

        monkeypatch.setattr(tools, "_load_dataset_panel", unlabelled)
        result = join_point_in_time(
            JoinPointInTimeInput(
                dataset_id=dataset_id, records=RECORDS, entity_scoped=True,
                fields=["eps"], prefix="event.",
            )
        )
        assert result.joined_dataset_id is None
        assert any("nothing to fit" in w for w in result.warnings)
        # The frame is still there to register by hand once it has a label.
        assert result.joined_uri


class TestTheWholePoint:
    def test_a_model_fits_on_the_joined_dataset(self, joined):
        """The end of the road the join exists for: an event feature in a
        fitted model. Before this, the joined panel could not be reached by
        any experiment at all."""
        result = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=joined.joined_dataset_id,
                spec={
                    "task": "regression",
                    "estimator": {"type": "ridge"},
                    "validation": {
                        "method": "walk_forward",
                        "train_window": 60,
                        "test_window": 20,
                        "min_folds": 1,
                    },
                    "random_seed": 0,
                },
            )
        )
        assert result.model_id
        # The joined column reached the fit: the importance summary has one
        # entry per feature the model was given.
        assert "event.eps" in result.feature_importance_summary
