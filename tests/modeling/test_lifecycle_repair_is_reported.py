"""
A torn promotion log is reported where the model's history is read.

An interrupted append leaves a fragment at the end of `promotions.jsonl`;
the next reader cuts it off and keeps it in a `.promotions.torn-*` side
file. That repair was only logged, to a logger the package gives just a
NullHandler, so whoever read the model's history to decide whether to trust
it never learned that a decision may be missing from it. The model card
(`inspect_model`'s summary) and `promote_model` now name every fragment set
aside, and its side file.
"""

from uuid import uuid4

import pytest

from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    InspectModelInput,
    ListModelsInput,
    PromoteModelInput,
)
from standard_quant_tools.modeling.agent.tools import (
    inspect_model,
    list_models,
    promote_model,
)
from standard_quant_tools.modeling.registry.lifecycle import (
    PROMOTIONS_FILE,
    current_stage,
    promote,
    torn_fragments,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

REASON = "reviewed the walk-forward evidence"
FRAGMENT = b'{"actor": "reviewer-2", "evidence": ["backtest_ref=sqt://eq'


def _tear(model_id: str) -> None:
    with open(_artifacts.run_dir(model_id) / PROMOTIONS_FILE, "ab") as handle:
        handle.write(FRAGMENT)


def _side_files(model_id: str):
    return sorted(_artifacts.run_dir(model_id).glob(".promotions.torn-*"))


class TestTheLogSaysWhatWasSetAside:
    @pytest.fixture(autouse=True)
    def _own_runs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))

    @staticmethod
    def _placeholder_model() -> str:
        model_id = f"mdl_{uuid4().hex[:12]}"
        directory = _artifacts.run_dir(model_id)
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text("{}", encoding="utf-8")
        return model_id

    def test_a_log_never_torn_reports_nothing(self):
        """Null case."""
        model_id = self._placeholder_model()
        assert torn_fragments(model_id) == []
        promote(model_id, "validated", REASON)
        assert torn_fragments(model_id) == []

    def test_a_repair_names_its_side_file_and_size(self):
        model_id = self._placeholder_model()
        promote(model_id, "validated", REASON)
        _tear(model_id)

        assert current_stage(model_id) == "validated"  # the read repairs

        [side] = _side_files(model_id)
        [note] = torn_fragments(model_id)
        assert side.name in note and f"{len(FRAGMENT)} byte(s)" in note
        assert "record it again" in note


class TestTheToolsCarryIt:
    def test_the_model_card_and_the_next_promotion_name_the_side_file(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_torn_log")

        # Null case first: an intact log reports no repair anywhere.
        validated = promote_model(
            PromoteModelInput(model_id=model_id, to_stage="validated", reason=REASON)
        )
        assert validated.promotion_log_repairs == []
        card = inspect_model(InspectModelInput(model_id=model_id)).data
        assert card["promotion_log_repairs"] == []

        # A reader other than the card makes the repair ...
        _tear(model_id)
        list_models(ListModelsInput())
        [side] = _side_files(model_id)

        # ... and the card still says so, naming the side file.
        card = inspect_model(InspectModelInput(model_id=model_id)).data
        assert card["stage"] == "validated"
        [note] = card["promotion_log_repairs"]
        assert side.name in note

        staged = promote_model(
            PromoteModelInput(model_id=model_id, to_stage="staging", reason=REASON)
        )
        assert staged.to_stage == "staging"
        assert [side.name in n for n in staged.promotion_log_repairs] == [True]
        assert "promotion_log_repairs" in staged.model_dump()
