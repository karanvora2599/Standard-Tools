"""
A feature matrix computed elsewhere, trained on here.

WHAT THIS PINS. That the seam holds in both directions: a panel this
library did not build reaches `run_model_experiment` intact, and a panel
that has been edited underneath the reference does NOT.

The second is the one worth the file. Nothing is copied when a panel is
registered, so the file stays under its owner's control and can change --
which is exactly the situation `build_model_dataset`'s hash check was
written for and has never had to face, because it owned every panel it
verified.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import RegisterExternalPanelInput
from standard_quant_tools.modeling.agent.tools import (
    _load_dataset_panel,
    register_external_panel,
)

DATES = 260
ENTITIES = ["AAA", "BBB", "CCC", "DDD"]


def _panel(*, dates: int = DATES, entities=None, seed: int = 4) -> pd.DataFrame:
    """
    A long panel with REAL signal in it.

    `alpha` genuinely predicts the target and `noise` genuinely does not,
    so an experiment run on this has a right answer and the tests can
    assert on it rather than only on "it ran".
    """
    entities = list(entities or ENTITIES)
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=dates, freq="B")
    rows = []
    for entity in entities:
        alpha = rng.normal(0, 1, dates)
        noise = rng.normal(0, 1, dates)
        target = 0.004 * alpha + rng.normal(0, 0.002, dates)
        rows.append(
            pd.DataFrame(
                {
                    "date": index,
                    "entity": entity,
                    "alpha": alpha,
                    "noise": noise,
                    "target": target,
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["date", "entity"])


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


@pytest.fixture
def panel_path(tmp_path) -> str:
    path = tmp_path / "features.parquet"
    _panel().to_parquet(path, index=False)
    return str(path)


def _register(path, **kwargs):
    payload = {"path": str(path), "horizon": 5}
    payload.update(kwargs)
    return register_external_panel(RegisterExternalPanelInput(**payload))


class TestNothingIsCopied:
    def test_registration_writes_a_record_and_no_panel(
        self, runs_dir, panel_path
    ) -> None:
        result = _register(panel_path)
        directory = _artifacts.run_dir(result.dataset_id)
        written = sorted(p.name for p in directory.iterdir())
        assert written == ["dataset_meta.json", "dataset_spec.json"], (
            f"registration wrote {written}; a panel.parquet here would be "
            "the second copy this path exists to avoid"
        )

    def test_the_record_points_at_the_original_file(self, runs_dir, panel_path) -> None:
        result = _register(panel_path)
        assert result.source_path == str(panel_path)
        meta = json.loads(
            (_artifacts.run_dir(result.dataset_id) / "dataset_meta.json").read_text()
        )
        assert meta["storage"] == "external"
        assert meta["panel_path"] == str(panel_path)

    def test_the_panel_loads_back_through_the_dataset_id(
        self, runs_dir, panel_path
    ) -> None:
        result = _register(panel_path)
        panel, meta, _directory = _load_dataset_panel(result.dataset_id)
        assert len(panel) == DATES * len(ENTITIES)
        assert set(meta["feature_ids"]) == {"alpha", "noise"}
        assert {"date", "entity", "target"} <= set(panel.columns)


class TestTheRecordIsAReadableDataset:
    def test_it_reports_what_it_found(self, runs_dir, panel_path) -> None:
        result = _register(panel_path)
        assert result.rows == DATES * len(ENTITIES)
        assert result.entities == ENTITIES
        assert result.feature_ids == ["alpha", "noise"]
        assert result.target_id == "forward_return:5"

    def test_the_spec_it_synthesizes_is_a_real_one(self, runs_dir, panel_path):
        """
        Not decoration. `run_model_experiment` verifies this spec's hash,
        bundles it into the registered model, and reads its universe and
        interval into the lineage, so a placeholder would put a false claim
        in all three.
        """
        from standard_quant_tools.modeling.specs import DatasetSpec

        result = _register(panel_path, interval="1s")
        spec_dict = json.loads(
            (_artifacts.run_dir(result.dataset_id) / "dataset_spec.json").read_text()
        )
        spec = DatasetSpec(**spec_dict)
        assert spec.provider == "external"
        assert spec.interval == "1s"
        assert spec.universe == ENTITIES
        assert [f.id for f in spec.features] == ["alpha", "noise"]
        assert spec.target.horizon == 5

    def test_the_spec_hash_verifies(self, runs_dir, panel_path) -> None:
        from standard_quant_tools.modeling.dataset.builder import dataset_spec_hash
        from standard_quant_tools.modeling.specs import DatasetSpec

        result = _register(panel_path)
        directory = _artifacts.run_dir(result.dataset_id)
        meta = json.loads((directory / "dataset_meta.json").read_text())
        spec = DatasetSpec(**json.loads((directory / "dataset_spec.json").read_text()))
        assert dataset_spec_hash(spec) == meta["spec_hash"]

    def test_row_loss_is_answerable(self, runs_dir, panel_path) -> None:
        from standard_quant_tools.modeling.agent.dataset_tools import (
            ExplainRowLossInput,
            explain_dataset_row_loss,
        )

        result = _register(panel_path)
        report = explain_dataset_row_loss(
            ExplainRowLossInput(dataset_id=result.dataset_id)
        )
        assert report.dataset_id == result.dataset_id
        assert {"alpha", "noise"} <= {c.column for c in report.columns}


class TestTheHorizonIsRequired:
    def test_it_cannot_be_omitted(self, runs_dir, panel_path) -> None:
        """
        The purge reads it. A missing horizon does not fail -- it disables
        the overlap purge silently, which is the failure mode that made it
        worth refusing rather than defaulting.
        """
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            RegisterExternalPanelInput(path=panel_path)

    def test_it_reaches_the_target_id(self, runs_dir, panel_path) -> None:
        assert _register(panel_path, horizon=21).target_id == "forward_return:21"


class TestTheColumnsAreCheckedFirst:
    def test_custom_column_names_are_accepted(self, runs_dir, tmp_path) -> None:
        frame = _panel().rename(
            columns={"date": "ts", "entity": "symbol", "target": "y"}
        )
        path = tmp_path / "renamed.parquet"
        frame.to_parquet(path, index=False)
        result = _register(
            path, date_column="ts", entity_column="symbol", target_column="y"
        )
        assert result.feature_ids == ["alpha", "noise"]
        panel, _meta, _d = _load_dataset_panel(result.dataset_id)
        assert {"date", "entity", "target"} <= set(panel.columns)

    def test_a_missing_column_is_refused_by_role(self, runs_dir, tmp_path) -> None:
        path = tmp_path / "noentity.parquet"
        _panel().drop(columns=["entity"]).to_parquet(path, index=False)
        with pytest.raises(ValidationError, match="entity_column"):
            _register(path)

    def test_feature_columns_can_be_narrowed(self, runs_dir, panel_path) -> None:
        result = _register(panel_path, feature_columns=["alpha"])
        assert result.feature_ids == ["alpha"]

    def test_naming_the_target_as_a_feature_is_refused(
        self, runs_dir, panel_path
    ) -> None:
        with pytest.raises(ValidationError, match="cannot be both"):
            _register(panel_path, feature_columns=["alpha", "target"])

    def test_an_unknown_feature_column_is_refused(self, runs_dir, panel_path):
        with pytest.raises(ValidationError, match="does not have"):
            _register(panel_path, feature_columns=["alpha", "absent"])

    def test_a_reserved_name_as_a_feature_is_refused(self, runs_dir, tmp_path):
        frame = _panel()
        frame["label_end_date"] = frame["date"]
        path = tmp_path / "reserved.parquet"
        frame.to_parquet(path, index=False)
        with pytest.raises(ValidationError, match="reserved"):
            _register(path, feature_columns=["alpha", "label_end_date"])

    def test_a_panel_with_no_features_is_refused(self, runs_dir, tmp_path):
        path = tmp_path / "bare.parquet"
        _panel()[["date", "entity", "target"]].to_parquet(path, index=False)
        with pytest.raises(ValidationError, match="no feature columns"):
            _register(path)

    def test_an_unparseable_date_is_refused(self, runs_dir, tmp_path) -> None:
        frame = _panel()
        frame["date"] = frame["date"].astype(str)
        frame.loc[:4, "date"] = "not a date"
        path = tmp_path / "baddate.parquet"
        frame.to_parquet(path, index=False)
        with pytest.raises(ValidationError, match="unparseable"):
            _register(path)

    def test_too_many_entities_is_refused(self, runs_dir, tmp_path) -> None:
        """A spec's universe holds 1000; a panel with more cannot be
        described by the record this writes, and failing here says so
        before anything is persisted."""
        frame = _panel(dates=2, entities=[f"S{i:04d}" for i in range(1100)])
        path = tmp_path / "wide.parquet"
        frame.to_parquet(path, index=False)
        with pytest.raises(ValidationError, match="at most 1000"):
            _register(path)


class TestIntegritySurvivesNotCopying:
    def test_an_edited_panel_fails_the_hash_check(
        self, runs_dir, tmp_path, panel_path
    ) -> None:
        """
        The check `build_model_dataset` has always had and has never had to
        use, because it owned every panel it verified. Here the file belongs
        to someone else.
        """
        result = _register(panel_path)
        edited = _panel()
        edited.loc[:9, "alpha"] = 99.0
        edited.to_parquet(panel_path, index=False)
        with pytest.raises(ValidationError, match="no longer matches"):
            _load_dataset_panel(result.dataset_id)

    def test_a_deleted_panel_says_what_happened(self, runs_dir, tmp_path) -> None:
        path = tmp_path / "transient.parquet"
        _panel().to_parquet(path, index=False)
        result = _register(path)
        path.unlink()
        with pytest.raises(ValidationError, match="only as good as the file"):
            _load_dataset_panel(result.dataset_id)

    def test_registration_warns_that_it_did_not_copy(
        self, runs_dir, panel_path
    ) -> None:
        result = _register(panel_path)
        assert any("REGISTERED BY REFERENCE" in w for w in result.warnings)
        assert any("score_model cannot run" in w for w in result.warnings)


class TestItTrainsAModel:
    """The point of the phase: a matrix this library did not build,
    trained on by the engine that has only ever seen its own."""

    @staticmethod
    def _spec():
        from standard_quant_tools.modeling.specs import (
            EstimatorSpec,
            ModelSpec,
            ValidationSpec,
        )

        return ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge"),
            validation=ValidationSpec(
                method="walk_forward",
                scheme="expanding",
                n_splits=3,
                train_window=120,
                test_window=30,
            ),
        )

    def test_an_experiment_runs_end_to_end(self, runs_dir, panel_path) -> None:
        from standard_quant_tools.modeling.agent.models import RunModelExperimentInput
        from standard_quant_tools.modeling.agent.tools import run_model_experiment

        dataset_id = _register(panel_path).dataset_id
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=self._spec())
        )
        assert result.model_id
        assert result.n_folds >= 3
        # `alpha` carries the signal by construction, so a panel that
        # arrived intact is one a ridge can fit. A mismapped column would
        # still 'run' and would not do this.
        assert result.oos_metrics["r2"] > 0.5, result.oos_metrics

    def test_the_model_finds_the_feature_that_carries_the_signal(
        self, runs_dir, panel_path
    ) -> None:
        """
        `alpha` predicts the target by construction and `noise` does not.
        A pipeline that silently mismapped columns would still 'run', so
        this asserts the panel arrived MEANING what it meant.
        """
        from standard_quant_tools.modeling.agent.models import (
            InspectModelInput,
            RunModelExperimentInput,
        )
        from standard_quant_tools.modeling.agent.tools import (
            inspect_model,
            run_model_experiment,
        )

        dataset_id = _register(panel_path).dataset_id
        trained = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=self._spec())
        )
        report = inspect_model(
            InspectModelInput(model_id=trained.model_id, view="feature_importance")
        )
        payload = report.model_dump()
        text = json.dumps(payload)
        assert "alpha" in text and "noise" in text


class TestScoreModelRefusesRatherThanGuessing:
    def test_it_names_the_reason_and_the_alternative(
        self, runs_dir, panel_path
    ) -> None:
        from standard_quant_tools.modeling.agent.models import (
            RunModelExperimentInput,
            ScoreModelInput,
        )
        from standard_quant_tools.modeling.agent.tools import (
            run_model_experiment,
            score_model,
        )

        dataset_id = _register(panel_path).dataset_id
        trained = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=dataset_id, spec=TestItTrainsAModel._spec()
            )
        )
        with pytest.raises(ValidationError, match="externally registered"):
            score_model(
                ScoreModelInput(
                    model_id=trained.model_id,
                    universe=ENTITIES,
                    as_of="2024-12-31",
                )
            )


class TestOtherLayouts:
    def test_a_csv_panel_registers(self, runs_dir, tmp_path) -> None:
        path = tmp_path / "features.csv"
        _panel().to_csv(path, index=False)
        result = _register(path)
        assert result.rows == DATES * len(ENTITIES)

    def test_a_partitioned_directory_registers(self, runs_dir, tmp_path) -> None:
        """The layout that makes copying worst: one file per day."""
        directory = tmp_path / "parts"
        directory.mkdir()
        frame = _panel()
        for index, (_date, chunk) in enumerate(frame.groupby("date")):
            chunk.to_parquet(directory / f"part-{index:04d}.parquet", index=False)
        result = _register(directory)
        assert result.rows == DATES * len(ENTITIES)
        panel, _meta, _d = _load_dataset_panel(result.dataset_id)
        assert len(panel) == DATES * len(ENTITIES)

    def test_a_label_end_column_is_carried(self, runs_dir, tmp_path) -> None:
        frame = _panel()
        frame["ends"] = frame["date"] + pd.Timedelta(days=7)
        path = tmp_path / "barrier.parquet"
        frame.to_parquet(path, index=False)
        result = _register(path, label_end_column="ends", target_type="triple_barrier")
        panel, _meta, _d = _load_dataset_panel(result.dataset_id)
        assert "label_end_date" in panel.columns
        assert "ends" not in result.feature_ids


def _multi_horizon_panel(seed: int = 11) -> pd.DataFrame:
    """
    One feature matrix, three labels, DELIBERATELY unequal signal.

    `alpha` predicts the 1-bar label strongly, the 5-bar one weakly and the
    20-bar one not at all. A selector that silently trained on the wrong
    column would still produce three models and three metrics; only the
    ORDERING of those metrics says the right column was used.

    The long horizons are also null on their final rows, the way a real
    forward label is, so "dropped per experiment" has something to drop.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=DATES, freq="B")
    rows = []
    for entity in ENTITIES:
        alpha = rng.normal(0, 1, DATES)
        noise = rng.normal(0, 1, DATES)
        frame = pd.DataFrame(
            {
                "date": index,
                "entity": entity,
                "alpha": alpha,
                "noise": noise,
                "ret_1": 0.01 * alpha + rng.normal(0, 0.0005, DATES),
                "ret_5": 0.01 * alpha + rng.normal(0, 0.004, DATES),
                "ret_20": rng.normal(0, 0.01, DATES),
            }
        )
        frame.loc[frame.index[-20:], "ret_20"] = np.nan
        frame.loc[frame.index[-5:], "ret_5"] = np.nan
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "entity"])


TARGETS = [
    {"name": "h1", "column": "ret_1", "horizon": 1},
    {"name": "h5", "column": "ret_5", "horizon": 5},
    {"name": "h20", "column": "ret_20", "horizon": 20},
]


class TestSeveralHorizonsInOnePanel:
    """
    The microstructure case: a book labelled at several horizons off
    identical features. One dataset per horizon would recompute and
    re-store the same matrix N times, and the models would stop being
    comparable because each would have been aligned separately.
    """

    @pytest.fixture
    def multi(self, tmp_path):
        path = tmp_path / "multi.parquet"
        _multi_horizon_panel().to_parquet(path, index=False)
        return str(path)

    def test_all_three_register_from_one_file(self, runs_dir, multi) -> None:
        result = register_external_panel(
            RegisterExternalPanelInput(path=multi, targets=TARGETS)
        )
        assert result.targets == ["h1", "h5", "h20"]
        assert result.target_id == "forward_return:1"
        assert result.feature_ids == ["alpha", "noise"]

    def test_the_panel_carries_every_label(self, runs_dir, multi) -> None:
        result = register_external_panel(
            RegisterExternalPanelInput(path=multi, targets=TARGETS)
        )
        panel, meta, _d = _load_dataset_panel(result.dataset_id)
        for name in ("h1", "h5", "h20"):
            assert f"target__{name}" in panel.columns
        # The primary is still an ordinary `target`, so everything that has
        # only ever seen one label keeps working on a multi-label panel.
        assert "target" in panel.columns
        assert panel["target"].equals(panel["target__h1"])
        assert [t["name"] for t in meta["targets"]] == ["h1", "h5", "h20"]

    def test_horizon_and_targets_are_mutually_exclusive(self, multi) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="EITHER"):
            RegisterExternalPanelInput(path=multi, horizon=5, targets=TARGETS)
        with pytest.raises(pydantic.ValidationError, match="EITHER"):
            RegisterExternalPanelInput(path=multi)

    def test_a_missing_target_column_names_the_target(self, runs_dir, multi) -> None:
        broken = [dict(t) for t in TARGETS]
        broken[1]["column"] = "not_a_column"
        with pytest.raises(ValidationError, match="target 'h5'"):
            register_external_panel(
                RegisterExternalPanelInput(path=multi, targets=broken)
            )

    def test_duplicate_names_are_refused(self, runs_dir, multi) -> None:
        dupes = [
            {"name": "h1", "column": "ret_1", "horizon": 1},
            {"name": "h1", "column": "ret_5", "horizon": 5},
        ]
        with pytest.raises(ValidationError, match="unique"):
            register_external_panel(
                RegisterExternalPanelInput(path=multi, targets=dupes)
            )


class TestSelectingAHorizon:
    @pytest.fixture
    def dataset_id(self, runs_dir, tmp_path):
        path = tmp_path / "multi.parquet"
        _multi_horizon_panel().to_parquet(path, index=False)
        return register_external_panel(
            RegisterExternalPanelInput(path=str(path), targets=TARGETS)
        ).dataset_id

    @staticmethod
    def _fit(dataset_id, target=None):
        from standard_quant_tools.modeling.agent.models import RunModelExperimentInput
        from standard_quant_tools.modeling.agent.tools import run_model_experiment

        return run_model_experiment(
            RunModelExperimentInput(
                dataset_id=dataset_id,
                spec=TestItTrainsAModel._spec(),
                target=target,
            )
        )

    def test_an_unknown_name_is_refused_with_the_list(self, dataset_id) -> None:
        with pytest.raises(ValidationError, match="h99"):
            self._fit(dataset_id, target="h99")

    def test_the_horizon_curve_recovers_the_planted_ordering(self, dataset_id) -> None:
        """
        THE TEST THAT MATTERS. `alpha` predicts h1 strongly, h5 weakly and
        h20 not at all, by construction. A selector that trained on the
        wrong column would still return three models and three r2 values --
        only their ORDER says the right label was used.
        """
        scores = {
            name: self._fit(dataset_id, target=name).oos_metrics["r2"]
            for name in ("h1", "h5", "h20")
        }
        assert scores["h1"] > scores["h5"] > scores["h20"], scores
        assert scores["h1"] > 0.8, scores
        assert scores["h20"] < 0.1, scores

    def test_each_model_records_its_own_horizon(self, dataset_id) -> None:
        from standard_quant_tools.modeling.agent.models import InspectModelInput
        from standard_quant_tools.modeling.agent.tools import inspect_model

        trained = self._fit(dataset_id, target="h20")
        report = inspect_model(
            InspectModelInput(model_id=trained.model_id, view="summary")
        )
        assert "forward_return:20" in json.dumps(report.model_dump(), default=str)

    def test_a_long_horizon_drops_only_its_own_rows(self, dataset_id) -> None:
        """
        h20 is null on its final 20 bars per entity and h1 is not. Dropping
        on the union would make the short-horizon model pay for the long
        one's warm-down.

        Asserted on the SELECTION rather than on out-of-sample row counts:
        which rows reach a fold is fold geometry, and with these windows the
        dropped tail sits past the last test window, so both experiments
        report the same n_oos_rows while genuinely training on different
        panels. The mechanism is what this pins.
        """
        from standard_quant_tools.modeling.agent.tools import _select_target

        panel, meta, _d = _load_dataset_panel(dataset_id)
        rows = {}
        for name in ("h1", "h5", "h20"):
            selected, target_id, notes = _select_target(panel, meta, name, dataset_id)
            rows[name] = len(selected)
            assert target_id.endswith(name.lstrip("h"))
        assert rows["h1"] > rows["h5"] > rows["h20"], rows
        # 4 entities x 20 unclosed bars
        assert rows["h1"] - rows["h20"] == 80, rows

    def test_the_drop_is_reported(self, dataset_id) -> None:
        from standard_quant_tools.modeling.agent.tools import _select_target

        panel, meta, _d = _load_dataset_panel(dataset_id)
        _selected, _target_id, notes = _select_target(panel, meta, "h20", dataset_id)
        assert any("THIS experiment only" in note for note in notes), notes

    def test_selecting_on_a_single_target_dataset_names_what_it_has(
        self, runs_dir, panel_path
    ) -> None:
        """A panel registered with one label records it as 'primary', so the
        refusal can list what is actually there rather than only saying no."""
        single = _register(panel_path).dataset_id
        with pytest.raises(ValidationError, match=r"carries \['primary'\]"):
            self._fit(single, target="h5")


class TestAnExecutionLabelSaysWhatItIs:
    """
    The point of widening the taxonomy. A fill probability computed in a
    C++ layer over the book used to have to be registered as
    `forward_return` -- a false claim in the manifest, and one that left
    the task/target check unable to tell a probability from a return.
    """

    @pytest.fixture
    def fills(self, tmp_path):
        """A panel whose label is whether a passive order filled."""
        rng = np.random.default_rng(17)
        index = pd.date_range("2024-01-01", periods=DATES, freq="B")
        rows = []
        for entity in ENTITIES:
            queue = rng.uniform(0, 1, DATES)
            spread = rng.uniform(0.5, 3.0, DATES)
            # Short queue and tight spread fill; the model has something
            # real to find, so a mislabelled column would show up as a
            # model that cannot find it.
            score = -2.0 * queue - 0.4 * spread + rng.normal(0, 0.15, DATES)
            rows.append(
                pd.DataFrame(
                    {
                        "date": index,
                        "entity": entity,
                        "queue_ahead": queue,
                        "spread_bps": spread,
                        "filled": (score > np.median(score)).astype(float),
                    }
                )
            )
        frame = pd.concat(rows, ignore_index=True).sort_values(["date", "entity"])
        path = tmp_path / "fills.parquet"
        frame.to_parquet(path, index=False)
        return str(path)

    def test_it_registers_under_its_own_name(self, runs_dir, fills) -> None:
        result = _register(
            fills,
            target_column="filled",
            target_type="fill_probability",
            horizon=1,
        )
        assert result.target_id == "fill_probability:1"
        assert result.feature_ids == ["queue_ahead", "spread_bps"]

    def test_the_manifest_records_the_real_label(self, runs_dir, fills) -> None:
        import json

        result = _register(
            fills,
            target_column="filled",
            target_type="fill_probability",
            horizon=1,
        )
        meta = json.loads(
            (_artifacts.run_dir(result.dataset_id) / "dataset_meta.json").read_text()
        )
        assert meta["target_id"] == "fill_probability:1"
        assert meta["targets"][0]["target_type"] == "fill_probability"

    def test_a_classifier_fits_it_and_a_regressor_is_refused(
        self, runs_dir, fills
    ) -> None:
        """
        The compatibility check earning its keep. A regressor on a 0/1
        label does not error -- it fits and reports a meaningless R2 --
        which is why the refusal has to come from the label's declared
        type rather than from the data.
        """
        from standard_quant_tools.modeling.agent.models import RunModelExperimentInput
        from standard_quant_tools.modeling.agent.tools import run_model_experiment
        from standard_quant_tools.modeling.specs import (
            EstimatorSpec,
            ModelSpec,
            ValidationSpec,
        )

        dataset_id = _register(
            fills,
            target_column="filled",
            target_type="fill_probability",
            horizon=1,
        ).dataset_id

        def spec(task, estimator):
            return ModelSpec(
                task=task,
                estimator=EstimatorSpec(type=estimator),
                validation=ValidationSpec(
                    method="walk_forward",
                    scheme="expanding",
                    n_splits=3,
                    train_window=120,
                    test_window=30,
                ),
            )

        trained = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=dataset_id, spec=spec("classification", "logistic")
            )
        )
        assert trained.model_id
        assert trained.oos_metrics.get("accuracy", 0) > 0.6, trained.oos_metrics

        with pytest.raises(ValidationError, match="expects one of"):
            run_model_experiment(
                RunModelExperimentInput(
                    dataset_id=dataset_id, spec=spec("regression", "ridge")
                )
            )

    def test_several_execution_horizons_register_together(
        self, runs_dir, tmp_path
    ) -> None:
        """A markout is measured at several distances from the same fill,
        which is the multi-horizon case again with a different label."""
        rng = np.random.default_rng(29)
        index = pd.date_range("2024-01-01", periods=DATES, freq="B")
        rows = []
        for entity in ENTITIES:
            flow = rng.normal(0, 1, DATES)
            rows.append(
                pd.DataFrame(
                    {
                        "date": index,
                        "entity": entity,
                        "ofi": flow,
                        "mk_1s": 0.6 * flow + rng.normal(0, 0.2, DATES),
                        "mk_30s": 0.2 * flow + rng.normal(0, 0.8, DATES),
                    }
                )
            )
        frame = pd.concat(rows, ignore_index=True).sort_values(["date", "entity"])
        path = tmp_path / "markout.parquet"
        frame.to_parquet(path, index=False)

        result = register_external_panel(
            RegisterExternalPanelInput(
                path=str(path),
                interval="1s",
                targets=[
                    {
                        "name": "mk1",
                        "column": "mk_1s",
                        "horizon": 1,
                        "target_type": "future_markout",
                    },
                    {
                        "name": "mk30",
                        "column": "mk_30s",
                        "horizon": 30,
                        "target_type": "future_markout",
                    },
                ],
            )
        )
        assert result.targets == ["mk1", "mk30"]
        assert result.target_id == "future_markout:1"

    def test_a_bar_built_dataset_cannot_ask_for_one(self) -> None:
        """`build_model_dataset` must refuse an execution label rather than
        hand back a forward return under its name."""
        from standard_quant_tools.modeling.dataset.target import build_target
        from standard_quant_tools.modeling.specs import TargetSpec

        close = pd.Series(
            100.0 + np.arange(60.0),
            index=pd.date_range("2024-01-01", periods=60, freq="B"),
        )
        with pytest.raises(ValidationError, match="register_external_panel"):
            build_target(close, TargetSpec(type="time_to_fill", horizon=5))
