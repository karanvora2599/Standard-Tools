"""
What the manifest recorded about itself, and the two universe facts a
build used to resolve in silence.

WHAT THIS PINS. Three of the fields written at registration decide
whether a later call is refused, and none of them could be read before
making that call: the information cutoff `score_model` gates `as_of` on
(the earliest legal scoring date), whether a conformal band was deployed
(the precondition for sizing on interval width), and the per-column
feature provenance scoring re-checks. `inspect_model(view="provenance")`
returns them, plus the environment that fitted the model against the one
asking now -- the only route to the current numerics, since the
capability report carries no numpy version, no BLAS and no thread count.

Beside it, two things a dataset build resolved without saying so: a
universe whose keys collide under the provider's symbol resolution (one
price series wearing two identities), and a calendar adopted from a venue
every key names, which is part of the dataset's identity.

Every detector here has a null case: an unmoved environment, a
point-only model, a dataset with distinct keys, a calendar that was
asked for rather than inferred.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.modeling.registry.environment as environment_module
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    EvaluateModelPortfolioInput,
    InspectModelInput,
    RegisterExternalPanelInput,
    ScoreModelInput,
    ValidateModelSpecInput,
)
from standard_quant_tools.modeling.agent.tools import (
    build_model_dataset,
    evaluate_model_portfolio,
    inspect_model,
    register_external_panel,
    score_model,
    validate_model_spec,
)
from standard_quant_tools.modeling.calendar import calendar_available
from standard_quant_tools.modeling.registry.environment import (
    environment_differences,
    environment_fingerprint,
)
from standard_quant_tools.modeling.registry.manifests import ModelManifest
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    ConformalSpec,
    EstimatorSpec,
    ModelSpec,
    PredictionTransformSpec,
    ValidationSpec,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

VENUE_QUALIFIED = ["AAA@XNYS", "BBB@XNYS", "CCC@XNYS"]

needs_calendars = pytest.mark.skipif(
    not calendar_available(), reason="exchange_calendars not installed"
)


def _provenance(model_id: str) -> dict:
    return inspect_model(InspectModelInput(model_id=model_id, view="provenance")).data


def _a_model_spec() -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )


class TestTheCutoffIsTheEarliestLegalScoringDate:
    """The field `score_model` gates on, read instead of guessed."""

    def test_the_cutoff_sits_past_the_last_feature_date(self, patched_multi_factory):
        """A horizon-5 forward-return label for a row dated t reads prices
        five bars past t, so the training data consumed dates the last
        FEATURE date does not mention. Both are in the view, and the
        cutoff is the later of the two."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_cutoff_is_readable"
        )
        data = _provenance(model_id)
        cutoff = pd.Timestamp(data["training_information_cutoff"])
        train_end = pd.Timestamp(data["train_end_date"])
        assert cutoff > train_end
        assert data["warnings"] == []

    def test_scoring_refuses_up_to_the_cutoff_and_runs_the_day_after(
        self, patched_multi_factory
    ):
        """The gate refuses `as_of <= cutoff`, so the cutoff itself is the
        last refused date and the first legal one is the day after. Both
        sides of that boundary are asserted; reading the field is what
        replaces discovering it from the refusal."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_cutoff_boundary"
        )
        cutoff = pd.Timestamp(_provenance(model_id)["training_information_cutoff"])
        universe = ["AAA", "BBB", "CCC"]
        for offset in (-1, 0):
            as_of = str((cutoff + pd.Timedelta(days=offset)).date())
            with pytest.raises(ValidationError, match="training information cutoff"):
                score_model(
                    ScoreModelInput(model_id=model_id, as_of=as_of, universe=universe)
                )
        first_legal = str((cutoff + pd.Timedelta(days=1)).date())
        result = score_model(
            ScoreModelInput(model_id=model_id, as_of=first_legal, universe=universe)
        )
        assert result.n_entities == 3


class TestTheConformalBandIsReadableBeforeItIsNeeded:
    """`has_conformal` is the precondition for sizing on interval width."""

    def test_a_banded_model_says_so_and_the_sizing_then_runs(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(),
            dataset_id="ds_band_is_reported",
            model_spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                intervals=ConformalSpec(alpha=0.1),
                random_seed=1,
            ),
        )
        distribution = _provenance(model_id)["distribution"]
        assert distribution["has_conformal"] is True
        assert distribution["alpha"] == pytest.approx(0.1)
        assert distribution["raw"]["conformal"]["radius"] > 0
        result = evaluate_model_portfolio(
            EvaluateModelPortfolioInput(
                model_id=model_id,
                transform=PredictionTransformSpec(method="uncertainty_scaled"),
            )
        )
        assert np.isfinite(result.metrics["sharpe_ratio"])

    def test_a_point_model_says_so_and_the_same_sizing_refuses(
        self, patched_multi_factory
    ):
        """The null case. Before the field was readable, this refusal was
        the only way to learn the transform was unavailable."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_no_band_is_reported"
        )
        distribution = _provenance(model_id)["distribution"]
        assert distribution["has_conformal"] is False
        assert distribution["quantile_levels"] == []
        assert distribution["alpha"] is None
        with pytest.raises(ValidationError, match="ModelSpec.intervals"):
            evaluate_model_portfolio(
                EvaluateModelPortfolioInput(
                    model_id=model_id,
                    transform=PredictionTransformSpec(method="uncertainty_scaled"),
                )
            )


class TestTheEnvironmentDiff:
    """What computed the model, against what is asking now."""

    def test_a_bumped_numpy_is_exactly_one_difference(
        self, patched_multi_factory, monkeypatch
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_environment_moved"
        )
        moved = copy.deepcopy(environment_fingerprint())
        moved["packages"]["numpy"] = "99.0.0"
        monkeypatch.setattr(
            environment_module, "environment_fingerprint", lambda: moved
        )
        data = _provenance(model_id)
        environment = data["environment"]
        assert environment["matches"] is False
        assert list(environment["differences"]) == ["packages.numpy"]
        assert environment["differences"]["packages.numpy"]["current"] == "99.0.0"
        assert environment["trained"]["packages"]["numpy"] != "99.0.0"
        assert data["warnings"] == [
            "environment moved since training: packages.numpy was "
            f"{environment['differences']['packages.numpy']['trained']!r}, "
            "is now '99.0.0'."
        ]

    def test_the_unmoved_environment_matches(self, patched_multi_factory):
        """The null case: fitted and asked on one machine in one process."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_environment_unmoved"
        )
        environment = _provenance(model_id)["environment"]
        assert environment["differences"] == {}
        assert environment["matches"] is True
        assert environment["trained"] == environment["current"]

    def test_a_key_on_one_side_only_is_a_difference_with_none(self):
        """Dotted keys, and absence reported rather than skipped -- 'this
        manifest predates the field' and 'this package is not installed'
        are different facts."""
        differences = environment_differences(
            {"packages": {"numpy": "2.0.0", "lightgbm": None}, "python": "3.12.1"},
            {
                "packages": {"numpy": "2.0.0"},
                "python": "3.12.1",
                "blas": {"blas": "mkl"},
            },
        )
        assert differences == {
            "blas.blas": {"trained": None, "current": "mkl"},
            "packages.lightgbm": {"trained": None, "current": None},
        }
        assert environment_differences({}, {}) == {}


class TestAManifestWithoutTheCutoffField:
    def test_the_view_reports_none_and_names_the_weaker_guarantee(
        self, patched_multi_factory
    ):
        """A manifest written before the cutoff existed loads with the
        field absent. The view answers None and says which guarantee is
        actually in force, rather than raising on a key that is not
        there."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_cutoff_predates_the_field"
        )
        path = Path(_artifacts.run_dir(model_id)) / "manifest.json"
        payload = json.loads(path.read_text())
        payload.pop("training_information_cutoff")
        path.write_text(json.dumps(payload))
        assert load_manifest(model_id).training_information_cutoff is None

        data = _provenance(model_id)
        assert data["training_information_cutoff"] is None
        assert data["train_end_date"] == payload["train_end_date"]
        weaker = [w for w in data["warnings"] if "train_end_date" in w]
        assert len(weaker) == 1
        assert "weaker guarantee" in weaker[0]


class TestFormatsIsDeclaredOnce:
    def test_one_declaration_and_the_round_trip_keeps_it(self, patched_multi_factory):
        """`formats` was declared twice with identical comment blocks, so
        the second silently shadowed the first -- the same field, but a
        reader could not tell which one was live. `model_fields` cannot
        show a duplicate (a dict collapses it), so the source is what is
        counted."""
        source = inspect.getsource(ModelManifest)
        assert source.count("formats: List[str]") == 1
        assert "formats" in ModelManifest.model_fields

        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_formats_round_trip"
        )
        formats = _provenance(model_id)["formats"]
        assert "joblib" in formats
        assert formats == load_manifest(model_id).formats
        written = json.loads(
            (Path(_artifacts.run_dir(model_id)) / "manifest.json").read_text()
        )
        assert written["formats"] == formats


class TestTheOtherViewsAreUnchanged:
    def test_lineage_returns_the_keys_it_always_did(self, patched_multi_factory):
        """The expensive package verification stays in `lineage` and the
        new view neither borrows from it nor changes it."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_lineage_is_untouched"
        )
        lineage = inspect_model(
            InspectModelInput(model_id=model_id, view="lineage")
        ).data
        assert set(lineage) == {
            "dataset_id",
            "dataset_hash",
            "data_sources",
            "oos_predictions_uri",
            "random_seed",
            "git_commit_sha",
            "package_version",
            "created_at_utc",
            "dataset_warnings",
            "environment",
            "package",
        }
        assert lineage["package"]["ok"] is True
        assert "package" not in _provenance(model_id)


class TestCollidingEntityKeys:
    """Two keys, one provider symbol: one series under two identities."""

    def test_the_build_refuses_before_a_single_bar_is_fetched(
        self, patched_multi_factory
    ):
        with pytest.raises(ValidationError, match="both fetch as 'AAA'"):
            build_model_dataset(
                BuildModelDatasetInput(
                    spec=_dataset_spec(universe=["AAA@XNYS", "AAA@XASX", "BBB"])
                )
            )
        assert patched_multi_factory.get_ohlcv_async.call_count == 0
        assert patched_multi_factory.get_ohlcv.call_count == 0

    def test_the_spec_check_sees_a_registered_panel_s_collision(
        self, tmp_path, monkeypatch
    ):
        """A panel registered by reference fetched nothing, so it CAN
        carry a colliding pair; everything that later fetches prices for
        it would refuse. The check reads the dataset's metadata, so the
        refusal arrives with no provider constructed at all."""
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": pd.date_range("2024-01-01", periods=60, freq="B"),
                        "entity": entity,
                        "alpha": np.linspace(0, 1, 60),
                        "target": np.linspace(0, 0.01, 60),
                    }
                )
                for entity in ("BHP@XNYS", "BHP@XASX")
            ],
            ignore_index=True,
        )
        path = tmp_path / "colliding_keys.parquet"
        frame.to_parquet(path, index=False)
        registered = register_external_panel(
            RegisterExternalPanelInput(path=str(path), horizon=5)
        )

        provider_factory = MagicMock(
            side_effect=AssertionError("no provider may be constructed here")
        )
        monkeypatch.setattr(DataFactory, "get_provider", provider_factory)
        result = validate_model_spec(
            ValidateModelSpecInput(
                spec=_a_model_spec(), dataset_id=registered.dataset_id
            )
        )
        assert result.valid is False
        collisions = [p for p in result.problems if "both fetch as" in p.problem]
        assert len(collisions) == 1
        assert collisions[0].where == "dataset_id"
        assert "distinct provider symbols" in collisions[0].suggestion
        assert provider_factory.call_count == 0

    def test_distinct_keys_are_reported_as_no_problem(self, patched_multi_factory):
        """The null case: venue-qualified keys that resolve to different
        symbols are not a collision."""
        built = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec(universe=VENUE_QUALIFIED))
        )
        result = validate_model_spec(
            ValidateModelSpecInput(spec=_a_model_spec(), dataset_id=built.dataset_id)
        )
        assert [p for p in result.problems if "both fetch as" in p.problem] == []


@needs_calendars
class TestTheAdoptedCalendarIsNamed:
    """A calendar nothing asked for is part of the dataset's identity."""

    def test_both_the_build_and_the_check_name_it(self, patched_multi_factory):
        built = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec(universe=VENUE_QUALIFIED))
        )
        adopted = [
            w
            for w in built.warnings
            if w.startswith("calendar adopted from the universe's venue: XNYS")
        ]
        assert len(adopted) == 1

        meta = json.loads(
            (
                Path(_artifacts.run_dir(built.dataset_id)) / "dataset_meta.json"
            ).read_text()
        )
        assert meta["calendar_adopted_from_venue"] == "XNYS"

        checked = validate_model_spec(
            ValidateModelSpecInput(spec=_a_model_spec(), dataset_id=built.dataset_id)
        )
        assert [
            w
            for w in checked.warnings
            if w.startswith("calendar adopted from the universe's venue: XNYS")
        ]

    def test_an_explicit_calendar_is_not_reported_as_adopted(
        self, patched_multi_factory
    ):
        """The null case. The same universe, the same resulting calendar,
        but asked for -- so nothing was inferred and nothing is said."""
        built = build_model_dataset(
            BuildModelDatasetInput(
                spec=_dataset_spec(universe=VENUE_QUALIFIED, calendar="XNYS")
            )
        )
        assert [w for w in built.warnings if "calendar adopted" in w] == []
        meta = json.loads(
            (
                Path(_artifacts.run_dir(built.dataset_id)) / "dataset_meta.json"
            ).read_text()
        )
        assert meta["calendar_adopted_from_venue"] is None

        checked = validate_model_spec(
            ValidateModelSpecInput(spec=_a_model_spec(), dataset_id=built.dataset_id)
        )
        assert [w for w in checked.warnings if "calendar adopted" in w] == []
