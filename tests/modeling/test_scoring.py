"""Tests for modeling.scoring.score_model: missing_entities reporting,
as_of validation, and that the reconstructed scoring DatasetSpec is
actually re-validated (not silently bypassed via model_copy)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

from .conftest import make_ohlcv, make_provider_mock


def _dataset_spec(**overrides) -> DatasetSpec:
    defaults = dict(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    defaults.update(overrides)
    return DatasetSpec(**defaults)


def _train_a_model_with_spec(
    spec: DatasetSpec, dataset_id: str = "ds_scoring_test"
) -> str:
    """Builds+trains a model exactly the way build_model_dataset +
    run_model_experiment would, but persists dataset_spec.json by hand
    (the agent tool normally does this; scoring.py depends on it
    existing) -- returns the trained model_id."""
    from pathlib import Path

    from standard_quant_tools.modeling import artifacts as _artifacts

    built = build_dataset(spec)
    panel_uri = _artifacts.save_artifact(
        built["panel"], run_id=dataset_id, name="panel"
    )
    directory = Path(panel_uri).parent
    _artifacts.save_json(directory, "dataset_spec", spec.model_dump())

    model_spec = ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )
    dataset = {
        "panel": built["panel"],
        "feature_ids": built["feature_ids"],
        "target_id": built["target_id"],
        "data_hash": built["data_hash"],
        # Mirrors what the run_model_experiment agent tool passes, so the
        # registered model bundles (and content-verifies) its own copy of
        # the training spec rather than reading the dataset directory's.
        "spec_hash": built["spec_hash"],
        "dataset_spec": spec.model_dump(),
    }
    result = run_experiment(dataset, model_spec, dataset_id=dataset_id)
    return result["model_id"]


def _train_a_model(patched_multi_factory) -> str:
    return _train_a_model_with_spec(_dataset_spec())


class TestScoreModelReliability:
    def test_missing_entities_reported_not_silently_dropped(self, monkeypatch):
        """CCC gets only 8 bars of OHLCV -- not enough for RSI's 14-bar
        lookback to ever produce a non-NaN value, so CCC contributes zero
        rows to both the training panel and the scoring snapshot. It
        must show up in missing_entities, not be silently absent from a
        result that otherwise looks like it succeeded for everyone."""

        def _fetch(symbol):
            if symbol == "CCC":
                return make_ohlcv(symbol, n=8)
            return make_ohlcv(symbol)

        provider = make_provider_mock(_fetch)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        spec = _dataset_spec(
            features=[FeatureSpec(id="technical.rsi")], universe=["AAA", "BBB"]
        )
        model_id = _train_a_model_with_spec(spec)

        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )
        assert result["missing_entities"] == ["CCC"]
        assert result["n_entities"] == 2

    def test_malformed_as_of_raises_validation_error_not_raw_pandas_error(
        self, patched_multi_factory
    ):
        model_id = _train_a_model(patched_multi_factory)
        with pytest.raises(ValidationError, match="not a valid date"):
            score_model(model_id=model_id, as_of="not-a-date", universe=["AAA"])

    def test_scoring_spec_reconstruction_is_actually_validated(
        self, patched_multi_factory
    ):
        """Regression test for the model_copy(update=...) bug: passing a
        universe with duplicate symbols to score_model must be rejected
        by DatasetSpec's validator (raised as a pydantic ValidationError,
        the same as any other invalid DatasetSpec construction), not
        silently accepted the way model_copy(update=...) would have."""
        model_id = _train_a_model(patched_multi_factory)
        with pytest.raises(PydanticValidationError, match="duplicate symbols"):
            score_model(model_id=model_id, as_of="2023-12-29", universe=["AAA", "AAA"])

    def test_successful_score_has_no_missing_entities(self, patched_multi_factory):
        model_id = _train_a_model(patched_multi_factory)
        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )
        assert result["missing_entities"] == []
        assert result["n_entities"] == 3


class TestSingleEffectiveScoreDate:
    """
    score_model took each entity's OWN latest surviving row
    (`groupby("entity").tail(1)`), so a symbol that stopped trading earlier
    contributed an older bar and was returned inside what the response
    called one `as_of` cross-section.

    For a cross-sectional model that is not a smaller cross-section — it is
    a ranking that no longer compares contemporaneous information.
    `missing_entities` never caught it, because it only saw entities with NO
    row at all, so a stale entity looked like a fully successful score.
    """

    def _provider_with_a_stale_symbol(self, monkeypatch, stale_bars: int = 5):
        """CCC has plenty of history (so it is NOT missing) but its last bar
        predates everyone else's by `stale_bars` sessions."""

        def _fetch(symbol):
            df = make_ohlcv(symbol)
            if symbol == "CCC":
                return df.iloc[:-stale_bars]
            return df

        provider = make_provider_mock(_fetch)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

    def test_stale_entity_is_excluded_and_reported(self, monkeypatch):
        self._provider_with_a_stale_symbol(monkeypatch)
        spec = _dataset_spec(
            features=[FeatureSpec(id="technical.rsi")], universe=["AAA", "BBB"]
        )
        model_id = _train_a_model_with_spec(spec, dataset_id="ds_stale")

        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )

        assert "CCC" in result["stale_entities"], (
            "a symbol scored from an older bar must be reported, not silently "
            "folded into the cross-section"
        )
        # Reported with the date it actually had, so the caller can judge.
        assert result["stale_entities"]["CCC"] < result["effective_score_date"]
        # And excluded rather than scored on that older bar.
        assert result["n_entities"] == 2
        # "no data at all" and "older data" are different conditions.
        assert "CCC" not in result["missing_entities"]

    def test_all_returned_predictions_share_one_date(self, monkeypatch):
        self._provider_with_a_stale_symbol(monkeypatch)
        spec = _dataset_spec(
            features=[FeatureSpec(id="technical.rsi")], universe=["AAA", "BBB"]
        )
        model_id = _train_a_model_with_spec(spec, dataset_id="ds_stale_uniform")

        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )

        from standard_quant_tools.modeling import artifacts as _artifacts

        preds = _artifacts.load_artifact(result["predictions_uri"])
        assert preds["date"].nunique() == 1, "cross-section mixes observation dates"
        assert (
            pd.Timestamp(preds["date"].iloc[0]).strftime("%Y-%m-%d")
            == result["effective_score_date"]
        )

    def test_effective_score_date_is_reported_separately_from_as_of(
        self, patched_multi_factory
    ):
        """
        as_of is what was REQUESTED; effective_score_date is what the data
        actually supported. They differ whenever the most recent bar at or
        before as_of is earlier — a holiday, a weekend, a provider window
        ending earlier — and the caller must not have to assume they match.
        """
        model_id = _train_a_model(patched_multi_factory)
        result = score_model(
            model_id=model_id, as_of="2023-12-31", universe=["AAA", "BBB"]
        )
        assert result["as_of"] == "2023-12-31"
        assert result["effective_score_date"] <= result["as_of"]
        assert result["effective_score_date"]  # populated, not left blank


class TestMaxStaleness:
    """
    A single cross-section date makes predictions internally consistent but
    says nothing about how OLD that date is. A universe whose data stopped
    months ago still produces a perfectly uniform — and entirely stale —
    cross-section, which previously came back looking like a completely
    successful score.
    """

    def test_staleness_days_is_always_reported(self, patched_multi_factory):
        model_id = _train_a_model(patched_multi_factory)
        result = score_model(
            model_id=model_id, as_of="2023-12-31", universe=["AAA", "BBB"]
        )
        assert "staleness_days" in result
        assert result["staleness_days"] >= 0
        # It must actually agree with the two dates it sits between.
        gap = (
            pd.Timestamp(result["as_of"]) - pd.Timestamp(result["effective_score_date"])
        ).days
        assert result["staleness_days"] == gap

    def test_stale_universe_rejected_when_a_limit_is_set(self, patched_multi_factory):
        """The whole universe's data ends well before as_of — not a
        per-symbol gap, which is why the message says so."""
        model_id = _train_a_model(patched_multi_factory)
        with pytest.raises(ValidationError, match="max_staleness_days"):
            score_model(
                model_id=model_id,
                as_of="2023-12-31",
                universe=["AAA", "BBB"],
                lookback_days=800,
                max_staleness_days=1,
            )

    def test_no_limit_means_no_check(self, patched_multi_factory):
        """Default stays permissive: how much staleness is still
        decision-useful is a property of the strategy, not something
        score_model can pick on the caller's behalf."""
        model_id = _train_a_model(patched_multi_factory)
        result = score_model(
            model_id=model_id, as_of="2023-12-31", universe=["AAA", "BBB"]
        )
        assert result["n_entities"] > 0

    def test_generous_limit_passes(self, patched_multi_factory):
        model_id = _train_a_model(patched_multi_factory)
        result = score_model(
            model_id=model_id,
            as_of="2023-12-31",
            universe=["AAA", "BBB"],
            max_staleness_days=3650,
        )
        assert result["n_entities"] == 2


class TestUniverseScopeFeatureLock:
    """
    score_model deliberately permits a different scoring universe, which is
    correct for entity-scope features: AAPL's RSI does not change because
    MSFT was added to the request.

    It is wrong for a UNIVERSE-scope feature. factors.pca_loading and
    pca_factor_return are computed from the entire universe's return matrix,
    so a model trained on one set and scored on another receives a different
    factor basis under the same feature column — a silently different
    variable, not a smaller sample.
    """

    def test_different_universe_rejected_when_a_pca_feature_is_used(
        self, patched_multi_factory
    ):
        spec = _dataset_spec(
            universe=["AAA", "BBB", "CCC"],
            features=[
                FeatureSpec(id="technical.rsi"),
                FeatureSpec(id="factors.pca_loading"),
            ],
        )
        model_id = _train_a_model_with_spec(spec, dataset_id="ds_univ_lock")

        with pytest.raises(ValidationError, match="universe-scope feature"):
            score_model(
                model_id=model_id,
                as_of="2023-12-29",
                universe=["AAA", "BBB"],  # a subset is still a different basis
            )

    def test_same_universe_still_scores(self, patched_multi_factory):
        spec = _dataset_spec(
            universe=["AAA", "BBB", "CCC"],
            features=[
                FeatureSpec(id="technical.rsi"),
                FeatureSpec(id="factors.pca_loading"),
            ],
        )
        model_id = _train_a_model_with_spec(spec, dataset_id="ds_univ_same")

        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["CCC", "AAA", "BBB"]
        )
        # Order must not matter — it's a set, not a sequence.
        assert result["n_entities"] == 3

    def test_entity_scope_only_model_still_allows_a_new_universe(
        self, patched_multi_factory
    ):
        """The permission this lock must NOT take away: with only
        entity-scope features, each symbol's values are independent of which
        other symbols were requested."""
        spec = _dataset_spec(
            universe=["AAA", "BBB"],
            features=[FeatureSpec(id="technical.rsi")],
        )
        model_id = _train_a_model_with_spec(spec, dataset_id="ds_entity_only")

        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )
        assert result["n_entities"] >= 2


class TestFeatureImplementationDrift:
    """
    The manifest recorded each column's implementation hash at
    registration, but score_model never compared it against today's code.
    So editing a feature function and then scoring an existing model fed
    the registered estimator a differently-defined input under the same
    column name — the provenance recorded that something had changed, but
    only after the fact, for anyone who went looking.
    """

    def test_changed_feature_implementation_blocks_scoring(
        self, patched_multi_factory, monkeypatch
    ):
        from standard_quant_tools.modeling.features.registry import FEATURE_REGISTRY

        model_id = _train_a_model(patched_multi_factory)

        # Swap in a different implementation for a feature the model uses,
        # exactly as editing its source would.
        original = FEATURE_REGISTRY["technical.rsi"]

        def _different_rsi(ohlcv, context, period: int = 14):
            return ohlcv["Close"].pct_change(period) * 100.0

        monkeypatch.setattr(
            FEATURE_REGISTRY["technical.rsi"], "fn", _different_rsi, raising=True
        )

        with pytest.raises(ValidationError, match="implementation of"):
            score_model(model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB"])
        assert FEATURE_REGISTRY["technical.rsi"] is original  # monkeypatch scope only

    def test_unchanged_implementation_still_scores(self, patched_multi_factory):
        """The guard must not fire on an untouched codebase."""
        model_id = _train_a_model(patched_multi_factory)
        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB"]
        )
        assert result["n_entities"] == 2
