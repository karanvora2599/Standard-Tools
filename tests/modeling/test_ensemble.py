"""
Combining models, and the four ways it goes quietly wrong.

WHAT THESE PIN.

  1. Averaging LEVELS from models on different scales produces a number
     dominated by whichever has the wider spread. That is a fact about its
     units, not about its skill, and it is invisible in the result.
  2. Averaging a probability with a return is arithmetic on incomparable
     quantities and is refused rather than performed.
  3. Two models that agree add nothing, and the ensemble's own score cannot
     tell you so -- the pairwise correlation can.
  4. Models validated over different windows share fewer rows than either
     has, and the shortest one silently shortens the ensemble.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.ensemble import combine_predictions

DATES = pd.date_range("2024-01-01", periods=40, freq="B")
ENTITIES = ["AAA", "BBB", "CCC", "DDD"]


class _Manifest:
    def __init__(self, model_id, task, target_id, uri):
        self.model_id = model_id
        self.task = task
        self.target_id = target_id
        self.oos_predictions_uri = uri
        self.content_hashes = {}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A stand-in registry: model_id -> a persisted predictions frame."""
    store = {}

    def _register(model_id, frame, *, task="regression", target_id="forward_return:5"):
        path = tmp_path / f"{model_id}.parquet"
        frame.to_parquet(path)
        store[model_id] = _Manifest(model_id, task, target_id, str(path))
        return model_id

    monkeypatch.setattr(
        "standard_quant_tools.modeling.ensemble.load_manifest",
        lambda model_id: store[model_id],
    )
    # The hash check is exercised in tests/modeling/test_provenance_*; here
    # the manifests carry no recorded hash, so verification is a no-op.
    monkeypatch.setattr(
        "standard_quant_tools.modeling.ensemble._artifacts.verify_file",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "standard_quant_tools.modeling.ensemble._artifacts.load_artifact",
        lambda uri: pd.read_parquet(uri),
    )
    _register.store = store
    return _register


def _predictions(values, dates=DATES, entities=ENTITIES) -> pd.DataFrame:
    rows = []
    for i, date in enumerate(dates):
        for j, entity in enumerate(entities):
            rows.append(
                {"date": date, "entity": entity, "prediction": float(values(i, j))}
            )
    return pd.DataFrame(rows)


class TestScaleDoesNotDecideTheEnsemble:
    def test_mean_is_dominated_by_the_wider_series(self, registry) -> None:
        """
        THE REASON rank_mean IS THE DEFAULT. `wide` says the same thing as
        `narrow` about ordering and says it a thousand times louder, so
        their level-average is essentially `wide` alone.
        """
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, (40, 4))
        registry("narrow", _predictions(lambda i, j: base[i, j] * 0.001))
        registry("wide", _predictions(lambda i, j: -base[i, j] * 1000.0))

        averaged = combine_predictions(["narrow", "wide"], method="mean")
        # The wide series is the negative of the narrow one, so a fair
        # combination is near zero and a dominated one tracks `wide`.
        assert abs(averaged["predictions"]["prediction"]).mean() > 1.0

        ranked = combine_predictions(["narrow", "wide"], method="rank_mean")
        # Opposite orderings cancel: every rank pair sums to zero.
        assert abs(ranked["predictions"]["prediction"]).max() < 1e-9

    def test_rank_mean_is_scale_free(self, registry) -> None:
        rng = np.random.default_rng(2)
        base = rng.normal(0, 1, (40, 4))
        registry("a", _predictions(lambda i, j: base[i, j]))
        registry("b", _predictions(lambda i, j: base[i, j] * 5000.0))
        ranked = combine_predictions(["a", "b"], method="rank_mean")
        # Identical orderings, so the combination equals either one's rank.
        assert ranked["correlations"]["a|b"] == pytest.approx(1.0)


class TestIncomparableQuantitiesAreRefused:
    def test_a_probability_and_a_return_do_not_average(self, registry) -> None:
        rng = np.random.default_rng(3)
        registry("reg", _predictions(lambda i, j: rng.normal(0, 0.01)))
        registry(
            "clf",
            _predictions(lambda i, j: rng.uniform(0, 1)),
            task="classification",
            target_id="forward_direction:5",
        )
        with pytest.raises(ValidationError, match="incomparable units"):
            combine_predictions(["reg", "clf"])

    def test_regression_and_ranking_do_combine(self, registry) -> None:
        """Both emit a continuous score whose ordering is the meaning."""
        rng = np.random.default_rng(4)
        registry("reg", _predictions(lambda i, j: rng.normal(0, 0.01)))
        registry(
            "rnk",
            _predictions(lambda i, j: rng.normal(0, 0.01)),
            task="ranking",
        )
        assert combine_predictions(["reg", "rnk"])["n_rows"] > 0

    def test_different_targets_warn_rather_than_refuse(self, registry) -> None:
        rng = np.random.default_rng(5)
        registry("h5", _predictions(lambda i, j: rng.normal()))
        registry(
            "h20",
            _predictions(lambda i, j: rng.normal()),
            target_id="forward_return:20",
        )
        report = combine_predictions(["h5", "h20"])
        assert any("DIFFERENT targets" in w for w in report["warnings"])


class TestAgreementIsReported:
    def test_two_identical_models_correlate_at_one(self, registry) -> None:
        """
        The number that says the ensemble was pointless. Its own score
        cannot show this -- it looks exactly like a good single model.
        """
        rng = np.random.default_rng(6)
        base = rng.normal(0, 1, (40, 4))
        registry("x", _predictions(lambda i, j: base[i, j]))
        registry("y", _predictions(lambda i, j: base[i, j] + 1e-9))
        report = combine_predictions(["x", "y"], method="mean")
        assert report["correlations"]["x|y"] == pytest.approx(1.0, abs=1e-6)

    def test_uncorrelated_models_are_visible_as_such(self, registry) -> None:
        rng = np.random.default_rng(7)
        registry("p", _predictions(lambda i, j: rng.normal()))
        registry("q", _predictions(lambda i, j: rng.normal()))
        report = combine_predictions(["p", "q"], method="mean")
        assert abs(report["correlations"]["p|q"]) < 0.4


class TestCoverage:
    def test_only_rows_every_model_covered_are_combined(self, registry) -> None:
        rng = np.random.default_rng(8)
        registry("long", _predictions(lambda i, j: rng.normal()))
        registry(
            "short",
            _predictions(lambda i, j: rng.normal(), dates=DATES[:20]),
        )
        report = combine_predictions(["long", "short"], method="mean")
        assert report["rows_per_model"] == {"long": 160, "short": 80}
        assert report["rows_covered_by_all"] == 80
        assert any("covered by EVERY model" in w for w in report["warnings"])

    def test_models_sharing_no_rows_are_refused(self, registry) -> None:
        rng = np.random.default_rng(9)
        registry("early", _predictions(lambda i, j: rng.normal(), dates=DATES[:15]))
        registry("late", _predictions(lambda i, j: rng.normal(), dates=DATES[25:]))
        with pytest.raises(ValidationError, match="nothing to combine"):
            combine_predictions(["early", "late"])


class TestTheArgumentsAreCheckedRatherThanGuessed:
    def test_one_model_is_not_an_ensemble(self, registry) -> None:
        registry("solo", _predictions(lambda i, j: 0.0))
        with pytest.raises(ValidationError, match="at least two"):
            combine_predictions(["solo"])

    def test_a_repeated_model_is_refused_not_double_weighted(self, registry) -> None:
        """A model listed twice is a weighting decision disguised as a typo."""
        rng = np.random.default_rng(10)
        registry("a", _predictions(lambda i, j: rng.normal()))
        registry("b", _predictions(lambda i, j: rng.normal()))
        with pytest.raises(ValidationError, match="repeats"):
            combine_predictions(["a", "b", "a"])

    def test_weights_without_the_weighted_method_are_refused(self, registry) -> None:
        """Silently ignoring a weighting the caller asked for is how an
        ensemble ends up not being the one anybody designed."""
        rng = np.random.default_rng(11)
        registry("a", _predictions(lambda i, j: rng.normal()))
        registry("b", _predictions(lambda i, j: rng.normal()))
        with pytest.raises(ValidationError, match="ignores them"):
            combine_predictions(["a", "b"], method="mean", weights=[0.7, 0.3])

    def test_weighted_needs_one_weight_per_model(self, registry) -> None:
        rng = np.random.default_rng(12)
        registry("a", _predictions(lambda i, j: rng.normal()))
        registry("b", _predictions(lambda i, j: rng.normal()))
        with pytest.raises(ValidationError, match="one weight per model"):
            combine_predictions(["a", "b"], method="weighted", weights=[1.0])

    def test_weights_summing_to_zero_are_refused(self, registry) -> None:
        rng = np.random.default_rng(13)
        registry("a", _predictions(lambda i, j: rng.normal()))
        registry("b", _predictions(lambda i, j: rng.normal()))
        with pytest.raises(ValidationError, match="sum to zero"):
            combine_predictions(["a", "b"], method="weighted", weights=[1.0, -1.0])

    def test_weighting_actually_weights(self, registry) -> None:
        registry("a", _predictions(lambda i, j: 1.0))
        registry("b", _predictions(lambda i, j: 3.0))
        report = combine_predictions(["a", "b"], method="weighted", weights=[3.0, 1.0])
        assert report["predictions"]["prediction"].iloc[0] == pytest.approx(1.5)

    def test_an_unknown_method_names_the_real_ones(self, registry) -> None:
        registry("a", _predictions(lambda i, j: 0.0))
        registry("b", _predictions(lambda i, j: 0.0))
        with pytest.raises(ValidationError, match="rank_mean"):
            combine_predictions(["a", "b"], method="geometric")


class TestSingleEntityDatesHaveNoCrossSection:
    def test_rank_mean_drops_them_rather_than_calling_them_average(
        self, registry
    ) -> None:
        """A rank needs someone to rank against. 0.0 would read as 'exactly
        average' rather than 'no information'."""
        rng = np.random.default_rng(14)
        registry("a", _predictions(lambda i, j: rng.normal(), entities=["AAA"]))
        registry("b", _predictions(lambda i, j: rng.normal(), entities=["AAA"]))
        with pytest.raises(ValidationError, match="no cross-section"):
            combine_predictions(["a", "b"], method="rank_mean")

    def test_mean_still_works_on_a_single_entity(self, registry) -> None:
        registry("a", _predictions(lambda i, j: 2.0, entities=["AAA"]))
        registry("b", _predictions(lambda i, j: 4.0, entities=["AAA"]))
        report = combine_predictions(["a", "b"], method="mean")
        assert report["predictions"]["prediction"].iloc[0] == pytest.approx(3.0)
