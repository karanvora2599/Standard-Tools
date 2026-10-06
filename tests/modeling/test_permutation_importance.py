"""
Permutation importance: what the model loses when a feature is scrambled.

`run_feature_ablation` was the only model-relative importance here, and the
library calls it "EXPENSIVE -- 40 features at 8 folds is 328 fits" and tells
agents to narrow away from it, which inverts the question it answers: you
narrow using the thing you wanted the answer in order to narrow. Permutation
costs one PREDICT per feature per repeat per fold instead of one FIT.

They do not measure the same thing. Ablation REFITS without the feature, so
the remaining features take over its job -- "would a model built without
this have been worse". Permutation keeps the fitted model and destroys the
feature's information -- "does THIS model use it". A feature with a perfect
substitute scores near zero under ablation and high under permutation, and
the last test here plants exactly that case.

The shuffle is WITHIN the date. A global one would move a value into
another date's cross-section, breaking the panel as well as the feature, so
the drop would confound the two.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.validation.permutation import (
    permutation_importance,
    shuffle_within_date,
    summarize_permutation,
)


def _codes(*counts):
    out = []
    for index, count in enumerate(counts):
        out.extend([index] * count)
    return np.asarray(out)


class TestTheShuffleStaysInsideTheDate:
    def test_each_date_keeps_its_own_values(self):
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        codes = _codes(3, 3)
        out = shuffle_within_date(values, codes, np.random.default_rng(0))
        assert sorted(out[:3]) == [1.0, 2.0, 3.0]
        assert sorted(out[3:]) == [10.0, 20.0, 30.0]

    def test_a_one_row_date_is_unchanged(self):
        """Nothing to permute it with, and leaving it alone keeps the
        baseline and the permuted score comparable on those rows."""
        values = np.array([7.0, 1.0, 2.0])
        out = shuffle_within_date(values, _codes(1, 2), np.random.default_rng(0))
        assert out[0] == 7.0

    def test_it_does_permute_something(self):
        """Or every drop would be zero and the measure would be vacuous."""
        values = np.arange(50.0)
        codes = np.zeros(50, dtype=int)
        out = shuffle_within_date(values, codes, np.random.default_rng(0))
        assert not np.array_equal(values, out)

    def test_rows_need_not_arrive_sorted_by_date(self):
        """The codes group the rows; the frame is not reordered."""
        values = np.array([1.0, 10.0, 2.0, 20.0])
        codes = np.asarray([0, 1, 0, 1])
        out = shuffle_within_date(values, codes, np.random.default_rng(3))
        assert sorted([out[0], out[2]]) == [1.0, 2.0]
        assert sorted([out[1], out[3]]) == [10.0, 20.0]


class TestTheDrop:
    @staticmethod
    def _panel(n_dates=40, per_date=10, seed=0):
        rng = np.random.default_rng(seed)
        rows = n_dates * per_date
        signal = rng.normal(size=rows)
        noise = rng.normal(size=rows)
        dates = np.repeat(
            np.arange(n_dates).astype("datetime64[D]").astype("datetime64[ns]"),
            per_date,
        )
        X = pd.DataFrame({"signal": signal, "noise": noise})
        y = signal * 2.0 + rng.normal(scale=0.1, size=rows)
        return X, y, dates

    def test_a_used_feature_drops_more_than_an_unused_one(self):
        X, y, dates = self._panel()

        def score(frame):
            # A model that reads `signal` and ignores `noise`.
            return -float(np.mean((frame["signal"].to_numpy() * 2.0 - y) ** 2))

        result = permutation_importance(score, X, dates, n_repeats=3, seed=0)
        columns = result["columns"]
        assert columns["signal"]["mean_drop"] > columns["noise"]["mean_drop"]
        # The unused one is not merely smaller: it is nothing.
        assert columns["noise"]["mean_drop"] == pytest.approx(0.0, abs=1e-12)

    def test_the_baseline_is_the_unpermuted_score(self):
        X, y, dates = self._panel()
        score = lambda frame: float(frame["signal"].sum())  # noqa: E731
        result = permutation_importance(score, X, dates, n_repeats=1, seed=0)
        assert result["baseline"] == pytest.approx(float(X["signal"].sum()))

    def test_a_negative_drop_is_reported_not_clipped(self):
        """The model scoring BETTER on scrambled values is evidence the
        feature is noise it is fitting. Clipping to zero would hide it."""
        X, _y, dates = self._panel()
        calls = {"n": 0}

        def score(frame):
            calls["n"] += 1
            return 0.0 if calls["n"] == 1 else 1.0

        result = permutation_importance(score, X, dates, columns=["noise"], n_repeats=2, seed=0)
        assert result["columns"]["noise"]["mean_drop"] < 0

    def test_the_spread_across_repeats_comes_back(self):
        X, y, dates = self._panel()
        score = lambda frame: -float(np.mean((frame["signal"].to_numpy() * 2.0 - y) ** 2))  # noqa: E731
        result = permutation_importance(score, X, dates, n_repeats=4, seed=1)
        record = result["columns"]["signal"]
        assert record["n_repeats"] == 4
        assert record["std_drop"] > 0

    def test_the_same_seed_gives_the_same_answer(self):
        X, y, dates = self._panel()
        score = lambda frame: -float(np.mean((frame["signal"].to_numpy() * 2.0 - y) ** 2))  # noqa: E731
        first = permutation_importance(score, X, dates, n_repeats=3, seed=7)
        second = permutation_importance(score, X, dates, n_repeats=3, seed=7)
        assert first == second

    def test_only_the_named_columns_are_measured(self):
        X, y, dates = self._panel()
        score = lambda frame: float(frame.sum().sum())  # noqa: E731
        result = permutation_importance(score, X, dates, columns=["noise"], n_repeats=1)
        assert set(result["columns"]) == {"noise"}


class TestItRefusesRatherThanGuess:
    def test_mismatched_dates(self):
        X = pd.DataFrame({"f": [1.0, 2.0]})
        with pytest.raises(ValidationError, match="one date per row"):
            permutation_importance(lambda f: 0.0, X, np.array([1]), n_repeats=1)

    def test_an_unknown_column(self):
        X = pd.DataFrame({"f": [1.0, 2.0]})
        with pytest.raises(ValidationError, match="no such column"):
            permutation_importance(
                lambda f: 0.0, X, _codes(2), columns=["ghost"], n_repeats=1
            )

    def test_zero_repeats(self):
        X = pd.DataFrame({"f": [1.0, 2.0]})
        with pytest.raises(ValidationError, match="at least 1"):
            permutation_importance(lambda f: 0.0, X, _codes(2), n_repeats=0)

    def test_a_non_finite_baseline(self):
        """No drop from it would mean anything."""
        X = pd.DataFrame({"f": [1.0, 2.0]})
        with pytest.raises(ValidationError, match="baseline score is not finite"):
            permutation_importance(
                lambda f: float("nan"), X, _codes(2), n_repeats=1
            )


class TestTheSummary:
    def test_the_spread_across_folds_is_what_it_reports(self):
        """A feature that mattered in one fold and not the others mattered
        in one regime; a mean alone hides that."""
        folds = [
            {"baseline": 1.0, "columns": {"a": {"mean_drop": 1.0}, "b": {"mean_drop": 0.5}}},
            {"baseline": 1.0, "columns": {"a": {"mean_drop": 0.0}, "b": {"mean_drop": 0.5}}},
        ]
        summary = summarize_permutation(folds)
        assert summary["a"]["mean_drop"] == pytest.approx(0.5)
        assert summary["b"]["mean_drop"] == pytest.approx(0.5)
        # Same mean, and only one of them is stable.
        assert summary["a"]["std_drop_across_folds"] > summary["b"]["std_drop_across_folds"]

    def test_a_feature_missing_from_a_fold_is_not_counted_against_it(self):
        folds = [
            {"columns": {"a": {"mean_drop": 1.0}, "b": {"mean_drop": 2.0}}},
            {"columns": {"a": {"mean_drop": 1.0}}},
        ]
        summary = summarize_permutation(folds)
        assert summary["a"]["n_folds"] == 2
        assert summary["b"]["n_folds"] == 1
        assert summary["b"]["mean_drop"] == pytest.approx(2.0)

    def test_no_folds_is_an_empty_summary(self):
        assert summarize_permutation([]) == {}


class TestItAnswersADifferentQuestionFromAblation:
    def test_a_duplicated_feature_still_scores(self):
        """The case that separates the two measures. Ablation would refit
        and the twin would take over, so dropping either costs nothing.
        Permutation keeps the fitted model, which reads BOTH, so
        scrambling either one really does hurt it.
        """
        rng = np.random.default_rng(0)
        rows = 300
        signal = rng.normal(size=rows)
        X = pd.DataFrame({"a": signal, "twin": signal.copy()})
        dates = np.repeat(
            np.arange(30).astype("datetime64[D]").astype("datetime64[ns]"), 10
        )
        y = signal * 2.0

        def score(frame):
            # A model that averages the two, which is what a fit on
            # perfectly collinear inputs tends to produce.
            blended = (frame["a"].to_numpy() + frame["twin"].to_numpy())
            return -float(np.mean((blended - y) ** 2))

        result = permutation_importance(score, X, dates, n_repeats=3, seed=0)
        assert result["columns"]["a"]["mean_drop"] > 0
        assert result["columns"]["twin"]["mean_drop"] > 0


class TestThroughTheEngine:
    """The fold loop is the only place this is cheap: the estimator is
    fitted and the test rows are sliced. It is opt-in because it costs
    predictions, unlike `coef_` which the estimator computed while
    fitting."""

    def test_a_run_reports_it_when_asked(self, patched_multi_factory):
        from standard_quant_tools.modeling.dataset.builder import build_dataset
        from standard_quant_tools.modeling.engine import run_experiment

        from .test_scoring import _dataset_spec

        dataset = build_dataset(_dataset_spec())
        spec = _spec(permutation={"n_repeats": 2, "seed": 0})
        result = run_experiment(dataset, spec, "ds_perm", register=False)
        summary = result["validation_report"]["permutation_importance"]
        assert summary, "nothing came back"
        assert set(summary) == {"technical.rsi", "market.momentum"}
        for record in summary.values():
            assert record["n_folds"] >= 1
            assert "std_drop_across_folds" in record

    def test_it_is_absent_rather_than_empty_when_not_asked(
        self, patched_multi_factory
    ):
        """An absent measurement and a measured zero are different."""
        from standard_quant_tools.modeling.dataset.builder import build_dataset
        from standard_quant_tools.modeling.engine import run_experiment

        from .test_scoring import _dataset_spec

        dataset = build_dataset(_dataset_spec())
        result = run_experiment(dataset, _spec(), "ds_perm_off", register=False)
        assert result["validation_report"]["permutation_importance"] is None


def _spec(permutation=None):
    from standard_quant_tools.modeling.specs import (
        EstimatorSpec,
        ModelSpec,
        ValidationSpec,
    )

    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge"),
        validation=ValidationSpec(
            method="walk_forward", train_window=120, test_window=40, min_folds=1
        ),
        permutation_importance=permutation,
        random_seed=0,
    )
