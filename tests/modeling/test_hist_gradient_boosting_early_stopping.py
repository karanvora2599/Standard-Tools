"""
hist_gradient_boosting stops early on its training window's last dates (see
the CHANGELOG entry of 2026-10-04).

scikit-learn's default, `early_stopping='auto'`, holds out a SHUFFLED 10% of
the training rows above 10,000 of them and stops when the loss on those
rows has not improved for 10 iterations. On a panel those rows sit among
the rows fitted, and an h-day label shares outcomes with its neighbours, so
the stopping point was chosen on rows that are not out of sample in time.
The validation rows are now the window's last `validation_fraction` of its
dates, the rows whose labels reach them are left out of the fit, and
scikit-learn is handed them as `X_val`. `early_stopping`,
`validation_fraction` and `n_iter_no_change` are spec parameters.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.engine import _fit, run_experiment
from standard_quant_tools.modeling.estimators import trees
from standard_quant_tools.modeling.estimators.registry import (
    allowed_params,
    validate_param_value,
    validate_params,
)
from standard_quant_tools.modeling.estimators.trees import (
    AUTO_EARLY_STOPPING_ROWS,
    early_stopping_warnings,
    fit_takes_validation_set,
    time_ordered_validation,
)
from standard_quant_tools.modeling.samples import SampleIndex
from standard_quant_tools.modeling.specs import (
    ConformalSpec,
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)

needs_validation_set = pytest.mark.skipif(
    not fit_takes_validation_set(),
    reason="this scikit-learn's HistGradientBoosting fit takes no X_val (1.7+)",
)


def _rows(n_entities, n_dates, h=5, seed=0, classification=False):
    """A balanced daily panel as fit arrays: features, an h-day label, and
    the sample index with each row's label end h bars ahead."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n_dates + h)
    dates = np.repeat(days[:n_dates].to_numpy(), n_entities)
    ends = np.repeat(days[h : n_dates + h].to_numpy(), n_entities)
    X = rng.normal(size=(n_dates * n_entities, 3))
    y = X[:, 0] * 0.3 + rng.normal(size=len(X))
    if classification:
        y = (y > 0).astype(float)
    entities = np.tile(np.arange(n_entities), n_dates)
    return X, y, SampleIndex(dates=dates, entities=entities, label_end=ends)


def _masks_by_hand(index, fraction, h):
    """The rule written out: the last n - floor(n * (1 - fraction)) dates
    validate; rows on the h dates before them, and rows whose label ends on
    or after the first of them, are left out."""
    window = np.unique(index.dates)
    n_fit_dates = int(np.floor(len(window) * (1.0 - fraction)))
    first = window[n_fit_dates]
    validation = index.dates >= first
    fitted = (index.dates < window[n_fit_dates - h]) & (index.label_end < first)
    return fitted, validation


class TestTheSchema:
    def test_the_three_are_spec_parameters_of_both_tasks(self):
        """A spec can name the stopping rule's three settings for either task;
        they were refused as unknown parameters."""
        for task in ("regression", "classification"):
            names = allowed_params(task, "hist_gradient_boosting")
            assert {"early_stopping", "validation_fraction", "n_iter_no_change"} <= set(
                names
            )

    def test_early_stopping_takes_auto_true_or_false_only(self):
        """scikit-learn's three values, and nothing that only compares equal to
        them: 1 == True in Python, so a membership test alone would run
        early_stopping=1 as True."""
        for value in ("auto", True, False):
            validate_params(
                "regression", "hist_gradient_boosting", {"early_stopping": value}
            )
        # 1 == True in Python; a number is refused rather than read as a flag.
        for value in (1, 0, "true", "True", None, 0.5):
            with pytest.raises(ValidationError, match="early_stopping"):
                validate_params(
                    "regression", "hist_gradient_boosting", {"early_stopping": value}
                )

    def test_validation_fraction_is_a_share_written_as_a_fraction(self):
        """A share of the window's dates between 0.01 and 0.5. A whole number
        is refused because scikit-learn reads an int here as a row count."""
        for value in (0.01, 0.1, 0.5):
            validate_param_value(
                "regression", "hist_gradient_boosting", "validation_fraction", value
            )
        for value in (0.0, 0.6, 1, True, None):
            with pytest.raises(ValidationError, match="validation_fraction"):
                validate_param_value(
                    "regression", "hist_gradient_boosting", "validation_fraction", value
                )

    def test_n_iter_no_change_is_a_positive_whole_number(self):
        """The patience is a count of iterations: a fraction or a bool is refused."""
        validate_param_value(
            "classification", "hist_gradient_boosting", "n_iter_no_change", 25
        )
        for value in (0, 2.5, True):
            with pytest.raises(ValidationError, match="n_iter_no_change"):
                validate_param_value(
                    "classification",
                    "hist_gradient_boosting",
                    "n_iter_no_change",
                    value,
                )

    def test_the_rule_s_settings_are_refused_beside_early_stopping_false(self):
        """With early stopping off the two settings would be ignored without a
        word, so the combination is refused by name; beside 'auto' they apply."""
        for name, value in (("validation_fraction", 0.2), ("n_iter_no_change", 5)):
            with pytest.raises(ValidationError, match=name):
                validate_params(
                    "regression",
                    "hist_gradient_boosting",
                    {"early_stopping": False, name: value},
                )
            validate_params(
                "regression",
                "hist_gradient_boosting",
                {"early_stopping": "auto", name: value},
            )


class TestTheValidationBlock:
    def test_it_is_the_last_tenth_of_the_dates_not_of_the_rows(self):
        """On a panel whose dates carry different row counts, the block is the
        last tenth of the DATES, not of the rows."""
        # 50 dates of 10 rows, then 50 dates of 200: the last 10 dates hold
        # 2,000 of 10,500 rows, so a tenth of the rows would be a different
        # block.
        days = pd.bdate_range("2021-01-01", periods=100).to_numpy()
        dates = np.concatenate([np.repeat(days[:50], 10), np.repeat(days[50:], 200)])
        block, reason = time_ordered_validation(dates, None, 0, 0.1)
        assert reason is None
        assert np.array_equal(np.unique(dates[block["validation_rows"]]), days[90:])
        assert block["n_validation_dates"] == 10
        assert block["validation_rows"].sum() == 2_000
        assert block["validation_start"] == str(pd.Timestamp(days[90]).date())
        assert block["validation_end"] == str(pd.Timestamp(days[-1]).date())

    def test_no_fitted_label_reaches_the_block(self):
        """An entity that skips dates reaches its h-th bar later than the
        panel's calendar does; its rows are dropped by their recorded label
        end, so no fitted label ends inside the block."""
        # One entity skips every other date, so its 5-bar labels end later
        # than five of the panel's dates: those rows reach the block from
        # before the date embargo and must be dropped by their recorded end.
        days = pd.bdate_range("2021-01-01", periods=120)
        frames = []
        for entity in range(4):
            own = days[::2] if entity == 0 else days
            ends = list(own[5:]) + [pd.NaT] * 5
            frames.append(pd.DataFrame({"date": own, "end": ends, "entity": entity}))
        frame = pd.concat(frames, ignore_index=True).dropna()
        dates = frame["date"].to_numpy()
        ends = frame["end"].to_numpy()
        block, _ = time_ordered_validation(dates, ends, 5, 0.1)
        first = np.unique(dates[block["validation_rows"]]).min()
        fitted = block["fit_rows"]
        assert (ends[fitted] < first).all()
        assert (dates[fitted] < first).all()
        # The ragged entity's reach is what the date embargo alone misses.
        window = np.unique(dates)
        last_by_date = window[np.searchsorted(window, first) - 5 - 1]
        reaching = (dates <= last_by_date) & (ends >= first)
        assert reaching.any() and not (fitted & reaching).any()
        assert block["n_embargoed_rows"] == int(
            len(dates) - fitted.sum() - block["validation_rows"].sum()
        )

    @pytest.mark.parametrize("fraction", [0.1, 0.25])
    @pytest.mark.parametrize("h", [1, 5, 12])
    @pytest.mark.parametrize("with_ends", [True, False])
    def test_the_masks_are_the_select_features_holdout_s(self, fraction, h, with_ends):
        """The block and the embargo are select_features' holdout and embargo,
        row for row, on a shuffled ragged panel, with and without recorded
        label ends. The rule is written on arrays for speed, so this holds the
        two together."""
        # The same rows `select_features` would select on and hold out,
        # embargoed by the same rule, on a ragged panel: entities enter late,
        # skip dates and carry recorded label ends of their own.
        from standard_quant_tools.modeling.analysis.feature_selection import (
            _selection_cutoff,
            _split_at_holdout,
        )

        rng = np.random.default_rng(h)
        days = pd.bdate_range("2021-01-01", periods=160)
        frames = []
        for entity in range(6):
            own = days[entity * 3 :: 1 + entity % 3]
            ends = list(own[h:]) + [pd.NaT] * min(h, len(own))
            frames.append(
                pd.DataFrame({"date": own, "label_end_date": ends[: len(own)]})
            )
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
        if not with_ends:
            frame = frame[["date"]]
        window, cutoff = _selection_cutoff(frame, None, fraction)
        split = _split_at_holdout(frame, window, cutoff, h, "test")
        block, _ = time_ordered_validation(
            frame["date"].to_numpy(),
            frame["label_end_date"].to_numpy() if with_ends else None,
            h,
            fraction,
        )
        assert np.array_equal(
            np.flatnonzero(block["fit_rows"]), np.sort(split["selection"].index)
        )
        assert np.array_equal(
            np.flatnonzero(block["validation_rows"]), np.sort(split["holdout"].index)
        )
        assert block["n_embargoed_rows"] == split["embargo_rows"]

    def test_the_rule_written_out(self):
        """On a balanced panel the masks are the rule as written: the last 20
        of 200 dates validate, and the 5 dates before them are embargoed."""
        _X, _y, index = _rows(7, 200, h=5)
        block, _ = time_ordered_validation(index.dates, index.label_end, 5, 0.1)
        fitted, validation = _masks_by_hand(index, 0.1, 5)
        assert np.array_equal(block["fit_rows"], fitted)
        assert np.array_equal(block["validation_rows"], validation)
        assert block["embargo_dates"] == 5

    def test_without_a_horizon_the_recorded_ends_alone_embargo(self):
        """A target_id with no horizon leaves the recorded label ends, which
        alone keep every fitted label out of the block."""
        _X, _y, index = _rows(3, 100, h=4)
        block, _ = time_ordered_validation(index.dates, index.label_end, None, 0.1)
        first = np.unique(index.dates[block["validation_rows"]]).min()
        assert np.array_equal(block["fit_rows"], index.label_end < first)

    def test_without_label_ends_the_horizon_s_dates_embargo(self):
        """With no recorded ends the embargo is the horizon's dates before the block."""
        _X, _y, index = _rows(3, 100, h=4)
        block, _ = time_ordered_validation(index.dates, None, 4, 0.1)
        window = np.unique(index.dates)
        assert np.array_equal(block["fit_rows"], index.dates <= window[89 - 4])

    def test_a_window_too_short_says_why(self):
        """Six dates cannot hold a validation date and a five-date embargo with
        a date left to fit; the reason names the counts."""
        _X, _y, index = _rows(3, 6, h=5)
        block, reason = time_ordered_validation(index.dates, index.label_end, 5, 0.1)
        assert block is None
        assert "6 date(s)" in reason and "no date to fit on" in reason


@needs_validation_set
class TestTheFit:
    @pytest.mark.parametrize("classification", [False, True])
    def test_auto_above_the_threshold_stops_on_the_block(self, classification):
        """Above 10,000 rows 'auto' fits exactly what scikit-learn fits when
        handed the block by hand as X_val, for the regressor and the
        classifier, and the report counts the same rows."""
        X, y, index = _rows(60, 200, classification=classification)
        assert len(y) > AUTO_EARLY_STOPPING_ROWS
        cls = (
            HistGradientBoostingClassifier
            if classification
            else HistGradientBoostingRegressor
        )
        model = cls(random_state=3)
        stopping = _fit(model, X, y, None, index=index, horizon=5)
        fitted, validation = _masks_by_hand(index, 0.1, 5)
        by_hand = cls(random_state=3, early_stopping=True).fit(
            X[fitted], y[fitted], X_val=X[validation], y_val=y[validation]
        )
        assert model.do_early_stopping_ and model.n_iter_ == by_hand.n_iter_
        assert np.array_equal(model.predict(X), by_hand.predict(X))
        report = stopping.report(model)
        assert report["applied"] and report["n_iter"] == model.n_iter_
        assert report["n_fit_rows"] == fitted.sum()
        assert report["n_validation_rows"] == validation.sum()

    def test_weights_are_split_with_the_rows(self):
        """Sample weights follow their rows: the fitted rows' weights to the
        fit, the block's to sample_weight_val."""
        X, y, index = _rows(60, 200)
        weights = np.linspace(0.5, 1.5, len(y))
        model = HistGradientBoostingRegressor(random_state=0)
        _fit(model, X, y, weights, index=index, horizon=5)
        fitted, validation = _masks_by_hand(index, 0.1, 5)
        by_hand = HistGradientBoostingRegressor(
            random_state=0, early_stopping=True
        ).fit(
            X[fitted],
            y[fitted],
            sample_weight=weights[fitted],
            X_val=X[validation],
            y_val=y[validation],
            sample_weight_val=weights[validation],
        )
        assert np.array_equal(model.predict(X), by_hand.predict(X))

    def test_early_stopping_false_fits_the_whole_window(self):
        """early_stopping=False is left to scikit-learn: every row, max_iter
        iterations, the same model as a plain fit."""
        X, y, index = _rows(60, 200)
        model = HistGradientBoostingRegressor(early_stopping=False, max_iter=30)
        assert _fit(model, X, y, None, index=index, horizon=5) is None
        plain = HistGradientBoostingRegressor(early_stopping=False, max_iter=30).fit(
            X, y
        )
        assert not model.do_early_stopping_ and model.n_iter_ == 30
        assert np.array_equal(model.predict(X), plain.predict(X))

    @pytest.mark.parametrize("classification", [False, True])
    def test_auto_at_or_below_the_threshold_is_unchanged(self, classification):
        """At 10,000 rows scikit-learn's 'auto' does not stop early, and the
        fit is the plain fit, bit for bit, for both tasks."""
        X, y, index = _rows(50, 200, classification=classification)
        assert len(y) == AUTO_EARLY_STOPPING_ROWS
        cls = (
            HistGradientBoostingClassifier
            if classification
            else HistGradientBoostingRegressor
        )
        model = cls(random_state=1)
        assert _fit(model, X, y, None, index=index, horizon=5) is None
        plain = cls(random_state=1).fit(X, y)
        assert not model.do_early_stopping_ and model.early_stopping == "auto"
        assert np.array_equal(model.predict(X), plain.predict(X))

    def test_true_below_the_threshold_stops_on_the_block_too(self):
        """early_stopping=True applies whatever the row count."""
        X, y, index = _rows(10, 200)
        model = HistGradientBoostingRegressor(early_stopping=True, random_state=0)
        stopping = _fit(model, X, y, None, index=index, horizon=5)
        assert stopping.block is not None and model.do_early_stopping_

    def test_the_settings_reach_the_rule(self):
        """validation_fraction sizes the block in dates."""
        X, y, index = _rows(60, 200)
        model = HistGradientBoostingRegressor(
            validation_fraction=0.25, n_iter_no_change=3, random_state=0
        )
        stopping = _fit(model, X, y, None, index=index, horizon=5)
        fitted, validation = _masks_by_hand(index, 0.25, 5)
        assert np.array_equal(stopping.block["validation_rows"], validation)
        assert stopping.block["n_validation_dates"] == 50

    def test_true_on_a_window_that_cannot_hold_the_block_is_refused(self):
        """An explicit request that cannot run on a time-ordered block is
        refused by name rather than fitted another way."""
        X, y, index = _rows(10, 6)
        model = HistGradientBoostingRegressor(early_stopping=True)
        with pytest.raises(ValidationError, match="early_stopping=True needs a time"):
            _fit(model, X, y, None, index=index, horizon=5)

    def test_auto_on_such_a_window_fits_every_row_and_notes_it(self):
        """Above 10,000 rows on too few dates, 'auto' fits every row without
        early stopping -- never the shuffled split -- and the run's warning
        counts the fit."""
        # 10,200 rows on six dates: above the threshold, and a 5-day embargo
        # leaves no date to fit on.
        X, y, index = _rows(1_700, 6)
        notes = []
        model = HistGradientBoostingRegressor(max_iter=20)
        stopping = _fit(model, X, y, None, index=index, horizon=5, notes=notes)
        assert stopping.block is None and stopping.off_kind == "window"
        assert notes == [stopping]
        assert not model.do_early_stopping_ and model.n_iter_ == 20
        plain = HistGradientBoostingRegressor(max_iter=20, early_stopping=False).fit(
            X, y
        )
        assert np.array_equal(model.predict(X), plain.predict(X))
        (warning,) = early_stopping_warnings(notes)
        assert "1 fit(s)" in warning and "fewest dates in one: 6" in warning

    def test_a_classifier_whose_fitted_rows_hold_one_class_is_not_split(self):
        """A block that would leave the classifier one class to fit on is
        refused under True rather than raised from inside scikit-learn."""
        X, y, index = _rows(60, 200, classification=True)
        # One class before the validation block, the other inside it.
        window = np.unique(index.dates)
        y = (index.dates >= window[180]).astype(float)
        model = HistGradientBoostingClassifier(early_stopping=True)
        with pytest.raises(ValidationError, match="hold one class"):
            _fit(model, X, y, None, index=index, horizon=5)


@needs_validation_set
def test_the_shuffled_split_stops_later_on_overlapping_labels():
    """
    The leak, on a panel built to show it: 20-day labels on daily rows,
    each sharing 19 days of outcome with the next, and the date itself as a
    feature. Rows held out at random sit beside fitted rows with nearly the
    same label, so the held-out loss keeps falling while boosting memorizes
    dates; on the window's last dates nothing memorized carries over. With
    one seed and one panel: 155 iterations shuffled, 12 time-ordered.
    """
    rng = np.random.default_rng(0)
    n_entities, n_dates, h = 20, 300, 20
    days = pd.bdate_range("2020-01-01", periods=n_dates + h)
    shocks = rng.normal(size=(n_dates + h, n_entities))
    level = np.vstack([np.zeros((1, n_entities)), np.cumsum(shocks, axis=0)])
    y = (level[h + 1 : n_dates + h + 1] - level[1 : n_dates + 1]).reshape(-1)
    t = np.repeat(np.arange(n_dates), n_entities).astype(float)
    entity = np.tile(np.arange(n_entities), n_dates)
    X = np.column_stack([t, entity, rng.normal(size=(len(t), 2))])
    index = SampleIndex(
        dates=np.repeat(days[:n_dates].to_numpy(), n_entities),
        entities=entity,
        label_end=np.repeat(days[h : n_dates + h].to_numpy(), n_entities),
    )
    shuffled = HistGradientBoostingRegressor(
        early_stopping=True, max_iter=300, random_state=0
    ).fit(X, y)
    ordered = HistGradientBoostingRegressor(
        early_stopping=True, max_iter=300, random_state=0
    )
    _fit(ordered, X, y, None, index=index, horizon=h)
    assert shuffled.n_iter_ >= 5 * ordered.n_iter_
    assert ordered.n_iter_ < 30


class TestAnOlderScikitLearn:
    """Simulated: the check for `X_val` answers no, as before 1.7."""

    @pytest.fixture(autouse=True)
    def _no_validation_set(self, monkeypatch):
        monkeypatch.setattr(trees, "fit_takes_validation_set", lambda: False)

    def test_true_is_refused_naming_the_release(self):
        """Under a scikit-learn without X_val an explicit request is refused,
        naming the release that has it."""
        X, y, index = _rows(10, 200)
        with pytest.raises(ValidationError, match="scikit-learn 1.7 or later"):
            _fit(
                HistGradientBoostingRegressor(early_stopping=True),
                X,
                y,
                None,
                index=index,
                horizon=5,
            )

    def test_true_is_refused_before_any_data_is_read(self):
        """The refusal comes before the panel is read -- for the spec's own
        params and for a search axis that lists True."""
        spec = _spec({"early_stopping": True})
        with pytest.raises(ValidationError, match="scikit-learn 1.7 or later"):
            run_experiment({"panel": None}, spec, "ds", register=False)
        searched = _spec(
            {},
            search=SearchSpec(
                param_grid={"early_stopping": [False, True]}, inner_splits=2
            ),
        )
        with pytest.raises(ValidationError, match="scikit-learn 1.7 or later"):
            run_experiment({"panel": None}, searched, "ds", register=False)

    def test_auto_fits_every_row_without_early_stopping_and_says_so(self):
        """'auto' under such a scikit-learn fits every row without early
        stopping rather than stop on a shuffled share of them."""
        X, y, index = _rows(60, 200)
        notes = []
        model = HistGradientBoostingRegressor(max_iter=25)
        stopping = _fit(model, X, y, None, index=index, horizon=5, notes=notes)
        assert stopping.off_kind == "scikit-learn" and notes == [stopping]
        plain = HistGradientBoostingRegressor(max_iter=25, early_stopping=False).fit(
            X, y
        )
        assert not model.do_early_stopping_ and model.n_iter_ == 25
        assert np.array_equal(model.predict(X), plain.predict(X))

    def test_a_run_warns_once_with_the_count(self):
        """One warning for the run, counting every fit above the threshold, and
        each fold record says early stopping was off."""
        result = run_experiment(
            _dataset(), _spec({"max_iter": 15}), "ds", register=False
        )
        (warning,) = [w for w in result["warnings"] if "cannot take a validation" in w]
        # Two folds and the refit are above 10,000 rows.
        assert "so 3 fit(s)" in warning
        for fold in result["validation_report"]["folds"]:
            assert fold["early_stopping"]["applied"] is False
            assert fold["early_stopping"]["n_iter"] == 15


def _dataset(n_entities=40, n_dates=400, h=5, classification=False):
    X, y, index = _rows(n_entities, n_dates, h=h, classification=classification)
    panel = pd.DataFrame(
        {
            "date": index.dates,
            "entity": [f"E{e}" for e in index.entities],
            "a": X[:, 0],
            "b": X[:, 1],
            "c": X[:, 2],
            "target": y,
            "label_end_date": index.label_end,
        }
    )
    kind = "forward_direction" if classification else "forward_return"
    return {
        "panel": panel,
        "feature_ids": ["a", "b", "c"],
        "target_id": f"{kind}:{h}",
        "data_hash": f"early-stopping-{classification}-{n_entities}",
    }


def _spec(params, task="regression", estimator="hist_gradient_boosting", **extra):
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(
            type=estimator,
            params=params,
            calibration=extra.pop("calibration", "none"),
        ),
        validation=ValidationSpec(
            train_window=300, test_window=50, embargo=0, min_folds=1
        ),
        random_seed=1,
        **extra,
    )


@needs_validation_set
class TestARun:
    def test_each_fold_and_the_refit_record_their_block(self):
        """Each fold's block is the end of the window it fitted, its three
        parts add up to the fold's training rows, and the refit's block ends on
        the panel's last date."""
        dataset = _dataset()
        result = run_experiment(dataset, _spec({"max_iter": 40}), "ds", register=False)
        report = result["validation_report"]
        assert report["folds"]
        for fold in report["folds"]:
            stopping = fold["early_stopping"]
            assert stopping["applied"] is True
            # The block is the end of the window the fold fitted.
            assert stopping["validation_end"] == fold["train_end"]
            assert stopping["validation_start"] > fold["train_start"]
            assert (
                stopping["n_fit_rows"]
                + stopping["n_validation_rows"]
                + stopping["n_embargoed_rows"]
                == fold["n_train_rows"]
            )
            assert stopping["embargo_dates"] == 5
            assert 1 <= stopping["n_iter"] <= 40
        refit = report["refit_early_stopping"]
        last = str(pd.Timestamp(dataset["panel"]["date"].max()).date())
        assert refit["applied"] is True and refit["validation_end"] == last
        assert refit["n_validation_dates"] == 40
        assert not any("early_stopping" in w for w in result["warnings"])

    def test_classification_and_calibration(self):
        """Under calibration every calibration fit stops on the window's block,
        and the fold reports each one's iterations."""
        result = run_experiment(
            _dataset(classification=True),
            _spec({"max_iter": 30}, task="classification", calibration="sigmoid"),
            "ds",
            register=False,
        )
        for fold in result["validation_report"]["folds"]:
            stopping = fold["early_stopping"]
            assert stopping["applied"] is True
            # One booster per calibration fold, each stopped on the block.
            assert len(stopping["n_iter"]) == 3

    def test_below_the_threshold_and_for_other_estimators_nothing_is_recorded(self):
        """A fit the library leaves as scikit-learn runs it adds nothing to the
        report, so those manifests keep their shape."""
        small = run_experiment(
            _dataset(n_entities=20), _spec({"max_iter": 20}), "ds", register=False
        )
        ridge = run_experiment(
            _dataset(), _spec({}, estimator="ridge"), "ds", register=False
        )
        for result in (small, ridge):
            report = result["validation_report"]
            assert "refit_early_stopping" not in report
            assert all("early_stopping" not in fold for fold in report["folds"])

    def test_the_search_s_fits_stop_on_their_own_windows(self, monkeypatch):
        """Search candidates, folds, the final search and the refit each cut
        the block from their own rows, at the end of them."""
        seen = []
        real = engine.prepare_early_stopping

        def spy(estimator, y, index, horizon):
            stopping = real(estimator, y, index, horizon)
            if stopping is not None:
                seen.append((len(y), stopping.block, index.dates))
            return stopping

        monkeypatch.setattr(engine, "prepare_early_stopping", spy)
        spec = _spec(
            {"early_stopping": True},
            search=SearchSpec(param_grid={"max_iter": [10, 20]}, inner_splits=2),
        )
        run_experiment(_dataset(n_entities=20), spec, "ds", register=False)
        # Inner candidates, outer folds, the final search and the refit:
        # every fit split its own rows, whatever their count, with its
        # validation block at the end of them.
        assert len({rows for rows, _block, _dates in seen}) > 3
        for _rows, block, dates in seen:
            assert block is not None
            validated = dates[block["validation_rows"]]
            assert validated.max() == dates.max()
            assert dates[block["fit_rows"]].max() < validated.min()

    def test_conformal_interval_fits_stop_on_their_own_windows(self, monkeypatch):
        """The conformal-interval fits are histogram boosting fits too, and
        each cuts the block from its own rows."""
        calls = []
        real = engine.prepare_early_stopping

        def spy(estimator, y, index, horizon):
            stopping = real(estimator, y, index, horizon)
            calls.append(stopping is not None and stopping.block is not None)
            return stopping

        monkeypatch.setattr(engine, "prepare_early_stopping", spy)
        result = run_experiment(
            _dataset(n_entities=60),
            _spec({"max_iter": 20}, intervals=ConformalSpec(calibration_folds=3)),
            "ds",
            register=False,
        )
        # Per fold and for the refit: the point fit and three calibration
        # fits, each above 10,000 rows and each split.
        n_folds = len(result["validation_report"]["folds"])
        assert len(calls) == 4 * (n_folds + 1) and all(calls)
