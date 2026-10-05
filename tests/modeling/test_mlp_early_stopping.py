"""
The MLPs stop early on their training window's last dates (see the
CHANGELOG entry of 2026-10-04).

scikit-learn's MLP, under `early_stopping=True`, holds out a SHUFFLED
`validation_fraction` of the training rows (stratified for the classifier)
and stops when the score on them has not improved by `tol` for
`n_iter_no_change` epochs. On a panel those rows sit among the rows fitted,
whose overlapping labels share their outcomes. The engine now hands
`PanelMLPRegressor` and `PanelMLPClassifier` the block histogram boosting
stops on -- the window's last 10% of dates, with the label horizon
embargoed before them -- as `X_val`, and the subclasses run scikit-learn's
own rule on those rows. `early_stopping=False`, the default, fits as before.
"""

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
import sklearn.neural_network._multilayer_perceptron as sklearn_mlp
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.validation import has_fit_parameter

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.engine import _fit, run_experiment
from standard_quant_tools.modeling.estimators import neural
from standard_quant_tools.modeling.estimators.neural import (
    PanelMLPClassifier,
    PanelMLPRegressor,
)
from standard_quant_tools.modeling.samples import SampleIndex
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)

#: scikit-learn's MLP takes sample weights from 1.7.
needs_weights = pytest.mark.skipif(
    "sample_weight" not in inspect.signature(MLPRegressor.fit).parameters,
    reason="this scikit-learn's MLP fit takes no sample_weight (1.7+)",
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


def _mlp(classification, **params):
    cls = PanelMLPClassifier if classification else PanelMLPRegressor
    return cls(**{"n_hidden_units": 8, "max_iter": 80, "random_state": 3, **params})


def _same_fit(a, b):
    """Two fitted MLPs hold the same epochs, scores and weights, bit for bit."""
    assert a.n_iter_ == b.n_iter_
    assert a.loss_curve_ == b.loss_curve_
    assert a.validation_scores_ == b.validation_scores_
    assert a.best_validation_score_ == b.best_validation_score_
    for mine, theirs in zip(a.coefs_ + a.intercepts_, b.coefs_ + b.intercepts_):
        assert np.array_equal(mine, theirs)


class TestScikitLearnsRuleOnTheRowsHandedIn:
    @pytest.mark.parametrize(
        "weighted", [False, pytest.param(True, marks=needs_weights)]
    )
    @pytest.mark.parametrize("classification", [False, True])
    def test_it_is_scikit_learns_early_stopping_given_the_same_split(
        self, monkeypatch, classification, weighted
    ):
        """Handed the split scikit-learn's own early stopping draws, the
        subclass runs the same epochs and keeps the same weights: the same
        validation scores, best score, epoch count, loss curve and
        coefficients, bit for bit, for both tasks, weighted and not. The
        epochs are not shuffled, so the random draws the split itself takes
        are the only ones the two runs do not share."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1_200, 5))
        y = X[:, 0] * 0.5 + np.sin(X[:, 1]) + rng.normal(scale=0.5, size=1_200)
        if classification:
            y = (y > 0).astype(int)
        weights = rng.uniform(0.5, 1.5, size=1_200) if weighted else None
        split = {}
        real_split = sklearn_mlp.train_test_split

        def recording_split(*arrays, **kwargs):
            split["parts"] = real_split(*arrays, **kwargs)
            return split["parts"]

        monkeypatch.setattr(sklearn_mlp, "train_test_split", recording_split)
        sklearn_cls = MLPClassifier if classification else MLPRegressor
        theirs = sklearn_cls(
            hidden_layer_sizes=(16,),
            early_stopping=True,
            max_iter=300,
            random_state=4,
            shuffle=False,
        )
        theirs.fit(X, y, **({} if weights is None else {"sample_weight": weights}))
        monkeypatch.setattr(sklearn_mlp, "train_test_split", real_split)
        if weighted:
            X_fit, X_val, y_fit, y_val, w_fit, w_val = split["parts"]
            extra = {"sample_weight": w_fit, "sample_weight_val": w_val}
        else:
            X_fit, X_val, y_fit, y_val = split["parts"]
            extra = {}
        # scikit-learn splits the targets as it trains on them: a column,
        # binarized for the classifier.
        if classification:
            y_fit = theirs._label_binarizer.inverse_transform(y_fit)
            y_val = theirs._label_binarizer.inverse_transform(y_val)
        else:
            y_fit, y_val = y_fit.ravel(), y_val.ravel()

        cls = PanelMLPClassifier if classification else PanelMLPRegressor
        mine = cls(n_hidden_units=16, early_stopping=True, max_iter=300, random_state=4)
        mine.shuffle = False
        mine.fit(X_fit, y_fit, X_val=X_val, y_val=y_val, **extra)
        _same_fit(mine, theirs)
        assert np.array_equal(mine.predict(X), theirs.predict(X))
        # It stopped on the rule, not on max_iter.
        assert mine.n_iter_ < 300

    @pytest.mark.parametrize("classification", [False, True])
    def test_with_shuffled_epochs_the_rule_holds(self, classification):
        """With scikit-learn's default shuffled epochs: one validation score
        per epoch; training stops at the first epoch where the score has
        failed to beat the best before it by tol for more than
        n_iter_no_change epochs in a row; the weights kept are the first best
        epoch's, which score best_validation_score_ on the rows handed in."""
        X, y, _index = _rows(20, 100, classification=classification)
        X_fit, y_fit, X_val, y_val = X[:1_800], y[:1_800], X[1_800:], y[1_800:]
        model = _mlp(classification, early_stopping=True, max_iter=400)
        model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
        scores = model.validation_scores_
        assert len(scores) == model.n_iter_
        best, stale, stop = -np.inf, 0, None
        for epoch, score in enumerate(scores, start=1):
            stale = stale + 1 if score < best + model.tol else 0
            best = max(best, score)
            if stale > model.n_iter_no_change:
                stop = epoch
                break
        assert stop == model.n_iter_ < 400
        assert model.best_validation_score_ == max(scores)
        assert model.best_iteration() == int(np.argmax(scores)) + 1
        metric = accuracy_score if classification else r2_score
        assert metric(y_val, model.predict(X_val)) == model.best_validation_score_

    def test_the_estimator_keeps_its_settings_and_not_the_rows(self):
        """After a fit on a validation set the estimator's parameters are
        those it was built with -- early_stopping still True -- and the
        validation rows are not kept on it, so they are not pickled."""
        X, y, _index = _rows(10, 60)
        model = _mlp(False, early_stopping=True)
        params = model.get_params()
        model.fit(X[:500], y[:500], X_val=X[500:], y_val=y[500:])
        assert model.get_params() == params and model.early_stopping is True
        assert "_held_out" not in vars(model)

    def test_the_validation_arguments_are_checked(self):
        """A validation set beside early_stopping=False, half of one, one
        row, or the wrong width is refused rather than ignored."""
        X, y, _index = _rows(10, 30)
        with pytest.raises(ValueError, match="early_stopping is False"):
            _mlp(False).fit(X, y, X_val=X[:10], y_val=y[:10])
        with pytest.raises(ValueError, match="given together"):
            _mlp(False, early_stopping=True).fit(X, y, X_val=X[:10])
        with pytest.raises(ValueError, match="1 row"):
            _mlp(False, early_stopping=True).fit(X, y, X_val=X[:1], y_val=y[:1])
        with pytest.raises(ValueError, match="2 features"):
            _mlp(False, early_stopping=True).fit(X, y, X_val=X[:10, :2], y_val=y[:10])


class _MLPRegressorBefore(MLPRegressor):
    """PanelMLPRegressor as it was before the validation set, reproduced so
    the default fit can be held to it."""

    def __init__(
        self,
        n_hidden_units=64,
        n_hidden_layers=1,
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=False,
        random_state=None,
    ):
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        super().__init__(
            hidden_layer_sizes=tuple([int(n_hidden_units)] * int(n_hidden_layers)),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )

    def fit(self, X, y, **kwargs):
        self.hidden_layer_sizes = tuple(
            [int(self.n_hidden_units)] * int(self.n_hidden_layers)
        )
        return super().fit(X, y, **kwargs)


class _MLPClassifierBefore(MLPClassifier):
    """PanelMLPClassifier as it was before the validation set."""

    def __init__(
        self,
        n_hidden_units=64,
        n_hidden_layers=1,
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=False,
        random_state=None,
    ):
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        super().__init__(
            hidden_layer_sizes=tuple([int(n_hidden_units)] * int(n_hidden_layers)),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )

    def fit(self, X, y, **kwargs):
        self.hidden_layer_sizes = tuple(
            [int(self.n_hidden_units)] * int(self.n_hidden_layers)
        )
        return super().fit(X, y, **kwargs)


class TestTheDefaultIsUnchanged:
    @pytest.mark.parametrize(
        "weighted", [False, pytest.param(True, marks=needs_weights)]
    )
    @pytest.mark.parametrize("classification", [False, True])
    def test_early_stopping_false_fits_as_before_bit_for_bit(
        self, classification, weighted
    ):
        """early_stopping=False is left to scikit-learn: through the engine's
        fit, on a deterministic panel, the fitted state pickles to the same
        bytes as the class before this change fitted plainly, and predicts
        the same numbers; the engine records no stopping rule. Whether
        calibration sees a sample_weight parameter is unchanged too."""
        X, y, index = _rows(20, 150, classification=classification)
        weights = np.linspace(0.5, 1.5, len(y)) if weighted else None
        mine = _mlp(classification, max_iter=40)
        assert _fit(mine, X, y, weights, index=index, horizon=5) is None
        before_cls = _MLPClassifierBefore if classification else _MLPRegressorBefore
        before = before_cls(n_hidden_units=8, max_iter=40, random_state=3)
        before.fit(X, y, **({} if weights is None else {"sample_weight": weights}))
        assert pickle.dumps(mine.__getstate__()) == pickle.dumps(before.__getstate__())
        assert np.array_equal(mine.predict(X), before.predict(X))
        assert has_fit_parameter(mine, "sample_weight") == has_fit_parameter(
            before, "sample_weight"
        )

    def test_a_default_run_records_nothing(self):
        """A run at the default adds no early_stopping record to a fold or
        the refit, so its manifest keeps its shape."""
        result = run_experiment(
            _dataset(n_entities=20), _spec({"max_iter": 20}), "ds", register=False
        )
        report = result["validation_report"]
        assert "refit_early_stopping" not in report
        assert all("early_stopping" not in fold for fold in report["folds"])


class TestTheEngineHandsItTheBlock:
    @pytest.mark.parametrize("classification", [False, True])
    def test_the_validation_rows_are_the_last_tenth_of_the_dates(self, classification):
        """Through the engine's fit the MLP validates on the window's last
        20 of 200 dates, the 5 dates before them embargoed, and is the same
        fit as one handed that block by hand; the record counts its rows and
        names the epoch whose weights were kept."""
        X, y, index = _rows(10, 200, classification=classification)
        model = _mlp(classification, early_stopping=True)
        stopping = _fit(model, X, y, None, index=index, horizon=5)
        fitted, validation = _masks_by_hand(index, 0.1, 5)
        assert np.array_equal(stopping.block["fit_rows"], fitted)
        assert np.array_equal(stopping.block["validation_rows"], validation)
        by_hand = _mlp(classification, early_stopping=True).fit(
            X[fitted], y[fitted], X_val=X[validation], y_val=y[validation]
        )
        _same_fit(model, by_hand)
        report = stopping.report(model)
        assert report["applied"] and report["n_validation_dates"] == 20
        assert report["n_fit_rows"] == fitted.sum()
        assert report["n_validation_rows"] == validation.sum()
        assert report["n_iter"] == model.n_iter_
        assert report["best_iter"] == model.best_iteration() <= model.n_iter_

    def test_no_fitted_row_s_label_reaches_the_block(self):
        """An entity that skips dates reaches its 5th bar later than the
        panel's calendar does. The MLP is fitted only on rows whose labels
        end before the block's first date, and dated before it."""
        days = pd.bdate_range("2021-01-01", periods=120)
        frames = []
        for entity in range(4):
            own = days[::2] if entity == 0 else days
            ends = list(own[5:]) + [pd.NaT] * 5
            frames.append(pd.DataFrame({"date": own, "end": ends, "entity": entity}))
        frame = pd.concat(frames, ignore_index=True).dropna()
        index = SampleIndex(
            dates=frame["date"].to_numpy(),
            entities=frame["entity"].to_numpy(),
            label_end=frame["end"].to_numpy(),
        )
        rng = np.random.default_rng(5)
        X = rng.normal(size=(len(frame), 3))
        y = X[:, 0] + rng.normal(size=len(frame))
        seen = {}
        model = _mlp(False, early_stopping=True)
        real_fit = PanelMLPRegressor.fit

        def recording_fit(self, X_fit, y_fit, **kwargs):
            seen["X"], seen["X_val"] = X_fit, kwargs["X_val"]
            return real_fit(self, X_fit, y_fit, **kwargs)

        model.fit = recording_fit.__get__(model)
        stopping = _fit(model, X, y, None, index=index, horizon=5)
        fitted = stopping.block["fit_rows"]
        first = index.dates[stopping.block["validation_rows"]].min()
        assert (index.label_end[fitted] < first).all()
        assert (index.dates[fitted] < first).all()
        # The ragged entity's labels reach the block from before the date
        # embargo; those rows were not fitted.
        window = np.unique(index.dates)
        before_embargo = index.dates <= window[np.searchsorted(window, first) - 6]
        reaching = before_embargo & (index.label_end >= first)
        assert reaching.any() and not (fitted & reaching).any()
        assert np.array_equal(seen["X"], X[fitted])
        assert np.array_equal(seen["X_val"], X[stopping.block["validation_rows"]])

    @needs_weights
    def test_weights_follow_their_rows(self):
        """Sample weights follow their rows: the fitted rows' weights to the
        fit, the block's to the validation score."""
        X, y, index = _rows(10, 200)
        weights = np.linspace(0.5, 1.5, len(y))
        model = _mlp(False, early_stopping=True)
        _fit(model, X, y, weights, index=index, horizon=5)
        fitted, validation = _masks_by_hand(index, 0.1, 5)
        by_hand = _mlp(False, early_stopping=True).fit(
            X[fitted],
            y[fitted],
            sample_weight=weights[fitted],
            X_val=X[validation],
            y_val=y[validation],
            sample_weight_val=weights[validation],
        )
        _same_fit(model, by_hand)


class TestRefusals:
    def test_a_window_too_short_for_the_block_is_refused(self):
        """Six dates cannot hold a validation date and a five-date embargo
        with a date left to fit; the request is refused by name rather than
        fitted another way."""
        X, y, index = _rows(10, 6)
        with pytest.raises(ValidationError, match="estimator 'mlp': early_stopping"):
            _fit(_mlp(False, early_stopping=True), X, y, None, index=index, horizon=5)

    def test_the_advice_names_what_an_mlp_can_set(self):
        """The MLP has no 'auto' and no validation_fraction parameter, so the
        refusal does not offer them."""
        X, y, index = _rows(10, 6)
        with pytest.raises(ValidationError) as caught:
            _fit(_mlp(False, early_stopping=True), X, y, None, index=index, horizon=5)
        message = str(caught.value)
        assert "Set early_stopping=False, or widen" in message
        assert "'auto'" not in message and "validation_fraction" not in message

    def test_a_classifier_whose_fitted_rows_hold_one_class_is_refused(self):
        """A block that would leave the classifier one class to fit on is
        refused by name."""
        X, _y, index = _rows(10, 200, classification=True)
        window = np.unique(index.dates)
        y = (index.dates >= window[180]).astype(float)
        with pytest.raises(ValidationError, match="hold one class"):
            _fit(_mlp(True, early_stopping=True), X, y, None, index=index, horizon=5)

    def test_a_block_of_one_row_is_refused(self):
        """scikit-learn's early stopping scores at least two validation
        rows; a block whose last date holds one row is refused by name."""
        X, y, index = _rows(3, 10, h=1)
        keep = np.ones(len(y), dtype=bool)
        keep[-2:] = False
        index = SampleIndex(
            dates=index.dates[keep],
            entities=index.entities[keep],
            label_end=index.label_end[keep],
        )
        with pytest.raises(ValidationError, match="needs at least 2"):
            _fit(
                _mlp(False, early_stopping=True),
                X[keep],
                y[keep],
                None,
                index=index,
                horizon=1,
            )

    def test_a_fit_without_a_sample_index_is_refused(self):
        """With no dates to order the rows by there is no block to cut."""
        X, y, _index = _rows(10, 50)
        with pytest.raises(ValidationError, match="no sample index"):
            _fit(_mlp(False, early_stopping=True), X, y, None, index=None)

    def test_without_the_methods_it_is_refused_before_any_data_is_read(
        self, monkeypatch
    ):
        """Simulated: a scikit-learn whose MLP lacks the methods the
        validation set is handed through. early_stopping=True is refused
        before the panel is read -- in params or on a search axis -- and the
        estimator refuses a validation set rather than ignore it."""
        monkeypatch.setattr(neural, "_scoring_arguments", lambda: None)
        with pytest.raises(ValidationError, match="cannot be handed one"):
            run_experiment(
                {"panel": None}, _spec({"early_stopping": True}), "ds", register=False
            )
        searched = _spec(
            {},
            search=SearchSpec(
                param_grid={"early_stopping": [False, True]}, inner_splits=2
            ),
        )
        with pytest.raises(ValidationError, match="cannot be handed one"):
            run_experiment({"panel": None}, searched, "ds", register=False)
        X, y, _index = _rows(10, 30)
        with pytest.raises(TypeError, match="lacks the methods"):
            _mlp(False, early_stopping=True).fit(X, y, X_val=X[:10], y_val=y[:10])


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
        "data_hash": f"mlp-early-stopping-{classification}-{n_entities}",
    }


def _spec(params, task="regression", **extra):
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(
            type="mlp",
            params={"n_hidden_units": 8, "random_state": 0, **params},
            calibration=extra.pop("calibration", "none"),
        ),
        validation=ValidationSpec(
            train_window=300, test_window=50, embargo=0, min_folds=1
        ),
        random_seed=1,
        **extra,
    )


class TestARun:
    def test_each_fold_and_the_refit_record_their_block(self):
        """Each fold's block is the end of the window it fitted, its three
        parts add up to the fold's training rows, and the refit's block ends
        on the panel's last date; each record gives the epochs run and the
        epoch whose weights were kept."""
        dataset = _dataset(n_entities=20)
        result = run_experiment(
            dataset,
            _spec({"early_stopping": True, "max_iter": 60}),
            "ds",
            register=False,
        )
        report = result["validation_report"]
        assert report["folds"]
        for fold in report["folds"]:
            stopping = fold["early_stopping"]
            assert stopping["applied"] is True
            assert stopping["validation_end"] == fold["train_end"]
            assert stopping["validation_start"] > fold["train_start"]
            assert (
                stopping["n_fit_rows"]
                + stopping["n_validation_rows"]
                + stopping["n_embargoed_rows"]
                == fold["n_train_rows"]
            )
            assert stopping["embargo_dates"] == 5
            assert 0 <= stopping["best_iter"] <= stopping["n_iter"] <= 60
        refit = report["refit_early_stopping"]
        last = str(pd.Timestamp(dataset["panel"]["date"].max()).date())
        assert refit["applied"] is True and refit["validation_end"] == last
        assert refit["n_validation_dates"] == 40
        assert not any("early_stopping" in w for w in result["warnings"])

    def test_classification_under_calibration(self):
        """Under calibration every calibration fit stops on the window's
        block, and the fold reports each one's epochs."""
        result = run_experiment(
            _dataset(n_entities=20, classification=True),
            _spec(
                {"early_stopping": True, "max_iter": 40},
                task="classification",
                calibration="sigmoid",
            ),
            "ds",
            register=False,
        )
        for fold in result["validation_report"]["folds"]:
            stopping = fold["early_stopping"]
            assert stopping["applied"] is True
            assert len(stopping["n_iter"]) == len(stopping["best_iter"]) == 3


def _leak_panel():
    """20-day labels on daily rows, each sharing 19 days of outcome with the
    next; the features are the date, one column per entity and two noise
    columns, z-scored as the engine scales them. Nothing in them predicts a
    label, but together the date and the entity locate it."""
    rng = np.random.default_rng(0)
    n_entities, n_dates, h = 10, 300, 20
    days = pd.bdate_range("2020-01-01", periods=n_dates + h)
    shocks = rng.normal(size=(n_dates + h, n_entities))
    level = np.vstack([np.zeros((1, n_entities)), np.cumsum(shocks, axis=0)])
    y = (level[h + 1 : n_dates + h + 1] - level[1 : n_dates + 1]).reshape(-1)
    t = np.repeat(np.arange(n_dates), n_entities).astype(float)
    entity = np.tile(np.arange(n_entities), n_dates)
    X = np.column_stack([t, np.eye(n_entities)[entity], rng.normal(size=(len(t), 2))])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    index = SampleIndex(
        dates=np.repeat(days[:n_dates].to_numpy(), n_entities),
        entities=entity,
        label_end=np.repeat(days[h : n_dates + h].to_numpy(), n_entities),
    )
    return X, y, index, h


def test_the_shuffled_split_stops_later_on_overlapping_labels():
    """
    The leak: rows held out at random sit beside fitted rows of the same
    entity with nearly the same label, so their score keeps rising while
    the network memorizes each entity's path; on the window's last dates
    nothing memorized carries over. Over six seeds, scikit-learn's shuffled
    split ran 61 to 303 epochs (1,121 in all) and scored a best R2 of 0.10
    to 0.25; the time-ordered block 12 to 49 epochs (210) and 0.005 to
    0.063. A single seed's epoch count moves with the last bits of the
    matrix products, which differ between platforms, so the test holds the
    six together.
    """
    X, y, index, h = _leak_panel()
    shuffled_epochs, ordered_epochs, shuffled_best, ordered_best = [], [], [], []
    for seed in range(6):
        shuffled = PanelMLPRegressor(
            early_stopping=True, max_iter=500, random_state=seed
        ).fit(X, y)
        ordered = PanelMLPRegressor(
            early_stopping=True, max_iter=500, random_state=seed
        )
        _fit(ordered, X, y, None, index=index, horizon=h)
        shuffled_epochs.append(shuffled.n_iter_)
        ordered_epochs.append(ordered.n_iter_)
        shuffled_best.append(shuffled.best_validation_score_)
        ordered_best.append(ordered.best_validation_score_)
    assert sum(shuffled_epochs) >= 3 * sum(ordered_epochs)
    assert np.median(shuffled_best) >= 2 * max(np.median(ordered_best), 0.0)
