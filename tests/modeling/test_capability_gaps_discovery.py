"""
The three discovery tools: the venue catalog, the estimator allowlist's
bounds, and a feature spec's warm-up.

Every answer here is planted rather than recomputed by the test: the
venue codes, the five logistic solvers, the 2000-tree and 4096-leaf
ceilings, momentum's 900 bars at `lookback=900` and hurst's 500 at
`window=500` are all facts about the library that the tools must report,
not numbers derived by calling the same helper twice. Each detector also
gets a null case -- the calendar library absent, an interval with no
venue, an estimator that is not registered, a feature that consumes no
bars -- because a tool that warns about everything and a tool that warns
about nothing are equally useless.

One thing is deliberately NOT pinned: how many venue codes exist. That
count moves with the installed `exchange_calendars` release, so a test
asserting it would fail on an upgrade that broke nothing.
"""

import math
from importlib.util import find_spec
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import calendar as calendar_module
from standard_quant_tools.modeling.agent.discovery_models import (
    DescribeEstimatorInput,
    DescribeExchangeCalendarInput,
    EstimateFeatureWarmupInput,
)
from standard_quant_tools.modeling.agent.discovery_tools import (
    describe_estimator,
    describe_exchange_calendar,
    estimate_feature_warmup,
)
from standard_quant_tools.modeling.estimators.boosting import OPTIONAL_ESTIMATORS
from standard_quant_tools.modeling.estimators.registry import (
    ESTIMATOR_REGISTRY,
    allowed_params,
)
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

from .conftest import make_ohlcv, mock_metadata
from .test_scoring import _train_a_model_with_spec

_HAS_CALENDARS = calendar_module.calendar_available()

_needs_calendars = pytest.mark.skipif(
    not _HAS_CALENDARS,
    reason="the optional exchange_calendars package is not installed here",
)


# ── describe_exchange_calendar ─────────────────────────────────────────


@_needs_calendars
class TestExchangeCalendarCatalog:
    """The venue codes DatasetSpec.calendar accepts are listable, which
    before this tool meant reading the eight names that fit inside a
    refusal message."""

    def test_listing_returns_sorted_codes_including_nyse_london_and_crypto(self):
        result = describe_exchange_calendar(DescribeExchangeCalendarInput())

        assert result.available is True
        # The codes themselves are planted; the COUNT is not, because it
        # moves with the installed exchange_calendars release.
        assert {"XNYS", "XLON", "24/7"} <= set(result.calendar_names)
        assert result.calendar_names == sorted(result.calendar_names)
        assert result.n_calendars == len(result.calendar_names)
        assert result.calendar is None
        assert result.warnings == []

    def test_name_contains_filters_the_codes_and_keeps_the_total(self):
        result = describe_exchange_calendar(
            DescribeExchangeCalendarInput(name_contains="XL")
        )

        assert "XLON" in result.calendar_names
        assert all("xl" in name.lower() for name in result.calendar_names)
        # The unfiltered total still reported, so a filter cannot make the
        # catalog look smaller than it is.
        assert result.n_calendars > len(result.calendar_names)


@_needs_calendars
class TestExchangeCalendarResolution:
    """One venue's session arithmetic -- the numbers that annualize an
    intraday statistic, and which interval needs them at all."""

    def test_nyse_session_counts_are_read_off_the_calendar_not_assumed(self):
        result = describe_exchange_calendar(
            DescribeExchangeCalendarInput(calendar="XNYS")
        )

        assert result.calendar == "XNYS"
        # Counted over complete years with holidays in it, which is why it
        # is not the 252 convention.
        assert result.sessions_per_year is not None
        assert 249.0 < result.sessions_per_year < 253.0
        assert result.sessions_per_year != 252.0
        assert result.session_minutes == 390.0

    def test_hourly_bars_on_nyse_count_the_partial_last_bar(self):
        result = describe_exchange_calendar(
            DescribeExchangeCalendarInput(calendar="XNYS", interval="1h")
        )

        assert result.interval_minutes == 60
        # Seven, not six: a 6.5-hour session emits a stub bar and the
        # provider delivers it.
        assert result.bars_per_session == 7
        assert result.periods_per_year == 1761
        assert result.warnings == []

    def test_daily_interval_reports_no_bars_per_session_and_says_why(self):
        """A daily-or-coarser interval is not an error here. The bars-per-
        session computation refuses it; this tool quotes that refusal as a
        warning and answers the rest of the question."""
        result = describe_exchange_calendar(
            DescribeExchangeCalendarInput(calendar="XNYS", interval="1d")
        )

        assert result.bars_per_session is None
        assert result.periods_per_year is None
        assert result.interval_minutes is None
        # Still resolved, because the venue question was answered.
        assert result.sessions_per_year is not None
        assert any(
            "not an intraday interval" in w and "calendar arithmetic" in w
            for w in result.warnings
        )

    def test_intraday_interval_without_a_calendar_warns_rather_than_guessing(self):
        """The null case for the session arithmetic: bars per session is a
        property of the VENUE, so an interval alone gets minutes per bar
        and an explanation, never an invented session length."""
        result = describe_exchange_calendar(
            DescribeExchangeCalendarInput(interval="1h")
        )

        assert result.interval_minutes == 60
        assert result.bars_per_session is None
        assert result.periods_per_year is None
        assert any("properties of a VENUE" in w for w in result.warnings)

    def test_unknown_code_is_refused_with_the_message_a_dataset_spec_gives(self):
        """The point of the tool: a code that passes here passes
        DatasetSpec, because the refusal is literally the same one."""
        with pytest.raises(ValidationError) as from_tool:
            describe_exchange_calendar(DescribeExchangeCalendarInput(calendar="NOPE"))

        with pytest.raises(PydanticValidationError) as from_spec:
            DatasetSpec(
                universe=["AAA"],
                start="2022-01-01",
                end="2023-01-01",
                features=[FeatureSpec(id="technical.rsi")],
                target=TargetSpec(horizon=5),
                calendar="NOPE",
            )

        assert str(from_tool.value) in str(from_spec.value)
        assert "is not an exchange_calendars name" in str(from_tool.value)


class TestExchangeCalendarWithoutTheLibrary:
    """The optional package absent: listing degrades to an empty catalog
    with a warning, and only RESOLVING a venue is refused."""

    def test_listing_without_the_library_reports_unavailable_rather_than_raising(
        self, monkeypatch
    ):
        monkeypatch.setattr(calendar_module, "calendar_available", lambda: False)

        result = describe_exchange_calendar(
            DescribeExchangeCalendarInput(interval="1h")
        )

        assert result.available is False
        assert result.calendar_names == []
        assert result.n_calendars == 0
        # Minutes per bar is a regex over the interval and needs nothing
        # installed, so it is still answered.
        assert result.interval_minutes == 60
        assert any("exchange_calendars" in w for w in result.warnings)

    def test_resolving_a_calendar_without_the_library_is_refused_by_name(
        self, monkeypatch
    ):
        monkeypatch.setattr(calendar_module, "calendar_available", lambda: False)

        with pytest.raises(ValidationError, match="exchange_calendars"):
            describe_exchange_calendar(DescribeExchangeCalendarInput(calendar="XNYS"))


# ── describe_estimator ─────────────────────────────────────────────────


class TestEstimatorParameterBounds:
    """The bounds the allowlist enforces on every fit, which used to be
    discoverable only by tripping them."""

    def test_logistic_reports_five_solvers_and_the_penalty_matrix(self):
        result = describe_estimator(
            DescribeEstimatorInput(task="classification", name="logistic")
        )

        assert result.n_estimators_described == 1
        entry = result.estimators[0]
        assert entry.params["solver"].choices == [
            "lbfgs",
            "newton-cg",
            "sag",
            "liblinear",
            "saga",
        ]
        # The rule no per-parameter bound can express: l1 is implemented
        # by two of those five.
        notes = " ".join(entry.compatibility_notes)
        assert "liblinear" in notes and "saga" in notes
        assert "elasticnet" in notes and "l1_ratio" in notes

    def test_sgd_reports_a_different_loss_set_per_task(self):
        """The most decision-changing note in the online module: a
        classifier here is asked for probabilities unconditionally, so the
        hinge losses are not offered at all."""
        both = describe_estimator(DescribeEstimatorInput(name="sgd"))
        by_task = {entry.task: entry for entry in both.estimators}

        classification = by_task["classification"].params["loss"]
        regression = by_task["regression"].params["loss"]

        assert classification.choices == ["log_loss", "modified_huber"]
        assert "huber" in (regression.choices or [])
        assert "squared_error" in (regression.choices or [])
        assert classification.choices != regression.choices
        assert "predict_proba" in classification.note

    def test_mlp_reports_two_bounded_integers_not_the_sklearn_tuple(self):
        """An agent that knows scikit-learn guesses `hidden_layer_sizes`
        and is refused; the architecture here is two bounded integers."""
        result = describe_estimator(
            DescribeEstimatorInput(task="regression", name="mlp")
        )
        entry = result.estimators[0]

        assert "n_hidden_units" in entry.params
        assert "n_hidden_layers" in entry.params
        assert "hidden_layer_sizes" not in entry.params
        assert entry.params["n_hidden_units"].maximum == 512
        assert entry.params["n_hidden_layers"].maximum == 3

    def test_tree_and_leaf_ceilings_are_reported_with_their_reason(self):
        """The two resource budgets an unbounded request would blow
        through. lightgbm is installed in this environment, which is what
        makes num_leaves reachable as a registered entry."""
        result = describe_estimator(
            DescribeEstimatorInput(task="regression", name="lightgbm")
        )
        entry = result.estimators[0]

        assert entry.available is True
        assert entry.requires_library == "lightgbm"
        assert entry.params["n_estimators"].maximum == 2000
        assert entry.params["n_estimators"].note != ""
        assert entry.params["num_leaves"].maximum == 4096
        # The one estimator argument the engine sets itself, one fit per
        # requested quantile, so it is not a params key.
        assert entry.quantile_param == "alpha"
        assert "num_leaves" not in (entry.class_path or "")


class TestEstimatorAvailability:
    """What is installed here, what is merely declared, and how an
    unknown name is refused."""

    def test_include_unavailable_names_every_optional_pair_with_its_library(self):
        """The optional estimators are a STATIC declaration, so their
        names and bounds exist on every machine. That is what makes a
        ranking model nameable where its library is not installed."""
        result = describe_estimator(DescribeEstimatorInput(include_unavailable=True))
        described = {(entry.task, entry.name): entry for entry in result.estimators}

        assert len(OPTIONAL_ESTIMATORS) == 8
        for key, (library, _schema) in OPTIONAL_ESTIMATORS.items():
            assert key in described, key
            entry = described[key]
            assert entry.requires_library == library
            assert entry.available is (find_spec(library) is not None)
            # Bounds are described either way; only the class path needs
            # something imported.
            assert entry.params
            if not entry.available:
                assert entry.class_path is None

        assert ("ranking", "lightgbm_ranker") in described
        assert ("ranking", "xgboost_ranker") in described

    def test_unknown_pair_is_refused_with_the_allowlist_message(self):
        """The null case: an estimator that is neither registered nor
        declared optional is refused exactly as a spec naming it would
        be."""
        with pytest.raises(ValidationError) as excinfo:
            describe_estimator(
                DescribeEstimatorInput(task="regression", name="transformer")
            )

        message = str(excinfo.value)
        assert "unknown estimator name='transformer'" in message
        assert "task='regression'" in message
        assert "ridge" in message

    def test_unknown_task_is_refused_naming_the_tasks(self):
        with pytest.raises(ValidationError, match="not a supervised task"):
            describe_estimator(DescribeEstimatorInput(task="clustering"))


class TestEstimatorDescriptionMatchesTheAllowlist:
    """The anti-drift invariant: what this tool describes is what the
    allowlist accepts, entry for entry."""

    @pytest.mark.parametrize("task,name", sorted(ESTIMATOR_REGISTRY))
    def test_every_registered_entry_describes_exactly_its_allowed_params(
        self, task, name
    ):
        result = describe_estimator(DescribeEstimatorInput(task=task, name=name))
        entry = result.estimators[0]

        assert entry.available is True
        assert set(allowed_params(task, name)) == set(entry.params)
        assert entry.class_path

    def test_calibration_is_reported_for_classification_and_absent_elsewhere(self):
        """`calibration` is not in the capability report at all, and it
        decides outcomes rather than polishing them."""
        described = describe_estimator(
            DescribeEstimatorInput(include_unavailable=True)
        ).estimators

        for entry in described:
            if entry.task == "classification":
                assert entry.calibration is not None, entry.name
                assert entry.calibration.choices == ["none", "isotonic", "sigmoid"]
                assert entry.calibration.default == "none"
                assert entry.calibration.folds_default == 3
                assert entry.calibration.folds_minimum == 2
                assert entry.calibration.folds_maximum == 10
                # The sentence that changes a decision: a raw forest
                # selected zero rows where the calibrated one selected 194.
                assert "194" in entry.calibration.note
                assert "proba_threshold=0.9" in entry.calibration.note
            else:
                assert entry.calibration is None, entry.name

    def test_unfiltered_description_warns_about_its_own_size(self):
        result = describe_estimator(DescribeEstimatorInput())

        assert result.n_estimators_described == len(ESTIMATOR_REGISTRY)
        assert any("KB of JSON" in w for w in result.warnings)

    def test_a_filtered_description_carries_no_size_warning(self):
        """The null case for the size warning: asking for one estimator is
        the behaviour the warning is trying to produce."""
        result = describe_estimator(
            DescribeEstimatorInput(task="regression", name="ridge")
        )

        assert result.warnings == []


# ── estimate_feature_warmup ────────────────────────────────────────────


class TestFeatureWarmup:
    """Bars of history a feature spec burns at its REQUESTED parameters,
    which is a different number from the catalog's the moment a window is
    overridden."""

    def test_overridden_lookback_binds_and_the_catalog_number_stays_behind(self):
        """market.momentum is catalogued at 20 bars and consumes 900 when
        asked for 900 -- the catalog understates it by 45x, and nothing
        before this reported the difference."""
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[
                    FeatureSpec(id="market.momentum", params={"lookback": 900}),
                    FeatureSpec(id="technical.rsi"),
                ]
            )
        )

        assert result.bars_required == 900 + result.deepest_lag
        assert result.deepest_lag == 0
        assert result.binding_feature == "market.momentum"

        momentum = result.per_feature["market.momentum"]
        assert momentum.declared == 20
        assert momentum.resolved == 900

        # The feature that does not bind is reported and is not the one to
        # shorten.
        rsi = result.per_feature["technical.rsi"]
        assert rsi.declared == rsi.resolved == 14

    def test_a_window_parameter_diverges_from_the_declared_lookback(self):
        """statistical.hurst is catalogued at 200 and consumes 500 at
        window=500: the window suffix rule, not the parameter's name."""
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[FeatureSpec(id="statistical.hurst", params={"window": 500})]
            )
        )

        entry = result.per_feature["statistical.hurst"]
        assert entry.declared == 200
        assert entry.resolved == 500
        assert entry.declared != entry.resolved
        assert result.bars_required == 500

    def test_an_aliased_spec_is_keyed_by_its_alias(self):
        """The alias is the panel's column name, and the only key that
        distinguishes the same feature requested twice at two windows."""
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[
                    FeatureSpec(
                        id="market.momentum",
                        params={"lookback": 20},
                        alias="mom_20",
                    ),
                    FeatureSpec(
                        id="market.momentum",
                        params={"lookback": 252},
                        alias="mom_252",
                    ),
                ]
            )
        )

        assert set(result.per_feature) == {"mom_20", "mom_252"}
        assert "market.momentum" not in result.per_feature
        assert result.per_feature["mom_252"].resolved == 252
        assert result.binding_feature == "mom_252"
        assert result.bars_required == 252

    def test_lags_are_warm_up_too_and_are_charged_once(self):
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[FeatureSpec(id="technical.rsi", lags=[1, 5])]
            )
        )

        assert result.deepest_lag == 5
        assert result.bars_required == 14 + 5
        assert result.per_feature["technical.rsi"].lags == [1, 5]
        assert result.per_feature["technical.rsi"].deepest_lag == 5

    def test_a_point_in_time_feature_contributes_nothing_and_never_binds(self):
        """The null case for the binding detector: a feature that reads
        filings rather than bars costs no bar warm-up, and its freshness
        is a staleness bound in a different unit."""
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[
                    FeatureSpec(id="fundamental.diluted_eps"),
                    FeatureSpec(id="technical.rsi"),
                ]
            )
        )

        entry = result.per_feature["fundamental.diluted_eps"]
        assert entry.point_in_time is True
        assert entry.declared == 0
        assert entry.resolved == 0
        assert result.binding_feature == "technical.rsi"
        assert result.bars_required == 14
        assert any("fundamental.diluted_eps" in w for w in result.warnings)
        assert any("max_staleness_days" in w for w in result.warnings)

    def test_a_duplicate_output_name_is_named_in_warnings(self):
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[
                    FeatureSpec(id="market.momentum", params={"lookback": 20}),
                    FeatureSpec(id="market.momentum", params={"lookback": 300}),
                ]
            )
        )

        assert len(result.per_feature) == 1
        assert result.per_feature["market.momentum"].resolved == 300
        assert any("more than once" in w for w in result.warnings)

    def test_an_unknown_feature_id_is_refused_with_the_registry_remedy(self):
        """The null case for the resolution path: the registry's refusal,
        unchanged, so an id this tool accepts is one the builder accepts."""
        with pytest.raises(ValidationError) as excinfo:
            estimate_feature_warmup(
                EstimateFeatureWarmupInput(
                    features=[FeatureSpec(id="technical.telepathy")]
                )
            )

        message = str(excinfo.value)
        assert "unknown feature id 'technical.telepathy'" in message
        assert "list_features()" in message


class TestFeatureWarmupInCalendarDays:
    """Bars are not the unit a scoring history window is given in, and the
    conversion is a property of the interval and the venue."""

    def test_daily_bars_convert_with_the_conventional_session_count(self):
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[FeatureSpec(id="market.momentum", params={"lookback": 252})]
            )
        )

        assert result.bars_required == 252
        # One trading year of bars is one calendar year of days.
        assert result.calendar_days_estimate == pytest.approx(365.25, abs=0.5)

    @_needs_calendars
    def test_a_named_venue_replaces_the_252_convention(self):
        without = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[FeatureSpec(id="market.momentum", params={"lookback": 300})]
            )
        )
        with_venue = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[FeatureSpec(id="market.momentum", params={"lookback": 300})],
                calendar="XNYS",
            )
        )

        assert without.calendar_days_estimate == pytest.approx(434.8, abs=1.0)
        # NYSE has fewer than 252 sessions a year, so the same bar count
        # spans slightly MORE calendar days.
        assert with_venue.calendar_days_estimate > without.calendar_days_estimate

    def test_intraday_without_a_calendar_leaves_the_day_estimate_empty(self):
        """The null case for the conversion: guessing a session length
        would be wrong by whatever factor the venue differs by, so the
        field is empty and the warning says what to pass."""
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[FeatureSpec(id="technical.rsi", lags=[1, 5])],
                interval="1h",
            )
        )

        assert result.bars_required == 19
        assert result.calendar_days_estimate is None
        assert any("bars per session" in w for w in result.warnings)

    @_needs_calendars
    def test_intraday_with_a_calendar_converts_through_bars_per_session(self):
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[FeatureSpec(id="technical.rsi", lags=[1, 5])],
                interval="1h",
                calendar="XNYS",
            )
        )

        # 19 hourly bars is under three NYSE sessions, which is days, not
        # a month -- the same bar count at '1d' would be nearly a month.
        assert result.calendar_days_estimate is not None
        assert result.calendar_days_estimate < 10.0


class TestFeatureWarmupSizesAScoringWindow:
    """`score_model(lookback_days=)` is a calendar-day window a human
    supplies by hand, against features that may need hundreds of bars.
    Nothing in the library derived it before this."""

    def test_the_estimate_scores_a_model_the_default_window_cannot(self, monkeypatch):
        # The shared per-symbol provider ignores the requested window and
        # always returns the same bars, which would make lookback_days
        # unobservable. This one honours start/end, which is what the
        # argument under test controls.
        bars = pd.date_range("2022-01-01", periods=900, freq="B")
        last_bar = str(bars[-1].date())

        def _fetch(symbol, start, end, interval="1d"):
            return make_ohlcv(symbol, n=900).loc[str(start) : str(end)]

        provider = MagicMock()
        provider.get_ohlcv.side_effect = _fetch
        provider.get_ohlcv_async = AsyncMock(side_effect=_fetch)
        provider.get_metadata.side_effect = lambda symbol, interval="1d": mock_metadata(
            symbol, interval
        )
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        feature = FeatureSpec(id="market.momentum", params={"lookback": 450})
        estimate = estimate_feature_warmup(
            EstimateFeatureWarmupInput(features=[feature])
        )
        assert estimate.bars_required == 450
        assert estimate.calendar_days_estimate is not None
        # The number the default was never going to be right for.
        assert estimate.calendar_days_estimate > 400

        model_id = _train_a_model_with_spec(
            DatasetSpec(
                universe=["AAA", "BBB"],
                start="2022-01-01",
                # Trained well before the scoring date: score_model refuses
                # an as_of at or before the training cutoff for its own
                # reasons, which is a separate check from this one.
                end="2024-06-28",
                features=[feature],
                target=TargetSpec(horizon=5),
            ),
            dataset_id="ds_long_warmup_feature",
            model_spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(train_window=120, test_window=25, embargo=2),
                random_seed=1,
            ),
        )

        # 400 calendar days is about 286 bars, and this feature needs 450:
        # every row is NaN and the refusal points at the lookback windows.
        with pytest.raises(ValidationError, match="lookback"):
            score_model(model_id=model_id, as_of=last_bar, universe=["AAA", "BBB"])

        scored = score_model(
            model_id=model_id,
            as_of=last_bar,
            universe=["AAA", "BBB"],
            lookback_days=int(math.ceil(estimate.calendar_days_estimate)),
        )
        assert scored["n_entities"] == 2
        assert scored["as_of"] == last_bar
