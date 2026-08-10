"""Tests for modeling.features: every built-in feature against the
underlying primitive it wraps, plus FEATURE_REGISTRY behavior."""

import pandas as pd
import pytest

from standard_quant_tools.analysis.hurst import rolling_hurst
from standard_quant_tools.analysis.pca import pca_returns
from standard_quant_tools.analysis.regression import rolling_beta
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.momentum import rsi
from standard_quant_tools.indicators.trend import adx
from standard_quant_tools.metrics.volatility_estimators import yang_zhang_volatility
from standard_quant_tools.modeling.dataset.alignment import build_returns_panel
from standard_quant_tools.modeling.features.base import (
    FeatureContext,
    FeatureDefinition,
    FeatureScope,
    TemporalSupport,
)
from standard_quant_tools.modeling.features.registry import (
    FEATURE_REGISTRY,
    get_feature,
    list_features,
    register_feature,
)

from .conftest import make_ohlcv

_CONTEXT = FeatureContext()


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return make_ohlcv("AAA")


@pytest.fixture(scope="module")
def benchmark_close() -> pd.Series:
    return make_ohlcv("SPY")["Close"]


class TestBuiltInFeaturesMatchUnderlyingPrimitives:
    def test_technical_rsi(self, ohlcv):
        definition = get_feature("technical.rsi")
        result = definition.fn(ohlcv, _CONTEXT, period=14)
        expected = rsi(ohlcv["Close"], period=14)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_technical_adx(self, ohlcv):
        definition = get_feature("technical.adx")
        result = definition.fn(ohlcv, _CONTEXT, period=14)
        expected = adx(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=14)["ADX"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_market_momentum(self, ohlcv):
        definition = get_feature("market.momentum")
        result = definition.fn(ohlcv, _CONTEXT, lookback=20)
        expected = ohlcv["Close"].pct_change(periods=20)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_market_new_high_breakout_excludes_own_bar(self, ohlcv):
        """Same look-ahead-safe convention as analysis/rally.py: today's
        own bar must be excluded from its own breakout comparison."""
        definition = get_feature("market.new_high_breakout")
        result = definition.fn(ohlcv, _CONTEXT, period=20)
        breakout_high = ohlcv["High"].rolling(20).max().shift(1)
        expected = (ohlcv["Close"] > breakout_high).astype(float)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_risk_realized_volatility(self, ohlcv):
        definition = get_feature("risk.realized_volatility")
        result = definition.fn(ohlcv, _CONTEXT, period=20)
        expected = yang_zhang_volatility(
            ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=20
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_risk_rolling_beta(self, ohlcv, benchmark_close):
        definition = get_feature("risk.rolling_beta")
        context = FeatureContext(benchmark_close=benchmark_close)
        result = definition.fn(ohlcv, context, window=60)
        asset_returns = ohlcv["Close"].pct_change().dropna()
        benchmark_returns = benchmark_close.pct_change().dropna()
        expected = rolling_beta(asset_returns, benchmark_returns, window=60)["Rolling_Beta"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_risk_rolling_beta_requires_benchmark_in_context(self, ohlcv):
        definition = get_feature("risk.rolling_beta")
        with pytest.raises(ValidationError, match="benchmark_close"):
            definition.fn(ohlcv, FeatureContext(), window=60)

    def test_statistical_hurst(self, ohlcv):
        definition = get_feature("statistical.hurst")
        result = definition.fn(ohlcv, _CONTEXT, window=200, method="dfa")
        returns = ohlcv["Close"].pct_change().dropna()
        expected = rolling_hurst(returns, window=200, method="dfa")
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestUniverseScopePcaFeatures:
    @pytest.fixture(scope="class")
    def returns_panel(self) -> pd.DataFrame:
        close_by_entity = {s: make_ohlcv(s)["Close"] for s in ("AAA", "BBB", "CCC")}
        return build_returns_panel(close_by_entity)

    def test_pca_loading_shape_and_columns(self, returns_panel):
        definition = get_feature("factors.pca_loading")
        result = definition.fn(returns_panel, _CONTEXT, window=100, refit_every=10)
        assert list(result.columns) == list(returns_panel.columns)
        assert len(result) == len(returns_panel)

    def test_pca_loading_forward_filled_between_refits(self, returns_panel):
        """Loadings only change at refit points (every `refit_every`
        bars) -- between refits, consecutive rows must be identical."""
        definition = get_feature("factors.pca_loading")
        result = definition.fn(returns_panel, _CONTEXT, window=100, refit_every=10)
        valid = result.dropna()
        # Any two consecutive rows strictly inside one refit block are equal.
        mid_block = valid.iloc[102:105]
        pd.testing.assert_series_equal(mid_block.iloc[0], mid_block.iloc[1], check_names=False)

    def test_pca_factor_return_shared_across_entities(self, returns_panel):
        """The same macro factor return applies to every entity on a
        given date -- every column must be identical row-wise."""
        definition = get_feature("factors.pca_factor_return")
        result = definition.fn(returns_panel, _CONTEXT, window=100, refit_every=10)
        for col in result.columns[1:]:
            pd.testing.assert_series_equal(
                result[col], result[returns_panel.columns[0]], check_names=False
            )

    def test_pca_factor_return_matches_hand_computed_projection(self, returns_panel):
        """At the first refit point, the factor return equals that
        date's realized return dotted with the freshly fit PC1 loadings
        -- the exact projection the docstring describes."""
        definition = get_feature("factors.pca_factor_return")
        window, refit_every = 100, 10
        result = definition.fn(returns_panel, _CONTEXT, window=window, refit_every=refit_every)

        refit_pos = window - 1  # first index where i+1 >= window and (i+1-window) % refit_every == 0
        window_slice = returns_panel.iloc[refit_pos + 1 - window : refit_pos + 1]
        pca_result = pca_returns(window_slice, n_components=1)
        loadings = pca_result["loadings"]["PC1"]
        expected = float(returns_panel.iloc[refit_pos].to_numpy() @ loadings.to_numpy())

        assert result.iloc[refit_pos, 0] == pytest.approx(expected)


class TestPcaFeatureParamValidation:
    """window/refit_every feed a range() step -- an unvalidated
    refit_every=0 used to crash with Python's cryptic 'range() arg 3
    must not be zero' instead of a clear, attributable ValidationError."""

    @pytest.fixture(scope="class")
    def returns_panel(self) -> pd.DataFrame:
        close_by_entity = {s: make_ohlcv(s)["Close"] for s in ("AAA", "BBB")}
        return build_returns_panel(close_by_entity)

    @pytest.mark.parametrize("feature_id", ["factors.pca_loading", "factors.pca_factor_return"])
    def test_zero_refit_every_raises_validation_error(self, returns_panel, feature_id):
        definition = get_feature(feature_id)
        with pytest.raises(ValidationError, match="refit_every"):
            definition.fn(returns_panel, _CONTEXT, window=100, refit_every=0)

    @pytest.mark.parametrize("feature_id", ["factors.pca_loading", "factors.pca_factor_return"])
    def test_negative_refit_every_raises_validation_error(self, returns_panel, feature_id):
        definition = get_feature(feature_id)
        with pytest.raises(ValidationError, match="refit_every"):
            definition.fn(returns_panel, _CONTEXT, window=100, refit_every=-5)

    @pytest.mark.parametrize("feature_id", ["factors.pca_loading", "factors.pca_factor_return"])
    def test_window_below_two_raises_validation_error(self, returns_panel, feature_id):
        definition = get_feature(feature_id)
        with pytest.raises(ValidationError, match="window"):
            definition.fn(returns_panel, _CONTEXT, window=1, refit_every=10)


class TestFeatureRegistry:
    def test_all_nine_built_in_features_registered(self):
        expected_ids = {
            "technical.rsi",
            "technical.adx",
            "market.momentum",
            "market.new_high_breakout",
            "risk.realized_volatility",
            "risk.rolling_beta",
            "statistical.hurst",
            "factors.pca_loading",
            "factors.pca_factor_return",
        }
        assert expected_ids <= set(FEATURE_REGISTRY.keys())

    def test_entity_scope_features_are_entity_scope(self):
        for feature_id in (
            "technical.rsi",
            "technical.adx",
            "market.momentum",
            "market.new_high_breakout",
            "risk.realized_volatility",
            "risk.rolling_beta",
            "statistical.hurst",
        ):
            assert get_feature(feature_id).scope == FeatureScope.ENTITY

    def test_pca_features_are_universe_scope(self):
        assert get_feature("factors.pca_loading").scope == FeatureScope.UNIVERSE
        assert get_feature("factors.pca_factor_return").scope == FeatureScope.UNIVERSE

    def test_all_built_in_features_are_pit_safe(self):
        """Phase 1 has no fundamentals feature -- every built-in must be
        PIT_SAFE, or the leakage check would reject build_model_dataset
        for perfectly ordinary technical/market/risk/factor features."""
        for definition in FEATURE_REGISTRY.values():
            assert definition.temporal_support == TemporalSupport.PIT_SAFE

    def test_get_unknown_feature_raises(self):
        with pytest.raises(ValidationError, match="unknown feature id"):
            get_feature("nonexistent.feature")

    def test_register_duplicate_id_raises_without_overwrite(self):
        dummy = FeatureDefinition(
            id="test.dummy_duplicate",
            description="d",
            fn=lambda ohlcv, context, **p: ohlcv["Close"],
            temporal_support=TemporalSupport.PIT_SAFE,
            lookback=0,
        )
        register_feature(dummy)
        try:
            with pytest.raises(ValidationError, match="already registered"):
                register_feature(dummy)
            register_feature(dummy, overwrite=True)  # must not raise
        finally:
            del FEATURE_REGISTRY["test.dummy_duplicate"]

    def test_list_features_filters_by_category(self):
        technical_only = list_features(category="technical")
        assert {d.id for d in technical_only} == {"technical.rsi", "technical.adx"}

    def test_list_features_no_category_returns_full_catalog(self):
        assert len(list_features()) == len(FEATURE_REGISTRY)
