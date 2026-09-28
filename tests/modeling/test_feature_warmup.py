"""
How much history a recursive feature needs before its value stops
depending on where that history starts.

`resolved_lookback` counts the bars to a feature's first output. For an EMA
or a Wilder smoother the first output still carries its start value, so an
RSI computed from 2016 and one computed from 2010 disagree for over a
hundred bars after it -- rows that are not NaN, so no alignment drops them,
and a scoring window started elsewhere than the training build computes
them differently. `resolved_warmup` is the second quantity: the bars until
the start value's weight is below a tolerance. These tests plant the
answer by computing each feature on a full history and on the same history
truncated, and asking where the two agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.agent.discovery_models import (
    EstimateFeatureWarmupInput,
)
from standard_quant_tools.modeling.agent.discovery_tools import (
    estimate_feature_warmup,
)
from standard_quant_tools.modeling.features.base import FeatureContext
from standard_quant_tools.modeling.features.params import (
    WARMUP_TOLERANCE,
    is_recursive,
    resolve_params,
    resolved_lookback,
    resolved_warmup,
    smoother_decay_bars,
)
from standard_quant_tools.modeling.features.registry import get_feature
from standard_quant_tools.modeling.scoring import _unconverged_window_warning
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec


def _ohlcv(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.01, n))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=pd.bdate_range("2014-01-02", periods=n),
    )


def _worst_error_after(feature_id, params, bars, *, n=1600, starts=(400, 800)):
    """The largest |truncated - full| / std(full) at any bar at least
    `bars` after the truncated history's start, over two seeds and two
    start points."""
    definition = get_feature(feature_id)
    resolved = resolve_params(definition, params)
    context = FeatureContext(interval="1d")
    worst = 0.0
    for seed in (21, 22):
        frame = _ohlcv(n, seed)
        full = pd.Series(
            definition.fn(frame, context, **resolved), index=frame.index
        ).astype(float)
        scale = float(full.std()) or 1.0
        for start in starts:
            truncated = pd.Series(
                definition.fn(frame.iloc[start:], context, **resolved),
                index=frame.index[start:],
            ).astype(float)
            error = (
                np.abs(
                    truncated.iloc[bars:].to_numpy()
                    - full.iloc[start + bars :].to_numpy()
                )
                / scale
            )
            finite = error[np.isfinite(error)]
            worst = max(worst, float(finite.max()) if finite.size else np.inf)
    return worst, resolved


RECURSIVE = [
    ("technical.rsi", {}),
    ("risk.atr_pct", {}),
    ("technical.macd_histogram", {}),
    ("technical.adx", {}),
    ("technical.rsi", {"period": 30}),
    ("technical.adx", {"period": 20}),
    ("technical.macd_histogram", {"fast": 8, "slow": 21, "signal": 5}),
]


class TestTheConvergedWarmUpIsEnough:
    @pytest.mark.parametrize("feature_id,params", RECURSIVE)
    def test_truncated_by_the_warm_up_the_values_agree(self, feature_id, params):
        """Planted: after resolved_warmup bars the truncated history's
        value is the full history's to within 1e-3 of a standard
        deviation, at every later bar."""
        definition = get_feature(feature_id)
        bars = resolved_warmup(definition, resolve_params(definition, params))
        worst, _ = _worst_error_after(feature_id, params, bars)
        assert worst < 1e-3

    @pytest.mark.parametrize("feature_id,params", RECURSIVE)
    def test_truncated_by_the_first_output_they_do_not(self, feature_id, params):
        """The test has teeth: at resolved_lookback -- the first output --
        the two histories still disagree by a visible fraction of the
        feature's spread."""
        definition = get_feature(feature_id)
        bars = resolved_lookback(definition, resolve_params(definition, params))
        worst, _ = _worst_error_after(feature_id, params, bars)
        assert worst > 0.05

    def test_parabolic_sar_couples_within_its_declared_warm_up(self):
        """A state machine: two runs agree exactly once they reverse on
        the same bar, and the declared 100 bars covers that."""
        definition = get_feature("market.psar_trend")
        resolved = resolve_params(definition, {})
        bars = resolved_warmup(definition, resolved)
        assert bars == 100
        worst, _ = _worst_error_after("market.psar_trend", {}, bars)
        assert worst == 0.0
        at_first_output, _ = _worst_error_after(
            "market.psar_trend", {}, resolved_lookback(definition, resolved)
        )
        assert at_first_output > 0.0


class TestTheClosedForms:
    def test_the_decay_of_one_smoother(self):
        # (13/14)^n <= 1e-4 first at n = 125; EMA(26) at n = 120.
        assert smoother_decay_bars(1.0 / 14.0, 1e-4) == 125
        assert smoother_decay_bars(2.0 / 27.0, 1e-4) == 120
        assert smoother_decay_bars(1.0, 1e-4) == 0

    @pytest.mark.parametrize(
        "feature_id,params,expected",
        [
            ("technical.rsi", {}, 14 + 125),
            ("risk.atr_pct", {}, 14 + 125),
            ("technical.macd_histogram", {}, 26 + 120 + 42),
            ("technical.adx", {}, 187),
            ("market.psar_trend", {}, 100),
            ("market.psar_trend", {"af_start": 0.01}, 200),
        ],
    )
    def test_the_defaults(self, feature_id, params, expected):
        definition = get_feature(feature_id)
        assert resolved_warmup(definition, resolve_params(definition, params)) == (
            expected
        )
        assert is_recursive(definition)

    def test_it_scales_with_the_period(self):
        rsi = get_feature("technical.rsi")
        short = resolved_warmup(rsi, resolve_params(rsi, {"period": 14}))
        long = resolved_warmup(rsi, resolve_params(rsi, {"period": 50}))
        assert 3.4 < long / short < 3.8

    def test_a_looser_tolerance_needs_fewer_bars(self):
        rsi = get_feature("technical.rsi")
        resolved = resolve_params(rsi, {})
        assert resolved_warmup(rsi, resolved, 1e-3) < resolved_warmup(rsi, resolved)
        assert WARMUP_TOLERANCE == 1e-4

    @pytest.mark.parametrize(
        "feature_id,params",
        [
            ("market.momentum", {}),
            ("risk.realized_volatility", {}),
            ("statistical.hurst", {"window": 300}),
            ("volume.obv_roc", {}),
        ],
    )
    def test_a_finite_window_is_warm_at_its_first_output(self, feature_id, params):
        """The null case: no start value to forget."""
        definition = get_feature(feature_id)
        resolved = resolve_params(definition, params)
        assert resolved_warmup(definition, resolved) == resolved_lookback(
            definition, resolved
        )
        assert not is_recursive(definition)


class TestTheEstimatorReportsIt:
    def test_converged_beside_resolved_and_a_second_total(self):
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[
                    FeatureSpec(id="technical.rsi", lags=[5]),
                    FeatureSpec(id="market.momentum"),
                ]
            )
        )
        rsi = result.per_feature["technical.rsi"]
        momentum = result.per_feature["market.momentum"]
        # What bars_required always meant is unchanged.
        assert rsi.resolved == 14 and result.bars_required == 20 + 5
        assert rsi.converged == 139 and rsi.recursive
        assert momentum.converged == momentum.resolved == 20
        assert not momentum.recursive
        assert result.bars_required_converged == 139 + 5
        assert result.converged_binding_feature == "technical.rsi"
        assert result.calendar_days_converged == pytest.approx(144 / 252 * 365.25)
        assert result.warmup_tolerance == WARMUP_TOLERANCE
        assert any("recursive feature" in w for w in result.warnings)

    def test_no_recursive_feature_no_second_number_and_no_warning(self):
        """The null case."""
        result = estimate_feature_warmup(
            EstimateFeatureWarmupInput(
                features=[FeatureSpec(id="market.momentum", params={"lookback": 60})]
            )
        )
        assert result.bars_required_converged == result.bars_required == 60
        assert result.calendar_days_converged == result.calendar_days_estimate
        assert not any("recursive feature" in w for w in result.warnings)


def _spec(features) -> DatasetSpec:
    return DatasetSpec(
        universe=["AAA", "BBB"],
        start="2020-01-01",
        end="2022-12-31",
        features=features,
        target=TargetSpec(horizon=5),
    )


class TestAShortScoringWindowIsNamed:
    def test_a_window_shorter_than_the_converged_warm_up_warns(self):
        """120 calendar days hold about 83 bars; RSI(14) needs 139 before
        its value is the one a longer history gives."""
        warning = _unconverged_window_warning(
            _spec([FeatureSpec(id="technical.rsi")]), 120, "score_model"
        )
        assert warning is not None
        assert "technical.rsi" in warning and "139" in warning
        assert "lookback_days of at least 202" in warning

    def test_the_default_window_covers_the_default_parameters(self):
        features = [
            FeatureSpec(id="technical.rsi"),
            FeatureSpec(id="technical.adx"),
            FeatureSpec(id="technical.macd_histogram"),
            FeatureSpec(id="risk.atr_pct"),
            FeatureSpec(id="market.psar_trend"),
        ]
        assert _unconverged_window_warning(_spec(features), 400, "score_model") is None

    def test_a_finite_window_feature_never_warns(self):
        """The null case: however short the window, a finite window's first
        output is already the value a longer history gives; running out of
        rows is the refusal's business, not this warning's."""
        features = [FeatureSpec(id="market.momentum", params={"lookback": 200})]
        assert _unconverged_window_warning(_spec(features), 100, "score_model") is None
