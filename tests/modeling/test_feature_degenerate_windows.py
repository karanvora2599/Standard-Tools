"""
Regression tests for the feature-level findings: degenerate windows,
warm-up handling, and signed feature importance.

The common thread is a feature answering confidently where it had no
information — a fabricated 0.0 across every warm-up bar, an inf that
rejected an entire panel, a halt reported as a dropped row, and a
stability metric that rated the least stable feature perfectly stable.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.features.base import FeatureContext
from standard_quant_tools.modeling.features.registry import get_feature
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec
from standard_quant_tools.modeling.validation.diagnostics import (
    fold_feature_importance,
    summarize_importance,
)

from .conftest import make_ohlcv, make_provider_mock

_CONTEXT = FeatureContext()


def _frame(close, high=None, low=None, volume=None) -> pd.DataFrame:
    n = len(close)
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close if high is None else np.asarray(high, dtype=float),
            "Low": close if low is None else np.asarray(low, dtype=float),
            "Close": close,
            "Volume": np.full(n, 1e6) if volume is None else np.asarray(volume, float),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


# ── market.new_high_breakout warm-up ───────────────────────────────────


class TestBreakoutWarmUp:
    """`NaN > x` is False, so `.astype(float)` turned every warm-up bar
    into a confident "no breakout" — the one feature in the catalog that
    never produced NaN, and so never let alignment drop its own warm-up."""

    def _feature(self, frame, period=20):
        return get_feature("market.new_high_breakout").fn(
            frame, _CONTEXT, period=period
        )

    def test_warm_up_is_nan_not_zero(self):
        result = self._feature(_frame(np.linspace(100, 130, 40)))
        assert result.iloc[:20].isna().all()
        assert result.iloc[20:].notna().all()

    def test_declared_lookback_matches_what_is_consumed(self):
        """lookback=20 described nothing observable while the feature
        emitted a value for every bar from the first."""
        definition = get_feature("market.new_high_breakout")
        result = self._feature(_frame(np.linspace(100, 130, 40)))
        assert int(result.isna().sum()) == definition.lookback

    def test_a_real_breakout_is_still_detected(self):
        close = np.concatenate([np.full(25, 100.0), [120.0]])
        result = self._feature(_frame(close, high=close))
        assert result.iloc[-1] == 1.0

    def test_a_non_breakout_is_still_zero(self):
        close = np.concatenate([np.full(25, 100.0), [99.0]])
        result = self._feature(_frame(close, high=close))
        assert result.iloc[-1] == 0.0

    def test_warm_up_rows_no_longer_reach_the_panel(self, monkeypatch):
        """The consequence that made this worth fixing: a dataset built on
        this feature alone began `period` bars early, with fabricated
        negatives in exactly the rows a breakout model cares about."""
        from standard_quant_tools.data.factory import DataFactory

        provider = make_provider_mock(lambda symbol: make_ohlcv(symbol, n=120))
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        built = build_dataset(
            DatasetSpec(
                universe=["AAA"],
                start="2022-01-01",
                end="2023-12-31",
                features=[FeatureSpec(id="market.new_high_breakout")],
                target=TargetSpec(horizon=5),
            )
        )
        panel = built["panel"]
        first_bar = make_ohlcv("AAA", n=120).index[0]
        # 20 warm-up bars are gone; the panel no longer starts at bar 0.
        assert panel["date"].min() > first_bar


# ── risk.atr_pct non-positive denominator ──────────────────────────────


class TestAtrPctDegenerateDenominator:
    def _feature(self, frame, period=14):
        return get_feature("risk.atr_pct").fn(frame, _CONTEXT, period=period)

    def test_zero_close_yields_nan_not_inf(self):
        close = np.concatenate([np.linspace(100, 110, 25), [0.0], [105.0] * 4])
        result = self._feature(_frame(close, high=close * 1.01, low=close * 0.99))
        assert not np.isinf(result.to_numpy()).any()
        assert np.isnan(result.iloc[25])

    def test_negative_close_yields_nan(self):
        close = np.concatenate([np.linspace(100, 110, 25), [-5.0], [105.0] * 4])
        result = self._feature(_frame(close, high=np.abs(close), low=np.abs(close)))
        assert np.isnan(result.iloc[25])

    def test_normal_prices_are_unaffected(self):
        rng = np.random.default_rng(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
        result = self._feature(_frame(close, high=close * 1.01, low=close * 0.99))
        valid = result.dropna()
        assert len(valid) > 40
        assert (valid > 0).all()

    def test_one_bad_bar_no_longer_rejects_the_whole_panel(self, monkeypatch):
        """The reason this was more than a bad row: inf fails
        build_dataset's finite-value guard, which rejects the ENTIRE panel
        — so one zero print in one symbol failed the whole build, blaming
        the feature rather than the data. Same failure mode volume.obv_roc
        was already fixed for."""
        from standard_quant_tools.data.factory import DataFactory

        def _fetch(symbol):
            frame = make_ohlcv(symbol, n=200)
            if symbol == "BBB":
                frame.iloc[100, frame.columns.get_loc("Close")] = 0.0
            return frame

        provider = make_provider_mock(_fetch)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        built = build_dataset(
            DatasetSpec(
                universe=["AAA", "BBB"],
                start="2022-01-01",
                end="2023-12-31",
                features=[FeatureSpec(id="risk.atr_pct")],
                target=TargetSpec(horizon=5),
            )
        )
        # AAA survives in full; only BBB's affected rows are missing.
        assert not built["panel"].empty
        assert set(built["panel"]["entity"].unique()) == {"AAA", "BBB"}


# ── risk.bollinger_pct_b degenerate band ───────────────────────────────


class TestBollingerPctBFlatWindow:
    def _feature(self, frame, period=20):
        return get_feature("risk.bollinger_pct_b").fn(frame, _CONTEXT, period=period)

    def test_flat_window_is_the_middle_band_not_nan(self):
        """A halted or stale-quoted symbol collapses both bands onto the
        mean. Close equals that mean exactly, so 0.5 is what %B is defined
        to be there — not a fallback, and better than dropping the row."""
        result = self._feature(_frame(np.full(30, 100.0)))
        assert result.iloc[-1] == 0.5

    def test_warm_up_stays_nan(self):
        """The failure this guard must not introduce: conflating "the bands
        collapsed" with "there are not yet `period` bars" would fabricate a
        0.5 for rows with no window at all."""
        result = self._feature(_frame(np.full(30, 100.0)))
        assert result.iloc[:19].isna().all()

    def test_normal_data_is_unchanged(self):
        rng = np.random.default_rng(7)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 80))
        result = self._feature(_frame(close))
        valid = result.dropna()
        # Most observations sit inside the bands; none is degenerate.
        assert ((valid >= 0) & (valid <= 1)).mean() > 0.7
        assert not (valid == 0.5).all()

    def test_a_halt_followed_by_a_jump_stays_bounded(self):
        """Guards the claim in the docstring: only the exactly-degenerate
        case needed handling, because a jump enters the standard deviation
        that scales it rather than blowing the ratio up."""
        close = np.concatenate([np.full(20, 100.0), [101.0]])
        result = self._feature(_frame(close))
        assert np.isfinite(result.iloc[-1])
        assert abs(result.iloc[-1]) < 10


# ── volume.vwap_deviation ──────────────────────────────────────────────


class TestVwapDeviationDenominator:
    def test_zero_volume_window_is_nan_not_inf(self):
        close = np.linspace(100, 110, 40)
        volume = np.concatenate([np.full(20, 1e6), np.zeros(20)])
        result = get_feature("volume.vwap_deviation").fn(
            _frame(close, volume=volume), _CONTEXT, period=20
        )
        assert not np.isinf(result.to_numpy()).any()

    def test_normal_data_is_unchanged(self):
        rng = np.random.default_rng(3)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
        result = get_feature("volume.vwap_deviation").fn(
            _frame(close), _CONTEXT, period=20
        )
        assert result.dropna().abs().max() < 1.0


# ── Signed feature importance ──────────────────────────────────────────


class _FakeLinear:
    def __init__(self, coef):
        self.coef_ = np.asarray(coef, dtype=float)


class _FakeTree:
    def __init__(self, importances):
        self.feature_importances_ = np.asarray(importances, dtype=float)


class _FakeOpaque:
    """Neither coef_ nor feature_importances_ — e.g.
    HistGradientBoostingRegressor."""


class TestSignedFeatureImportance:
    IDS = ["stable_pos", "stable_neg", "sign_flipper"]

    def _summary(self):
        folds = [
            fold_feature_importance(_FakeLinear([0.5, -0.5, 0.5]), self.IDS),
            fold_feature_importance(_FakeLinear([0.5, -0.5, -0.5]), self.IDS),
            fold_feature_importance(_FakeLinear([0.5, -0.5, 0.5]), self.IDS),
            fold_feature_importance(_FakeLinear([0.5, -0.5, -0.5]), self.IDS),
        ]
        return summarize_importance(folds, self.IDS)

    def test_a_sign_flipping_feature_is_no_longer_perfectly_stable(self):
        """The finding: taking |coef| FIRST made +0.5, -0.5, +0.5, -0.5 —
        the maximally unstable case and a textbook sign of fitting noise —
        come out as 0.5 every fold, std exactly 0.0, i.e. reported as
        perfectly stable by the number whose whole job was catching it."""
        summary = self._summary()
        assert summary["sign_flipper"]["std"] == pytest.approx(0.0)
        assert summary["sign_flipper"]["signed_std"] == pytest.approx(0.5)
        assert summary["sign_flipper"]["sign_consistency"] == pytest.approx(0.5)

    def test_a_stable_feature_is_still_reported_as_stable(self):
        summary = self._summary()
        assert summary["stable_pos"]["signed_std"] == pytest.approx(0.0)
        assert summary["stable_pos"]["sign_consistency"] == pytest.approx(1.0)

    def test_direction_is_recoverable(self):
        """A stably negative coefficient is a working contrarian signal;
        magnitude alone made it indistinguishable from a positive one."""
        summary = self._summary()
        assert summary["stable_pos"]["signed_mean"] == pytest.approx(0.5)
        assert summary["stable_neg"]["signed_mean"] == pytest.approx(-0.5)
        assert summary["stable_pos"]["mean"] == summary["stable_neg"]["mean"]

    def test_magnitude_keys_keep_their_meaning(self):
        """mean/std are unchanged so existing manifests stay comparable."""
        summary = self._summary()
        for fid in self.IDS:
            assert summary[fid]["mean"] == pytest.approx(0.5)

    def test_tree_importances_report_no_direction(self):
        """Tree importances are non-negative by construction, so a sign
        statistic over them would report perfect directional agreement for
        a quantity that has no direction."""
        folds = [
            fold_feature_importance(_FakeTree([0.6, 0.3, 0.1]), self.IDS)
            for _ in range(3)
        ]
        summary = summarize_importance(folds, self.IDS)
        assert summary["stable_pos"]["mean"] == pytest.approx(0.6)
        assert np.isnan(summary["stable_pos"]["signed_mean"])
        assert np.isnan(summary["stable_pos"]["sign_consistency"])

    def test_an_estimator_exposing_neither_is_all_nan(self):
        folds = [fold_feature_importance(_FakeOpaque(), self.IDS) for _ in range(3)]
        summary = summarize_importance(folds, self.IDS)
        for key in ("mean", "std", "signed_mean", "sign_consistency"):
            assert np.isnan(summary["stable_pos"][key])

    def test_multiclass_coefficients_are_not_silently_misattributed(self):
        """(n_classes, n_features) ravels to n_classes * n_features, and
        zip() kept the first n_features — reporting class 0's coefficients
        as THE importances and dropping every other class without a word."""
        fold = fold_feature_importance(
            _FakeLinear([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), self.IDS
        )
        assert all(np.isnan(v) for v in fold.values.values())

    def test_zero_coefficients_do_not_vote_on_direction(self):
        """A coefficient driven to exactly 0.0 (routine under L1) expresses
        no direction; letting it vote would let a feature the model
        DISCARDED in most folds read as directionally consistent."""
        ids = ["mostly_zero"]
        folds = [
            fold_feature_importance(_FakeLinear([0.0]), ids),
            fold_feature_importance(_FakeLinear([0.0]), ids),
            fold_feature_importance(_FakeLinear([-0.5]), ids),
        ]
        summary = summarize_importance(folds, ids)
        assert summary["mostly_zero"]["sign_consistency"] == pytest.approx(1.0)
        assert summary["mostly_zero"]["signed_mean"] < 0

    def test_all_zero_coefficients_have_no_direction(self):
        ids = ["dropped"]
        folds = [fold_feature_importance(_FakeLinear([0.0]), ids) for _ in range(3)]
        summary = summarize_importance(folds, ids)
        assert np.isnan(summary["dropped"]["sign_consistency"])
        assert summary["dropped"]["mean"] == pytest.approx(0.0)


class TestSignedImportanceReachesTheModel:
    def test_trained_model_reports_signed_importance(self, patched_multi_factory):
        from standard_quant_tools.modeling.engine import run_experiment
        from standard_quant_tools.modeling.registry.model_registry import load_manifest
        from standard_quant_tools.modeling.specs import (
            EstimatorSpec,
            ModelSpec,
            ValidationSpec,
        )

        built = build_dataset(
            DatasetSpec(
                universe=["AAA", "BBB"],
                start="2022-01-01",
                end="2023-12-31",
                features=[
                    FeatureSpec(id="technical.rsi"),
                    FeatureSpec(id="market.momentum"),
                ],
                target=TargetSpec(horizon=5),
            )
        )
        result = run_experiment(
            built,
            ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            ),
            dataset_id="ds_signed",
        )
        summary = result["feature_importance_summary"]["technical.rsi"]
        assert {"mean", "std", "signed_mean", "signed_std", "sign_consistency"} <= set(
            summary
        )
        # Persisted, not just returned — inspect_model reads the manifest.
        manifest = load_manifest(result["model_id"])
        assert "signed_mean" in manifest.feature_importance_summary["technical.rsi"]


def test_no_feature_produces_inf_on_ordinary_data(patched_multi_factory):
    """A catalog-wide guard rather than one test per feature: inf does not
    corrupt a row, it fails the finite-value check and rejects the whole
    panel, so any feature acquiring one breaks every dataset containing it."""
    from standard_quant_tools.modeling.features.registry import FEATURE_REGISTRY

    spec = DatasetSpec(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id=fid) for fid in sorted(FEATURE_REGISTRY)],
        target=TargetSpec(horizon=5),
    )
    try:
        built = build_dataset(spec)
    except ValidationError as exc:  # pragma: no cover — the regression itself
        pytest.fail(f"a catalog feature produced a non-finite value: {exc}")
    numeric = built["panel"].select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
