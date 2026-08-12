"""
Regression tests for the P0 correctness findings in the modeling runtime.

Every test here pins a defect that the pre-existing 2,099-test suite did
not catch, and each docstring records *why* it was missed — in several
cases the existing test asserted the wrong thing (a wrapper against its
own broken primitive) or happened to pick parameters that masked the bug
(embargo == horizon).
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.pca import pca_returns
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.target import (
    build_label_end_dates,
    build_target,
)
from standard_quant_tools.modeling.features.params import (
    resolve_params,
    resolved_lookback,
)
from standard_quant_tools.modeling.features.registry import get_feature
from standard_quant_tools.modeling.specs import TargetSpec

# ── P0-1/2: forward-target leakage across walk-forward folds ────────────────


class TestLabelOverlapPurge:
    """
    `target[t]` reads `Close[t+horizon]`, so a training row's LABEL is only
    resolved once bar t+horizon prints. WalkForwardSplit is never given the
    horizon — it only honours an integer `embargo` — so with
    horizon=20/embargo=0 the last 20 training labels were built from
    test-period prices.

    The pre-existing engine tests used embargo == horizon, which
    accidentally satisfied the missing invariant and hid it.
    """

    def test_label_end_is_horizon_bars_ahead(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        close = pd.Series(np.arange(10, dtype=float), index=idx)
        ends = build_label_end_dates(close, TargetSpec(horizon=3))
        assert ends.iloc[0] == idx[3]
        assert ends.iloc[6] == idx[9]
        # The final `horizon` rows have no resolved label.
        assert ends.tail(3).isna().all()

    def test_label_end_follows_entity_calendar_not_global_offset(self):
        """
        The reason purging must use a real timestamp rather than an integer
        embargo: `horizon` counts THIS ENTITY'S bars. With missing trading
        days, t+horizon entity bars is a different calendar date than
        t+horizon global panel dates, so an integer embargo under-purges.
        """
        idx = pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06", "2024-01-07"]
        )
        close = pd.Series(np.arange(5, dtype=float), index=idx)
        ends = build_label_end_dates(close, TargetSpec(horizon=2))
        # 2 entity bars ahead of 2024-01-01 is 2024-01-05, i.e. +4 calendar
        # days -- an embargo of 2 *dates* would not have covered it.
        assert ends.iloc[0] == pd.Timestamp("2024-01-05")

    def test_label_end_consistent_with_target_definition(self):
        """The purge is only correct if label_end_date marks the same bar
        build_target actually read."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        close = pd.Series(100 + np.arange(30, dtype=float), index=idx)
        spec = TargetSpec(horizon=5)
        target = build_target(close, spec)
        ends = build_label_end_dates(close, spec)
        t = 3
        expected = close.loc[ends.iloc[t]] / close.iloc[t] - 1.0
        assert np.isclose(target.iloc[t], expected)

    def test_purge_predicate_drops_overlapping_training_rows(self):
        """
        The engine's purge condition itself: with horizon=20 and embargo=0,
        20 training rows have labels reaching into the test window and must
        be dropped.
        """
        idx = pd.date_range("2024-01-01", periods=300, freq="D")
        close = pd.Series(100 + np.arange(300, dtype=float), index=idx)
        ends = build_label_end_dates(close, TargetSpec(horizon=20))

        train_dates = idx[:200]
        first_test_date = idx[200]
        train_ends = ends.loc[train_dates]
        overlapping = (train_ends >= first_test_date).sum()
        assert overlapping == 20, "horizon=20, embargo=0 leaks exactly 20 labels"
        kept = train_ends[train_ends < first_test_date]
        assert len(kept) == 180
        assert kept.max() < first_test_date


# ── P0-3/4: negative feature parameters create future leakage ───────────────


class TestFeatureParamValidation:
    """
    `FeatureSpec.params` was an unrestricted `Dict[str, object]` splatted
    straight into the feature. `market.momentum` passes `lookback` to
    `Series.pct_change(periods=lookback)`, and pandas reads a NEGATIVE
    period as a FORWARD window — so `lookback=-20` made the feature at t
    read `Close[t+20]` while its declared TemporalSupport stayed PIT_SAFE.
    The point-in-time gate only ever inspected the static label.
    """

    @pytest.mark.parametrize("feature_id", ["market.momentum", "volume.obv_roc"])
    @pytest.mark.parametrize("bad", [-20, -1, 0])
    def test_non_positive_window_rejected(self, feature_id, bad):
        definition = get_feature(feature_id)
        with pytest.raises(ValidationError, match=r">= 1"):
            resolve_params(definition, {"lookback": bad})

    def test_negative_lookback_would_have_read_the_future(self):
        """Demonstrates the leak the validation now prevents, so the test
        fails loudly if the guard is ever removed."""
        close = pd.Series(np.arange(100, 140, dtype=float))
        forward = close.pct_change(periods=-20)
        assert np.isclose(forward.iloc[0], close.iloc[0] / close.iloc[20] - 1)
        with pytest.raises(ValidationError):
            resolve_params(get_feature("market.momentum"), {"lookback": -20})

    def test_unknown_param_name_is_a_validation_error(self):
        """Previously surfaced as a raw TypeError from inside the feature."""
        with pytest.raises(ValidationError, match="unknown parameter"):
            resolve_params(get_feature("market.momentum"), {"lookbak": 20})

    def test_wrong_type_rejected(self):
        with pytest.raises(ValidationError, match="must be a number"):
            resolve_params(get_feature("market.momentum"), {"lookback": "20"})

    def test_non_finite_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            resolve_params(get_feature("risk.atr_pct"), {"period": float("inf")})

    def test_valid_params_pass_through_and_merge_defaults(self):
        definition = get_feature("market.momentum")
        assert resolve_params(definition, {}) == {"lookback": 20}
        assert resolve_params(definition, {"lookback": 60}) == {"lookback": 60}

    def test_resolved_lookback_tracks_params_not_the_static_label(self):
        """`FeatureDefinition.lookback` is registered against the DEFAULT
        params, so it stays 20 even when 500 bars are requested."""
        definition = get_feature("market.momentum")
        assert definition.lookback == 20
        assert (
            resolved_lookback(definition, resolve_params(definition, {"lookback": 500}))
            == 500
        )


# ── P0-5: volume.obv_roc produced non-finite values on ordinary data ───────


class TestObvRocFinite:
    """
    OBV is a cumulative sum seeded at exactly 0.0 (bar 0 has no prior close,
    so its direction is 0). `obv.pct_change(20)` therefore divides by
    OBV[0] == 0 at its first valid row and returns +/-inf, and
    build_dataset's finite guard then rejected the ENTIRE panel.

    The old feature test compared the wrapper to
    `obv(...).pct_change(20)` — i.e. asserted the broken behavior — and
    never ran the feature through build_dataset.
    """

    @staticmethod
    def _ohlcv(n=120, seed=0):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": np.full(n, 1_000_000.0),
            },
            index=idx,
        )

    def test_old_formulation_was_non_finite(self):
        from standard_quant_tools.indicators.volume import obv

        df = self._ohlcv()
        old = obv(df["Close"], df["Volume"]).pct_change(20)
        assert not np.isfinite(old.dropna().to_numpy(dtype=float)).all()

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_current_feature_is_finite(self, seed):
        definition = get_feature("volume.obv_roc")
        out = definition.fn(self._ohlcv(seed=seed), None, lookback=20).dropna()
        assert len(out) > 0
        assert np.isfinite(out.to_numpy(dtype=float)).all()

    def test_bounded_by_traded_volume(self):
        definition = get_feature("volume.obv_roc")
        out = definition.fn(self._ohlcv(), None, lookback=20).dropna()
        assert out.abs().max() <= 1.0 + 1e-12

    def test_zero_volume_window_is_nan_not_inf(self):
        df = self._ohlcv()
        df.loc[df.index[:60], "Volume"] = 0.0
        out = get_feature("volume.obv_roc").fn(df, None, lookback=20)
        assert out.iloc[30:59].isna().all()
        assert np.isfinite(out.dropna().to_numpy(dtype=float)).all()


# ── P0-6: PCA power iteration returned the wrong principal component ───────


class TestPcaPowerIterationStartVector:
    """
    Power iteration started from the uniform vector [1..1]/sqrt(n). A
    spread factor with loadings proportional to [1,-1] is exactly
    orthogonal to that, so the first matvec was the zero vector and the
    routine reported the ZERO-eigenvalue direction as PC1.

    Existing tests only used same-sign common-factor structures, where the
    uniform start happens to be close to PC1 already.
    """

    @staticmethod
    def _spread_panel(m=400, seed=7):
        rng = np.random.default_rng(seed)
        spread = rng.normal(0, 1, m)
        noise = rng.normal(0, 0.01, (m, 2))
        return pd.DataFrame({"A": spread + noise[:, 0], "B": -spread + noise[:, 1]})

    def test_spread_factor_matches_svd(self):
        R = self._spread_panel()
        svd = pca_returns(R, n_components=1, method="svd")
        power = pca_returns(R, n_components=1, method="power_iteration")
        np.testing.assert_allclose(
            power["explained_variance_ratio"].to_numpy(),
            svd["explained_variance_ratio"].to_numpy(),
            atol=1e-6,
        )
        # The dominant component really does explain ~everything here --
        # guards against both paths degenerating to the same wrong answer.
        assert float(svd["explained_variance_ratio"].iloc[0]) > 0.99

    def test_loadings_match_svd_up_to_sign(self):
        R = self._spread_panel()
        svd = pca_returns(R, n_components=1, method="svd")["loadings"].to_numpy()
        power = pca_returns(R, n_components=1, method="power_iteration")[
            "loadings"
        ].to_numpy()
        assert np.allclose(power, svd, atol=1e-6) or np.allclose(power, -svd, atol=1e-6)

    def test_same_sign_market_factor_still_matches(self):
        """The case that already worked must keep working."""
        rng = np.random.default_rng(3)
        m = 400
        mkt = rng.normal(0, 1, m)
        R = pd.DataFrame(
            {
                c: mkt * b + rng.normal(0, 0.05, m)
                for c, b in zip("ABCD", [1.0, 0.9, 1.1, 0.8])
            }
        )
        svd = pca_returns(R, n_components=2, method="svd")
        power = pca_returns(R, n_components=2, method="power_iteration")
        np.testing.assert_allclose(
            power["explained_variance_ratio"].to_numpy(),
            svd["explained_variance_ratio"].to_numpy(),
            atol=1e-6,
        )

    def test_rank_deficient_matrix_does_not_raise(self):
        rng = np.random.default_rng(11)
        base = rng.normal(0, 1, 200)
        R = pd.DataFrame({"A": base, "B": base * 2.0, "C": base * -1.5})
        out = pca_returns(R, n_components=2, method="power_iteration")
        assert np.isfinite(out["explained_variance_ratio"].to_numpy()).all()

    def test_zero_variance_matrix_does_not_raise(self):
        R = pd.DataFrame({"A": np.ones(50), "B": np.ones(50)})
        out = pca_returns(R, n_components=1, method="power_iteration")
        assert np.isfinite(out["explained_variance_ratio"].to_numpy()).all()

    def test_invalid_method_rejected(self):
        """`method` was only a type annotation; anything unrecognized
        silently fell through to the SVD branch."""
        with pytest.raises(ValueError, match="method must be"):
            pca_returns(self._spread_panel(), n_components=1, method="oops")

    def test_non_positive_n_components_rejected(self):
        with pytest.raises(ValueError, match="n_components must be"):
            pca_returns(self._spread_panel(), n_components=0)


# ── Full-refit information cutoff (second-pass P0-1) ────────────────────────


class TestTrainingInformationCutoff:
    """
    The full-refit estimator's information cutoff is max(label_end_date),
    NOT max(date).

    A row dated t with a horizon-h forward-return target reads Close[t+h]
    to build its label, so the deployed model has indirectly seen prices h
    bars past the last feature date. score_model gated on train_end_date
    (the feature date), leaving a horizon-wide window -- ~28 calendar days
    at h=20 -- in which it accepted an as_of whose future the model had
    already consumed, and returned a future-trained prediction that looked
    point-in-time.

    The earlier P0 pass added label_end_date and used it for fold purging,
    but the full-refit manifest cutoff kept using the feature date, so this
    survived that pass.
    """

    def test_label_end_max_exceeds_feature_date_max(self):
        """The gap the old guard left open, measured directly."""
        dates = pd.date_range("2026-01-01", periods=120, freq="B")
        close = pd.Series(np.linspace(100, 150, 120), index=dates)
        ends = build_label_end_dates(close, TargetSpec(horizon=20))
        panel = pd.DataFrame({"date": dates, "label_end_date": ends.values}).dropna()

        feature_end = pd.Timestamp(panel["date"].max())
        label_end = pd.Timestamp(panel["label_end_date"].max())
        assert label_end > feature_end
        # 20 business days of forward target -> ~28 calendar days unguarded.
        assert (label_end - feature_end).days >= 20

    def test_manifest_records_label_aware_cutoff(self, patched_multi_factory):
        from standard_quant_tools.modeling.registry.model_registry import (
            load_manifest,
        )

        from .test_scoring import _dataset_spec, _train_a_model_with_spec

        horizon = 20
        spec = _dataset_spec(target=TargetSpec(horizon=horizon))
        model_id = _train_a_model_with_spec(spec, dataset_id="ds_cutoff")
        manifest = load_manifest(model_id)

        assert manifest.training_information_cutoff is not None
        cutoff = pd.Timestamp(manifest.training_information_cutoff)
        feature_end = pd.Timestamp(manifest.train_end_date)
        # The cutoff must sit strictly beyond the last feature date --
        # that difference IS the bug this pins.
        assert cutoff > feature_end

    def test_score_model_rejects_as_of_inside_the_horizon_window(
        self, patched_multi_factory
    ):
        """
        The regression proper: an as_of after the last FEATURE date but at
        or before the label cutoff used to be accepted.
        """
        from standard_quant_tools.modeling.registry.model_registry import (
            load_manifest,
        )
        from standard_quant_tools.modeling.scoring import score_model

        from .test_scoring import _dataset_spec, _train_a_model_with_spec

        spec = _dataset_spec(target=TargetSpec(horizon=20))
        model_id = _train_a_model_with_spec(spec, dataset_id="ds_cutoff_reject")
        manifest = load_manifest(model_id)

        feature_end = pd.Timestamp(manifest.train_end_date)
        cutoff = pd.Timestamp(manifest.training_information_cutoff)
        inside = feature_end + pd.Timedelta(days=1)
        assert inside <= cutoff, "fixture must exercise the previously-open window"

        with pytest.raises(ValidationError, match="training information cutoff"):
            score_model(
                model_id=model_id,
                as_of=inside.strftime("%Y-%m-%d"),
                universe=["AAA", "BBB"],
            )


class TestPowerIterationZeroMatvec:
    """
    The residual convergence check was skipped whenever the computed
    eigenvalue was ~0, on the reasoning that a null direction has nothing
    to converge to. But "we found a direction with no variance" is
    ambiguous on its own: it is legitimate only when there is no variance
    LEFT to find.

    Because the start vector is fixed and deterministic, a matrix whose
    true loading is exactly orthogonal to it can be constructed. Then
    `working @ start == 0` despite genuine variance, the null direction was
    accepted, and the SVD fallback never fired — rarer than the original
    uniform-start bug, but the same silent failure.
    """

    @staticmethod
    def _orthogonal_to_start_panel():
        n_assets = 4
        start = np.random.default_rng(0).standard_normal(n_assets)
        start /= np.linalg.norm(start)
        # Gram-Schmidt a basis vector against `start`.
        w = np.array([1.0, 0.0, 0.0, 0.0]) - start * start[0]
        w /= np.linalg.norm(w)
        assert abs(float(w @ start)) < 1e-12, "fixture must be truly orthogonal"
        z = np.random.default_rng(3).standard_normal(300)
        return pd.DataFrame(np.outer(z, w), columns=[f"A{i}" for i in range(n_assets)])

    def test_dominant_component_orthogonal_to_start_still_found(self):
        R = self._orthogonal_to_start_panel()
        svd = pca_returns(R, n_components=1, method="svd")
        power = pca_returns(R, n_components=1, method="power_iteration")
        np.testing.assert_allclose(
            power["explained_variance_ratio"].to_numpy(),
            svd["explained_variance_ratio"].to_numpy(),
            atol=1e-6,
        )

    def test_genuine_null_space_is_not_treated_as_failure(self):
        """A fully deflated / zero-variance matrix has no remaining
        variance, so a zero eigenvalue there is correct, not a failure —
        the guard must distinguish the two."""
        R = pd.DataFrame({"A": np.ones(60), "B": np.ones(60)})
        out = pca_returns(R, n_components=2, method="power_iteration")
        assert np.isfinite(out["explained_variance_ratio"].to_numpy()).all()


class TestIntervalAwareAnnualization:
    """
    The realized-volatility features annualized with sqrt(252) regardless
    of DatasetSpec.interval, so an "annualized" volatility at a weekly bar
    was wrong by a fixed multiplicative factor while still looking like a
    volatility.
    """

    @staticmethod
    def _ohlcv(n=300):
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        rng = np.random.default_rng(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
        return pd.DataFrame(
            {
                "Open": close * 1.001,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": 1e6,
            },
            index=idx,
        )

    def test_weekly_scales_by_its_own_constant(self):
        from standard_quant_tools.modeling.features.base import FeatureContext

        fn = get_feature("risk.realized_volatility").fn
        df = self._ohlcv()
        daily = fn(df, FeatureContext(interval="1d"), period=20).dropna()
        weekly = fn(df, FeatureContext(interval="1wk"), period=20).dropna()
        # Same per-bar series, different annualization: exactly sqrt(252/52).
        np.testing.assert_allclose(
            (daily / weekly).to_numpy(),
            np.full(len(daily), (252 / 52) ** 0.5),
            rtol=1e-9,
        )

    def test_missing_interval_stays_daily(self):
        """Back-compat: a caller that predates the field must not silently
        change scale."""
        from standard_quant_tools.modeling.features.base import FeatureContext

        fn = get_feature("risk.realized_volatility").fn
        df = self._ohlcv()
        np.testing.assert_allclose(
            fn(df, FeatureContext(), period=20).dropna().to_numpy(),
            fn(df, FeatureContext(interval="1d"), period=20).dropna().to_numpy(),
        )

    @pytest.mark.parametrize(
        "feature_id",
        [
            "risk.realized_volatility",
            "risk.parkinson_volatility",
            "risk.garman_klass_volatility",
        ],
    )
    def test_intraday_is_rejected_not_mis_scaled(self, feature_id):
        """
        Bars per year at "1h" depends on the venue's session length, which
        this package has no calendar for. Guessing one would make the
        number wrong for every other market while still looking precise.
        """
        from standard_quant_tools.modeling.features.base import FeatureContext

        fn = get_feature(feature_id).fn
        with pytest.raises(ValidationError, match="cannot annualize"):
            fn(self._ohlcv(), FeatureContext(interval="1h"), period=20)
