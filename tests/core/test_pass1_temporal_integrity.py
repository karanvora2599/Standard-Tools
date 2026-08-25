"""
Regression tests for Pass 1 of the full-codebase audit: the findings that
produce a temporally wrong answer, a security hole, or a silently benign
reading of missing data.

Each test pins the CAUSE, not just the corrected output, so a change that
reintroduces the mechanism fails with an explanation rather than a bare
assertion mismatch. Every number quoted in a docstring was measured against
the pre-fix code.
"""

import json
import math
import pathlib
import shutil
import tempfile
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import _shared as shared_tools
from standard_quant_tools.error import AuditIntegrityError, ValidationError


class _Marker:
    """Module-level so joblib can pickle it — a class defined inside a test
    method is not importable from the pickle and fails to dump."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


# ── 1. Model registry: manifest.json is the commit point ────────────────


class TestManifestIsRequiredToLoad:
    """
    _expected_hash() swallowed a ValidationError from load_manifest() and
    returned None, and verify_file() treats expected=None as "skip
    verification". So deleting manifest.json -- strictly easier than forging
    a digest inside it -- downgraded every integrity check at once, and
    joblib.load executes code from the file it is handed.
    """

    def _registered(self, tmp_path, monkeypatch, tag="ORIGINAL"):
        import joblib

        from standard_quant_tools.modeling import artifacts as A
        from standard_quant_tools.modeling.registry.manifests import ModelManifest

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        mid = "mdl_probe"
        d = pathlib.Path(A.run_dir(mid))
        d.mkdir(parents=True, exist_ok=True)

        joblib.dump(_Marker(tag), d / "model.joblib")
        manifest = ModelManifest(
            model_id=mid,
            version=1,
            task="regression",
            estimator_type="ridge",
            estimator_params={},
            feature_ids=["f"],
            target_id="t",
            dataset_id="ds",
            dataset_hash="x" * 16,
            validation_method="walk_forward",
            oos_metrics={},
            feature_importance_summary={},
            n_folds=2,
            oos_predictions_uri="u",
            random_seed=1,
            created_at_utc="2026-01-01T00:00:00Z",
            content_hashes={"model.joblib": A.hash_file(d / "model.joblib")},
        )
        (d / "manifest.json").write_text(manifest.model_dump_json())
        return mid, d, manifest

    def test_intact_package_loads(self, tmp_path, monkeypatch):
        from standard_quant_tools.modeling.registry import model_registry as R

        mid, _, _ = self._registered(tmp_path, monkeypatch)
        assert R.load_model(mid).tag == "ORIGINAL"

    def test_tampered_artifact_with_manifest_present_is_refused(
        self, tmp_path, monkeypatch
    ):
        import joblib

        from standard_quant_tools.modeling.registry import model_registry as R

        mid, d, _ = self._registered(tmp_path, monkeypatch)

        joblib.dump(_Marker("TAMPERED"), d / "model.joblib")
        with pytest.raises(
            ValidationError, match="has changed since it was registered"
        ):
            R.load_model(mid)

    def test_deleting_the_manifest_no_longer_bypasses_verification(
        self, tmp_path, monkeypatch
    ):
        """
        The bypass. Before the fix this DESERIALIZED the tampered file --
        the same input the previous test refuses -- because a missing
        manifest was read as "no expected hash" rather than "the package is
        not intact".
        """
        import joblib

        from standard_quant_tools.modeling.registry import model_registry as R

        mid, d, _ = self._registered(tmp_path, monkeypatch)

        joblib.dump(_Marker("TAMPERED"), d / "model.joblib")
        (d / "manifest.json").unlink()
        with pytest.raises(ValidationError, match="no registered model"):
            R.load_model(mid)

    def test_legacy_manifest_without_a_digest_still_loads(self, tmp_path, monkeypatch):
        """
        The one case that legitimately yields expected=None: a VALID manifest
        that simply predates content hashing. Refusing these would make an
        upgrade look like mass corruption, so the back-compat path must
        survive the fix.
        """
        from standard_quant_tools.modeling.registry import model_registry as R

        mid, d, manifest = self._registered(tmp_path, monkeypatch)
        (d / "manifest.json").write_text(
            manifest.model_copy(update={"content_hashes": {}}).model_dump_json()
        )
        assert R.load_model(mid).tag == "ORIGINAL"


# ── 2. Strategy parameters: the look-ahead one ──────────────────────────


class TestStrategyParameterContract:
    """
    Not one of the eight registered strategies validated a single parameter.
    momentum_timeseries(lookback=-20) reached Close.pct_change(periods=-20),
    and a negative period makes pandas look FORWARD: standing at bar 25 it
    returns close[25]/close[45] - 1. Direct look-ahead from an
    ordinary-looking integer, reachable from the agent surface because
    BacktestInput.parameters was an unconstrained Dict[str, Any].
    """

    def _ohlcv(self, n=120):
        idx = pd.date_range("2023-01-02", periods=n, freq="B")
        close = pd.Series(np.linspace(100.0, 140.0, n), index=idx)
        return pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": 1_000_000.0,
            },
            index=idx,
        )

    def test_pandas_really_does_read_forward_on_a_negative_period(self):
        """Pins the mechanism itself, so the guard is never mistaken for
        excess caution about a merely odd input."""
        df = self._ohlcv()
        forward = df["Close"].pct_change(periods=-20)
        expected = df["Close"].iloc[25] / df["Close"].iloc[45] - 1.0
        assert forward.iloc[25] == pytest.approx(expected)
        assert forward.iloc[-20:].isna().all(), "the NaNs sit at the FUTURE end"

    @pytest.mark.parametrize("bad", [-20, 0])
    def test_non_positive_window_rejected_at_the_registry(self, bad):
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        with pytest.raises(ValidationError, match="must be >= 1"):
            STRATEGY_REGISTRY["momentum_timeseries"](self._ohlcv(), lookback=bad)

    def test_rejection_names_look_ahead_as_the_reason(self):
        from standard_quant_tools.backtest.strategy_params import (
            resolve_strategy_params,
        )

        with pytest.raises(ValidationError, match="FORWARD window"):
            resolve_strategy_params("momentum_timeseries", {"lookback": -20})

    def test_nan_threshold_rejected_because_it_makes_a_strategy_inert(self):
        """NaN fails every comparison, so it does not tighten a strategy --
        it silently flattens it to never-in-market, which looks exactly like
        a strategy that honestly found no trades."""
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        with pytest.raises(ValidationError, match="must be finite"):
            STRATEGY_REGISTRY["adx_trend"](self._ohlcv(), adx_threshold=float("nan"))

    def test_unknown_parameter_rejected_rather_than_swallowed(self):
        """Every signature ends in **_, so a typo silently ran the default
        and the caller believed it had configured something."""
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        with pytest.raises(ValidationError, match="unknown parameter"):
            STRATEGY_REGISTRY["sma_crossover"](self._ohlcv(), fast_perio=5)

    def test_every_registry_entry_is_covered(self):
        """A single unwrapped entry is a call site with no contract."""
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        for name, fn in STRATEGY_REGISTRY.items():
            assert hasattr(fn, "__wrapped__"), f"{name} unvalidated"

    def test_all_eight_still_run_on_their_defaults(self):
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        df = self._ohlcv()
        for name, fn in STRATEGY_REGISTRY.items():
            assert len(fn(df)) == len(df), name

    def test_grid_style_relations_are_not_enforced_on_the_hot_path(self):
        """
        backtest_grid sweeps a rectangle that necessarily contains fast >=
        slow, and it does not catch per-combination errors — so enforcing
        relations inside the registry would abort a whole sweep over points
        a search should simply score badly and move on from.
        """
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        signals = STRATEGY_REGISTRY["sma_crossover"](
            self._ohlcv(), fast_period=50, slow_period=10
        )
        assert len(signals) == 120

    def test_but_a_deliberate_single_config_does_enforce_them(self):
        from standard_quant_tools.backtest.strategy_params import (
            resolve_strategy_params,
        )

        with pytest.raises(ValidationError, match="must be <"):
            resolve_strategy_params(
                "sma_crossover", {"fast_period": 50, "slow_period": 10}
            )


# ── 3. The engine's look-ahead warning must reach the agent ─────────────


class TestLookAheadWarningReachesTheAgent:
    def test_close_fill_warning_is_propagated(self):
        """
        run_strategy has always emitted a look-ahead caveat for
        fill_price="close". _run_backtest rebuilt BacktestResult without it
        and the model had no field for it, so the engine knew the simulation
        might contain look-ahead while the agent-facing output said nothing.
        """
        from standard_quant_tools.agent import tools as T
        from standard_quant_tools.agent.models import BacktestInput

        n = 120
        idx = pd.date_range("2023-01-02", periods=n, freq="B")
        close = pd.Series(np.linspace(100.0, 140.0, n), index=idx)
        df = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": 1_000_000.0,
            },
            index=idx,
        )
        signals = pd.Series(
            np.where(np.arange(n) % 20 < 10, 1, 0), index=idx, dtype=float
        )
        inp = BacktestInput(
            symbol="AAA",
            start_date="2023-01-02",
            end_date="2024-01-01",
            strategy_type="sma_crossover",
            fill_price="close",
        )
        result = shared_tools._run_backtest(inp, df, signals)
        assert any("look" in w.lower() or "fill_price" in w for w in result.warnings)

        conservative = shared_tools._run_backtest(
            inp.model_copy(update={"fill_price": "next_open"}), df, signals
        )
        assert conservative.warnings == [], "next_open carries no such caveat"

    def test_fill_price_and_strategy_type_are_constrained(self):
        from pydantic import ValidationError as PydanticValidationError

        from standard_quant_tools.agent.models import BacktestInput

        base = dict(symbol="A", start_date="2023-01-02", end_date="2024-01-01")
        with pytest.raises(PydanticValidationError):
            BacktestInput(strategy_type="sma_crossover", fill_price="magic", **base)
        with pytest.raises(PydanticValidationError):
            BacktestInput(strategy_type="hallucinated_strategy", **base)

    def test_all_registered_strategies_are_accepted_by_the_schema(self):
        """The description used to list four of eight, so half the registry
        was undiscoverable from the schema."""
        from standard_quant_tools.agent.models import BacktestInput
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        for name in STRATEGY_REGISTRY:
            BacktestInput(
                symbol="A",
                start_date="2023-01-02",
                end_date="2024-01-01",
                strategy_type=name,
            )


# ── 4. Signal-panel calendar preservation ───────────────────────────────


class TestSignalPanelCalendarPreservation:
    """
    run_strategy intersects price dates with signal dates and takes
    pct_change() over what REMAINS, so a sparse signal deletes the
    intervening trading days rather than holding through them. Measured on a
    120-bar daily series driven by identical exposure: annualized volatility
    0.0241 with a daily signal against 0.7735 with the same signal sampled
    monthly -- a 32x distortion of risk from the same prices.
    """

    def _fixture(self, n=250):
        idx = pd.date_range("2023-01-02", periods=n, freq="B")
        rng = np.random.default_rng(3)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n)), index=idx)
        df = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": 1_000_000.0,
            },
            index=idx,
        )
        daily = pd.DataFrame({"AAA": pd.Series(1.0, index=idx)})
        monthly = pd.DataFrame({"AAA": pd.Series(1.0, index=idx).resample("ME").last()})
        return {"AAA": df}, daily, monthly

    def test_sparse_signal_keeps_the_full_price_calendar(self):
        from standard_quant_tools.backtest.panel import run_signal_panel_backtest

        price, daily, monthly = self._fixture()
        n_daily = len(run_signal_panel_backtest(price, daily)["portfolio_returns"])
        n_monthly = len(run_signal_panel_backtest(price, monthly)["portfolio_returns"])
        assert n_monthly == n_daily, "a 12-row signal must not shrink 250 price bars"

    def test_hold_and_flat_are_different_and_both_available(self):
        from standard_quant_tools.backtest.panel import run_signal_panel_backtest

        price, _, monthly = self._fixture()
        held = run_signal_panel_backtest(price, monthly, signal_calendar_policy="hold")
        flat = run_signal_panel_backtest(price, monthly, signal_calendar_policy="flat")
        assert (
            held["portfolio_metrics"]["annualized_volatility"]
            > flat["portfolio_metrics"]["annualized_volatility"]
        ), "holding through the month carries more risk"

    def test_error_policy_refuses_to_guess(self):
        from standard_quant_tools.backtest.panel import run_signal_panel_backtest

        price, _, monthly = self._fixture()
        with pytest.raises(ValidationError, match="signal covers"):
            run_signal_panel_backtest(price, monthly, signal_calendar_policy="error")

    def test_hold_does_not_backfill_before_the_first_signal(self):
        """Back-filling would be look-ahead: no view had been expressed yet."""
        from standard_quant_tools.backtest.panel import _align_signal_to_calendar

        idx = pd.date_range("2023-01-02", periods=10, freq="B")
        sparse = pd.Series([1.0], index=[idx[5]])
        aligned = _align_signal_to_calendar(sparse, idx, "hold", "AAA")
        assert (aligned.iloc[:5] == 0.0).all()
        assert (aligned.iloc[5:] == 1.0).all()

    def test_empty_universe_rejected_before_dividing_by_zero(self):
        from standard_quant_tools.backtest.panel import run_signal_panel_backtest

        price, _, _ = self._fixture()
        with pytest.raises(ValidationError, match="no columns"):
            run_signal_panel_backtest(price, pd.DataFrame())


# ── 5. Intraday timestamps are a canonical instant ──────────────────────


class TestIntradayTimezoneCanonicalization:
    """
    tz_localize(None) without converting first keeps the LOCAL wall clock, so
    London 15:00 BST (14:00 UTC) and New York 15:00 EDT (19:00 UTC) both
    became naive 15:00 -- five hours apart, indexed identically, and joined
    as one instant by any cross-market correlation, PCA or panel.
    """

    def _tz_frame(self, tz):
        return pd.DataFrame(
            {"Close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-06-03 15:00", periods=3, freq="h", tz=tz),
        )

    def test_two_venues_no_longer_look_simultaneous(self):
        from standard_quant_tools.data._cache import _normalize_ohlcv_index

        london = _normalize_ohlcv_index(self._tz_frame("Europe/London"), "1h")
        newyork = _normalize_ohlcv_index(self._tz_frame("America/New_York"), "1h")
        assert not london.index.equals(newyork.index)

    def test_each_bar_keeps_its_true_utc_instant(self):
        from standard_quant_tools.data._cache import _normalize_ohlcv_index

        london = _normalize_ohlcv_index(self._tz_frame("Europe/London"), "1h")
        newyork = _normalize_ohlcv_index(self._tz_frame("America/New_York"), "1h")
        assert str(london.index[0]) == "2024-06-03 14:00:00"
        assert str(newyork.index[0]) == "2024-06-03 19:00:00"

    @pytest.mark.parametrize("tz", ["America/New_York", "Asia/Tokyo", "Europe/London"])
    def test_daily_bars_keep_their_local_trading_date(self, tz):
        """
        Daily deliberately does NOT convert. A daily bar is identified by its
        local session date, and converting first would shift it -- Tokyo
        2024-06-03 00:00 JST is 2024-06-02 15:00 UTC, i.e. the wrong day.
        """
        from standard_quant_tools.data._cache import _normalize_ohlcv_index

        df = pd.DataFrame(
            {"Close": [1.0, 2.0]},
            index=pd.date_range("2024-06-03", periods=2, tz=tz),
        )
        assert str(_normalize_ohlcv_index(df, "1d").index[0].date()) == "2024-06-03"

    def test_cache_format_version_was_bumped(self):
        """v2 intraday files hold local wall-clock times, so serving one
        would answer the same request with a different instant than a live
        fetch."""
        from standard_quant_tools.data._cache import _CACHE_FORMAT_VERSION

        assert _CACHE_FORMAT_VERSION == "v3"


# ── 6/7. Audit chain fails closed ───────────────────────────────────────


class TestAuditChainFailsClosed:
    """
    A corrupted tail returned None, which the caller turned into the genesis
    hash -- so the writer silently STARTED A NEW CHAIN and kept appending as
    though the trail had just begun. A tamper-evident log that quietly
    re-genesises on damage is no longer evidence of anything.
    """

    def _writer(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path))
        from standard_quant_tools.audit.writer import AuditWriter

        return AuditWriter()

    def _record(self, i):
        from standard_quant_tools.audit.models import DecisionRecord

        return DecisionRecord(
            request_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            tool_name=f"t{i}",
            input={"i": i},
            output={"o": i},
            cpp_available=False,
            duration_ms=1.0,
            status="success",
        )

    def test_clean_chain_links_record_to_record(self, tmp_path, monkeypatch):
        w = self._writer(tmp_path, monkeypatch)
        w.write(self._record(1))
        path = pathlib.Path(w.write(self._record(2)))
        lines = path.read_text().splitlines()
        first, second = json.loads(lines[0]), json.loads(lines[1])
        assert second["prev_record_hash"] == first["record_hash"]

    def test_corrupt_tail_refuses_the_append(self, tmp_path, monkeypatch):
        w = self._writer(tmp_path, monkeypatch)
        w.write(self._record(1))
        path = pathlib.Path(w.write(self._record(2)))
        lines = path.read_text().splitlines()
        lines[-1] = '{"tool_name": "t2", "record_hash": TRUNCA'
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(AuditIntegrityError, match="corrupt"):
            w.write(self._record(3))

    def test_record_without_a_hash_also_refuses(self, tmp_path, monkeypatch):
        w = self._writer(tmp_path, monkeypatch)
        path = pathlib.Path(w.write(self._record(1)))
        path.write_text(json.dumps({"tool_name": "t1"}) + "\n")
        with pytest.raises(AuditIntegrityError, match="no record_hash"):
            w.write(self._record(2))

    def test_a_fresh_trail_is_still_a_legitimate_genesis(self, tmp_path, monkeypatch):
        """Absent/empty and corrupt are different states; only the second is
        an error."""
        from standard_quant_tools.audit.paths import _GENESIS_HASH

        w = self._writer(tmp_path, monkeypatch)
        path = pathlib.Path(w.write(self._record(1)))
        first = json.loads(path.read_text().splitlines()[0])
        assert first["prev_record_hash"] == _GENESIS_HASH


# ── 8. Unknown is not zero ──────────────────────────────────────────────


class TestUnknownIsNotZero:
    """
    Three places used a valid-looking number as a failure sentinel, and each
    biased toward the reassuring answer.
    """

    def test_beta_not_estimable_is_nan_not_zero(self):
        from standard_quant_tools.analysis.regression import calculate_beta

        a = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-06-01", periods=3))
        b = pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2020-01-01", periods=3))
        assert math.isnan(calculate_beta(a, b)["beta"])

    def test_treynor_does_not_turn_a_missing_beta_into_a_ratio(self):
        """beta == 0 previously returned 0.0 -- a plausible-looking
        risk-adjusted return built from no overlapping benchmark data."""
        from standard_quant_tools.metrics.risk_metrics import treynor_ratio

        a = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-06-01", periods=3))
        b = pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2020-01-01", periods=3))
        assert math.isnan(treynor_ratio(a, b))

    @pytest.mark.parametrize("adv", [0.0, -5.0, float("nan")])
    def test_unknown_liquidity_is_not_free(self, adv):
        """
        adv_participation(1e9, adv=0) returned 0.0 -- the score of a trade so
        small it barely touches the market -- so the ticker with NO volume
        data ranked as the easiest in the universe to trade. The honest
        comparison is 0.0 against 100.0 (100x ADV) for a real baseline.
        """
        from standard_quant_tools.backtest.constraints import adv_participation
        from standard_quant_tools.backtest.costs import impact_cost

        assert math.isnan(adv_participation(1e9, adv))
        assert math.isnan(impact_cost(1e9, adv, 0.30))

    def test_a_real_baseline_still_measures_a_real_cost(self):
        from standard_quant_tools.backtest.constraints import adv_participation
        from standard_quant_tools.backtest.costs import impact_cost

        assert adv_participation(1e9, 1e7) == pytest.approx(100.0)
        assert impact_cost(1e9, 1e7, 0.30) > 0.0

    def test_nan_volume_no_longer_evades_the_positivity_guard(self):
        """NaN satisfies neither `<= 0` nor `> 0`, so it passed a guard
        written as a comparison and produced a NaN answer that looked
        computed."""
        from standard_quant_tools.backtest.constraints import days_to_liquidate

        with pytest.raises(ValidationError, match="finite"):
            days_to_liquidate(1e6, float("nan"), 0.1)


# ── 9. Optimizer scalars ────────────────────────────────────────────────


class TestOptimizerScalarValidation:
    def _returns(self):
        rng = np.random.default_rng(5)
        return pd.DataFrame(
            {"A": rng.normal(0.0008, 0.012, 600), "B": rng.normal(0.0006, 0.014, 600)}
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(objective="max_sharpe", risk_free_rate=float("nan")),
            dict(objective="target_volatility", target_volatility=float("nan")),
            dict(objective="target_return", target_return=float("nan")),
        ],
    )
    def test_nan_scalars_rejected_rather_than_producing_nan_weights(self, kwargs):
        """
        Every domain guard here is a comparison, and NaN makes all of them
        False -- so `if target_volatility <= 0` never fired for NaN. The
        result was {ticker: nan} weights reported with converged=True.
        """
        from standard_quant_tools.portfolio.optimize import mean_variance_optimize

        with pytest.raises(ValidationError, match="must be finite"):
            mean_variance_optimize(
                self._returns(), allow_short=True, max_weight=None, **kwargs
            )

    @pytest.mark.parametrize("bad", [0, -252, 2.5])
    def test_periods_per_year_must_be_a_positive_integer(self, bad):
        from standard_quant_tools.portfolio.optimize import mean_variance_optimize

        with pytest.raises(ValidationError, match="periods_per_year"):
            mean_variance_optimize(
                self._returns(),
                objective="min_volatility",
                allow_short=True,
                max_weight=None,
                periods_per_year=bad,
            )

    def test_valid_scalars_still_solve(self):
        from standard_quant_tools.portfolio.optimize import mean_variance_optimize

        result = mean_variance_optimize(
            self._returns(),
            objective="min_volatility",
            allow_short=True,
            max_weight=None,
        )
        assert result["converged"]
        assert all(np.isfinite(v) for v in result["weights"].values())
