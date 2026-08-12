"""
Regression tests for the data/runtime architecture cluster.

Four things the modeling runtime did silently:

  1. Fetched through `DataFactory.get_provider()` with no arguments, so
     every dataset came from the default provider and no model recorded
     which source it was trained on.
  2. Had no `interval`, so daily bars were an unstated assumption rather
     than a choice.
  3. Fetched the universe one symbol at a time, and (via asyncio.gather's
     default) would have reported only the first failure had it been made
     concurrent naively.
  4. Never read the provider's own point_in_time/survivorship_free
     self-report, never reported partial history, and never mentioned that
     complete-case alignment for PCA features truncates the panel — while
     `BuildModelDatasetResult.warnings` sat unpopulated.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import BuildModelDatasetInput
from standard_quant_tools.modeling.agent.tools import build_model_dataset
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.coverage import (
    entity_coverage_warnings,
    intersection_warnings,
    interval_warnings,
    provider_guarantee_warnings,
)
from standard_quant_tools.modeling.dataset.fetch import fetch_universe_ohlcv
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec

from .conftest import make_ohlcv, make_provider_mock, mock_metadata


def _spec(**overrides) -> DatasetSpec:
    base = dict(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi")],
        target=TargetSpec(horizon=5),
    )
    base.update(overrides)
    return DatasetSpec(**base)


# ── Provider selection ─────────────────────────────────────────────────


class TestProviderSelection:
    def test_default_is_yfinance(self):
        assert _spec().provider == "yfinance"

    def test_spec_provider_reaches_the_factory(self, monkeypatch):
        """Previously DataFactory.get_provider() was called with no
        arguments, so `provider` in the spec could not have had any
        effect even once the field existed."""
        seen = []
        provider = make_provider_mock(make_ohlcv)

        def _get_provider(source="yfinance", *a, **kw):
            seen.append(source)
            return provider

        monkeypatch.setattr(DataFactory, "get_provider", _get_provider)
        build_dataset(_spec(provider="polygon"))
        assert seen and seen[0] == "polygon"

    def test_unknown_provider_rejected_at_the_boundary(self):
        with pytest.raises(ValueError):
            _spec(provider="definitely_not_a_provider")

    def test_provider_is_part_of_the_spec_hash(self, monkeypatch):
        """provider/interval live in DatasetSpec, so they are covered by
        spec_hash and bundled into the model — that is what makes a
        model's lineage able to say what it was trained on."""
        provider = make_provider_mock(make_ohlcv)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        yf_hash = build_dataset(_spec(provider="yfinance"))["spec_hash"]
        pg_hash = build_dataset(_spec(provider="polygon"))["spec_hash"]
        assert yf_hash != pg_hash

    def test_no_credential_field_exists_on_the_spec(self):
        """The spec is written to disk, hashed into the model and embedded
        in decision records. A key field here would leak into all three, so
        credentials stay in the environment."""
        fields = set(DatasetSpec.model_fields)
        assert not (fields & {"api_key", "apikey", "token", "secret", "password"})


# ── Interval ───────────────────────────────────────────────────────────


class TestInterval:
    def test_default_is_daily(self):
        assert _spec().interval == "1d"

    def test_interval_is_passed_to_the_provider(self, monkeypatch):
        seen = []

        def _fetch_async(symbol, start, end, interval="1d"):
            seen.append(interval)
            return make_ohlcv(symbol)

        provider = make_provider_mock(make_ohlcv)
        provider.get_ohlcv_async = AsyncMock(side_effect=_fetch_async)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        build_dataset(_spec(interval="1h"))
        assert seen and set(seen) == {"1h"}

    def test_benchmark_uses_the_same_interval(self, monkeypatch):
        """A benchmark fetched at a different interval than the universe
        would misalign risk.rolling_beta without any error."""
        seen = []
        provider = make_provider_mock(make_ohlcv)
        provider.get_ohlcv.side_effect = lambda s, start, end, interval="1d": (
            seen.append(interval) or make_ohlcv(s)
        )
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        build_dataset(_spec(interval="1wk"))
        # The universe goes through the async path; the benchmark is the
        # sync one, so this records the benchmark fetch specifically.
        assert seen == ["1wk"]

    def test_daily_interval_produces_no_interval_warning(self):
        assert interval_warnings("1d") == []

    def test_non_daily_interval_warns_about_daily_calibration(self):
        """
        The surviving caveat is that feature/target windows count BARS, so
        a default calibrated for daily bars means something else at another
        interval.

        This test used to also require the word "annualize", because the
        volatility features annualized with a daily constant regardless of
        interval — a knowingly wrong absolute scale. That is now fixed at
        the source (features/risk.py resolves the constant from the
        interval, and rejects intervals where none exists without an
        exchange calendar), so warning about it would be describing
        behavior the code no longer has.
        """
        (message,) = interval_warnings("1h")
        assert "1h" in message
        assert (
            "252" in message
        ), "the concrete default must be named, not just 'defaults'"
        assert "BARS" in message


# ── Concurrent fetch ───────────────────────────────────────────────────


class TestConcurrentFetch:
    def test_universe_is_fetched_through_the_async_path(self, monkeypatch):
        provider = make_provider_mock(make_ohlcv)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        build_dataset(_spec())
        assert provider.get_ohlcv_async.await_count == 3

    def test_all_failures_reported_together(self):
        def _fetch(symbol):
            if symbol in {"AAA", "CCC"}:
                raise ConnectionError(f"{symbol} down")
            return make_ohlcv(symbol)

        provider = make_provider_mock(_fetch)
        with pytest.raises(ValidationError) as excinfo:
            fetch_universe_ohlcv(
                provider, ["AAA", "BBB", "CCC"], "2022-01-01", "2023-01-01"
            )
        message = str(excinfo.value)
        assert "'AAA'" in message and "'CCC'" in message and "2 of 3" in message

    def test_failures_are_reported_in_a_stable_order(self):
        """Concurrent completion order is nondeterministic; an error
        message that reorders itself between runs is not diffable and
        makes a flaky-looking report out of a deterministic failure."""

        def _fetch(symbol):
            raise ConnectionError("down")

        provider = make_provider_mock(_fetch)
        messages = []
        for _ in range(3):
            with pytest.raises(ValidationError) as excinfo:
                fetch_universe_ohlcv(
                    provider, ["CCC", "AAA", "BBB"], "2022-01-01", "2023-01-01"
                )
            messages.append(str(excinfo.value))
        assert len(set(messages)) == 1

    def test_result_order_follows_the_requested_universe(self):
        provider = make_provider_mock(make_ohlcv)
        frames = fetch_universe_ohlcv(
            provider, ["CCC", "AAA", "BBB"], "2022-01-01", "2023-01-01"
        )
        assert list(frames) == ["CCC", "AAA", "BBB"]

    def test_empty_frame_is_a_named_failure_not_an_empty_dataset(self):
        def _fetch(symbol):
            if symbol == "BBB":
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            return make_ohlcv(symbol)

        provider = make_provider_mock(_fetch)
        with pytest.raises(ValidationError, match="'BBB'"):
            fetch_universe_ohlcv(provider, ["AAA", "BBB"], "2022-01-01", "2023-01-01")

    def test_falls_back_to_sync_for_a_provider_without_the_async_method(self):
        """A duck-typed provider implementing only get_ohlcv must degrade
        to a slower fetch, not raise from inside a coroutine."""

        class SyncOnlyProvider:
            def get_ohlcv(self, symbol, start, end, interval="1d"):
                return make_ohlcv(symbol)

        frames = fetch_universe_ohlcv(
            SyncOnlyProvider(), ["AAA", "BBB"], "2022-01-01", "2023-01-01"
        )
        assert set(frames) == {"AAA", "BBB"}

    def test_works_inside_a_running_event_loop(self):
        """asyncio.run raises outright if a loop is already running, which
        would make build_dataset unusable from a notebook or an async agent
        runtime."""
        provider = make_provider_mock(make_ohlcv)

        async def _call():
            return fetch_universe_ohlcv(
                provider, ["AAA", "BBB"], "2022-01-01", "2023-01-01"
            )

        frames = asyncio.run(_call())
        assert set(frames) == {"AAA", "BBB"}

    def test_sequential_path_also_reports_every_failure(self):
        """The two paths must fail identically — an error that depends on
        whether a loop happened to be running is not reproducible."""

        class SyncOnlyProvider:
            def get_ohlcv(self, symbol, start, end, interval="1d"):
                raise ConnectionError(f"{symbol} down")

        with pytest.raises(ValidationError) as excinfo:
            fetch_universe_ohlcv(
                SyncOnlyProvider(), ["AAA", "BBB"], "2022-01-01", "2023-01-01"
            )
        message = str(excinfo.value)
        assert "'AAA'" in message and "'BBB'" in message and "2 of 2" in message

    @staticmethod
    def _peak_in_flight(n_symbols: int) -> int:
        in_flight = 0
        peak = 0

        async def _fetch_async(symbol, start, end, interval="1d"):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return make_ohlcv(symbol)

        provider = MagicMock()
        provider.get_ohlcv_async = _fetch_async
        fetch_universe_ohlcv(
            provider, [f"S{i}" for i in range(n_symbols)], "2022-01-01", "2023-01-01"
        )
        return peak

    def test_fetches_actually_overlap(self, monkeypatch):
        """The point of the change. Asserting only an upper bound would
        pass just as happily if every fetch ran one after another, which is
        the behaviour being replaced — so the lower bound is the test that
        matters."""
        monkeypatch.setenv("SQT_MODELING_FETCH_CONCURRENCY", "8")
        assert self._peak_in_flight(8) > 1

    def test_concurrency_limit_is_respected(self, monkeypatch):
        monkeypatch.setenv("SQT_MODELING_FETCH_CONCURRENCY", "2")
        peak = self._peak_in_flight(8)
        assert peak == 2, "the semaphore, not chance, must be the binding constraint"

    def test_malformed_concurrency_env_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("SQT_MODELING_FETCH_CONCURRENCY", "not-a-number")
        provider = make_provider_mock(make_ohlcv)
        frames = fetch_universe_ohlcv(
            provider, ["AAA", "BBB"], "2022-01-01", "2023-01-01"
        )
        assert set(frames) == {"AAA", "BBB"}


# ── Provider guarantees ────────────────────────────────────────────────


class TestProviderGuaranteeWarnings:
    def test_both_guarantees_absent_produces_two_warnings(self):
        warnings = provider_guarantee_warnings(mock_metadata())
        assert len(warnings) == 2
        joined = " ".join(warnings)
        assert "point_in_time=False" in joined
        assert "survivorship_free=False" in joined

    def test_a_fully_guaranteed_provider_produces_none(self):
        warnings = provider_guarantee_warnings(
            mock_metadata(survivorship_free=True, point_in_time=True)
        )
        assert warnings == []

    def test_missing_metadata_is_not_an_error(self):
        """
        Still not an error — it must not raise or fail the build. But it is
        no longer SILENT.

        This previously asserted `== []`, which made a failed metadata
        lookup indistinguishable from a provider that guarantees everything
        (the case asserted directly above). Both produced "no warnings", so
        a transient get_metadata failure silently suppressed exactly the
        provenance caveat this function exists to surface.
        """
        warnings = provider_guarantee_warnings(None)
        assert len(warnings) == 1
        assert "could not be determined" in warnings[0]
        # And it must not be confusable with a clean bill of health.
        assert provider_guarantee_warnings(None) != provider_guarantee_warnings(
            mock_metadata(survivorship_free=True, point_in_time=True)
        )

    def test_a_provider_without_get_metadata_still_builds(self, monkeypatch):
        """Provenance is a note, not a precondition: a provider that cannot
        report metadata must not fail the build."""
        provider = make_provider_mock(make_ohlcv)
        del provider.get_metadata
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        built = build_dataset(_spec())
        assert not built["panel"].empty

    def test_survivorship_warning_reaches_the_tool_result(self, patched_multi_factory):
        result = build_model_dataset(BuildModelDatasetInput(spec=_spec()))
        assert any("survivorship_free=False" in w for w in result.warnings)


# ── Coverage diagnostics ───────────────────────────────────────────────


class TestCoverageWarnings:
    def test_full_coverage_produces_no_warning(self):
        frames = {s: make_ohlcv(s, n=400) for s in ("AAA", "BBB")}
        assert entity_coverage_warnings(frames, "2022-01-01", "2023-07-15") == []

    def test_a_late_listing_is_reported(self):
        frames = {"AAA": make_ohlcv("AAA", n=400), "BBB": make_ohlcv("BBB", n=400)}
        # CCC lists two-thirds of the way through the window.
        frames["CCC"] = frames["AAA"].iloc[300:].copy()
        (message,) = entity_coverage_warnings(frames, "2022-01-01", "2023-07-15")
        assert "CCC" in message
        assert "100/400" in message
        assert "AAA" not in message, "only the short entity should be named"

    def test_a_universe_starting_after_the_requested_window_is_reported(self):
        frames = {"AAA": make_ohlcv("AAA", n=400)}
        messages = entity_coverage_warnings(frames, "2015-01-01", "2023-07-15")
        assert any("requested start 2015-01-01" in m for m in messages)

    def test_a_universe_ending_before_the_requested_window_is_reported(self):
        frames = {"AAA": make_ohlcv("AAA", n=400)}
        messages = entity_coverage_warnings(frames, "2022-01-01", "2030-01-01")
        assert any("requested end 2030-01-01" in m for m in messages)

    def test_a_few_missing_bars_do_not_trigger_the_warning(self):
        """Isolated gaps (a halt, a local holiday) are normal; the warning
        exists for a materially short history, and one that fires on every
        real universe would be ignored."""
        frames = {"AAA": make_ohlcv("AAA", n=400), "BBB": make_ohlcv("BBB", n=400)}
        frames["BBB"] = frames["BBB"].drop(frames["BBB"].index[50:53])
        assert entity_coverage_warnings(frames, "2022-01-01", "2023-07-15") == []


class TestIntersectionWarnings:
    def _frames_with_one_short_history(self):
        frames = {"AAA": make_ohlcv("AAA", n=400), "BBB": make_ohlcv("BBB", n=400)}
        frames["CCC"] = frames["AAA"].iloc[300:].copy()
        return frames

    def _returns_panel(self, frames):
        from standard_quant_tools.modeling.dataset.alignment import build_returns_panel

        return build_returns_panel({s: f["Close"] for s, f in frames.items()})

    def test_silent_truncation_is_reported_when_pca_features_are_used(self):
        frames = self._frames_with_one_short_history()
        panel = self._returns_panel(frames)
        (message,) = intersection_warnings(frames, panel, True)
        assert "CCC" in message, "the binding constraint must be named"
        assert "25%" in message

    def test_not_reported_when_no_universe_scope_feature_is_requested(self):
        """Without a PCA feature the returns panel is computed but never
        consumed, so the intersection costs nothing and warning about it
        would be noise."""
        frames = self._frames_with_one_short_history()
        panel = self._returns_panel(frames)
        assert intersection_warnings(frames, panel, False) == []

    def test_aligned_histories_produce_no_warning(self):
        frames = {s: make_ohlcv(s, n=400) for s in ("AAA", "BBB")}
        panel = self._returns_panel(frames)
        assert intersection_warnings(frames, panel, True) == []

    def test_the_truncation_is_real_not_hypothetical(self):
        """Guards the premise of the warning: with a universe-scope feature
        requested, one short history really does cut the whole panel down —
        it is not merely a bookkeeping note."""
        frames = self._frames_with_one_short_history()
        panel = self._returns_panel(frames)
        union = pd.DatetimeIndex([])
        for frame in frames.values():
            union = union.union(pd.DatetimeIndex(frame.index))
        assert len(panel.index) < 0.3 * len(union)


# ── Warnings reach the caller and the model ────────────────────────────


class TestWarningsPropagation:
    def test_build_dataset_returns_warnings(self, patched_multi_factory):
        built = build_dataset(_spec())
        assert isinstance(built["warnings"], list)
        assert built["warnings"], "the mock provider reports neither guarantee"

    def test_tool_result_carries_them(self, patched_multi_factory):
        result = build_model_dataset(BuildModelDatasetInput(spec=_spec()))
        assert result.warnings == build_dataset(_spec())["warnings"]

    def test_warnings_are_persisted_with_the_dataset(self, patched_multi_factory):
        from standard_quant_tools.modeling import artifacts as _artifacts

        result = build_model_dataset(BuildModelDatasetInput(spec=_spec()))
        meta = _artifacts.load_json(
            str(_artifacts.run_dir(result.dataset_id) / "dataset_meta.json")
        )
        assert meta["warnings"] == result.warnings
        assert meta["provider"] == "yfinance"
        assert meta["interval"] == "1d"

    def test_warnings_travel_onto_the_trained_model(self, patched_multi_factory):
        """The caveats must sit next to the OOS metrics they qualify —
        inspect_model(view='lineage') is where someone decides months later
        whether to trust the numbers."""
        from standard_quant_tools.modeling.agent.models import (
            InspectModelInput,
            RunModelExperimentInput,
        )
        from standard_quant_tools.modeling.agent.tools import (
            inspect_model,
            run_model_experiment,
        )
        from standard_quant_tools.modeling.specs import (
            EstimatorSpec,
            ModelSpec,
            ValidationSpec,
        )

        dataset = build_model_dataset(BuildModelDatasetInput(spec=_spec()))
        experiment = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=dataset.dataset_id,
                spec=ModelSpec(
                    task="regression",
                    estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                    validation=ValidationSpec(
                        train_window=150, test_window=30, embargo=5
                    ),
                ),
            )
        )
        lineage = inspect_model(
            InspectModelInput(model_id=experiment.model_id, view="lineage")
        )
        assert lineage.data["dataset_warnings"] == dataset.warnings

    def test_a_clean_provider_and_universe_produce_no_warnings(self, monkeypatch):
        """The warnings must be informative rather than constant: a
        provider making both guarantees, aligned histories and a daily
        interval produce an empty list, so a non-empty one means
        something."""
        provider = make_provider_mock(
            lambda symbol: make_ohlcv(symbol, n=400),
            survivorship_free=True,
            point_in_time=True,
        )
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        built = build_dataset(_spec(start="2022-01-01", end="2023-07-15"))
        assert built["warnings"] == []


class TestNoBehaviourChangeForExistingSpecs:
    def test_panel_is_identical_to_the_pre_change_defaults(self, patched_multi_factory):
        """provider/interval have defaults, so a spec written before they
        existed must build exactly the same panel."""
        built = build_dataset(_spec())
        panel = built["panel"]
        assert list(panel.columns) == [
            "date",
            "entity",
            "technical.rsi",
            "target",
            "label_end_date",
        ]
        assert np.isfinite(panel["technical.rsi"].to_numpy()).all()
        assert sorted(panel["entity"].unique()) == ["AAA", "BBB", "CCC"]
