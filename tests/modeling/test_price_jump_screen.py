"""
A dataset build screens its bars for split-sized moves, and adjusts the
splits a caller declares.

Databento serves unadjusted bars and says so (`adjusted=False`), and
nothing under modeling read the flag: a live 2022-2026 panel of 30 names
spanned six splits, 25 labels and 1,131 feature rows read one as a -67% to
-95% return, and about half of a ridge model's rank IC came from those
rows (the CHANGELOG entry of 2026-10-04). These tests hold what the build
says about such a move, that saying it never changes the panel, that a
split declared in `DatasetSpec.corporate_actions` is adjusted the same way
at build and at scoring, and that a spec without the new field hashes as
it did before the field existed.
"""

import inspect
import json
import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools import _split_screen, constants
from standard_quant_tools.backtest import screens
from standard_quant_tools.data import quality
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.metadata import DataSetMetadata
from standard_quant_tools.data.quality import (
    detect_split_like_moves,
    split_ratio_label,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    BuildModelDatasetResult,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.tools import (
    build_model_dataset,
    run_model_experiment,
)
from standard_quant_tools.modeling.dataset.builder import (
    build_dataset,
    dataset_spec_hash,
)
from standard_quant_tools.modeling.dataset.coverage import (
    provider_guarantee_warnings,
)
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

N_BARS = 750
SPLIT_ENTITY = "BBB"
SPLIT_DATE = "2023-09-01"
UNIVERSE = ["AAA", "BBB", "CCC"]


# ── Bars ────────────────────────────────────────────────────────────────


def _adjusted(symbol: str, n: int = N_BARS) -> pd.DataFrame:
    """Smooth synthetic bars (1.2% daily volatility, so no move nears 35%)
    on business days from 2022-01-03, seeded by the symbol."""
    seed = sum(ord(c) * 31**i for i, c in enumerate(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.001, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.004, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.004, n))),
            "Close": close,
            "Volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=pd.bdate_range("2022-01-03", periods=n),
    )


def _unadjusted(frame: pd.DataFrame, ex_date: str, ratio: float) -> pd.DataFrame:
    """The bars an unadjusted feed serves for a split: every bar before the
    ex-date at the pre-split price, ratio times the adjusted one."""
    raw = frame.copy()
    before = raw.index < pd.Timestamp(ex_date)
    for column in ("Open", "High", "Low", "Close"):
        raw.loc[before, column] = raw.loc[before, column] * ratio
    raw.loc[before, "Volume"] = raw.loc[before, "Volume"] / ratio
    return raw


def _hand_adjusted(raw: pd.DataFrame, ex_date: str, ratio: float) -> pd.DataFrame:
    """The reference a declared split must reproduce: prices before the
    ex-date divided by the ratio, volume multiplied by it."""
    out = raw.copy()
    before = out.index < pd.Timestamp(ex_date)
    for column in ("Open", "High", "Low", "Close"):
        out.loc[before, column] = out.loc[before, column] / ratio
    out.loc[before, "Volume"] = out.loc[before, "Volume"] * ratio
    return out


def _frames(split=(SPLIT_ENTITY, SPLIT_DATE, 10.0), benchmark_split=None):
    frames = {s: _adjusted(s) for s in UNIVERSE + ["SPY"]}
    if split is not None:
        entity, ex, ratio = split
        frames[entity] = _unadjusted(frames[entity], ex, ratio)
    if benchmark_split is not None:
        frames["SPY"] = _unadjusted(frames["SPY"], *benchmark_split)
    return frames


def _provider(frames, adjusted=False, attrs=None):
    """A provider serving `frames` sliced to the requested window. `adjusted`
    is what get_metadata reports (None: get_metadata fails); `attrs` is
    stamped on every frame as attrs['adjusted'] when given."""

    def fetch(symbol, start, end, interval="1d"):
        frame = frames[symbol]
        zone = frame.index.tz
        out = frame[
            (frame.index >= pd.Timestamp(start, tz=zone))
            & (frame.index <= pd.Timestamp(end, tz=zone))
        ].copy()
        if attrs is not None:
            out.attrs["adjusted"] = attrs
        return out

    provider = MagicMock()
    provider.get_ohlcv.side_effect = fetch
    provider.get_ohlcv_async = AsyncMock(side_effect=fetch)
    if adjusted is None:
        provider.get_metadata.side_effect = RuntimeError("metadata unavailable")
    else:
        provider.get_metadata.side_effect = lambda s, interval="1d": DataSetMetadata(
            provider="databento",
            adjusted=adjusted,
            survivorship_free=True,
            point_in_time=True,
            frequency=interval,
            timezone="UTC",
        )
    return provider


@pytest.fixture
def serve(monkeypatch):
    """Route DataFactory to a provider built by `_provider`."""

    def _serve(frames, **kwargs):
        provider = _provider(frames, **kwargs)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)
        return provider

    return _serve


def _spec(**overrides) -> DatasetSpec:
    kwargs = dict(
        universe=UNIVERSE,
        start="2022-01-03",
        end="2023-12-29",
        features=[FeatureSpec(id="market.momentum", params={"lookback": 20})],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
        provider="databento",
    )
    kwargs.update(overrides)
    return DatasetSpec(**kwargs)


def _jump_warning(warnings):
    hits = [w for w in warnings if w.startswith("PRICE JUMPS")]
    assert len(hits) <= 1
    return hits[0] if hits else None


# ── The detector ────────────────────────────────────────────────────────


class TestTheDetector:
    def test_the_backtest_and_the_build_read_one_threshold(self):
        """One object, not two equal numbers: the backtest's screen and the
        build's cannot drift apart."""
        assert screens.SPLIT_SCREEN_THRESHOLD is constants.SPLIT_SCREEN_THRESHOLD
        assert quality.SPLIT_SCREEN_THRESHOLD is constants.SPLIT_SCREEN_THRESHOLD
        default = inspect.signature(detect_split_like_moves).parameters["threshold"]
        assert default.default is constants.SPLIT_SCREEN_THRESHOLD
        assert constants.SPLIT_SCREEN_THRESHOLD == 0.35

    def test_the_backtest_and_the_build_run_one_rule(self):
        """Which bars are named is one function, and the tolerance and the
        ratios it names below the threshold are one object each, read by
        both screens (the CHANGELOG entry of 2026-10-04)."""
        assert screens.screen_moves is _split_screen.screen_moves
        assert quality.screen_moves is _split_screen.screen_moves
        assert quality.SPLIT_RATIO_TOLERANCE is _split_screen.SPLIT_RATIO_TOLERANCE
        assert (
            screens.RATIOS_NAMED_BELOW_THRESHOLD
            is quality.RATIOS_NAMED_BELOW_THRESHOLD
            is _split_screen.RATIOS_NAMED_BELOW_THRESHOLD
        )
        assert _split_screen.RATIOS_NAMED_BELOW_THRESHOLD == (1.5,)
        assert _split_screen.SPLIT_RATIO_TOLERANCE == 0.10

    @pytest.mark.parametrize(
        "ratio,label", [(2.0, "2:1"), (3.0, "3:1"), (10.0, "10:1"), (20.0, "20:1")]
    )
    def test_a_split_is_named_after_its_ratio(self, ratio, label):
        """A split written into smooth bars is listed on its ex-date with its
        ratio, in the new-shares-per-old-share unit corporate_actions takes."""
        close = _unadjusted(_adjusted("XYZ"), "2023-01-03", ratio)["Close"]
        [move] = detect_split_like_moves(close)
        assert move["date"] == "2023-01-03"
        assert move["split_ratio"] == ratio
        assert move["ratio_error"] < 0.05
        assert math.isclose(move["close_move"], 1 / ratio - 1, abs_tol=0.05)
        assert split_ratio_label(move["split_ratio"]) == label

    def test_a_reverse_split_is_named_the_other_way_up(self):
        """A 1:10 reverse split is a +900% bar, named with ratio 0.1."""
        close = _unadjusted(_adjusted("XYZ"), "2023-01-03", 0.1)["Close"]
        [move] = detect_split_like_moves(close)
        assert move["split_ratio"] == pytest.approx(0.1)
        assert move["close_move"] > 5
        assert split_ratio_label(move["split_ratio"]) == "1:10"

    def test_a_three_for_two_split_below_the_threshold_is_named(self):
        """-33% does not cross 35%; it is named because it is the size of a
        3:2 split (the CHANGELOG entry of 2026-10-04 closes the gap the
        entry before it left)."""
        close = _unadjusted(_adjusted("XYZ"), "2023-01-03", 1.5)["Close"]
        [move] = detect_split_like_moves(close)
        assert move["date"] == "2023-01-03"
        assert move["split_ratio"] == 1.5 and move["ratio_error"] < 0.05
        assert -0.35 < move["close_move"] < -0.30
        assert detect_split_like_moves(close, threshold=0.30) == [move]

    def test_a_move_between_ratios_is_listed_but_not_named(self):
        """-42%: the price ratio 1.72 is 0.14 from 3:2 and 0.15 from 2:1 on a
        log scale, both beyond the 0.10 that names a ratio."""
        close = _adjusted("XYZ")["Close"].copy()
        close.iloc[300:] *= 0.58
        [move] = detect_split_like_moves(close)
        assert move["split_ratio"] is None
        assert move["ratio_error"] > quality.SPLIT_RATIO_TOLERANCE

    def test_quiet_bars_and_a_frame_argument(self):
        """Bars at 1.2% daily volatility hold no 35% move; a frame is read by
        its Close, and one bar has no move at all."""
        frame = _adjusted("XYZ")
        assert detect_split_like_moves(frame) == []
        assert detect_split_like_moves(frame["Close"].iloc[:1]) == []

    def test_a_threshold_that_is_not_a_size_is_refused(self):
        """A threshold of zero would list every bar; it is refused by name."""
        with pytest.raises(ValidationError, match="positive"):
            detect_split_like_moves(_adjusted("XYZ")["Close"], threshold=0)


# ── The build's screen ──────────────────────────────────────────────────


class TestTheBuildScreen:
    def test_a_split_is_named_with_the_labels_and_rows_that_read_it(self, serve):
        """A 10:1 split in BBB under a 5-bar label and a 20-bar momentum:
        the 5 labels dated before it that end on or after it, and the 20
        rows from it on, read it."""
        serve(_frames())
        built = build_dataset(_spec())
        [jump] = built["price_jumps"]
        assert jump["entity"] == SPLIT_ENTITY and jump["role"] == "entity"
        assert jump["date"] == SPLIT_DATE
        assert jump["split_ratio"] == 10.0
        assert jump["labels"] == 5
        assert jump["feature_rows"] == 20
        assert jump["reach_bars"] == 20
        assert built["bars_adjusted"] is False

        warning = _jump_warning(built["warnings"])
        assert warning.startswith(
            "PRICE JUMPS: 1 close-to-close move beyond 35% in the bars this "
            f"dataset was built from: BBB {SPLIT_DATE} (-90%, within "
        )
        assert "of a 10:1 split)" in warning
        assert "The provider reports adjusted=False" in warning
        n_rows = len(built["panel"])
        assert f"In this panel 5 of {n_rows:,} targets span one" in warning
        assert "20 rows (" in warning
        assert "DatasetSpec.corporate_actions" in warning
        assert "dataset_meta.json lists each one as price_jumps." in warning

    def test_the_rows_named_are_the_rows_that_read_it(self, serve):
        """The recorded rows are BBB's 20 panel rows from the split on."""
        serve(_frames())
        built = build_dataset(_spec())
        panel = built["panel"]
        rows = panel.loc[built["price_jump_rows"][0]]
        assert set(rows["entity"]) == {SPLIT_ENTITY}
        assert rows["date"].min() == pd.Timestamp(SPLIT_DATE)
        bars = _frames()[SPLIT_ENTITY].index
        at = bars.get_loc(pd.Timestamp(SPLIT_DATE))
        assert rows["date"].max() == bars[at + 19]

    def test_the_screen_never_changes_the_panel(self, serve):
        """The same bars under adjusted=False and adjusted=True build the
        same panel and the same hashes; only the words differ."""
        serve(_frames(), adjusted=False)
        unadjusted = build_dataset(_spec())
        serve(_frames(), adjusted=True)
        adjusted = build_dataset(_spec())
        assert unadjusted["data_hash"] == adjusted["data_hash"]
        assert unadjusted["spec_hash"] == adjusted["spec_hash"]
        pd.testing.assert_frame_equal(unadjusted["panel"], adjusted["panel"])

    def test_adjusted_true_says_genuine_move_or_bad_print(self, serve):
        """On bars the provider says are adjusted, a move is not a split, and
        the remedy of declaring one is not offered."""
        serve(_frames(), adjusted=True)
        warning = _jump_warning(build_dataset(_spec())["warnings"])
        assert "adjusted=True, so each is a genuine move or a bad print" in warning
        assert "corporate_actions" not in warning

    def test_unknown_says_unknown(self, serve):
        """No metadata and no stamp on the frames: not known, and said so."""
        serve(_frames(), adjusted=None)
        built = build_dataset(_spec())
        assert built["bars_adjusted"] is None
        assert "Whether these bars are split-adjusted is not known here" in (
            _jump_warning(built["warnings"])
        )

    def test_the_frames_stamp_answers_when_the_metadata_cannot(self, serve):
        """Carbon's bridge can lose the metadata; frames that all say
        adjusted=False still settle the flag."""
        serve(_frames(), adjusted=None, attrs=False)
        built = build_dataset(_spec())
        assert built["bars_adjusted"] is False
        assert "adjusted=False" in _jump_warning(built["warnings"])

    def test_a_universe_feature_spreads_one_split_to_every_entity(self, serve):
        """factors.pca_loading reads every entity's returns, so BBB's split
        reaches AAA's and CCC's rows too."""
        serve(_frames())
        spec = _spec(
            features=[
                FeatureSpec(id="market.momentum", params={"lookback": 20}),
                FeatureSpec(id="factors.pca_loading"),
            ],
            end="2024-11-15",
        )
        built = build_dataset(spec)
        [jump] = built["price_jumps"]
        rows = built["panel"].loc[built["price_jump_rows"][0]]
        assert set(rows["entity"]) == set(UNIVERSE)
        assert jump["feature_rows"] > 3 * 20

    def test_a_benchmark_split_reaches_the_features_that_read_it(self, serve):
        """A split in the benchmark is read by rolling beta on every
        entity; with no feature reading the benchmark it is in no row and
        is not listed."""
        serve(_frames(split=None, benchmark_split=("2023-06-01", 4.0)))
        beta = build_dataset(
            _spec(
                features=[
                    FeatureSpec(id="market.momentum", params={"lookback": 20}),
                    FeatureSpec(id="risk.rolling_beta", params={"window": 60}),
                ]
            )
        )
        [jump] = beta["price_jumps"]
        assert jump["role"] == "benchmark" and jump["entity"] == "SPY"
        assert jump["labels"] == 0 and jump["reach_bars"] == 60
        assert jump["feature_rows"] == 3 * 60
        assert "SPY (benchmark) 2023-06-01 (-75%" in _jump_warning(beta["warnings"])

        momentum_only = build_dataset(_spec())
        assert momentum_only["price_jumps"] == []

    def test_six_are_listed_then_the_count(self, serve):
        """Nine moves: the warning names six and counts the other three, so it
        stays one readable line."""
        frames = _frames(split=None)
        for i, symbol in enumerate(UNIVERSE):
            for k in range(3):
                ex = str(
                    pd.bdate_range("2022-09-01", periods=400)[100 * k + 7 * i].date()
                )
                frames[symbol] = _unadjusted(frames[symbol], ex, 2.0)
        serve(frames)
        warning = _jump_warning(build_dataset(_spec())["warnings"])
        assert warning.startswith("PRICE JUMPS: 9 close-to-close moves beyond 35%")
        assert " and 3 more." in warning
        assert warning.count("of a 2:1 split)") == 6

    def test_quiet_bars_give_no_warning_and_an_empty_record(self, serve):
        """No move, no warning; the empty list says the bars were screened."""
        serve(_frames(split=None))
        built = build_dataset(_spec())
        assert built["price_jumps"] == []
        assert _jump_warning(built["warnings"]) is None

    def test_a_cross_sectional_target_names_the_labels_it_moves(self, serve):
        """A rank is computed across each date's entities, so the split
        moves the other two entities' labels on BBB's five spanning dates.
        On bars shaped like the live panel the rank labels that changed
        were 397, against 25 that span a split themselves."""
        serve(_frames())
        built = build_dataset(
            _spec(target=TargetSpec(type="forward_return_rank", horizon=5))
        )
        assert (
            "The target is computed across each date's entities, so the labels "
            "of the other 10 rows on those 5 dates can move too."
            in _jump_warning(built["warnings"])
        )
        serve(_frames())
        plain = build_dataset(_spec())
        assert "computed across each date's entities" not in _jump_warning(
            plain["warnings"]
        )

    def test_a_multi_horizon_panel_counts_each_label_once(self, serve):
        """horizons [5, 10]: the primary `target` repeats `target__h5`, so
        the labels are the h5 column's 5 and the h10 column's 10."""
        serve(_frames())
        built = build_dataset(_spec(target=TargetSpec(horizons=[5, 10])))
        [jump] = built["price_jumps"]
        assert jump["labels"] == 15
        n_rows = len(built["panel"])
        assert f"15 of {2 * n_rows:,} targets" in _jump_warning(built["warnings"])


class TestAThreeForTwoSplit:
    """A 3:2 split moves -33%, below the 35% threshold, and the build used
    to read it as a return without a word. It is named now, counted like
    any other move, and still only named (the CHANGELOG entry of
    2026-10-04)."""

    def test_it_is_named_with_the_labels_and_rows_that_read_it(self, serve):
        serve(_frames(split=(SPLIT_ENTITY, SPLIT_DATE, 1.5)))
        built = build_dataset(_spec())
        [jump] = built["price_jumps"]
        assert (jump["entity"], jump["date"]) == (SPLIT_ENTITY, SPLIT_DATE)
        assert jump["split_ratio"] == 1.5
        assert (jump["labels"], jump["feature_rows"]) == (5, 20)
        warning = _jump_warning(built["warnings"])
        assert warning.startswith(
            "PRICE JUMPS: 1 close-to-close move beyond 35%, or falls of 26% to "
            "35% near a 3:2 split, in the bars this dataset was built from: "
            f"BBB {SPLIT_DATE} (-3"
        )
        assert "of a 3:2 split)" in warning

    def test_null_a_move_beyond_the_threshold_keeps_the_old_header(self, serve):
        """The added clause appears only when a listed move is below 35%."""
        serve(_frames())
        warning = _jump_warning(build_dataset(_spec())["warnings"])
        assert warning.startswith("PRICE JUMPS: 1 close-to-close move beyond 35% in")

    def test_naming_it_changes_nothing_in_the_panel(self, serve):
        """Warnings only: the same 3:2 bars build the same panel and hashes
        whatever the provider says about adjustment."""
        frames = _frames(split=(SPLIT_ENTITY, SPLIT_DATE, 1.5))
        serve(frames, adjusted=False)
        unadjusted = build_dataset(_spec())
        serve(frames, adjusted=True)
        adjusted = build_dataset(_spec())
        assert unadjusted["data_hash"] == adjusted["data_hash"]
        pd.testing.assert_frame_equal(unadjusted["panel"], adjusted["panel"])
        assert len(unadjusted["price_jumps"]) == len(adjusted["price_jumps"]) == 1

    def test_declared_it_is_adjusted_and_no_longer_named(self, serve):
        serve(_frames(split=(SPLIT_ENTITY, SPLIT_DATE, 1.5)))
        built = build_dataset(_spec(corporate_actions=_declared(ratio=1.5)))
        assert built["price_jumps"] == []
        assert not any(
            w.startswith("DECLARED SPLIT NOT SEEN") for w in built["warnings"]
        )

    def test_declared_a_session_late_both_moves_are_named(self, serve):
        """The real -33% stays in the bars, and dividing the bars before the
        late date makes a +50% after it: both are named, the first only
        because of the 3:2 rule."""
        serve(_frames(split=(SPLIT_ENTITY, SPLIT_DATE, 1.5)))
        late = "2023-09-05"
        built = build_dataset(
            _spec(corporate_actions=_declared(ex_date=late, ratio=1.5))
        )
        assert [j["date"] for j in built["price_jumps"]] == [SPLIT_DATE, late]
        assert [j["split_ratio"] for j in built["price_jumps"]] == [
            1.5,
            pytest.approx(1 / 1.5),
        ]
        assert any(w.startswith("DECLARED SPLIT NOT SEEN") for w in built["warnings"])


class TestTheProviderLine:
    def test_unadjusted_bars_are_said_to_give_price_returns(self):
        """Databento's adjusted=False becomes one line: labels are price
        returns, a dividend is a price drop, a split a price fall."""
        metadata = DataSetMetadata(
            provider="databento",
            adjusted=False,
            survivorship_free=True,
            point_in_time=True,
            frequency="1d",
            timezone="UTC",
        )
        [line] = provider_guarantee_warnings(metadata)
        assert line.startswith(
            "provider 'databento' serves unadjusted bars (adjusted=False): a "
            "dividend is a price drop on its ex-date, so return targets built "
            "here are price returns, not the total returns"
        )
        assert "nothing here adjusts a dividend" in line

    def test_adjusting_and_silent_providers_get_no_line(self):
        """Only a provider that says False gets the line: True does not, and a
        mock's non-bool attribute is silence."""
        metadata = DataSetMetadata(
            provider="yfinance",
            adjusted=True,
            survivorship_free=True,
            point_in_time=True,
            frequency="1d",
            timezone="UTC",
        )
        assert provider_guarantee_warnings(metadata) == []
        silent = MagicMock(point_in_time=True, survivorship_free=True)
        assert provider_guarantee_warnings(silent) == []


# ── What a dataset records ──────────────────────────────────────────────


class TestTheRecord:
    def test_dataset_meta_records_the_moves_and_the_flag(self, serve):
        """dataset_meta.json carries price_jumps, bars_adjusted and the
        declared table's record on every build."""
        serve(_frames())
        result = build_model_dataset(BuildModelDatasetInput(spec=_spec()))
        meta = json.loads(
            (_artifacts.run_dir(result.dataset_id) / "dataset_meta.json").read_text()
        )
        assert meta["bars_adjusted"] is False
        [jump] = meta["price_jumps"]
        assert jump["entity"] == SPLIT_ENTITY and jump["labels"] == 5
        assert meta["corporate_actions_applied"] == []
        assert any(w.startswith("PRICE JUMPS") for w in result.warnings)

    def test_the_tool_result_keeps_its_fields(self):
        """The findings go to dataset_meta.json and the warnings; the tool
        result's fields are the ones they were."""
        assert sorted(BuildModelDatasetResult.model_fields) == [
            "data_sources",
            "dataset_id",
            "drop_attribution",
            "entities",
            "feature_ids",
            "rows",
            "target_id",
            "warnings",
        ]

    def _model_spec(self):
        return ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            random_seed=1,
        )

    def test_a_databento_dataset_from_before_the_screen_gets_a_note(self, serve):
        """No `price_jumps` key: nothing on record says whether a split is
        inside, and the experiment's manifest says so."""
        serve(_frames(split=None))
        built = build_model_dataset(BuildModelDatasetInput(spec=_spec()))
        meta_path = _artifacts.run_dir(built.dataset_id) / "dataset_meta.json"
        meta = json.loads(meta_path.read_text())
        for key in ("price_jumps", "bars_adjusted", "corporate_actions_applied"):
            meta.pop(key)
        meta_path.write_text(json.dumps(meta))

        result = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=built.dataset_id, spec=self._model_spec()
            )
        )
        notes = [
            w
            for w in load_manifest(result.model_id).dataset_warnings
            if "before builds screened for splits" in w
        ]
        assert notes == [
            f"NOTE: dataset {built.dataset_id!r} was built from Databento's "
            "unadjusted bars before builds screened for splits, so nothing on "
            "record says whether a split falls inside it; a split in these bars "
            "is a price fall that every label and feature spanning it reads as a "
            "return. Rebuild it to have any split named."
        ]

    def test_a_screened_dataset_gets_no_note(self, serve):
        """A dataset built with the screen records price_jumps, so it gets no
        note."""
        serve(_frames(split=None))
        built = build_model_dataset(BuildModelDatasetInput(spec=_spec()))
        result = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=built.dataset_id, spec=self._model_spec()
            )
        )
        assert not any(
            "before builds screened for splits" in w
            for w in load_manifest(result.model_id).dataset_warnings
        )


# ── The declared table ──────────────────────────────────────────────────


def _declared(entity=SPLIT_ENTITY, ex_date=SPLIT_DATE, ratio=10.0):
    return [{"entity": entity, "ex_date": ex_date, "split_ratio": ratio}]


class TestDeclaredSplits:
    def test_an_adjusted_build_equals_a_hand_adjusted_reference(self, serve):
        """Declared on raw bars, the panel is bit for bit the panel built
        from bars adjusted by hand: prices before the ex-date divided by
        10, volume multiplied by 10."""
        serve(_frames())
        declared = build_dataset(_spec(corporate_actions=_declared()))

        reference = _frames()
        reference[SPLIT_ENTITY] = _hand_adjusted(
            reference[SPLIT_ENTITY], SPLIT_DATE, 10.0
        )
        serve(reference)
        by_hand = build_dataset(_spec())

        pd.testing.assert_frame_equal(declared["panel"], by_hand["panel"])
        assert declared["data_hash"] == by_hand["data_hash"]
        assert declared["spec_hash"] != by_hand["spec_hash"]

    def test_an_adjusted_split_is_not_screened(self, serve):
        """A correctly declared split leaves no move for the screen, and the
        note says what was adjusted and that dividends were not."""
        serve(_frames())
        built = build_dataset(_spec(corporate_actions=_declared()))
        assert built["price_jumps"] == []
        assert _jump_warning(built["warnings"]) is None
        [record] = built["corporate_actions_applied"]
        assert record["status"] == "applied" and record["matches_raw_move"] is True
        [note] = [w for w in built["warnings"] if w.startswith("DECLARED SPLITS")]
        assert note.startswith(
            "DECLARED SPLITS: 1 split from DatasetSpec.corporate_actions adjusted "
            f"before any feature or label was computed: BBB before {SPLIT_DATE} "
            "(10:1; the raw close moved -90% into"
        )
        assert "dividends are not adjusted" in note

    def test_a_wrong_date_warns_and_the_screen_names_what_it_created(self, serve):
        """One session late: the bar declared does not move by the ratio, the
        real split is still in the bars, and a +900% bar now follows it."""
        serve(_frames())
        late = "2023-09-05"
        built = build_dataset(_spec(corporate_actions=_declared(ex_date=late)))
        [not_seen] = [
            w for w in built["warnings"] if w.startswith("DECLARED SPLIT NOT SEEN")
        ]
        assert not_seen.startswith(f"DECLARED SPLIT NOT SEEN: BBB 10:1 on {late}")
        assert "not the -90% a 10:1 split gives" in not_seen
        assert "the date or the ratio may be wrong" in not_seen
        dates = [j["date"] for j in built["price_jumps"]]
        assert dates == [SPLIT_DATE, late]
        assert (
            "These are the moves left after adjusting the 1 split declared in "
            "DatasetSpec.corporate_actions." in _jump_warning(built["warnings"])
        )

    def test_a_benchmark_that_is_an_entity_is_adjusted_with_it(self, serve):
        """SPY in the universe and as the benchmark: its declared split is
        taken out of both, so rolling beta reads no split either."""
        serve(_frames(split=None, benchmark_split=("2023-06-01", 4.0)))
        built = build_dataset(
            _spec(
                universe=["AAA", "BBB", "SPY"],
                features=[FeatureSpec(id="risk.rolling_beta", params={"window": 60})],
                corporate_actions=_declared(
                    entity="SPY", ex_date="2023-06-01", ratio=4.0
                ),
            )
        )
        assert built["price_jumps"] == []
        spy = built["panel"][built["panel"]["entity"] == "SPY"]
        assert np.allclose(spy["risk.rolling_beta"], 1.0)

    def test_a_split_outside_the_bars_changes_nothing_and_says_so(self, serve):
        """A split before the first bar has no earlier bar to adjust: the panel
        is unchanged and the warning says so."""
        serve(_frames(split=None))
        plain = build_dataset(_spec())
        built = build_dataset(_spec(corporate_actions=_declared(ex_date="2021-06-01")))
        assert built["data_hash"] == plain["data_hash"]
        assert any(
            w.startswith("DECLARED SPLIT OUTSIDE THE BARS: BBB 10:1 on 2021-06-01")
            for w in built["warnings"]
        )

    def test_a_declared_split_on_adjusted_bars_is_refused_before_the_fetch(self, serve):
        """Adjusting bars the provider already adjusted would divide twice;
        get_metadata says so before any bar is fetched."""
        provider = serve(_frames(), adjusted=True)
        with pytest.raises(ValidationError) as caught:
            build_dataset(_spec(corporate_actions=_declared()))
        message = str(caught.value)
        assert f"BBB on {SPLIT_DATE} (10:1)" in message
        assert "adjusted=True (get_metadata)" in message
        provider.get_ohlcv_async.assert_not_called()

    def test_a_frame_stamp_of_adjusted_is_refused_too(self, serve):
        """With no metadata, frames that all say adjusted=True refuse a
        declared split after the fetch."""
        serve(_frames(), adjusted=None, attrs=True)
        with pytest.raises(ValidationError, match=f"BBB on {SPLIT_DATE}"):
            build_dataset(_spec(corporate_actions=_declared()))


class TestTheDeclaredTableModel:
    def test_the_entity_must_be_in_the_universe(self):
        """A split for an entity the dataset does not hold adjusts nothing; it
        is refused at the spec."""
        with pytest.raises(PydanticValidationError, match="not in the universe"):
            _spec(corporate_actions=_declared(entity="ZZZ"))

    @pytest.mark.parametrize("ratio", [1.0, 0.0, -2.0, float("inf")])
    def test_the_ratio_must_be_a_split(self, ratio):
        """A ratio of 1 adjusts nothing, and zero, a negative or an infinity is
        no ratio."""
        with pytest.raises(PydanticValidationError):
            _spec(corporate_actions=_declared(ratio=ratio))

    @pytest.mark.parametrize("ex_date", ["not a date", "", "2023-09-01 10:30"])
    def test_the_ex_date_must_be_a_date(self, ex_date):
        """An ex-date is a session date: unparseable, empty and time-of-day
        values are refused."""
        with pytest.raises(PydanticValidationError):
            _spec(corporate_actions=_declared(ex_date=ex_date))

    def test_one_split_per_entity_and_date(self):
        """Two entries for one entity and ex-date would compound silently; one
        is required."""
        with pytest.raises(PydanticValidationError, match="twice"):
            _spec(corporate_actions=_declared() + _declared(ratio=2.0))


class TestTheSpecHash:
    """A spec without corporate_actions hashes exactly as it did before the
    field existed, so every persisted dataset keeps verifying."""

    def test_a_spec_without_it_hashes_as_before(self):
        """Pinned: version-2 hashes recorded before the field existed (the
        second is the live 30-name Databento dataset's)."""
        small = DatasetSpec(
            universe=["AAA", "BBB", "CCC"],
            start="2022-01-01",
            end="2023-12-31",
            features=[
                FeatureSpec(id="technical.rsi"),
                FeatureSpec(id="market.momentum"),
            ],
            target=TargetSpec(horizon=5),
            benchmark="SPY",
        )
        assert dataset_spec_hash(small) == (
            "c2ee7b06e76c725ec205b83f7583a80f7cf8449f47c61bd9e98be496d175e523"
        )
        live = DatasetSpec(
            universe=[
                "AAPL",
                "MSFT",
                "NVDA",
                "AMZN",
                "GOOGL",
                "META",
                "AVGO",
                "TSLA",
                "JPM",
                "V",
                "UNH",
                "XOM",
                "JNJ",
                "WMT",
                "PG",
                "MA",
                "HD",
                "CVX",
                "MRK",
                "ABBV",
                "KO",
                "PEP",
                "COST",
                "ADBE",
                "CRM",
                "BAC",
                "TMO",
                "MCD",
                "CSCO",
                "ACN",
            ],
            start="2022-01-03",
            end="2026-09-30",
            features=[
                FeatureSpec(id="technical.rsi", params={"period": 14}, alias="rsi_14"),
                FeatureSpec(
                    id="market.momentum", params={"lookback": 20}, alias="mom_20"
                ),
                FeatureSpec(
                    id="market.momentum", params={"lookback": 126}, alias="mom_126"
                ),
                FeatureSpec(
                    id="risk.realized_volatility",
                    params={"period": 20},
                    alias="rvol_20",
                ),
                FeatureSpec(
                    id="risk.rolling_beta", params={"window": 60}, alias="beta_60"
                ),
                FeatureSpec(
                    id="volume.volume_surprise",
                    params={"period": 20},
                    alias="volsurp_20",
                ),
                FeatureSpec(
                    id="risk.bollinger_pct_b", params={"period": 20}, alias="pctb_20"
                ),
                FeatureSpec(id="technical.macd_histogram", alias="macdh"),
            ],
            target=TargetSpec(type="forward_return_rank", horizon=5),
            provider="databento",
        )
        assert dataset_spec_hash(live) == (
            "4b2faa1a597d06e71a82e3bcb34c195e30973344c52ca335ee8b2d276377a3b1"
        )

    def test_an_empty_table_is_the_same_spec(self):
        """An explicit empty table is the default, and the default is excluded
        from the hash."""
        assert dataset_spec_hash(_spec(corporate_actions=[])) == dataset_spec_hash(
            _spec()
        )

    def test_a_declared_split_changes_it_and_order_does_not(self):
        """A declared split is part of the dataset's identity; its order and
        the spelling of its date are not."""
        two = [
            {"entity": "BBB", "ex_date": SPLIT_DATE, "split_ratio": 10.0},
            {"entity": "AAA", "ex_date": "2023-03-01", "split_ratio": 2.0},
        ]
        assert dataset_spec_hash(_spec(corporate_actions=two)) != dataset_spec_hash(
            _spec()
        )
        assert dataset_spec_hash(_spec(corporate_actions=two)) == dataset_spec_hash(
            _spec(corporate_actions=list(reversed(two)))
        )
        spelled = [dict(two[0], ex_date="2023-09-01T00:00:00"), two[1]]
        assert dataset_spec_hash(_spec(corporate_actions=spelled)) == (
            dataset_spec_hash(_spec(corporate_actions=two))
        )


# ── Scoring ─────────────────────────────────────────────────────────────


def _train(spec: DatasetSpec) -> str:
    built = build_model_dataset(BuildModelDatasetInput(spec=spec))
    result = run_model_experiment(
        RunModelExperimentInput(
            dataset_id=built.dataset_id,
            spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                random_seed=1,
            ),
        )
    )
    return result.model_id


#: Momentum over 126 bars: a split on 2023-09-01 is inside what the row
#: scored on 2024-01-31 reads (about 108 bars later).
_LONG = [
    FeatureSpec(id="market.momentum", params={"lookback": 20}, alias="mom_20"),
    FeatureSpec(id="market.momentum", params={"lookback": 126}, alias="mom_126"),
]
_AS_OF = "2024-01-31"


class TestScoring:
    def test_a_split_the_scored_row_reads_is_named(self, serve):
        """Trained before the split; scored after it on raw bars."""
        serve(_frames())
        model_id = _train(_spec(features=_LONG, end="2023-06-30"))
        result = score_model(model_id=model_id, as_of=_AS_OF, universe=UNIVERSE)
        hits = [w for w in result["warnings"] if w.startswith("BBB:")]
        assert len(hits) == 1
        assert hits[0].startswith(
            f"BBB: a -90% close-to-close move on {SPLIT_DATE} (within "
        )
        assert (
            "of a 10:1 split) is inside the 126 bars this score's features read, "
            "and the provider reports adjusted=False, so the score reads the "
            "split as a return." in hits[0]
        )

    def test_a_split_beyond_the_scored_rows_reach_is_not(self, serve):
        """Momentum over 20 bars cannot reach back 108 bars."""
        serve(_frames())
        model_id = _train(_spec(end="2023-06-30"))
        result = score_model(model_id=model_id, as_of=_AS_OF, universe=UNIVERSE)
        assert not any(w.startswith("BBB:") for w in result["warnings"])

    def test_scoring_adjusts_as_the_training_build_did(self, serve):
        """The bundled spec declares the split, so the scoring rebuild
        adjusts the same bars: no warning, and BBB's scored momentum is
        the hand-adjusted bars' value."""
        serve(_frames())
        model_id = _train(
            _spec(features=_LONG, corporate_actions=_declared(), end="2023-12-29")
        )
        result = score_model(model_id=model_id, as_of=_AS_OF, universe=UNIVERSE)
        assert not any(w.startswith("BBB:") for w in result["warnings"])

        features = pd.read_parquet(result["features_uri"]).set_index("entity")
        close = _hand_adjusted(_frames()[SPLIT_ENTITY], SPLIT_DATE, 10.0)["Close"]
        at = close.index.get_loc(pd.Timestamp(_AS_OF))
        expected = close.iloc[at] / close.iloc[at - 126] - 1.0
        assert features.loc[SPLIT_ENTITY, "mom_126"] == pytest.approx(
            expected, rel=1e-12
        )
        raw = _frames()[SPLIT_ENTITY]["Close"]
        assert raw.iloc[at] / raw.iloc[at - 126] - 1.0 < -0.8

    def test_a_scored_subset_keeps_only_its_own_declared_splits(self, serve):
        """BBB's split is declared; scoring AAA alone must not be refused
        for naming an entity outside the scored universe."""
        serve(_frames())
        model_id = _train(
            _spec(features=_LONG, corporate_actions=_declared(), end="2023-12-29")
        )
        result = score_model(model_id=model_id, as_of=_AS_OF, universe=["AAA"])
        assert result["n_entities"] == 1

    def test_bars_stamped_in_utc_score(self, serve):
        """Databento's daily bars are UTC midnights, so the panel's dates are
        zone-aware; the staleness subtraction against a date-only `as_of`
        raised TypeError before it was taken on one clock."""
        frames = {s: f.tz_localize("UTC") for s, f in _frames().items()}
        serve(frames)
        model_id = _train(_spec(features=_LONG, end="2023-06-30"))
        result = score_model(model_id=model_id, as_of=_AS_OF, universe=UNIVERSE)
        assert result["effective_score_date"] == _AS_OF
        assert result["staleness_days"] == 0
        assert any(w.startswith("BBB: a -90%") for w in result["warnings"])


# ── The new feature ─────────────────────────────────────────────────────


class TestMacdHistogramPct:
    def test_it_is_the_histogram_over_close(self):
        """The new feature is the existing histogram divided by Close, nothing
        else."""
        from standard_quant_tools.indicators.trend import macd
        from standard_quant_tools.modeling.features.base import FeatureContext
        from standard_quant_tools.modeling.features.registry import get_feature

        ohlcv = _adjusted("XYZ")
        result = get_feature("technical.macd_histogram_pct").fn(
            ohlcv, FeatureContext(), fast=12, slow=26, signal=9
        )
        expected = macd(ohlcv["Close"], fast=12, slow=26, signal=9)["Histogram"] / (
            ohlcv["Close"]
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_a_non_positive_close_is_nan_not_inf(self):
        """Guarded like risk.atr_pct: a zero Close gives NaN, which alignment
        drops, not an infinity that refuses the panel."""
        from standard_quant_tools.modeling.features.base import FeatureContext
        from standard_quant_tools.modeling.features.registry import get_feature

        ohlcv = _adjusted("XYZ")
        ohlcv.iloc[200, ohlcv.columns.get_loc("Close")] = 0.0
        result = get_feature("technical.macd_histogram_pct").fn(ohlcv, FeatureContext())
        assert np.isnan(result.iloc[200])
        assert not np.isinf(result.to_numpy(dtype=float)).any()

    def test_the_raw_histogram_keeps_its_implementation_hash(self):
        """Its description gained a sentence; its function did not change,
        so models trained on it still score (pinned before the change)."""
        from standard_quant_tools.modeling.registry.feature_provenance import (
            feature_implementation_hash,
        )

        assert feature_implementation_hash("technical.macd_histogram") == (
            "3b61d7ec8155129e"
        )

    def test_it_is_comparable_across_price_levels(self):
        """The same path at 10x the price: the raw histogram is 10x, the
        ratio is the same."""
        from standard_quant_tools.modeling.features.base import FeatureContext
        from standard_quant_tools.modeling.features.registry import get_feature

        low = _adjusted("XYZ")
        high = low.copy()
        for column in ("Open", "High", "Low", "Close"):
            high[column] = high[column] * 10
        raw = get_feature("technical.macd_histogram").fn
        pct = get_feature("technical.macd_histogram_pct").fn
        pd.testing.assert_series_equal(
            raw(high, FeatureContext()), raw(low, FeatureContext()) * 10, rtol=1e-9
        )
        pd.testing.assert_series_equal(
            pct(high, FeatureContext()), pct(low, FeatureContext()), rtol=1e-12
        )
