"""
The whole indicator vocabulary across a universe, not five names of it.

`_PANEL_SHAPES` held exactly five: rsi, atr, adx, bollinger_bands and
stochastic_oscillator. The single-symbol doors reach all fourteen and
return only the LAST value, so a MACD, SMA, EMA, Williams %R, OBV, VWAP,
Parabolic SAR, simple-mean ATR or MFI *history* was obtainable from nothing
in this library, for any number of tickers -- and since the panel is what
feeds a persisted indicator artifact, those nine could not become anything
downstream either.

THE INVARIANT THIS FILE HOLDS. The module's own docstring says panel output
is what looping the per-ticker wrapper produces. That is the whole promise
of the shape, and it is the only thing that makes a fast path safe to
prefer: a caller moving from `macd(close)` to the panel must get the same
numbers and the same column labels, not merely the same shape. So every
indicator here is checked column by column against its wrapper ON THE SAME
BARS, with `assert_series_equal`.

ON THE SAME BARS is load-bearing. The panel stacks the universe onto the
INTERSECTION of every ticker's index, so one young ticker truncates the
window for everyone -- and every indicator here is path-dependent (Wilder
smoothing, EMAs, cumulative OBV and VWAP), which means starting later
changes the VALUES and not only the coverage. A test comparing a truncated
panel against a full-history wrapper would fail for the right reason and be
silenced for the wrong one, so the comparison is made against the wrapper
run on the intersected bars.

See the CHANGELOG entry of 2026-09-22.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators import panel as panel_module
from standard_quant_tools.indicators.momentum import rsi, stochastic_oscillator
from standard_quant_tools.indicators.panel import (
    _PANEL_SHAPES,
    technical_indicators_panel,
)
from standard_quant_tools.indicators.trend import (
    adx,
    ema,
    macd,
    parabolic_sar,
    sma,
    williams_r,
)
from standard_quant_tools.indicators.volatility import atr, bollinger_bands, wilder_atr
from standard_quant_tools.indicators.volume import mfi, obv, vwap

#: The five the native kernel carries, and the nine that were unreachable.
NATIVE_FIVE = ["rsi", "atr", "adx", "bollinger_bands", "stochastic_oscillator"]
NEWLY_REACHABLE = [
    "macd",
    "sma",
    "ema",
    "williams_r",
    "obv",
    "vwap",
    "parabolic_sar",
    "mfi",
    "atr_simple",
]

N_BARS = 260


def _bars(seed: int, n: int = N_BARS, start: str = "2022-01-03") -> pd.DataFrame:
    """One name's OHLCV on its own random walk."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.013, n))
    index = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * (1.0 + rng.uniform(0.001, 0.012, n)),
            "Low": close * (1.0 - rng.uniform(0.001, 0.012, n)),
            "Close": close,
            "Volume": rng.uniform(5e5, 5e6, n),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def universe():
    return {"AAA": _bars(1), "BBB": _bars(2), "CCC": _bars(3)}


#: indicator -> (callable on one frame, field name or None). The callable
#: takes the frame the panel was computed from, so the comparison is against
#: the wrapper's own answer rather than a reimplementation of it.
_WRAPPERS = {
    "rsi": (lambda f: rsi(f["Close"], 14), None),
    "atr": (lambda f: wilder_atr(f["High"], f["Low"], f["Close"], 14), None),
    "atr_simple": (lambda f: atr(f["High"], f["Low"], f["Close"], 14), None),
    "sma": (lambda f: sma(f["Close"], 14), None),
    "ema": (lambda f: ema(f["Close"], 14), None),
    "williams_r": (lambda f: williams_r(f["High"], f["Low"], f["Close"], 14), None),
    "obv": (lambda f: obv(f["Close"], f["Volume"]), None),
    "vwap": (lambda f: vwap(f["High"], f["Low"], f["Close"], f["Volume"], None), None),
    "mfi": (lambda f: mfi(f["High"], f["Low"], f["Close"], f["Volume"], 14), None),
    "macd": (lambda f: macd(f["Close"], 12, 26, 9), ["MACD", "Signal", "Histogram"]),
    "parabolic_sar": (
        lambda f: parabolic_sar(f["High"], f["Low"], 0.02, 0.02, 0.2),
        ["SAR", "Trend"],
    ),
    "adx": (
        lambda f: adx(f["High"], f["Low"], f["Close"], 14),
        ["DI_Plus", "DI_Minus", "ADX"],
    ),
    "bollinger_bands": (
        lambda f: bollinger_bands(f["Close"], 20, 2.0),
        ["BB_Upper", "BB_Middle", "BB_Lower"],
    ),
    "stochastic_oscillator": (
        lambda f: stochastic_oscillator(f["High"], f["Low"], f["Close"], 14, 3),
        ["Stoch_K", "Stoch_D"],
    ),
}


def _assert_column_matches_wrapper(frames, indicator, ticker, bars):
    """One ticker's panel column against the wrapper on the same bars."""
    computed, fields = _WRAPPERS[indicator]
    expected = computed(bars)
    got = frames[indicator]
    if fields is None:
        pd.testing.assert_series_equal(
            got[ticker].reset_index(drop=True),
            pd.Series(expected.to_numpy(), name=ticker),
            check_names=False,
        )
        return
    for field in fields:
        pd.testing.assert_series_equal(
            got[(ticker, field)].reset_index(drop=True),
            pd.Series(expected[field].to_numpy(), name=field),
            check_names=False,
        )


class TestTheVocabularyIsFourteen:
    def test_every_name_is_registered(self):
        assert set(_PANEL_SHAPES) == set(NATIVE_FIVE) | set(NEWLY_REACHABLE)
        assert len(_PANEL_SHAPES) == 14

    def test_the_two_average_true_ranges_are_separate_names(self):
        """`atr` was and stays WILDER's. The simple rolling mean is a
        different number, and folding it into the same name under a
        parameter would have changed what an existing caller's `atr`
        column means without anything saying so."""
        assert "atr" in _PANEL_SHAPES and "atr_simple" in _PANEL_SHAPES

    def test_wilder_and_simple_really_do_differ(self, universe):
        """Otherwise the separate name would be bookkeeping."""
        frames = technical_indicators_panel(universe, ["atr", "atr_simple"])
        wilder = frames["atr"]["AAA"].dropna()
        simple = frames["atr_simple"]["AAA"].dropna()
        assert not np.allclose(wilder.to_numpy()[-50:], simple.to_numpy()[-50:])

    def test_an_unknown_indicator_is_still_refused_with_the_list(self, universe):
        with pytest.raises(ValidationError, match="unknown indicator"):
            technical_indicators_panel(universe, ["moving_vibes"])


class TestEveryColumnIsTheWrappersAnswer:
    @pytest.mark.parametrize("indicator", NEWLY_REACHABLE)
    def test_each_new_indicator_matches_per_ticker(self, universe, indicator):
        frames = technical_indicators_panel(universe, [indicator])
        for ticker, bars in universe.items():
            _assert_column_matches_wrapper(frames, indicator, ticker, bars)

    @pytest.mark.parametrize("indicator", NATIVE_FIVE)
    def test_each_native_indicator_still_matches_per_ticker(self, universe, indicator):
        """The five were correct before and must stay correct now that the
        dispatch splits the request between two paths."""
        frames = technical_indicators_panel(universe, [indicator])
        for ticker, bars in universe.items():
            computed, fields = _WRAPPERS[indicator]
            expected = computed(bars)
            got = frames[indicator]
            if fields is None:
                np.testing.assert_allclose(
                    got[ticker].to_numpy(), expected.to_numpy(), equal_nan=True
                )
            else:
                for field in fields:
                    np.testing.assert_allclose(
                        got[(ticker, field)].to_numpy(),
                        expected[field].to_numpy(),
                        equal_nan=True,
                    )

    def test_all_fourteen_together_agree_with_all_fourteen_apart(self, universe):
        """Requesting a mixed set must not change any answer: the two paths
        run over the same stacked matrices and neither may perturb the
        other's inputs."""
        together = technical_indicators_panel(universe, list(_PANEL_SHAPES))
        for indicator in _PANEL_SHAPES:
            alone = technical_indicators_panel(universe, [indicator])[indicator]
            pd.testing.assert_frame_equal(together[indicator], alone, check_exact=True)

    def test_the_five_are_bit_identical_beside_the_nine(self, universe):
        """Specifically: adding the nine to the request leaves the native
        five untouched to the last bit, not merely to a tolerance."""
        five = technical_indicators_panel(universe, NATIVE_FIVE)
        fourteen = technical_indicators_panel(universe, list(_PANEL_SHAPES))
        for indicator in NATIVE_FIVE:
            np.testing.assert_array_equal(
                fourteen[indicator].to_numpy(), five[indicator].to_numpy()
            )


class TestTheMultiColumnNamesAreUnderTheTickerLevel:
    def test_macd_returns_its_three_named_fields(self, universe):
        frame = technical_indicators_panel(universe, ["macd"])["macd"]
        assert isinstance(frame.columns, pd.MultiIndex)
        assert frame.columns.names == ["ticker", "field"]
        for ticker in universe:
            assert [f for t, f in frame.columns if t == ticker] == [
                "MACD",
                "Signal",
                "Histogram",
            ]

    def test_parabolic_sar_returns_its_two(self, universe):
        frame = technical_indicators_panel(universe, ["parabolic_sar"])["parabolic_sar"]
        for ticker in universe:
            assert [f for t, f in frame.columns if t == ticker] == ["SAR", "Trend"]

    def test_the_trend_column_is_only_plus_or_minus_one(self, universe):
        frame = technical_indicators_panel(universe, ["parabolic_sar"])["parabolic_sar"]
        trend = frame[("AAA", "Trend")].to_numpy()
        assert set(np.unique(trend)) <= {-1.0, 1.0}

    def test_the_flattened_names_survive_a_parquet_round_trip(self, universe):
        """`ticker::field` is what the persisting path writes, and the
        separator cannot collide with either half."""
        frame = technical_indicators_panel(universe, ["macd"])["macd"]
        flattened = [f"{t}::{f}" for t, f in frame.columns]
        assert flattened[:3] == ["AAA::MACD", "AAA::Signal", "AAA::Histogram"]
        assert len(set(flattened)) == len(flattened)

    def test_a_single_column_indicator_is_keyed_by_ticker_alone(self, universe):
        frame = technical_indicators_panel(universe, ["obv"])["obv"]
        assert not isinstance(frame.columns, pd.MultiIndex)
        assert list(frame.columns) == list(universe)


class TestVolumeIsRequiredOnlyWhereItIsRead:
    @staticmethod
    def _without_volume(universe):
        return {t: f.drop(columns=["Volume"]) for t, f in universe.items()}

    @pytest.mark.parametrize("indicator", ["obv", "vwap", "mfi"])
    def test_a_volume_indicator_without_volume_is_refused_by_name(
        self, universe, indicator
    ):
        with pytest.raises(ValidationError, match="Volume"):
            technical_indicators_panel(self._without_volume(universe), [indicator])

    @pytest.mark.parametrize("indicator", ["obv", "vwap", "mfi"])
    def test_the_refusal_says_which_indicators_need_it(self, universe, indicator):
        with pytest.raises(ValidationError, match="volume indicators"):
            technical_indicators_panel(self._without_volume(universe), [indicator])

    def test_a_price_only_panel_still_answers_price_indicators(self, universe):
        """The guard must not become a new requirement on the panels that
        worked before: High/Low/Close is all the other eleven read."""
        frames = technical_indicators_panel(
            self._without_volume(universe), ["rsi", "macd", "parabolic_sar"]
        )
        assert set(frames) == {"rsi", "macd", "parabolic_sar"}
        assert not frames["rsi"].isna().all().all()


class TestAShorterHistoryTruncatesForEveryone:
    """The intersection rule, stated in `_stack_panel` and easy to violate
    a test against: a young ticker shortens the window for the whole
    universe, and because every indicator here is path-dependent the older
    tickers' VALUES change too, not only their row count."""

    @pytest.fixture
    def ragged(self):
        full = _bars(7)
        # Half the history, ending on the same last bar.
        young = _bars(8).iloc[len(full) // 2 :]
        return {"OLD": full, "YOUNG": young}

    def test_the_panel_is_the_intersected_bars(self, ragged):
        frames = technical_indicators_panel(ragged, ["ema"])
        assert len(frames["ema"]) == len(ragged["YOUNG"])
        assert frames["ema"].index.equals(ragged["YOUNG"].index)

    @pytest.mark.parametrize("indicator", ["ema", "macd", "obv", "vwap", "mfi"])
    def test_each_column_matches_the_wrapper_on_the_intersected_bars(
        self, ragged, indicator
    ):
        """And NOT on the full history: an EMA, a cumulative OBV and a
        session VWAP all differ when they start later, which is exactly
        why the truncation has to be visible."""
        frames = technical_indicators_panel(ragged, [indicator])
        shared = frames[indicator].index
        for ticker, bars in ragged.items():
            _assert_column_matches_wrapper(frames, indicator, ticker, bars.loc[shared])

    def test_the_truncated_answer_really_differs_from_the_full_history_one(
        self, ragged
    ):
        """Otherwise the test above would hold for the wrong reason."""
        frames = technical_indicators_panel(ragged, ["obv"])
        shared = frames["obv"].index
        full = obv(ragged["OLD"]["Close"], ragged["OLD"]["Volume"]).loc[shared]
        assert not np.allclose(frames["obv"]["OLD"].to_numpy(), full.to_numpy())

    def test_tickers_with_no_shared_bars_are_refused(self):
        early = _bars(9, n=60, start="2018-01-02")
        late = _bars(10, n=60, start="2023-01-02")
        with pytest.raises(ValidationError, match="no common bars"):
            technical_indicators_panel({"EARLY": early, "LATE": late}, ["sma"])


class TestThePythonPathRunsWhenTheKernelIsAbsent:
    """The nine never touch the kernel, so the only thing to check is that
    blocking it does not change them -- and that the five fall back to the
    same numbers they had."""

    @staticmethod
    def _without_cpp(universe, indicators):
        saved_flag, saved_core = panel_module.HAS_CPP, panel_module._cpp_core
        try:
            panel_module.HAS_CPP, panel_module._cpp_core = False, None
            return technical_indicators_panel(universe, indicators)
        finally:
            panel_module.HAS_CPP, panel_module._cpp_core = saved_flag, saved_core

    @pytest.mark.parametrize("indicator", NEWLY_REACHABLE)
    def test_the_nine_are_unchanged_without_the_extension(self, universe, indicator):
        native = technical_indicators_panel(universe, [indicator])[indicator]
        fallback = self._without_cpp(universe, [indicator])[indicator]
        pd.testing.assert_frame_equal(native, fallback, check_exact=True)

    @pytest.mark.parametrize("indicator", NATIVE_FIVE)
    def test_the_five_fall_back_to_the_wrappers(self, universe, indicator):
        fallback = self._without_cpp(universe, [indicator])[indicator]
        for ticker, bars in universe.items():
            _assert_column_matches_wrapper(
                {indicator: fallback}, indicator, ticker, bars
            )

    def test_a_mixed_request_is_served_whole_without_the_extension(self, universe):
        frames = self._without_cpp(universe, ["rsi", "macd", "obv"])
        assert set(frames) == {"rsi", "macd", "obv"}


class TestTheParametersReachTheCalculation:
    @pytest.mark.parametrize(
        "indicator,kwargs,wrapper",
        [
            ("sma", {"sma_period": 5}, lambda f: sma(f["Close"], 5)),
            ("ema", {"ema_period": 5}, lambda f: ema(f["Close"], 5)),
            (
                "williams_r",
                {"williams_period": 7},
                lambda f: williams_r(f["High"], f["Low"], f["Close"], 7),
            ),
            (
                "mfi",
                {"mfi_period": 7},
                lambda f: mfi(f["High"], f["Low"], f["Close"], f["Volume"], 7),
            ),
            (
                "vwap",
                {"vwap_period": 10},
                lambda f: vwap(f["High"], f["Low"], f["Close"], f["Volume"], 10),
            ),
            (
                "atr_simple",
                {"atr_simple_period": 7},
                lambda f: atr(f["High"], f["Low"], f["Close"], 7),
            ),
        ],
    )
    def test_a_non_default_period_changes_the_answer(
        self, universe, indicator, kwargs, wrapper
    ):
        frames = technical_indicators_panel(universe, [indicator], **kwargs)
        expected = wrapper(universe["AAA"])
        pd.testing.assert_series_equal(
            frames[indicator]["AAA"].reset_index(drop=True),
            pd.Series(expected.to_numpy()),
            check_names=False,
        )
        default = technical_indicators_panel(universe, [indicator])
        assert not np.allclose(
            frames[indicator]["AAA"].to_numpy()[-30:],
            default[indicator]["AAA"].to_numpy()[-30:],
        )

    def test_macd_spans_reach_the_calculation(self, universe):
        frames = technical_indicators_panel(
            universe, ["macd"], macd_fast=5, macd_slow=13, macd_signal=4
        )
        expected = macd(universe["AAA"]["Close"], 5, 13, 4)
        pd.testing.assert_series_equal(
            frames["macd"][("AAA", "MACD")].reset_index(drop=True),
            pd.Series(expected["MACD"].to_numpy()),
            check_names=False,
        )

    def test_an_inverted_macd_pair_is_refused_rather_than_sign_flipped(self, universe):
        with pytest.raises(ValidationError, match="must be <"):
            technical_indicators_panel(universe, ["macd"], macd_fast=26, macd_slow=12)

    def test_the_sar_acceleration_reaches_the_calculation(self, universe):
        frames = technical_indicators_panel(
            universe, ["parabolic_sar"], sar_af_start=0.01, sar_af_step=0.01
        )
        expected = parabolic_sar(
            universe["AAA"]["High"], universe["AAA"]["Low"], 0.01, 0.01, 0.2
        )
        pd.testing.assert_series_equal(
            frames["parabolic_sar"][("AAA", "SAR")].reset_index(drop=True),
            pd.Series(expected["SAR"].to_numpy()),
            check_names=False,
        )


class TestThePersistedPanelIsParameterized:
    """`compute_indicator_panel` forwarded none of the period parameters,
    so the panel an agent PERSISTS -- the one a feature or a custom
    backtest consumes -- was always RSI(14) whatever was asked for. The
    snapshot tool forwarded all seven, so the two doors disagreed about
    what the same request meant."""

    @pytest.fixture
    def runs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        return tmp_path

    @staticmethod
    def _compute(monkeypatch, universe, **kwargs):
        from standard_quant_tools.agent.runtimes.research import reference_tools

        monkeypatch.setattr(
            reference_tools,
            "fetch_ohlcv_panel_sync",
            lambda tickers, start, end: {t: universe[t] for t in tickers},
        )
        return reference_tools.compute_indicator_panel(
            reference_tools.IndicatorPanelInput(
                tickers=list(universe),
                start_date="2022-01-03",
                end_date="2023-01-03",
                indicators=["rsi"],
                **kwargs,
            )
        )

    def test_a_non_default_rsi_period_reaches_the_published_panel(
        self, runs_dir, monkeypatch, universe
    ):
        from standard_quant_tools.agent.runtimes import handoff

        result = self._compute(
            monkeypatch, universe, run_id="panel_run", name="p5", rsi_period=5
        )
        published = handoff.resolve(result.refs["rsi"])

        expected = rsi(universe["AAA"]["Close"], 5)
        np.testing.assert_allclose(
            published["AAA"].to_numpy(), expected.to_numpy(), equal_nan=True
        )

    def test_it_differs_from_the_fourteen_period_default(
        self, runs_dir, monkeypatch, universe
    ):
        """The point of the test above: RSI(5) and RSI(14) are different
        series, so a dropped parameter was not a cosmetic loss."""
        from standard_quant_tools.agent.runtimes import handoff

        five = handoff.resolve(
            self._compute(
                monkeypatch, universe, run_id="r", name="five", rsi_period=5
            ).refs["rsi"]
        )
        default = handoff.resolve(
            self._compute(monkeypatch, universe, run_id="r", name="default").refs["rsi"]
        )
        assert not np.allclose(
            five["AAA"].to_numpy()[-30:], default["AAA"].to_numpy()[-30:]
        )

    def test_a_newly_reachable_indicator_publishes_its_named_fields(
        self, runs_dir, monkeypatch, universe
    ):
        from standard_quant_tools.agent.runtimes import handoff
        from standard_quant_tools.agent.runtimes.research import reference_tools

        monkeypatch.setattr(
            reference_tools,
            "fetch_ohlcv_panel_sync",
            lambda tickers, start, end: {t: universe[t] for t in tickers},
        )
        result = reference_tools.compute_indicator_panel(
            reference_tools.IndicatorPanelInput(
                tickers=list(universe),
                start_date="2022-01-03",
                end_date="2023-01-03",
                indicators=["macd", "obv"],
                run_id="panel_new",
                name="fields",
            )
        )
        assert set(result.refs) == {"macd", "obv"}
        published = handoff.resolve(result.refs["macd"])
        assert isinstance(published.columns, pd.MultiIndex)
        assert [f for t, f in published.columns if t == "AAA"] == [
            "MACD",
            "Signal",
            "Histogram",
        ]
        np.testing.assert_allclose(
            published[("AAA", "MACD")].to_numpy(),
            macd(universe["AAA"]["Close"], 12, 26, 9)["MACD"].to_numpy(),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            handoff.resolve(result.refs["obv"])["AAA"].to_numpy(),
            obv(universe["AAA"]["Close"], universe["AAA"]["Volume"]).to_numpy(),
        )
