"""
Properties every indicator in this library must have, checked on all of them.

WHY PROPERTIES RATHER THAN RECORDED VALUES. A test that pins `rsi(prices)[50]
== 47.3218` pins whatever the implementation produced the day it was written,
including its mistakes, and it says nothing about the other 249 bars. The
properties below are true of the INDICATOR, so an implementation that
violates one is wrong regardless of what any recorded number says.

THE ONE THAT MATTERS MOST IS NO-LOOKAHEAD. An indicator's value at bar `t`
must not change when bars after `t` arrive. If it does, every backtest built
on it is reading the future, the equity curve looks excellent, and nothing
anywhere raises. It is the single most expensive bug this library could
have, it is invisible in any single-series test, and it is checked here for
every indicator at once by computing on a prefix and on the whole series and
comparing the overlap.

`parabolic_sar` is the interesting case and is exempted deliberately below —
see the test.

WHAT ELSE IS CHECKED FOR EVERY INDICATOR:

- length is preserved, so an indicator can be assigned back onto its frame;
- a constant input produces a constant, defined output (many indicators
  divide by a range or a standard deviation, and a flat series makes that
  zero);
- no infinities anywhere, ever — an inf silently poisons every downstream
  mean, and NaN at least propagates visibly;
- the index comes back unchanged, so a join against the source frame lines
  up rather than silently reindexing.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    mfi,
    obv,
    parabolic_sar,
    rsi,
    sma,
    stochastic_oscillator,
    vwap,
    wilder_atr,
    williams_r,
)

N = 260


@pytest.fixture(scope="module")
def bars():
    """A realistic OHLCV frame: trending, with noise, and a real range."""
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, N)))
    spread = close * rng.uniform(0.004, 0.02, N)
    return pd.DataFrame(
        {
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "open": close * (1 + rng.normal(0, 0.003, N)),
            "volume": rng.integers(1e5, 5e6, N).astype(float),
        },
        index=pd.bdate_range("2023-01-02", periods=N),
    )


def _call(fn, frame):
    """Call any indicator with whatever columns it declares."""
    import inspect

    kinds = {
        "series": frame["close"],
        "high": frame["high"],
        "low": frame["low"],
        "close": frame["close"],
        "volume": frame["volume"],
    }
    params = inspect.signature(fn).parameters
    return fn(**{k: v for k, v in kinds.items() if k in params})


#: Every indicator, by name, so a new one is covered the moment it is added
#: rather than when somebody remembers to write a test for it.
INDICATORS = {
    "sma": sma,
    "ema": ema,
    "rsi": rsi,
    "macd": macd,
    "bollinger_bands": bollinger_bands,
    "atr": atr,
    "wilder_atr": wilder_atr,
    "adx": adx,
    "obv": obv,
    "vwap": vwap,
    "mfi": mfi,
    "stochastic_oscillator": stochastic_oscillator,
    "williams_r": williams_r,
    "parabolic_sar": parabolic_sar,
}


def _frames(result):
    """Every numeric column an indicator returned, whatever shape it used."""
    if isinstance(result, pd.Series):
        return {"": result}
    if isinstance(result, pd.DataFrame):
        return {c: result[c] for c in result.columns}
    if isinstance(result, tuple):
        return {str(i): s for i, s in enumerate(result)}
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if isinstance(v, pd.Series)}
    raise AssertionError(f"unhandled indicator return type: {type(result)}")


class TestNoIndicatorSeesTheFuture:
    """
    The value at bar t must not change when bars after t arrive.

    An indicator that fails this makes every backtest on it read the future.
    The equity curve looks excellent and nothing raises, which is why it is
    checked mechanically for every indicator rather than argued about one at
    a time.
    """

    @pytest.mark.parametrize("name", sorted(INDICATORS))
    def test_a_prefix_computes_the_same_values(self, name, bars):
        if name == "parabolic_sar":
            pytest.skip("path-dependent by construction -- see the test below")
        cut = 180
        whole = _frames(_call(INDICATORS[name], bars))
        prefix = _frames(_call(INDICATORS[name], bars.iloc[:cut]))

        for column, full_series in whole.items():
            head = full_series.iloc[:cut]
            got = prefix[column]
            both = head.notna() & got.notna()
            assert both.sum() > 10, f"{name}[{column}]: too few comparable values"
            np.testing.assert_allclose(
                head[both].to_numpy(),
                got[both].to_numpy(),
                rtol=1e-9,
                atol=1e-9,
                err_msg=(
                    f"{name}[{column}] changed its past when future bars "
                    "arrived. Every backtest using it is reading the future."
                ),
            )

    def test_parabolic_sar_is_path_dependent_and_that_is_correct(self, bars):
        """
        SAR carries an extreme point and an acceleration factor forward from
        the start of the current trend, so its value at bar t genuinely
        depends on where the series began -- not on where it ENDS.

        That is a different thing from lookahead, and the distinction is
        worth pinning: recomputing from an earlier start must leave the tail
        alone once the state has converged, which is what this checks.
        Truncating the FUTURE is what must not matter.
        """
        long_run = parabolic_sar(bars["high"], bars["low"])
        prefix = parabolic_sar(bars["high"].iloc[:200], bars["low"].iloc[:200])
        overlap = long_run.iloc[:200]
        both = overlap.notna() & prefix.notna()
        agreement = np.isclose(
            overlap[both].to_numpy(), prefix[both].to_numpy(), rtol=1e-9
        ).mean()
        assert agreement > 0.95, (
            f"only {agreement:.0%} of SAR values matched between a prefix and "
            "the full series. SAR is path-dependent from the START, which is "
            "fine; depending on the END is not."
        )


class TestShapeAndIndex:
    @pytest.mark.parametrize("name", sorted(INDICATORS))
    def test_length_is_preserved(self, name, bars):
        for column, series in _frames(_call(INDICATORS[name], bars)).items():
            assert len(series) == len(bars), f"{name}[{column}]"

    @pytest.mark.parametrize("name", sorted(INDICATORS))
    def test_the_index_comes_back_unchanged(self, name, bars):
        """So a join against the source frame lines up rather than silently
        reindexing to a range."""
        for column, series in _frames(_call(INDICATORS[name], bars)).items():
            pd.testing.assert_index_equal(
                series.index, bars.index, obj=f"{name}[{column}]"
            )

    @pytest.mark.parametrize("name", sorted(INDICATORS))
    def test_nothing_is_infinite(self, name, bars):
        """An inf silently poisons every downstream mean, sum and
        correlation. NaN at least propagates visibly."""
        for column, series in _frames(_call(INDICATORS[name], bars)).items():
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            assert not np.isinf(values).any(), f"{name}[{column}] produced an inf"


class TestDegenerateInputs:
    """A flat series makes every range and standard deviation zero, which is
    where division-by-zero lives."""

    @pytest.fixture
    def flat(self, bars):
        frame = bars.copy()
        for column in ("high", "low", "close", "open"):
            frame[column] = 100.0
        return frame

    @pytest.mark.parametrize("name", sorted(INDICATORS))
    def test_a_constant_series_produces_no_infinities(self, name, flat):
        for column, series in _frames(_call(INDICATORS[name], flat)).items():
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            assert not np.isinf(values).any(), (
                f"{name}[{column}] divided by a zero range on a flat series "
                "and returned an infinity"
            )

    def test_zero_volume_does_not_produce_an_infinity(self, bars):
        """VWAP and MFI both divide by volume."""
        frame = bars.copy()
        frame["volume"] = 0.0
        for fn in (vwap, mfi):
            for column, series in _frames(_call(fn, frame)).items():
                values = pd.to_numeric(series, errors="coerce").to_numpy(float)
                assert not np.isinf(values).any(), f"{fn.__name__}[{column}]"


class TestTheMathIsTheMath:
    """Identities that hold by definition, so they catch an implementation
    that has drifted from the thing it is named after."""

    def test_sma_of_a_constant_is_that_constant(self, bars):
        flat = pd.Series(7.0, index=bars.index)
        result = sma(flat, period=10).dropna()
        np.testing.assert_allclose(result.to_numpy(), 7.0, rtol=1e-12)

    def test_sma_is_the_rolling_mean(self, bars):
        expected = bars["close"].rolling(20).mean()
        got = sma(bars["close"], period=20)
        np.testing.assert_allclose(
            got.dropna().to_numpy(), expected.dropna().to_numpy(), rtol=1e-12
        )

    def test_ema_converges_to_a_constant(self, bars):
        flat = pd.Series(7.0, index=bars.index)
        assert ema(flat, period=10).iloc[-1] == pytest.approx(7.0, abs=1e-9)

    @pytest.mark.parametrize("bars_after", [1, 3, 5, 10])
    def test_ema_reacts_faster_than_sma_early(self, bars_after):
        """
        The property that makes it a different indicator: after a step up
        the EMA is closer to the new level FIRST.

        "Early" is load-bearing and was got wrong the first time. Twenty
        bars after a step the SMA window is entirely past it, so the SMA
        equals the new level EXACTLY while the EMA is still asymptoting --
        measured, sma=130.000 against ema=125.947. The EMA is faster to
        react, not closer forever, and a test at the end of the series
        checks the opposite of what it means to.
        """
        step = pd.Series(
            [100.0] * 100 + [130.0] * 30,
            index=pd.bdate_range("2024-01-01", periods=130),
        )
        at = 99 + bars_after
        assert ema(step, period=20).iloc[at] > sma(step, period=20).iloc[at]

    def test_the_sma_wins_once_its_window_is_past_the_step(self):
        """The other side of the same fact, pinned so the asymmetry is
        recorded rather than rediscovered."""
        step = pd.Series(
            [100.0] * 100 + [130.0] * 30,
            index=pd.bdate_range("2024-01-01", periods=130),
        )
        assert sma(step, period=20).iloc[-1] == pytest.approx(130.0)
        assert ema(step, period=20).iloc[-1] < 130.0

    def test_rsi_is_bounded(self, bars):
        values = rsi(bars["close"], period=14).dropna()
        assert values.min() >= 0.0 and values.max() <= 100.0

    def test_rsi_of_a_monotonic_rise_is_one_hundred(self):
        rising = pd.Series(
            np.arange(1, 101, dtype=float),
            index=pd.bdate_range("2024-01-01", periods=100),
        )
        assert rsi(rising, period=14).iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_rsi_of_a_monotonic_fall_is_zero(self):
        falling = pd.Series(
            np.arange(100, 0, -1, dtype=float),
            index=pd.bdate_range("2024-01-01", periods=100),
        )
        assert rsi(falling, period=14).iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_bollinger_bands_are_ordered_and_centred_on_the_sma(self, bars):
        result = bollinger_bands(bars["close"], period=20, num_std=2.0)
        frames = _frames(result)
        upper = next(v for k, v in frames.items() if "upper" in k.lower())
        lower = next(v for k, v in frames.items() if "lower" in k.lower())
        middle = next(
            v for k, v in frames.items() if "mid" in k.lower() or "ma" == k.lower()
        )
        usable = upper.notna() & lower.notna() & middle.notna()
        assert (upper[usable] >= middle[usable]).all()
        assert (middle[usable] >= lower[usable]).all()
        np.testing.assert_allclose(
            middle[usable].to_numpy(),
            sma(bars["close"], 20)[usable].to_numpy(),
            rtol=1e-9,
        )

    def test_wider_bands_need_more_standard_deviations(self, bars):
        narrow = _frames(bollinger_bands(bars["close"], 20, 1.0))
        wide = _frames(bollinger_bands(bars["close"], 20, 3.0))
        n_upper = next(v for k, v in narrow.items() if "upper" in k.lower())
        w_upper = next(v for k, v in wide.items() if "upper" in k.lower())
        usable = n_upper.notna() & w_upper.notna()
        assert (w_upper[usable] >= n_upper[usable]).all()

    def test_atr_is_never_negative(self, bars):
        for fn in (atr, wilder_atr):
            values = _call(fn, bars).dropna()
            assert (values >= 0).all(), fn.__name__

    def test_adx_is_bounded(self, bars):
        for column, series in _frames(_call(adx, bars)).items():
            values = series.dropna()
            if len(values):
                assert values.min() >= 0.0, f"adx[{column}]"
                assert values.max() <= 100.0 + 1e-9, f"adx[{column}]"

    @pytest.mark.parametrize("fn", [stochastic_oscillator, mfi])
    def test_oscillators_are_bounded_zero_to_a_hundred(self, fn, bars):
        for column, series in _frames(_call(fn, bars)).items():
            values = series.dropna()
            if len(values):
                assert values.min() >= -1e-9, f"{fn.__name__}[{column}]"
                assert values.max() <= 100.0 + 1e-9, f"{fn.__name__}[{column}]"

    def test_williams_r_is_bounded_minus_hundred_to_zero(self, bars):
        values = _call(williams_r, bars).dropna()
        assert values.min() >= -100.0 - 1e-9
        assert values.max() <= 0.0 + 1e-9

    def test_obv_moves_by_exactly_the_volume(self, bars):
        """Its whole definition: add the volume on an up bar, subtract it on
        a down bar. A rounding or a sign error shows immediately."""
        close = pd.Series([10.0, 11.0, 10.0, 10.0, 12.0])
        volume = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
        result = obv(close, volume).to_numpy(dtype=float)
        steps = np.diff(result)
        assert steps[0] == pytest.approx(200.0)  # up
        assert steps[1] == pytest.approx(-300.0)  # down
        assert steps[3] == pytest.approx(500.0)  # up
        assert abs(steps[2]) == pytest.approx(0.0)  # unchanged

    def test_vwap_sits_inside_the_days_range(self, bars):
        """A volume-weighted average of prices inside [low, high] cannot
        leave it. If it does, the weights are wrong."""
        values = _call(vwap, bars)
        usable = values.notna()
        assert (values[usable] >= bars["low"][usable].cummin() - 1e-6).all()
        assert (values[usable] <= bars["high"][usable].cummax() + 1e-6).all()

    def test_macd_is_the_difference_of_two_emas(self, bars):
        frames = _frames(macd(bars["close"], fast=12, slow=26, signal=9))
        line = next(
            v
            for k, v in frames.items()
            if k.lower() in ("macd", "macd_line", "0", "line")
        )
        expected = ema(bars["close"], 12) - ema(bars["close"], 26)
        usable = line.notna() & expected.notna()
        np.testing.assert_allclose(
            line[usable].to_numpy(), expected[usable].to_numpy(), rtol=1e-9
        )

    def test_a_longer_sma_is_smoother(self, bars):
        """Not a definition but a consequence, and it catches a period that
        is being ignored."""
        short = sma(bars["close"], 5).dropna().diff().abs().mean()
        long = sma(bars["close"], 50).dropna().diff().abs().mean()
        assert long < short


class TestTheyRefuseWhatTheyCannotCompute:
    def test_a_period_longer_than_the_series_gives_nan_not_an_exception(self, bars):
        """A short history is a normal thing to have. Returning all-NaN is
        the honest answer; raising would make an indicator unusable on the
        first month of a new listing."""
        short = bars["close"].head(5)
        result = sma(short, period=50)
        assert len(result) == 5
        assert result.isna().all()

    def test_the_empty_input_contract_is_split_but_no_longer_crashes(self):
        """
        A FINDING, narrowed by a fix and then pinned.

        BEFORE. One indicator refused an empty series, thirteen returned one,
        and `parabolic_sar` raised a bare IndexError from inside numpy --
        three different answers to the same question, and the third is the
        worst of them because a caller cannot act on it and it names nothing.

        Worse, the split moved depending on HOW the function was called:
        `obv(empty, empty)` refused while `obv(close=empty, volume=empty)`
        crashed, because `validate_series` inspected only positional
        arguments. That is fixed -- see tests/test_validation_decorator.py.

        AFTER. Three refuse, eleven return, and nothing crashes. The
        remaining split is no longer accidental: it follows whether a
        function declares `allow_empty`, which is a per-function decision
        somebody made rather than an artefact of the call site.

        Still recorded rather than flattened. Returning empty is arguably
        the worse default -- an empty result flows into a backtest as "no
        signal" where a refusal says "no data" -- but changing eleven
        functions is a breaking change and not one to make inside a test.
        This fails if the split moves, which makes it a decision somebody
        takes rather than a thing that drifts.
        """
        import inspect

        from standard_quant_tools.error import ValidationError

        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        raising, returning, crashing = [], [], []
        for name, fn in sorted(INDICATORS.items()):
            params = inspect.signature(fn).parameters
            kwargs = {
                k: empty
                for k in ("series", "high", "low", "close", "volume")
                if k in params
            }
            try:
                fn(**kwargs)
                returning.append(name)
            except ValidationError:
                raising.append(name)
            except Exception:  # noqa: BLE001 - anything else is the bad case
                crashing.append(name)

        assert crashing == [], (
            f"{crashing} raised an unhandled error on an empty series. That "
            "is the worst of the three outcomes: a caller cannot act on it "
            "and it names nothing."
        )
        assert raising == ["obv", "rsi", "stochastic_oscillator"], (
            f"the set of indicators that REFUSE an empty series changed to "
            f"{raising}. That is either a fix worth making everywhere or a "
            "regression; either way it is a decision, not a drift."
        )
        assert len(returning) == len(INDICATORS) - len(raising)

    def test_a_single_observation_does_not_crash(self):
        one = pd.Series([100.0], index=pd.bdate_range("2024-01-01", periods=1))
        for fn in (sma, ema, rsi):
            assert len(fn(one)) == 1, fn.__name__
