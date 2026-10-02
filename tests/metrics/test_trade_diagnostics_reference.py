"""
`trade_excursions` and `exposure_stats` locate every trade at once, and must
answer exactly as the per-trade loops they replaced.

Each reference below is the implementation as it stood before that change,
kept verbatim so the comparison is against the old code itself rather than
against a restatement of it (see the CHANGELOG entry of 2026-10-01). Every
comparison is exact: same frame, same dtypes, same NaN positions, same dict.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics import diagnostics
from standard_quant_tools.metrics.diagnostics import exposure_stats, trade_excursions

logger = logging.getLogger("reference_trade_diagnostics")


# ── the implementations before the change, verbatim ──────────────────────


def _reference_trade_excursions(
    trade_log: pd.DataFrame, price_data: pd.DataFrame
) -> pd.DataFrame:
    if trade_log.empty:
        result = trade_log.copy()
        result["mae_pct"] = pd.Series(dtype=float)
        result["mfe_pct"] = pd.Series(dtype=float)
        return result

    mae_list: List[float] = []
    mfe_list: List[float] = []
    for _, row in trade_log.iterrows():
        window = price_data.loc[row["entry_date"] : row["exit_date"]]
        entry_price = float(row["entry_price"])
        if window.empty or not np.isfinite(entry_price) or entry_price <= 0:
            mae_list.append(float("nan"))
            mfe_list.append(float("nan"))
            continue

        is_long = row["direction"] == "long"
        high, low = float(window["High"].max()), float(window["Low"].min())
        if is_long:
            mfe = (high - entry_price) / entry_price
            mae = (low - entry_price) / entry_price
        else:
            mfe = (entry_price - low) / entry_price
            mae = (entry_price - high) / entry_price
        mfe_list.append(round(mfe * 100, 4))
        mae_list.append(round(mae * 100, 4))

    result = trade_log.copy()
    result["mae_pct"] = mae_list
    result["mfe_pct"] = mfe_list
    return result


def _reference_exposure_stats(
    executed_signal: pd.Series,
    trade_log: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    values = executed_signal.to_numpy(dtype=float)
    if len(values) and not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValidationError(
            f"executed_signal contains {n_bad} non-finite value(s). A NaN "
            "position satisfies `!= 0`, so it would be counted as time in the "
            "market while making every exposure average NaN."
        )
    if len(values) == 0:
        return {
            "time_in_market": 0.0,
            "avg_gross_exposure": 0.0,
            "avg_net_exposure": 0.0,
            "pct_long": 0.0,
            "pct_short": 0.0,
            "avg_holding_period_bars": None,
        }

    avg_holding_period_bars: Optional[float] = None
    if trade_log is not None and not trade_log.empty:
        idx = executed_signal.index
        holding_bars: List[int] = []
        for _, row in trade_log.iterrows():
            try:
                entry_pos = idx.get_loc(row["entry_date"])
                exit_pos = idx.get_loc(row["exit_date"])
            except KeyError:
                continue
            if not isinstance(entry_pos, int) or not isinstance(exit_pos, int):
                logger.warning(
                    "[exposure_stats] ambiguous index position for trade "
                    "%s -> %s (duplicate timestamps?) — excluded from "
                    "avg_holding_period_bars",
                    row["entry_date"],
                    row["exit_date"],
                )
                continue
            holding_bars.append(exit_pos - entry_pos)
        if holding_bars:
            avg_holding_period_bars = round(float(np.mean(holding_bars)), 2)

    return {
        "time_in_market": round(float((values != 0).mean()), 4),
        "avg_gross_exposure": round(float(np.abs(values).mean()), 4),
        "avg_net_exposure": round(float(values.mean()), 4),
        "pct_long": round(float((values > 0).mean()), 4),
        "pct_short": round(float((values < 0).mean()), 4),
        "avg_holding_period_bars": avg_holding_period_bars,
    }


# ── inputs shaped like a backtest's ──────────────────────────────────────


def _prices(n_bars: int = 2100, seed: int = 0, tz: Optional[str] = None):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_bars)))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.01, n_bars)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.01, n_bars)))
    index = pd.bdate_range("2018-01-02", periods=n_bars, tz=tz)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close}, index=index
    )


def _trades(price: pd.DataFrame, n_trades: int = 805, seed: int = 0):
    """Back-to-back trades, as the engine's trade log lays them out."""
    rng = np.random.default_rng(seed)
    n = len(price)
    cuts = np.sort(rng.choice(np.arange(1, n - 1), size=2 * n_trades, replace=False))
    entries, exits = cuts[0::2], cuts[1::2]
    direction = np.where(rng.random(n_trades) < 0.5, "long", "short")
    close = price["Close"].to_numpy()
    return pd.DataFrame(
        {
            "entry_date": price.index[entries],
            "exit_date": price.index[exits],
            "direction": direction,
            "entry_price": np.round(close[entries], 4),
            "exit_price": np.round(close[exits], 4),
            "position_size": np.where(direction == "long", 1.0, -1.0),
            "return_pct": rng.normal(0.0, 2.0, n_trades),
        }
    )


def _executed(price: pd.DataFrame, log: pd.DataFrame) -> pd.Series:
    signal = pd.Series(0.0, index=price.index)
    for entry, exit_, side in zip(
        log["entry_date"], log["exit_date"], log["direction"]
    ):
        signal.loc[entry:exit_] = 1.0 if side == "long" else -1.0
    return signal


def _same_excursions(log: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Both answers, compared exactly; an exception must be the same one."""
    try:
        expected = _reference_trade_excursions(log, price)
    except Exception as error:  # noqa: BLE001 -- the type is what is compared
        with pytest.raises(type(error)):
            trade_excursions(log, price)
        return None
    actual = trade_excursions(log, price)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for column in ("mae_pct", "mfe_pct"):
        assert np.array_equal(
            actual[column].to_numpy(), expected[column].to_numpy(), equal_nan=True
        )
    return actual


# ── trade_excursions ─────────────────────────────────────────────────────


class TestTradeExcursionsMatchesThePerTradeSlice:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_engine_shaped_log_is_identical(self, seed):
        price = _prices(seed=seed)
        log = _trades(price, seed=seed)
        # The located path is the one under test, not the fallback.
        assert diagnostics._window_bounds(log, price) is not None
        out = _same_excursions(log, price)
        assert out["mae_pct"].notna().all()

    def test_tz_aware_index(self):
        price = _prices(n_bars=300, tz="America/New_York")
        log = _trades(price, n_trades=90)
        assert diagnostics._window_bounds(log, price) is not None
        _same_excursions(log, price)

    def test_empty_trade_log(self):
        log = pd.DataFrame(
            columns=["entry_date", "exit_date", "direction", "entry_price"]
        )
        _same_excursions(log, _prices(n_bars=10))

    def test_one_trade(self):
        price = _prices(n_bars=50)
        _same_excursions(_trades(price, n_trades=1, seed=4), price)

    def test_trade_at_the_first_and_last_bar(self):
        price = _prices(n_bars=60)
        index = price.index
        log = pd.DataFrame(
            {
                "entry_date": [index[0], index[0], index[-1], index[10]],
                "exit_date": [index[-1], index[0], index[-1], index[-1]],
                "direction": ["long", "short", "long", "short"],
                "entry_price": [100.0, 100.0, 100.0, 100.0],
            }
        )
        _same_excursions(log, price)

    def test_dates_absent_from_the_price_index(self):
        price = _prices(n_bars=40)
        index = price.index
        weekend = index[3] + pd.Timedelta(days=(5 - index[3].dayofweek) % 7 or 7)
        log = pd.DataFrame(
            {
                "entry_date": [
                    weekend,  # a Saturday: no bar, the window starts after it
                    index[0] - pd.Timedelta(days=30),  # before the data
                    index[-1] + pd.Timedelta(days=5),  # after the data: empty
                    index[5] + pd.Timedelta(hours=12),  # between two bars
                    index[20],
                ],
                "exit_date": [
                    index[12],
                    index[2],
                    index[-1] + pd.Timedelta(days=9),
                    index[9] + pd.Timedelta(hours=1),
                    index[10],  # exit before entry: empty
                ],
                "direction": ["long", "short", "long", "short", "long"],
                "entry_price": [100.0] * 5,
            }
        )
        out = _same_excursions(log, price)
        assert out["mae_pct"].isna().tolist() == [False, False, True, False, True]

    def test_duplicate_timestamps_in_a_sorted_index(self):
        price = _prices(n_bars=30)
        doubled = pd.concat([price, price.iloc[[4, 4, 9, 17]]]).sort_index(
            kind="stable"
        )
        doubled.iloc[5, doubled.columns.get_loc("High")] = 500.0  # a repeat of bar 4
        index = price.index
        log = pd.DataFrame(
            {
                "entry_date": [index[4], index[9], index[0], index[17]],
                "exit_date": [index[4], index[17], index[4], index[17]],
                "direction": ["long", "short", "long", "long"],
                "entry_price": [100.0, 100.0, 100.0, 100.0],
            }
        )
        assert diagnostics._window_bounds(log, doubled) is not None
        out = _same_excursions(log, doubled)
        # Every repeat of a boundary date is inside the window, as .loc has it.
        assert out.loc[0, "mfe_pct"] == pytest.approx(400.0)

    def test_nan_prices(self):
        price = _prices(n_bars=80)
        price.iloc[[3, 4, 5, 6, 40], price.columns.get_loc("High")] = np.nan
        price.iloc[[3, 4, 5, 6, 41], price.columns.get_loc("Low")] = np.nan
        index = price.index
        log = pd.DataFrame(
            {
                # Bars 3-6 are all-NaN: a window of only those is NaN, not 0.
                "entry_date": [index[3], index[2], index[39], index[10]],
                "exit_date": [index[6], index[8], index[42], index[12]],
                "direction": ["long", "short", "long", "short"],
                "entry_price": [100.0, np.nan, 100.0, -1.0],
            }
        )
        out = _same_excursions(log, price)
        assert out["mae_pct"].isna().tolist() == [True, True, False, True]

    def test_inf_and_integer_prices(self):
        price = _prices(n_bars=40)
        price["High"] = np.round(price["High"]).astype(np.int64)
        price["Low"] = np.round(price["Low"]).astype(np.int64)
        log = _trades(price, n_trades=12, seed=5)
        assert diagnostics._window_bounds(log, price) is not None
        _same_excursions(log, price)
        price = _prices(n_bars=40)
        price.iloc[7, price.columns.get_loc("High")] = np.inf
        price.iloc[8, price.columns.get_loc("Low")] = -np.inf
        _same_excursions(_trades(price, n_trades=12, seed=6), price)

    def test_overlapping_and_unsorted_trades(self):
        price = _prices(n_bars=400)
        rng = np.random.default_rng(9)
        entries = rng.integers(0, 380, 60)
        exits = entries + rng.integers(0, 200, 60)
        exits = np.minimum(exits, 399)
        log = pd.DataFrame(
            {
                "entry_date": price.index[entries],
                "exit_date": price.index[exits],
                "direction": rng.choice(["long", "short"], 60),
                "entry_price": rng.uniform(50, 150, 60),
            }
        )
        _same_excursions(log, price)

    @pytest.mark.parametrize(
        "case", ["unsorted index", "string dates", "other unit", "nat", "no high"]
    )
    def test_inputs_off_the_common_path(self, case):
        # The first four are declined by the located path and sliced per
        # trade; the last is located and raises the same KeyError.
        price = _prices(n_bars=60)
        log = _trades(price, n_trades=15, seed=3)
        if case == "unsorted index":
            price = price.iloc[::-1]
        elif case == "string dates":
            log["entry_date"] = log["entry_date"].dt.strftime("%Y-%m-%d")
        elif case == "other unit":
            log["exit_date"] = log["exit_date"].astype("datetime64[s]")
        elif case == "nat":
            log.loc[3, "exit_date"] = pd.NaT
        elif case == "no high":
            price = price.drop(columns="High")
        _same_excursions(log, price)

    def test_no_high_column_is_not_read_when_no_trade_is_measurable(self):
        price = _prices(n_bars=20).drop(columns=["High", "Low"])
        log = _trades(_prices(n_bars=20), n_trades=3)
        log["entry_price"] = np.nan
        out = _same_excursions(log, price)
        assert out["mae_pct"].isna().all()

    def test_planted_excursions(self):
        index = pd.bdate_range("2024-01-01", periods=6)
        price = pd.DataFrame(
            {
                "High": [10.0, 12.0, 15.0, 11.0, 9.0, 10.0],
                "Low": [9.0, 8.0, 14.0, 6.0, 8.0, 9.0],
            },
            index=index,
        )
        log = pd.DataFrame(
            {
                "entry_date": [index[1], index[3]],
                "exit_date": [index[3], index[5]],
                "direction": ["long", "short"],
                "entry_price": [10.0, 10.0],
            }
        )
        out = _same_excursions(log, price)
        # Long over bars 1-3: high 15, low 6. Short over 3-5: high 11, low 6.
        assert out["mfe_pct"].tolist() == [50.0, 40.0]
        assert out["mae_pct"].tolist() == [-40.0, -10.0]

    def test_null_case_a_flat_price_never_moves(self):
        index = pd.bdate_range("2024-01-01", periods=30)
        price = pd.DataFrame({"High": 100.0, "Low": 100.0}, index=index)
        log = _trades(price.assign(Close=100.0), n_trades=8, seed=2)
        log["entry_price"] = 100.0
        out = _same_excursions(log, price)
        assert (out[["mae_pct", "mfe_pct"]].abs() == 0.0).all().all()


# ── exposure_stats ───────────────────────────────────────────────────────


def _same_exposure(signal: pd.Series, log: Optional[pd.DataFrame], caplog=None):
    try:
        expected = _reference_exposure_stats(signal, log)
    except Exception as error:  # noqa: BLE001
        with pytest.raises(type(error)):
            exposure_stats(signal, log)
        return None
    actual = exposure_stats(signal, log)
    assert actual == expected
    for key in actual:
        assert type(actual[key]) is type(expected[key]), key
    return actual


class TestExposureStatsMatchesThePerTradeLookup:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_engine_shaped_log_is_identical(self, seed):
        price = _prices(seed=seed)
        log = _trades(price, seed=seed)
        signal = _executed(price, log)
        assert isinstance(diagnostics._holding_bars(signal.index, log), np.ndarray)
        out = _same_exposure(signal, log)
        assert out["avg_holding_period_bars"] is not None

    def test_tz_aware_index(self):
        price = _prices(n_bars=200, tz="UTC")
        log = _trades(price, n_trades=40)
        _same_exposure(_executed(price, log), log)

    def test_no_log_empty_log_and_empty_signal(self):
        price = _prices(n_bars=20)
        signal = pd.Series(0.0, index=price.index)
        _same_exposure(signal, None)
        _same_exposure(signal, pd.DataFrame(columns=["entry_date", "exit_date"]))
        _same_exposure(pd.Series(dtype=float), _trades(price, n_trades=3))

    def test_one_trade_and_the_first_and_last_bar(self):
        price = _prices(n_bars=25)
        index = price.index
        log = pd.DataFrame({"entry_date": [index[0]], "exit_date": [index[-1]]})
        out = _same_exposure(pd.Series(1.0, index=index), log)
        assert out["avg_holding_period_bars"] == 24.0

    def test_dates_absent_from_the_index_are_skipped(self):
        price = _prices(n_bars=40)
        index = price.index
        log = pd.DataFrame(
            {
                "entry_date": [
                    index[2],
                    index[2] + pd.Timedelta(hours=3),
                    index[10],
                    pd.NaT,
                ],
                "exit_date": [
                    index[6],
                    index[8],
                    index[-1] + pd.Timedelta(days=3),
                    index[12],
                ],
            }
        )
        out = _same_exposure(pd.Series(1.0, index=index), log)
        assert out["avg_holding_period_bars"] == 4.0

    def test_nat_on_the_index(self):
        index = pd.DatetimeIndex(
            list(pd.bdate_range("2024-01-01", periods=5)) + [pd.NaT]
        )
        log = pd.DataFrame(
            {"entry_date": [index[0], index[5]], "exit_date": [index[5], index[5]]}
        )
        _same_exposure(pd.Series(1.0, index=index), log)

    def test_duplicate_timestamps_warn_and_skip_as_before(self, caplog):
        index = pd.bdate_range("2024-01-01", periods=10)
        index = index.insert(4, index[3])
        log = pd.DataFrame(
            {
                "entry_date": [index[0], index[3], index[5]],
                "exit_date": [index[2], index[6], index[9]],
            }
        )
        signal = pd.Series(1.0, index=index)
        with caplog.at_level(logging.WARNING):
            out = _same_exposure(signal, log)

        def emitted(name):
            return [r.getMessage() for r in caplog.records if r.name == name]

        before = emitted(logger.name)
        assert len(before) == 1
        assert emitted(diagnostics.logger.name) == before
        assert out["avg_holding_period_bars"] == 3.0

    def test_missing_date_columns_and_other_dtypes(self):
        price = _prices(n_bars=30)
        log = _trades(price, n_trades=6)
        signal = _executed(price, log)
        _same_exposure(signal, log.drop(columns="exit_date"))
        strings = log.assign(entry_date=log["entry_date"].dt.strftime("%Y-%m-%d"))
        _same_exposure(signal, strings)
        _same_exposure(
            signal, log.assign(exit_date=log["exit_date"].astype("datetime64[s]"))
        )
        _same_exposure(signal.reset_index(drop=True), log)

    def test_planted_holding_period(self):
        index = pd.bdate_range("2024-01-01", periods=12)
        log = pd.DataFrame(
            {
                "entry_date": [index[0], index[4], index[7]],
                "exit_date": [index[3], index[5], index[11]],
            }
        )
        out = _same_exposure(pd.Series(1.0, index=index), log)
        assert out["avg_holding_period_bars"] == round((3 + 1 + 4) / 3, 2)

    def test_null_case_no_trade_located(self):
        index = pd.bdate_range("2024-01-01", periods=12)
        elsewhere = pd.bdate_range("2030-01-01", periods=2)
        log = pd.DataFrame({"entry_date": [elsewhere[0]], "exit_date": [elsewhere[1]]})
        out = _same_exposure(pd.Series(0.0, index=index), log)
        assert out["avg_holding_period_bars"] is None

    def test_non_finite_signal_still_refused(self):
        index = pd.bdate_range("2024-01-01", periods=3)
        with pytest.raises(ValidationError):
            exposure_stats(pd.Series([1.0, np.nan, 0.0], index=index))
