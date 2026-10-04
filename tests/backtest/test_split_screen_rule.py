"""
The backtest's split screen and the dataset build's name the same bars,
and both name a 3:2 split.

A 3:2 split moves -33.3%, below the 35% both screens used, so a backtest or
a dataset built across one compounded it as a return without a word. A fall
within 10% (on a log scale) of a 3:2 split -- 26.3% to 39.7% -- is now
named by both, through one rule (`_split_screen.screen_moves`). Warnings
only; no price is adjusted. See the CHANGELOG entry of 2026-10-04.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools import _split_screen
from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.backtest.screens import split_screen_warnings
from standard_quant_tools.data.quality import (
    detect_split_like_moves,
    nearest_split_ratio,
)


def _closes(n: int = 300, seed: int = 11, vol: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1 + rng.normal(0.0003, vol, n))


def _series(closes: np.ndarray) -> pd.Series:
    return pd.Series(closes, index=pd.bdate_range("2023-01-02", periods=len(closes)))


def _with_moves(moves: dict, n: int = 300) -> pd.Series:
    """Quiet bars with a move of `moves[i]` into bar i, the bars after it
    carrying the new level (as an unadjusted split leaves them)."""
    closes = _closes(n)
    for at, move in moves.items():
        closes[at:] *= 1.0 + move
    return _series(closes)


def _listed_dates(warning: str) -> list:
    return re.findall(r"(\d{4}-\d{2}-\d{2}) \(", warning)


class TestTheRule:
    @pytest.mark.parametrize(
        "move,named",
        [
            (-0.25, False),  # a 4:3 split's size: not listed
            (-0.20, False),  # a 5:4 split's size: not listed
            (-0.26, False),
            (-0.27, True),
            (-1 / 3, True),  # 3:2
            (-0.349, True),
            (-0.36, True),  # beyond the threshold
            (+0.30, False),  # a rise below the threshold is no forward split
            (+0.36, True),
        ],
    )
    def test_which_moves_are_named(self, move, named):
        screened = _split_screen.screen_moves(np.array([100.0, 100.0 * (1 + move)]))
        assert bool(screened.flagged[0]) is named
        assert bool(screened.by_ratio[0]) is (named and abs(move) <= 0.35)

    def test_the_band_edges(self):
        """The fall named below 35% runs from 26.3% (the price ratio
        1.5 / e^0.1) to the threshold."""
        smallest, largest = _split_screen.below_threshold_band()
        assert smallest == pytest.approx(1 - 1 / (1.5 * math.exp(-0.1)))
        assert round(smallest, 3) == 0.263 and largest == 0.35
        edge = 1.5 * math.exp(-0.1)  # the smallest price ratio named
        for factor, named in ((edge * (1 + 1e-6), True), (edge * (1 - 1e-6), False)):
            screened = _split_screen.screen_moves(np.array([100.0, 100.0 / factor]))
            assert bool(screened.flagged[0]) is named

    def test_a_bar_named_by_the_rule_is_named_3_2_by_the_build(self):
        """The rule's distance is the build's: on a fine grid across the
        whole band, every bar the rule names below the threshold is named
        after 3:2, never "not near a split ratio"."""
        prior = 97.31
        for factor in np.linspace(1.30, 1.70, 4001):
            current = prior / factor
            screened = _split_screen.screen_moves(np.array([prior, current]))
            if screened.by_ratio[0]:
                ratio, distance = nearest_split_ratio(prior / current)
                assert ratio == 1.5 and distance <= _split_screen.SPLIT_RATIO_TOLERANCE

    def test_missing_and_non_positive_closes_are_not_named_by_ratio(self):
        values = np.array([100.0, np.nan, 66.0, 0.0, 66.0, -100.0, -66.0])
        screened = _split_screen.screen_moves(values)
        assert not screened.by_ratio.any()


class TestBothScreensNameTheSameBars:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_on_bars_with_moves_of_every_kind(self, seed):
        rng = np.random.default_rng(seed)
        positions = sorted(rng.choice(np.arange(20, 280), size=5, replace=False))
        sizes = [-1 / 3, -0.30, -0.5, -0.9, -0.24, 0.30, 0.6, -0.27]
        moves = {int(p): float(rng.choice(sizes)) for p in positions}
        prices = _with_moves(moves)
        [warning] = split_screen_warnings(prices, False) or [""]
        build = [m["date"] for m in detect_split_like_moves(prices)]
        assert _listed_dates(warning) == build[:5]

    def test_quiet_bars_name_nothing_in_either(self):
        prices = _series(_closes(2000, vol=0.02))
        assert split_screen_warnings(prices, None) == []
        assert detect_split_like_moves(prices) == []


class TestTheBacktestWarning:
    def test_a_3_2_split_is_named_in_the_backtest(self):
        prices = _with_moves({150: -1 / 3})
        frame = pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": 1e6,
            }
        )
        frame.attrs["adjusted"] = False
        result = run_strategy(frame, pd.Series(1.0, index=frame.index))
        [warning] = [w for w in result["warnings"] if w.startswith("SPLIT SCREEN")]
        date = prices.index[150].date()
        move = prices.pct_change(fill_method=None).iloc[150]
        assert -0.35 < move < -0.30
        assert warning.startswith(
            "SPLIT SCREEN: 1 bar(s) move more than 35% close to close, or fall "
            "26% to 35% as a 3:2 split does (-33%): "
            f"{date} ({move:+.1%}, near a 3:2 split). The provider reports "
            "adjusted=False, so a split is a real bar here"
        )

    def test_null_moves_beyond_35_keep_the_old_words(self):
        """Without a 3:2-sized bar the warning is the one it always was."""
        prices = _with_moves({60: -0.5, 200: -0.9})
        [warning] = split_screen_warnings(prices, True)
        moves = prices.pct_change(fill_method=None)
        assert warning == (
            "SPLIT SCREEN: 2 bar(s) move more than 35% close to close: "
            f"{prices.index[60].date()} ({moves.iloc[60]:+.1%}), "
            f"{prices.index[200].date()} ({moves.iloc[200]:+.1%}). The provider "
            "reports adjusted=True, so this is either a genuine move or a bad "
            "print; check the bar before trusting the result."
        )

    def test_the_portfolio_engine_names_it_per_ticker(self):
        prices = _with_moves({150: -1 / 3}, n=260)
        other = _series(_closes(260, seed=5))
        price_data = {
            t: pd.DataFrame(
                {
                    "Open": p,
                    "High": p * 1.01,
                    "Low": p * 0.99,
                    "Close": p,
                    "Volume": 1e6,
                }
            )
            for t, p in (("AAA", prices), ("BBB", other))
        }
        weights = pd.DataFrame({"AAA": [0.5], "BBB": [0.5]}, index=[prices.index[0]])
        result = run_portfolio_simulation(price_data, weights)
        named = [w for w in result["warnings"] if "SPLIT SCREEN" in w]
        assert len(named) == 1 and named[0].startswith("AAA: SPLIT SCREEN: 1 bar(s)")
        assert "near a 3:2 split" in named[0]
