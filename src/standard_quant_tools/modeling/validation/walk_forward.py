"""
WalkForwardSplit — no existing generic time-series splitter in this
codebase to reuse (checked backtest/: only strategy-level walk-forward
backtesting, not a sklearn-style splitter), so this is genuinely new.

Yields (train_positions, test_positions) over a sorted, unique date
array, walking forward one `test_window` at a time, with an `embargo` gap
between each fold's train and test window so a feature's lookback can't
bleed across the boundary. engine.py maps these date-positions back to
panel row masks (a caller with a long entity-stacked panel passes
`panel['date'].unique()` sorted here, not the panel itself).
"""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError


class WalkForwardSplit:
    def __init__(self, train_window: int, test_window: int, embargo: int = 0):
        if train_window <= 0 or test_window <= 0:
            raise ValidationError(
                f"train_window and test_window must be > 0, got "
                f"({train_window}, {test_window})"
            )
        if embargo < 0:
            raise ValidationError(f"embargo must be >= 0, got {embargo}")
        self.train_window = train_window
        self.test_window = test_window
        self.embargo = embargo

    def split(self, dates: pd.Index) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(dates)
        start = 0
        while start + self.train_window + self.embargo + self.test_window <= n:
            train_end = start + self.train_window
            test_start = train_end + self.embargo
            test_end = test_start + self.test_window
            yield np.arange(start, train_end), np.arange(test_start, test_end)
            start += self.test_window

    def n_splits(self, dates: pd.Index) -> int:
        """How many folds `split(dates)` will actually yield — engine.py
        uses this to raise a clear error before fitting anything if the
        dataset is too short for even one fold, rather than silently
        returning zero folds' worth of metrics."""
        return sum(1 for _ in self.split(dates))
