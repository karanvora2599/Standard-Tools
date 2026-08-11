"""Tests for modeling.validation.walk_forward.WalkForwardSplit: fold
boundaries, embargo enforcement, no train/test overlap."""

import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.validation.walk_forward import WalkForwardSplit


@pytest.fixture
def dates() -> pd.Index:
    return pd.Index(pd.date_range("2020-01-01", periods=100, freq="B"))


class TestWalkForwardSplit:
    def test_first_fold_starts_at_zero(self, dates):
        splitter = WalkForwardSplit(train_window=50, test_window=10, embargo=0)
        train_pos, test_pos = next(splitter.split(dates))
        assert train_pos[0] == 0
        assert train_pos[-1] == 49
        assert test_pos[0] == 50
        assert test_pos[-1] == 59

    def test_embargo_gap_excluded_from_both_train_and_test(self, dates):
        splitter = WalkForwardSplit(train_window=50, test_window=10, embargo=5)
        train_pos, test_pos = next(splitter.split(dates))
        assert train_pos[-1] == 49
        assert test_pos[0] == 55  # 49 + 1 (embargo start) + 5 (embargo length)
        gap = set(range(train_pos[-1] + 1, test_pos[0]))
        assert len(gap) == 5
        assert not gap & set(train_pos)
        assert not gap & set(test_pos)

    def test_folds_walk_forward_by_test_window(self, dates):
        splitter = WalkForwardSplit(train_window=50, test_window=10, embargo=0)
        folds = list(splitter.split(dates))
        assert len(folds) >= 2
        assert (
            folds[1][0][0] == folds[0][0][0] + 10
        )  # second fold's train starts one test_window later

    def test_no_train_test_overlap_across_all_folds(self, dates):
        splitter = WalkForwardSplit(train_window=30, test_window=10, embargo=2)
        for train_pos, test_pos in splitter.split(dates):
            assert not set(train_pos.tolist()) & set(test_pos.tolist())

    def test_n_splits_matches_manual_count(self, dates):
        splitter = WalkForwardSplit(train_window=50, test_window=10, embargo=0)
        assert splitter.n_splits(dates) == len(list(splitter.split(dates)))

    def test_too_short_dataset_yields_zero_folds(self, dates):
        splitter = WalkForwardSplit(train_window=90, test_window=90, embargo=0)
        assert splitter.n_splits(dates) == 0
        assert list(splitter.split(dates)) == []

    def test_invalid_windows_raise(self):
        with pytest.raises(ValidationError):
            WalkForwardSplit(train_window=0, test_window=10)
        with pytest.raises(ValidationError):
            WalkForwardSplit(train_window=10, test_window=-1)
        with pytest.raises(ValidationError):
            WalkForwardSplit(train_window=10, test_window=10, embargo=-1)
