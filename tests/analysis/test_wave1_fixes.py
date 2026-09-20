"""
Four small dishonesties in `analysis`, each planted.

`calculate_beta` returned an R-squared of 0.0 for a constant series
against its own NaN-not-zero policy; `pca_returns` raised a raw
ValueError where every sibling raises ValidationError; the cointegration
scan's pure-Python fallback filled the intercept with NaN and the lag
with 0, a real-looking value for "not reported"; and nothing pinned that
the C++ and Python half-lives agree.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis import cointegration, regression
from standard_quant_tools.analysis.cointegration import (
    cointegration_test,
    scan_cointegrated_pairs,
)
from standard_quant_tools.analysis.pca import pca_returns
from standard_quant_tools.analysis.regression import calculate_beta
from standard_quant_tools.error import ValidationError


def _pairs_frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n)
    b = pd.Series(np.cumsum(rng.normal(size=n)) + 100.0, index=index)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(scale=0.5)
    a = 5.0 + 1.5 * b + noise
    c = pd.Series(np.cumsum(rng.normal(size=n)) + 50.0, index=index)
    return pd.DataFrame({"A": a, "B": b, "C": c})


class TestBetaRSquared:
    def test_a_constant_series_has_no_variance_to_explain(self, monkeypatch):
        monkeypatch.setattr(regression, "HAS_CPP", False)
        index = pd.bdate_range("2023-01-02", periods=30)
        constant = pd.Series(0.01, index=index)
        market = pd.Series(np.random.default_rng(0).normal(size=30), index=index)
        result = calculate_beta(constant, market)
        assert np.isnan(result["r_squared"])
        assert result["beta"] == pytest.approx(0.0, abs=1e-12)


class TestPcaRefusals:
    def test_they_are_validation_errors_like_every_sibling(self):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(50, 3)), columns=list("xyz")
        )
        with pytest.raises(ValidationError, match="method must be"):
            pca_returns(frame, method="bogus")
        with pytest.raises(ValidationError, match="at least 2 observations"):
            pca_returns(frame.head(1))
        # Still a ValueError for anyone who caught that before.
        with pytest.raises(ValueError):
            pca_returns(frame, method="bogus")


class TestTheScanFallback:
    def test_the_intercept_is_the_ols_identity_and_the_lag_is_unknown(
        self, monkeypatch
    ):
        monkeypatch.setattr(cointegration, "HAS_CPP", False)
        frame = _pairs_frame()
        scanned = scan_cointegrated_pairs(frame, pairs=[("A", "B")])
        row = scanned.iloc[0]
        single = cointegration_test(frame["A"], frame["B"])
        assert row["hedge_ratio"] == pytest.approx(single["hedge_ratio"])
        expected_intercept = (
            frame["A"].mean() - single["hedge_ratio"] * frame["B"].mean()
        )
        assert row["intercept"] == pytest.approx(expected_intercept)
        assert abs(row["intercept"] - 5.0) < 2.0
        assert row["optimal_lag"] == -1

    @pytest.mark.skipif(not cointegration.HAS_CPP, reason="C++ extension not built")
    def test_the_two_backends_agree_on_the_half_life(self):
        frame = _pairs_frame()
        batch = scan_cointegrated_pairs(frame, pairs=[("A", "B"), ("A", "C")])
        for (a, b), (_index, row) in zip([("A", "B"), ("A", "C")], batch.iterrows()):
            single = cointegration_test(frame[a], frame[b])
            assert row["hedge_ratio"] == pytest.approx(single["hedge_ratio"], rel=1e-6)
            if np.isfinite(single["half_life_days"]):
                assert row["half_life_days"] == pytest.approx(
                    single["half_life_days"], rel=1e-3
                )
