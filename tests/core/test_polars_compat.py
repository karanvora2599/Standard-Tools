"""
Dual-backend numerical-equivalence tests: for every function in this
library that accepts Polars input (see Documentation/13_polars_support.md
for the current list), assert it returns identical results whether given
a pandas or a polars Series/DataFrame built from the same underlying data.

This file is the template for verifying each function added in future
phases of Polars support — one class per function, one fixture with the
shared input data, one test asserting pandas/polars agreement.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools._compat import HAS_POLARS

pytestmark = pytest.mark.skipif(
    not HAS_POLARS, reason="Polars dual-backend tests require polars installed"
)

if HAS_POLARS:
    import polars as pl

from standard_quant_tools.analysis.hurst import hurst_exponent


@pytest.fixture(scope="module")
def sample_data():
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 2000)


class TestHurstExponentDualBackend:
    def test_dfa_identical_across_backends(self, sample_data):
        pd_result = hurst_exponent(pd.Series(sample_data), method="dfa")
        pl_result = hurst_exponent(pl.Series(sample_data), method="dfa")
        assert pd_result == pl_result

    def test_rs_identical_across_backends(self, sample_data):
        pd_result = hurst_exponent(pd.Series(sample_data), method="rs")
        pl_result = hurst_exponent(pl.Series(sample_data), method="rs")
        assert pd_result == pl_result

    def test_identical_with_nulls_and_nans_present(self):
        """Regression coverage: pandas' .dropna() and polars'
        .drop_nulls().drop_nans() (via to_clean_numpy) must agree exactly
        on which values get dropped, not just on clean data."""
        rng = np.random.default_rng(7)
        data = rng.normal(0, 1, 500).tolist()
        data[10] = np.nan
        data[20] = None

        pd_result = hurst_exponent(pd.Series(data))
        pl_result = hurst_exponent(pl.Series(data))
        assert pd_result == pl_result

    def test_custom_min_max_window_identical_across_backends(self, sample_data):
        pd_result = hurst_exponent(
            pd.Series(sample_data), min_window=20, max_window=200
        )
        pl_result = hurst_exponent(
            pl.Series(sample_data), min_window=20, max_window=200
        )
        assert pd_result == pl_result
