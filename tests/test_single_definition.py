"""
The things that used to be defined in many places must stay defined in one.

Consolidating duplicates is a one-off edit; keeping them consolidated is not.
Every copy this test guards was written by someone who needed a normal CDF
or a trading-day count and reasonably wrote one, because nothing told them
where the existing one lived. This test is what tells them.

It is deliberately a source scan rather than an import check. Importing
proves the names resolve, which they would even with nine copies -- the
question here is how many definitions exist, and that is a property of the
files.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "standard_quant_tools"


def _python_files():
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def _definitions(pattern: str, exclude: str):
    """Files defining `pattern` at module level, excluding the canonical home."""
    hits = []
    for path in _python_files():
        if path.name == exclude:
            continue
        if re.search(pattern, path.read_text(encoding="utf-8"), re.M):
            hits.append(path.relative_to(SRC).as_posix())
    return sorted(hits)


class TestSpecialFunctionsAreDefinedOnce:
    """`_special.py` replaced fourteen copies of six functions. Two of the
    pairs had drifted, and both drifts were at the edge of the domain: one
    `_norm_ppf` returned +inf where the other raised, and one `_f_sf`
    returned a p-value of 0.0 for a test with zero denominator degrees of
    freedom where the other returned 1.0."""

    @pytest.mark.parametrize(
        "name",
        ["_norm_cdf", "_norm_pdf", "_norm_ppf", "_betainc", "_betacf", "_f_sf"],
    )
    def test_no_module_reimplements_it(self, name):
        offenders = _definitions(rf"^def {name}\(", "_special.py")
        assert offenders == [], (
            f"{name} is implemented in {offenders}. It lives in `_special.py` "
            f"-- import it (`from standard_quant_tools._special import ...`) "
            f"and alias it if the private name is needed locally."
        )

    def test_the_canonical_ones_exist(self):
        from standard_quant_tools import _special

        for name in ("norm_cdf", "norm_pdf", "norm_ppf", "betainc", "betacf", "f_sf"):
            assert callable(getattr(_special, name))


class TestSpecialFunctionsAreStillCorrect:
    """Consolidation is only safe if the surviving implementation is right.
    scipy is not a dependency, so this skips where it is absent rather than
    vendoring a table of expected values that would itself go stale."""

    def test_against_scipy(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        from standard_quant_tools import _special as S

        rng = np.random.default_rng(0)

        for x in rng.normal(0, 3, 500):
            assert S.norm_cdf(x) == pytest.approx(scipy_stats.norm.cdf(x), abs=1e-12)
            assert S.norm_pdf(x) == pytest.approx(scipy_stats.norm.pdf(x), abs=1e-12)

        for p in rng.uniform(1e-9, 1 - 1e-9, 500):
            # Acklam's stated accuracy is ~1.15e-9.
            assert S.norm_ppf(p) == pytest.approx(scipy_stats.norm.ppf(p), abs=5e-9)

        for _ in range(500):
            d1 = int(rng.integers(1, 40))
            d2 = int(rng.integers(1, 200))
            f = float(rng.uniform(1e-3, 50))
            assert S.f_sf(f, d1, d2) == pytest.approx(
                scipy_stats.f.sf(f, d1, d2), abs=1e-10
            )

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_norm_ppf_refuses_the_bounds_rather_than_returning_infinity(self, bad):
        """The divergence that motivated picking this variant. At 1e17 trials
        `1 - 1/n` collapses to exactly 1.0, and the copy that returned +inf
        fed it straight into a deflated Sharpe ratio."""
        from standard_quant_tools._special import norm_ppf
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError, match="strictly in"):
            norm_ppf(bad)

    def test_f_sf_returns_one_not_zero_for_degenerate_degrees_of_freedom(self):
        """The other divergence. A p-value of 0.0 is maximum significance,
        and this test has nothing to measure."""
        from standard_quant_tools._special import f_sf

        assert f_sf(2.0, 3, 0) == 1.0
        assert f_sf(2.0, 0, 50) == 1.0
        assert f_sf(-1.0, 3, 50) == 1.0


class TestTradingDaysIsDefinedOnce:
    """252 was defined nine times under two spellings, and the copy that
    named itself canonically had no importers."""

    def test_no_module_defines_the_literal(self):
        offenders = _definitions(r"^TRADING_DAYS(_PER_YEAR)? = 252$", "constants.py")
        assert offenders == [], (
            f"252 is defined in {offenders}. It lives in `constants.py` as "
            f"TRADING_DAYS_PER_YEAR -- import it and alias it locally if the "
            f"short name is part of that module's public API."
        )

    def test_every_surviving_name_is_the_same_object(self):
        """Five of the eight were in their module's `__all__`, so they are
        public API and had to keep resolving."""
        import importlib

        from standard_quant_tools.constants import TRADING_DAYS_PER_YEAR

        for module in (
            "analysis.derivatives",
            "analysis.diagnostics",
            "analysis.inference",
            "backtesting.overfitting",
            "backtesting.trade_analysis",
            "delta_one.hedging",
            "portfolio.construction",
        ):
            mod = importlib.import_module(f"standard_quant_tools.{module}")
            assert mod.TRADING_DAYS == TRADING_DAYS_PER_YEAR, module

        from standard_quant_tools.delta_one.daycount import TRADING_DAYS_PER_YEAR as d

        assert d == TRADING_DAYS_PER_YEAR


class TestDeflatedSharpeFormulasAreDefinedOnce:
    """Two `deflated_sharpe_ratio` functions, one taking summary statistics
    and one taking a return series, each with its own copy of the two
    formulas that make a deflated Sharpe deflated. They agreed to rounding,
    which is what a duplicate looks like right up until it does not."""

    def test_euler_mascheroni_is_defined_once(self):
        offenders = _definitions(r"^_?EULER_MASCHERONI = 0\.", "constants.py")
        assert offenders == [], f"defined in {offenders}; it lives in constants.py"

    def test_the_expected_maximum_is_not_written_out_twice(self):
        """The `(1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N e))` expansion
        should appear only inside `expected_max_sharpe`."""
        offenders = []
        for path in _python_files():
            text = path.read_text(encoding="utf-8")
            if re.search(r"1\.0 / \(n_trials \* math\.e\)", text):
                offenders.append(path.relative_to(SRC).as_posix())
        assert offenders == ["backtest/robustness.py"], (
            f"the expected-maximum expansion appears in {offenders}; it belongs "
            f"in `backtest.robustness.expected_max_sharpe` alone"
        )

    def test_both_entry_points_agree(self):
        """The wrapper takes returns and the core takes summary statistics, so
        this drives them to the same place by hand and checks they land
        together. They must, now that they share both formulas."""
        import math

        import pandas as pd

        from standard_quant_tools.backtest.robustness import deflated_sharpe_ratio
        from standard_quant_tools.backtesting.overfitting import (
            deflated_sharpe_ratio as from_returns,
        )

        rng = np.random.default_rng(4)
        for n, trials in [(300, 10), (500, 50), (750, 200), (1000, 5), (1200, 1000)]:
            series = pd.Series(rng.normal(0.0006, 0.011, n))
            values = series.to_numpy()
            mean, std = values.mean(), values.std(ddof=1)
            centred = (values - mean) / std

            wrapped = from_returns(series, n_trials=trials)
            core = deflated_sharpe_ratio(
                observed_sharpe=mean / std,
                sharpe_trials_std=math.sqrt(1.0 / n),
                n_trials=trials,
                n_obs=n,
                skew=float((centred**3).mean()),
                kurtosis=float((centred**4).mean()),
            )
            # The core rounds its output to 6 places; that is the only gap.
            assert wrapped["deflated_sharpe_probability"] == pytest.approx(
                core["deflated_sharpe_ratio"], abs=1e-6
            )


class TestTheConsolidatedHelpersOnDegenerateInput:
    """The consolidation pass made these single points of failure for the
    whole library and pinned none of them. Every function here had zero test
    references until this class existed, and each one was wrong on an input
    a caller can produce.
    """

    def test_norm_cdf_array_handles_an_empty_array(self):
        """`np.vectorize` infers its dtype by calling the function once, so
        size-0 raised `ValueError: cannot call vectorize on size 0 inputs`.
        Reached from `multi_factor`'s scipy-absent p-value path, which is
        itself never executed in this environment."""
        from standard_quant_tools._special import norm_cdf_array

        out = norm_cdf_array([])
        assert out.shape == (0,)
        assert out.dtype == np.dtype(float)

    def test_norm_cdf_array_agrees_with_the_scalar_version(self):
        from standard_quant_tools._special import norm_cdf, norm_cdf_array

        xs = np.random.default_rng(0).normal(0, 3, 200)
        assert np.allclose(norm_cdf_array(xs), [norm_cdf(x) for x in xs])

    @pytest.mark.parametrize("block_size", [0, -1, -3])
    def test_block_indices_refuses_a_block_size_below_one(self, block_size):
        """It used to clamp up to 1, which is an IID resample returned under
        the name of a block bootstrap -- destroying exactly the serial
        correlation a caller chose this function to preserve. The agent
        boundary guards it with ge=1; the library API did not."""
        from standard_quant_tools._resampling import block_indices
        from standard_quant_tools.error import ValidationError

        rng = np.random.default_rng(0)
        with pytest.raises(ValidationError, match="block_size"):
            block_indices(10, block_size, rng)

    @pytest.mark.parametrize("n", [0, -5])
    def test_block_indices_refuses_an_empty_source(self, n):
        """`n=0` raised a bare ZeroDivisionError from inside math.ceil."""
        from standard_quant_tools._resampling import block_indices
        from standard_quant_tools.error import ValidationError

        rng = np.random.default_rng(0)
        with pytest.raises(ValidationError, match="n must be"):
            block_indices(n, 5, rng)

    def test_block_indices_still_clamps_a_block_larger_than_the_source(self):
        """Clamping DOWN is correct and documented -- it reproduces what the
        concatenating version did. Only clamping up was the bug."""
        from standard_quant_tools._resampling import block_indices

        rng = np.random.default_rng(0)
        out = block_indices(10, 999, rng)
        assert len(out) == 10
        assert out.min() >= 0 and out.max() < 10

    def test_annualized_mean_cov_scales_covariance_linearly(self):
        """THE MUTATION THAT SURVIVED 2,149 TESTS. Changing `* ppy` to
        `* sqrt(ppy)` here understates every reported portfolio volatility
        by 15.87x across 71 live calls and no test noticed, because
        optimizer weights are invariant to a uniform scaling of the
        covariance and every magnitude assertion elsewhere is computed from
        the same mutated matrix."""
        import pandas as pd

        from standard_quant_tools.portfolio.optimize import annualized_mean_cov

        rng = np.random.default_rng(0)
        frame = pd.DataFrame(rng.normal(0.0005, 0.012, (500, 4)))

        mu, cov = annualized_mean_cov(frame, 252)
        mu_1, cov_1 = annualized_mean_cov(frame, 1)

        # Linear in periods_per_year, both of them -- not sqrt, not mixed.
        assert np.allclose(mu, mu_1 * 252)
        assert np.allclose(cov, cov_1 * 252)

        # And an absolute anchor, so a uniform rescale of BOTH cannot pass.
        assert np.allclose(cov, frame.cov().to_numpy() * 252)
        assert np.allclose(mu, frame.mean().to_numpy() * 252)

        # A daily vol of 1.2% is ~19% annualized. sqrt-scaling gives 1.2%.
        annual_vol = float(np.sqrt(cov[0, 0]))
        assert 0.15 < annual_vol < 0.24, annual_vol

    def test_sharpe_standard_error_factor_uses_the_raw_fourth_moment(self):
        """THE OTHER SURVIVING MUTATION. `(kurtosis - 1.0)` -> `(kurtosis -
        3.0)` passes 1,300 tests, because every deflated-Sharpe test asserts
        an ordering or a 0..1 range and both survive a monotone transform.
        3.0 is normal here, not 0.0."""
        from standard_quant_tools.backtest.robustness import (
            sharpe_standard_error_factor,
        )

        # Normal moments: skew 0, kurtosis 3 -> factor sqrt(1 + SR^2/2).
        got = sharpe_standard_error_factor(0.1, skew=0.0, kurtosis=3.0)
        assert got == pytest.approx(np.sqrt(1.0 + (3.0 - 1.0) / 4.0 * 0.01))

        # Negative skew widens it; fat tails widen it.
        assert sharpe_standard_error_factor(0.1, -1.0, 3.0) > got
        assert sharpe_standard_error_factor(0.1, 0.0, 9.0) > got
