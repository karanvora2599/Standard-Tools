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
