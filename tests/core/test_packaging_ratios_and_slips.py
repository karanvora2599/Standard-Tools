"""
Regression tests for the three closing items of the full-codebase audit:

  * #150 packaging — the native extension is part of the normal build path
  * #115/#116 — one canonical unit and definition per FinancialRatios field
  * the slip audit — fixes that a parallel path could reach around

The slip tests are the ones worth reading. Two of the four slips were defects
in code written *during* this audit, which is the point: a fix is not finished
when the function it targets is correct, only when no sibling path does the
same job unguarded.
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
IDX = pd.date_range("2023-01-02", periods=120, freq="B")


def _ohlcv(close_values=None, n=120):
    close = pd.Series(
        close_values if close_values is not None else np.linspace(100.0, 140.0, n),
        index=IDX[:n],
    )
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=close.index,
    )


# ── #150 packaging ──────────────────────────────────────────────────────


class TestNativeExtensionIsPartOfTheBuild:
    """
    `pip install .` used a pure-Python backend, so it produced a package
    WITHOUT _sqt_core no matter what the machine could compile — "installed
    Standard Tools" could mean two materially different runtimes and nothing
    in the install output said which one you had.
    """

    def _pyproject(self):
        return (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    def _cmakelists(self):
        return (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    def test_build_backend_can_compile(self):
        text = self._pyproject()
        assert "scikit_build_core.build" in text
        assert "flit_core.buildapi" not in text

    def test_wheel_packages_point_at_src_layout(self):
        """Without this the backend looks for a top-level package directory
        and silently builds an empty wheel."""
        assert 'wheel.packages = ["src/standard_quant_tools"]' in self._pyproject()

    def test_release_build_type_is_pinned(self):
        """A debug build of this extension is several times slower, which
        would silently undo the entire reason it exists."""
        assert 'cmake.build-type = "Release"' in self._pyproject()

    def test_an_install_rule_exists_for_the_extension(self):
        """A wheel is staged in an isolated directory and carries only what
        CMake INSTALLS. Without the rule, scikit-build-core produced a wheel
        with no extension in it — a silent pure-Python package that looked
        like a successful native build."""
        cpp = (
            REPO_ROOT / "src" / "standard_quant_tools" / "_cpp" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        assert "install(TARGETS _sqt_core" in cpp
        assert "DESTINATION standard_quant_tools" in cpp

    def test_a_missing_toolchain_degrades_instead_of_failing(self):
        """
        _sqt_core is an OPTIONAL fast path — every function it accelerates has
        a Python fallback. Requiring a C++ compiler to install would turn an
        accelerator into a hard dependency, which is the opposite of what
        shipping it in the build was for.
        """
        text = self._cmakelists()
        assert "LANGUAGES NONE" in text
        assert "enable_language(CXX OPTIONAL)" in text
        assert "CMAKE_CXX_COMPILER_LOADED" in text

    def test_check_language_is_not_used(self):
        """
        check_language() runs a separate try-compile that does not inherit the
        Visual Studio generator's toolchain discovery, so it reported "no
        compiler" on a machine with a working MSVC install and broke the
        ordinary developer configure. Verified against the previous revision
        in the same shell: the old form found MSVC 19.44 while check_language
        did not.
        """
        assert "check_language(CXX)" not in self._cmakelists()

    def test_strict_mode_exists_for_ci(self):
        """A silent skip in CI means a green build that quietly tested only
        the fallback path."""
        assert "SQT_REQUIRE_NATIVE" in self._cmakelists()


# ── #115/#116 financial ratios ──────────────────────────────────────────


class TestFinancialRatioContract:
    """
    The shared field names implied an interchangeability that did not exist:
    yfinance reports `debtToEquity` as a PERCENTAGE (150.5) while Polygon
    computes a plain RATIO (1.505), so `debt_equity_max=2.0` admitted nearly
    every company on one provider and nearly none on the other.
    """

    def test_every_field_has_a_declared_unit(self):
        from standard_quant_tools.data.base import FinancialRatios
        from standard_quant_tools.data.ratios import CANONICAL_UNITS

        declared = set(CANONICAL_UNITS)
        modelled = set(FinancialRatios.model_fields) - {"definition_notes"}
        assert modelled == declared, "every field needs a stated unit"

    def test_every_field_has_a_declared_formula(self):
        from standard_quant_tools.data.ratios import (
            CANONICAL_UNITS,
            FIELD_DEFINITIONS,
        )

        assert set(FIELD_DEFINITIONS) == set(CANONICAL_UNITS)

    def test_percent_conversion_is_exact(self):
        from standard_quant_tools.data.ratios import percent_to_fraction

        assert percent_to_fraction(150.5) == pytest.approx(1.505)
        assert percent_to_fraction(15.0) == pytest.approx(0.15)
        assert percent_to_fraction(None) is None

    def test_conversion_is_unconditional_not_magnitude_based(self):
        """
        A magnitude test would silently rewrite a genuine 1500% return on
        equity, which companies with very small equity really do report. The
        conversion is driven by the PROVIDER's documented unit instead.
        """
        from standard_quant_tools.data.ratios import percent_to_fraction

        assert percent_to_fraction(0.15) == pytest.approx(0.0015)

    def test_implausible_values_are_reported_not_corrected(self):
        """
        yfinance changed dividendYield from a fraction to a percentage between
        releases, so which arrives depends on the installed version. A warning
        surfaces that; a silent rescale would corrupt a real outlier.
        """
        from standard_quant_tools.data.base import FinancialRatios
        from standard_quant_tools.data.ratios import implausible_value_warnings

        ratios = FinancialRatios(return_on_equity=15.0)
        warnings = implausible_value_warnings(ratios)
        assert warnings and "return_on_equity" in warnings[0]
        assert ratios.return_on_equity == 15.0, "value must be left unchanged"

    def test_canonical_values_produce_no_warning(self):
        from standard_quant_tools.data.base import FinancialRatios
        from standard_quant_tools.data.ratios import implausible_value_warnings

        clean = FinancialRatios(
            return_on_equity=0.15, profit_margins=0.25, dividend_yield=0.015
        )
        assert implausible_value_warnings(clean) == []

    def test_definition_differences_are_declared_not_hidden(self):
        """
        A unit difference is mechanical and is converted. A DEFINITION
        difference is not: Polygon derives debt_to_equity from total
        LIABILITIES (payables, deferred revenue, leases) where Bloomberg uses
        total DEBT, so it reads systematically higher for reasons unrelated to
        leverage. The value is kept and the basis is stated.
        """
        from standard_quant_tools.data.base import FinancialRatios

        assert "definition_notes" in FinancialRatios.model_fields
        assert FinancialRatios().definition_notes == {}

    def test_yfinance_debt_to_equity_is_converted(self):
        """The one yfinance field whose unit is not canonical."""
        source = (
            REPO_ROOT / "src" / "standard_quant_tools" / "data" / "yfinance_provider.py"
        ).read_text(encoding="utf-8")
        assert 'percent_to_fraction(info.get("debtToEquity"))' in source

    def test_polygon_declares_its_liabilities_basis(self):
        source = (
            REPO_ROOT / "src" / "standard_quant_tools" / "data" / "polygon_provider.py"
        ).read_text(encoding="utf-8")
        assert "total LIABILITIES" in source
        assert "definition_notes=notes" in source


# ── the slip audit ──────────────────────────────────────────────────────


class TestFixesDoNotSlip:
    """
    Each of these is a place where a fix was correct in the function it
    targeted and reachable around through a sibling path. Two were introduced
    during this audit.
    """

    def test_the_grid_batch_path_enforces_positive_prices(self):
        """
        run_strategy gained the positive-price contract; backtest_grid's C++
        batch path kept only the finiteness check, so a Close of -5.0 ran
        through an entire parameter sweep and returned a full results table —
        because -5.0 is perfectly finite.
        """
        from standard_quant_tools.backtest.engine import backtest_grid

        bad = np.linspace(100.0, 140.0, 120)
        bad[10] = -5.0
        with pytest.raises(ValidationError, match="non-positive"):
            backtest_grid(
                _ohlcv(bad),
                "sma_crossover",
                {"fast_period": [5], "slow_period": [20]},
                n_workers=1,
            )

    def test_the_grid_enforces_the_strategy_parameter_contract(self):
        """Wrapping STRATEGY_REGISTRY rather than each call site is what makes
        this hold through the ProcessPoolExecutor worker too."""
        from standard_quant_tools.backtest.engine import backtest_grid

        with pytest.raises(ValidationError, match="must be >= 1"):
            backtest_grid(
                _ohlcv(), "momentum_timeseries", {"lookback": [-20]}, n_workers=1
            )

    def test_a_nan_trade_return_is_not_a_breakeven(self):
        """
        A hole the breakeven fix itself opened. Moving from `~is_win` to
        explicit `> 0` / `< 0` tests made NaN satisfy NEITHER, so a NaN trade
        was silently bucketed with the flat trades — counted in the win-rate
        denominator, excluded from both averages, and treated as a
        streak-breaker. The earlier two-way split had at least called it a
        loss. Neither is right.
        """
        from standard_quant_tools.metrics.diagnostics import trade_expectancy

        with pytest.raises(ValidationError, match="non-finite"):
            trade_expectancy(pd.DataFrame({"return_pct": [1.0, float("nan"), -1.0]}))

    def test_ordinary_trade_logs_still_summarize(self):
        from standard_quant_tools.metrics.diagnostics import trade_expectancy

        result = trade_expectancy(pd.DataFrame({"return_pct": [1.0, 0.0, -1.0]}))
        assert result["max_consecutive_losses"] == 1

    def test_the_unwrapped_registry_is_marked_internal(self):
        """
        _RAW_STRATEGIES must exist as the input to the wrapping step, so it
        cannot be removed — but calling out of it skips the validation that
        makes a negative lookback (direct look-ahead) unreachable. It is
        private and prominently marked.
        """
        from standard_quant_tools.backtest import strategies

        assert not hasattr(strategies, "RAW_STRATEGIES"), "must stay private"
        source = (
            REPO_ROOT / "src" / "standard_quant_tools" / "backtest" / "strategies.py"
        ).read_text(encoding="utf-8")
        marker = source.index("_RAW_STRATEGIES = {")
        preamble = source[max(0, marker - 900) : marker]
        assert "INTERNAL ONLY" in preamble
        assert "look-ahead" in preamble

    def test_every_public_strategy_goes_through_validation(self):
        from standard_quant_tools.backtest.strategies import (
            _RAW_STRATEGIES,
            STRATEGY_REGISTRY,
        )

        assert set(STRATEGY_REGISTRY) == set(_RAW_STRATEGIES)
        for name, fn in STRATEGY_REGISTRY.items():
            assert fn is not _RAW_STRATEGIES[name], f"{name} is unwrapped"

    def test_a_forgotten_interval_warns_before_flattening(self, caplog):
        """
        The `interval` default is "1d" for back-compat, so a caller who simply
        FORGETS it gets the old collapsing behaviour on intraday data — the
        exact bug the function was rewritten to fix, reachable again by
        omission rather than intent.
        """
        import logging

        from standard_quant_tools.data._cache import _normalize_ohlcv_index

        frame = pd.DataFrame(
            {"Close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-06-03 15:00", periods=3, freq="h"),
        )
        with caplog.at_level(logging.WARNING):
            _normalize_ohlcv_index(frame)  # interval omitted
        assert any("time component" in r.message for r in caplog.records)

    def test_a_genuine_daily_frame_does_not_warn(self, caplog):
        import logging

        from standard_quant_tools.data._cache import _normalize_ohlcv_index

        frame = pd.DataFrame(
            {"Close": [1.0, 2.0]},
            index=pd.date_range("2024-06-03", periods=2, freq="D"),
        )
        with caplog.at_level(logging.WARNING):
            _normalize_ohlcv_index(frame, "1d")
        assert not [r for r in caplog.records if "time component" in r.message]

    def test_portfolio_metrics_inherits_build_portfolio_hygiene(self):
        """A sibling entry point that would otherwise skip the checks added to
        build_portfolio."""
        from standard_quant_tools.portfolio.portfolio import portfolio_metrics

        rng = np.random.default_rng(0)
        rets = pd.DataFrame(rng.normal(0.0005, 0.012, (300, 3)), columns=list("ABC"))
        rets.iloc[5, 0] = float("inf")
        with pytest.raises(ValidationError, match="infinite"):
            portfolio_metrics(rets, [1 / 3] * 3)
