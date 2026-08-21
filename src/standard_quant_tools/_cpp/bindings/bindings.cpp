#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "sqt/hurst.hpp"
#include "sqt/indicators.hpp"
#include "sqt/cointegration.hpp"
#include "sqt/backtest.hpp"
#include "sqt/rolling_regression.hpp"
#include "sqt/monte_carlo.hpp"
#include "sqt/garch.hpp"
#include "sqt/signal_state_machines.hpp"
#include "sqt/numerics.hpp"

namespace py = pybind11;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Forces the input to be a C-contiguous float64 array -- but c_style and
// forcecast alone say nothing about ndim, so a caller passing a 2-D array
// (or any other shape) previously flowed through silently, flattened, and
// misinterpreted as if it were the 1-D series every kernel below assumes.
// require_1d() (called at the top of every lambda taking an Array1D
// parameter) is what actually enforces the "1-D" half of this type's name.
using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;

static void require_1d(const Array1D& arr, const char* name) {
    if (arr.ndim() != 1)
        throw std::invalid_argument(
            std::string(name) + " must be a 1-D array, got ndim=" +
            std::to_string(arr.ndim()));
}

// ── Strict/zero-copy binding support ────────────────────────────────────────
//
// Array1D (above) uses `forcecast`: any input that isn't already exactly
// float64 + C-contiguous gets silently copied before the kernel ever sees
// it -- convenient, but for a caller who already has a correctly-typed
// contiguous array and is calling one of the highest-value large-array
// entry points (rolling_beta, rolling_factor_loadings,
// simulate_forward_paths, batch_run_strategy, technical_indicators,
// rolling_hurst), that copy is a real, avoidable cost. The `_zerocopy`
// sibling bindings below take an untyped py::array and validate dtype/
// layout manually via this helper -- raising a clear, actionable error
// instead of pybind11's own generic "incompatible function arguments"
// message on a mismatch -- then cast without `forcecast`, so a
// correctly-typed input is used in place with zero copy.
using Array1DStrict = py::array_t<double, py::array::c_style>;

static Array1DStrict require_strict_f64_1d(const py::array& arr, const char* name) {
    if (!py::isinstance<py::array_t<double>>(arr) ||
        !(arr.flags() & py::array::c_style)) {
        throw std::invalid_argument(
            std::string(name) + " must already be a C-contiguous float64 "
            "array for this zero-copy binding (no implicit dtype/layout "
            "conversion happens here) -- use the non-'_zerocopy' binding "
            "for automatic conversion.");
    }
    auto typed = arr.cast<Array1DStrict>();
    if (typed.ndim() != 1)
        throw std::invalid_argument(
            std::string(name) + " must be a 1-D array, got ndim=" +
            std::to_string(typed.ndim()));
    return typed;
}

// 2-D counterpart of require_strict_f64_1d, for factors/signals matrices.
static py::array_t<double, py::array::c_style> require_strict_f64_2d(
    const py::array& arr, const char* name)
{
    if (!py::isinstance<py::array_t<double>>(arr) ||
        !(arr.flags() & py::array::c_style)) {
        throw std::invalid_argument(
            std::string(name) + " must already be a C-contiguous float64 "
            "array for this zero-copy binding (no implicit dtype/layout "
            "conversion happens here) -- use the non-'_zerocopy' binding "
            "for automatic conversion.");
    }
    auto typed = arr.cast<py::array_t<double, py::array::c_style>>();
    if (typed.ndim() != 2)
        throw std::invalid_argument(
            std::string(name) + " must be a 2-D array, got ndim=" +
            std::to_string(typed.ndim()));
    return typed;
}

// ── Shared argument validators ──────────────────────────────────────────────
//
// Scope is deliberately narrow: SCALAR CONFIGURATION parameters only.
//
// These bindings validated shape (ndim, matching lengths) and nothing else, so
// a direct native call could pass a configuration value with no meaning and get
// a confident-looking number back. Measured on the pre-validation build:
//
//   run_strategy(..., initial_capital=0)    -> total_return=nan, sharpe=9.99
//   run_strategy(..., initial_capital=-100) -> total_return=+1.7%
//   run_strategy(..., periods_per_year=-1)  -> annualized_volatility=nan
//   run_strategy(..., commission_pct=-0.1)  -> +23.2%, profitable from costs
//
// A wrong scalar here silently corrupts every number in the result, and there
// is no sentinel convention covering it -- note the first case returned NaN for
// total_return but a decisive-looking 9.99 Sharpe from the same call.
//
// What is deliberately NOT validated here: input DATA and per-indicator
// window/period arguments. Those already have a documented contract in this
// codebase -- degenerate arguments and bad bars yield NaN, not exceptions --
// and it exists for a reason the tests state outright: build_dataset's
// finite-value guard rejects an ENTIRE panel, so one zero print in one symbol
// used to fail a whole multi-entity build and blame the feature rather than the
// data (tests/modeling/test_feature_degenerate_windows.py::
// test_one_bad_bar_no_longer_rejects_the_whole_panel, and the matching
// all-NaN-not-raise tests in tests/cpp_bindings/). Adding finiteness, OHLC
// invariant or positive-period throws at this layer reintroduces exactly that
// failure mode. NaN propagation is the project's chosen answer for bad data;
// these validators only cover the arguments it was never meant to cover.
//
// Everything here throws std::invalid_argument, which pybind11 surfaces as a
// Python ValueError -- the same type the Python-side validators raise, so a
// caller cannot tell which layer rejected the call.

static void require_positive(double v, const char* name, const char* fn) {
    if (!(v > 0.0) || !std::isfinite(v))
        throw std::invalid_argument(
            std::string(fn) + ": " + name + " must be finite and > 0, got " +
            std::to_string(v));
}

static void require_non_negative(double v, const char* name, const char* fn) {
    if (!(v >= 0.0) || !std::isfinite(v))
        throw std::invalid_argument(
            std::string(fn) + ": " + name + " must be finite and >= 0, got " +
            std::to_string(v));
}

static void require_positive_int(int v, const char* name, const char* fn) {
    if (v <= 0)
        throw std::invalid_argument(
            std::string(fn) + ": " + name + " must be >= 1, got " + std::to_string(v));
}

// Grouped, because listing the individual require_* calls at each binding
// is what let two of them ship with none at all.
//
// batch_run_strategy validated all four; batch_run_strategy_zerocopy -- same
// kernel, same arguments, added later as a "strict/zero-copy variant [...]
// same semantics otherwise" -- validated none. Measured on the shipped
// build, the exact failure the block above was written to prevent:
//
//   batch_run_strategy(...,  initial_capital=0) -> ValueError, correctly
//   batch_run_strategy_zerocopy(..., 0)         -> [0.0, nan, 0.0129, 6.595]
//
// A NaN total_return beside a decisive-looking 6.6 Sharpe. Same story for
// simulate_forward_paths_zerocopy, which skipped the positive-int checks and
// reported a vaguer error from the kernel's own defence-in-depth instead.
//
// One call per binding now, so the question a reviewer has to answer is
// "does this binding validate?" rather than "are all four lines present and
// spelled with the right function name?".
static void require_backtest_scalars(
    double initial_capital, double commission_pct, double slippage_pct,
    double periods_per_year, const char* fn)
{
    require_positive(initial_capital, "initial_capital", fn);
    require_non_negative(commission_pct, "commission_pct", fn);
    require_non_negative(slippage_pct, "slippage_pct", fn);
    require_positive(periods_per_year, "periods_per_year", fn);
}

static void require_simulation_scalars(
    int horizon_days, int n_simulations, int block_size,
    double initial_capital, const char* fn)
{
    require_positive_int(horizon_days, "horizon_days", fn);
    require_positive_int(n_simulations, "n_simulations", fn);
    require_positive_int(block_size, "block_size", fn);
    require_positive(initial_capital, "initial_capital", fn);
}

static py::dict hurst_result_to_dict(const sqt::HurstResult& r) {
    py::dict d;
    d["hurst"]          = r.hurst;
    d["regime"]         = r.regime;
    d["fit_r_squared"]  = r.fit_r_squared;
    d["method"]         = r.method;
    d["n_obs"]          = static_cast<py::ssize_t>(r.n_obs);
    return d;
}

// ── Module definition ─────────────────────────────────────────────────────────
//
// Every binding below follows the same GIL-release shape: extract raw
// pointers/sizes/plain-C++-value arguments from the py:: types FIRST (while
// still holding the GIL -- pybind11's array buffer access and argument
// casting are Python-API calls), then release the GIL for the duration of
// the actual sqt:: kernel call (the only part of each binding doing
// nontrivial CPU work with no Python API calls of its own), then let
// py::gil_scoped_release's destructor reacquire the GIL before touching any
// py:: type again to build the return value. This lets multiple Python
// threads run these kernels concurrently instead of serializing on the GIL
// even though NumPy released it for nothing -- the C++ call itself never
// touched a Python object once past argument extraction.

PYBIND11_MODULE(_sqt_core, m) {
    m.doc() =
        "SQT C++ core — high-performance implementations of computationally "
        "intensive functions.  Import via the public Python modules; do not "
        "call these entry-points directly.";

    // ── Hurst exponent ────────────────────────────────────────────────────────

    m.def(
        "hurst_dfa",
        [](Array1D arr, int min_window, int max_window) -> py::dict {
            require_1d(arr, "arr");
            const double* arr_ptr = arr.data();
            const auto    n       = arr.size();
            sqt::HurstResult r;
            {
                py::gil_scoped_release release;
                r = sqt::hurst_exponent(arr_ptr, n, "dfa", min_window, max_window);
            }
            return hurst_result_to_dict(r);
        },
        py::arg("arr"),
        py::arg("min_window") = 10,
        py::arg("max_window") = -1,
        "Hurst exponent via Detrended Fluctuation Analysis (DFA-1).\n\n"
        "Pass max_window=-1 to auto-select (n//4).");

    m.def(
        "hurst_rs",
        [](Array1D arr, int min_window, int max_window) -> py::dict {
            require_1d(arr, "arr");
            const double* arr_ptr = arr.data();
            const auto    n       = arr.size();
            sqt::HurstResult r;
            {
                py::gil_scoped_release release;
                r = sqt::hurst_exponent(arr_ptr, n, "rs", min_window, max_window);
            }
            return hurst_result_to_dict(r);
        },
        py::arg("arr"),
        py::arg("min_window") = 10,
        py::arg("max_window") = -1,
        "Hurst exponent via Rescaled Range (R/S) analysis.\n\n"
        "Pass max_window=-1 to auto-select (n//2).  Prefer DFA for n < 2000.");

    m.def(
        "rolling_hurst",
        [](Array1D arr, int window, int step,
           const std::string& method, int min_window) -> py::array_t<double>
        {
            require_1d(arr, "arr");
            const double* arr_ptr = arr.data();
            const auto    n       = arr.size();
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_hurst_into(arr_ptr, n, window, step, method, min_window, out_ptr);
            }
            return out;
        },
        py::arg("arr"),
        py::arg("window")     = 200,
        py::arg("step")       = 1,
        py::arg("method")     = "dfa",
        py::arg("min_window") = 10,
        "Rolling Hurst exponent in a single C++ pass — no Python re-entry per bar.\n\n"
        "Returns a 1-D float64 array of length n; first (window-1) values are NaN.");

    m.def(
        "rolling_hurst_zerocopy",
        [](py::array arr, int window, int step,
           const std::string& method, int min_window) -> py::array_t<double>
        {
            auto arr_typed = require_strict_f64_1d(arr, "arr");
            const double* arr_ptr = arr_typed.data();
            const auto    n       = arr_typed.size();
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_hurst_into(arr_ptr, n, window, step, method, min_window, out_ptr);
            }
            return out;
        },
        py::arg("arr"),
        py::arg("window")     = 200,
        py::arg("step")       = 1,
        py::arg("method")     = "dfa",
        py::arg("min_window") = 10,
        "Strict/zero-copy variant of rolling_hurst() -- `arr` must already be "
        "a C-contiguous float64 array (raises instead of implicitly copying "
        "on a mismatch). Same semantics/output otherwise.");

    // ── RSI ───────────────────────────────────────────────────────────────────

    m.def(
        "rsi",
        [](Array1D arr, int period) -> py::array_t<double> {
            require_1d(arr, "arr");
            const double* arr_ptr = arr.data();
            const auto    n       = arr.size();
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rsi_into(arr_ptr, n, period, out_ptr);
            }
            return out;
        },
        py::arg("arr"),
        py::arg("period") = 14,
        "RSI via Wilder's smoothing (SMA seed, then alpha=1/period).\n\n"
        "Returns a 1-D float64 array of length n; first `period` values are NaN.");

    // ── ADX ───────────────────────────────────────────────────────────────────

    m.def(
        "adx",
        [](Array1D high, Array1D low, Array1D close, int period) -> py::array_t<double> {
            require_1d(high, "high");
            require_1d(low, "low");
            require_1d(close, "close");
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const double* high_ptr  = high.data();
            const double* low_ptr   = low.data();
            const double* close_ptr = close.data();
            const auto    n         = high.size();
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(3)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::adx_into(high_ptr, low_ptr, close_ptr, n, period, out_ptr);
            }
            return out;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("period") = 14,
        "ADX with DI+ and DI- (Wilder's smoothing).\n\n"
        "Returns a 2-D float64 array of shape (n, 3):\n"
        "  col 0 = DI+, col 1 = DI-, col 2 = ADX.\n"
        "First `period` rows have NaN in DI+/DI-; ADX starts at row 2*period-1.");

    // ── Parabolic SAR ─────────────────────────────────────────────────────────

    m.def(
        "parabolic_sar",
        [](Array1D high, Array1D low,
           double af_start, double af_step, double af_max) -> py::array_t<double>
        {
            require_1d(high, "high");
            require_1d(low, "low");
            if (high.size() != low.size())
                throw std::invalid_argument("high and low must have equal length");
            const double* high_ptr = high.data();
            const double* low_ptr  = low.data();
            const auto    n        = high.size();
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(2)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::parabolic_sar_into(high_ptr, low_ptr, n, af_start, af_step, af_max, out_ptr);
            }
            return out;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("af_start") = 0.02,
        py::arg("af_step")  = 0.02,
        py::arg("af_max")   = 0.2,
        "Parabolic SAR state machine.\n\n"
        "Returns a 2-D float64 array of shape (n, 2):\n"
        "  col 0 = SAR, col 1 = Trend (1.0 rising, -1.0 falling).");

    // ── Wilder's ATR ─────────────────────────────────────────────────────────

    m.def(
        "wilder_atr",
        [](Array1D high, Array1D low, Array1D close, int period) -> py::array_t<double> {
            require_1d(high, "high");
            require_1d(low, "low");
            require_1d(close, "close");
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const double* high_ptr  = high.data();
            const double* low_ptr   = low.data();
            const double* close_ptr = close.data();
            const auto    n         = high.size();
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::wilder_atr_into(high_ptr, low_ptr, close_ptr, n, period, out_ptr);
            }
            return out;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("period") = 14,
        "Wilder's ATR (Average True Range with Wilder's smoothing).\n\n"
        "TR[0]=high[0]-low[0]; TR[i]=max(H-L,|H-C_prev|,|L-C_prev|) for i>=1.\n"
        "Seed: ATR[period-1]=mean(TR[0..period-1]).\n"
        "Forward: ATR[i]=(ATR[i-1]*(period-1)+TR[i])/period.\n\n"
        "Returns a 1-D float64 array of length n; first period-1 values are NaN.");

    // ── Backtest kernel ───────────────────────────────────────────────────────

    m.def(
        "run_strategy",
        [](Array1D prices, Array1D signals,
           double initial_capital, double commission_pct, double slippage_pct,
           double periods_per_year,
           std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>
               ref_prices)
        -> py::dict {
            require_1d(prices, "prices");
            require_1d(signals, "signals");
            if (prices.size() != signals.size())
                throw std::invalid_argument("prices and signals must have equal length");
            require_backtest_scalars(initial_capital, commission_pct, slippage_pct,
                                     periods_per_year, "run_strategy");
            const double* prices_ptr  = prices.data();
            const double* signals_ptr = signals.data();
            const auto    n           = prices.size();
            // Resolved BEFORE the GIL is released: request() touches the
            // Python object, so doing it inside the released region would be
            // a use of the interpreter without holding the GIL.
            const double* ref_ptr = nullptr;
            if (ref_prices.has_value()) {
                require_1d(*ref_prices, "ref_prices");
                if (ref_prices->size() != prices.size())
                    throw std::invalid_argument(
                        "ref_prices must have the same length as prices");
                ref_ptr = ref_prices->data();
            }
            sqt::BacktestResult r;
            {
                py::gil_scoped_release release;
                r = sqt::run_strategy(
                    prices_ptr, signals_ptr, n,
                    initial_capital, commission_pct, slippage_pct, periods_per_year,
                    ref_ptr);
            }

            py::array_t<double> eq(static_cast<py::ssize_t>(r.equity_curve.size()));
            std::copy(r.equity_curve.begin(), r.equity_curve.end(), eq.mutable_data());

            py::dict d;
            d["final_equity"]          = r.final_equity;
            d["total_return"]          = r.total_return;
            d["annualized_volatility"] = r.annualized_vol;
            d["sharpe_ratio"]          = r.sharpe_ratio;
            d["sortino_ratio"]         = r.sortino_ratio;
            d["max_drawdown"]          = r.max_drawdown;
            d["calmar_ratio"]          = r.calmar_ratio;
            d["num_trades"]            = r.num_trades;
            d["win_rate"]              = r.win_rate;
            d["profit_factor"]         = r.profit_factor;
            d["avg_trade_return_pct"]  = r.avg_trade_return_pct;
            d["equity_curve"]          = eq;
            return d;
        },
        py::arg("prices"),
        py::arg("signals"),
        py::arg("initial_capital") = 10'000.0,
        py::arg("commission_pct")  = 0.001,
        py::arg("slippage_pct")    = 0.0005,
        // Bars per year for the annualized metrics. Python resolves the
        // calendar and passes the number; the kernel stays
        // calendar-agnostic. Defaults to 252 so existing callers are
        // unchanged.
        py::arg("periods_per_year") = 252.0,
        // Optional per-bar fill price. None -> close-to-close (the
        // historical behaviour); an array -> the two-leg
        // overnight/intraday decomposition engine.py uses for
        // next_open / hl2_exploratory, so the more realistic execution
        // model is no longer confined to the Python path.
        py::arg("ref_prices") = py::none(),
        "Vectorized backtest kernel — identical algorithm to run_strategy in engine.py.\n\n"
        "One-bar lag execution: executed[i] = signals[i-1].\n"
        "Returns a dict with keys: final_equity, total_return, annualized_volatility,\n"
        "sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, num_trades,\n"
        "win_rate, profit_factor, avg_trade_return_pct, equity_curve.");

    // ── Batch backtest ────────────────────────────────────────────────────────

    m.def(
        "batch_backtest_crossover",
        [](Array1D prices,
           py::array_t<double, py::array::c_style | py::array::forcecast> indicators,
           py::array_t<int, py::array::c_style | py::array::forcecast> pair_idx,
           double initial_capital, double commission_pct, double slippage_pct,
           double periods_per_year,
           std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>
               ref_prices)
        -> py::array_t<double>
        {
            require_1d(prices, "prices");
            require_backtest_scalars(initial_capital, commission_pct, slippage_pct,
                                     periods_per_year, "batch_backtest_crossover");
            auto ind_buf  = indicators.request();
            auto pair_buf = pair_idx.request();
            if (ind_buf.ndim != 2)
                throw std::invalid_argument("indicators must be 2-D (n_unique, n_bars)");
            if (pair_buf.ndim != 2 || pair_buf.shape[1] != 2)
                throw std::invalid_argument("pair_idx must be 2-D (num_combos, 2)");

            const auto n          = static_cast<std::size_t>(prices.size());
            const auto n_unique   = static_cast<std::size_t>(ind_buf.shape[0]);
            const auto num_combos = static_cast<std::size_t>(pair_buf.shape[0]);
            if (static_cast<std::size_t>(ind_buf.shape[1]) != n)
                throw std::invalid_argument("indicators.shape[1] must equal len(prices)");

            // Bounds-checked HERE, where an out-of-range row is a caller
            // error worth naming, rather than left to the kernel where it
            // would be an out-of-bounds read.
            const int* pair_ptr = static_cast<const int*>(pair_buf.ptr);
            for (std::size_t i = 0; i < num_combos * 2; ++i) {
                if (pair_ptr[i] < 0 ||
                    static_cast<std::size_t>(pair_ptr[i]) >= n_unique) {
                    throw std::invalid_argument(
                        "pair_idx contains a row index outside indicators");
                }
            }

            const double* ref_ptr = nullptr;
            if (ref_prices.has_value()) {
                require_1d(*ref_prices, "ref_prices");
                if (static_cast<std::size_t>(ref_prices->size()) != n)
                    throw std::invalid_argument(
                        "ref_prices must have the same length as prices");
                ref_ptr = ref_prices->data();
            }

            constexpr py::ssize_t kNumCols = 11;
            py::array_t<double> out(
                {static_cast<py::ssize_t>(num_combos), kNumCols});
            double* out_ptr = out.mutable_data();
            const double* p_ptr = prices.data();
            const double* i_ptr = static_cast<const double*>(ind_buf.ptr);
            {
                py::gil_scoped_release release;
                const auto results = sqt::batch_backtest_crossover(
                    p_ptr, i_ptr, n, n_unique, pair_ptr, num_combos,
                    initial_capital, commission_pct, slippage_pct,
                    periods_per_year, ref_ptr);
                for (std::size_t i = 0; i < results.size(); ++i) {
                    const auto& r = results[i];
                    double* row = out_ptr + i * static_cast<std::size_t>(kNumCols);
                    row[0]  = r.final_equity;
                    row[1]  = r.total_return;
                    row[2]  = r.annualized_vol;
                    row[3]  = r.sharpe_ratio;
                    row[4]  = r.sortino_ratio;
                    row[5]  = r.max_drawdown;
                    row[6]  = r.calmar_ratio;
                    row[7]  = r.win_rate;
                    row[8]  = r.profit_factor;
                    row[9]  = static_cast<double>(r.num_trades);
                    row[10] = r.avg_trade_return_pct;
                }
            }
            return out;
        },
        py::arg("prices"),
        py::arg("indicators"),
        py::arg("pair_idx"),
        py::arg("initial_capital")  = 10000.0,
        py::arg("commission_pct")   = 0.001,
        py::arg("slippage_pct")     = 0.0005,
        py::arg("periods_per_year") = 252.0,
        py::arg("ref_prices")       = py::none(),
        "Fused crossover grid: builds each combination's signal from two rows "
        "of `indicators` and backtests it immediately, so no (num_combos x "
        "n_bars) signal matrix is ever materialized. "
        "Returns a flat (num_combos, 11) array in the same column order as "
        "batch_run_strategy."
    );

    m.def(
        "batch_run_strategy",
        [](Array1D prices,
           py::array_t<double, py::array::c_style | py::array::forcecast> signals_2d,
           double initial_capital, double commission_pct, double slippage_pct,
           double periods_per_year,
           std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>
               ref_prices)
        -> py::array_t<double>
        {
            require_1d(prices, "prices");
            require_backtest_scalars(initial_capital, commission_pct, slippage_pct,
                                     periods_per_year, "batch_run_strategy");
            auto prices_buf  = prices.request();
            auto signals_buf = signals_2d.request();

            if (signals_buf.ndim != 2)
                throw std::invalid_argument("signals must be a 2-D array (num_tests, n_bars)");

            const auto n         = static_cast<std::size_t>(prices_buf.shape[0]);
            const auto num_tests = static_cast<std::size_t>(signals_buf.shape[0]);

            if (static_cast<std::size_t>(signals_buf.shape[1]) != n)
                throw std::invalid_argument("signals.shape[1] must equal len(prices)");

            const double* p_ptr = static_cast<const double*>(prices_buf.ptr);
            const double* s_ptr = static_cast<const double*>(signals_buf.ptr);

            // 11 metric columns, fixed order -- see docstring below. Returning a
            // flat (num_tests, 11) array instead of a Python list of dicts means
            // building num_tests Python objects is no longer part of this call
            // at all; the Python side builds one DataFrame directly from the
            // array instead of iterating a list of dicts first. Column order
            // here MUST stay in sync with backtest_grid's _BATCH_METRIC_COLUMNS
            // in backtest/engine.py.
            constexpr py::ssize_t kNumCols = 11;
            py::array_t<double> out(
                {static_cast<py::ssize_t>(num_tests), kNumCols});
            double* out_ptr = out.mutable_data();
            const double* ref_ptr = nullptr;
            if (ref_prices.has_value()) {
                require_1d(*ref_prices, "ref_prices");
                if (static_cast<std::size_t>(ref_prices->size()) != n)
                    throw std::invalid_argument(
                        "ref_prices must have the same length as prices");
                ref_ptr = ref_prices->data();
            }
            {
                py::gil_scoped_release release;
                const auto results = sqt::batch_run_strategy(
                    p_ptr, s_ptr, n, num_tests,
                    initial_capital, commission_pct, slippage_pct, periods_per_year,
                    ref_ptr);
                for (std::size_t i = 0; i < results.size(); ++i) {
                    const auto& r = results[i];
                    double* row = out_ptr + i * static_cast<std::size_t>(kNumCols);
                    row[0]  = r.final_equity;
                    row[1]  = r.total_return;
                    row[2]  = r.annualized_vol;
                    row[3]  = r.sharpe_ratio;
                    row[4]  = r.sortino_ratio;
                    row[5]  = r.max_drawdown;
                    row[6]  = r.calmar_ratio;
                    row[7]  = r.win_rate;
                    row[8]  = r.profit_factor;
                    row[9]  = static_cast<double>(r.num_trades);
                    row[10] = r.avg_trade_return_pct;
                }
            }
            return out;
        },
        py::arg("prices"),
        py::arg("signals"),
        py::arg("initial_capital") = 10'000.0,
        py::arg("commission_pct")  = 0.001,
        py::arg("slippage_pct")    = 0.0005,
        // Bars per year for the annualized metrics. Python resolves the
        // calendar and passes the number; the kernel stays
        // calendar-agnostic. Defaults to 252 so existing callers are
        // unchanged.
        py::arg("periods_per_year") = 252.0,
        // Optional per-bar fill price. None -> close-to-close (the
        // historical behaviour); an array -> the two-leg
        // overnight/intraday decomposition engine.py uses for
        // next_open / hl2_exploratory, so the more realistic execution
        // model is no longer confined to the Python path.
        py::arg("ref_prices") = py::none(),
        "Batch vectorized backtest — run all parameter combinations in one C++ call.\n\n"
        "signals must be a 2-D float64 array of shape (num_tests, n_bars).\n"
        "Returns a 2-D float64 array of shape (num_tests, 11), one row per\n"
        "test in input order. Columns (fixed order): final_equity,\n"
        "total_return, annualized_volatility, sharpe_ratio, sortino_ratio,\n"
        "max_drawdown, calmar_ratio, win_rate, profit_factor, num_trades\n"
        "(stored as float, cast back to int on the Python side),\n"
        "avg_trade_return_pct. equity_curve is NOT included, to save memory.");

    m.def(
        "batch_run_strategy_zerocopy",
        [](py::array prices_obj, py::array signals_obj,
           double initial_capital, double commission_pct, double slippage_pct,
           double periods_per_year,
           std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>
               ref_prices)
        -> py::array_t<double>
        {
            require_backtest_scalars(initial_capital, commission_pct, slippage_pct,
                                     periods_per_year, "batch_run_strategy_zerocopy");
            auto prices_arr  = require_strict_f64_1d(prices_obj, "prices");
            auto signals_arr = require_strict_f64_2d(signals_obj, "signals");
            auto prices_buf  = prices_arr.request();
            auto signals_buf = signals_arr.request();

            const auto n         = static_cast<std::size_t>(prices_buf.shape[0]);
            const auto num_tests = static_cast<std::size_t>(signals_buf.shape[0]);

            if (static_cast<std::size_t>(signals_buf.shape[1]) != n)
                throw std::invalid_argument("signals.shape[1] must equal len(prices)");

            const double* p_ptr = static_cast<const double*>(prices_buf.ptr);
            const double* s_ptr = static_cast<const double*>(signals_buf.ptr);

            constexpr py::ssize_t kNumCols = 11;
            py::array_t<double> out(
                {static_cast<py::ssize_t>(num_tests), kNumCols});
            double* out_ptr = out.mutable_data();
            const double* ref_ptr = nullptr;
            if (ref_prices.has_value()) {
                require_1d(*ref_prices, "ref_prices");
                if (static_cast<std::size_t>(ref_prices->size()) != n)
                    throw std::invalid_argument(
                        "ref_prices must have the same length as prices");
                ref_ptr = ref_prices->data();
            }
            {
                py::gil_scoped_release release;
                const auto results = sqt::batch_run_strategy(
                    p_ptr, s_ptr, n, num_tests,
                    initial_capital, commission_pct, slippage_pct, periods_per_year,
                    ref_ptr);
                for (std::size_t i = 0; i < results.size(); ++i) {
                    const auto& r = results[i];
                    double* row = out_ptr + i * static_cast<std::size_t>(kNumCols);
                    row[0]  = r.final_equity;
                    row[1]  = r.total_return;
                    row[2]  = r.annualized_vol;
                    row[3]  = r.sharpe_ratio;
                    row[4]  = r.sortino_ratio;
                    row[5]  = r.max_drawdown;
                    row[6]  = r.calmar_ratio;
                    row[7]  = r.win_rate;
                    row[8]  = r.profit_factor;
                    row[9]  = static_cast<double>(r.num_trades);
                    row[10] = r.avg_trade_return_pct;
                }
            }
            return out;
        },
        py::arg("prices"),
        py::arg("signals"),
        py::arg("initial_capital") = 10'000.0,
        py::arg("commission_pct")  = 0.001,
        py::arg("slippage_pct")    = 0.0005,
        // Bars per year for the annualized metrics. Python resolves the
        // calendar and passes the number; the kernel stays
        // calendar-agnostic. Defaults to 252 so existing callers are
        // unchanged.
        py::arg("periods_per_year") = 252.0,
        // Optional per-bar fill price. None -> close-to-close (the
        // historical behaviour); an array -> the two-leg
        // overnight/intraday decomposition engine.py uses for
        // next_open / hl2_exploratory, so the more realistic execution
        // model is no longer confined to the Python path.
        py::arg("ref_prices") = py::none(),
        "Strict/zero-copy variant of batch_run_strategy() -- `prices`/"
        "`signals` must already be C-contiguous float64 arrays (raises "
        "instead of implicitly copying on a mismatch). Same "
        "semantics/output/validation otherwise.");

    // ── 2-variable OLS ────────────────────────────────────────────────────────

    m.def(
        "ols2",
        [](Array1D y, Array1D x) -> py::dict {
            require_1d(y, "y");
            require_1d(x, "x");
            if (y.size() != x.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y.data();
            const double* x_ptr = x.data();
            const auto    n     = y.size();
            sqt::Ols2Result r;
            {
                py::gil_scoped_release release;
                r = sqt::ols2(y_ptr, x_ptr, n);
            }
            py::dict d;
            d["intercept"] = r.intercept;
            d["slope"]     = r.slope;
            d["r_squared"] = r.r_squared;
            return d;
        },
        py::arg("y"),
        py::arg("x"),
        "2-variable OLS: y = intercept + slope * x.\n\n"
        "Closed-form normal equations — avoids LAPACK for this 2×2 system.\n"
        "Returns a dict with keys: intercept, slope, r_squared.");

    // ── Rolling factor loadings ───────────────────────────────────────────────

    m.def(
        "rolling_factor_loadings",
        [](Array1D y_arr,
           py::array_t<double, py::array::c_style | py::array::forcecast> factors_arr,
           int window) -> py::array_t<double>
        {
            require_1d(y_arr, "y");
            auto y_buf  = y_arr.request();
            auto f_buf  = factors_arr.request();

            if (f_buf.ndim != 2)
                throw std::invalid_argument("factors must be a 2-D array (n, k)");
            if (y_buf.shape[0] != f_buf.shape[0])
                throw std::invalid_argument("len(y) must equal factors.shape[0]");

            const auto n = static_cast<std::size_t>(y_buf.shape[0]);
            const auto k = static_cast<std::size_t>(f_buf.shape[1]);
            // Checked: `k` is factors.shape[1] as the caller supplied it, and
            // this is the output array's column count. k+1 is formed in
            // size_t space so the intercept column cannot overflow the check
            // itself. (bindings.cpp compiles with /wd4244 /wd4267 for
            // pybind11's own py::ssize_t conversions, so a silent narrowing
            // here would not even warn.)
            const int  p = sqt::numerics::checked_narrow_to_int(
                k + 1, "rolling_factor_loadings: intercept + factor count");

            const double* y_ptr = static_cast<const double*>(y_buf.ptr);
            const double* f_ptr = static_cast<const double*>(f_buf.ptr);

            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(p)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_factor_loadings_into(y_ptr, f_ptr, n, k, window, out_ptr);
            }
            return out;
        },
        py::arg("y"),
        py::arg("factors"),
        py::arg("window"),
        "Rolling OLS factor loadings (per-window rank-revealing QR).\n\n"
        "y      : 1-D float64 array of length n (asset returns).\n"
        "factors: 2-D float64 array of shape (n, k).\n"
        "window : rolling window size in bars.\n\n"
        "Returns a 2-D float64 array of shape (n, k+1):\n"
        "  col 0 = alpha (intercept); cols 1..k = factor loadings.\n"
        "First (window-1) rows are NaN, as is any row whose window is\n"
        "rank-deficient (duplicated or perfectly collinear factors).");

    m.def(
        "rolling_factor_loadings_zerocopy",
        [](py::array y_obj, py::array factors_obj, int window) -> py::array_t<double>
        {
            auto y_arr       = require_strict_f64_1d(y_obj, "y");
            auto factors_arr = require_strict_f64_2d(factors_obj, "factors");
            auto y_buf = y_arr.request();
            auto f_buf = factors_arr.request();

            if (y_buf.shape[0] != f_buf.shape[0])
                throw std::invalid_argument("len(y) must equal factors.shape[0]");

            const auto n = static_cast<std::size_t>(y_buf.shape[0]);
            const auto k = static_cast<std::size_t>(f_buf.shape[1]);
            // Checked: `k` is factors.shape[1] as the caller supplied it, and
            // this is the output array's column count. k+1 is formed in
            // size_t space so the intercept column cannot overflow the check
            // itself. (bindings.cpp compiles with /wd4244 /wd4267 for
            // pybind11's own py::ssize_t conversions, so a silent narrowing
            // here would not even warn.)
            const int  p = sqt::numerics::checked_narrow_to_int(
                k + 1, "rolling_factor_loadings: intercept + factor count");

            const double* y_ptr = static_cast<const double*>(y_buf.ptr);
            const double* f_ptr = static_cast<const double*>(f_buf.ptr);

            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(p)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_factor_loadings_into(y_ptr, f_ptr, n, k, window, out_ptr);
            }
            return out;
        },
        py::arg("y"),
        py::arg("factors"),
        py::arg("window"),
        "Strict/zero-copy variant of rolling_factor_loadings() -- `y`/"
        "`factors` must already be C-contiguous float64 arrays (raises "
        "instead of implicitly copying on a mismatch). Same "
        "semantics/output otherwise.");

    // ── Rolling beta ──────────────────────────────────────────────────────────

    m.def(
        "rolling_beta",
        [](Array1D y_arr, Array1D x_arr, int window) -> py::array_t<double>
        {
            require_1d(y_arr, "y");
            require_1d(x_arr, "x");
            if (y_arr.size() != x_arr.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y_arr.data();
            const double* x_ptr = x_arr.data();
            const auto    n     = static_cast<std::size_t>(y_arr.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_beta_into(y_ptr, x_ptr, n, window, out_ptr);
            }
            return out;
        },
        py::arg("y"),
        py::arg("x"),
        py::arg("window"),
        "Rolling OLS beta using incremental O(1) sum updates.\n\n"
        "Returns a 1-D float64 array of length n;\n"
        "first (window-1) values are NaN.");

    m.def(
        "rolling_beta_zerocopy",
        [](py::array y_obj, py::array x_obj, int window) -> py::array_t<double>
        {
            auto y_arr = require_strict_f64_1d(y_obj, "y");
            auto x_arr = require_strict_f64_1d(x_obj, "x");
            if (y_arr.size() != x_arr.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y_arr.data();
            const double* x_ptr = x_arr.data();
            const auto    n     = static_cast<std::size_t>(y_arr.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_beta_into(y_ptr, x_ptr, n, window, out_ptr);
            }
            return out;
        },
        py::arg("y"),
        py::arg("x"),
        py::arg("window"),
        "Strict/zero-copy variant of rolling_beta() -- `y`/`x` must already "
        "be C-contiguous float64 arrays (raises instead of implicitly "
        "copying on a mismatch). Same semantics/output otherwise.");

    // ── Bollinger Bands ───────────────────────────────────────────────────────

    m.def(
        "bollinger_bands",
        [](Array1D prices, int period, double num_std) -> py::array_t<double>
        {
            require_1d(prices, "prices");
            const double* prices_ptr = prices.data();
            const auto    n          = static_cast<std::size_t>(prices.size());
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(3)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::bollinger_bands_into(prices_ptr, n, period, num_std, out_ptr);
            }
            return out;
        },
        py::arg("prices"),
        py::arg("period")  = 20,
        py::arg("num_std") = 2.0,
        "Bollinger Bands — fused sliding mean+std in one pass.\n\n"
        "Returns a 2-D float64 array of shape (n, 3):\n"
        "  col 0 = Upper, col 1 = Middle (SMA), col 2 = Lower.\n"
        "First (period-1) rows are NaN.");

    // ── Stochastic Oscillator ─────────────────────────────────────────────────

    m.def(
        "stochastic_oscillator",
        [](Array1D high, Array1D low, Array1D close,
           int k_period, int d_period) -> py::array_t<double>
        {
            require_1d(high, "high");
            require_1d(low, "low");
            require_1d(close, "close");
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const double* high_ptr  = high.data();
            const double* low_ptr   = low.data();
            const double* close_ptr = close.data();
            const auto    n         = static_cast<std::size_t>(high.size());
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(2)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::stochastic_oscillator_into(
                    high_ptr, low_ptr, close_ptr, n, k_period, d_period, out_ptr);
            }
            return out;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("k_period") = 14,
        py::arg("d_period") = 3,
        "Stochastic Oscillator — fused sliding min+max in one pass.\n\n"
        "Returns a 2-D float64 array of shape (n, 2):\n"
        "  col 0 = %%K, col 1 = %%D.\n"
        "First (k_period-1) rows have NaN in %%K;\n"
        "first (k_period + d_period - 2) rows have NaN in %%D.");

    // ── Fused technical indicators ────────────────────────────────────────────

    m.def(
        "technical_indicators",
        [](Array1D high, Array1D low, Array1D close,
           bool compute_rsi, int rsi_period,
           bool compute_adx, int adx_period,
           bool compute_atr, int atr_period,
           bool compute_bollinger, int bollinger_period, double bollinger_num_std,
           bool compute_stochastic, int stoch_k_period, int stoch_d_period) -> py::dict
        {
            require_1d(high, "high");
            require_1d(low, "low");
            require_1d(close, "close");
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const double* high_ptr  = high.data();
            const double* low_ptr   = low.data();
            const double* close_ptr = close.data();
            const auto    n         = static_cast<std::size_t>(high.size());

            sqt::TechnicalIndicatorsConfig cfg;
            cfg.compute_rsi        = compute_rsi;
            cfg.rsi_period         = rsi_period;
            cfg.compute_adx        = compute_adx;
            cfg.adx_period         = adx_period;
            cfg.compute_atr        = compute_atr;
            cfg.atr_period         = atr_period;
            cfg.compute_bollinger  = compute_bollinger;
            cfg.bollinger_period   = bollinger_period;
            cfg.bollinger_num_std  = bollinger_num_std;
            cfg.compute_stochastic = compute_stochastic;
            cfg.stoch_k_period     = stoch_k_period;
            cfg.stoch_d_period     = stoch_d_period;

            sqt::TechnicalIndicatorsResult r;
            {
                py::gil_scoped_release release;
                r = sqt::technical_indicators(high_ptr, low_ptr, close_ptr, n, cfg);
            }

            py::dict d;
            if (compute_rsi) {
                py::array_t<double> arr(static_cast<py::ssize_t>(n));
                std::copy(r.rsi.begin(), r.rsi.end(), arr.mutable_data());
                d["rsi"] = arr;
            }
            if (compute_adx) {
                py::array_t<double> arr({static_cast<py::ssize_t>(n), py::ssize_t(3)});
                std::copy(r.adx.begin(), r.adx.end(), arr.mutable_data());
                d["adx"] = arr;
            }
            if (compute_atr) {
                py::array_t<double> arr(static_cast<py::ssize_t>(n));
                std::copy(r.atr.begin(), r.atr.end(), arr.mutable_data());
                d["atr"] = arr;
            }
            if (compute_bollinger) {
                py::array_t<double> arr({static_cast<py::ssize_t>(n), py::ssize_t(3)});
                std::copy(r.bollinger.begin(), r.bollinger.end(), arr.mutable_data());
                d["bollinger_bands"] = arr;
            }
            if (compute_stochastic) {
                py::array_t<double> arr({static_cast<py::ssize_t>(n), py::ssize_t(2)});
                std::copy(r.stochastic.begin(), r.stochastic.end(), arr.mutable_data());
                d["stochastic_oscillator"] = arr;
            }
            return d;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("compute_rsi")        = false,
        py::arg("rsi_period")         = 14,
        py::arg("compute_adx")        = false,
        py::arg("adx_period")         = 14,
        py::arg("compute_atr")        = false,
        py::arg("atr_period")         = 14,
        py::arg("compute_bollinger")  = false,
        py::arg("bollinger_period")   = 20,
        py::arg("bollinger_num_std")  = 2.0,
        py::arg("compute_stochastic") = false,
        py::arg("stoch_k_period")     = 14,
        py::arg("stoch_d_period")     = 3,
        "Fused multi-indicator call: computes whichever of RSI/ADX/ATR/\n"
        "Bollinger Bands/Stochastic Oscillator are requested in ONE native\n"
        "call instead of up to 5 separate Python/C++ boundary crossings --\n"
        "each indicator's own algorithm and output shape are unchanged\n"
        "(this is pure orchestration, calling the same *_into kernels the\n"
        "individual rsi()/adx()/wilder_atr()/bollinger_bands()/\n"
        "stochastic_oscillator() bindings use).\n\n"
        "Returns a dict containing only the keys for indicators actually\n"
        "requested: 'rsi' (n,), 'adx' (n,3), 'atr' (n,), 'bollinger_bands'\n"
        "(n,3), 'stochastic_oscillator' (n,2) -- same shapes/column layout\n"
        "as each indicator's own standalone binding.");

    m.def(
        "technical_indicators_zerocopy",
        [](py::array high_obj, py::array low_obj, py::array close_obj,
           bool compute_rsi, int rsi_period,
           bool compute_adx, int adx_period,
           bool compute_atr, int atr_period,
           bool compute_bollinger, int bollinger_period, double bollinger_num_std,
           bool compute_stochastic, int stoch_k_period, int stoch_d_period) -> py::dict
        {
            auto high  = require_strict_f64_1d(high_obj, "high");
            auto low   = require_strict_f64_1d(low_obj, "low");
            auto close = require_strict_f64_1d(close_obj, "close");
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const double* high_ptr  = high.data();
            const double* low_ptr   = low.data();
            const double* close_ptr = close.data();
            const auto    n         = static_cast<std::size_t>(high.size());

            sqt::TechnicalIndicatorsConfig cfg;
            cfg.compute_rsi        = compute_rsi;
            cfg.rsi_period         = rsi_period;
            cfg.compute_adx        = compute_adx;
            cfg.adx_period         = adx_period;
            cfg.compute_atr        = compute_atr;
            cfg.atr_period         = atr_period;
            cfg.compute_bollinger  = compute_bollinger;
            cfg.bollinger_period   = bollinger_period;
            cfg.bollinger_num_std  = bollinger_num_std;
            cfg.compute_stochastic = compute_stochastic;
            cfg.stoch_k_period     = stoch_k_period;
            cfg.stoch_d_period     = stoch_d_period;

            sqt::TechnicalIndicatorsResult r;
            {
                py::gil_scoped_release release;
                r = sqt::technical_indicators(high_ptr, low_ptr, close_ptr, n, cfg);
            }

            py::dict d;
            if (compute_rsi) {
                py::array_t<double> arr(static_cast<py::ssize_t>(n));
                std::copy(r.rsi.begin(), r.rsi.end(), arr.mutable_data());
                d["rsi"] = arr;
            }
            if (compute_adx) {
                py::array_t<double> arr({static_cast<py::ssize_t>(n), py::ssize_t(3)});
                std::copy(r.adx.begin(), r.adx.end(), arr.mutable_data());
                d["adx"] = arr;
            }
            if (compute_atr) {
                py::array_t<double> arr(static_cast<py::ssize_t>(n));
                std::copy(r.atr.begin(), r.atr.end(), arr.mutable_data());
                d["atr"] = arr;
            }
            if (compute_bollinger) {
                py::array_t<double> arr({static_cast<py::ssize_t>(n), py::ssize_t(3)});
                std::copy(r.bollinger.begin(), r.bollinger.end(), arr.mutable_data());
                d["bollinger_bands"] = arr;
            }
            if (compute_stochastic) {
                py::array_t<double> arr({static_cast<py::ssize_t>(n), py::ssize_t(2)});
                std::copy(r.stochastic.begin(), r.stochastic.end(), arr.mutable_data());
                d["stochastic_oscillator"] = arr;
            }
            return d;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("compute_rsi")        = false,
        py::arg("rsi_period")         = 14,
        py::arg("compute_adx")        = false,
        py::arg("adx_period")         = 14,
        py::arg("compute_atr")        = false,
        py::arg("atr_period")         = 14,
        py::arg("compute_bollinger")  = false,
        py::arg("bollinger_period")   = 20,
        py::arg("bollinger_num_std")  = 2.0,
        py::arg("compute_stochastic") = false,
        py::arg("stoch_k_period")     = 14,
        py::arg("stoch_d_period")     = 3,
        "Strict/zero-copy variant of technical_indicators() -- `high`/`low`/"
        "`close` must already be C-contiguous float64 arrays (raises "
        "instead of implicitly copying on a mismatch). Same "
        "semantics/output otherwise.");

    m.def(
        "technical_indicators_panel",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> high,
           py::array_t<double, py::array::c_style | py::array::forcecast> low,
           py::array_t<double, py::array::c_style | py::array::forcecast> close,
           bool compute_rsi, int rsi_period,
           bool compute_adx, int adx_period,
           bool compute_atr, int atr_period,
           bool compute_bollinger, int bollinger_period, double bollinger_num_std,
           bool compute_stochastic, int stoch_k_period, int stoch_d_period) -> py::dict
        {
            auto h_buf = high.request();
            auto l_buf = low.request();
            auto c_buf = close.request();
            if (h_buf.ndim != 2 || l_buf.ndim != 2 || c_buf.ndim != 2)
                throw std::invalid_argument(
                    "high, low and close must each be 2-D (n_tickers, n_bars)");
            if (h_buf.shape[0] != l_buf.shape[0] || h_buf.shape[0] != c_buf.shape[0] ||
                h_buf.shape[1] != l_buf.shape[1] || h_buf.shape[1] != c_buf.shape[1])
                throw std::invalid_argument(
                    "high, low and close must have identical shapes");

            const auto n_tickers = static_cast<std::size_t>(h_buf.shape[0]);
            const auto n_bars    = static_cast<std::size_t>(h_buf.shape[1]);
            const auto nt = static_cast<py::ssize_t>(n_tickers);
            const auto nb = static_cast<py::ssize_t>(n_bars);

            sqt::TechnicalIndicatorsConfig cfg;
            cfg.compute_rsi        = compute_rsi;
            cfg.rsi_period         = rsi_period;
            cfg.compute_adx        = compute_adx;
            cfg.adx_period         = adx_period;
            cfg.compute_atr        = compute_atr;
            cfg.atr_period         = atr_period;
            cfg.compute_bollinger  = compute_bollinger;
            cfg.bollinger_period   = bollinger_period;
            cfg.bollinger_num_std  = bollinger_num_std;
            cfg.compute_stochastic = compute_stochastic;
            cfg.stoch_k_period     = stoch_k_period;
            cfg.stoch_d_period     = stoch_d_period;

            // Allocated here, while the GIL is held; the kernel writes into
            // them directly, so nothing is copied on the way back.
            py::array_t<double> a_rsi, a_adx, a_atr, a_bb, a_stoch;
            sqt::TechnicalIndicatorsPanelOut dest;
            if (compute_rsi) {
                a_rsi = py::array_t<double>({nt, nb});
                dest.rsi = a_rsi.mutable_data();
            }
            if (compute_adx) {
                a_adx = py::array_t<double>({nt, nb, py::ssize_t(3)});
                dest.adx = a_adx.mutable_data();
            }
            if (compute_atr) {
                a_atr = py::array_t<double>({nt, nb});
                dest.atr = a_atr.mutable_data();
            }
            if (compute_bollinger) {
                a_bb = py::array_t<double>({nt, nb, py::ssize_t(3)});
                dest.bollinger = a_bb.mutable_data();
            }
            if (compute_stochastic) {
                a_stoch = py::array_t<double>({nt, nb, py::ssize_t(2)});
                dest.stochastic = a_stoch.mutable_data();
            }

            const double* h_ptr = static_cast<const double*>(h_buf.ptr);
            const double* l_ptr = static_cast<const double*>(l_buf.ptr);
            const double* c_ptr = static_cast<const double*>(c_buf.ptr);
            {
                py::gil_scoped_release release;
                sqt::technical_indicators_panel(h_ptr, l_ptr, c_ptr,
                                                 n_tickers, n_bars, cfg, dest);
            }

            py::dict d;
            if (compute_rsi)        d["rsi"] = a_rsi;
            if (compute_adx)        d["adx"] = a_adx;
            if (compute_atr)        d["atr"] = a_atr;
            if (compute_bollinger)  d["bollinger_bands"] = a_bb;
            if (compute_stochastic) d["stochastic_oscillator"] = a_stoch;
            return d;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("compute_rsi")        = false,
        py::arg("rsi_period")         = 14,
        py::arg("compute_adx")        = false,
        py::arg("adx_period")         = 14,
        py::arg("compute_atr")        = false,
        py::arg("atr_period")         = 14,
        py::arg("compute_bollinger")  = false,
        py::arg("bollinger_period")   = 20,
        py::arg("bollinger_num_std")  = 2.0,
        py::arg("compute_stochastic") = false,
        py::arg("stoch_k_period")     = 14,
        py::arg("stoch_d_period")     = 3,
        "technical_indicators() over a whole universe in one call.\\n\\n"
        "high/low/close are 2-D float64 (n_tickers, n_bars); row t is\\n"
        "ticker t. Tickers are computed in parallel.\\n\\n"
        "Returns a dict containing only the requested keys, each with the\\n"
        "ticker axis prepended to the single-series shape: 'rsi'\\n"
        "(n_tickers, n_bars), 'adx' (n_tickers, n_bars, 3), 'atr'\\n"
        "(n_tickers, n_bars), 'bollinger_bands' (n_tickers, n_bars, 3),\\n"
        "'stochastic_oscillator' (n_tickers, n_bars, 2).\\n\\n"
        "Bit-identical to calling technical_indicators() once per ticker --\\n"
        "each row goes through the same kernels.");

    // ── Engle-Granger cointegration ───────────────────────────────────────────

    m.def(
        "engle_granger",
        [](Array1D y0, Array1D y1, int max_lag, bool use_aic) -> py::dict {
            require_1d(y0, "y0");
            require_1d(y1, "y1");
            if (y0.size() != y1.size())
                throw std::invalid_argument("y0 and y1 must have equal length");
            const double* y0_ptr = y0.data();
            const double* y1_ptr = y1.data();
            const auto    n      = y0.size();
            sqt::CointResult r;
            {
                py::gil_scoped_release release;
                r = sqt::engle_granger(y0_ptr, y1_ptr, n, max_lag, use_aic);
            }
            py::dict d;
            d["intercept"]     = r.intercept;
            d["hedge_ratio"]   = r.hedge_ratio;
            d["adf_statistic"] = r.adf_statistic;
            d["optimal_lag"]   = r.optimal_lag;
            d["p_value"]       = r.p_value;
            d["cv_1pct"]       = r.cv_1pct;
            d["cv_5pct"]       = r.cv_5pct;
            d["cv_10pct"]      = r.cv_10pct;
            d["half_life"]     = r.half_life;
            d["n_obs"]         = r.n_obs;
            d["cointegrated"]  = r.cointegrated;
            return d;
        },
        py::arg("y0"),
        py::arg("y1"),
        py::arg("max_lag") = -1,
        py::arg("use_aic") = true,
        "Engle-Granger two-step cointegration test.\n\n"
        "Step 1: OLS of y0 on y1 → hedge_ratio and spread.\n"
        "Step 2: ADF test on the spread (MacKinnon 2010 critical values).\n"
        "Step 3: AR(1) half-life of the spread.\n\n"
        "Returns a dict with keys: intercept, hedge_ratio, adf_statistic,\n"
        "optimal_lag, p_value, cv_1pct, cv_5pct, cv_10pct, half_life,\n"
        "n_obs, cointegrated.");

    m.def(
        "batch_engle_granger",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> prices,
           py::array_t<int, py::array::c_style | py::array::forcecast> pairs,
           int max_lag, bool use_aic) -> py::array_t<double>
        {
            auto p_buf = prices.request();
            auto q_buf = pairs.request();
            if (p_buf.ndim != 2)
                throw std::invalid_argument(
                    "prices must be a 2-D array (n_tickers, n_bars)");
            if (q_buf.ndim != 2 || q_buf.shape[1] != 2)
                throw std::invalid_argument("pairs must be a 2-D array (n_pairs, 2)");

            const auto n_tickers = static_cast<std::size_t>(p_buf.shape[0]);
            const auto n_bars    = static_cast<std::size_t>(p_buf.shape[1]);
            const auto n_pairs   = static_cast<std::size_t>(q_buf.shape[0]);

            constexpr py::ssize_t kCols = sqt::kBatchCointCols;
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n_pairs), kCols});
            double* out_ptr = out.mutable_data();
            const double* p_ptr = static_cast<const double*>(p_buf.ptr);
            const int*    q_ptr = static_cast<const int*>(q_buf.ptr);
            {
                py::gil_scoped_release release;
                sqt::batch_engle_granger(p_ptr, n_tickers, n_bars, q_ptr, n_pairs,
                                          max_lag, use_aic, out_ptr);
            }
            return out;
        },
        py::arg("prices"),
        py::arg("pairs"),
        py::arg("max_lag") = -1,
        py::arg("use_aic") = true,
        "Engle-Granger over many pairs in one native call.\\n\\n"
        "prices : 2-D float64 (n_tickers, n_bars), already aligned onto a\\n"
        "         common index by the caller -- this kernel never sees an\\n"
        "         index and does no date alignment.\\n"
        "pairs  : 2-D int32 (n_pairs, 2), row indices into `prices`.\\n\\n"
        "Returns a 2-D float64 array of shape (n_pairs, 11), one row per pair\\n"
        "in input order. Columns (fixed order): intercept, hedge_ratio,\\n"
        "adf_statistic, optimal_lag, p_value, cv_1pct, cv_5pct, cv_10pct,\\n"
        "half_life, n_obs, cointegrated (0.0/1.0).\\n\\n"
        "Bit-identical to calling engle_granger() once per pair, and\\n"
        "independent of thread count. Raises ValueError if a pairs row\\n"
        "references a ticker outside the panel.");

    // ── Monte Carlo (moving-block bootstrap) ──────────────────────────────────

    m.def(
        "simulate_forward_paths",
        [](Array1D values, int horizon_days, int n_simulations, int block_size,
           double initial_capital, py::object seed) -> py::array_t<double>
        {
            require_1d(values, "values");
            require_simulation_scalars(horizon_days, n_simulations, block_size,
                                       initial_capital, "simulate_forward_paths");
            const bool has_seed = !seed.is_none();
            const unsigned long long seed_val =
                has_seed ? seed.cast<unsigned long long>() : 0ULL;
             // horizon_days<=0 or n_simulations<=0 must raise, not silently
            // return a degenerate empty/zero-shaped array -- checked
            // explicitly here rather than only inferred from a result-size
            // mismatch below, since 0 * anything == 0 would otherwise make
            // an all-zero "expected" size indistinguishable from the
            // correctly-empty result these inputs actually produce.
            if (horizon_days <= 0 || n_simulations <= 0)
                throw std::invalid_argument(
                    "simulate_forward_paths: horizon_days and n_simulations must both be > 0");

            const double* values_ptr = values.data();
            const auto    n          = static_cast<std::size_t>(values.size());

            py::array_t<double> out(
                {static_cast<py::ssize_t>(n_simulations), static_cast<py::ssize_t>(horizon_days)});
            double* out_ptr = out.mutable_data();
            bool ok;
            {
                py::gil_scoped_release release;
                ok = sqt::simulate_forward_paths_into(
                    values_ptr, n, horizon_days, n_simulations, block_size,
                    initial_capital, seed_val, has_seed, out_ptr);
            }

            if (!ok)
                throw std::invalid_argument(
                    "simulate_forward_paths: invalid input (check block_size in "
                    "(0, len(values)] and initial_capital > 0)");

            return out;
        },
        py::arg("values"),
        py::arg("horizon_days"),
        py::arg("n_simulations"),
        py::arg("block_size"),
        py::arg("initial_capital"),
        py::arg("seed") = py::none(),
        "Moving-block bootstrap Monte Carlo forward simulation.\n\n"
        "Returns a 2-D float64 array of shape (n_simulations, horizon_days):\n"
        "  out[i, t] = simulated equity of path i at bar t.\n\n"
        "Each path is independently seeded (derived from `seed` and its own "
        "path index) so this does NOT reproduce numpy's PCG64 bit stream -- "
        "the same seed gives different concrete numbers than the pure-Python "
        "fallback, though repeat calls with the same seed on this path are "
        "bit-identical. Raises ValueError if horizon_days/n_simulations <= 0, "
        "values is empty, block_size is not in (0, len(values)], or "
        "initial_capital is non-positive.");

    m.def(
        "simulate_forward_paths_zerocopy",
        [](py::array values_obj, int horizon_days, int n_simulations, int block_size,
           double initial_capital, py::object seed) -> py::array_t<double>
        {
            require_simulation_scalars(horizon_days, n_simulations, block_size,
                                       initial_capital,
                                       "simulate_forward_paths_zerocopy");
            auto values = require_strict_f64_1d(values_obj, "values");
            const bool has_seed = !seed.is_none();
            const unsigned long long seed_val =
                has_seed ? seed.cast<unsigned long long>() : 0ULL;
            if (horizon_days <= 0 || n_simulations <= 0)
                throw std::invalid_argument(
                    "simulate_forward_paths_zerocopy: horizon_days and n_simulations must both be > 0");

            const double* values_ptr = values.data();
            const auto    n          = static_cast<std::size_t>(values.size());

            py::array_t<double> out(
                {static_cast<py::ssize_t>(n_simulations), static_cast<py::ssize_t>(horizon_days)});
            double* out_ptr = out.mutable_data();
            bool ok;
            {
                py::gil_scoped_release release;
                ok = sqt::simulate_forward_paths_into(
                    values_ptr, n, horizon_days, n_simulations, block_size,
                    initial_capital, seed_val, has_seed, out_ptr);
            }

            if (!ok)
                throw std::invalid_argument(
                    "simulate_forward_paths_zerocopy: invalid input (check block_size in "
                    "(0, len(values)] and initial_capital > 0)");

            return out;
        },
        py::arg("values"),
        py::arg("horizon_days"),
        py::arg("n_simulations"),
        py::arg("block_size"),
        py::arg("initial_capital"),
        py::arg("seed") = py::none(),
        "Strict/zero-copy variant of simulate_forward_paths() -- `values` "
        "must already be a C-contiguous float64 array (raises instead of "
        "implicitly copying on a mismatch). Same semantics/output/validation "
        "otherwise.");

    m.def(
        "simulate_forward_paths_terminal",
        [](Array1D values, int horizon_days, int n_simulations, int block_size,
           double initial_capital, py::object seed) -> py::array_t<double>
        {
            require_1d(values, "values");
            require_simulation_scalars(horizon_days, n_simulations, block_size,
                                       initial_capital, "simulate_forward_paths_terminal");
            const bool has_seed = !seed.is_none();
            const unsigned long long seed_val =
                has_seed ? seed.cast<unsigned long long>() : 0ULL;
            if (horizon_days <= 0 || n_simulations <= 0)
                throw std::invalid_argument(
                    "simulate_forward_paths_terminal: horizon_days and n_simulations must both be > 0");

            const double* values_ptr = values.data();
            const auto    n          = static_cast<std::size_t>(values.size());

            py::array_t<double> out(static_cast<py::ssize_t>(n_simulations));
            double* out_ptr = out.mutable_data();
            bool ok;
            {
                py::gil_scoped_release release;
                ok = sqt::simulate_forward_paths_terminal_into(
                    values_ptr, n, horizon_days, n_simulations, block_size,
                    initial_capital, seed_val, has_seed, out_ptr);
            }

            if (!ok)
                throw std::invalid_argument(
                    "simulate_forward_paths_terminal: invalid input (check block_size in "
                    "(0, len(values)] and initial_capital > 0)");

            return out;
        },
        py::arg("values"),
        py::arg("horizon_days"),
        py::arg("n_simulations"),
        py::arg("block_size"),
        py::arg("initial_capital"),
        py::arg("seed") = py::none(),
        "Terminal-only variant of simulate_forward_paths(): identical RNG/"
        "block-bootstrap core, but returns only each path's TERMINAL equity "
        "(1-D float64 array, length n_simulations) instead of the full "
        "(n_simulations, horizon_days) path matrix -- for memory-constrained "
        "large-simulation use where only the terminal distribution is needed. "
        "For identical (seed, inputs), result[i] == "
        "simulate_forward_paths(...)[i, -1] exactly. Same validation/error "
        "conventions as simulate_forward_paths().");

    // ── GARCH(1,1) variance recursion ─────────────────────────────────────────

    m.def(
        "garch11_variance_recursion",
        [](Array1D resid_sq, double omega, double alpha, double beta) -> py::array_t<double> {
            require_1d(resid_sq, "resid_sq");
            const double* resid_sq_ptr = resid_sq.data();
            const auto    n            = static_cast<std::size_t>(resid_sq.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::garch11_variance_recursion_into(resid_sq_ptr, n, omega, alpha, beta, out_ptr);
            }
            return out;
        },
        py::arg("resid_sq"),
        py::arg("omega"),
        py::arg("alpha"),
        py::arg("beta"),
        "GARCH(1,1) conditional variance recursion.\n\n"
        "sigma2[0] = max(mean(resid_sq), 1e-12); sigma2[t] = "
        "max(omega + alpha*resid_sq[t-1] + beta*sigma2[t-1], 1e-12) for t >= 1.\n"
        "Returns a 1-D float64 array of the same length as resid_sq.");

    m.def(
        "garch11_neg_loglik",
        [](Array1D resid_sq, double omega, double alpha, double beta,
           bool penalize) -> double
        {
            require_1d(resid_sq, "resid_sq");
            const double* resid_sq_ptr = resid_sq.data();
            const auto    n            = static_cast<std::size_t>(resid_sq.size());
            double nll;
            {
                py::gil_scoped_release release;
                nll = sqt::garch11_neg_loglik(resid_sq_ptr, n, omega, alpha, beta, penalize);
            }
            return nll;
        },
        py::arg("resid_sq"),
        py::arg("omega"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("penalize") = true,
        "GARCH(1,1) negative log-likelihood -- fuses the variance recursion\n"
        "and the NLL reduction into one native call, so a scipy.optimize\n"
        "objective evaluation never round-trips a full sigma2 array across\n"
        "the Python/C++ boundary just to reduce it to a scalar.\n\n"
        "nll = 0.5 * sum(log(2*pi) + log(sigma2) + resid_sq/sigma2);\n"
        "if penalize and (alpha+beta) >= 1.0: nll += 1e6*((alpha+beta)-1)**2.\n"
        "Returns a single float (0.0 if resid_sq is empty).");

    m.def(
        "garch11_neg_loglik_grad",
        [](Array1D resid_sq, double omega, double alpha, double beta,
           bool penalize) -> py::tuple
        {
            require_1d(resid_sq, "resid_sq");
            const double* resid_sq_ptr = resid_sq.data();
            const auto    n            = static_cast<std::size_t>(resid_sq.size());
            double nll;
            double grad[3];
            {
                py::gil_scoped_release release;
                nll = sqt::garch11_neg_loglik_grad(
                    resid_sq_ptr, n, omega, alpha, beta, penalize, grad);
            }
            py::array_t<double> grad_out(3);
            std::copy(grad, grad + 3, grad_out.mutable_data());
            return py::make_tuple(nll, grad_out);
        },
        py::arg("resid_sq"),
        py::arg("omega"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("penalize") = true,
        "GARCH(1,1) negative log-likelihood AND its analytic gradient\n"
        "w.r.t. (omega, alpha, beta), computed in one fused pass -- for\n"
        "scipy.optimize's jac=True convention (fun returns (value, grad)),\n"
        "so an optimizer using the gradient pays for one recursion per\n"
        "iteration, not two.\n\n"
        "Returns a tuple (nll: float, grad: 1-D float64 array of length 3\n"
        "[d/domega, d/dalpha, d/dbeta]).");

    // ── Kalman filters (time-varying hedge ratio) ─────────────────────────────

    m.def(
        "kalman_filter_1state",
        [](Array1D y, Array1D x, double delta, double observation_noise) -> py::dict {
            require_1d(y, "y");
            require_1d(x, "x");
            if (y.size() != x.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y.data();
            const double* x_ptr = x.data();
            const auto    n     = static_cast<std::size_t>(y.size());
            sqt::Kalman1StateResult r;
            {
                py::gil_scoped_release release;
                r = sqt::kalman_filter_1state(y_ptr, x_ptr, n, delta, observation_noise);
            }

            py::array_t<double> beta(static_cast<py::ssize_t>(r.beta.size()));
            std::copy(r.beta.begin(), r.beta.end(), beta.mutable_data());
            py::array_t<double> gain(static_cast<py::ssize_t>(r.gain.size()));
            std::copy(r.gain.begin(), r.gain.end(), gain.mutable_data());
            py::array_t<double> innovation(static_cast<py::ssize_t>(r.innovation.size()));
            std::copy(r.innovation.begin(), r.innovation.end(), innovation.mutable_data());

            py::dict d;
            d["beta"] = beta;
            d["gain"] = gain;
            d["innovation"] = innovation;
            return d;
        },
        py::arg("y"),
        py::arg("x"),
        py::arg("delta"),
        py::arg("observation_noise"),
        "1-state (slope-only) Kalman filter for a time-varying hedge ratio.\n\n"
        "Returns a dict with keys: beta, gain, innovation (each length n).\n"
        "All-empty if delta is not in (0,1) or observation_noise <= 0.");

    m.def(
        "kalman_filter_2state",
        [](Array1D y, Array1D x, double delta, double observation_noise) -> py::dict {
            require_1d(y, "y");
            require_1d(x, "x");
            if (y.size() != x.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y.data();
            const double* x_ptr = x.data();
            const auto    n     = static_cast<std::size_t>(y.size());
            sqt::Kalman2StateResult r;
            {
                py::gil_scoped_release release;
                r = sqt::kalman_filter_2state(y_ptr, x_ptr, n, delta, observation_noise);
            }

            py::array_t<double> alpha(static_cast<py::ssize_t>(r.alpha.size()));
            std::copy(r.alpha.begin(), r.alpha.end(), alpha.mutable_data());
            py::array_t<double> beta(static_cast<py::ssize_t>(r.beta.size()));
            std::copy(r.beta.begin(), r.beta.end(), beta.mutable_data());
            py::array_t<double> gain(static_cast<py::ssize_t>(r.gain.size()));
            std::copy(r.gain.begin(), r.gain.end(), gain.mutable_data());
            py::array_t<double> innovation(static_cast<py::ssize_t>(r.innovation.size()));
            std::copy(r.innovation.begin(), r.innovation.end(), innovation.mutable_data());

            py::dict d;
            d["alpha"] = alpha;
            d["beta"] = beta;
            d["gain"] = gain;
            d["innovation"] = innovation;
            return d;
        },
        py::arg("y"),
        py::arg("x"),
        py::arg("delta"),
        py::arg("observation_noise"),
        "2-state (intercept + slope) Kalman filter for a time-varying hedge ratio.\n\n"
        "Returns a dict with keys: alpha, beta, gain, innovation (each length n).\n"
        "All-empty if delta is not in (0,1) or observation_noise <= 0.");

    // ── Signal state machines (Donchian / VWAP-reversion hysteresis) ──────────

    m.def(
        "donchian_state_machine",
        [](Array1D close, Array1D entry_max, Array1D exit_min) -> py::array_t<double> {
            require_1d(close, "close");
            require_1d(entry_max, "entry_max");
            require_1d(exit_min, "exit_min");
            if (close.size() != entry_max.size() || close.size() != exit_min.size())
                throw std::invalid_argument("close, entry_max, exit_min must have equal length");
            const double* close_ptr     = close.data();
            const double* entry_max_ptr = entry_max.data();
            const double* exit_min_ptr  = exit_min.data();
            const auto    n             = static_cast<std::size_t>(close.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::donchian_state_machine_into(
                    close_ptr, entry_max_ptr, exit_min_ptr, n, out_ptr);
            }
            return out;
        },
        py::arg("close"),
        py::arg("entry_max"),
        py::arg("exit_min"),
        "Donchian breakout entry/exit hysteresis: 1.0=long, 0.0=flat.\n\n"
        "A NaN in entry_max/exit_min (rolling warmup) does not update the\n"
        "position state for that bar; output carries the position already\n"
        "held instead of hardcoding 0.0.");

    m.def(
        "vwap_reversion_state_machine",
        [](Array1D close, Array1D vwap, double entry_threshold) -> py::array_t<double> {
            require_1d(close, "close");
            require_1d(vwap, "vwap");
            if (close.size() != vwap.size())
                throw std::invalid_argument("close and vwap must have equal length");
            const double* close_ptr = close.data();
            const double* vwap_ptr  = vwap.data();
            const auto    n         = static_cast<std::size_t>(close.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::vwap_reversion_state_machine_into(
                    close_ptr, vwap_ptr, entry_threshold, n, out_ptr);
            }
            return out;
        },
        py::arg("close"),
        py::arg("vwap"),
        py::arg("entry_threshold"),
        "VWAP mean-reversion entry/exit hysteresis: 1.0=long, 0.0=flat.\n\n"
        "A NaN in vwap (rolling warmup) does not update the position state\n"
        "for that bar; output carries the position already held instead of\n"
        "hardcoding 0.0.");
}
