#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "sqt/hurst.hpp"
#include "sqt/indicators.hpp"
#include "sqt/cointegration.hpp"
#include "sqt/backtest.hpp"
#include "sqt/rolling_regression.hpp"
#include "sqt/monte_carlo.hpp"
#include "sqt/garch.hpp"
#include "sqt/signal_state_machines.hpp"
#include "sqt/numerics.hpp"
#include "sqt/panel_stats.hpp"
#include "sqt/build_info.hpp"

namespace py = pybind11;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Forces the input to be a C-contiguous float64 array -- but c_style and
// forcecast alone say nothing about ndim, so a caller passing a 2-D array
// (or any other shape) previously flowed through silently, flattened, and
// misinterpreted as if it were the 1-D series every kernel below assumes.
// require_1d() (called at the top of every lambda taking an Array1D
// parameter) is what actually enforces the "1-D" half of this type's name.
using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Templated over the element type and flags so an integer array is checked
// as itself. A single Array1D overload accepted an int64 code array only by
// implicitly CONVERTING it -- a full float64 copy made just to read ndim --
// and could not be called at all on the bindings whose arrays are not
// float64, which is how two of them flattened a (4, 2) input and answered.
template <typename T, int Flags>
static void require_1d(const py::array_t<T, Flags>& arr, const char* name) {
    if (arr.ndim() != 1)
        throw std::invalid_argument(
            std::string(name) + " must be a 1-D array, got ndim=" +
            std::to_string(arr.ndim()));
}

// ── Index arrays ─────────────────────────────────────────────────────────────
//
// Every integer index a binding receives -- date and entity codes, pair rows,
// rebalance bars, timestamps -- comes through exact_int64. They used to be
// declared py::array_t<Int, forcecast>, and forcecast is numpy's UNSAFE cast:
// it wraps an int64 2**33+2 to 2 when the target is int32, floors a float 1.9
// to 1, and wraps a uint64 2**64-1 to -1, all before any bounds check below
// can see the value. Measured: batch_engle_granger([[2**33+2, 2]]) answered
// for a series against itself (hedge ratio 1.0, cointegrated), and a wrapped
// date code handed back uninitialised memory.
//
// So: integer dtypes only (an empty array of any dtype is still accepted, as
// is a Python list of ints), every value converted to int64 exactly, and a
// range checked in int64 BEFORE anything is narrowed. The argument arrives as
// a py::object rather than a py::array because pybind11's py::array caster
// refuses a list outright, where the old forcecast parameters converted one.
using IndexArray = py::array_t<long long, py::array::c_style | py::array::forcecast>;

static IndexArray exact_int64(const py::object& obj, const char* name, const char* fn) {
    const py::array a = py::array::ensure(obj);
    if (!a)
        throw std::invalid_argument(
            std::string(fn) + ": " + name + " must be an integer array");
    const char kind = a.dtype().kind();
    if (a.size() > 0 && kind != 'i' && kind != 'u')
        throw std::invalid_argument(
            std::string(fn) + ": " + name + " must be an integer array, got dtype " +
            std::string(py::str(a.dtype())) +
            " -- a non-integer index would be truncated before it could be "
            "range-checked; convert it with .astype(np.int64) once its values "
            "are known to be whole");
    if (a.size() > 0 && kind == 'u' && a.itemsize() == 8) {
        // The one integer dtype int64 cannot hold exactly.
        const auto u = py::array_t<std::uint64_t,
                                   py::array::c_style | py::array::forcecast>::ensure(a);
        const std::uint64_t* p = u.data();
        const auto limit =
            static_cast<std::uint64_t>(std::numeric_limits<long long>::max());
        for (py::ssize_t i = 0; i < u.size(); ++i) {
            if (p[i] > limit)
                throw std::invalid_argument(
                    std::string(fn) + ": " + name + "[" + std::to_string(i) +
                    "] = " + std::to_string(p[i]) + " is outside the int64 range");
        }
    }
    // Exact now: every remaining source dtype fits in int64.
    IndexArray out = IndexArray::ensure(a);
    if (!out)
        throw std::invalid_argument(
            std::string(fn) + ": " + name + " could not be read as int64");
    return out;
}

// Every value in [0, upper), checked in int64 before any narrowing. `hint`
// names the usual cause, so the refusal says what to do about it.
static void require_indices_below(const IndexArray& a, long long upper,
                                  const char* name, const char* fn,
                                  const char* hint = "") {
    const long long* p = a.data();
    for (py::ssize_t i = 0; i < a.size(); ++i) {
        if (p[i] < 0 || p[i] >= upper)
            throw std::invalid_argument(
                std::string(fn) + ": " + name + "[" + std::to_string(i) + "] = " +
                std::to_string(p[i]) + " is outside [0, " + std::to_string(upper) +
                ")" + hint);
    }
}

// Row indices for a kernel that takes `int`, after the range check. `upper`
// is checked against INT_MAX first, so every value below it narrows exactly.
static std::vector<int> narrow_indices_to_int(const IndexArray& a, long long upper,
                                              const char* fn) {
    if (upper > static_cast<long long>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            std::string(fn) + ": " + std::to_string(upper) +
            " rows is outside what the kernel's int indices can address");
    std::vector<int> out(static_cast<std::size_t>(a.size()));
    const long long* p = a.data();
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<int>(p[i]);
    return out;
}

// Date and entity codes come from pd.factorize, which codes a NaT or NaN key
// as -1; that is the usual way a code falls outside its range.
static constexpr const char* kFactorizeHint =
    "; every row needs a code in that range, and pd.factorize gives -1 for a "
    "NaT or NaN key, so drop or fill those rows first";

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

// Any sign, but a number. A risk-free rate may be negative -- policy rates
// have been -- but a NaN one made Sharpe NaN and Sortino +inf from the same
// call, and +inf is the Sortino that reads as "no downside at all".
static void require_finite(double v, const char* name, const char* fn) {
    if (!std::isfinite(v))
        throw std::invalid_argument(
            std::string(fn) + ": " + name + " must be finite, got " +
            std::to_string(v));
}

// An enum code from Python. An unknown one used to fall into whichever
// branch the kernel tests last -- fill=7 ran as Close, commission_model=9 as
// percentage -- so a typo answered as a different configuration.
static void require_one_of(int v, std::initializer_list<int> allowed,
                           const char* name, const char* fn) {
    if (std::find(allowed.begin(), allowed.end(), v) == allowed.end()) {
        std::string codes;
        for (int a : allowed) codes += (codes.empty() ? "" : ", ") + std::to_string(a);
        throw std::invalid_argument(
            std::string(fn) + ": " + name + " must be one of {" + codes +
            "}, got " + std::to_string(v));
    }
}

// Grouped, because listing the individual require_* calls at each binding
// is what let two of them ship with none at all.
//
// The pair that proved it are gone now -- `batch_run_strategy` validated all
// four scalars, and its zero-copy sibling, same kernel and same arguments,
// added later as a "strict/zero-copy variant [...] same semantics
// otherwise", validated none. Measured on the shipped build:
//
//   batch_run_strategy(..., initial_capital=0)  -> ValueError, correctly
//   the _zerocopy sibling                       -> [0.0, nan, 0.0129, 6.595]
//
// A NaN total_return beside a decisive-looking 6.6 Sharpe. The lesson is
// what survives the deletion: one call per binding, so the question a
// reviewer answers is "does this binding validate?" rather than "are all
// four lines present and spelled with the right function name?".
static void require_backtest_scalars(
    double initial_capital, double commission_pct, double slippage_pct,
    double periods_per_year, double risk_free_rate, const char* fn)
{
    require_positive(initial_capital, "initial_capital", fn);
    require_non_negative(commission_pct, "commission_pct", fn);
    require_non_negative(slippage_pct, "slippage_pct", fn);
    require_positive(periods_per_year, "periods_per_year", fn);
    require_finite(risk_free_rate, "risk_free_rate", fn);
}

// The same lesson for the portfolio account, which shipped with fifteen
// scalars and no call at all. Measured on that build: slippage_pct=-0.5 tripled
// the final equity, a NaN max_gross_leverage silently switched the limit off
// (`x > NaN` is false), and fill=7 ran as Close. The bounds are the ones
// backtest/portfolio_engine.py and costs._cost_rate apply before they call
// here, so a direct caller is held to the same rules as the Python engine.
static void require_portfolio_scalars(const sqt::PortfolioCosts& c, const char* fn) {
    require_positive(c.initial_capital, "initial_capital", fn);
    require_non_negative(c.commission_pct, "commission_pct", fn);
    require_non_negative(c.sell_commission_pct, "sell_commission_pct", fn);
    require_non_negative(c.slippage_pct, "slippage_pct", fn);
    require_positive(c.max_gross_leverage, "max_gross_leverage", fn);
    require_positive(c.max_position_pct, "max_position_pct", fn);
    require_non_negative(c.borrow_fee_bps, "borrow_fee_bps", fn);
    require_non_negative(c.margin_interest_rate, "margin_interest_rate", fn);
    require_one_of(c.fill, {sqt::kFillClose, sqt::kFillNextOpen, sqt::kFillHl2},
                   "fill", fn);
    require_one_of(c.commission_model,
                   {sqt::kCommissionPct, sqt::kCommissionPerShare},
                   "commission_model", fn);
    require_non_negative(c.per_share_rate, "per_share_rate", fn);
    require_non_negative(c.min_commission, "min_commission", fn);
    // use_impact_model needs nothing: pybind11's bool caster refuses a
    // non-bool before this runs.
    require_non_negative(c.impact_coefficient, "impact_coefficient", fn);
    // 0 is the kernel's "no cap" (Python sends 0.0 for None); a negative,
    // NaN or infinite cap used to switch the cap off just as silently.
    require_non_negative(c.max_adv_participation, "max_adv_participation", fn);
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

    // ── Build stamp ───────────────────────────────────────────────────────────
    // Which sources and which build produced this binary. The package reads
    // `source_digest` at import and refuses an extension built from sources
    // other than the ones beside it, so an old copy cannot keep answering
    // for code that has since changed. A read-only mapping, because it is a
    // statement about the binary: nothing at runtime can make it truer.
    {
        const sqt::BuildInfo info = sqt::build_info();
        py::dict stamp;
        stamp["source_digest"] = info.source_digest;
        stamp["source_files"]  = info.source_files;
        stamp["build_type"]    = info.build_type;
        stamp["native_arch"]   = info.native_arch;
        stamp["compiler"]      = info.compiler;
        stamp["openmp"] = (info.openmp[0] != '\0')
                              ? py::object(py::str(info.openmp))
                              : py::object(py::none());
        stamp["pgo"] = info.pgo;
        m.attr("__build_info__") =
            py::module_::import("types").attr("MappingProxyType")(stamp);
    }

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
               ref_prices,
           double risk_free_rate)
        -> py::dict {
            require_1d(prices, "prices");
            require_1d(signals, "signals");
            if (prices.size() != signals.size())
                throw std::invalid_argument("prices and signals must have equal length");
            require_backtest_scalars(initial_capital, commission_pct, slippage_pct,
                                     periods_per_year, risk_free_rate, "run_strategy");
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
                    ref_ptr, risk_free_rate);
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
        // Annualized risk-free rate. Subtracted per period from every
        // return before Sharpe and Sortino, matching
        // metrics/risk_metrics.py. Defaults to 0.0 -- the value this
        // kernel always assumed -- so no existing result moves.
        py::arg("risk_free_rate") = 0.0,
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
           py::object pair_idx_obj,
           double initial_capital, double commission_pct, double slippage_pct,
           double periods_per_year,
           std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>
               ref_prices,
           double risk_free_rate)
        -> py::array_t<double>
        {
            constexpr const char* fn = "batch_backtest_crossover";
            require_1d(prices, "prices");
            require_backtest_scalars(initial_capital, commission_pct, slippage_pct,
                                     periods_per_year, risk_free_rate, fn);
            auto ind_buf  = indicators.request();
            const IndexArray pair_idx = exact_int64(pair_idx_obj, "pair_idx", fn);
            if (ind_buf.ndim != 2)
                throw std::invalid_argument("indicators must be 2-D (n_unique, n_bars)");
            if (pair_idx.ndim() != 2 || pair_idx.shape(1) != 2)
                throw std::invalid_argument("pair_idx must be 2-D (num_combos, 2)");

            const auto n          = static_cast<std::size_t>(prices.size());
            const auto n_unique   = static_cast<std::size_t>(ind_buf.shape[0]);
            const auto num_combos = static_cast<std::size_t>(pair_idx.shape(0));
            if (static_cast<std::size_t>(ind_buf.shape[1]) != n)
                throw std::invalid_argument("indicators.shape[1] must equal len(prices)");

            // Bounds-checked HERE, in int64 and before narrowing, where an
            // out-of-range row is a caller error worth naming, rather than
            // left to the kernel where it would be an out-of-bounds read.
            require_indices_below(pair_idx, static_cast<long long>(n_unique),
                                  "pair_idx", fn,
                                  " (a row of indicators)");
            const std::vector<int> pairs_int = narrow_indices_to_int(
                pair_idx, static_cast<long long>(n_unique), fn);
            const int* pair_ptr = pairs_int.data();

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
                    periods_per_year, ref_ptr, risk_free_rate);
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
        // Annualized risk-free rate. Subtracted per period from every
        // return before Sharpe and Sortino, matching
        // metrics/risk_metrics.py. Defaults to 0.0 -- the value this
        // kernel always assumed -- so no existing result moves.
        py::arg("risk_free_rate") = 0.0,
        "Fused crossover grid: builds each combination's signal from two rows "
        "of `indicators` and backtests it immediately, so no (num_combos x "
        "n_bars) signal matrix is ever materialized. pair_idx is a "
        "(num_combos, 2) integer array of rows of `indicators`; a float or "
        "out-of-range index raises ValueError rather than being truncated or "
        "wrapped. "
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
               ref_prices,
           double risk_free_rate)
        -> py::array_t<double>
        {
            require_1d(prices, "prices");
            require_backtest_scalars(initial_capital, commission_pct, slippage_pct,
                                     periods_per_year, risk_free_rate,
                                     "batch_run_strategy");
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
                    ref_ptr, risk_free_rate);
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
        // Annualized risk-free rate. Subtracted per period from every
        // return before Sharpe and Sortino, matching
        // metrics/risk_metrics.py. Defaults to 0.0 -- the value this
        // kernel always assumed -- so no existing result moves.
        py::arg("risk_free_rate") = 0.0,
        "Batch vectorized backtest — run all parameter combinations in one C++ call.\n\n"
        "signals must be a 2-D float64 array of shape (num_tests, n_bars).\n"
        "Returns a 2-D float64 array of shape (num_tests, 11), one row per\n"
        "test in input order. Columns (fixed order): final_equity,\n"
        "total_return, annualized_volatility, sharpe_ratio, sortino_ratio,\n"
        "max_drawdown, calmar_ratio, win_rate, profit_factor, num_trades\n"
        "(stored as float, cast back to int on the Python side),\n"
        "avg_trade_return_pct. equity_curve is NOT included, to save memory.");

    m.def(
        "run_portfolio_simulation",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> close,
           py::array_t<double, py::array::c_style | py::array::forcecast> exec_prices,
           py::array_t<double, py::array::c_style | py::array::forcecast> weights,
           py::object rebal_bars_obj,
           py::array_t<double, py::array::c_style | py::array::forcecast> day_gaps,
           double initial_capital, double commission_pct,
           double sell_commission_pct, double slippage_pct,
           double max_gross_leverage, double max_position_pct,
           double borrow_fee_bps, double margin_interest_rate,
           int fill,
           py::object dollar_volume, py::object volatility,
           int commission_model, double per_share_rate, double min_commission,
           bool use_impact_model, double impact_coefficient,
           double max_adv_participation) -> py::dict
        {
            constexpr const char* fn = "run_portfolio_simulation";
            const IndexArray rebal_bars = exact_int64(rebal_bars_obj, "rebal_bars", fn);
            auto c_buf = close.request();
            auto x_buf = exec_prices.request();
            auto w_buf = weights.request();
            auto r_buf = rebal_bars.request();
            auto g_buf = day_gaps.request();
            if (c_buf.ndim != 2 || x_buf.ndim != 2 || w_buf.ndim != 2)
                throw std::invalid_argument(
                    "close, exec_prices and weights must each be 2-D");
            if (r_buf.ndim != 1 || g_buf.ndim != 1)
                throw std::invalid_argument(
                    "rebal_bars and day_gaps must each be 1-D");

            const auto n_bars    = static_cast<std::size_t>(c_buf.shape[0]);
            const auto n_tickers = static_cast<std::size_t>(c_buf.shape[1]);
            const auto n_rebal   = static_cast<std::size_t>(w_buf.shape[0]);
            if (static_cast<std::size_t>(x_buf.shape[0]) != n_bars ||
                static_cast<std::size_t>(x_buf.shape[1]) != n_tickers)
                throw std::invalid_argument(
                    "exec_prices must have the same shape as close");
            if (static_cast<std::size_t>(w_buf.shape[1]) != n_tickers)
                throw std::invalid_argument(
                    "weights.shape[1] must equal close.shape[1]");
            if (static_cast<std::size_t>(r_buf.shape[0]) != n_rebal)
                throw std::invalid_argument(
                    "rebal_bars must have one entry per weights row");
            if (static_cast<std::size_t>(g_buf.shape[0]) != n_bars)
                throw std::invalid_argument(
                    "day_gaps must have one entry per bar");

            sqt::PortfolioCosts costs;
            costs.initial_capital      = initial_capital;
            costs.commission_pct       = commission_pct;
            costs.sell_commission_pct  = sell_commission_pct;
            costs.slippage_pct         = slippage_pct;
            costs.max_gross_leverage   = max_gross_leverage;
            costs.max_position_pct     = max_position_pct;
            costs.borrow_fee_bps       = borrow_fee_bps;
            costs.margin_interest_rate = margin_interest_rate;
            costs.fill                 = fill;
            costs.commission_model      = commission_model;
            costs.per_share_rate        = per_share_rate;
            costs.min_commission        = min_commission;
            costs.use_impact_model      = use_impact_model;
            costs.impact_coefficient    = impact_coefficient;
            costs.max_adv_participation = max_adv_participation;
            require_portfolio_scalars(costs, fn);

            // The two arrays that are configuration rather than data, held
            // to what the Python engine guarantees by construction. The
            // kernel skips a rebalance row that is out of order and never
            // reaches one at or past the last bar -- both silently, as fewer
            // executed rebalances than rows -- and a NaN day gap turned the
            // financing charge, and every equity value after it, into NaN.
            require_indices_below(rebal_bars, static_cast<long long>(n_bars),
                                  "rebal_bars", fn, " (a bar of close)");
            const long long* rb = rebal_bars.data();
            for (std::size_t i = 1; i < n_rebal; ++i) {
                if (rb[i] <= rb[i - 1])
                    throw std::invalid_argument(
                        std::string(fn) + ": rebal_bars must be strictly "
                        "increasing, got " + std::to_string(rb[i - 1]) +
                        " then " + std::to_string(rb[i]) + " at position " +
                        std::to_string(i) + "; sort the weights rows by date "
                        "and merge duplicates");
            }
            const double* gaps = static_cast<const double*>(g_buf.ptr);
            for (std::size_t i = 0; i < n_bars; ++i) {
                if (!(gaps[i] >= 0.0) || !std::isfinite(gaps[i]))
                    throw std::invalid_argument(
                        std::string(fn) + ": day_gaps[" + std::to_string(i) +
                        "] must be finite and >= 0, got " +
                        std::to_string(gaps[i]));
            }

            // Held at this scope, not inside the branch: the pointers below
            // are borrowed from these arrays and must not outlive them.
            using Mat = py::array_t<double,
                                    py::array::c_style | py::array::forcecast>;
            Mat dv_arr, vol_arr;
            const double* dv_ptr  = nullptr;
            const double* vol_ptr = nullptr;

            auto take_panel = [&](py::object src, Mat& hold, const char* name)
                -> const double* {
                if (src.is_none()) return nullptr;
                hold = src.cast<Mat>();
                auto b = hold.request();
                if (b.ndim != 2 ||
                    static_cast<std::size_t>(b.shape[0]) != n_bars ||
                    static_cast<std::size_t>(b.shape[1]) != n_tickers)
                    throw std::invalid_argument(
                        std::string(name) +
                        " must have the same shape as close");
                return static_cast<const double*>(b.ptr);
            };
            dv_ptr  = take_panel(dollar_volume, dv_arr, "dollar_volume");
            vol_ptr = take_panel(volatility, vol_arr, "volatility");

            // Refused here rather than dereferenced null in the kernel. A
            // caller who asked for an ADV cap or the impact model and gave
            // no volume panel wants a liquidity-aware run the data cannot
            // support, and silently dropping the constraint would satisfy
            // it by default.
            if ((max_adv_participation > 0.0 || use_impact_model) && !dv_ptr)
                throw std::invalid_argument(
                    "max_adv_participation and use_impact_model both need a "
                    "dollar_volume panel");
            if (use_impact_model && !vol_ptr)
                throw std::invalid_argument(
                    "use_impact_model needs a volatility panel");

            const auto nb = static_cast<py::ssize_t>(n_bars);
            py::array_t<double> eq(nb), csh(nb), grs(nb), net(nb);
            py::array_t<double> reb({static_cast<py::ssize_t>(n_rebal),
                                     py::ssize_t(3)});

            const double*    c_ptr = static_cast<const double*>(c_buf.ptr);
            const double*    x_ptr = static_cast<const double*>(x_buf.ptr);
            const double*    w_ptr = static_cast<const double*>(w_buf.ptr);
            const long long* r_ptr = static_cast<const long long*>(r_buf.ptr);
            const double*    g_ptr = static_cast<const double*>(g_buf.ptr);
            double* eq_ptr  = eq.mutable_data();
            double* csh_ptr = csh.mutable_data();
            double* grs_ptr = grs.mutable_data();
            double* net_ptr = net.mutable_data();
            double* reb_ptr = n_rebal ? reb.mutable_data() : nullptr;

            sqt::PortfolioSimError err;
            std::size_t n_executed = 0;
            double peak_position = 0.0;
            {
                py::gil_scoped_release release;
                n_executed = sqt::run_portfolio_simulation(
                    c_ptr, x_ptr, w_ptr, r_ptr, g_ptr,
                    n_bars, n_tickers, n_rebal, costs,
                    eq_ptr, csh_ptr, grs_ptr, net_ptr, reb_ptr,
                    &peak_position, &err, dv_ptr, vol_ptr);
            }

            py::dict d;
            d["equity"]      = eq;
            d["cash"]        = csh;
            d["gross"]       = grs;
            d["net"]         = net;
            d["rebalances"]  = reb;
            d["n_executed"]  = static_cast<py::ssize_t>(n_executed);
            d["peak_position"] = peak_position;
            d["status"]      = err.status;
            d["bar"]         = static_cast<py::ssize_t>(err.bar);
            d["ticker"]      = err.ticker;
            d["value"]       = err.value;
            return d;
        },
        py::arg("close"),
        py::arg("exec_prices"),
        py::arg("weights"),
        py::arg("rebal_bars"),
        py::arg("day_gaps"),
        py::arg("initial_capital")      = 10'000.0,
        py::arg("commission_pct")       = 0.001,
        py::arg("sell_commission_pct")  = 0.001,
        py::arg("slippage_pct")         = 0.0005,
        py::arg("max_gross_leverage")   = 1.0,
        py::arg("max_position_pct")     = 1.0,
        py::arg("borrow_fee_bps")       = 0.0,
        py::arg("margin_interest_rate") = 0.0,
        py::arg("fill")                 = 0,
        // Defaulted to None/off so every existing call -- including the
        // Python engine's own, before it learned to pass them -- keeps its
        // exact previous behaviour.
        py::arg("dollar_volume")        = py::none(),
        py::arg("volatility")           = py::none(),
        py::arg("commission_model")     = 0,
        py::arg("per_share_rate")       = 0.0,
        py::arg("min_commission")       = 0.0,
        py::arg("use_impact_model")     = false,
        py::arg("impact_coefficient")   = 1.0,
        py::arg("max_adv_participation") = 0.0,
        "Shared-cash multi-asset portfolio simulation.\n\n"
        "Runs every configuration backtest/portfolio_engine.py accepts: the\n"
        "pct and per_share commission models (commission_model 0 / 1), the\n"
        "square-root impact model (use_impact_model, with dollar_volume and\n"
        "volatility panels) and the ADV participation cap\n"
        "(max_adv_participation > 0, with a dollar_volume panel; 0 = no cap).\n\n"
        "close/exec_prices are (n_bars, n_tickers); weights is\n"
        "(n_rebal, n_tickers); rebal_bars is the bar index each weights row\n"
        "triggers at, an integer array strictly increasing within\n"
        "[0, n_bars); day_gaps is calendar days since the previous bar, for\n"
        "financing accrual, each finite and >= 0.\n\n"
        "fill: 0 = Close, 1 = next Open, 2 = (High+Low)/2.\n\n"
        "Every scalar is held to the Python engine's bounds (capital,\n"
        "leverage and position limits finite and > 0; rates, fees and the\n"
        "impact coefficient finite and >= 0; fill and commission_model one of\n"
        "their codes) and a violation raises ValueError.\n\n"
        "Returns a dict with equity/cash/gross/net (n_bars each), rebalances\n"
        "(n_rebal, 3) of turnover_pct/gross_leverage_after/n_positions,\n"
        "n_executed, and a status/bar/ticker/value quartet describing why the\n"
        "simulation stopped -- status 0 means it ran to the end. Rebalance\n"
        "rows from n_executed on are NaN (a next_open trigger on the last bar\n"
        "never executes). After an early stop, bars the simulation never\n"
        "marked are NaN: from `bar` on for a failed rebalance, from `bar`+1\n"
        "when equity reached zero at `bar`. The caller raises on a non-zero\n"
        "status; this never does, so the exact message stays in Python.");

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
        "  col 0 = %K, col 1 = %D.\n"
        "First (k_period-1) rows have NaN in %K;\n"
        "first (k_period + d_period - 2) rows have NaN in %D.");

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
        "technical_indicators() over a whole universe in one call.\n\n"
        "high/low/close are 2-D float64 (n_tickers, n_bars); row t is\n"
        "ticker t. Tickers are computed in parallel.\n\n"
        "Returns a dict containing only the requested keys, each with the\n"
        "ticker axis prepended to the single-series shape: 'rsi'\n"
        "(n_tickers, n_bars), 'adx' (n_tickers, n_bars, 3), 'atr'\n"
        "(n_tickers, n_bars), 'bollinger_bands' (n_tickers, n_bars, 3),\n"
        "'stochastic_oscillator' (n_tickers, n_bars, 2).\n\n"
        "Bit-identical to calling technical_indicators() once per ticker --\n"
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
           py::object pairs_obj,
           int max_lag, bool use_aic) -> py::array_t<double>
        {
            constexpr const char* fn = "batch_engle_granger";
            auto p_buf = prices.request();
            const IndexArray pairs = exact_int64(pairs_obj, "pairs", fn);
            if (p_buf.ndim != 2)
                throw std::invalid_argument(
                    "prices must be a 2-D array (n_tickers, n_bars)");
            if (pairs.ndim() != 2 || pairs.shape(1) != 2)
                throw std::invalid_argument("pairs must be a 2-D array (n_pairs, 2)");

            const auto n_tickers = static_cast<std::size_t>(p_buf.shape[0]);
            const auto n_bars    = static_cast<std::size_t>(p_buf.shape[1]);
            const auto n_pairs   = static_cast<std::size_t>(pairs.shape(0));

            // In int64, before narrowing to the kernel's int: a value that
            // only wrapped into range used to pass the kernel's own check.
            require_indices_below(pairs, static_cast<long long>(n_tickers),
                                  "pairs", fn, " (a ticker row of prices)");
            const std::vector<int> pairs_int = narrow_indices_to_int(
                pairs, static_cast<long long>(n_tickers), fn);

            constexpr py::ssize_t kCols = sqt::kBatchCointCols;
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n_pairs), kCols});
            double* out_ptr = out.mutable_data();
            const double* p_ptr = static_cast<const double*>(p_buf.ptr);
            const int*    q_ptr = pairs_int.data();
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
        "Engle-Granger over many pairs in one native call.\n\n"
        "prices : 2-D float64 (n_tickers, n_bars), already aligned onto a\n"
        "         common index by the caller -- this kernel never sees an\n"
        "         index and does no date alignment.\n"
        "pairs  : 2-D integer (n_pairs, 2), row indices into `prices`. Any\n"
        "         integer dtype; a float array is refused rather than\n"
        "         truncated.\n\n"
        "Returns a 2-D float64 array of shape (n_pairs, 11), one row per pair\n"
        "in input order. Columns (fixed order): intercept, hedge_ratio,\n"
        "adf_statistic, optimal_lag, p_value, cv_1pct, cv_5pct, cv_10pct,\n"
        "half_life, n_obs, cointegrated (0.0/1.0).\n\n"
        "Bit-identical to calling engle_granger() once per pair, and\n"
        "independent of thread count. Raises ValueError if a pairs row\n"
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

    // ── Modeling panel statistics ─────────────────────────────────────────
    m.def(
        "fit_preprocess_stats",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> values,
           double q_low, double q_high) -> py::dict
        {
            auto buf = values.request();
            if (buf.ndim != 2)
                throw std::invalid_argument(
                    "values must be 2-D (n_rows, n_cols)");
            if (!(q_low >= 0.0 && q_low <= 1.0) || !(q_high >= 0.0 && q_high <= 1.0))
                throw std::invalid_argument(
                    "q_low and q_high must each lie in [0, 1]");
            if (!(q_low < q_high))
                throw std::invalid_argument("need q_low < q_high");

            const auto n_rows = static_cast<std::size_t>(buf.shape[0]);
            const auto n_cols = static_cast<std::size_t>(buf.shape[1]);
            const auto nc = static_cast<py::ssize_t>(n_cols);

            py::array_t<double> a_lo(nc), a_hi(nc), a_mean(nc), a_std(nc);
            sqt::PreprocessStats out{a_lo.mutable_data(), a_hi.mutable_data(),
                                     a_mean.mutable_data(), a_std.mutable_data()};
            const double* ptr = values.data();
            bool ok = true;
            // An infinity is refused, not fitted. It has no finite neighbour
            // to interpolate a quantile towards and no z-score: the kernel's
            // bound came out +inf where pandas' came out NaN, and either way
            // the clipped mean was infinite and every transformed value -inf
            // or NaN. NaN stays a missing value the fit skips. Scanned
            // without the GIL; the refusal is raised once it is held again.
            std::size_t inf_col = n_cols;
            {
                py::gil_scoped_release release;
                for (std::size_t k = 0; k < n_rows * n_cols; ++k) {
                    if (std::isinf(ptr[k])) { inf_col = k % n_cols; break; }
                }
                if (inf_col == n_cols)
                    ok = sqt::fit_preprocess_stats(ptr, n_rows, n_cols,
                                                   q_low, q_high, out);
            }
            if (inf_col != n_cols)
                throw std::invalid_argument(
                    "fit_preprocess_stats: column " + std::to_string(inf_col) +
                    " holds an infinite value. An infinity has no quantile to "
                    "winsorize to and no z-score; mark it missing with NaN, "
                    "which the fit skips, or repair or drop the row");
            if (!ok)
                throw std::runtime_error(
                    "fit_preprocess_stats: could not allocate a column buffer");

            py::dict d;
            d["lo"] = a_lo;
            d["hi"] = a_hi;
            d["mean"] = a_mean;
            d["std"] = a_std;
            return d;
        },
        py::arg("values"),
        py::arg("q_low"),
        py::arg("q_high"),
        "Per-column winsorize bounds and clipped moments for a row-major\n"
        "(n_rows, n_cols) feature panel.\n\n"
        "Reproduces pandas exactly, including the conventions that are\n"
        "pandas' choice rather than mathematical necessity: quantiles are\n"
        "LINEARLY INTERPOLATED at h=(n-1)*q, the standard deviation is\n"
        "ddof=1, and NaN is skipped rather than propagated. An infinite\n"
        "value raises ValueError naming its column: it has no quantile to\n"
        "winsorize to and no z-score.\n\n"
        "A column with no finite values returns NaN bounds and mean with\n"
        "std=1.0; so does a constant column, so the caller's division stays\n"
        "defined.");

    m.def(
        "apply_preprocess_stats",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> values,
           py::array_t<double, py::array::c_style | py::array::forcecast> lo,
           py::array_t<double, py::array::c_style | py::array::forcecast> hi,
           py::array_t<double, py::array::c_style | py::array::forcecast> mean,
           py::array_t<double, py::array::c_style | py::array::forcecast> stdev)
            -> py::array_t<double>
        {
            auto buf = values.request();
            if (buf.ndim != 2)
                throw std::invalid_argument(
                    "values must be 2-D (n_rows, n_cols)");
            const auto n_rows = static_cast<std::size_t>(buf.shape[0]);
            const auto n_cols = static_cast<std::size_t>(buf.shape[1]);
            const auto expected = static_cast<py::ssize_t>(n_cols);
            if (lo.size() != expected || hi.size() != expected ||
                mean.size() != expected || stdev.size() != expected)
                throw std::invalid_argument(
                    "lo, hi, mean and std must each have one entry per column");

            // The stats are caller-supplied -- apply_preprocessing reads them
            // back from a persisted preprocessing_stats.json -- and the kernel
            // divides by std and clips to [lo, hi] without asking. std = 0
            // answered +/-inf and NaN, a negative std flipped every sign, and
            // lo > hi pinned every value to hi (where pandas' clip swaps the
            // bounds, so the two backends disagreed). NaN lo/hi/mean stay
            // legal: fit_preprocess_stats emits them for an all-NaN column,
            // with std = 1.0.
            const double* lo_p = lo.data();
            const double* hi_p = hi.data();
            const double* sd_p = stdev.data();
            for (py::ssize_t c = 0; c < expected; ++c) {
                if (!(sd_p[c] > 0.0) || !std::isfinite(sd_p[c]))
                    throw std::invalid_argument(
                        "apply_preprocess_stats: std[" + std::to_string(c) +
                        "] must be finite and > 0, got " + std::to_string(sd_p[c]) +
                        "; refit the statistics with fit_preprocess_stats, which "
                        "never produces one");
                if (lo_p[c] > hi_p[c])
                    throw std::invalid_argument(
                        "apply_preprocess_stats: lo[" + std::to_string(c) + "] = " +
                        std::to_string(lo_p[c]) + " is above hi[" +
                        std::to_string(c) + "] = " + std::to_string(hi_p[c]) +
                        "; the clip bounds must satisfy lo <= hi");
            }

            py::array_t<double> out({buf.shape[0], buf.shape[1]});
            sqt::PreprocessStats stats{
                const_cast<double*>(lo.data()), const_cast<double*>(hi.data()),
                const_cast<double*>(mean.data()), const_cast<double*>(stdev.data())};
            const double* ptr = values.data();
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::apply_preprocess_stats(ptr, n_rows, n_cols, stats, out_ptr);
            }
            return out;
        },
        py::arg("values"),
        py::arg("lo"),
        py::arg("hi"),
        py::arg("mean"),
        py::arg("std"),
        "Clip to [lo, hi] then standardize, in one fused pass.\n\n"
        "The Python form allocates two full-panel temporaries per column\n"
        "(the clip result and the standardized result); this allocates one\n"
        "output array and nothing else. NaN passes through untouched, which\n"
        "is what Series.clip does with a missing value.\n\n"
        "Raises ValueError for a std that is not finite and > 0, or a column\n"
        "whose lo is above its hi. NaN lo/hi/mean are accepted: they are what\n"
        "fit_preprocess_stats reports for an all-NaN column.");


    m.def(
        "cross_sectional_correlation",
        [](Array1D y_true,
           Array1D y_pred,
           py::object date_codes_obj,
           py::ssize_t n_dates,
           bool spearman) -> py::array_t<double>
        {
            constexpr const char* fn = "cross_sectional_correlation";
            // 1-D first: a (4, 2) input used to be flattened and answered.
            require_1d(y_true, "y_true");
            require_1d(y_pred, "y_pred");
            const IndexArray date_codes = exact_int64(date_codes_obj, "date_codes", fn);
            require_1d(date_codes, "date_codes");
            if (y_true.size() != y_pred.size() || y_true.size() != date_codes.size())
                throw std::invalid_argument(
                    "y_true, y_pred and date_codes must have the same length");
            if (n_dates < 0)
                throw std::invalid_argument("n_dates must be >= 0");
            require_indices_below(date_codes, static_cast<long long>(n_dates),
                                  "date_codes", fn, kFactorizeHint);

            const auto n_rows = static_cast<std::size_t>(y_true.size());
            py::array_t<double> out(n_dates);
            const double* yt = y_true.data();
            const double* yp = y_pred.data();
            const long long* codes = date_codes.data();
            double* out_ptr = out.mutable_data();
            bool ok = true;
            {
                py::gil_scoped_release release;
                ok = sqt::cross_sectional_correlation(
                    yt, yp, codes, n_rows, static_cast<std::size_t>(n_dates),
                    spearman, out_ptr);
            }
            if (!ok)
                throw std::runtime_error(
                    "cross_sectional_correlation: could not allocate a buffer");
            return out;
        },
        py::arg("y_true"),
        py::arg("y_pred"),
        py::arg("date_codes"),
        py::arg("n_dates"),
        py::arg("spearman"),
        "Per-date correlation between two aligned columns.\n\n"
        "Rows need not be sorted by date: the kernel counting-sorts them in\n"
        "O(n_rows), which replaces the caller's argsort and the two gathers\n"
        "that followed it. NaN PAIRS are dropped, matching Series.corr;\n"
        "infinities are kept, since pandas treats only NaN as missing.\n\n"
        "Returns one value per date, 0.0 where the correlation is undefined\n"
        "(fewer than two usable pairs, or a constant cross-section) -- the\n"
        "same 0.0-not-NaN contract the Python _safe_corr established.\n\n"
        "The POOLED correlation is this with n_dates=1 and all codes 0, so\n"
        "both share one implementation rather than drifting apart.\n\n"
        "All three arrays are 1-D; date_codes is an integer array with every\n"
        "code in [0, n_dates). Anything else raises ValueError.");

    m.def(
        "standardize_by_date",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> values,
           py::object date_codes_obj,
           py::ssize_t n_dates,
           double clip_sigma) -> py::array_t<double>
        {
            constexpr const char* fn = "standardize_by_date";
            auto buf = values.request();
            if (buf.ndim != 2)
                throw std::invalid_argument("values must be 2-D (n_rows, n_cols)");
            const IndexArray date_codes = exact_int64(date_codes_obj, "date_codes", fn);
            require_1d(date_codes, "date_codes");
            if (date_codes.size() != buf.shape[0])
                throw std::invalid_argument(
                    "date_codes must have one entry per row of values");
            if (n_dates < 0)
                throw std::invalid_argument("n_dates must be >= 0");
            if (!(clip_sigma >= 0.0))
                throw std::invalid_argument("clip_sigma must be >= 0");
            // A row whose code is outside [0, n_dates) belongs to no
            // cross-section; the kernel used to leave it unwritten, and the
            // caller got uninitialised memory back for that row.
            require_indices_below(date_codes, static_cast<long long>(n_dates),
                                  "date_codes", fn, kFactorizeHint);

            const auto n_rows = static_cast<std::size_t>(buf.shape[0]);
            const auto n_cols = static_cast<std::size_t>(buf.shape[1]);
            py::array_t<double> out({buf.shape[0], buf.shape[1]});
            const double* ptr = values.data();
            const long long* codes = date_codes.data();
            double* out_ptr = out.mutable_data();
            bool ok = true;
            {
                py::gil_scoped_release release;
                ok = sqt::standardize_by_date(ptr, n_rows, n_cols, codes,
                                              static_cast<std::size_t>(n_dates),
                                              clip_sigma, out_ptr);
            }
            if (!ok)
                throw std::runtime_error(
                    "standardize_by_date: could not allocate a buffer");
            return out;
        },
        py::arg("values"),
        py::arg("date_codes"),
        py::arg("n_dates"),
        py::arg("clip_sigma"),
        "Standardize every column within each date's cross-section.\n\n"
        "Subtracts that date's mean and divides by its ddof=1 standard\n"
        "deviation, then clips to +/- clip_sigma (0 disables). A date with no\n"
        "dispersion has every entity exactly at the mean, so those rows come\n"
        "back 0.0 rather than NaN -- NaN would drop the whole date\n"
        "downstream. NaN inputs are skipped by the moments and preserved in\n"
        "the output.\n\n"
        "date_codes is an integer array with every code in [0, n_dates);\n"
        "anything else raises ValueError. pd.factorize codes a NaT date as -1,\n"
        "so those rows need a code of their own or must be dropped first.");
    m.def(
        "rank_by_date",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> values,
           py::object date_codes_obj,
           py::ssize_t n_dates) -> py::array_t<double>
        {
            constexpr const char* fn = "rank_by_date";
            auto buf = values.request();
            if (buf.ndim != 2)
                throw std::invalid_argument("values must be 2-D (n_rows, n_cols)");
            const IndexArray date_codes = exact_int64(date_codes_obj, "date_codes", fn);
            require_1d(date_codes, "date_codes");
            if (date_codes.size() != buf.shape[0])
                throw std::invalid_argument(
                    "date_codes must have one entry per row of values");
            if (n_dates < 0)
                throw std::invalid_argument("n_dates must be >= 0");
            // Same reason as standardize_by_date: an out-of-range row was
            // never written.
            require_indices_below(date_codes, static_cast<long long>(n_dates),
                                  "date_codes", fn, kFactorizeHint);

            const auto n_rows = static_cast<std::size_t>(buf.shape[0]);
            const auto n_cols = static_cast<std::size_t>(buf.shape[1]);
            py::array_t<double> out({buf.shape[0], buf.shape[1]});
            const double* ptr = values.data();
            const long long* codes = date_codes.data();
            double* out_ptr = out.mutable_data();
            bool ok = true;
            {
                py::gil_scoped_release release;
                ok = sqt::rank_by_date(ptr, n_rows, n_cols, codes,
                                       static_cast<std::size_t>(n_dates), out_ptr);
            }
            if (!ok)
                throw std::runtime_error(
                    "rank_by_date: could not allocate a buffer");
            return out;
        },
        py::arg("values"),
        py::arg("date_codes"),
        py::arg("n_dates"),
        "Average rank of every value within its own date's cross-section.\n\n"
        "Ranks are 1-based and ties take the mean of the ordinals they span,\n"
        "matching Series.rank(method='average'). Ranking is per COLUMN within\n"
        "each date, so several models' predictions rank in one call. NaN is\n"
        "skipped by the ranking and preserved in the output, and does not\n"
        "shift the ranks of the values that are present.\n\n"
        "date_codes is an integer array with every code in [0, n_dates);\n"
        "anything else raises ValueError, as in standardize_by_date.");
    m.def(
        "permutation_null_ic",
        [](Array1D target,
           Array1D values,
           py::object date_codes_obj,
           py::ssize_t n_dates,
           py::ssize_t n_permutations,
           std::uint64_t seed,
           bool spearman) -> py::array_t<double>
        {
            constexpr const char* fn = "permutation_null_ic";
            require_1d(target, "target");
            require_1d(values, "values");
            const IndexArray date_codes = exact_int64(date_codes_obj, "date_codes", fn);
            require_1d(date_codes, "date_codes");
            if (target.size() != values.size() ||
                target.size() != date_codes.size())
                throw std::invalid_argument(
                    "target, values and date_codes must be the same length");
            if (n_dates < 0)
                throw std::invalid_argument("n_dates must be >= 0");
            if (n_permutations < 0)
                throw std::invalid_argument("n_permutations must be >= 0");
            require_indices_below(date_codes, static_cast<long long>(n_dates),
                                  "date_codes", fn, kFactorizeHint);

            py::array_t<double> out(n_permutations);
            const double* t_ptr = target.data();
            const double* v_ptr = values.data();
            const long long* codes = date_codes.data();
            const auto n_rows = static_cast<std::size_t>(target.size());
            double* out_ptr = out.mutable_data();
            bool ok = true;
            {
                py::gil_scoped_release release;
                ok = sqt::permutation_null_ic(
                    t_ptr, v_ptr, codes, n_rows,
                    static_cast<std::size_t>(n_dates),
                    static_cast<std::size_t>(n_permutations), seed, spearman,
                    out_ptr);
            }
            if (!ok)
                throw std::runtime_error(
                    "permutation_null_ic: could not allocate a buffer");
            return out;
        },
        py::arg("target"),
        py::arg("values"),
        py::arg("date_codes"),
        py::arg("n_dates"),
        py::arg("n_permutations"),
        py::arg("seed"),
        py::arg("spearman"),
        "Null distribution of a mean cross-sectional IC under within-date\n"
        "shuffling. One entry per draw: shuffle values inside each date,\n"
        "correlate against target within each date, average over the dates\n"
        "carrying at least two rows. For spearman the ranks are taken ONCE\n"
        "and shuffled directly, since permuting values permutes their\n"
        "ranks. `seed` reproduces a run within THIS backend only -- it is\n"
        "not numpy's PCG64 stream, the same contract\n"
        "simulate_forward_paths states. date_codes is an integer array with\n"
        "every code in [0, n_dates); anything else raises ValueError.");
    m.def(
        "label_uniqueness",
        [](py::object dates_obj,
           py::object label_end_obj,
           py::object entity_codes_obj,
           py::ssize_t n_entities) -> py::array_t<double>
        {
            constexpr const char* fn = "label_uniqueness";
            // Timestamps have no range to check -- NaT is INT64_MIN and
            // legal -- but they are still converted exactly: a float
            // timestamp was truncated on the way in.
            const IndexArray dates = exact_int64(dates_obj, "dates", fn);
            const IndexArray label_end = exact_int64(label_end_obj, "label_end", fn);
            const IndexArray entity_codes =
                exact_int64(entity_codes_obj, "entity_codes", fn);
            // 1-D: a (4, 2) input used to be flattened into eight labels.
            require_1d(dates, "dates");
            require_1d(label_end, "label_end");
            require_1d(entity_codes, "entity_codes");
            if (dates.size() != label_end.size() ||
                dates.size() != entity_codes.size())
                throw std::invalid_argument(
                    "dates, label_end and entity_codes must have the same length");
            if (n_entities < 0)
                throw std::invalid_argument("n_entities must be >= 0");
            // A row outside [0, n_entities) was dropped from the concurrency
            // count and left at weight 1.0, while the Python path grouped it
            // with the other unmatched rows -- the backends disagreed.
            require_indices_below(entity_codes, static_cast<long long>(n_entities),
                                  "entity_codes", fn, kFactorizeHint);

            const auto n_rows = static_cast<std::size_t>(dates.size());
            py::array_t<double> out(dates.size());
            const long long* d = dates.data();
            const long long* e = label_end.data();
            const long long* c = entity_codes.data();
            double* out_ptr = out.mutable_data();
            bool ok = true;
            {
                py::gil_scoped_release release;
                ok = sqt::label_uniqueness(d, e, c, n_rows,
                                           static_cast<std::size_t>(n_entities),
                                           out_ptr);
            }
            if (!ok)
                throw std::runtime_error(
                    "label_uniqueness: could not allocate a buffer");
            return out;
        },
        py::arg("dates"),
        py::arg("label_end"),
        py::arg("entity_codes"),
        py::arg("n_entities"),
        "Average uniqueness of each row's label, within its entity.\n\n"
        "Timestamps are nanoseconds since the epoch; numpy's NaT (INT64_MIN)\n"
        "marks a label that never resolves and spans only its own bar. Rows\n"
        "need not be sorted. Concurrency is accumulated with a difference\n"
        "array, which is O(n) where sweeping every label's span would be\n"
        "O(n * horizon).\n\n"
        "Returns weights normalized to mean 1, so enabling weighting does\n"
        "not also rescale the effective regularization strength.\n\n"
        "All three arrays are 1-D integer arrays; entity_codes must lie in\n"
        "[0, n_entities). Anything else raises ValueError.");
}
