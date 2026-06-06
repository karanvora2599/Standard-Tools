#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sqt/hurst.hpp"
#include "sqt/indicators.hpp"

namespace py = pybind11;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Force the input to be a 1-D, C-contiguous float64 array.
using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;

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

PYBIND11_MODULE(_sqt_core, m) {
    m.doc() =
        "SQT C++ core — high-performance implementations of computationally "
        "intensive functions.  Import via the public Python modules; do not "
        "call these entry-points directly.";

    // ── Hurst exponent ────────────────────────────────────────────────────────

    m.def(
        "hurst_dfa",
        [](Array1D arr, int min_window, int max_window) -> py::dict {
            return hurst_result_to_dict(
                sqt::hurst_exponent(arr.data(), arr.size(), "dfa", min_window, max_window));
        },
        py::arg("arr"),
        py::arg("min_window") = 10,
        py::arg("max_window") = -1,
        "Hurst exponent via Detrended Fluctuation Analysis (DFA-1).\n\n"
        "Pass max_window=-1 to auto-select (n//4).");

    m.def(
        "hurst_rs",
        [](Array1D arr, int min_window, int max_window) -> py::dict {
            return hurst_result_to_dict(
                sqt::hurst_exponent(arr.data(), arr.size(), "rs", min_window, max_window));
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
            auto result = sqt::rolling_hurst(
                arr.data(), arr.size(), window, step, method, min_window);
            // Return a new 1-D numpy array (copy of result vector)
            py::array_t<double> out(static_cast<py::ssize_t>(result.size()));
            std::copy(result.begin(), result.end(), out.mutable_data());
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
            auto result = sqt::rsi(arr.data(), arr.size(), period);
            py::array_t<double> out(static_cast<py::ssize_t>(result.size()));
            std::copy(result.begin(), result.end(), out.mutable_data());
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
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const auto n = high.size();
            auto result  = sqt::adx(high.data(), low.data(), close.data(), n, period);
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(3)});
            std::copy(result.begin(), result.end(), out.mutable_data());
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
            if (high.size() != low.size())
                throw std::invalid_argument("high and low must have equal length");
            const auto n = high.size();
            auto result  = sqt::parabolic_sar(
                high.data(), low.data(), n, af_start, af_step, af_max);
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(2)});
            std::copy(result.begin(), result.end(), out.mutable_data());
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
}
