#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sqt/hurst.hpp"

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
}
