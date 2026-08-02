#include "sqt/hurst.hpp"

#include "sqt/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#ifdef SQT_HAS_OPENMP
#include <omp.h>
#endif

namespace sqt {

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::string classify(double h) {
    if (h > 0.55) return "trending";
    if (h < 0.45) return "mean_reverting";
    return "random_walk";
}

}  // namespace


// ── Utility functions ─────────────────────────────────────────────────────────

std::vector<int> log_sizes(int min_w, int max_w, int n_points) {
    std::unordered_set<int> seen;
    std::vector<int>        sizes;

    const double log_min = std::log10(static_cast<double>(min_w));
    const double log_max = std::log10(static_cast<double>(max_w));

    for (int i = 0; i < n_points; ++i) {
        const double t  = (n_points > 1) ? static_cast<double>(i) / (n_points - 1) : 0.0;
        const int    sz = static_cast<int>(std::pow(10.0, log_min + t * (log_max - log_min)));
        if (sz >= min_w && sz <= max_w && seen.insert(sz).second) {
            sizes.push_back(sz);
        }
    }
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}


std::pair<double, double> ols_slope_r2(
    const std::vector<double>& x,
    const std::vector<double>& y)
{
    const int m = static_cast<int>(x.size());
    if (m < 2) return {0.0, 0.0};

    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < m; ++i) {
        sx  += x[i];
        sy  += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }

    const double denom = m * sxx - sx * sx;
    if (std::abs(denom) < 1e-14) return {0.0, 0.0};

    const double slope     = (m * sxy - sx * sy) / denom;
    const double intercept = (sy - slope * sx) / m;

    const double y_mean = sy / m;
    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < m; ++i) {
        const double residual = y[i] - (intercept + slope * x[i]);
        ss_res += residual * residual;
        const double dev = y[i] - y_mean;
        ss_tot += dev * dev;
    }
    const double r2 = (ss_tot > 1e-14) ? 1.0 - ss_res / ss_tot : 0.0;

    return {slope, r2};
}


// ── DFA-1 ─────────────────────────────────────────────────────────────────────

namespace {

// Per-thread scratch reused across every rolling window that thread
// processes, instead of a fresh allocation per window. Declared here
// (ahead of dfa_impl/dfa_onepass) so both can reference it.
struct RollingHurstScratch {
    std::vector<double> y;
};

// Shared implementation behind both the public dfa() and the internal
// scratch-based rolling_hurst_into() fast path. `y_scratch`, if non-null, is
// a caller-owned buffer reused across many calls (one per rolling window)
// instead of a fresh `std::vector<double> y(n)` allocation every call;
// passing nullptr reproduces dfa()'s original always-allocate-locally
// behavior exactly, so dfa() itself (public, standalone-tested) stays
// byte-identical in every way that matters -- same code path, just an extra
// indirection through a pointer that's always null on that path.
std::pair<std::vector<double>, std::vector<double>>
dfa_impl(const double* arr, std::size_t n, int min_w, int max_w, int n_points,
          std::vector<double>* y_scratch)
{
    // Step 1: mean-centred cumulative sum
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += arr[i];
    mean /= static_cast<double>(n);

    std::vector<double> local_y;
    std::vector<double>& y = y_scratch ? *y_scratch : local_y;
    y.resize(n);
    double cs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cs  += arr[i] - mean;
        y[i] = cs;
    }

    const auto sizes = log_sizes(min_w, max_w, n_points);
    std::vector<double> flucts, valid_sizes;

    for (const int sz : sizes) {
        // n_chunks is n/sz, which can exceed INT_MAX for a large series
        // even though sz itself (a window size) fits comfortably in int --
        // compute and iterate it in size_t space rather than narrowing n
        // first, which would silently wrap for n > INT_MAX.
        const std::size_t sz_sz    = static_cast<std::size_t>(sz);
        const std::size_t n_chunks = n / sz_sz;
        if (n_chunks < 2) continue;   // need ≥ 2 chunks for a meaningful fit

        // Precompute x-statistics (same for every chunk of size sz)
        const double x_mean = (sz - 1) * 0.5;
        double x_var = 0.0;
        for (int j = 0; j < sz; ++j) {
            const double d = j - x_mean;
            x_var += d * d;
        }
        x_var /= sz;  // mean of squared deviations, matching numpy's .mean()

        double rms_acc = 0.0;
        for (std::size_t chunk = 0; chunk < n_chunks; ++chunk) {
            const double* seg      = y.data() + chunk * sz_sz;
            double        seg_mean = 0.0;
            for (int j = 0; j < sz; ++j) seg_mean += seg[j];
            seg_mean /= sz;

            // Analytic linear-detrend coefficients
            // b = mean((x-x_mean)*(seg-seg_mean)) / x_var
            //   = [sum_j (j-x_mean)*(seg_j-seg_mean) / sz] / x_var
            double cross = 0.0;
            for (int j = 0; j < sz; ++j)
                cross += (j - x_mean) * (seg[j] - seg_mean);
            const double b = (x_var > 0.0) ? cross / (sz * x_var) : 0.0;
            const double a = seg_mean - b * x_mean;

            // RMS of residuals for this chunk (matches (residuals**2).mean())
            double rms = 0.0;
            for (int j = 0; j < sz; ++j) {
                const double r = seg[j] - (a + b * j);
                rms += r * r;
            }
            rms_acc += rms / sz;
        }

        flucts.push_back(std::sqrt(rms_acc / static_cast<double>(n_chunks)));
        valid_sizes.push_back(static_cast<double>(sz));
    }

    return {valid_sizes, flucts};
}

// One-pass reformulation of dfa_impl()'s per-chunk math, used ONLY by
// hurst_exponent_scratch()'s "dfa" branch (item H) -- the public dfa()/
// dfa_impl() above stay on the original 3-pass arithmetic permanently, so
// this genuine reassociation never touches the standalone-tested public
// function. Two algebraic identities (both exact at the OLS optimum, not
// approximations) collapse dfa_impl()'s 3 passes over each chunk down to 1:
//
//   cross = sum_j (j-x_mean)*(y_j-seg_mean)
//         = sum_j(j*y_j) - x_mean*sum_j(y_j)      [the seg_mean cross-term
//                                                    cancels exactly, since
//                                                    sum_j(j-x_mean) == 0
//                                                    on the fixed integer
//                                                    grid 0..sz-1]
//         = S_jy - x_mean*Sy
//
//   SSE = sum_j (y_j - (a+b*j))^2 = Syy - a*Sy - b*S_jy   [standard OLS
//         sufficient-statistics identity -- holds exactly because (a,b) are
//         the actual least-squares fit, giving Sum(residual)=0 and
//         Sum(j*residual)=0 at the optimum]
//
// So one pass accumulating Sy, Syy, S_jy per chunk is enough; no second
// pass for `cross` and no third pass for the residual sum of squares.
// x_var also replaced with its closed form (sz^2-1)/12 -- population
// variance of the integers 0..sz-1 -- instead of a per-size loop.
//
// This is a genuine floating-point reassociation (sum-of-squares style
// accumulation is less numerically robust than deviation-from-mean style,
// in general) -- NOT assumed bit-identical to dfa_impl(); see
// tests/cpp/test_hurst.cpp's tolerance-gated comparison, including
// deliberately ill-conditioned inputs, before this is trusted.
std::pair<std::vector<double>, std::vector<double>>
dfa_onepass(const double* arr, std::size_t n, int min_w, int max_w, int n_points,
             RollingHurstScratch& scratch)
{
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += arr[i];
    mean /= static_cast<double>(n);

    std::vector<double>& y = scratch.y;
    y.resize(n);
    double cs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cs  += arr[i] - mean;
        y[i] = cs;
    }

    const auto sizes = log_sizes(min_w, max_w, n_points);
    std::vector<double> flucts, valid_sizes;

    for (const int sz : sizes) {
        // See dfa_impl()'s matching comment: n_chunks = n/sz can exceed
        // INT_MAX for a large series even though sz itself fits in int.
        const std::size_t sz_sz    = static_cast<std::size_t>(sz);
        const std::size_t n_chunks = n / sz_sz;
        if (n_chunks < 2) continue;

        const double x_mean = (sz - 1) * 0.5;
        // Closed form: mean of squared deviations of 0..sz-1 from x_mean.
        const double x_var = (static_cast<double>(sz) * sz - 1.0) / 12.0;

        double rms_acc = 0.0;
        for (std::size_t chunk = 0; chunk < n_chunks; ++chunk) {
            const double* seg = y.data() + chunk * sz_sz;

            double Sy = 0.0, Syy = 0.0, S_jy = 0.0;
            for (int j = 0; j < sz; ++j) {
                const double yj = seg[j];
                Sy   += yj;
                Syy  += yj * yj;
                S_jy += j * yj;
            }
            const double seg_mean = Sy / sz;

            const double cross = S_jy - x_mean * Sy;
            const double b = (x_var > 0.0) ? cross / (sz * x_var) : 0.0;
            const double a = seg_mean - b * x_mean;

            const double sse = Syy - a * Sy - b * S_jy;
            rms_acc += sse / sz;
        }

        flucts.push_back(std::sqrt(rms_acc / static_cast<double>(n_chunks)));
        valid_sizes.push_back(static_cast<double>(sz));
    }

    return {valid_sizes, flucts};
}

}  // namespace

std::pair<std::vector<double>, std::vector<double>>
dfa(const double* arr, std::size_t n, int min_w, int max_w, int n_points) {
    return dfa_impl(arr, n, min_w, max_w, n_points, nullptr);
}


// ── R/S analysis ──────────────────────────────────────────────────────────────

std::pair<std::vector<double>, std::vector<double>>
rs_analysis(const double* arr, std::size_t n, int min_w, int max_w, int n_points) {
    const auto sizes = log_sizes(min_w, max_w, n_points);
    std::vector<double> rs_vals, valid_sizes;

    for (const int sz : sizes) {
        // See dfa_impl()'s matching comment: n_chunks = n/sz can exceed
        // INT_MAX for a large series even though sz itself fits in int.
        const std::size_t sz_sz    = static_cast<std::size_t>(sz);
        const std::size_t n_chunks = n / sz_sz;
        if (n_chunks < 1) continue;

        double      rs_acc = 0.0;
        std::size_t count  = 0;

        for (std::size_t chunk = 0; chunk < n_chunks; ++chunk) {
            const double* c    = arr + chunk * sz_sz;
            double        mean = 0.0;
            for (int j = 0; j < sz; ++j) mean += c[j];
            mean /= sz;

            // Range of cumulative mean-adjusted deviations
            double cs = 0.0, R_max = -1e300, R_min = 1e300;
            // Variance (ddof=1 to match numpy chunk.std(ddof=1))
            double var = 0.0;
            for (int j = 0; j < sz; ++j) {
                cs += c[j] - mean;
                if (cs > R_max) R_max = cs;
                if (cs < R_min) R_min = cs;
                const double d = c[j] - mean;
                var += d * d;
            }
            const double R = R_max - R_min;
            const double S = (sz > 1) ? std::sqrt(var / (sz - 1)) : 0.0;

            if (S > 0.0) {
                rs_acc += R / S;
                ++count;
            }
        }

        if (count > 0) {
            rs_vals.push_back(rs_acc / static_cast<double>(count));
            valid_sizes.push_back(static_cast<double>(sz));
        }
    }

    return {valid_sizes, rs_vals};
}


// ── hurst_exponent ────────────────────────────────────────────────────────────

HurstResult hurst_exponent(
    const double*      arr,
    std::size_t        n,
    const std::string& method,
    int                min_window,
    int                max_window)
{
    const HurstResult nan_result{kNaN, "unknown", kNaN, method, n};

    // Anything other than exactly "dfa"/"rs" used to fall through the
    // if/else below straight to R/S -- silently substituting a different
    // (and, per the header's own doc comment, upward-biased-for-short-series)
    // estimator for a caller's typo'd or invalid method string, rather than
    // honoring the documented "dfa" or "rs" contract. Reject explicitly.
    if (method != "dfa" && method != "rs")
        throw std::invalid_argument(
            "hurst_exponent: method must be exactly \"dfa\" or \"rs\", got \"" +
            method + "\"");

    // min_window <= 0 would otherwise reach log10() in log_sizes() with a
    // non-positive argument (NaN, not a crash, but relying on that NaN to
    // propagate cleanly through every downstream branch is fragile) —
    // reject explicitly instead.
    if (min_window <= 0) return nan_result;

    // Auto-select max window: n/4 for DFA (less biased), n/2 for R/S.
    // Computed in size_t space (n can exceed INT_MAX for a large series)
    // and only narrowed to int -- window sizes are int throughout this
    // file's public API -- via a checked cast that fails loud instead of
    // silently wrapping if a series is so large the auto-selected window
    // itself wouldn't fit in an int.
    const std::size_t max_w_auto_sz = (method == "dfa") ? (n / 4) : (n / 2);
    const int max_w_auto = numerics::checked_narrow_to_int(
        max_w_auto_sz, "hurst_exponent: auto-selected max_window");
    const int max_w = (max_window <= 0)
        ? max_w_auto
        : std::min(max_window, max_w_auto);

    // Need at least 4 min-windows of data. Compared in size_t space (not
    // narrowing n to int first) and with min_window*4 computed in size_t
    // space too, since min_window is caller-supplied and could itself be
    // large enough for an int*int multiplication to overflow.
    if (n < static_cast<std::size_t>(min_window) * 4 || min_window >= max_w)
        return nan_result;

    std::vector<double> sizes, values;
    if (method == "dfa") {
        auto [s, v] = dfa(arr, n, min_window, max_w);
        sizes  = std::move(s);
        values = std::move(v);
    } else {
        auto [s, v] = rs_analysis(arr, n, min_window, max_w);
        sizes  = std::move(s);
        values = std::move(v);
    }

    // Need ≥ 3 points for a meaningful log-log fit
    if (sizes.size() < 3) return nan_result;

    // All fluctuations/RS values must be positive for log
    for (const double v : values)
        if (v <= 0.0) return nan_result;

    std::vector<double> log_s(sizes.size()), log_v(values.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        log_s[i] = std::log(sizes[i]);
        log_v[i] = std::log(values[i]);
    }

    auto [h, r2] = ols_slope_r2(log_s, log_v);

    // std::clamp's behavior is unspecified if the value being clamped is
    // NaN (the standard requires v/lo/hi to be well-ordered by <, which NaN
    // never is) -- guard explicitly rather than relying on clamp+classify's
    // string-threshold comparisons (all false for NaN) to silently fall
    // through to a plausible-looking but wrong regime label.
    if (std::isnan(h)) return nan_result;

    h = std::clamp(h, 0.0, 1.5);

    return {h, classify(h), r2, method, n};
}

namespace {

// Mirrors hurst_exponent() exactly, except the "dfa" branch calls
// dfa_impl(..., &scratch.y) instead of the public dfa() -- reusing the
// scratch buffer instead of allocating a fresh one per call. The "rs"
// branch is unchanged (calls rs_analysis() directly, as hurst_exponent()
// does) -- not worth threading scratch through for its small
// n_points-bounded vectors. Used only by rolling_hurst_into() below; the
// public hurst_exponent() is untouched.
HurstResult hurst_exponent_scratch(
    const double* arr, std::size_t n, const std::string& method,
    int min_window, int max_window, RollingHurstScratch& scratch)
{
    const HurstResult nan_result{kNaN, "unknown", kNaN, method, n};

    if (min_window <= 0) return nan_result;

    // See hurst_exponent()'s matching comment: computed in size_t space and
    // checked-narrowed rather than truncating n to int first.
    const std::size_t max_w_auto_sz = (method == "dfa") ? (n / 4) : (n / 2);
    const int max_w_auto = numerics::checked_narrow_to_int(
        max_w_auto_sz, "hurst_exponent_scratch: auto-selected max_window");
    const int max_w = (max_window <= 0)
        ? max_w_auto
        : std::min(max_window, max_w_auto);

    if (n < static_cast<std::size_t>(min_window) * 4 || min_window >= max_w)
        return nan_result;

    std::vector<double> sizes, values;
    if (method == "dfa") {
        auto [s, v] = dfa_onepass(arr, n, min_window, max_w, /*n_points=*/20, scratch);
        sizes  = std::move(s);
        values = std::move(v);
    } else {
        auto [s, v] = rs_analysis(arr, n, min_window, max_w);
        sizes  = std::move(s);
        values = std::move(v);
    }

    if (sizes.size() < 3) return nan_result;

    for (const double v : values)
        if (v <= 0.0) return nan_result;

    std::vector<double> log_s(sizes.size()), log_v(values.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        log_s[i] = std::log(sizes[i]);
        log_v[i] = std::log(values[i]);
    }

    auto [h, r2] = ols_slope_r2(log_s, log_v);

    if (std::isnan(h)) return nan_result;

    h = std::clamp(h, 0.0, 1.5);

    return {h, classify(h), r2, method, n};
}

}  // namespace

// ── rolling_hurst ─────────────────────────────────────────────────────────────

void rolling_hurst_into(
    const double* SQT_RESTRICT      arr,
    std::size_t        n,
    int                window,
    int                step,
    const std::string& method,
    int                min_window,
    double* SQT_RESTRICT            out)
{
    std::fill(out, out + n, kNaN);

    // Validate eagerly rather than relying on the first hurst_exponent_scratch()
    // call inside the loop below to catch it -- for a short input (n <
    // window) that loop runs zero times, so a bad method string would
    // otherwise silently produce an all-NaN result instead of raising. Also
    // ensures no exception can be thrown from inside the parallel region
    // below -- throwing across an #pragma omp for boundary is undefined
    // behavior / terminates the process.
    if (method != "dfa" && method != "rs")
        throw std::invalid_argument(
            "rolling_hurst: method must be exactly \"dfa\" or \"rs\", got \"" +
            method + "\"");

    // step <= 0 makes the loop below non-progressing (i += step never
    // advances, or moves backward) — an infinite native loop that hangs
    // the process rather than raising. window <= 0 is equally nonsensical
    // (the slice below would be empty or reversed). Reject both up front.
    if (step <= 0 || window <= 0) return;
    if (n == 0 || static_cast<std::size_t>(window) > n) return;

    // Precompute the window-position count so the parallel loop below can
    // use a counted, unit-stride induction variable derived from it --
    // OpenMP's canonical-for-loop form technically permits `i += step`
    // directly (a loop-invariant increment is allowed by the spec), but
    // this codebase's only prior OpenMP loop (monte_carlo.cpp) is
    // unit-stride, so there was no local precedent confirming every
    // targeted compiler accepts the strided form cleanly; a counted
    // rewrite is unambiguously canonical everywhere and was confirmed to
    // build correctly on this project's MSVC toolchain.
    //
    // Computed and iterated in size_t/long long, not int: n (and so the
    // window-position count, particularly at step==1) can exceed INT_MAX
    // for a large series -- narrowing to int here would silently wrap.
    // long long (not int) is used for the OpenMP induction variable
    // itself, matching the precedent already established in
    // backtest.cpp's batch_run_strategy: MSVC's OpenMP 2.0 canonical-for
    // form requires a signed integer induction variable, and long long
    // qualifies just as int does, while covering the full practical range.
    const std::size_t win_sz  = static_cast<std::size_t>(window);
    const std::size_t last_i  = n - 1;
    const std::size_t step_sz = static_cast<std::size_t>(step);
    const long long count = static_cast<long long>((last_i - (win_sz - 1)) / step_sz + 1);

#ifdef SQT_HAS_OPENMP
    #pragma omp parallel if(count > 1)
#endif
    {
        RollingHurstScratch scratch;
        scratch.y.reserve(win_sz);

#ifdef SQT_HAS_OPENMP
        #pragma omp for schedule(static)
#endif
        for (long long idx = 0; idx < count; ++idx) {
            const std::size_t i = (win_sz - 1) + static_cast<std::size_t>(idx) * step_sz;
            const auto result = hurst_exponent_scratch(
                arr + (i - win_sz + 1),
                win_sz,
                method,
                min_window,
                /*max_window=*/-1,   // auto per chunk size
                scratch);
            out[i] = result.hurst;
        }
    }
}

std::vector<double> rolling_hurst(
    const double*      arr,
    std::size_t        n,
    int                window,
    int                step,
    const std::string& method,
    int                min_window)
{
    std::vector<double> out(n);
    rolling_hurst_into(arr, n, window, step, method, min_window, out.data());
    return out;
}

}  // namespace sqt
