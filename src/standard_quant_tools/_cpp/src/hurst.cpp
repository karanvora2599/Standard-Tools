#include "sqt/hurst.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

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

std::pair<std::vector<double>, std::vector<double>>
dfa(const double* arr, std::size_t n, int min_w, int max_w, int n_points) {
    // Step 1: mean-centred cumulative sum
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += arr[i];
    mean /= static_cast<double>(n);

    std::vector<double> y(n);
    double cs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cs  += arr[i] - mean;
        y[i] = cs;
    }

    const auto sizes = log_sizes(min_w, max_w, n_points);
    std::vector<double> flucts, valid_sizes;

    for (const int sz : sizes) {
        const int n_chunks = static_cast<int>(n) / sz;
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
        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const double* seg      = y.data() + static_cast<std::size_t>(chunk * sz);
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

        flucts.push_back(std::sqrt(rms_acc / n_chunks));
        valid_sizes.push_back(static_cast<double>(sz));
    }

    return {valid_sizes, flucts};
}


// ── R/S analysis ──────────────────────────────────────────────────────────────

std::pair<std::vector<double>, std::vector<double>>
rs_analysis(const double* arr, std::size_t n, int min_w, int max_w, int n_points) {
    const auto sizes = log_sizes(min_w, max_w, n_points);
    std::vector<double> rs_vals, valid_sizes;

    for (const int sz : sizes) {
        const int n_chunks = static_cast<int>(n) / sz;
        if (n_chunks < 1) continue;

        double rs_acc = 0.0;
        int    count  = 0;

        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const double* c    = arr + static_cast<std::size_t>(chunk * sz);
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
            rs_vals.push_back(rs_acc / count);
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

    // min_window <= 0 would otherwise reach log10() in log_sizes() with a
    // non-positive argument (NaN, not a crash, but relying on that NaN to
    // propagate cleanly through every downstream branch is fragile) —
    // reject explicitly instead.
    if (min_window <= 0) return nan_result;

    // Auto-select max window: n/4 for DFA (less biased), n/2 for R/S
    const int max_w_auto = (method == "dfa")
        ? static_cast<int>(n) / 4
        : static_cast<int>(n) / 2;
    const int max_w = (max_window <= 0)
        ? max_w_auto
        : std::min(max_window, max_w_auto);

    // Need at least 4 min-windows of data
    if (static_cast<int>(n) < min_window * 4 || min_window >= max_w)
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
    h = std::clamp(h, 0.0, 1.5);

    return {h, classify(h), r2, method, n};
}


// ── rolling_hurst ─────────────────────────────────────────────────────────────

std::vector<double> rolling_hurst(
    const double*      arr,
    std::size_t        n,
    int                window,
    int                step,
    const std::string& method,
    int                min_window)
{
    std::vector<double> out(n, kNaN);

    // step <= 0 makes the loop below non-progressing (i += step never
    // advances, or moves backward) — an infinite native loop that hangs
    // the process rather than raising. window <= 0 is equally nonsensical
    // (the slice below would be empty or reversed). Reject both up front.
    if (step <= 0 || window <= 0) return out;

    for (int i = window - 1; i < static_cast<int>(n); i += step) {
        const auto result = hurst_exponent(
            arr + static_cast<std::size_t>(i - window + 1),
            static_cast<std::size_t>(window),
            method,
            min_window,
            /*max_window=*/-1);   // auto per chunk size
        out[i] = result.hurst;
    }

    return out;
}

}  // namespace sqt
