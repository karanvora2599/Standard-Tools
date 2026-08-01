#include "sqt/cointegration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sqt {

namespace {

constexpr double kNaN  = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf  = std::numeric_limits<double>::infinity();
constexpr int    kMaxK = 16;  // max regressors in ADF system (max_lag + 2)
constexpr double kKalmanPriorVariance = 1.0e4;


// ── Gaussian elimination (partial pivoting) ───────────────────────────────────
// Solves A x = b in place (modifies both A and b).
// A is stored row-major, size k×k. Returns false if singular.

static bool gauss_elim(double A[], double b[], int k) {
    for (int col = 0; col < k; ++col) {
        // Partial pivot
        int pivot = col;
        for (int row = col + 1; row < k; ++row)
            if (std::abs(A[row * k + col]) > std::abs(A[pivot * k + col]))
                pivot = row;

        if (std::abs(A[pivot * k + col]) < 1e-14) return false;

        if (pivot != col) {
            for (int j = 0; j < k; ++j)
                std::swap(A[col * k + j], A[pivot * k + j]);
            std::swap(b[col], b[pivot]);
        }

        // Eliminate below
        for (int row = col + 1; row < k; ++row) {
            const double f = A[row * k + col] / A[col * k + col];
            for (int j = col; j < k; ++j)
                A[row * k + j] -= f * A[col * k + j];
            b[row] -= f * b[col];
        }
    }

    // Back-substitution
    for (int col = k - 1; col >= 0; --col) {
        b[col] /= A[col * k + col];
        for (int row = 0; row < col; ++row)
            b[row] -= A[row * k + col] * b[col];
    }
    return true;
}


// ── OLS via normal equations ──────────────────────────────────────────────────
// Returns: beta (length k), RSS, and the diagonal element of (X'X)^{-1}
// at position `diag_idx` (used for computing t-statistics).
// X is passed implicitly: caller fills XtX and Xty.

struct OlsCore {
    double beta[kMaxK];  // coefficients
    double rss;
    double diag_inv;     // (X'X)^{-1}[diag_idx, diag_idx]
    bool   ok;
};

static OlsCore ols_normal_eq(double XtX[], double Xty[], int k, int diag_idx) {
    OlsCore res;
    res.ok       = false;
    res.rss      = kNaN;
    res.diag_inv = kNaN;
    for (int i = 0; i < k; ++i) res.beta[i] = kNaN;

    // Solve for beta
    double A[kMaxK * kMaxK], b[kMaxK];
    std::copy(XtX, XtX + k * k, A);
    std::copy(Xty, Xty + k, b);
    if (!gauss_elim(A, b, k)) return res;
    std::copy(b, b + k, res.beta);

    // Solve for the `diag_idx`-th column of (X'X)^{-1}
    double A2[kMaxK * kMaxK], e[kMaxK] = {};
    std::copy(XtX, XtX + k * k, A2);
    e[diag_idx] = 1.0;
    if (!gauss_elim(A2, e, k)) return res;
    res.diag_inv = e[diag_idx];

    res.ok = true;
    return res;
}


// ── MacKinnon (2010) critical values ──────────────────────────────────────────
// Response surface: cv(T) = c_inf + c1/T + c2/T²
// For 2-variable cointegration, constant (trend="c").

static double mackinnon_cv(double level_pct, int n) {
    // Coefficients from MacKinnon (2010), Table 2, N=2, c.
    double c_inf, c1, c2;
    if (level_pct <= 1.0) {
        c_inf = -3.9001; c1 = -10.534; c2 = -30.030;
    } else if (level_pct <= 5.0) {
        c_inf = -3.3377; c1 = -5.967;  c2 = -8.982;
    } else {
        c_inf = -3.0462; c1 = -4.069;  c2 = -5.730;
    }
    const double T = static_cast<double>(n);
    return c_inf + c1 / T + c2 / (T * T);
}


// ── MacKinnon (2010) p-value ──────────────────────────────────────────────────
// Exact regression-surface algorithm for the 2-variable cointegration
// distribution (N=2, constant, "c"), not an interpolated lookup table.
//
// This is the real MacKinnon (1994/2010) response-surface method: fit a
// cubic (or, below a "star" breakpoint, quadratic) polynomial in the test
// statistic, then map through the standard normal CDF. The coefficients
// below are the exact regression-c/N=2 values statsmodels ships in
// tsa/adfvalues.py (_tau_maxs/_tau_mins/_tau_stars/_tau_smallps/_tau_largeps)
// -- reproducing statsmodels.tsa.stattools.mackinnonp(t, regression="c", N=2)
// to machine precision (verified numerically across [-20, 2] before this
// was written; see tests/test_cpp_cointegration.py). The previous
// hand-built 13-point lookup table this replaces was off by up to 0.08 in
// the middle of the distribution -- nowhere near the ±0.01-0.02 its own
// comment claimed.
//
// Reference: MacKinnon, J.G. 2010, "Critical Values for Cointegration
// Tests", Queen's Economics Department Working Paper No. 1227.

static double mackinnon_pvalue(double adf_stat) {
    constexpr double kTauMax  = 0.92;
    constexpr double kTauMin  = -18.86;
    constexpr double kTauStar = -2.62;

    if (adf_stat > kTauMax) return 1.0;
    if (adf_stat < kTauMin) return 0.0;

    double poly;
    if (adf_stat <= kTauStar) {
        // Quadratic branch, coefficients in ascending order [c0, c1, c2]
        constexpr double c0 = 2.92, c1 = 1.5012, c2 = 0.039796;
        poly = c2 * adf_stat * adf_stat + c1 * adf_stat + c0;
    } else {
        // Cubic branch, coefficients in ascending order [c0, c1, c2, c3]
        constexpr double c0 = 2.1945, c1 = 0.64695, c2 = -0.29198, c3 = -0.042377;
        poly = ((c3 * adf_stat + c2) * adf_stat + c1) * adf_stat + c0;
    }

    // Standard normal CDF via erf (no scipy/statsmodels dependency needed).
    return 0.5 * (1.0 + std::erf(poly / std::sqrt(2.0)));
}


// ── AR(1) half-life ───────────────────────────────────────────────────────────
// Fits Δy_t = alpha + beta * y_{t-1} and returns -ln(2)/beta.
// Returns +inf when beta >= 0 (not mean-reverting).

static double ar1_halflife(const double* y, std::size_t n) {
    if (n < 4) return kInf;

    // Build Δy and lagged y using the 2-var OLS helper
    const std::size_t m = n - 1;
    std::vector<double> dy(m), lag(m);
    for (std::size_t t = 0; t < m; ++t) {
        dy[t]  = y[t + 1] - y[t];
        lag[t] = y[t];
    }

    const auto res = ols2(dy.data(), lag.data(), m);
    const double beta = res.slope;
    // beta>=0.0 is false for NaN under IEEE754 (all NaN comparisons are
    // false), so a degenerate zero-variance lag series -- ols2's own
    // det<1e-14 guard returns slope=NaN for that case -- would otherwise
    // fall through to -log(2)/NaN = NaN instead of the same "not mean-
    // reverting" +inf sentinel a non-negative beta already gets. A
    // zero-variance predictor carries no information about mean
    // reversion, so it belongs in the same bucket as beta>=0, not a
    // separate silent NaN.
    if (!(beta < 0.0)) return kInf;
    return -std::log(2.0) / beta;
}

}  // namespace


// ── Public: ols2 ─────────────────────────────────────────────────────────────

Ols2Result ols2(const double* y, const double* x, std::size_t n) {
    Ols2Result r;
    r.residuals.resize(n, kNaN);
    r.intercept = r.slope = r.r_squared = kNaN;

    if (n < 2) return r;

    // Compute sums for normal equations
    double s1 = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (std::size_t i = 0; i < n; ++i) {
        s1  += 1.0;
        sx  += x[i];
        sy  += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
        syy += y[i] * y[i];
    }

    // Normal equations: [n, sx; sx, sxx] [b0; b1] = [sy; sxy]
    const double det = s1 * sxx - sx * sx;
    if (std::abs(det) < 1e-14) return r;

    r.intercept = (sy * sxx - sxy * sx) / det;
    r.slope     = (s1 * sxy - sx * sy) / det;

    double ss_res = 0, ss_tot = 0;
    const double y_mean = sy / s1;
    for (std::size_t i = 0; i < n; ++i) {
        const double pred = r.intercept + r.slope * x[i];
        r.residuals[i]    = y[i] - pred;
        ss_res += r.residuals[i] * r.residuals[i];
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    r.r_squared = (ss_tot > 0) ? 1.0 - ss_res / ss_tot : 0.0;
    return r;
}


// ── Public: adf_test ─────────────────────────────────────────────────────────

AdfResult adf_test(const double* y, std::size_t n, int max_lag, bool use_aic) {
    AdfResult out{kNaN, 0, kNaN};
    if (n < 4) return out;

    // Auto max lag: floor(12 * (n/100)^(1/4)), capped at (n-2)/3
    if (max_lag < 0) {
        max_lag = static_cast<int>(std::floor(12.0 * std::pow(n / 100.0, 0.25)));
        max_lag = std::min(max_lag, static_cast<int>((n - 2) / 3));
    }
    max_lag = std::min(max_lag, kMaxK - 2);  // never exceed kMaxK regressors

    // Pre-compute first differences
    const std::size_t nd = n - 1;
    std::vector<double> dy(nd);
    for (std::size_t i = 0; i < nd; ++i) dy[i] = y[i + 1] - y[i];

    // Degenerate input: y is (numerically) constant, so every regressor in
    // every candidate lag's design matrix (y_{t-1}, and every lagged Δy)
    // has zero variance -- the per-lag OLS solve below is singular for
    // every p, not just some, so the loop would never update best_t/best_ic
    // away from their initial NaN/+inf sentinels. This happens for real,
    // not just in theory: an Engle-Granger spread built from two perfectly
    // (or near-perfectly) collinear series is exactly y≡0. A constant
    // series is the strongest possible evidence AGAINST a unit root, not
    // "no evidence either way" -- statsmodels' adfuller()/coint() converge
    // on adf_statistic=-inf, p_value=0.0 for this exact case (verified
    // empirically here), so match that rather than surfacing NaN.
    {
        bool all_zero_diff = true;
        for (double d : dy) {
            if (d != 0.0) { all_zero_diff = false; break; }
        }
        if (all_zero_diff) {
            out.statistic   = -kInf;
            out.optimal_lag = 0;
            out.ic_min      = -kInf;
            return out;
        }
    }

    double best_ic  = kInf;
    double best_t   = kNaN;
    int    best_lag = 0;

    for (int p = 0; p <= max_lag; ++p) {
        // Number of usable observations: T = n - 1 - p
        const int T = static_cast<int>(n) - 1 - p;
        if (T < p + 3) break;  // too few obs for the k = p+2 regressors

        const int k = p + 2;  // constant + y_{t-1} + p lags of Δy

        // Build X'X (k×k) and X'y (k) by iterating over t = p+1 .. n-1
        double XtX[kMaxK * kMaxK] = {};
        double Xty[kMaxK]         = {};

        for (int t = p + 1; t < static_cast<int>(n); ++t) {
            // x[0]=1, x[1]=y[t-1], x[j+1]=dy[t-j] for j=1..p
            double xrow[kMaxK];
            xrow[0] = 1.0;
            xrow[1] = y[t - 1];
            // Δy_{t-j} = y[t-j] - y[t-j-1] = dy[t-j-1]
            for (int j = 1; j <= p; ++j) xrow[j + 1] = dy[t - 1 - j];

            const double response = dy[t - 1];  // Δy_t = y[t] - y[t-1] = dy[t-1]

            for (int i = 0; i < k; ++i) {
                for (int jj = i; jj < k; ++jj)
                    XtX[i * k + jj] += xrow[i] * xrow[jj];
                Xty[i] += xrow[i] * response;
            }
        }
        // Symmetrize
        for (int i = 0; i < k; ++i)
            for (int jj = i + 1; jj < k; ++jj)
                XtX[jj * k + i] = XtX[i * k + jj];

        // Solve
        const auto r = ols_normal_eq(XtX, Xty, k, /*diag_idx=*/1);
        if (!r.ok) continue;

        // RSS from beta: RSS = y'y - beta' X'y
        double yty = 0;
        for (int t = p + 1; t < static_cast<int>(n); ++t)
            yty += dy[t - 1] * dy[t - 1];

        double bXty = 0;
        for (int i = 0; i < k; ++i) bXty += r.beta[i] * Xty[i];
        const double rss = yty - bXty;
        if (rss <= 0) {
            // Degenerate: the regression fits perfectly (residual variance
            // is identically zero) -- happens when y itself is constant
            // (e.g. an Engle-Granger spread from two perfectly collinear
            // series). For a unit-root test this is the strongest possible
            // evidence AGAINST a unit root, not "no evidence either way" --
            // statsmodels' adfuller()/coint() converge on
            // adf_statistic=-inf, p_value=0.0 in this exact case (verified
            // empirically here). Previously this candidate was silently
            // skipped for every lag, so a perfectly collinear pair fell
            // through to the loop's NaN initial value instead of the
            // maximally-stationary result the math actually supports.
            if (best_ic > -kInf) {
                best_ic  = -kInf;
                best_lag = p;
                best_t   = -kInf;
            }
            continue;
        }

        const double df  = static_cast<double>(T - k);
        const double sig2 = rss / df;

        // Information criterion
        const double ic = use_aic
            ? std::log(sig2) + 2.0 * k / T
            : std::log(sig2) + std::log(static_cast<double>(T)) * k / T;

        if (ic < best_ic) {
            best_ic  = ic;
            best_lag = p;
            // t-statistic for φ (coefficient at index 1)
            const double se_phi = std::sqrt(sig2 * r.diag_inv);
            best_t = (se_phi > 0) ? r.beta[1] / se_phi : kNaN;
        }
    }

    out.statistic   = best_t;
    out.optimal_lag = best_lag;
    out.ic_min      = best_ic;
    return out;
}


// ── Public: engle_granger ────────────────────────────────────────────────────

CointResult engle_granger(
    const double* y0, const double* y1, std::size_t n,
    int max_lag, bool use_aic)
{
    CointResult r;
    r.intercept     = kNaN;
    r.hedge_ratio   = kNaN;
    r.adf_statistic = kNaN;
    r.optimal_lag   = 0;
    r.p_value       = kNaN;
    r.cv_1pct       = kNaN;
    r.cv_5pct       = kNaN;
    r.cv_10pct      = kNaN;
    r.half_life     = kInf;
    r.n_obs         = static_cast<int>(n);
    r.cointegrated  = false;

    if (n < 8) return r;

    // ── Step 1: OLS y0 = intercept + hedge_ratio * y1 ────────────────────────
    const auto ols = ols2(y0, y1, n);
    r.intercept   = ols.intercept;
    r.hedge_ratio = ols.slope;

    // ── Step 2: ADF test on the spread (OLS residuals) ───────────────────────
    const auto adf = adf_test(ols.residuals.data(), n, max_lag, use_aic);
    r.adf_statistic = adf.statistic;
    r.optimal_lag   = adf.optimal_lag;

    // ── Step 3: MacKinnon (2010) critical values and p-value ─────────────────
    r.cv_1pct  = mackinnon_cv(1.0,  static_cast<int>(n));
    r.cv_5pct  = mackinnon_cv(5.0,  static_cast<int>(n));
    r.cv_10pct = mackinnon_cv(10.0, static_cast<int>(n));

    if (!std::isnan(adf.statistic)) {
        r.p_value     = mackinnon_pvalue(adf.statistic);
        r.cointegrated = (r.p_value < 0.05);
    }

    // ── Step 4: AR(1) half-life of the spread ────────────────────────────────
    r.half_life = ar1_halflife(ols.residuals.data(), n);

    return r;
}

// ── Kalman filters (time-varying hedge ratio) ─────────────────────────────────
//
// Sequential predict/update recursion -- state at t depends on state at
// t-1, so this can't be vectorized in plain numpy, same shape as the
// already-ported RSI/PSAR indicators. Matches _kalman_filter_1state /
// _kalman_filter_2state in analysis/cointegration.py exactly.

Kalman1StateResult kalman_filter_1state(
    const double* y, const double* x, std::size_t n,
    double delta, double observation_noise)
{
    Kalman1StateResult result;
    if (n == 0 || !(delta > 0.0 && delta < 1.0) || !(observation_noise > 0.0)) {
        return result;
    }

    result.beta.resize(n);
    result.gain.resize(n);
    result.innovation.resize(n);

    const double vw = delta / (1.0 - delta);
    double beta_prev = 0.0;
    double p_prev = kKalmanPriorVariance;

    for (std::size_t t = 0; t < n; ++t) {
        const double r = p_prev + vw;
        const double y_hat = beta_prev * x[t];
        const double q = r * x[t] * x[t] + observation_noise;
        const double e = y[t] - y_hat;
        const double k = r * x[t] / q;

        const double beta_t = beta_prev + k * e;
        const double p_t = r - k * x[t] * r;

        result.beta[t] = beta_t;
        result.gain[t] = k;
        result.innovation[t] = e;

        beta_prev = beta_t;
        p_prev = p_t;
    }

    return result;
}

Kalman2StateResult kalman_filter_2state(
    const double* y, const double* x, std::size_t n,
    double delta, double observation_noise)
{
    Kalman2StateResult result;
    if (n == 0 || !(delta > 0.0 && delta < 1.0) || !(observation_noise > 0.0)) {
        return result;
    }

    result.alpha.resize(n);
    result.beta.resize(n);
    result.gain.resize(n);
    result.innovation.resize(n);

    const double vw = delta / (1.0 - delta);
    double alpha_prev = 0.0;
    double beta_prev = 0.0;
    double p00 = kKalmanPriorVariance, p01 = 0.0, p11 = kKalmanPriorVariance;

    for (std::size_t t = 0; t < n; ++t) {
        const double r00 = p00 + vw;
        const double r01 = p01;
        const double r11 = p11 + vw;

        const double xt = x[t];
        const double q = r00 + 2.0 * r01 * xt + r11 * xt * xt + observation_noise;
        const double e = y[t] - (alpha_prev + beta_prev * xt);

        const double rx0 = r00 + r01 * xt;
        const double rx1 = r01 + r11 * xt;
        const double k0 = rx0 / q;
        const double k1 = rx1 / q;

        const double alpha_t = alpha_prev + k0 * e;
        const double beta_t = beta_prev + k1 * e;

        const double p00_t = r00 - k0 * rx0;
        const double p01_t = r01 - k0 * rx1;
        const double p11_t = r11 - k1 * rx1;

        result.alpha[t] = alpha_t;
        result.beta[t] = beta_t;
        result.gain[t] = k1;
        result.innovation[t] = e;

        alpha_prev = alpha_t;
        beta_prev = beta_t;
        p00 = p00_t;
        p01 = p01_t;
        p11 = p11_t;
    }

    return result;
}

}  // namespace sqt
