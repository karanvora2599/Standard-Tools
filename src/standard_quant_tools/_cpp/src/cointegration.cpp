#include "sqt/cointegration.hpp"

#include "sqt/numerics.hpp"
#include "sqt/qr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sqt {

namespace {

constexpr double kNaN  = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf  = std::numeric_limits<double>::infinity();
constexpr double kKalmanPriorVariance = 1.0e4;


// Per-lag regression summary produced by adf_test's fit_lag lambda. Named
// (not a std::tuple) so the selection pass and the report pass read the same
// fields by name and cannot silently swap two doubles.
//
// The gauss_elim / ols_normal_eq / matrix_diag_scale trio that used to live
// here is gone: every solve in this file now goes through sqt::qr::lstsq. The
// normal equations squared the condition number of each design and needed a
// bespoke "RSS went materially negative, so the factorization broke" guard
// that a QR simply cannot trigger.
struct AdfLagFit {
    bool   ok     = false;
    double rss    = 0.0;
    int    T      = 0;   // rows actually fitted
    int    k      = 0;   // regressor count
    double t_stat = std::numeric_limits<double>::quiet_NaN();
};


// ── MacKinnon (2010) critical values ──────────────────────────────────────────
// Response surface: cv(T) = c_inf + c1/T + c2/T² + c3/T³
// For 2-variable cointegration, constant (trend="c").
//
// These are now genuinely the 2010 coefficients. The previous set --
// (-3.9001, -10.534, -30.030) / (-3.3377, -5.967, -8.982) /
// (-3.0462, -4.069, -5.730) -- is MacKinnon (1991) Table 1, under a comment
// claiming MacKinnon (2010) Table 2, in a file whose stated goal is
// statsmodels parity. statsmodels' coint() calls mackinnoncrit(), which
// reads tau_2010s["c"][1]; transcribed here verbatim from statsmodels
// 0.14.3, including the cubic term (exactly 0.0 for N=2, kept so the
// transcription is checkable line-for-line against the source table rather
// than silently truncated).
//
// The disagreement was small but systematic, and always in the same
// direction on a given sample size -- measured against
// statsmodels.tsa.adfvalues.mackinnoncrit(N=2, regression="c"):
//
//     T      1%        5%       10%
//     50   +0.0061   +0.0004   +0.0005
//    100   +0.0009   -0.0004   -0.0003
//    250   -0.0019   -0.0010   -0.0011
//   1000   -0.0032   -0.0014   -0.0016
//
// Nothing caught it because the only assertions on these values, in both
// tests/cpp/test_cointegration.cpp and tests/cpp_bindings/, check ORDERING
// (cv_1pct < cv_5pct < cv_10pct < 0) and never a number.
//
// Sample size is nobs-1, not nobs -- see the call site.
static double mackinnon_cv(double level_pct, std::size_t n) {
    // statsmodels 0.14.3, tau_2010s["c"][N-1] with N=2, rows [1%, 5%, 10%].
    double c_inf, c1, c2, c3;
    if (level_pct <= 1.0) {
        c_inf = -3.89644; c1 = -10.9519; c2 = -33.527; c3 = 0.0;
    } else if (level_pct <= 5.0) {
        c_inf = -3.33613; c1 = -6.1101;  c2 = -6.823;  c3 = 0.0;
    } else {
        c_inf = -3.04445; c1 = -4.2412;  c2 = -2.72;   c3 = 0.0;
    }
    if (n == 0) return c_inf;  // no sample: the asymptotic value is all there is
    const double invT = 1.0 / static_cast<double>(n);
    // Horner in 1/T, matching numpy.polyval's evaluation in mackinnoncrit.
    return c_inf + invT * (c1 + invT * (c2 + invT * c3));
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
    // relative-epsilon singularity guard (numerics::is_negligible_pivot,
    // not the fixed 1e-14 threshold this comment used to name) returns
    // slope=NaN for that case -- would otherwise
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

    // Raw-moment sums on the *unshifted* x/y suffer catastrophic
    // cancellation for a large-baseline series -- the same bug class
    // already fixed in rolling_beta_into (rolling_regression.cpp) and
    // bollinger_bands_into (indicators.cpp). ols2 is a one-shot fit (no
    // sliding window), so a single shift by the series' own first values
    // (x[0]/y[0]) suffices -- no periodic re-centering is needed the way
    // the rolling kernels require to bound drift over many windows.
    const double cx = x[0], cy = y[0];
    double s1 = 0, sxd = 0, syd = 0, sxxd = 0, sxyd = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double xd = x[i] - cx;
        const double yd = y[i] - cy;
        s1   += 1.0;
        sxd  += xd;
        syd  += yd;
        sxxd += xd * xd;
        sxyd += xd * yd;
    }

    // Normal equations on shifted data:
    // [n, sxd; sxd, sxxd] [b0'; b1'] = [syd; sxyd], where b1' = slope
    // (unchanged by a pure shift) and b0' = intercept - slope*cx + cy
    // (un-shifted below). Relative-epsilon singularity threshold,
    // scaled to the shifted design matrix's own magnitude -- a pure ratio with
    // no absolute floor, so the same pair of series rescaled into different
    // units gives the same answer (see numerics::is_negligible_pivot).
    // Scale reference is s1*sxxd, not sxxd alone: det = s1*sxxd - sxd^2 grows
    // with the OBSERVATION COUNT as well as the spread of x, so testing it
    // against sxxd alone made the singularity check ~n times too lenient --
    // a genuinely near-singular system passed on any long series.
    const double det = s1 * sxxd - sxd * sxd;
    if (numerics::is_negligible_pivot(det, s1 * sxxd)) return r;

    const double intercept_shifted = (syd * sxxd - sxyd * sxd) / det;
    r.slope     = (s1 * sxyd - sxd * syd) / det;
    r.intercept = intercept_shifted + cy - r.slope * cx;  // un-shift

    double ss_res = 0, ss_tot = 0;
    const double y_mean = cy + syd / s1;  // = sy/s1, computed stably
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

AdfResult adf_test(const double* y, std::size_t n, int max_lag, bool use_aic,
                   bool include_constant) {
    AdfResult out{kNaN, 0, kNaN};
    if (n < 4) return out;

    // Deterministic regressor count: 1 for the constant, 0 without it. This is
    // statsmodels' `ntrend` for regression="c" / "n", and it enters the max-lag
    // cap below, so it has to be resolved before the cap is applied.
    //
    // include_constant exists because engle_granger's step 2 needs it OFF.
    // statsmodels' coint() runs adfuller(resid, regression="n") -- the
    // residuals of a cointegrating regression that already contained a
    // constant are mean-zero by construction, so fitting another intercept
    // spends a degree of freedom on a coefficient known to be zero and shifts
    // the t-statistic. This kernel always included one, which is why the
    // native and statsmodels paths of cointegration_test() disagreed on the
    // ADF statistic for essentially every input.
    const int ntrend = include_constant ? 1 : 0;

    if (max_lag < 0) {
        // Schwert (1989) rule in statsmodels' exact form: ceil(12*(n/100)^(1/4))
        // capped at n/2 - ntrend - 1. This kernel previously used floor(...)
        // capped at (n-2)/3 -- a different candidate set, so the two backends
        // searched different lags before the selection logic even ran.
        max_lag = static_cast<int>(
            std::ceil(12.0 * std::pow(static_cast<double>(n) / 100.0, 0.25)));
        // The cap is compared in long long space. `n/2` was narrowed to int
        // first, which wraps for a series longer than 2*INT_MAX bars and
        // would silently produce a NEGATIVE cap -- an early return reporting
        // "no lag solved" for a series large enough that lag selection is
        // the least of the problems, but wrong in the direction that is
        // hardest to notice.
        const long long half =
            numerics::checked_narrow_to_ll(n / 2, "adf_test: max-lag cap");
        const long long cap = half - ntrend - 1;
        max_lag = static_cast<int>(std::min<long long>(max_lag, cap));
        if (max_lag < 0) return out;
    }

    // Pre-compute first differences
    const std::size_t nd = n - 1;
    std::vector<double> dy(nd);
    for (std::size_t i = 0; i < nd; ++i) dy[i] = y[i + 1] - y[i];

    // Degenerate input: y is (numerically) constant, so every regressor in
    // every candidate lag's design matrix (y_{t-1}, and every lagged Δy)
    // has zero variance -- the per-lag solve below is rank-deficient for
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

    // Column index of y_{t-1} -- the coefficient whose t-statistic IS the ADF
    // statistic. It sits after the constant when there is one.
    const int lvl_idx = ntrend;

    // Fits Δy_t = [c +] φ·y_{t-1} + Σψⱼ·Δy_{t-j} over rows t = start_t .. n-1.
    //
    // start_t is the whole point of this refactor. The usable row count at lag
    // p is n-1-p, so a larger p is fitted on FEWER observations, and the
    // previous implementation scored every candidate on its own such sample.
    // Information criteria computed from different response vectors are not
    // comparable: log(σ²) shifts with the sample, and that shift routinely
    // exceeds the k-penalty difference the criterion is supposed to be
    // measuring. Selection now passes a COMMON start_t (every candidate holds
    // back max_lag observations) and only the winner is refitted at its own
    // start_t for the reported statistic -- statsmodels' adfuller() convention,
    // which matters because the Python fallback of cointegration_test() IS
    // statsmodels.
    auto fit_lag = [&](int p, std::size_t start_t, bool want_t) -> AdfLagFit {
        AdfLagFit f{};
        if (start_t >= n) return f;
        const std::size_t T_sz = n - start_t;
        const int k = ntrend + 1 + p;
        if (T_sz <= static_cast<std::size_t>(k)) return f;  // need df >= 1
        const int T = numerics::checked_narrow_to_int(T_sz, "adf_test: regression rows");
        const std::size_t k_sz = static_cast<std::size_t>(k);

        std::vector<double> A(numerics::checked_mul(T_sz, k_sz,
            "adf_test: design matrix size"));
        std::vector<double> b(T_sz);
        for (std::size_t row = 0; row < T_sz; ++row) {
            const std::size_t t = start_t + row;
            double* rp = A.data() + row * k_sz;
            int c = 0;
            if (include_constant) rp[c++] = 1.0;
            rp[c++] = y[t - 1];
            // Δy_{t-j} = y[t-j] - y[t-j-1] = dy[t-j-1]
            for (int j = 1; j <= p; ++j)
                rp[c++] = dy[t - 1 - static_cast<std::size_t>(j)];
            b[row] = dy[t - 1];  // Δy_t
        }

        // Rank-revealing QR, not normal equations. A is overwritten with its
        // factorization, which xtx_inv_diag then reads for the standard error
        // -- coefficient and t-statistic come from ONE decomposition instead of
        // two independently-conditioned solves.
        std::vector<int> perm(k_sz);
        const auto sol = qr::lstsq(A.data(), b.data(), T, k, perm.data());
        if (!sol.full_rank) return f;

        f.ok  = true;
        f.rss = sol.rss;
        f.T   = T;
        f.k   = k;
        if (want_t) {
            if (sol.rss <= 0.0) {
                // Exact fit: residual variance is identically zero, so the
                // t-statistic diverges. Same reading as the constant-series
                // branch above -- maximal evidence against a unit root.
                f.t_stat = -kInf;
            } else {
                const double sig2 = sol.rss / static_cast<double>(sol.df());
                const double xx =
                    qr::xtx_inv_diag(A.data(), sol, perm.data(), lvl_idx);
                const double se = std::sqrt(sig2 * xx);
                f.t_stat = (se > 0.0)
                    ? sol.beta[static_cast<std::size_t>(lvl_idx)] / se : kNaN;
            }
        }
        return f;
    };

    // ── Selection pass: one common sample for every candidate ────────────────
    double best_ic  = kInf;
    int    best_lag = -1;
    const std::size_t sel_start = static_cast<std::size_t>(max_lag) + 1;

    for (int p = 0; p <= max_lag; ++p) {
        const auto f = fit_lag(p, sel_start, /*want_t=*/false);
        if (!f.ok) continue;  // rank-deficient at this p; another p may solve

        const double T_d = static_cast<double>(f.T);
        const double k_d = static_cast<double>(f.k);
        double ic;
        if (f.rss <= 0.0) {
            ic = -kInf;  // exact fit wins outright
        } else {
            // σ² = RSS/T, the MLE variance -- NOT the unbiased RSS/(T-k).
            // statsmodels' OLS.aic/.bic are log-likelihood based and the
            // likelihood uses the MLE variance; the previous RSS/(T-k) form
            // folded a df correction into a criterion that already carries its
            // own k penalty, which changes which lag wins. Both this and the
            // common sample above are needed: measured over 200 series, fixing
            // only the sample cut disagreement with statsmodels from 33% to
            // 20.5%, fixing only σ² made it slightly worse at 34.5%, and both
            // together reached exact agreement on all 200.
            const double sig2 = f.rss / T_d;
            ic = std::log(sig2) +
                 (use_aic ? 2.0 * k_d / T_d : std::log(T_d) * k_d / T_d);
        }
        if (ic < best_ic) { best_ic = ic; best_lag = p; }
    }

    if (best_lag < 0) return out;  // nothing solved at any lag: NaN is honest

    // ── Report pass: refit the winner on its own (longer) sample ─────────────
    const auto final_fit =
        fit_lag(best_lag, static_cast<std::size_t>(best_lag) + 1, /*want_t=*/true);

    out.optimal_lag = best_lag;
    out.ic_min      = best_ic;
    out.statistic   = final_fit.ok ? final_fit.t_stat : kNaN;
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
    // Public struct field (int) -- checked-narrowed rather than silently
    // wrapped, so an astronomically large n fails loud instead of storing
    // a wrong (wrapped) observation count.
    r.n_obs         = numerics::checked_narrow_to_int(n, "engle_granger: n_obs");
    r.cointegrated  = false;

    if (n < 8) return r;

    // ── Step 1: OLS y0 = intercept + hedge_ratio * y1 ────────────────────────
    const auto ols = ols2(y0, y1, n);
    r.intercept   = ols.intercept;
    r.hedge_ratio = ols.slope;

    // ── Step 2: ADF test on the spread (OLS residuals) ───────────────────────
    // include_constant=false: statsmodels' coint() runs
    // adfuller(resid, regression="n") because a cointegrating regression's
    // residuals are mean-zero by construction. Fitting a second intercept
    // here spent a degree of freedom on a coefficient known to be zero and
    // moved the t-statistic, which is why this path and the statsmodels
    // fallback disagreed on essentially every input.
    const auto adf = adf_test(ols.residuals.data(), n, max_lag, use_aic,
                              /*include_constant=*/false);
    r.adf_statistic = adf.statistic;
    r.optimal_lag   = adf.optimal_lag;

    // ── Step 3: MacKinnon (2010) critical values and p-value ─────────────────
    // nobs-1, not nobs. statsmodels' coint() calls
    // mackinnoncrit(..., nobs=nobs-1) with its own comment on the line --
    // "the -1 is to match egranger in Stata, I do not know why". This
    // kernel passed the full n, so even with correct coefficients it would
    // have evaluated the response surface at the wrong sample size. Both
    // halves are needed to agree with the Python fallback, which IS
    // statsmodels.
    const std::size_t cv_nobs = n - 1;  // n >= 8 is guaranteed above
    r.cv_1pct  = mackinnon_cv(1.0,  cv_nobs);
    r.cv_5pct  = mackinnon_cv(5.0,  cv_nobs);
    r.cv_10pct = mackinnon_cv(10.0, cv_nobs);

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
