/**
 * C++ unit tests for sqt::rolling_factor_loadings and sqt::rolling_beta
 * (rolling_regression.cpp).
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_rolling_regression
 */

#include "sqt/rolling_regression.hpp"

#include "sqt/isa_dispatch.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

// ── Tiny assertion helpers ────────────────────────────────────────────────────

static int g_tests_run    = 0;
static int g_tests_failed = 0;

#define CHECK(cond) \
    do { \
        ++g_tests_run; \
        if (!(cond)) { \
            ++g_tests_failed; \
            std::fprintf(stderr, "FAIL  %s  line %d: %s\n", __func__, __LINE__, #cond); \
        } \
    } while (false)

#define CHECK_NEAR(a, b, tol) \
    CHECK(std::abs((a) - (b)) <= (tol))

#define CHECK_TRUE(cond)   CHECK(cond)
#define CHECK_FALSE(cond)  CHECK(!(cond))
#define CHECK_NAN(val)     CHECK(std::isnan(val))
#define CHECK_NOT_NAN(val) CHECK(!std::isnan(val))

// ── Deterministic pseudo-random generator (no <random> dependency, matches
//    the style used elsewhere in this test suite for reproducible inputs) ──

static double pseudo_random(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint64_t x = state;
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33;
    return (static_cast<double>(x >> 11) / 9007199254740992.0) * 2.0 - 1.0;  // [-1, 1)
}

// ── Independent reference solver: dense Gaussian elimination with partial
//    pivoting on the normal equations, sharing NO code with cholesky_solve()
//    or build_normal_equations() -- an honest cross-check of the production
//    (lower-triangle-only) implementation, not a copy of it. ──────────────

static bool gauss_solve(std::vector<double> A, std::vector<double> b, int p,
                         std::vector<double>& x) {
    for (int col = 0; col < p; ++col) {
        int pivot = col;
        double best = std::abs(A[col * p + col]);
        for (int r = col + 1; r < p; ++r) {
            const double v = std::abs(A[r * p + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-12) return false;
        if (pivot != col) {
            for (int c = 0; c < p; ++c) std::swap(A[col * p + c], A[pivot * p + c]);
            std::swap(b[col], b[pivot]);
        }
        for (int r = col + 1; r < p; ++r) {
            const double factor = A[r * p + col] / A[col * p + col];
            for (int c = col; c < p; ++c) A[r * p + c] -= factor * A[col * p + c];
            b[r] -= factor * b[col];
        }
    }
    x.assign(p, 0.0);
    for (int r = p - 1; r >= 0; --r) {
        double s = b[r];
        for (int c = r + 1; c < p; ++c) s -= A[r * p + c] * x[c];
        x[r] = s / A[r * p + r];
    }
    return true;
}

// Builds the FULL (all p^2 entries, including upper triangle) normal
// equations directly from a window of bars -- an independent re-derivation,
// not a call into build_normal_equations().
static void reference_beta(
    const double* y, const double* factors, int start, int end,
    int k, int p, std::vector<double>& beta_out, bool& ok)
{
    std::vector<double> XtX(p * p, 0.0), Xty(p, 0.0);
    for (int i = start; i < end; ++i) {
        const double* fi = factors + i * k;
        for (int r = 0; r < p; ++r) {
            const double xr = (r == 0) ? 1.0 : fi[r - 1];
            for (int c = 0; c < p; ++c) {
                XtX[r * p + c] += xr * ((c == 0) ? 1.0 : fi[c - 1]);
            }
            Xty[r] += xr * y[i];
        }
    }
    ok = gauss_solve(XtX, Xty, p, beta_out);
}

// ── rolling_factor_loadings tests ───────────────────────────────────────────

static void test_nan_prefix_and_shape() {
    const int n = 50, k = 2, window = 10;
    std::vector<double> y(n), factors(n * k);
    std::uint64_t state = 42;
    for (int i = 0; i < n; ++i) {
        y[i] = pseudo_random(state);
        for (int f = 0; f < k; ++f) factors[i * k + f] = pseudo_random(state);
    }
    auto out = sqt::rolling_factor_loadings(y.data(), factors.data(), n, k, window);
    CHECK(out.size() == static_cast<std::size_t>(n) * (k + 1));
    for (int i = 0; i < window - 1; ++i)
        for (int j = 0; j <= k; ++j) CHECK_NAN(out[i * (k + 1) + j]);
    CHECK_NOT_NAN(out[(window - 1) * (k + 1)]);
}

static void test_single_factor_recovers_known_coefficients() {
    // y = 3.0 + 2.0*x, no noise -> every window's fit must recover this exactly.
    const int n = 40, k = 1, window = 15;
    std::vector<double> x(n), y(n);
    std::uint64_t state = 7;
    for (int i = 0; i < n; ++i) {
        x[i] = pseudo_random(state) * 10.0;
        y[i] = 3.0 + 2.0 * x[i];
    }
    auto out = sqt::rolling_factor_loadings(y.data(), x.data(), n, k, window);
    for (int i = window - 1; i < n; ++i) {
        CHECK_NEAR(out[i * (k + 1) + 0], 3.0, 1e-8);
        CHECK_NEAR(out[i * (k + 1) + 1], 2.0, 1e-8);
    }
}

static void test_matches_independent_reference_multi_factor() {
    // Cross-check against a from-scratch Gaussian-elimination reference
    // (full p^2 normal equations, sharing no code with the production
    // lower-triangle-only path) across many random windows -- this is the
    // real correctness proof that removing the upper-triangle computation
    // didn't change any observable output.
    const int n = 200, k = 3, p = k + 1, window = 25;
    std::vector<double> y(n), factors(n * k);
    std::uint64_t state = 12345;
    for (int i = 0; i < n; ++i) {
        y[i] = pseudo_random(state) * 5.0;
        for (int f = 0; f < k; ++f) factors[i * k + f] = pseudo_random(state) * 3.0;
    }
    auto out = sqt::rolling_factor_loadings(y.data(), factors.data(), n, k, window);

    int checked = 0;
    for (int i = window - 1; i < n; i += 7) {  // sample every 7th bar
        std::vector<double> ref_beta;
        bool ok = false;
        reference_beta(y.data(), factors.data(), i - window + 1, i + 1, k, p, ref_beta, ok);
        if (!ok) continue;  // singular window -- production code would also emit NaN here
        for (int j = 0; j < p; ++j)
            CHECK_NEAR(out[i * p + j], ref_beta[j], 1e-8);
        ++checked;
    }
    CHECK_TRUE(checked > 10);  // sanity: the loop above actually exercised real windows
}

static void test_singular_window_produces_nan() {
    // Two identical factor columns -> XtX singular -> cholesky_solve fails
    // -> NaN, exactly like the pre-existing Python-level test of this case.
    const int n = 40, k = 2, window = 20;
    std::vector<double> y(n), factors(n * k);
    std::uint64_t state = 99;
    for (int i = 0; i < n; ++i) {
        y[i] = pseudo_random(state);
        const double v = pseudo_random(state);
        factors[i * k + 0] = v;
        factors[i * k + 1] = v;  // duplicate column
    }
    auto out = sqt::rolling_factor_loadings(y.data(), factors.data(), n, k, window);
    CHECK_NAN(out[(window - 1) * (k + 1) + 0]);
}

static void test_large_magnitude_recovers_known_coefficients() {
    // Same construction as test_single_factor_recovers_known_coefficients,
    // but x is scaled to ~1e6 magnitude instead of O(1) -- exercises
    // cholesky_solve's relative-epsilon singularity threshold (replacing a
    // fixed absolute `s <= 1e-14` that didn't scale with the matrix's own
    // magnitude): XtX's diagonal entries here are ~1e12, so a threshold
    // that stayed fixed at an O(1)-scale absolute value could fail to
    // reject a genuinely near-singular window at this scale (see the
    // companion NaN test below). Deliberately zero-mean-ish random scaling
    // (not a huge constant offset + tiny variation) -- an offset-based
    // construction hits a separate, out-of-scope raw-moment catastrophic-
    // cancellation issue in build_normal_equations' uncentered sums, which
    // this test isn't targeting.
    const int n = 40, k = 1, window = 15;
    std::vector<double> x(n), y(n);
    std::uint64_t state = 2024;
    for (int i = 0; i < n; ++i) {
        x[i] = pseudo_random(state) * 1.0e6;
        y[i] = 3.0 + 2.0 * x[i];
    }
    auto out = sqt::rolling_factor_loadings(y.data(), x.data(), n, k, window);
    for (int i = window - 1; i < n; ++i) {
        CHECK_NEAR(out[i * (k + 1) + 0], 3.0, 1e-4);
        CHECK_NEAR(out[i * (k + 1) + 1], 2.0, 1e-12);
    }
}

static void test_singular_window_at_large_magnitude_produces_nan() {
    // Same duplicate-column construction as test_singular_window_produces_nan,
    // but scaled to ~1e6 magnitude -- proves the relative-epsilon threshold
    // isn't accidentally MORE permissive at large scale than the old fixed
    // absolute threshold was: a genuinely singular window (two identical
    // factor columns) must still be rejected (NaN output) regardless of
    // the matrix's magnitude, since two identical columns make the exact
    // same direction singular whether their magnitude is O(1) or O(1e6).
    const int n = 40, k = 2, window = 20;
    std::vector<double> y(n), factors(n * k);
    std::uint64_t state = 9901;
    for (int i = 0; i < n; ++i) {
        y[i] = pseudo_random(state) * 1.0e6;
        const double v = pseudo_random(state) * 1.0e6;
        factors[i * k + 0] = v;
        factors[i * k + 1] = v;  // duplicate column
    }
    auto out = sqt::rolling_factor_loadings(y.data(), factors.data(), n, k, window);
    CHECK_NAN(out[(window - 1) * (k + 1) + 0]);
}

static void test_window_larger_than_n_all_nan() {
    const int n = 10, k = 1, window = 50;
    std::vector<double> y(n), x(n);
    std::uint64_t state = 3;
    for (int i = 0; i < n; ++i) { y[i] = pseudo_random(state); x[i] = pseudo_random(state); }
    auto out = sqt::rolling_factor_loadings(y.data(), x.data(), n, k, window);
    for (double v : out) CHECK_NAN(v);
}

static void test_underdetermined_window_all_nan() {
    // window < k+2 leaves fewer observations than the k+1 coefficients being
    // estimated, so every window is underdetermined and the whole result is
    // NaN. Pinned on the C++ side too because the PYTHON fallback used to
    // disagree here: numpy.linalg.lstsq returned its minimum-norm solution
    // instead of NaN, so the same call produced numbers or NaN depending only
    // on whether the extension was built. analysis/multi_factor.py now
    // short-circuits to NaN before dispatching, matching this kernel.
    const int n = 40, k = 1;
    std::vector<double> y(n), x(n);
    std::uint64_t state = 7;
    for (int i = 0; i < n; ++i) { y[i] = pseudo_random(state); x[i] = pseudo_random(state); }

    for (int window = 1; window <= k + 1; ++window) {
        auto out = sqt::rolling_factor_loadings(y.data(), x.data(), n, k, window);
        for (double v : out) CHECK_NAN(v);
    }

    // window == k+2 is the smallest DETERMINED window -- must produce values,
    // so the guard above can't silently swallow every legitimate call too.
    auto ok = sqt::rolling_factor_loadings(y.data(), x.data(), n, k, k + 2);
    bool any_finite = false;
    for (double v : ok) if (!std::isnan(v)) { any_finite = true; break; }
    CHECK_TRUE(any_finite);
}

// ── rolling_beta tests (no existing C++ coverage before this file) ─────────

static void test_rolling_beta_recovers_known_slope() {
    // y = 1.5*x + noise-free -> beta must converge to exactly 1.5.
    const int n = 60, window = 20;
    std::vector<double> x(n), y(n);
    std::uint64_t state = 55;
    for (int i = 0; i < n; ++i) {
        x[i] = pseudo_random(state) * 4.0;
        y[i] = 1.5 * x[i];
    }
    auto out = sqt::rolling_beta(y.data(), x.data(), n, window);
    for (int i = window - 1; i < n; ++i) CHECK_NEAR(out[i], 1.5, 1e-6);
}

static void test_rolling_beta_nan_prefix() {
    const int n = 30, window = 10;
    std::vector<double> x(n), y(n);
    std::uint64_t state = 21;
    for (int i = 0; i < n; ++i) { x[i] = pseudo_random(state); y[i] = pseudo_random(state); }
    auto out = sqt::rolling_beta(y.data(), x.data(), n, window);
    for (int i = 0; i < window - 1; ++i) CHECK_NAN(out[i]);
    CHECK_NOT_NAN(out[window - 1]);
}

// ── rolling_beta: runtime ISA dispatch (item L) ─────────────────────────────
//
// rolling_beta_into dispatches to an AVX2+FMA reduction when
// detect_isa_features().avx2 is true, otherwise the portable scalar path.
// NOT bit-identical between the two (SIMD lane accumulation reorders the
// sum) -- tolerance-gated here, comparing the AVX2 path (this machine is
// AVX2-capable, so this is the real dispatched path in every other test in
// this file) against the scalar path forced via the test-only override
// hook -- the only practical way to exercise "runs correctly on a
// non-AVX2 CPU" without physical access to one.

static void test_rolling_beta_avx2_matches_scalar_tolerance() {
    struct Case { int n, window; bool huge_baseline; };
    const Case cases[] = {
        {500, 60, false},
        {500, 60, true},   // large-baseline cancellation stress, same shape
                            // as the existing large-baseline fix this file
                            // already covers for the scalar path
        {200, 7, false},   // window not a multiple of 4 -> exercises the
                            // AVX2 kernel's scalar tail
        {37, 37, false},   // window == n (single output value)
    };

    for (const auto& c : cases) {
        std::vector<double> x(c.n), y(c.n);
        std::uint64_t state = 909;
        for (int i = 0; i < c.n; ++i) {
            double xv = pseudo_random(state) * 3.0;
            double yv = pseudo_random(state) * 3.0;
            if (c.huge_baseline) { xv += 1e9; yv += 1e9; }
            x[i] = xv;
            y[i] = yv;
        }

        sqt::reset_isa_features_override_for_testing();
        auto out_avx2 = sqt::rolling_beta(y.data(), x.data(), c.n, c.window);

        sqt::force_isa_features_for_testing({false, false});
        auto out_scalar = sqt::rolling_beta(y.data(), x.data(), c.n, c.window);
        sqt::reset_isa_features_override_for_testing();

        for (int i = c.window - 1; i < c.n; ++i) {
            if (std::isnan(out_scalar[i])) {
                CHECK_NAN(out_avx2[i]);
            } else {
                CHECK_NEAR(out_avx2[i], out_scalar[i], 1e-6);
            }
        }
    }
}

static void test_rolling_beta_forced_scalar_path_correct() {
    // With the AVX2 path forced off, the scalar path alone must still
    // recover a known slope exactly (same assertion as
    // test_rolling_beta_recovers_known_slope, but with the dispatch
    // explicitly pinned to scalar rather than relying on this machine
    // happening to route there).
    sqt::force_isa_features_for_testing({false, false});
    const int n = 60, window = 20;
    std::vector<double> x(n), y(n);
    std::uint64_t state = 55;
    for (int i = 0; i < n; ++i) {
        x[i] = pseudo_random(state) * 4.0;
        y[i] = 1.5 * x[i];
    }
    auto out = sqt::rolling_beta(y.data(), x.data(), n, window);
    sqt::reset_isa_features_override_for_testing();

    for (int i = window - 1; i < n; ++i) CHECK_NEAR(out[i], 1.5, 1e-6);
}

int main() {
    test_nan_prefix_and_shape();
    test_single_factor_recovers_known_coefficients();
    test_matches_independent_reference_multi_factor();
    test_singular_window_produces_nan();
    test_large_magnitude_recovers_known_coefficients();
    test_singular_window_at_large_magnitude_produces_nan();
    test_window_larger_than_n_all_nan();
    test_underdetermined_window_all_nan();
    test_rolling_beta_recovers_known_slope();
    test_rolling_beta_nan_prefix();
    test_rolling_beta_avx2_matches_scalar_tolerance();
    test_rolling_beta_forced_scalar_path_correct();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
