/**
 * C++ unit tests for sqt::garch11_variance_recursion (GARCH(1,1) kernel).
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_garch
 */

#include "sqt/garch.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
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


// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_empty_input_returns_empty() {
    auto result = sqt::garch11_variance_recursion(nullptr, 0, 1e-6, 0.05, 0.9);
    CHECK(result.empty());
}

static void test_returns_correct_length() {
    std::vector<double> resid_sq(200, 1e-4);
    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), 1e-6, 0.05, 0.9);
    CHECK(result.size() == 200);
}

static void test_first_value_is_mean_of_resid_sq() {
    std::vector<double> resid_sq = {1e-4, 4e-4, 9e-4, 1.6e-3};
    double mean = 0.0;
    for (double v : resid_sq) mean += v;
    mean /= static_cast<double>(resid_sq.size());

    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), 1e-6, 0.05, 0.9);
    CHECK_NEAR(result[0], mean, 1e-12);
}

static void test_recursion_matches_hand_computation() {
    // sigma2[1] = omega + alpha*resid_sq[0] + beta*sigma2[0]
    std::vector<double> resid_sq = {2e-4, 5e-4, 1e-4};
    const double omega = 1e-6, alpha = 0.08, beta = 0.85;

    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), omega, alpha, beta);

    const double expected_sigma2_1 = omega + alpha * resid_sq[0] + beta * result[0];
    CHECK_NEAR(result[1], expected_sigma2_1, 1e-12);

    const double expected_sigma2_2 = omega + alpha * resid_sq[1] + beta * result[1];
    CHECK_NEAR(result[2], expected_sigma2_2, 1e-12);
}

static void test_floor_at_min_sigma2() {
    // omega negative and alpha/beta zero drives every subsequent value
    // below the floor -- must clamp to 1e-12, not go negative.
    std::vector<double> resid_sq(30, 0.0);
    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), -1.0, 0.0, 0.0);
    bool all_floored = true;
    for (double v : result) {
        if (v < 1e-12) { all_floored = false; break; }
    }
    CHECK(all_floored);
}

static void test_single_observation() {
    std::vector<double> resid_sq = {3e-4};
    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), 1e-6, 0.05, 0.9);
    CHECK(result.size() == 1);
    CHECK_NEAR(result[0], 3e-4, 1e-12);
}


// ── Tests: garch11_neg_loglik ────────────────────────────────────────────────

// Independent reference for the fused NLL: computed from
// garch11_variance_recursion's own output (already covered by the tests
// above) plus a separate reduction loop, deliberately not sharing any code
// with garch11_neg_loglik's internal fused loop -- this only agrees with
// the fused version if the fusion is actually correct, not just
// self-consistent.
static double reference_neg_loglik(
    const std::vector<double>& resid_sq,
    double omega, double alpha, double beta, bool penalize)
{
    auto sigma2 = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), omega, alpha, beta);
    double nll = 0.0;
    const double log_2pi = std::log(2.0 * 3.14159265358979323846);
    for (std::size_t t = 0; t < resid_sq.size(); ++t) {
        nll += log_2pi + std::log(sigma2[t]) + resid_sq[t] / sigma2[t];
    }
    nll *= 0.5;
    if (penalize) {
        const double persistence = alpha + beta;
        if (persistence >= 1.0) {
            const double d = persistence - 1.0;
            nll += 1.0e6 * d * d;
        }
    }
    return nll;
}

static void test_neg_loglik_empty_input_returns_zero() {
    auto result = sqt::garch11_neg_loglik(nullptr, 0, 1e-6, 0.05, 0.9, true);
    CHECK_NEAR(result, 0.0, 1e-15);
}

static void test_neg_loglik_matches_reference_no_penalty() {
    std::vector<double> resid_sq = {2e-4, 5e-4, 1e-4, 3e-4, 8e-4, 1.5e-4};
    const double omega = 1e-6, alpha = 0.08, beta = 0.85;  // persistence < 1
    const double result = sqt::garch11_neg_loglik(
        resid_sq.data(), resid_sq.size(), omega, alpha, beta, true);
    const double expected = reference_neg_loglik(resid_sq, omega, alpha, beta, true);
    CHECK_NEAR(result, expected, 1e-9);
}

static void test_neg_loglik_penalty_branch_matches_reference() {
    // persistence = alpha+beta = 1.05 >= 1.0 -> penalty term must apply.
    std::vector<double> resid_sq = {1e-4, 2e-4, 3e-4, 4e-4};
    const double omega = 1e-6, alpha = 0.2, beta = 0.85;  // persistence = 1.05
    const double with_penalty = sqt::garch11_neg_loglik(
        resid_sq.data(), resid_sq.size(), omega, alpha, beta, true);
    const double without_penalty = sqt::garch11_neg_loglik(
        resid_sq.data(), resid_sq.size(), omega, alpha, beta, false);
    CHECK(with_penalty > without_penalty);  // penalty strictly adds a positive term
    const double expected_with = reference_neg_loglik(resid_sq, omega, alpha, beta, true);
    const double expected_without = reference_neg_loglik(resid_sq, omega, alpha, beta, false);
    CHECK_NEAR(with_penalty, expected_with, 1e-9);
    CHECK_NEAR(without_penalty, expected_without, 1e-9);
}

static void test_neg_loglik_single_observation_matches_reference() {
    std::vector<double> resid_sq = {5e-4};
    const double omega = 1e-6, alpha = 0.05, beta = 0.9;
    const double result = sqt::garch11_neg_loglik(
        resid_sq.data(), resid_sq.size(), omega, alpha, beta, true);
    const double expected = reference_neg_loglik(resid_sq, omega, alpha, beta, true);
    CHECK_NEAR(result, expected, 1e-12);
}


// ── Tests: garch11_neg_loglik_grad (analytic gradient) ───────────────────────

// Simple LCG pseudo-random in (0, 1), seed-deterministic.
static std::vector<double> lcg_resid_sq(int n, unsigned seed, double scale) {
    std::vector<double> out(n);
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        const double u = static_cast<double>(state & 0x7FFFFFFFu) / 2147483648.0;
        const double r = (u - 0.5) * scale;  // pseudo-residual
        out[i] = r * r;
    }
    return out;
}

// Central-difference numerical gradient of garch11_neg_loglik, independent
// of garch11_neg_loglik_grad's own analytic derivation -- this is the hard
// gate the analytic gradient must pass before being trusted for anything.
// Step size is scaled per-parameter (not a single absolute h) since
// omega (~1e-6..1e-5) and alpha/beta (~0.05..0.95) differ by many orders
// of magnitude -- a single absolute h appropriate for alpha/beta would be
// a ~100%-of-magnitude perturbation for omega, dominated by truncation
// error in the numerical estimate itself, not a real analytic-gradient bug.
static void numerical_grad(
    const std::vector<double>& resid_sq,
    double omega, double alpha, double beta, bool penalize,
    double out_grad[3])
{
    auto f = [&](double o, double a, double b) {
        return sqt::garch11_neg_loglik(resid_sq.data(), resid_sq.size(), o, a, b, penalize);
    };
    const double h_omega = std::max(1e-10, std::abs(omega) * 1e-4);
    const double h_alpha = std::max(1e-8,  std::abs(alpha) * 1e-4);
    const double h_beta  = std::max(1e-8,  std::abs(beta)  * 1e-4);
    out_grad[0] = (f(omega + h_omega, alpha, beta) - f(omega - h_omega, alpha, beta)) / (2.0 * h_omega);
    out_grad[1] = (f(omega, alpha + h_alpha, beta) - f(omega, alpha - h_alpha, beta)) / (2.0 * h_alpha);
    out_grad[2] = (f(omega, alpha, beta + h_beta) - f(omega, alpha, beta - h_beta)) / (2.0 * h_beta);
}

static void test_analytic_gradient_matches_numerical_grid() {
    // Hard gate (performance architecture review item 3's gradient
    // stretch goal): sweep several (resid_sq, omega, alpha, beta) inputs,
    // well away from the flooring boundary (where the true gradient has a
    // real discontinuity central differences and the analytic formula
    // would legitimately disagree at), and require close agreement.
    struct Case { unsigned seed; double scale; int n; double omega, alpha, beta; };
    const Case cases[] = {
        {1,  0.02, 100, 1e-6, 0.05, 0.90},
        {2,  0.01, 250, 1e-5, 0.10, 0.80},
        {3,  0.03,  50, 1e-7, 0.02, 0.95},
        {4,  0.02, 400, 1e-6, 0.20, 0.70},
        {5,  0.05, 150, 1e-6, 0.08, 0.88},
    };
    for (const auto& c : cases) {
        auto resid_sq = lcg_resid_sq(c.n, c.seed, c.scale);
        double analytic[3];
        const double nll = sqt::garch11_neg_loglik_grad(
            resid_sq.data(), resid_sq.size(), c.omega, c.alpha, c.beta, true, analytic);

        double numeric[3];
        numerical_grad(resid_sq, c.omega, c.alpha, c.beta, true, numeric);

        // nll itself must match garch11_neg_loglik exactly (same formula).
        const double nll_only = sqt::garch11_neg_loglik(
            resid_sq.data(), resid_sq.size(), c.omega, c.alpha, c.beta, true);
        CHECK_NEAR(nll, nll_only, 1e-9);

        // Relative tolerance: omega's gradient can be numerically large
        // (1/sigma2 terms), so an absolute-only tolerance would be too
        // tight for it and too loose for alpha/beta -- use a mixed
        // absolute+relative check, standard practice for gradient checks.
        for (int k = 0; k < 3; ++k) {
            const double tol = 1e-3 + 1e-3 * std::abs(numeric[k]);
            CHECK_NEAR(analytic[k], numeric[k], tol);
        }
    }
}

static void test_analytic_gradient_penalty_branch_matches_numerical() {
    // persistence = alpha+beta = 1.05 >= 1.0 -> penalty term active, whose
    // own analytic gradient (2e6*persistence-2e6) must also check out.
    auto resid_sq = lcg_resid_sq(80, 42, 0.02);
    const double omega = 1e-6, alpha = 0.2, beta = 0.85;

    double analytic[3];
    sqt::garch11_neg_loglik_grad(
        resid_sq.data(), resid_sq.size(), omega, alpha, beta, true, analytic);

    double numeric[3];
    numerical_grad(resid_sq, omega, alpha, beta, true, numeric);

    for (int k = 0; k < 3; ++k) {
        const double tol = 1e-3 + 1e-3 * std::abs(numeric[k]);
        CHECK_NEAR(analytic[k], numeric[k], tol);
    }
}

static void test_grad_empty_input_returns_zero() {
    double grad[3];
    const double nll = sqt::garch11_neg_loglik_grad(nullptr, 0, 1e-6, 0.05, 0.9, true, grad);
    CHECK_NEAR(nll, 0.0, 1e-15);
    CHECK_NEAR(grad[0], 0.0, 1e-15);
    CHECK_NEAR(grad[1], 0.0, 1e-15);
    CHECK_NEAR(grad[2], 0.0, 1e-15);
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_empty_input_returns_empty();
    test_returns_correct_length();
    test_first_value_is_mean_of_resid_sq();
    test_recursion_matches_hand_computation();
    test_floor_at_min_sigma2();
    test_single_observation();

    test_neg_loglik_empty_input_returns_zero();
    test_neg_loglik_matches_reference_no_penalty();
    test_neg_loglik_penalty_branch_matches_reference();
    test_neg_loglik_single_observation_matches_reference();

    test_analytic_gradient_matches_numerical_grid();
    test_analytic_gradient_penalty_branch_matches_numerical();
    test_grad_empty_input_returns_zero();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
