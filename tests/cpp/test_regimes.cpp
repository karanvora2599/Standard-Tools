/**
 * C++ unit tests for sqt::regime_em_step.
 *
 * The Python-side tests hold the step, and detect_regimes as a whole, to
 * the numpy expressions bit for bit. Asserted here: the step written out
 * longhand on a small sample (which pins the (k, n) layouts and the
 * left-to-right sums), the convergence test at its boundary, and the three
 * floors: an observation whose densities all underflow divides by 1e-300,
 * an empty regime's count is read as 1e-12, and a variance never falls
 * below 1e-12.
 *
 * Run directly: build\tests\cpp\test_regimes.exe, or via CTest.
 */

#include "sqt/regimes.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_tests_run = 0;
static int g_tests_failed = 0;

static void expect(bool condition, const char* what) {
    ++g_tests_run;
    if (!condition) {
        ++g_tests_failed;
        std::printf("  FAIL: %s\n", what);
    }
}

namespace {

constexpr double kTwoPi = 2.0 * 3.141592653589793;

struct Step {
    std::vector<double> r, counts, means, vars, next;
    bool converged;
};

Step run(const std::vector<double>& x, const std::vector<double>& e,
         const std::vector<double>& means, const std::vector<double>& variances,
         const std::vector<double>& weights) {
    const std::size_t n = x.size(), k = variances.size();
    Step s{std::vector<double>(k * n), std::vector<double>(k), std::vector<double>(k),
           std::vector<double>(k), std::vector<double>(k * n), false};
    std::vector<double> totals(n);
    s.converged = sqt::regime_em_step(x.data(), n, e.data(), means.data(),
                                      variances.data(), weights.data(), k, s.r.data(),
                                      s.counts.data(), s.means.data(), s.vars.data(),
                                      s.next.data(), totals.data());
    return s;
}

}  // namespace

static void test_the_step_written_out() {
    std::printf("test_the_step_written_out\n");
    const std::vector<double> x = {-0.02, 0.001, 0.013, -0.004, 0.03};
    const std::vector<double> means = {-0.005, 0.01};
    const std::vector<double> variances = {1e-4, 4e-4};
    const std::vector<double> weights = {0.6, 0.4};
    const std::size_t n = x.size(), k = 2;
    std::vector<double> e(k * n);
    for (std::size_t j = 0; j < k; ++j)
        for (std::size_t i = 0; i < n; ++i) {
            const double d = x[i] - means[j];
            e[j * n + i] = std::exp((-0.5 * (d * d)) / variances[j]);
        }
    const Step s = run(x, e, means, variances, weights);

    // Longhand, observation by observation, as the numpy expressions read.
    std::vector<double> r(k * n), counts(k, 0.0), sum_rx(k, 0.0), spread(k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double total = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            r[j * n + i] = weights[j] * (e[j * n + i] / std::sqrt(kTwoPi * variances[j]));
            total += r[j * n + i];
        }
        for (std::size_t j = 0; j < k; ++j) r[j * n + i] /= total;
    }
    bool same_r = true;
    for (std::size_t i = 0; i < n * k; ++i) same_r = same_r && (s.r[i] == r[i]);
    expect(same_r, "responsibilities are (k, n), each observation normalized");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            counts[j] += r[j * n + i];
            sum_rx[j] += r[j * n + i] * x[i];
        }
    bool same = true;
    for (std::size_t j = 0; j < k; ++j) {
        const double mean = sum_rx[j] / counts[j];
        for (std::size_t i = 0; i < n; ++i) {
            const double d = x[i] - mean;
            spread[j] += r[j * n + i] * (d * d);
        }
        const double var = spread[j] / counts[j];
        same = same && s.counts[j] == counts[j] && s.means[j] == mean && s.vars[j] == var;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = x[i] - mean;
            same = same && s.next[j * n + i] == (-0.5 * (d * d)) / var;
        }
    }
    expect(same, "counts, means, variances and next exponents, in observation order");
    expect(!s.converged, "the means moved");

    // Handing the step its own new means back: converged exactly.
    const Step again = run(x, e, s.means, variances, weights);
    expect(again.converged, "unchanged means are close");
}

static void test_the_convergence_boundary() {
    std::printf("test_the_convergence_boundary\n");
    // Two identical regimes, so each new mean is the sample mean whatever
    // came in, and only the old means move.
    const std::vector<double> x = {1.0, 2.0, 3.0, 6.0};
    const std::vector<double> e(8, 1.0);
    const std::vector<double> v = {1.0, 1.0}, w = {0.5, 0.5};
    const Step first = run(x, e, {0.0, 0.0}, v, w);
    expect(first.means[0] == 3.0 && first.means[1] == 3.0, "the mean of the sample");
    // |new - old| <= 1e-10 + 1e-5 * |old|: inside and just outside it.
    const double tol = 1e-10 + 1e-05 * 3.0;
    expect(run(x, e, {3.0 + tol * 0.5, 3.0}, v, w).converged, "inside the tolerance");
    expect(!run(x, e, {3.0, 3.0 + tol * 2.0}, v, w).converged, "outside it, one regime");
    const double nan = std::nan("");
    expect(!run(x, e, {nan, 3.0}, v, w).converged, "a NaN old mean is never close");
}

static void test_the_floors() {
    std::printf("test_the_floors\n");
    // Regime 1's densities are all zero (its weight is 0), and the last
    // observation's densities are zero under both regimes.
    const std::vector<double> x = {1.0, 1.0, 1.0, 5.0};
    const std::vector<double> e = {1.0, 1.0, 1.0, 0.0,   // regime 0
                                   0.0, 0.0, 0.0, 0.0};  // regime 1
    const Step s = run(x, e, {1.0, 0.0}, {1.0, 1.0}, {1.0, 0.0});
    expect(s.r[3] == 0.0 && s.r[7] == 0.0, "an all-zero observation stays zero (/1e-300)");
    expect(s.counts[1] == 0.0, "the empty regime counts zero");
    expect(s.means[1] == 0.0, "and its mean is 0 / 1e-12");
    expect(s.vars[1] == 1e-12, "and its variance is floored");
    expect(s.vars[0] == 1e-12, "regime 0's points coincide: floored too");
    expect(s.means[0] == 1.0, "regime 0's mean is its points'");
}

int main() {
    std::printf("=== sqt regimes tests ===\n");
    test_the_step_written_out();
    test_the_convergence_boundary();
    test_the_floors();
    std::printf("\n%d assertion(s), %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
