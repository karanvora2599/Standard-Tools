/**
 * C++ unit tests for sqt::fit_preprocess_stats and
 * sqt::apply_preprocess_stats.
 *
 * These kernels replace a pandas expression, so the properties worth
 * asserting here are the ones a Python-side comparison would find hardest
 * to localize: the quantile interpolation rule, the ddof=1 divisor, NaN
 * being skipped by the moments but preserved by the transform, and the
 * degenerate columns where the Python path substitutes a value rather than
 * dividing by zero. The bit-level agreement with pandas itself is asserted
 * from Python, where pandas is available to compare against.
 *
 * Build:
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run directly:
 *   Windows : build\tests\cpp\Release\test_panel_stats.exe
 *   Linux   : ./build/tests/cpp/test_panel_stats
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V
 */

#include "sqt/panel_stats.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

// ── Tiny assertion helpers ────────────────────────────────────────────────────

static int g_tests_run = 0;
static int g_tests_failed = 0;

static void expect(bool condition, const char* what) {
    ++g_tests_run;
    if (!condition) {
        ++g_tests_failed;
        std::printf("  FAIL: %s\n", what);
    }
}

static void expect_near(double got, double want, double tol, const char* what) {
    ++g_tests_run;
    const bool ok = (std::isnan(got) && std::isnan(want)) ||
                    std::fabs(got - want) <= tol;
    if (!ok) {
        ++g_tests_failed;
        std::printf("  FAIL: %s (got %.17g, want %.17g)\n", what, got, want);
    }
}

namespace {

struct Fitted {
    std::vector<double> lo, hi, mean, stdev;

    explicit Fitted(std::size_t n_cols)
        : lo(n_cols), hi(n_cols), mean(n_cols), stdev(n_cols) {}

    sqt::PreprocessStats view() {
        return sqt::PreprocessStats{lo.data(), hi.data(), mean.data(),
                                    stdev.data()};
    }
};

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

// ── Quantile interpolation ────────────────────────────────────────────────────

static void test_quantile_is_linearly_interpolated() {
    std::printf("test_quantile_is_linearly_interpolated\n");
    // 0..10 in one column. For q=0.25, h = (11-1)*0.25 = 2.5, so pandas
    // returns x[2] + 0.5*(x[3]-x[2]) = 2.5 -- NOT x[2]=2, which is what a
    // bare nth_element would give. This single case is the whole reason
    // interpolated_quantile exists.
    std::vector<double> values(11);
    for (std::size_t i = 0; i < 11; ++i) values[i] = static_cast<double>(i);

    Fitted fitted(1);
    const bool ok =
        sqt::fit_preprocess_stats(values.data(), 11, 1, 0.25, 0.75, fitted.view());
    expect(ok, "fit succeeds");
    expect_near(fitted.lo[0], 2.5, 1e-15, "q=0.25 interpolates to 2.5");
    expect_near(fitted.hi[0], 7.5, 1e-15, "q=0.75 interpolates to 7.5");
}

static void test_quantile_endpoints() {
    std::printf("test_quantile_endpoints\n");
    std::vector<double> values{5.0, 1.0, 4.0, 2.0, 3.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 5, 1, 0.0, 1.0, fitted.view());
    expect_near(fitted.lo[0], 1.0, 1e-15, "q=0 is the minimum");
    expect_near(fitted.hi[0], 5.0, 1e-15, "q=1 is the maximum");
}

// ── Moments ───────────────────────────────────────────────────────────────────

static void test_std_uses_ddof_one() {
    std::printf("test_std_uses_ddof_one\n");
    // 1,2,3,4: mean 2.5. ddof=1 variance = 5/3, std = 1.29099...
    // ddof=0 would give sqrt(1.25) = 1.118..., so the two are easy to tell
    // apart. pandas' Series.std() is ddof=1 and fit_preprocessing uses it.
    std::vector<double> values{1.0, 2.0, 3.0, 4.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 4, 1, 0.0, 1.0, fitted.view());
    expect_near(fitted.mean[0], 2.5, 1e-15, "mean of 1..4");
    expect_near(fitted.stdev[0], std::sqrt(5.0 / 3.0), 1e-15, "ddof=1 std");
}

static void test_moments_are_of_the_clipped_column() {
    std::printf("test_moments_are_of_the_clipped_column\n");
    // An extreme value pulled inside the winsorize bounds must not drag the
    // mean: the Python path clips first, then takes the moments.
    std::vector<double> values{0.0, 1.0, 2.0, 3.0, 1000.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 5, 1, 0.0, 0.75, fitted.view());
    // q=0.75 over 5 points: h = 4*0.75 = 3 exactly, so hi = 3.0.
    expect_near(fitted.hi[0], 3.0, 1e-15, "upper bound is 3.0");
    // Clipped column is 0,1,2,3,3 -> mean 1.8.
    expect_near(fitted.mean[0], 1.8, 1e-15, "mean is of the CLIPPED column");
}

// ── Degenerate columns ────────────────────────────────────────────────────────

static void test_constant_column_gets_unit_std() {
    std::printf("test_constant_column_gets_unit_std\n");
    std::vector<double> values{7.0, 7.0, 7.0, 7.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 4, 1, 0.01, 0.99, fitted.view());
    expect_near(fitted.mean[0], 7.0, 1e-15, "mean of a constant column");
    // Zero dispersion: 1.0 keeps the caller's division defined and leaves
    // the standardized value at 0, which is what the column deserves.
    expect_near(fitted.stdev[0], 1.0, 0.0, "constant column std is 1.0");

    std::vector<double> out(4);
    sqt::apply_preprocess_stats(values.data(), 4, 1, fitted.view(), out.data());
    for (std::size_t i = 0; i < 4; ++i)
        expect_near(out[i], 0.0, 1e-15, "constant column standardizes to 0");
}

static void test_single_row_column() {
    std::printf("test_single_row_column\n");
    std::vector<double> values{42.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 1, 1, 0.01, 0.99, fitted.view());
    expect_near(fitted.lo[0], 42.0, 0.0, "single value is both bounds");
    expect_near(fitted.hi[0], 42.0, 0.0, "single value is both bounds");
    // One observation has no ddof=1 dispersion at all.
    expect_near(fitted.stdev[0], 1.0, 0.0, "one row -> std 1.0");
}

static void test_all_nan_column() {
    std::printf("test_all_nan_column\n");
    std::vector<double> values{kNaN, kNaN, kNaN};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 3, 1, 0.01, 0.99, fitted.view());
    expect(std::isnan(fitted.lo[0]), "all-NaN column has NaN bounds");
    expect(std::isnan(fitted.mean[0]), "all-NaN column has NaN mean");
    expect_near(fitted.stdev[0], 1.0, 0.0, "all-NaN column still gets std 1.0");
}

// ── NaN and infinity handling ─────────────────────────────────────────────────

static void test_nan_is_skipped_by_the_moments() {
    std::printf("test_nan_is_skipped_by_the_moments\n");
    // Series.quantile and Series.std ignore missing values, so the moments
    // here must match those of {1,2,3,4} exactly.
    std::vector<double> with_gaps{1.0, kNaN, 2.0, 3.0, kNaN, 4.0};
    std::vector<double> without{1.0, 2.0, 3.0, 4.0};
    Fitted a(1), b(1);
    sqt::fit_preprocess_stats(with_gaps.data(), 6, 1, 0.0, 1.0, a.view());
    sqt::fit_preprocess_stats(without.data(), 4, 1, 0.0, 1.0, b.view());
    expect_near(a.mean[0], b.mean[0], 0.0, "NaN does not change the mean");
    expect_near(a.stdev[0], b.stdev[0], 0.0, "NaN does not change the std");
    expect_near(a.lo[0], b.lo[0], 0.0, "NaN does not change the bounds");
}

static void test_nan_survives_the_transform() {
    std::printf("test_nan_survives_the_transform\n");
    // Series.clip leaves a missing value missing rather than pinning it to
    // a bound -- so a gap must come out the far side still a gap, not a
    // fabricated observation sitting exactly on the winsorize boundary.
    std::vector<double> values{1.0, kNaN, 3.0};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 3, 1, 0.0, 1.0, fitted.view());
    std::vector<double> out(3);
    sqt::apply_preprocess_stats(values.data(), 3, 1, fitted.view(), out.data());
    expect(std::isnan(out[1]), "NaN passes through apply untouched");
    expect(!std::isnan(out[0]) && !std::isnan(out[2]), "other rows transform");
}

static void test_infinity_is_not_treated_as_missing() {
    std::printf("test_infinity_is_not_treated_as_missing\n");
    // pandas treats only NaN as missing. An infinity is a real, if
    // pathological, order statistic and must participate in the quantile.
    std::vector<double> values{1.0, 2.0, 3.0, kInf};
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 4, 1, 0.0, 1.0, fitted.view());
    expect(std::isinf(fitted.hi[0]), "inf is the maximum, not skipped");
}

// ── Multi-column layout ───────────────────────────────────────────────────────

static void test_columns_are_independent() {
    std::printf("test_columns_are_independent\n");
    // Row-major (4, 3): each column has a different scale, and reading the
    // wrong stride would mix them. Column 1 is deliberately the constant
    // one so a stride bug shows up as a wrong std rather than a near miss.
    std::vector<double> values{
        1.0, 5.0, 100.0,
        2.0, 5.0, 200.0,
        3.0, 5.0, 300.0,
        4.0, 5.0, 400.0,
    };
    Fitted fitted(3);
    sqt::fit_preprocess_stats(values.data(), 4, 3, 0.0, 1.0, fitted.view());
    expect_near(fitted.mean[0], 2.5, 1e-15, "column 0 mean");
    expect_near(fitted.mean[1], 5.0, 1e-15, "column 1 mean");
    expect_near(fitted.mean[2], 250.0, 1e-13, "column 2 mean");
    expect_near(fitted.stdev[1], 1.0, 0.0, "column 1 is constant -> std 1.0");
    expect_near(fitted.stdev[0], std::sqrt(5.0 / 3.0), 1e-15, "column 0 std");

    std::vector<double> out(12);
    sqt::apply_preprocess_stats(values.data(), 4, 3, fitted.view(), out.data());
    // Column 0 standardized: (1-2.5)/1.290994 = -1.161895
    expect_near(out[0], (1.0 - 2.5) / std::sqrt(5.0 / 3.0), 1e-14,
                "row 0, column 0 standardized");
    expect_near(out[1], 0.0, 1e-15, "row 0, column 1 (constant) -> 0");
    expect_near(out[2], (100.0 - 250.0) / fitted.stdev[2], 1e-13,
                "row 0, column 2 standardized");
}

static void test_apply_can_write_into_its_own_input() {
    std::printf("test_apply_can_write_into_its_own_input\n");
    // The header documents that values and out may alias; the engine does
    // not rely on it today, but a caller reading that promise should find
    // it true.
    std::vector<double> values{1.0, 2.0, 3.0, 4.0};
    std::vector<double> expected(4);
    Fitted fitted(1);
    sqt::fit_preprocess_stats(values.data(), 4, 1, 0.0, 1.0, fitted.view());
    sqt::apply_preprocess_stats(values.data(), 4, 1, fitted.view(),
                                expected.data());
    sqt::apply_preprocess_stats(values.data(), 4, 1, fitted.view(),
                                values.data());
    for (std::size_t i = 0; i < 4; ++i)
        expect_near(values[i], expected[i], 0.0, "in-place matches out-of-place");
}

static void test_empty_panel_is_a_no_op() {
    std::printf("test_empty_panel_is_a_no_op\n");
    Fitted fitted(1);
    fitted.lo[0] = fitted.hi[0] = fitted.mean[0] = fitted.stdev[0] = -1.0;
    const bool ok = sqt::fit_preprocess_stats(nullptr, 0, 0, 0.01, 0.99,
                                              fitted.view());
    expect(ok, "null/empty input reports success rather than failing");
    expect_near(fitted.lo[0], -1.0, 0.0, "untouched on an empty panel");
}

static void test_zero_rows_with_columns() {
    std::printf("test_zero_rows_with_columns\n");
    // No rows but real columns: every column is "all missing", so the
    // all-NaN rule applies rather than a division by zero.
    std::vector<double> values;  // empty, but n_cols = 2
    Fitted fitted(2);
    sqt::fit_preprocess_stats(values.data(), 0, 2, 0.01, 0.99, fitted.view());
    expect(std::isnan(fitted.mean[0]) && std::isnan(fitted.mean[1]),
           "zero rows -> NaN means");
    expect_near(fitted.stdev[0], 1.0, 0.0, "zero rows -> std 1.0");
}

int main() {
    std::printf("=== sqt panel_stats tests ===\n");
    test_quantile_is_linearly_interpolated();
    test_quantile_endpoints();
    test_std_uses_ddof_one();
    test_moments_are_of_the_clipped_column();
    test_constant_column_gets_unit_std();
    test_single_row_column();
    test_all_nan_column();
    test_nan_is_skipped_by_the_moments();
    test_nan_survives_the_transform();
    test_infinity_is_not_treated_as_missing();
    test_columns_are_independent();
    test_apply_can_write_into_its_own_input();
    test_empty_panel_is_a_no_op();
    test_zero_rows_with_columns();

    std::printf("\n%d assertion(s), %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
