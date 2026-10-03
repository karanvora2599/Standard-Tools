/**
 * C++ unit tests for sqt::pearson_correlation_into, pandas' nancorr.
 *
 * The Python suite (tests/cpp_bindings/test_cpp_correlation.py) holds the
 * kernel to DataFrame.corr() itself, bit for bit, on both pandas majors.
 * What is checked here is what the kernel owes on its own: the reference
 * loop below -- pandas/_libs/algos.pyx's nancorr, written out -- matched to
 * the bit on panels with and without missing values, in C and Fortran
 * order; hand-checked coefficients; the NaN rules (constant columns, one
 * row, an all-missing column, min_periods); infinity read as missing; an
 * exactly symmetric result; and the same bits on one thread as on many.
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_correlation
 */

#include "sqt/fp_contract.hpp"  // the reference loop is held to the kernel's rule
#include "sqt/correlation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_tests_run;                                                                \
        if (!(cond)) {                                                                \
            ++g_tests_failed;                                                         \
            std::fprintf(stderr, "FAIL  %s  line %d: %s\n", __func__, __LINE__, #cond); \
        }                                                                             \
    } while (false)

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// Both NaN, or the same bits.
bool same_cell(double a, double b) {
    return (std::isnan(a) && std::isnan(b)) || same_bits(a, b);
}

// pandas' nancorr(mat, cov=False, minp), as written in algos.pyx. `mat` is
// row-major (n_rows, n_cols).
std::vector<double> reference(const std::vector<double>& mat, std::size_t n_rows,
                              std::size_t n_cols, long long minp) {
    std::vector<double> result(n_cols * n_cols, -7.0);
    for (std::size_t xi = 0; xi < n_cols; ++xi) {
        for (std::size_t yi = 0; yi <= xi; ++yi) {
            long long nobs = 0;
            double ssqdmx = 0, ssqdmy = 0, covxy = 0, meanx = 0, meany = 0;
            for (std::size_t i = 0; i < n_rows; ++i) {
                const double vx = mat[i * n_cols + xi];
                const double vy = mat[i * n_cols + yi];
                if (std::isfinite(vx) && std::isfinite(vy)) {
                    nobs += 1;
                    const double dx = vx - meanx;
                    const double dy = vy - meany;
                    meanx += 1. / nobs * dx;
                    meany += 1. / nobs * dy;
                    ssqdmx += (vx - meanx) * dx;
                    ssqdmy += (vy - meany) * dy;
                    covxy += (vx - meanx) * dy;
                }
            }
            double r;
            if (nobs < minp) {
                r = kNaN;
            } else {
                const double divisor = std::sqrt(ssqdmx * ssqdmy);
                r = (divisor != 0) ? covxy / divisor : kNaN;
            }
            result[xi * n_cols + yi] = r;
            result[yi * n_cols + xi] = r;
        }
    }
    return result;
}

// The kernel on a row-major panel.
std::vector<double> kernel(const std::vector<double>& mat, std::size_t n_rows,
                           std::size_t n_cols, long long minp = 1) {
    std::vector<double> out(n_cols * n_cols, -7.0);
    sqt::pearson_correlation_into(mat.data(), n_rows, n_cols,
                                  static_cast<std::ptrdiff_t>(n_cols), 1, minp,
                                  out.data());
    return out;
}

// The kernel on the same panel stored column-major.
std::vector<double> kernel_fortran(const std::vector<double>& mat, std::size_t n_rows,
                                   std::size_t n_cols, long long minp = 1) {
    std::vector<double> f(mat.size());
    for (std::size_t i = 0; i < n_rows; ++i)
        for (std::size_t k = 0; k < n_cols; ++k) f[k * n_rows + i] = mat[i * n_cols + k];
    std::vector<double> out(n_cols * n_cols, -7.0);
    sqt::pearson_correlation_into(f.data(), n_rows, n_cols, 1,
                                  static_cast<std::ptrdiff_t>(n_rows), minp, out.data());
    return out;
}

bool all_same(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_cell(a[i], b[i])) return false;
    return true;
}

std::vector<double> noise(std::size_t n_rows, std::size_t n_cols, unsigned seed,
                          double scale = 0.012) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, scale);
    std::vector<double> m(n_rows * n_cols);
    for (double& v : m) v = normal(rng);
    return m;
}

}  // namespace

static void test_no_columns_writes_nothing() {
    sqt::pearson_correlation_into(nullptr, 0, 0, 0, 1, 1, nullptr);
    sqt::pearson_correlation_into(nullptr, 5, 0, 0, 1, 1, nullptr);
    CHECK(true);
}

static void test_no_rows_is_all_nan() {
    std::vector<double> out(9, -7.0);
    sqt::pearson_correlation_into(nullptr, 0, 3, 3, 1, 1, out.data());
    for (double v : out) CHECK(std::isnan(v));
    // min_periods 0 does not rescue it: the divisor is 0.
    sqt::pearson_correlation_into(nullptr, 0, 3, 3, 1, 0, out.data());
    for (double v : out) CHECK(std::isnan(v));
}

static void test_hand_checked_coefficients() {
    // y = 2x + 1 and z = -x: +1 and -1 to the last bit or two.
    const std::vector<double> m = {1, 3, -1, 2, 5, -2, 3, 7, -3, 4, 9, -4};
    const auto r = kernel(m, 4, 3);
    CHECK(std::fabs(r[0 * 3 + 1] - 1.0) < 1e-15);
    CHECK(std::fabs(r[0 * 3 + 2] + 1.0) < 1e-15);
    CHECK(std::fabs(r[1 * 3 + 2] + 1.0) < 1e-15);
    // The diagonal: s / sqrt(s * s) is exactly 1 when s * s neither
    // overflows nor underflows.
    for (int k = 0; k < 3; ++k) CHECK(same_bits(r[k * 3 + k], 1.0));
    // x = [1, 2, 3], y = [1, 3, 2]: cov 0.5, variances 1 and 1, so 0.5.
    const std::vector<double> m2 = {1, 1, 2, 3, 3, 2};
    const auto r2 = kernel(m2, 3, 2);
    CHECK(std::fabs(r2[1] - 0.5) < 1e-15);
    CHECK(all_same(r, reference(m, 4, 3, 1)));
    CHECK(all_same(r2, reference(m2, 3, 2, 1)));
}

static void test_a_constant_column_is_nan_everywhere_including_itself() {
    std::vector<double> m = noise(50, 3, 1);
    for (std::size_t i = 0; i < 50; ++i) m[i * 3 + 1] = 0.25;
    const auto r = kernel(m, 50, 3);
    for (std::size_t k = 0; k < 3; ++k) {
        CHECK(std::isnan(r[1 * 3 + k]));
        CHECK(std::isnan(r[k * 3 + 1]));
    }
    CHECK(!std::isnan(r[0 * 3 + 2]));
    CHECK(all_same(r, reference(m, 50, 3, 1)));
}

static void test_one_row_is_nan_everywhere() {
    const std::vector<double> m = {0.1, 0.2, 0.3};
    const auto r = kernel(m, 1, 3);
    for (double v : r) CHECK(std::isnan(v));
}

static void test_an_all_missing_column_is_nan_and_the_rest_unaffected() {
    std::vector<double> m = noise(40, 4, 2);
    for (std::size_t i = 0; i < 40; ++i) m[i * 4 + 2] = kNaN;
    const auto r = kernel(m, 40, 4);
    for (std::size_t k = 0; k < 4; ++k) CHECK(std::isnan(r[2 * 4 + k]));
    std::vector<double> without(40 * 3);
    for (std::size_t i = 0; i < 40; ++i) {
        without[i * 3 + 0] = m[i * 4 + 0];
        without[i * 3 + 1] = m[i * 4 + 1];
        without[i * 3 + 2] = m[i * 4 + 3];
    }
    const auto r3 = kernel(without, 40, 3);
    CHECK(same_bits(r[0 * 4 + 1], r3[0 * 3 + 1]));
    CHECK(same_bits(r[3 * 4 + 0], r3[2 * 3 + 0]));
    CHECK(same_bits(r[3 * 4 + 1], r3[2 * 3 + 1]));
}

static void test_infinity_is_a_missing_value() {
    std::vector<double> with_nan = noise(30, 3, 3);
    std::vector<double> with_inf = with_nan;
    with_nan[7 * 3 + 1] = kNaN;
    with_inf[7 * 3 + 1] = kInf;
    with_nan[19 * 3 + 2] = kNaN;
    with_inf[19 * 3 + 2] = -kInf;
    CHECK(all_same(kernel(with_nan, 30, 3), kernel(with_inf, 30, 3)));
    CHECK(all_same(kernel(with_inf, 30, 3), reference(with_inf, 30, 3, 1)));
}

static void test_min_periods_counts_the_pairs_complete_rows() {
    // Column 1 is complete on rows 0..3 only: four rows with column 0.
    std::vector<double> m = noise(8, 2, 4);
    for (std::size_t i = 4; i < 8; ++i) m[i * 2 + 1] = kNaN;
    const auto at4 = kernel(m, 8, 2, 4);
    const auto at5 = kernel(m, 8, 2, 5);
    CHECK(!std::isnan(at4[1]));
    CHECK(std::isnan(at5[1]));
    // Column 0 alone has eight rows, so its diagonal survives 8 and not 9.
    CHECK(!std::isnan(kernel(m, 8, 2, 8)[0]));
    CHECK(std::isnan(kernel(m, 8, 2, 9)[0]));
    CHECK(all_same(at4, reference(m, 8, 2, 4)));
    CHECK(all_same(at5, reference(m, 8, 2, 5)));
}

static void test_matches_the_reference_loop_bit_for_bit() {
    const std::size_t shapes[][2] = {{2, 2}, {3, 9}, {37, 5}, {250, 17}, {600, 64}};
    unsigned seed = 10;
    for (const auto& shape : shapes) {
        const std::size_t n_rows = shape[0], n_cols = shape[1];
        // Complete panel: every pair takes the shared-recursion path.
        std::vector<double> m = noise(n_rows, n_cols, seed++);
        CHECK(all_same(kernel(m, n_rows, n_cols), reference(m, n_rows, n_cols, 1)));
        CHECK(all_same(kernel_fortran(m, n_rows, n_cols),
                       reference(m, n_rows, n_cols, 1)));
        // Mixed: a third of the columns lose scattered rows, so pairs of
        // every kind -- complete/complete, complete/gapped, gapped/gapped --
        // are in one matrix.
        std::mt19937_64 rng(seed++);
        for (std::size_t k = 0; k < n_cols; k += 3)
            for (std::size_t i = 0; i < n_rows; ++i)
                if (rng() % 5 == 0) m[i * n_cols + k] = (rng() % 2) ? kNaN : kInf;
        const auto ref = reference(m, n_rows, n_cols, 1);
        CHECK(all_same(kernel(m, n_rows, n_cols), ref));
        CHECK(all_same(kernel_fortran(m, n_rows, n_cols), ref));
    }
    // Large and offset magnitudes, where the order of operations shows most.
    std::vector<double> big = noise(300, 12, 99, 1e3);
    for (double& v : big) v += 1e8;
    CHECK(all_same(kernel(big, 300, 12), reference(big, 300, 12, 1)));
    std::vector<double> tiny = noise(300, 12, 98, 1e-150);
    CHECK(all_same(kernel(tiny, 300, 12), reference(tiny, 300, 12, 1)));
}

static void test_the_result_is_exactly_symmetric() {
    std::vector<double> m = noise(120, 21, 7);
    for (std::size_t i = 0; i < 120; i += 9) m[i * 21 + 4] = kNaN;
    const auto r = kernel(m, 120, 21);
    bool symmetric = true;
    for (std::size_t a = 0; a < 21; ++a)
        for (std::size_t b = 0; b < 21; ++b)
            symmetric = symmetric && same_cell(r[a * 21 + b], r[b * 21 + a]);
    CHECK(symmetric);
}

static void test_one_thread_and_many_give_the_same_bits() {
#ifdef _OPENMP
    std::vector<double> m = noise(700, 90, 21);
    for (std::size_t i = 0; i < 700; i += 13) m[i * 90 + 30] = kNaN;
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto serial = kernel(m, 700, 90);
    omp_set_num_threads(threads > 1 ? threads : 4);
    const auto parallel = kernel(m, 700, 90);
    omp_set_num_threads(threads);
    CHECK(all_same(serial, parallel));
#else
    CHECK(true);
#endif
}

int main() {
    test_no_columns_writes_nothing();
    test_no_rows_is_all_nan();
    test_hand_checked_coefficients();
    test_a_constant_column_is_nan_everywhere_including_itself();
    test_one_row_is_nan_everywhere();
    test_an_all_missing_column_is_nan_and_the_rest_unaffected();
    test_infinity_is_a_missing_value();
    test_min_periods_counts_the_pairs_complete_rows();
    test_matches_the_reference_loop_bit_for_bit();
    test_the_result_is_exactly_symmetric();
    test_one_thread_and_many_give_the_same_bits();

    std::printf("%d/%d checks passed\n", g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed == 0 ? 0 : 1;
}
