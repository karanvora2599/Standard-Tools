// AVX2+FMA implementation of rolling_beta_into's window-reduction step
// (item L: runtime ISA dispatch demo). This translation unit is compiled
// unconditionally with AVX2+FMA codegen enabled (see CMakeLists.txt's
// set_source_files_properties for this file), independent of the opt-in
// SQT_NATIVE_ARCH flag that gates the rest of the extension's codegen --
// MSVC has no per-function ISA-target attribute, so isolating AVX2
// intrinsics into their own file compiled with /arch:AVX2 is the only
// portable way to keep the rest of the module's codegen unaffected.

#include "sqt/rolling_beta_avx2.hpp"

#include <immintrin.h>

namespace sqt {

void rolling_beta_reduce_avx2(
    const double* x,
    const double* y,
    std::size_t   start,
    int           window,
    double        cx,
    double        cy,
    double&       Sx,
    double&       Sy,
    double&       Sxy,
    double&       Sxx)
{
    __m256d vSx  = _mm256_setzero_pd();
    __m256d vSy  = _mm256_setzero_pd();
    __m256d vSxy = _mm256_setzero_pd();
    __m256d vSxx = _mm256_setzero_pd();
    const __m256d vcx = _mm256_set1_pd(cx);
    const __m256d vcy = _mm256_set1_pd(cy);

    const std::size_t end     = start + static_cast<std::size_t>(window);
    const std::size_t vec_end = start + (static_cast<std::size_t>(window) / 4) * 4;

    std::size_t j = start;
    for (; j < vec_end; j += 4) {
        const __m256d vx = _mm256_loadu_pd(x + j);
        const __m256d vy = _mm256_loadu_pd(y + j);
        const __m256d xd = _mm256_sub_pd(vx, vcx);
        const __m256d yd = _mm256_sub_pd(vy, vcy);
        vSx  = _mm256_add_pd(vSx, xd);
        vSy  = _mm256_add_pd(vSy, yd);
        vSxy = _mm256_fmadd_pd(xd, yd, vSxy);
        vSxx = _mm256_fmadd_pd(xd, xd, vSxx);
    }

    // Horizontal reduction of each 4-lane accumulator.
    double bufx[4], bufy[4], bufxy[4], bufxx[4];
    _mm256_storeu_pd(bufx,  vSx);
    _mm256_storeu_pd(bufy,  vSy);
    _mm256_storeu_pd(bufxy, vSxy);
    _mm256_storeu_pd(bufxx, vSxx);
    Sx  = bufx[0]  + bufx[1]  + bufx[2]  + bufx[3];
    Sy  = bufy[0]  + bufy[1]  + bufy[2]  + bufy[3];
    Sxy = bufxy[0] + bufxy[1] + bufxy[2] + bufxy[3];
    Sxx = bufxx[0] + bufxx[1] + bufxx[2] + bufxx[3];

    // Scalar tail for the remainder (window not a multiple of 4).
    for (; j < end; ++j) {
        const double xd = x[j] - cx;
        const double yd = y[j] - cy;
        Sx  += xd;
        Sy  += yd;
        Sxy += xd * yd;
        Sxx += xd * xd;
    }
}

}  // namespace sqt
