#pragma once

/**
 * Panel statistics for the modeling layer: per-column preprocessing and
 * per-date (cross-sectional) reductions.
 *
 * WHY THESE ARE NATIVE. Measured on a ridge walk-forward run, feature
 * preprocessing is 47-56% of the runtime -- more than the estimator fit and
 * the metrics combined, and an order of magnitude more than either. Per
 * fold at 2,000 entities the two `quantile` calls cost 166.5 ms and the
 * `clip` 60.2 ms, against 32 ms for all the pandas slicing around them. The
 * arithmetic is contiguous float64 with no Python object in the loop, which
 * is exactly the shape a kernel wins on.
 *
 * WHAT IS DELIBERATELY NOT HERE. The DataFrame slicing, the parquet write
 * and sklearn's fit are the other half of a run and no kernel reaches them.
 * See Development/modeling_native_plan.md for the ceiling arithmetic: with
 * preprocessing at ~50%, even an infinitely fast kernel caps the end-to-end
 * speedup at 2x.
 *
 * EXACTNESS IS THE CONTRACT. Every function here replaces a specific pandas
 * expression and must reproduce it, including the parts that are pandas
 * conventions rather than mathematical necessity -- linear-interpolated
 * quantiles, ddof=1 standard deviations, NaN skipped rather than
 * propagated. Where a rule is pandas' choice rather than the only sensible
 * one, the comment says so.
 */

#include <cstddef>

namespace sqt {

/**
 * Per-column winsorize bounds and standardization moments.
 *
 * Laid out as four parallel arrays rather than an array of structs because
 * the apply pass reads all four for one column together and the compiler
 * keeps them in registers; a struct-of-four would be the same, but this
 * matches how the caller already holds them.
 */
struct PreprocessStats {
    double* lo;    ///< lower winsorize bound, one per column
    double* hi;    ///< upper winsorize bound, one per column
    double* mean;  ///< mean of the CLIPPED column
    double* stdev; ///< ddof=1 standard deviation of the clipped column
};

/**
 * Fit winsorize bounds and clipped moments for every column.
 *
 * `values` is row-major (n_rows, n_cols) -- the layout `DataFrame.to_numpy()`
 * hands over, so no transpose is needed on the Python side.
 *
 * Reproduces, per column:
 *     lo, hi = col.quantile(q_low), col.quantile(q_high)
 *     clipped = col.clip(lo, hi)
 *     mean, std = clipped.mean(), clipped.std()      # std is ddof=1
 *
 * QUANTILES ARE LINEARLY INTERPOLATED, matching pandas' default: for the
 * sorted column and h = (n-1)*q, the value is
 *     x[floor(h)] + (h - floor(h)) * (x[floor(h)+1] - x[floor(h)])
 * A plain nth_element gives x[floor(h)] and would disagree with pandas on
 * almost every real column.
 *
 * NaN IS SKIPPED, not propagated -- again matching pandas, where
 * `Series.quantile` and `Series.std` both ignore missing values. A column
 * that is entirely NaN yields lo=hi=NaN and mean=NaN; std falls back to 1.0
 * so the caller's division is still defined, which is what
 * fit_preprocessing does in Python.
 *
 * A column with fewer than two finite values has no ddof=1 dispersion;
 * std is set to 1.0 there for the same reason.
 *
 * @return false if a working buffer could not be allocated (the caller then
 *         falls back to the Python path); true otherwise.
 */
bool fit_preprocess_stats(const double* values,
                          std::size_t n_rows,
                          std::size_t n_cols,
                          double q_low,
                          double q_high,
                          PreprocessStats out);

/**
 * Apply fitted stats: clip to [lo, hi], then (x - mean) / std, in place-safe
 * form into `out`.
 *
 * `values` and `out` are both row-major (n_rows, n_cols) and may alias.
 *
 * One fused pass with the inner loop over COLUMNS, which is the contiguous
 * direction in row-major and lets the small per-column parameter arrays stay
 * resident in L1 for the whole sweep. The Python version it replaces
 * allocates two full-panel temporaries per column (the clip result and the
 * standardized result); this allocates nothing.
 *
 * NaN passes through as NaN: a missing feature value stays missing rather
 * than being clipped to a bound, which is what `Series.clip` does.
 */
void apply_preprocess_stats(const double* values,
                            std::size_t n_rows,
                            std::size_t n_cols,
                            const PreprocessStats& stats,
                            double* out);

}  // namespace sqt
