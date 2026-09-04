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


/**
 * Per-date correlation between two aligned columns.
 *
 * `date_codes[i]` is the date index of row i, in [0, n_dates). Rows need
 * NOT be sorted: the kernel counting-sorts them into per-date buckets in
 * O(n_rows), which replaces the caller's O(n log n) argsort AND the two
 * gather passes that followed it.
 *
 * NaN PAIRS ARE DROPPED, matching Series.corr, which removes rows where
 * either side is missing BEFORE correlating and (for spearman) before
 * ranking. Infinities are kept: pandas treats only NaN as missing, so an
 * inf ranks as an extreme for spearman and voids pearson to the undefined
 * branch.
 *
 * `out_ic[d]` is that date's correlation, or 0.0 where it is undefined --
 * fewer than two usable pairs, or a constant cross-section. 0.0 rather than
 * NaN is the existing contract: the Python `_safe_corr` mapped NaN to 0.0
 * and the metric summaries depend on it.
 *
 * The POOLED correlation is this same computation with one segment, so
 * `_safe_corr` is served by calling this with n_dates=1 and all codes 0
 * rather than by a second kernel that could drift from it.
 *
 * @return false if a working buffer could not be allocated.
 */
bool cross_sectional_correlation(const double* y_true,
                                 const double* y_pred,
                                 const long long* date_codes,
                                 std::size_t n_rows,
                                 std::size_t n_dates,
                                 bool spearman,
                                 double* out_ic);

/**
 * Standardize every column within each date's cross-section.
 *
 * `values` and `out` are row-major (n_rows, n_cols) and may alias.
 * `date_codes[i]` is row i's date index, as above; rows need not be sorted.
 *
 * Per date and column: subtract the mean, divide by the ddof=1 standard
 * deviation, then clip to +/- clip_sigma (0 disables the clip). A date whose
 * cross-section is constant -- or which has a single usable entity -- has no
 * dispersion, and every entity in it sits exactly at the mean, so those rows
 * become 0.0 rather than NaN.
 *
 * NaN is skipped by the moments and preserved in the output, on the same
 * reasoning as the preprocessing kernel: a missing feature must stay missing
 * rather than be fabricated into an observation at the cross-section mean.
 *
 * @return false if a working buffer could not be allocated.
 */
bool standardize_by_date(const double* values,
                         std::size_t n_rows,
                         std::size_t n_cols,
                         const long long* date_codes,
                         std::size_t n_dates,
                         double clip_sigma,
                         double* out);


/**
 * Average rank of every value within its own date's cross-section.
 *
 * `values` and `out` are row-major (n_rows, n_cols) and may alias.
 * `date_codes[i]` is row i's date index; rows need not be sorted.
 *
 * Ranks are 1-based and ties take the mean of the ordinals they span --
 * `Series.rank(method="average")`, which is what the callers here mean by
 * a rank. Ranking is per COLUMN within each date, so a panel of several
 * models' predictions is ranked in one call.
 *
 * NaN is skipped by the ranking and preserved in the output: a missing
 * value has no position in an ordering, and giving it one would invent a
 * view the model never expressed. Its presence does not shift the ranks of
 * the values that ARE there -- a date with one name absent ranks the rest
 * 1..n-1, exactly as pandas does.
 *
 * @return false if a working buffer could not be allocated.
 */
bool rank_by_date(const double* values,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  const long long* date_codes,
                  std::size_t n_dates,
                  double* out);


/**
 * The null distribution of a mean cross-sectional IC under within-date
 * shuffling.
 *
 * For each of `n_permutations` draws: shuffle `values` inside each date,
 * correlate against `target` within each date, and average those
 * correlations over the dates that carry at least two rows. `out_null` gets
 * one mean IC per draw.
 *
 * WHY THIS IS ONE CALL. Done from Python the loop is 200 shuffles, 200
 * correlation calls and 200 round trips through pandas; measured at 18.6 s
 * for 200 draws over 504,000 rows, of which about 6 s was Series
 * construction alone. Fusing removes that entirely and lets the ranking
 * happen ONCE: shuffling values inside a date permutes their ranks, so for
 * spearman the ranks can be shuffled directly rather than recomputed per
 * draw. That is the saving, and it cannot be expressed in numpy -- an
 * attempt measured 0.6x because it replaces an O(n) counting sort with an
 * O(n log n) one.
 *
 * `date_codes[i]` is row i's date index; rows need not be sorted. Rows whose
 * target or value is not finite are dropped once, before any shuffling.
 *
 * SEEDING. `seed` reproduces a run WITHIN this backend only. This is not a
 * reimplementation of numpy's PCG64 bit stream, so the Python fallback
 * produces different draws from the same seed -- the same contract
 * `simulate_forward_paths` states, and for the same reason.
 *
 * @return false if a working buffer could not be allocated.
 */
bool permutation_null_ic(const double* target,
                         const double* values,
                         const long long* date_codes,
                         std::size_t n_rows,
                         std::size_t n_dates,
                         std::size_t n_permutations,
                         unsigned long long seed,
                         bool spearman,
                         double* out_null);


/**
 * Average uniqueness of each row's label, computed within its entity.
 *
 * Overlapping forward returns make consecutive rows largely redundant:
 * `effective_sample_size` has always reported that and nothing acted on it.
 * This is the weight that does -- the mean of 1/concurrency over the bars a
 * row's own label spans (Lopez de Prado, ch. 4).
 *
 * `dates` and `label_end` are nanoseconds since the epoch, with numpy's NaT
 * (INT64_MIN) marking a label that never resolves -- the final `horizon`
 * rows, which span only themselves. Rows need not be sorted; the kernel
 * buckets by entity and orders each entity by its OWN date axis, which is
 * the point of carrying label ends as timestamps: with entities on
 * different calendars, t+horizon of one entity's bars is not t+horizon of
 * the global panel's dates.
 *
 * Weights come back normalized to mean 1 over every row, so turning
 * weighting on does not also rescale the effective regularization strength.
 *
 * @return false if a working buffer could not be allocated.
 */
bool label_uniqueness(const long long* dates,
                      const long long* label_end,
                      const long long* entity_codes,
                      std::size_t n_rows,
                      std::size_t n_entities,
                      double* out_weights);

}  // namespace sqt
