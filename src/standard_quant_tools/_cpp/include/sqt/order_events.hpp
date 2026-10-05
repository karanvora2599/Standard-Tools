#pragma once

#include <cstddef>
#include <cstdint>

namespace sqt {

/**
 * The two passes of `analysis.order_events` over a market-by-order window:
 * the size resting ahead of each new order, and how long each order lived.
 *
 * WHY THIS IS NATIVE. Both are a Python loop over every event that keeps a
 * dict per order (and per price level): 4.0 s and 3.6 s on a 2,000,000-event
 * session, the cap the tool reads by default, against 0.16 s to read the
 * session's Parquet file -- 94% of `get_order_event_metrics`.
 *
 * THE SAME STATE MACHINE. The caller hands over what the Python loop
 * compares: each event's order as a code (pd.factorize, so two ids are one
 * order exactly when they are one dict key), its action as a code (below),
 * its side as a code, its price and size as the float64s the loop reads, the
 * snapshot flag, and its timestamp as an integer in the column's own unit.
 * The kernels then do what the loop does, in event order, with the same
 * float operations in the same order -- `+`, `-`, Python's `min` and `max`
 * as written -- so every size and count is the loop's to the bit. A price
 * level is the pair (side code, price), with -0.0 and +0.0 one level, as
 * they are one dict key. A CLEAR forgets every order and level; it is O(1)
 * here (a generation number), as dict.clear() is to the loop's reader.
 *
 * Neither kernel throws or allocates more than one entry per order and per
 * level. Each returns false when a buffer could not be allocated.
 */

/// Action codes. The Python maps each distinct action value to one of these
/// by the comparisons its loop makes (`== "R"`, `== "A"`, `in ("C", "F")`,
/// `== "C"`); anything else is kOrderOther, which both passes skip.
enum OrderAction : std::int8_t {
    kOrderAdd = 0,
    kOrderCancel = 1,
    kOrderFill = 2,
    kOrderClear = 3,
    kOrderOther = 4,
};

struct QueueAheadCounts {
    std::size_t n_ahead = 0;           // values written to `ahead`
    long long n_snapshot_orders = 0;
    long long n_unseeded_adds = 0;
    long long n_unseen_decrements = 0;
    double unseen_size = 0.0;          // summed in event order, from 0.0
};

/**
 * `queue_positions`: per event, a CLEAR forgets every order and level and
 * marks the book seen; an event whose price or size is not finite is
 * skipped; an ADD records the size resting at its level (a snapshot ADD
 * records nothing and marks the book seen) and then rests its own size
 * there; a CANCEL or FILL of an order this window added takes
 * min(size, remaining) from that order's level, floored at 0.0, and drops
 * the order once nothing remains; one of an order it never saw is counted,
 * with its size, as unseen.
 *
 * `order_codes` must lie in [0, n_orders) on every ADD, CANCEL and FILL
 * row, and `side_codes` must be >= 0 on every ADD row (the binding checks);
 * `ahead` has room for n_events values.
 */
bool order_queue_ahead(const long long* order_codes, std::size_t n_orders,
                       const std::int8_t* actions, const long long* side_codes,
                       const double* price, const double* size,
                       const std::uint8_t* snapshot, std::size_t n_events,
                       double* ahead, QueueAheadCounts& counts);

struct LifetimeCounts {
    std::size_t n_filled = 0;          // values written to `filled`
    std::size_t n_cancelled = 0;       // values written to `cancelled`
    long long still_resting = 0;       // added and never terminated
    long long terminated_without_an_add = 0;
    long long terminated_from_snapshot = 0;
    long long known_at_open = 0;       // snapshot orders never terminated
    bool overflowed = false;           // a duration left int64; nothing valid
};

/**
 * `order_lifetimes`: per event with a timestamp (`stamps[i]` is not
 * INT64_MIN, numpy's NaT), a snapshot ADD marks its order as resting at the
 * open; another ADD (re)starts its order's clock; a CANCEL or FILL of a
 * started order writes end - start, in the timestamps' unit, to `cancelled`
 * or `filled` in event order and stops the clock, and one of an unstarted
 * order is explained by the snapshot (once) or counted as left-censored.
 *
 * The durations are integers so the caller can turn them into seconds the
 * way pandas' Timedelta.total_seconds() does. A difference outside int64
 * sets `overflowed` and stops, where pandas raises; the caller then runs
 * the Python loop, which raises as it always did.
 */
bool order_lifetimes(const long long* order_codes, std::size_t n_orders,
                     const std::int8_t* actions, const std::uint8_t* snapshot,
                     const long long* stamps, std::size_t n_events,
                     long long* filled, long long* cancelled, LifetimeCounts& counts);

}  // namespace sqt
