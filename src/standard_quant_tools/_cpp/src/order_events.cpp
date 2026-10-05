#include "sqt/fp_contract.hpp"  // first: no contraction in this unit
#include "sqt/order_events.hpp"

#include "sqt/numerics.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace sqt {
namespace {

// A price level's identity: the side code and the price's bits, with -0.0
// read as +0.0 (the two are one dict key: equal, and hash(-0.0) == 0).
inline std::uint64_t price_key(double price) {
    const double canonical = (price == 0.0) ? 0.0 : price;
    std::uint64_t bits = 0;
    std::memcpy(&bits, &canonical, sizeof bits);
    return bits;
}

// Open addressing with linear probing over (side, price bits) -> level
// index. A level is never removed -- a CLEAR forgets values, not identities
// -- so there are no tombstones, and the table doubles at half full.
class LevelTable {
public:
    bool reserve(std::size_t expected) {
        std::size_t capacity = 16;
        while (capacity < 2 * expected) capacity *= 2;
        return rehash(capacity);
    }

    // The level's index, inserting it if new; SIZE_MAX if growing failed.
    std::size_t find_or_insert(long long side, std::uint64_t bits) {
        if (2 * (size_ + 1) > slots_.size() && !rehash(2 * slots_.size()))
            return std::numeric_limits<std::size_t>::max();
        std::size_t at = hash(side, bits) & (slots_.size() - 1);
        for (;;) {
            Slot& slot = slots_[at];
            if (slot.index == kEmpty) {
                slot = Slot{side, bits, size_};
                return size_++;
            }
            if (slot.side == side && slot.bits == bits) return slot.index;
            at = (at + 1) & (slots_.size() - 1);
        }
    }

    std::size_t size() const { return size_; }

private:
    static constexpr std::size_t kEmpty = std::numeric_limits<std::size_t>::max();
    struct Slot {
        long long side;
        std::uint64_t bits;
        std::size_t index;
    };

    static std::size_t hash(long long side, std::uint64_t bits) {
        std::uint64_t h = bits ^ (static_cast<std::uint64_t>(side) * 0x9E3779B97F4A7C15ULL);
        h ^= h >> 33;
        h *= 0xFF51AFD7ED558CCDULL;
        h ^= h >> 33;
        h *= 0xC4CEB9FE1A85EC53ULL;
        h ^= h >> 33;
        return static_cast<std::size_t>(h);
    }

    bool rehash(std::size_t capacity) {
        std::vector<Slot> next;
        try {
            next.assign(capacity, Slot{0, 0, kEmpty});
        } catch (const std::bad_alloc&) {
            return false;
        }
        for (const Slot& slot : slots_) {
            if (slot.index == kEmpty) continue;
            std::size_t at = hash(slot.side, slot.bits) & (capacity - 1);
            while (next[at].index != kEmpty) at = (at + 1) & (capacity - 1);
            next[at] = slot;
        }
        slots_.swap(next);
        return true;
    }

    std::vector<Slot> slots_;
    std::size_t size_ = 0;
};

}  // namespace

bool order_queue_ahead(const long long* order_codes, std::size_t n_orders,
                       const std::int8_t* actions, const long long* side_codes,
                       const double* price, const double* size,
                       const std::uint8_t* snapshot, std::size_t n_events,
                       double* ahead, QueueAheadCounts& counts) {
    counts = QueueAheadCounts{};
    // Per order: its level, the size still resting, and the generation it
    // was added in (current generation = present in the loop's `orders`).
    // Per level: the running total and its generation (current = present
    // in the loop's `resting`; absent reads as 0.0, as .get(key, 0.0) does).
    std::vector<std::size_t> order_level;
    std::vector<double> order_remaining;
    std::vector<std::size_t> order_generation;
    std::vector<double> level_resting;
    std::vector<std::size_t> level_generation;
    LevelTable levels;
    try {
        order_level.assign(n_orders, 0);
        order_remaining.assign(n_orders, 0.0);
        order_generation.assign(n_orders, 0);
    } catch (const std::bad_alloc&) {
        return false;
    }
    if (!levels.reserve(1024)) return false;

    std::size_t generation = 1;  // 0 marks "never present"
    bool seeded = false;
    std::size_t written = 0;
    double unseen_size = 0.0;

    for (std::size_t i = 0; i < n_events; ++i) {
        const std::int8_t action = actions[i];
        if (action == kOrderClear) {
            ++generation;
            seeded = true;
            continue;
        }
        const double p = price[i];
        const double s = size[i];
        if (!numerics::is_finite(p) || !numerics::is_finite(s)) continue;
        if (action == kOrderAdd) {
            const std::size_t level = levels.find_or_insert(side_codes[i], price_key(p));
            if (level == std::numeric_limits<std::size_t>::max()) return false;
            if (level >= level_resting.size()) {
                try {
                    level_resting.push_back(0.0);
                    level_generation.push_back(0);
                } catch (const std::bad_alloc&) {
                    return false;
                }
            }
            const double current =
                (level_generation[level] == generation) ? level_resting[level] : 0.0;
            if (snapshot[i]) {
                ++counts.n_snapshot_orders;
                seeded = true;
            } else {
                ahead[written++] = current;
                if (!seeded) ++counts.n_unseeded_adds;
            }
            level_resting[level] = current + s;
            level_generation[level] = generation;
            const auto order = static_cast<std::size_t>(order_codes[i]);
            order_level[order] = level;
            order_remaining[order] = s;
            order_generation[order] = generation;
        } else if (action == kOrderCancel || action == kOrderFill) {
            const auto order = static_cast<std::size_t>(order_codes[i]);
            if (order_generation[order] != generation) {
                ++counts.n_unseen_decrements;
                unseen_size += s;
                continue;
            }
            const std::size_t level = order_level[order];
            const double remaining = order_remaining[order];
            // Python's min(size, remaining) and max(0.0, x), as written:
            // the first argument unless the second is strictly smaller
            // (larger), so a NaN or a tie goes to the first.
            const double taken = (remaining < s) ? remaining : s;
            const double current =
                (level_generation[level] == generation) ? level_resting[level] : 0.0;
            const double reduced = current - taken;
            level_resting[level] = (reduced > 0.0) ? reduced : 0.0;
            level_generation[level] = generation;
            const double left = remaining - taken;
            if (left > 0.0) {
                order_remaining[order] = left;
            } else {
                order_generation[order] = 0;
            }
        }
    }
    counts.n_ahead = written;
    counts.unseen_size = unseen_size;
    return true;
}

bool order_lifetimes(const long long* order_codes, std::size_t n_orders,
                     const std::int8_t* actions, const std::uint8_t* snapshot,
                     const long long* stamps, std::size_t n_events,
                     long long* filled, long long* cancelled, LifetimeCounts& counts) {
    counts = LifetimeCounts{};
    constexpr long long kNaT = std::numeric_limits<long long>::min();
    constexpr long long kMax = std::numeric_limits<long long>::max();
    // Per order: whether it is in the loop's `added` (and since when), and
    // whether it is in `known_at_open`.
    std::vector<long long> started_at;
    std::vector<std::uint8_t> started;
    std::vector<std::uint8_t> known;
    try {
        started_at.assign(n_orders, 0);
        started.assign(n_orders, 0);
        known.assign(n_orders, 0);
    } catch (const std::bad_alloc&) {
        return false;
    }

    for (std::size_t i = 0; i < n_events; ++i) {
        const long long when = stamps[i];
        if (when == kNaT) continue;
        const std::int8_t action = actions[i];
        if (action == kOrderAdd) {
            const auto order = static_cast<std::size_t>(order_codes[i]);
            if (snapshot[i]) {
                known[order] = 1;
            } else {
                started[order] = 1;
                started_at[order] = when;
            }
        } else if (action == kOrderCancel || action == kOrderFill) {
            const auto order = static_cast<std::size_t>(order_codes[i]);
            if (!started[order]) {
                if (known[order]) {
                    known[order] = 0;
                    ++counts.terminated_from_snapshot;
                } else {
                    ++counts.terminated_without_an_add;
                }
                continue;
            }
            started[order] = 0;
            const long long start = started_at[order];
            // when - start without leaving int64: pandas raises there.
            if ((start < 0 && when > kMax + start) ||
                (start > 0 && when < kNaT + start)) {
                counts.overflowed = true;
                return true;
            }
            const long long lived = when - start;
            if (action == kOrderCancel) {
                cancelled[counts.n_cancelled++] = lived;
            } else {
                filled[counts.n_filled++] = lived;
            }
        }
    }
    for (std::size_t order = 0; order < n_orders; ++order) {
        counts.still_resting += started[order];
        counts.known_at_open += known[order];
    }
    return true;
}

}  // namespace sqt
