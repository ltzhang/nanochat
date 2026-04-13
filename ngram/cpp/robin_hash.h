#pragma once
// Robin Hood open-addressing hash table.
//
// Key  : fixed-width byte array (key_bytes = n × TOKEN_BYTES).
// Value: uint32_t count, saturating at UINT32_MAX.
// Empty slot sentinel: count == 0 (no valid entry has count 0).
// PSL (probe-sequence length) is stored in a separate parallel array so the
// main data array has the same layout as on-disk sorted-run entries.
//
// xxHash3 header: include common.h first, then this header.

#include "common.h"
#include <algorithm>
#include <vector>
#include <cassert>

#define XXH_STATIC_LINKING_ONLY
#define XXH_INLINE_ALL
#include "xxhash.h"

struct RobinTable {
    uint8_t *data;          // capacity × slot_bytes: [key | count(u32)]
    uint8_t *psl;           // capacity × 1: probe-sequence length per slot
    size_t   capacity;      // number of slots, power of 2
    size_t   mask;          // capacity - 1
    size_t   key_bytes_;
    size_t   slot_bytes_;
    size_t   occupied;
    size_t   max_occupied;  // capacity × 3 / 4
};

// Initialize.  mem_bytes is the memory budget for data + psl combined.
// Returns false on allocation failure.
inline bool robin_init(RobinTable *t, int n, size_t mem_bytes) {
    t->key_bytes_  = key_bytes(n);
    t->slot_bytes_ = slot_bytes(n);

    // Largest power-of-2 capacity s.t. cap*(slot_bytes+1) <= mem_bytes
    size_t cap = 1;
    while ((cap * 2) * (t->slot_bytes_ + 1) <= mem_bytes) cap <<= 1;

    t->capacity     = cap;
    t->mask         = cap - 1;
    t->max_occupied = cap * 3 / 4;
    t->occupied     = 0;

    t->data = static_cast<uint8_t *>(calloc(cap, t->slot_bytes_));
    t->psl  = static_cast<uint8_t *>(calloc(cap, 1));
    return t->data && t->psl;
}

inline void robin_free(RobinTable *t) {
    free(t->data); t->data = nullptr;
    free(t->psl);  t->psl  = nullptr;
}

inline void robin_clear(RobinTable *t) {
    memset(t->data, 0, t->capacity * t->slot_bytes_);
    memset(t->psl,  0, t->capacity);
    t->occupied = 0;
}

inline bool robin_needs_flush(const RobinTable *t) {
    return t->occupied >= t->max_occupied;
}

// Insert key or increment its count.  key must be key_bytes_ bytes wide.
inline void robin_increment(RobinTable *t, const uint8_t *key) {
    const size_t kb  = t->key_bytes_;
    const size_t sb  = t->slot_bytes_;
    uint64_t     h   = XXH3_64bits(key, kb);
    size_t       pos = h & t->mask;
    uint8_t      cur_psl = 0;

    // Phase 1: search for existing key
    while (true) {
        uint8_t *slot = t->data + pos * sb;
        uint32_t cnt;
        memcpy(&cnt, slot + kb, 4);

        if (cnt == 0 || t->psl[pos] < cur_psl) break;   // key absent

        if (memcmp(slot, key, kb) == 0) {
            if (cnt < UINT32_MAX) cnt++;
            memcpy(slot + kb, &cnt, 4);
            return;
        }
        pos = (pos + 1) & t->mask;
        cur_psl++;
    }

    // Phase 2: Robin Hood insertion of (key, 1) starting at pos / cur_psl
    uint8_t  ins_key[MAX_KEY_BYTES];
    uint32_t ins_cnt = 1;
    uint8_t  ins_psl = cur_psl;
    memcpy(ins_key, key, kb);

    while (true) {
        uint8_t *slot = t->data + pos * sb;
        uint32_t cnt;
        memcpy(&cnt, slot + kb, 4);

        if (cnt == 0) {
            memcpy(slot, ins_key, kb);
            memcpy(slot + kb, &ins_cnt, 4);
            t->psl[pos] = ins_psl;
            t->occupied++;
            return;
        }
        if (t->psl[pos] < ins_psl) {
            // Evict current occupant; continue inserting it
            uint8_t  tmp_key[MAX_KEY_BYTES];
            uint32_t tmp_cnt;
            uint8_t  tmp_psl = t->psl[pos];
            memcpy(tmp_key, slot, kb);
            memcpy(&tmp_cnt, slot + kb, 4);

            memcpy(slot, ins_key, kb);
            memcpy(slot + kb, &ins_cnt, 4);
            t->psl[pos] = ins_psl;

            memcpy(ins_key, tmp_key, kb);
            ins_cnt = tmp_cnt;
            ins_psl = tmp_psl;
        }
        pos = (pos + 1) & t->mask;
        ins_psl++;
    }
}

// Collect indices of occupied slots with count > prune_thresh,
// sorted lexicographically by key.
// Returns a heap-allocated array; caller must free().
// *out_count receives the number of elements.
inline uint32_t *robin_sorted_indices(const RobinTable *t,
                                      uint32_t          prune_thresh,
                                      size_t           *out_count) {
    const size_t kb = t->key_bytes_;
    const size_t sb = t->slot_bytes_;

    char ts[16];
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] sort: scanning %zu slots ...\n", ts, t->capacity);
    fflush(stdout);

    std::vector<uint32_t> idx;
    idx.reserve(t->occupied);
    Progress scan_prog;
    for (size_t i = 0; i < t->capacity; i++) {
        uint32_t cnt;
        memcpy(&cnt, t->data + i * sb + kb, 4);
        if (cnt > prune_thresh) idx.push_back(static_cast<uint32_t>(i));
        if ((i & ((1u << 20) - 1)) == 0 && scan_prog.tick()) {
            char ts2[16]; char sc[32];
            Progress::timestamp(ts2, sizeof(ts2));
            fmt_count(i, sc, sizeof(sc));
            printf("[%s] sort: scanned %s / %zu slots  collected %zu\n",
                   ts2, sc, t->capacity, idx.size());
            fflush(stdout);
        }
    }

    Progress::timestamp(ts, sizeof(ts));
    char ec[32]; fmt_count(idx.size(), ec, sizeof(ec));
    printf("[%s] sort: sorting %s entries ...\n", ts, ec);
    fflush(stdout);

    const uint8_t *base = t->data;
    std::sort(idx.begin(), idx.end(), [base, sb, kb](uint32_t a, uint32_t b) {
        return memcmp(base + a * sb, base + b * sb, kb) < 0;
    });

    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] sort: done\n", ts);
    fflush(stdout);

    *out_count = idx.size();
    uint32_t *arr = static_cast<uint32_t *>(malloc(idx.size() * sizeof(uint32_t)));
    if (arr) memcpy(arr, idx.data(), idx.size() * sizeof(uint32_t));
    return arr;
}
