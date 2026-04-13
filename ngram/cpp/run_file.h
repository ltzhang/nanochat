#pragma once
// Sorted-run binary file format (used for intermediate runs and the final
// frequency table):
//
//   Header (16 bytes):
//     char     magic[4]      = "NGRM"
//     uint16_t n             n-gram width (1–8)
//     uint16_t sort_by_key   1 if entries are sorted by raw key bytes, else 0
//     uint64_t entry_count
//
//   Entries: entry_count × slot_bytes(n) bytes
//     uint8_t  key[n × 4]
//     uint32_t count

#include "common.h"
#include <cstdio>
#include <cstring>

#pragma pack(push, 1)
struct RunHeader {
    char     magic[4];
    uint16_t n;
    uint16_t sort_by_key;
    uint64_t entry_count;
};
#pragma pack(pop)
static_assert(sizeof(RunHeader) == 16, "RunHeader must be 16 bytes");

// ── Writer ────────────────────────────────────────────────────────────────────
// Write pre-sorted entries one by one; updates entry_count in header on close.
struct RunWriter {
    FILE    *fp        = nullptr;
    int      n_        = 0;
    bool     sort_by_key_ = true;
    uint64_t written_  = 0;

    bool open(const char *path, int n, bool sort_by_key = true) {
        fp = fopen(path, "wb");
        if (!fp) return false;
        n_           = n;
        sort_by_key_ = sort_by_key;
        written_     = 0;
        RunHeader hdr{};
        memcpy(hdr.magic, "NGRM", 4);
        hdr.n           = static_cast<uint16_t>(n);
        hdr.sort_by_key = sort_by_key ? 1 : 0;
        fwrite(&hdr, sizeof(hdr), 1, fp);   // placeholder; rewritten on close
        return true;
    }

    void write_entry(const uint8_t *key, uint32_t count) {
        fwrite(key, key_bytes(n_), 1, fp);
        fwrite(&count, 4, 1, fp);
        written_++;
    }

    bool close() {
        if (!fp) return true;
        rewind(fp);
        RunHeader hdr{};
        memcpy(hdr.magic, "NGRM", 4);
        hdr.n           = static_cast<uint16_t>(n_);
        hdr.sort_by_key = sort_by_key_ ? 1 : 0;
        hdr.entry_count = written_;
        bool ok = (fwrite(&hdr, sizeof(hdr), 1, fp) == 1);
        ok &= (fclose(fp) == 0);
        fp = nullptr;
        return ok;
    }
};

// ── Reader ────────────────────────────────────────────────────────────────────
// Forward-sequential reader; cur_key / cur_count hold the current entry.
struct RunReader {
    FILE     *fp        = nullptr;
    RunHeader hdr{};
    uint64_t  pos_      = 0;
    bool      exhausted = true;

    uint8_t  cur_key[MAX_KEY_BYTES]{};
    uint32_t cur_count = 0;

    bool open(const char *path) {
        fp = fopen(path, "rb");
        if (!fp) return false;
        pos_ = 0;
        if (fread(&hdr, sizeof(hdr), 1, fp) != 1)              return false;
        if (memcmp(hdr.magic, "NGRM", 4) != 0)                 return false;
        if (hdr.n < 1 || hdr.n > MAX_N)                        return false;
        exhausted = (hdr.entry_count == 0);
        if (!exhausted) advance();
        return true;
    }

    // Load next entry into cur_key / cur_count; sets exhausted on EOF.
    void advance() {
        if (pos_ >= hdr.entry_count) { exhausted = true; return; }
        size_t kb = key_bytes(hdr.n);
        if (fread(cur_key, kb, 1, fp) != 1 ||
            fread(&cur_count, 4, 1, fp) != 1) { exhausted = true; return; }
        pos_++;
    }

    void close() { if (fp) { fclose(fp); fp = nullptr; } }
};
