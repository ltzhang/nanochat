#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cstdarg>

// ── Token type (current corpus format) ────────────────────────────────────────
using Token = uint32_t;

static constexpr size_t TOKEN_BYTES   = sizeof(Token);
static constexpr size_t MAX_N         = 8;
static constexpr size_t MAX_KEY_BYTES = MAX_N * 4;   // worst-case: n=8 × uint32

inline size_t key_bytes(int n)  { return (size_t)n * TOKEN_BYTES; }
inline size_t slot_bytes(int n) { return key_bytes(n) + sizeof(uint32_t); }

// ── Fatal error ───────────────────────────────────────────────────────────────
[[noreturn]] inline void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

// ── Progress logging ──────────────────────────────────────────────────────────
struct Progress {
    time_t last_log  = 0;
    int    interval  = 5;      // seconds between lines

    // Returns true when it is time to log; updates internal state.
    bool tick() {
        time_t now = time(nullptr);
        if (now - last_log >= interval) { last_log = now; return true; }
        return false;
    }

    // Fills buf with "HH:MM:SS".
    static void timestamp(char *buf, size_t len) {
        time_t now = time(nullptr);
        struct tm *t = localtime(&now);
        snprintf(buf, len, "%02d:%02d:%02d", t->tm_hour, t->tm_min, t->tm_sec);
    }
};

// ── Formatting helpers ────────────────────────────────────────────────────────
inline void fmt_bytes(uint64_t b, char *out, size_t len) {
    if      (b >= (1ULL << 30)) snprintf(out, len, "%.2f GB", b / (double)(1ULL << 30));
    else if (b >= (1ULL << 20)) snprintf(out, len, "%.2f MB", b / (double)(1ULL << 20));
    else                         snprintf(out, len, "%llu B",  (unsigned long long)b);
}

inline void fmt_count(uint64_t c, char *out, size_t len) {
    if      (c >= 1000000000ULL) snprintf(out, len, "%.2f B", c / 1.0e9);
    else if (c >= 1000000ULL)    snprintf(out, len, "%.2f M", c / 1.0e6);
    else if (c >= 1000ULL)       snprintf(out, len, "%.1f K", c / 1.0e3);
    else                          snprintf(out, len, "%llu",   (unsigned long long)c);
}
