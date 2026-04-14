// ngram count  -n N  -o FREQ_FILE  [options]  FILE...
//
// Streams corpus files, counts n-grams with a Robin Hood hash table, spills
// sorted runs to disk when the table is full, then k-way merges all runs into
// a final frequency table.
//
// N-grams do NOT span file boundaries.

#include "common.h"
#include "robin_hash.h"
#include "run_file.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <string>
#include <vector>
#include <cstring>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>

static_assert(TOKEN_BYTES == 4,
              "cmd_count expects corpus .bin files to store uint32 tokens");

static constexpr size_t READ_TOKENS = 1 << 23;   // 8 M tokens per read (~32 MB)
static constexpr size_t LOG_EVERY   = 1 << 20;   // check progress every 1 M n-grams

// ── Flush one sorted run to disk ──────────────────────────────────────────────
static void flush_run(RobinTable *t, int n, uint32_t prune_thresh,
                      const std::string &scratch, int run_idx,
                      std::vector<std::string> &run_paths,
                      uint64_t &total_scratch_bytes) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/ngram_run_%d_%d.bin",
             scratch.c_str(), static_cast<int>(getpid()), run_idx);

    char ts_pre[16];
    Progress::timestamp(ts_pre, sizeof(ts_pre));
    printf("[%s] flush: run #%d  occupied=%zu  capacity=%zu\n",
           ts_pre, run_idx, t->occupied, t->capacity);
    fflush(stdout);

    size_t count = 0;
    uint32_t *idx = robin_sorted_indices(t, prune_thresh, &count);
    if (!idx) die("flush_run: out of memory allocating index array");

    RunWriter w;
    if (!w.open(path, n)) die("flush_run: cannot open %s: %s", path, strerror(errno));

    const size_t kb = key_bytes(n);
    const size_t sb = slot_bytes(n);
    for (size_t i = 0; i < count; i++) {
        const uint8_t *slot = t->data + (size_t)idx[i] * sb;
        uint32_t cnt;
        memcpy(&cnt, slot + kb, 4);
        w.write_entry(slot, cnt);
    }
    free(idx);
    if (!w.close()) die("flush_run: write error on %s", path);

    uint64_t run_bytes = sizeof(RunHeader) + count * sb;
    total_scratch_bytes += run_bytes;

    char ts[16], rb[32], sb2[32];
    Progress::timestamp(ts, sizeof(ts));
    fmt_bytes(run_bytes, rb, sizeof(rb));
    fmt_bytes(total_scratch_bytes, sb2, sizeof(sb2));
    printf("[%s] flush: run #%d  entries=%zu  size=%s  scratch=%s\n",
           ts, run_idx, count, rb, sb2);
    fflush(stdout);

    run_paths.emplace_back(path);
    robin_clear(t);
}

// ── K-way merge + filter → output ────────────────────────────────────────────
struct HeapEntry {
    uint8_t key[MAX_KEY_BYTES];
    uint32_t count;
    int      run_idx;
};

static void merge_and_output(const std::vector<std::string> &run_paths,
                             int n, uint32_t min_count, int top_k,
                             const char *out_path) {
    const size_t kb = key_bytes(n);

    std::vector<RunReader> readers(run_paths.size());
    for (size_t i = 0; i < run_paths.size(); i++) {
        if (!readers[i].open(run_paths[i].c_str()))
            die("merge: cannot open run %s", run_paths[i].c_str());
    }

    // Min-heap by key (priority_queue is max-heap; invert comparator)
    auto cmp = [kb](const HeapEntry &a, const HeapEntry &b) {
        return memcmp(a.key, b.key, kb) > 0;
    };
    std::priority_queue<HeapEntry,
                        std::vector<HeapEntry>,
                        decltype(cmp)> heap(cmp);

    auto push_reader = [&](int i) {
        if (readers[i].exhausted) return;
        HeapEntry e;
        memcpy(e.key, readers[i].cur_key, kb);
        e.count   = readers[i].cur_count;
        e.run_idx = i;
        heap.push(e);
        readers[i].advance();
    };
    for (int i = 0; i < (int)readers.size(); i++) push_reader(i);

    // Top-K accumulator: min-heap by count of size top_k
    struct FreqEntry {
        uint8_t  key[MAX_KEY_BYTES];
        uint32_t count;
    };
    auto count_cmp = [](const FreqEntry &a, const FreqEntry &b) {
        return a.count > b.count;   // min-heap: smallest count on top
    };
    std::priority_queue<FreqEntry,
                        std::vector<FreqEntry>,
                        decltype(count_cmp)> top_heap(count_cmp);

    RunWriter writer;
    if (!writer.open(out_path, n))
        die("merge: cannot open output %s: %s", out_path, strerror(errno));

    uint8_t  cur_key[MAX_KEY_BYTES]{};
    uint64_t cur_sum = 0;
    bool     have    = false;
    uint64_t emitted = 0;
    uint64_t merged  = 0;
    Progress prog;

    auto emit = [&](const uint8_t *key, uint32_t count) {
        if (count < min_count) return;
        emitted++;
        if (top_k > 0) {
            FreqEntry fe;
            memcpy(fe.key, key, kb);
            fe.count = count;
            if ((int)top_heap.size() < top_k) {
                top_heap.push(fe);
            } else if (count > top_heap.top().count) {
                top_heap.pop();
                top_heap.push(fe);
            }
        } else {
            writer.write_entry(key, count);
        }
    };

    while (!heap.empty()) {
        HeapEntry top = heap.top(); heap.pop();
        push_reader(top.run_idx);
        merged++;

        if (have && memcmp(cur_key, top.key, kb) == 0) {
            cur_sum = std::min(cur_sum + top.count, (uint64_t)UINT32_MAX);
        } else {
            if (have) emit(cur_key, static_cast<uint32_t>(cur_sum));
            memcpy(cur_key, top.key, kb);
            cur_sum = top.count;
            have    = true;
        }

        if (merged % LOG_EVERY == 0 && prog.tick()) {
            char ts[16]; char mc[32]; char ec[32];
            Progress::timestamp(ts, sizeof(ts));
            fmt_count(merged, mc, sizeof(mc));
            fmt_count(emitted, ec, sizeof(ec));
            printf("[%s] merge: %s entries merged  %s passed filter\n",
                   ts, mc, ec);
            fflush(stdout);
        }
    }
    if (have) emit(cur_key, static_cast<uint32_t>(cur_sum));

    // Top-K: sort by key before writing
    if (top_k > 0) {
        std::vector<FreqEntry> results;
        while (!top_heap.empty()) { results.push_back(top_heap.top()); top_heap.pop(); }
        std::sort(results.begin(), results.end(),
                  [kb](const FreqEntry &a, const FreqEntry &b) {
                      return memcmp(a.key, b.key, kb) < 0;
                  });
        for (auto &fe : results) writer.write_entry(fe.key, fe.count);
    }

    if (!writer.close()) die("merge: write error on output %s", out_path);
    for (auto &r : readers) r.close();
}

// ── Subcommand entry point ────────────────────────────────────────────────────
static void usage_count() {
    fputs(
        "usage: ngram count -n N -o FREQ_FILE [options] FILE...\n"
        "\n"
        "Count n-gram frequencies across corpus files and write a binary\n"
        "frequency table suitable for downstream 'ngram process' steps.\n"
        "\n"
        "Required:\n"
        "  -n N          N-gram size (1–8)\n"
        "  -o FILE       Output frequency table\n"
        "  FILE...       One or more binary corpus files with little-endian\n"
        "                uint32 tokens (4 bytes per token)\n"
        "\n"
        "Options:\n"
        "  -m GB         Hash table memory in GB (default: 4)\n"
        "  -t T          Flush-prune threshold: drop entries with per-file\n"
        "                count <= T when spilling to disk (default: 1)\n"
        "                Set -t 0 to keep all entries at flush time.\n"
        "  -s DIR        Scratch directory for sorted run files (default: /tmp)\n"
        "  --min-count M Only emit n-grams with total count >= M (default: 1)\n"
        "  --top-k K     Only emit the K most frequent n-grams\n"
        "\n"
        "Examples:\n"
        "  # Count 2-grams across all corpus shards\n"
        "  ngram count -n 2 -o data/2gram.count.bin data/*.bin\n"
        "\n"
        "  # Count 3-grams, 8 GB table, keep only n-grams seen >= 10 times\n"
        "  ngram count -n 3 -m 8 --min-count 10 -o data/3gram.count.bin data/*.bin\n"
        "\n"
        "  # Count unigrams, emit only the top 100K most frequent\n"
        "  ngram count -n 1 --top-k 100000 -o data/1gram.count.bin data/*.bin\n",
        stderr);
}

int cmd_count(int argc, char **argv) {
    if (argc == 0) { usage_count(); return 1; }

    // Defaults
    int         n             = 0;
    size_t      mem_gb        = 4;
    uint32_t    prune_thresh  = 1;
    std::string scratch       = "/tmp";
    std::string out_path;
    uint32_t    min_count     = 1;
    int         top_k         = 0;
    std::vector<const char *> files;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_count(); return 0; }
        else if (!strcmp(argv[i], "-n") && i+1 < argc)           n            = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i+1 < argc)           mem_gb       = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i+1 < argc)           prune_thresh = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i+1 < argc)           scratch      = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc)           out_path     = argv[++i];
        else if (!strcmp(argv[i], "--min-count") && i+1 < argc)  min_count    = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-k")     && i+1 < argc)  top_k        = atoi(argv[++i]);
        else if (argv[i][0] != '-')                               files.push_back(argv[i]);
    }

    if (n < 1 || n > (int)MAX_N) { usage_count(); die("count: -n must be 1..%zu (got %d)", MAX_N, n); }
    if (out_path.empty())        { usage_count(); die("count: -o OUTPUT required"); }
    if (files.empty())           { usage_count(); die("count: no input files"); }

    RobinTable table;
    size_t mem_bytes = mem_gb * (1ULL << 30);
    if (!robin_init(&table, n, mem_bytes))
        die("count: failed to allocate %.0f GB hash table", (double)mem_gb);

    char ts[16]; char mb[32]; char bgt[32];
    Progress::timestamp(ts, sizeof(ts));
    fmt_bytes(mem_bytes, bgt, sizeof(bgt));
    fmt_bytes(table.capacity * table.slot_bytes_, mb, sizeof(mb));
    printf("[%s] count: n=%d  token_width=%zu  budget=%s  table_slots=%zu  data=%s\n",
           ts, n, TOKEN_BYTES, bgt, table.capacity, mb);
    fflush(stdout);

    std::vector<std::string> run_paths;
    uint64_t total_scratch_bytes = 0;
    int      run_idx             = 0;
    uint64_t total_tokens        = 0;
    uint64_t total_ngrams        = 0;
    uint64_t total_bytes         = 0;
    Progress prog;

    std::vector<Token> buf(READ_TOKENS + MAX_N);   // extra headroom for overlap
    size_t overlap = 0;   // tokens carried from previous buffer (within a file)

    for (const char *fpath : files) {
        FILE *fp = fopen(fpath, "rb");
        if (!fp) die("count: cannot open %s: %s", fpath, strerror(errno));

        struct stat st{};
        if (stat(fpath, &st) != 0)
            die("count: cannot stat %s: %s", fpath, strerror(errno));
        uint64_t file_bytes = static_cast<uint64_t>(st.st_size);
        if (file_bytes % TOKEN_BYTES != 0)
            die("count: file size of %s (%llu bytes) is not divisible by %zu-byte tokens",
                fpath,
                (unsigned long long)file_bytes,
                TOKEN_BYTES);
        total_bytes += file_bytes;
        overlap = 0;   // n-grams do not span file boundaries

        while (true) {
            size_t got = fread(buf.data() + overlap, TOKEN_BYTES,
                               READ_TOKENS, fp);
            if (got == 0) break;

            size_t total_in_buf = overlap + got;
            size_t ngrams_in_buf = (total_in_buf >= (size_t)n)
                                 ? (total_in_buf - (size_t)n + 1) : 0;

            for (size_t i = 0; i < ngrams_in_buf; i++) {
                robin_increment(&table, reinterpret_cast<const uint8_t *>(&buf[i]));
                total_ngrams++;

                if (total_ngrams % LOG_EVERY == 0 && prog.tick()) {
                    char ts2[16]; char tc[32]; char tb[32];
                    Progress::timestamp(ts2, sizeof(ts2));
                    fmt_count(total_ngrams, tc, sizeof(tc));
                    fmt_bytes(total_tokens * TOKEN_BYTES, tb, sizeof(tb));
                    printf("[%s] count: %s n-grams  %.2f GB read  "
                           "table %zu/%zu (%.0f%%)\n",
                           ts2, tc, total_tokens * TOKEN_BYTES / (double)(1ULL << 30),
                           table.occupied, table.max_occupied,
                           100.0 * table.occupied / table.max_occupied);
                    fflush(stdout);
                }

                if (robin_needs_flush(&table)) {
                    flush_run(&table, n, prune_thresh, scratch,
                              run_idx++, run_paths, total_scratch_bytes);
                }
            }

            total_tokens += got;
            // Carry last n-1 tokens for overlap within this file
            size_t carry = (total_in_buf >= (size_t)(n-1))
                         ? (size_t)(n-1) : total_in_buf;
            memmove(buf.data(), buf.data() + total_in_buf - carry,
                    carry * TOKEN_BYTES);
            overlap = carry;
        }
        fclose(fp);
    }

    // Final flush
    if (table.occupied > 0) {
        flush_run(&table, n, prune_thresh, scratch,
                  run_idx++, run_paths, total_scratch_bytes);
    }
    robin_free(&table);

    // K-way merge → frequency table
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] merge: %d runs  total_tokens=%llu\n",
           ts, run_idx, (unsigned long long)total_tokens);
    fflush(stdout);

    merge_and_output(run_paths, n, min_count, top_k, out_path.c_str());

    // Remove scratch run files
    for (auto &p : run_paths) remove(p.c_str());

    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] done.\n", ts);
    fflush(stdout);
    return 0;
}
