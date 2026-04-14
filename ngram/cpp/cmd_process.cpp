// ngram process <op> [options]
//
// Post-processing operations on binary frequency tables from "ngram count".
// -o is optional for all write commands; a descriptive default is derived
// from the input path when omitted.
//
//   filter-min  -i INPUT [-o OUTPUT] -t THRESHOLD
//       Keep only entries with count >= THRESHOLD.
//       Default output: <stem>.min<T><ext>
//
//   sort-count  -i INPUT [-o OUTPUT]
//       Re-order entries by count descending.
//       Prints top-k cutoff counts (k=1..10, 100, 1K, 10K, then every 100K).
//       Also writes the same table to <stem>_sort_hist.txt next to INPUT.
//       Default output: <stem>.sorted<ext>
//
//   to-text     -i INPUT [-o OUTPUT] [--vocab VOCAB_FILE]
//       Write human-readable text: one line per entry, tokens TAB count.
//       Without --vocab: space-separated token IDs.
//       With --vocab: detokenized text (tokens concatenated).
//       Default output: <stem>.txt
//
//   merge       -i INPUT [-i INPUT ...] [-o OUTPUT] [--min-count M]
//       K-way merge of multiple sorted frequency files.
//       Default output: <first-input-stem>.merged<ext>
//
//   split       -i INPUT -t THRESHOLD
//       Split INPUT into two files by count threshold (no -o; names are fixed).
//       Entries with count >= THRESHOLD -> <stem>.high_<T><ext>
//       Entries with count <  THRESHOLD -> <stem>.low_<T><ext>
//       Original file is not modified.

#include "common.h"
#include "run_file.h"

#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <cstring>
#include <cerrno>

static constexpr size_t LOG_EVERY = 1 << 20;   // progress every 1 M entries

// ── usage ─────────────────────────────────────────────────────────────────────
static void usage_process() {
    fputs(
        "usage: ngram process <op> [options]\n"
        "\n"
        "  filter-min  -i INPUT [-o OUTPUT] -t THRESHOLD\n"
        "      Keep entries with count >= THRESHOLD.\n"
        "      Default output: <stem>.min<T><ext>\n"
        "\n"
        "  sort-count  -i INPUT [-o OUTPUT]\n"
        "      Sort entries by count descending; print top-k cutoff table.\n"
        "      Default output: <stem>.sorted<ext>\n"
        "      Histogram log: <stem>_sort_hist.txt (next to input)\n"
        "\n"
        "  to-text     -i INPUT [-o OUTPUT] [--vocab VOCAB_FILE]\n"
        "      Dump as text: one line per entry, tokens TAB count.\n"
        "      Default output: <stem>.txt\n"
        "\n"
        "  merge       -i INPUT [-i INPUT ...] [-o OUTPUT] [--min-count M]\n"
        "      K-way merge of sorted frequency files.\n"
        "      Default output: <first-input-stem>.merged<ext>\n"
        "\n"
        "  split       -i INPUT -t THRESHOLD\n"
        "      Split into high/low files by count threshold.\n"
        "      High (>= T): <stem>.high_<T><ext>\n"
        "      Low  (<  T): <stem>.low_<T><ext>\n",
        stderr);
}

// ── helpers ───────────────────────────────────────────────────────────────────

// Given "/some/path/foo.bin" returns "foo" (basename without extension).
static std::string stem(const std::string &path) {
    // strip directory
    size_t slash = path.rfind('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    // strip extension
    size_t dot = base.rfind('.');
    if (dot != std::string::npos) base.resize(dot);
    return base;
}

// Given "/some/path/foo.bin" returns ".bin" (including dot) or "" if none.
static std::string ext(const std::string &path) {
    size_t slash = path.rfind('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.rfind('.');
    if (dot == std::string::npos) return "";
    return base.substr(dot);
}

// Given "/some/path/foo.bin" returns "/some/path/" (including trailing slash) or "".
static std::string dirpart(const std::string &path) {
    size_t slash = path.rfind('/');
    if (slash == std::string::npos) return "";
    return path.substr(0, slash + 1);
}

// Print and log top-k cutoff counts.  entries[] is sorted descending by count.
// Milestones: 1..10, 100, 1K, 10K, then every 100K.
static void print_topk_histogram(const std::vector<uint32_t> &counts,
                                  const std::string &in_path) {
    const uint64_t total = counts.size();

    // Build milestone list
    std::vector<uint64_t> milestones;
    for (uint64_t k = 1; k <= 10 && k <= total; k++) milestones.push_back(k);
    for (uint64_t k : {(uint64_t)100, (uint64_t)1000, (uint64_t)10000}) {
        if (k > 10 && k <= total) milestones.push_back(k);
    }
    for (uint64_t k = 100000; k <= total; k += 100000) milestones.push_back(k);
    // deduplicate (milestones are already increasing by construction, but just in case)
    milestones.erase(std::unique(milestones.begin(), milestones.end()), milestones.end());

    // Log file: placed next to the input, named <stem>_sort_hist.txt
    std::string log_path = dirpart(in_path) + stem(in_path) + "_sort_hist.txt";
    FILE *lf = fopen(log_path.c_str(), "w");
    if (!lf) {
        fprintf(stderr, "sort-count: warning: cannot open log %s: %s\n",
                log_path.c_str(), strerror(errno));
    }

    auto emit = [&](const char *line) {
        fputs(line, stdout);
        if (lf) fputs(line, lf);
    };

    char header[256];
    snprintf(header, sizeof(header),
             "top-k cutoff counts (source: %s, total=%llu entries):\n",
             in_path.c_str(), (unsigned long long)total);
    emit(header);

    for (uint64_t k : milestones) {
        uint32_t cutoff = counts[k - 1];   // 0-indexed; k-th entry is index k-1
        char line[128];
        if (k < 1000) {
            snprintf(line, sizeof(line), "  top %7llu : count >= %u\n",
                     (unsigned long long)k, cutoff);
        } else {
            snprintf(line, sizeof(line), "  top %6lluK : count >= %u\n",
                     (unsigned long long)(k / 1000), cutoff);
        }
        emit(line);
    }
    fflush(stdout);

    if (lf) {
        fclose(lf);
        printf("sort-count: histogram written to %s\n", log_path.c_str());
        fflush(stdout);
    }
}

// ── filter-min ────────────────────────────────────────────────────────────────
static int process_filter_min(int argc, char **argv) {
    std::string in_path, out_path;
    uint32_t threshold = 1;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_process(); return 0; }
        else if (!strcmp(argv[i], "-i") && i+1 < argc) in_path   = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path  = argv[++i];
        else if (!strcmp(argv[i], "-t") && i+1 < argc) threshold = (uint32_t)atoi(argv[++i]);
    }
    if (in_path.empty()) { usage_process(); die("process filter-min: -i INPUT required"); }
    if (out_path.empty()) out_path = dirpart(in_path) + stem(in_path) + ".min" + std::to_string(threshold) + ext(in_path);

    RunReader rr;
    if (!rr.open(in_path.c_str())) die("process filter-min: cannot open %s", in_path.c_str());
    int n = rr.hdr.n;

    RunWriter w;
    if (!w.open(out_path.c_str(), n, rr.hdr.sort_by_key != 0))
        die("process filter-min: cannot open %s: %s", out_path.c_str(), strerror(errno));

    uint64_t read = 0, written = 0;
    while (!rr.exhausted) {
        if (rr.cur_count >= threshold) { w.write_entry(rr.cur_key, rr.cur_count); written++; }
        rr.advance();
        read++;
    }
    rr.close();
    if (!w.close()) die("process filter-min: write error on %s", out_path.c_str());

    char ts[16]; char rc[32]; char wc[32];
    Progress::timestamp(ts, sizeof(ts));
    fmt_count(read,    rc, sizeof(rc));
    fmt_count(written, wc, sizeof(wc));
    printf("[%s] filter-min: %s read  %s written  threshold=%u\n", ts, rc, wc, threshold);
    fflush(stdout);
    return 0;
}

// ── sort-count ────────────────────────────────────────────────────────────────
static int process_sort_count(int argc, char **argv) {
    std::string in_path, out_path;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_process(); return 0; }
        else if (!strcmp(argv[i], "-i") && i+1 < argc) in_path  = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path = argv[++i];
    }
    if (in_path.empty()) { usage_process(); die("process sort-count: -i INPUT required"); }
    if (out_path.empty()) out_path = dirpart(in_path) + stem(in_path) + ".sorted" + ext(in_path);

    RunReader rr;
    if (!rr.open(in_path.c_str())) die("process sort-count: cannot open %s", in_path.c_str());
    int n = rr.hdr.n;
    const size_t kb = key_bytes(n);

    struct Entry {
        uint8_t  key[MAX_KEY_BYTES];
        uint32_t count;
    };

    char ts[16];
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] sort-count: loading %llu entries ...\n",
           ts, (unsigned long long)rr.hdr.entry_count);
    fflush(stdout);

    std::vector<Entry> entries;
    entries.reserve(rr.hdr.entry_count);
    while (!rr.exhausted) {
        Entry e;
        memcpy(e.key, rr.cur_key, kb);
        e.count = rr.cur_count;
        entries.push_back(e);
        rr.advance();
    }
    rr.close();

    char ec[32]; fmt_count(entries.size(), ec, sizeof(ec));
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] sort-count: sorting %s entries ...\n", ts, ec);
    fflush(stdout);

    std::sort(entries.begin(), entries.end(),
              [](const Entry &a, const Entry &b) { return a.count > b.count; });

    // Print top-k cutoff counts and log to file
    {
        std::vector<uint32_t> counts(entries.size());
        for (size_t i = 0; i < entries.size(); i++) counts[i] = entries[i].count;
        print_topk_histogram(counts, in_path);
    }

    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] sort-count: writing ...\n", ts);
    fflush(stdout);

    RunWriter w;
    if (!w.open(out_path.c_str(), n, false))
        die("process sort-count: cannot open %s: %s", out_path.c_str(), strerror(errno));
    for (const auto &e : entries) w.write_entry(e.key, e.count);
    if (!w.close()) die("process sort-count: write error on %s", out_path.c_str());

    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] sort-count: done  %s entries\n", ts, ec);
    fflush(stdout);
    return 0;
}

// ── to-text ───────────────────────────────────────────────────────────────────
// Load vocab file (tokenizer_vocab.txt produced by the tokenizer dump script).
// Format: token_id TAB bytes_repr TAB utf8_text  (# lines are comments)
// Returns a vector indexed by token_id; empty string means unknown.
static std::vector<std::string> load_vocab(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) die("to-text: cannot open vocab %s: %s", path, strerror(errno));

    std::vector<std::string> vocab;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        // field 1: token_id
        char *p = line;
        char *tab1 = strchr(p, '\t');
        if (!tab1) continue;
        *tab1 = '\0';
        uint32_t tid = (uint32_t)atoi(p);
        // field 2: bytes_repr (skip)
        p = tab1 + 1;
        char *tab2 = strchr(p, '\t');
        if (!tab2) continue;
        // field 3: utf8_text (strip trailing newline)
        p = tab2 + 1;
        size_t len = strlen(p);
        while (len > 0 && (p[len-1] == '\n' || p[len-1] == '\r')) p[--len] = '\0';

        if (tid >= vocab.size()) vocab.resize(tid + 1);
        vocab[tid] = p;
    }
    fclose(f);
    return vocab;
}

static int process_to_text(int argc, char **argv) {
    std::string in_path, out_path, vocab_path;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_process(); return 0; }
        else if (!strcmp(argv[i], "-i")      && i+1 < argc) in_path    = argv[++i];
        else if (!strcmp(argv[i], "-o")      && i+1 < argc) out_path   = argv[++i];
        else if (!strcmp(argv[i], "--vocab") && i+1 < argc) vocab_path = argv[++i];
    }
    if (in_path.empty()) { usage_process(); die("process to-text: -i INPUT required"); }
    if (out_path.empty()) out_path = dirpart(in_path) + stem(in_path) + ".txt";

    std::vector<std::string> vocab;
    if (!vocab_path.empty()) vocab = load_vocab(vocab_path.c_str());

    RunReader rr;
    if (!rr.open(in_path.c_str())) die("process to-text: cannot open %s", in_path.c_str());
    int n = rr.hdr.n;

    FILE *out = fopen(out_path.c_str(), "w");
    if (!out) die("process to-text: cannot open %s: %s", out_path.c_str(), strerror(errno));

    char ts[16];
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] to-text: writing %s ...\n", ts, out_path.c_str());
    fflush(stdout);

    uint64_t written = 0;
    Progress prog;
    while (!rr.exhausted) {
        if (vocab.empty()) {
            for (int i = 0; i < n; i++) {
                Token tok;
                memcpy(&tok, rr.cur_key + i * TOKEN_BYTES, TOKEN_BYTES);
                fprintf(out, i ? " %u" : "%u", (unsigned)tok);
            }
        } else {
            for (int i = 0; i < n; i++) {
                Token tok;
                memcpy(&tok, rr.cur_key + i * TOKEN_BYTES, TOKEN_BYTES);
                const char *s = (tok < (Token)vocab.size()) ? vocab[tok].c_str() : "?";
                fputs(s, out);
            }
        }
        fprintf(out, "\t%u\n", rr.cur_count);
        rr.advance();
        written++;

        if (written % LOG_EVERY == 0 && prog.tick()) {
            char ts2[16]; char wc[32];
            Progress::timestamp(ts2, sizeof(ts2));
            fmt_count(written, wc, sizeof(wc));
            printf("[%s] to-text: %s entries written\n", ts2, wc);
            fflush(stdout);
        }
    }
    rr.close();
    fclose(out);

    Progress::timestamp(ts, sizeof(ts));
    char wc[32]; fmt_count(written, wc, sizeof(wc));
    printf("[%s] to-text: done  %s entries\n", ts, wc);
    fflush(stdout);
    return 0;
}

// ── merge ─────────────────────────────────────────────────────────────────────
// K-way merge of multiple sorted frequency files with optional min-count filter.
static int process_merge(int argc, char **argv) {
    std::vector<std::string> in_paths;
    std::string out_path;
    uint32_t min_count = 1;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_process(); return 0; }
        else if (!strcmp(argv[i], "-i")          && i+1 < argc) in_paths.push_back(argv[++i]);
        else if (!strcmp(argv[i], "-o")          && i+1 < argc) out_path  = argv[++i];
        else if (!strcmp(argv[i], "--min-count") && i+1 < argc) min_count = (uint32_t)atoi(argv[++i]);
    }
    if (in_paths.empty()) { usage_process(); die("process merge: at least one -i INPUT required"); }
    if (out_path.empty()) out_path = dirpart(in_paths[0]) + stem(in_paths[0]) + ".merged" + ext(in_paths[0]);

    // Open all readers; validate n consistency.
    std::vector<RunReader> readers(in_paths.size());
    int n = 0;
    for (size_t i = 0; i < in_paths.size(); i++) {
        if (!readers[i].open(in_paths[i].c_str()))
            die("process merge: cannot open %s", in_paths[i].c_str());
        if (i == 0) {
            n = readers[i].hdr.n;
        } else if (readers[i].hdr.n != static_cast<uint16_t>(n)) {
            die("process merge: n mismatch: %s has n=%d, expected %d",
                in_paths[i].c_str(), (int)readers[i].hdr.n, n);
        }
    }

    const size_t kb = key_bytes(n);

    // Min-heap over readers, ordered by key.
    struct HeapEntry {
        uint8_t  key[MAX_KEY_BYTES];
        uint32_t count;
        int      reader_idx;
    };
    auto cmp = [kb](const HeapEntry &a, const HeapEntry &b) {
        return memcmp(a.key, b.key, kb) > 0;
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(cmp)> heap(cmp);

    auto push_reader = [&](int i) {
        if (readers[i].exhausted) return;
        HeapEntry e;
        memcpy(e.key, readers[i].cur_key, kb);
        e.count      = readers[i].cur_count;
        e.reader_idx = i;
        heap.push(e);
        readers[i].advance();
    };
    for (int i = 0; i < (int)readers.size(); i++) push_reader(i);

    RunWriter w;
    if (!w.open(out_path.c_str(), n))
        die("process merge: cannot open output %s: %s", out_path.c_str(), strerror(errno));

    uint8_t  cur_key[MAX_KEY_BYTES]{};
    uint64_t cur_sum = 0;
    bool     have    = false;
    uint64_t merged  = 0, emitted = 0;

    while (!heap.empty()) {
        HeapEntry top = heap.top(); heap.pop();
        push_reader(top.reader_idx);
        merged++;

        if (have && memcmp(cur_key, top.key, kb) == 0) {
            cur_sum = std::min(cur_sum + top.count, (uint64_t)UINT32_MAX);
        } else {
            if (have && cur_sum >= min_count) {
                w.write_entry(cur_key, static_cast<uint32_t>(cur_sum));
                emitted++;
            }
            memcpy(cur_key, top.key, kb);
            cur_sum = top.count;
            have    = true;
        }
    }
    if (have && cur_sum >= min_count) {
        w.write_entry(cur_key, static_cast<uint32_t>(cur_sum));
        emitted++;
    }

    for (auto &r : readers) r.close();
    if (!w.close()) die("process merge: write error on %s", out_path.c_str());

    char ts[16]; char mc[32]; char ec[32];
    Progress::timestamp(ts, sizeof(ts));
    fmt_count(merged,  mc, sizeof(mc));
    fmt_count(emitted, ec, sizeof(ec));
    printf("[%s] merge: %zu inputs  %s entries merged -> %s emitted"
           "  min_count=%u  out=%s\n",
           ts, in_paths.size(), mc, ec, min_count, out_path.c_str());
    fflush(stdout);
    return 0;
}

// ── split ─────────────────────────────────────────────────────────────────────
// Split a bin file into two by count threshold.
// Entries with count >= threshold -> <dir><stem>.high_<T><ext>
// Entries with count <  threshold -> <dir><stem>.low_<T><ext>
// Original file is not modified.
static int process_split(int argc, char **argv) {
    std::string in_path;
    uint32_t threshold = 0;
    bool threshold_set = false;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_process(); return 0; }
        else if (!strcmp(argv[i], "-i") && i+1 < argc) in_path   = argv[++i];
        else if (!strcmp(argv[i], "-t") && i+1 < argc) {
            threshold     = (uint32_t)atoi(argv[++i]);
            threshold_set = true;
        }
    }
    if (in_path.empty())  { usage_process(); die("process split: -i INPUT required"); }
    if (!threshold_set)   { usage_process(); die("process split: -t THRESHOLD required"); }

    // Derive output paths
    std::string d  = dirpart(in_path);
    std::string s  = stem(in_path);
    std::string e  = ext(in_path);
    char tsuf[64];
    snprintf(tsuf, sizeof(tsuf), "%u", threshold);

    std::string high_path = d + s + ".high_" + tsuf + e;
    std::string low_path  = d + s + ".low_"  + tsuf + e;

    RunReader rr;
    if (!rr.open(in_path.c_str())) die("process split: cannot open %s", in_path.c_str());
    int n = rr.hdr.n;
    bool sorted_by_key = rr.hdr.sort_by_key != 0;

    RunWriter wh, wl;
    if (!wh.open(high_path.c_str(), n, sorted_by_key))
        die("process split: cannot open %s: %s", high_path.c_str(), strerror(errno));
    if (!wl.open(low_path.c_str(), n, sorted_by_key))
        die("process split: cannot open %s: %s", low_path.c_str(), strerror(errno));

    char ts[16];
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] split: threshold=%u\n"
           "  high (>= %u) -> %s\n"
           "  low  (<  %u) -> %s\n",
           ts, threshold, threshold, high_path.c_str(), threshold, low_path.c_str());
    fflush(stdout);

    uint64_t n_high = 0, n_low = 0;
    while (!rr.exhausted) {
        if (rr.cur_count >= threshold) {
            wh.write_entry(rr.cur_key, rr.cur_count);
            n_high++;
        } else {
            wl.write_entry(rr.cur_key, rr.cur_count);
            n_low++;
        }
        rr.advance();
    }
    rr.close();

    if (!wh.close()) die("process split: write error on %s", high_path.c_str());
    if (!wl.close()) die("process split: write error on %s", low_path.c_str());

    char hc[32]; char lc[32];
    fmt_count(n_high, hc, sizeof(hc));
    fmt_count(n_low,  lc, sizeof(lc));
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] split: done  %s high  %s low\n", ts, hc, lc);
    fflush(stdout);
    return 0;
}

// ── Entry point ───────────────────────────────────────────────────────────────
int cmd_process(int argc, char **argv) {
    if (argc < 1 ||
        !strcmp(argv[0], "-h") || !strcmp(argv[0], "--help")) {
        usage_process();
        return argc < 1 ? 1 : 0;
    }
    if (!strcmp(argv[0], "filter-min"))  return process_filter_min(argc-1, argv+1);
    if (!strcmp(argv[0], "sort-count"))  return process_sort_count(argc-1, argv+1);
    if (!strcmp(argv[0], "to-text"))     return process_to_text(argc-1, argv+1);
    if (!strcmp(argv[0], "merge"))       return process_merge(argc-1, argv+1);
    if (!strcmp(argv[0], "split"))       return process_split(argc-1, argv+1);
    usage_process();
    fprintf(stderr, "process: unknown op '%s'\n", argv[0]);
    return 1;
}
