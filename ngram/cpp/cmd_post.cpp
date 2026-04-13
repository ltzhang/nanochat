// ngram process <op> [options]
//
// Post-processing operations on binary frequency tables from "ngram count".
//
//   filter-min  -i INPUT -o OUTPUT -t THRESHOLD
//       Keep only entries with count >= THRESHOLD.
//
//   count-range -i INPUT -o OUTPUT --min MIN [--max MAX]
//       Keep only entries with MIN <= count <= MAX.
//
//   sort-count  -i INPUT -o OUTPUT
//       Re-order entries by count descending.
//
//   to-text     -i INPUT -o OUTPUT [--vocab VOCAB_FILE]
//       Write human-readable text: one line per entry, tokens TAB count.
//       Without --vocab: space-separated token IDs.
//       With --vocab: detokenized text (tokens concatenated).

#include "common.h"
#include "run_file.h"

#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <cstring>
#include <cerrno>

static constexpr size_t LOG_EVERY = 1 << 20;   // progress every 1 M entries

// ── filter-min ────────────────────────────────────────────────────────────────
static int process_filter_min(int argc, char **argv) {
    std::string in_path, out_path;
    uint32_t threshold = 1;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-i") && i+1 < argc) in_path   = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path  = argv[++i];
        else if (!strcmp(argv[i], "-t") && i+1 < argc) threshold = (uint32_t)atoi(argv[++i]);
    }
    if (in_path.empty())  die("process filter-min: -i INPUT required");
    if (out_path.empty()) die("process filter-min: -o OUTPUT required");

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

// ── count-range ───────────────────────────────────────────────────────────────
static int process_count_range(int argc, char **argv) {
    std::string in_path, out_path;
    uint32_t min_count = 1, max_count = UINT32_MAX;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-i")    && i+1 < argc) in_path   = argv[++i];
        else if (!strcmp(argv[i], "-o")    && i+1 < argc) out_path  = argv[++i];
        else if (!strcmp(argv[i], "--min") && i+1 < argc) min_count = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max") && i+1 < argc) max_count = (uint32_t)atoi(argv[++i]);
    }
    if (in_path.empty())       die("process count-range: -i INPUT required");
    if (out_path.empty())      die("process count-range: -o OUTPUT required");
    if (min_count > max_count) die("process count-range: --min > --max");

    RunReader rr;
    if (!rr.open(in_path.c_str())) die("process count-range: cannot open %s", in_path.c_str());
    int n = rr.hdr.n;

    RunWriter w;
    if (!w.open(out_path.c_str(), n, rr.hdr.sort_by_key != 0))
        die("process count-range: cannot open %s: %s", out_path.c_str(), strerror(errno));

    uint64_t read = 0, written = 0;
    while (!rr.exhausted) {
        if (rr.cur_count >= min_count && rr.cur_count <= max_count) {
            w.write_entry(rr.cur_key, rr.cur_count);
            written++;
        }
        rr.advance();
        read++;
    }
    rr.close();
    if (!w.close()) die("process count-range: write error on %s", out_path.c_str());

    char ts[16]; char rc[32]; char wc[32];
    Progress::timestamp(ts, sizeof(ts));
    fmt_count(read,    rc, sizeof(rc));
    fmt_count(written, wc, sizeof(wc));
    printf("[%s] count-range: %s read  %s written  range=[%u,%u]\n",
           ts, rc, wc, min_count, max_count);
    fflush(stdout);
    return 0;
}

// ── sort-count ────────────────────────────────────────────────────────────────
static int process_sort_count(int argc, char **argv) {
    std::string in_path, out_path;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-i") && i+1 < argc) in_path  = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path = argv[++i];
    }
    if (in_path.empty())  die("process sort-count: -i INPUT required");
    if (out_path.empty()) die("process sort-count: -o OUTPUT required");

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

    // Build count histogram in one pass (entries are descending, so hist is too)
    {
        std::vector<std::pair<uint32_t, uint64_t>> hist;   // (count, num_ngrams)
        uint32_t cur_cnt = 0; uint64_t cur_run = 0;
        for (const auto &e : entries) {
            if (e.count != cur_cnt) {
                if (cur_run > 0) hist.emplace_back(cur_cnt, cur_run);
                cur_cnt = e.count; cur_run = 1;
            } else {
                cur_run++;
            }
        }
        if (cur_run > 0) hist.emplace_back(cur_cnt, cur_run);
        // prefix sums from front (highest count) → cumulative[i] = ngrams with count >= hist[i].first
        std::vector<uint64_t> cumulative(hist.size());
        uint64_t running = 0;
        for (size_t i = 0; i < hist.size(); i++) { running += hist[i].second; cumulative[i] = running; }
        // print ascending (reverse of descending hist)
        printf("count histogram (%zu distinct values):\n", hist.size());
        for (int i = (int)hist.size() - 1; i >= 0; i--) {
            char nc[32]; char cc[32];
            fmt_count(hist[i].second,  nc, sizeof(nc));
            fmt_count(cumulative[i],   cc, sizeof(cc));
            printf("  count %10u : %s n-grams  (>= %u: %s)\n",
                   hist[i].first, nc, hist[i].first, cc);
        }
        fflush(stdout);
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
        if      (!strcmp(argv[i], "-i")      && i+1 < argc) in_path    = argv[++i];
        else if (!strcmp(argv[i], "-o")      && i+1 < argc) out_path   = argv[++i];
        else if (!strcmp(argv[i], "--vocab") && i+1 < argc) vocab_path = argv[++i];
    }
    if (in_path.empty())  die("process to-text: -i INPUT required");
    if (out_path.empty()) die("process to-text: -o OUTPUT required");

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
// Usage: ngram process merge -i FILE [-i FILE ...] -o OUTPUT [--min-count M]
static int process_merge(int argc, char **argv) {
    std::vector<std::string> in_paths;
    std::string out_path;
    uint32_t min_count = 1;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-i")          && i+1 < argc) in_paths.push_back(argv[++i]);
        else if (!strcmp(argv[i], "-o")          && i+1 < argc) out_path  = argv[++i];
        else if (!strcmp(argv[i], "--min-count") && i+1 < argc) min_count = (uint32_t)atoi(argv[++i]);
    }
    if (in_paths.empty()) die("process merge: at least one -i INPUT required");
    if (out_path.empty()) die("process merge: -o OUTPUT required");

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

// ── Entry point ───────────────────────────────────────────────────────────────
int cmd_process(int argc, char **argv) {
    if (argc < 1) {
        fputs("usage: ngram process <op> [options]\n"
              "  filter-min  -i INPUT -o OUTPUT -t THRESHOLD\n"
              "  count-range -i INPUT -o OUTPUT --min MIN [--max MAX]\n"
              "  sort-count  -i INPUT -o OUTPUT\n"
              "  to-text     -i INPUT -o OUTPUT [--vocab VOCAB_FILE]\n"
              "  merge       -i INPUT [-i INPUT ...] -o OUTPUT [--min-count M]\n",
              stderr);
        return 1;
    }
    if (!strcmp(argv[0], "filter-min"))  return process_filter_min(argc-1, argv+1);
    if (!strcmp(argv[0], "count-range")) return process_count_range(argc-1, argv+1);
    if (!strcmp(argv[0], "sort-count"))  return process_sort_count(argc-1, argv+1);
    if (!strcmp(argv[0], "to-text"))     return process_to_text(argc-1, argv+1);
    if (!strcmp(argv[0], "merge"))       return process_merge(argc-1, argv+1);
    fprintf(stderr, "process: unknown op '%s'\n", argv[0]);
    return 1;
}
