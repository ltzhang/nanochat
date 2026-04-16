#include "common.h"
#include "run_file.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <limits.h>
#include <map>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <queue>
#include <string>
#include <vector>

namespace {

enum class BuildMode {
    PerNTopK,
    GlobalTopK,
    PerNPercent,
};

struct Candidate {
    uint8_t  key[MAX_KEY_BYTES]{};
    uint32_t count = 0;
    uint16_t n = 0;
};

struct PerNFile {
    int         n = 0;
    std::string path;
    uint64_t    entry_count = 0;
};

static void usage_build_vocab() {
    fputs(
        "usage: ngram build_vocab -i INPUT_DIR -o OUTPUT.tsv --mode MODE [options]\n"
        "\n"
        "Build a final UTF-8 n-gram vocabulary TSV from merged count tables.\n"
        "The TSV includes a header and columns:\n"
        "  ngram_id<TAB>n<TAB>token_ids<TAB>count<TAB>text\n"
        "\n"
        "Required:\n"
        "  -i DIR            Directory containing {n}gram.count.bin files\n"
        "  -o FILE           Output TSV path\n"
        "  --mode MODE       One of: per_n_topk, global_topk, per_n_percent\n"
        "\n"
        "Selection options:\n"
        "  --topn SPEC       For per_n_topk, e.g. 2:2000,3:3000,4:5000\n"
        "  --budget K        For global_topk, keep the best K entries overall\n"
        "  --percent SPEC    For per_n_percent, e.g. 2:5,3:2.5,4:1\n"
        "\n"
        "Optional:\n"
        "  --vocab FILE      Token vocabulary TSV for detokenized text output\n"
        "                    (auto-generated from NanoChat tokenizer if omitted)\n"
        "  --n-min N         Minimum order to include (default: 2)\n"
        "  --n-max N         Maximum order to include (default: 8)\n",
        stderr);
}

static std::string self_exe() {
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len < 0) die("build_vocab: readlink /proc/self/exe: %s", strerror(errno));
    buf[len] = '\0';
    return buf;
}

static std::string auto_find_py_script() {
    std::string exe = self_exe();
    size_t slash = exe.rfind('/');
    if (slash == std::string::npos) return "";
    std::string bin_dir = exe.substr(0, slash);
    size_t slash2 = bin_dir.rfind('/');
    std::string root = (slash2 == std::string::npos) ? bin_dir : bin_dir.substr(0, slash2);
    struct stat st{};
    const std::vector<std::string> candidates = {
        root + "/parquet_to_bin.py",
        root + "/ngram/parquet_to_bin.py",
    };
    for (const auto &candidate : candidates) {
        if (stat(candidate.c_str(), &st) == 0) return candidate;
    }
    return "";
}

static std::string trim_ascii(const std::string &s) {
    size_t lo = 0, hi = s.size();
    while (lo < hi && std::isspace(static_cast<unsigned char>(s[lo]))) lo++;
    while (hi > lo && std::isspace(static_cast<unsigned char>(s[hi - 1]))) hi--;
    return s.substr(lo, hi - lo);
}

static bool parse_uint64_str(const std::string &s, uint64_t *out) {
    if (s.empty()) return false;
    char *end = nullptr;
    errno = 0;
    unsigned long long v = strtoull(s.c_str(), &end, 10);
    if (errno != 0 || end != s.c_str() + s.size()) return false;
    *out = static_cast<uint64_t>(v);
    return true;
}

static bool parse_double_str(const std::string &s, double *out) {
    if (s.empty()) return false;
    char *end = nullptr;
    errno = 0;
    double v = strtod(s.c_str(), &end);
    if (errno != 0 || end != s.c_str() + s.size()) return false;
    *out = v;
    return true;
}

static std::map<int, uint64_t> parse_topn_spec(const std::string &spec) {
    std::map<int, uint64_t> out;
    size_t start = 0;
    while (start < spec.size()) {
        size_t comma = spec.find(',', start);
        std::string item = trim_ascii(spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (!item.empty()) {
            size_t colon = item.find(':');
            if (colon == std::string::npos) die("build_vocab: invalid --topn item '%s'", item.c_str());
            std::string n_str = trim_ascii(item.substr(0, colon));
            std::string k_str = trim_ascii(item.substr(colon + 1));
            uint64_t n_val = 0, k_val = 0;
            if (!parse_uint64_str(n_str, &n_val) || !parse_uint64_str(k_str, &k_val))
                die("build_vocab: invalid --topn item '%s'", item.c_str());
            if (n_val < 1 || n_val > MAX_N) die("build_vocab: n out of range in --topn: %s", item.c_str());
            out[(int)n_val] = k_val;
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

static std::map<int, double> parse_percent_spec(const std::string &spec) {
    std::map<int, double> out;
    size_t start = 0;
    while (start < spec.size()) {
        size_t comma = spec.find(',', start);
        std::string item = trim_ascii(spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (!item.empty()) {
            size_t colon = item.find(':');
            if (colon == std::string::npos) die("build_vocab: invalid --percent item '%s'", item.c_str());
            std::string n_str = trim_ascii(item.substr(0, colon));
            std::string p_str = trim_ascii(item.substr(colon + 1));
            uint64_t n_val = 0;
            double p_val = 0.0;
            if (!parse_uint64_str(n_str, &n_val) || !parse_double_str(p_str, &p_val))
                die("build_vocab: invalid --percent item '%s'", item.c_str());
            if (n_val < 1 || n_val > MAX_N) die("build_vocab: n out of range in --percent: %s", item.c_str());
            if (!(p_val >= 0.0 && p_val <= 100.0))
                die("build_vocab: percent must be in [0,100], got %s", item.c_str());
            out[(int)n_val] = p_val;
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

static std::string escape_text_field(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '\t') out += "\\t";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c < 0x20 || c == 0x7f) {
            char buf[5];
            snprintf(buf, sizeof(buf), "\\x%02X", (unsigned)c);
            out += buf;
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static std::string unescape_text_field(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        char c = s[++i];
        if (c == '\\') out.push_back('\\');
        else if (c == 't') out.push_back('\t');
        else if (c == 'n') out.push_back('\n');
        else if (c == 'r') out.push_back('\r');
        else if (c == 'x' && i + 2 < s.size()) {
            int hi = hex_value(s[i + 1]);
            int lo = hex_value(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back((char)((hi << 4) | lo));
                i += 2;
            } else {
                out.push_back('\\');
                out.push_back(c);
            }
        } else {
            out.push_back('\\');
            out.push_back(c);
        }
    }
    return out;
}

static void run_command_sync(const std::vector<std::string> &argv, const std::string &label) {
    std::vector<char *> args;
    args.reserve(argv.size() + 1);
    for (const auto &s : argv) args.push_back(const_cast<char *>(s.c_str()));
    args.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) die("build_vocab: fork failed for %s: %s", label.c_str(), strerror(errno));
    if (pid == 0) {
        execvp(args[0], args.data());
        fprintf(stderr, "execvp(%s) failed: %s\n", args[0], strerror(errno));
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
        die("build_vocab: waitpid failed for %s: %s", label.c_str(), strerror(errno));
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        die("build_vocab: command failed for %s", label.c_str());
}

static std::string generate_default_vocab_dump() {
    std::string py_script = auto_find_py_script();
    if (py_script.empty()) return "";
    size_t slash = py_script.rfind('/');
    if (slash == std::string::npos) return "";
    std::string ngram_dir = py_script.substr(0, slash);
    size_t slash2 = ngram_dir.rfind('/');
    if (slash2 == std::string::npos) return "";
    std::string repo_root = ngram_dir.substr(0, slash2);
    std::string out_path = "/tmp/ngram_build_vocab_tokdump_" + std::to_string(getpid()) + ".tsv";

    const char *script =
        "import sys\n"
        "repo_root, out_path = sys.argv[1], sys.argv[2]\n"
        "sys.path.insert(0, repo_root)\n"
        "from nanochat.tokenizer import get_tokenizer\n"
        "tok = get_tokenizer()\n"
        "def esc(s):\n"
        "    out = []\n"
        "    for ch in s:\n"
        "        o = ord(ch)\n"
        "        if ch == '\\\\': out.append('\\\\\\\\')\n"
        "        elif ch == '\\t': out.append('\\\\t')\n"
        "        elif ch == '\\n': out.append('\\\\n')\n"
        "        elif ch == '\\r': out.append('\\\\r')\n"
        "        elif o < 32 or o == 127: out.append(f'\\\\x{o:02X}')\n"
        "        else: out.append(ch)\n"
        "    return ''.join(out)\n"
        "with open(out_path, 'w', encoding='utf-8') as f:\n"
        "    for i in range(tok.get_vocab_size()):\n"
        "        f.write(f\"{i}\\t-\\t{esc(tok.decode([i]))}\\n\")\n";

    run_command_sync({"python3", "-c", script, repo_root, out_path}, "tokenizer vocab dump");
    return out_path;
}

static std::vector<std::string> load_vocab(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) die("build_vocab: cannot open vocab %s: %s", path, strerror(errno));

    std::vector<std::string> vocab;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        char *p = line;
        char *tab1 = strchr(p, '\t');
        if (!tab1) continue;
        *tab1 = '\0';
        uint32_t tid = (uint32_t)atoi(p);
        p = tab1 + 1;
        char *tab2 = strchr(p, '\t');
        if (!tab2) continue;
        p = tab2 + 1;
        size_t len = strlen(p);
        while (len > 0 && (p[len - 1] == '\n' || p[len - 1] == '\r')) p[--len] = '\0';

        if (tid >= vocab.size()) vocab.resize(tid + 1);
        vocab[tid] = unescape_text_field(p);
    }
    fclose(f);
    return vocab;
}

static std::vector<PerNFile> discover_input_files(const std::string &input_dir, int n_min, int n_max) {
    std::vector<PerNFile> files;
    for (int n = n_min; n <= n_max; n++) {
        std::string path = input_dir;
        if (!path.empty() && path.back() != '/') path.push_back('/');
        path += std::to_string(n) + "gram.count.bin";

        RunReader rr;
        if (!rr.open(path.c_str())) continue;
        if (rr.hdr.n != static_cast<uint16_t>(n)) {
            rr.close();
            die("build_vocab: header n mismatch in %s: expected %d, got %u",
                path.c_str(), n, (unsigned)rr.hdr.n);
        }
        files.push_back(PerNFile{n, path, rr.hdr.entry_count});
        rr.close();
    }
    if (files.empty()) {
        die("build_vocab: no {n}gram.count.bin files found in %s for n=%d..%d",
            input_dir.c_str(), n_min, n_max);
    }
    return files;
}

static int compare_candidate_global(const Candidate &a, const Candidate &b) {
    if (a.count != b.count) return (a.count > b.count) ? -1 : 1;
    if (a.n != b.n) return (a.n < b.n) ? -1 : 1;
    int cmp = memcmp(a.key, b.key, key_bytes(a.n));
    if (cmp < 0) return -1;
    if (cmp > 0) return 1;
    return 0;
}

static bool better_global(const Candidate &a, const Candidate &b) {
    return compare_candidate_global(a, b) < 0;
}

static bool better_same_n(const Candidate &a, const Candidate &b) {
    if (a.count != b.count) return a.count > b.count;
    return memcmp(a.key, b.key, key_bytes(a.n)) < 0;
}

struct WorseFirst {
    bool operator()(const Candidate &a, const Candidate &b) const {
        return better_global(a, b);
    }
};

static std::vector<Candidate> read_top_k_for_file(const PerNFile &file, uint64_t keep_k) {
    std::vector<Candidate> out;
    if (keep_k == 0) return out;

    RunReader rr;
    if (!rr.open(file.path.c_str())) die("build_vocab: cannot open %s", file.path.c_str());
    if (rr.hdr.n != static_cast<uint16_t>(file.n))
        die("build_vocab: n mismatch while reading %s", file.path.c_str());

    std::priority_queue<Candidate, std::vector<Candidate>, WorseFirst> heap;
    while (!rr.exhausted) {
        Candidate c;
        memcpy(c.key, rr.cur_key, key_bytes(file.n));
        c.count = rr.cur_count;
        c.n = (uint16_t)file.n;

        if (heap.size() < keep_k) {
            heap.push(c);
        } else if (better_same_n(c, heap.top())) {
            heap.pop();
            heap.push(c);
        }
        rr.advance();
    }
    rr.close();

    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    std::sort(out.begin(), out.end(), [](const Candidate &a, const Candidate &b) {
        return better_same_n(a, b);
    });
    return out;
}

static std::vector<Candidate> select_global_top_k(const std::vector<PerNFile> &files, uint64_t budget) {
    std::vector<Candidate> out;
    if (budget == 0) return out;

    std::priority_queue<Candidate, std::vector<Candidate>, WorseFirst> heap;
    for (const auto &file : files) {
        RunReader rr;
        if (!rr.open(file.path.c_str())) die("build_vocab: cannot open %s", file.path.c_str());
        while (!rr.exhausted) {
            Candidate c;
            memcpy(c.key, rr.cur_key, key_bytes(file.n));
            c.count = rr.cur_count;
            c.n = (uint16_t)file.n;
            if (heap.size() < budget) {
                heap.push(c);
            } else if (better_global(c, heap.top())) {
                heap.pop();
                heap.push(c);
            }
            rr.advance();
        }
        rr.close();
    }

    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    std::sort(out.begin(), out.end(), [](const Candidate &a, const Candidate &b) {
        if (a.n != b.n) return a.n < b.n;
        return better_same_n(a, b);
    });
    return out;
}

static std::string token_ids_to_string(const Candidate &c) {
    std::string out;
    char buf[32];
    for (int i = 0; i < c.n; i++) {
        Token tok;
        memcpy(&tok, c.key + i * TOKEN_BYTES, TOKEN_BYTES);
        if (i) out.push_back(' ');
        snprintf(buf, sizeof(buf), "%u", (unsigned)tok);
        out += buf;
    }
    return out;
}

static std::string detokenize_candidate(const Candidate &c, const std::vector<std::string> &vocab) {
    if (vocab.empty()) return "";
    std::string out;
    for (int i = 0; i < c.n; i++) {
        Token tok;
        memcpy(&tok, c.key + i * TOKEN_BYTES, TOKEN_BYTES);
        if (tok < (Token)vocab.size()) out += vocab[tok];
        else out.push_back('?');
    }
    return escape_text_field(out);
}

}  // namespace

int cmd_build_vocab(int argc, char **argv) {
    if (argc == 0) { usage_build_vocab(); return 1; }

    std::string input_dir, out_path, vocab_path, mode_str, topn_spec, percent_spec;
    int n_min = 2;
    int n_max = (int)MAX_N;
    uint64_t budget = 0;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_build_vocab(); return 0; }
        else if (!strcmp(argv[i], "-i")         && i + 1 < argc) input_dir = argv[++i];
        else if (!strcmp(argv[i], "-o")         && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--vocab")    && i + 1 < argc) vocab_path = argv[++i];
        else if (!strcmp(argv[i], "--mode")     && i + 1 < argc) mode_str = argv[++i];
        else if (!strcmp(argv[i], "--topn")     && i + 1 < argc) topn_spec = argv[++i];
        else if (!strcmp(argv[i], "--budget")   && i + 1 < argc) {
            if (!parse_uint64_str(argv[++i], &budget)) die("build_vocab: invalid --budget");
        }
        else if (!strcmp(argv[i], "--percent")  && i + 1 < argc) percent_spec = argv[++i];
        else if (!strcmp(argv[i], "--n-min")    && i + 1 < argc) n_min = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-max")    && i + 1 < argc) n_max = atoi(argv[++i]);
        else die("build_vocab: unknown argument '%s'", argv[i]);
    }

    if (input_dir.empty()) { usage_build_vocab(); die("build_vocab: -i DIR required"); }
    if (out_path.empty())  { usage_build_vocab(); die("build_vocab: -o FILE required"); }
    if (n_min < 1 || n_min > (int)MAX_N || n_max < 1 || n_max > (int)MAX_N || n_min > n_max)
        die("build_vocab: invalid n range %d..%d", n_min, n_max);

    BuildMode mode;
    if (mode_str == "per_n_topk") mode = BuildMode::PerNTopK;
    else if (mode_str == "global_topk") mode = BuildMode::GlobalTopK;
    else if (mode_str == "per_n_percent") mode = BuildMode::PerNPercent;
    else {
        usage_build_vocab();
        die("build_vocab: --mode must be one of per_n_topk, global_topk, per_n_percent");
    }

    std::map<int, uint64_t> topn_by_n;
    std::map<int, double> percent_by_n;
    if (mode == BuildMode::PerNTopK) {
        if (topn_spec.empty()) die("build_vocab: --topn required for --mode per_n_topk");
        topn_by_n = parse_topn_spec(topn_spec);
    } else if (mode == BuildMode::GlobalTopK) {
        if (budget == 0) die("build_vocab: --budget must be > 0 for --mode global_topk");
    } else if (mode == BuildMode::PerNPercent) {
        if (percent_spec.empty()) die("build_vocab: --percent required for --mode per_n_percent");
        percent_by_n = parse_percent_spec(percent_spec);
    }

    std::vector<PerNFile> files = discover_input_files(input_dir, n_min, n_max);
    std::vector<std::string> vocab;
    std::string temp_vocab_path;
    if (vocab_path.empty()) {
        temp_vocab_path = generate_default_vocab_dump();
        if (!temp_vocab_path.empty()) vocab_path = temp_vocab_path;
    }
    if (!vocab_path.empty()) vocab = load_vocab(vocab_path.c_str());

    char ts[16];
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] build_vocab: input=%s  files=%zu  mode=%s  out=%s\n",
           ts, input_dir.c_str(), files.size(), mode_str.c_str(), out_path.c_str());
    fflush(stdout);

    std::vector<Candidate> selected;
    if (mode == BuildMode::PerNTopK) {
        for (const auto &file : files) {
            auto it = topn_by_n.find(file.n);
            if (it == topn_by_n.end())
                die("build_vocab: missing --topn entry for %d-gram", file.n);
            std::vector<Candidate> cur = read_top_k_for_file(file, it->second);
            selected.insert(selected.end(), cur.begin(), cur.end());
        }
        std::sort(selected.begin(), selected.end(), [](const Candidate &a, const Candidate &b) {
            if (a.n != b.n) return a.n < b.n;
            return better_same_n(a, b);
        });
    } else if (mode == BuildMode::GlobalTopK) {
        selected = select_global_top_k(files, budget);
    } else {
        for (const auto &file : files) {
            auto it = percent_by_n.find(file.n);
            if (it == percent_by_n.end())
                die("build_vocab: missing --percent entry for %d-gram", file.n);
            uint64_t keep_k = 0;
            if (it->second > 0.0 && file.entry_count > 0) {
                keep_k = static_cast<uint64_t>(std::ceil((it->second / 100.0) * (double)file.entry_count));
                if (keep_k == 0) keep_k = 1;
            }
            std::vector<Candidate> cur = read_top_k_for_file(file, keep_k);
            selected.insert(selected.end(), cur.begin(), cur.end());
        }
        std::sort(selected.begin(), selected.end(), [](const Candidate &a, const Candidate &b) {
            if (a.n != b.n) return a.n < b.n;
            return better_same_n(a, b);
        });
    }

    FILE *out = fopen(out_path.c_str(), "w");
    if (!out) die("build_vocab: cannot open %s: %s", out_path.c_str(), strerror(errno));
    fprintf(out, "ngram_id\tn\ttoken_ids\tcount\ttext\n");

    for (size_t i = 0; i < selected.size(); i++) {
        const Candidate &c = selected[i];
        std::string token_ids = token_ids_to_string(c);
        std::string text = detokenize_candidate(c, vocab);
        fprintf(out, "%zu\t%u\t%s\t%u\t%s\n",
                i + 1, (unsigned)c.n, token_ids.c_str(), c.count, text.c_str());
    }
    fclose(out);

    std::map<int, uint64_t> counts_by_n;
    for (const auto &c : selected) counts_by_n[c.n]++;

    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] build_vocab: wrote %zu entries to %s\n", ts, selected.size(), out_path.c_str());
    for (const auto &kv : counts_by_n) {
        char cc[32];
        fmt_count(kv.second, cc, sizeof(cc));
        printf("[%s]   %d-gram: %s kept\n", ts, kv.first, cc);
    }
    if (!temp_vocab_path.empty()) remove(temp_vocab_path.c_str());
    fflush(stdout);
    return 0;
}
