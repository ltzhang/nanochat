#include "common.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <limits.h>

namespace {

struct TrieNode {
    uint32_t ngram_id = 0;
    std::unordered_map<Token, int> children;
};

struct SourceFile {
    std::string bin_path;
    std::string parquet_path;
};

static void usage_ngram_input_gen() {
    fputs(
        "usage: ngram ngram_input_gen --lexicon VOCAB.tsv [INPUT_PATH] [options]\n"
        "\n"
        "Generate aligned .ngram.bin sidecars from a built n-gram vocabulary.\n"
        "INPUT_PATH may be a .bin file, a .parquet file, or a directory.\n"
        "If omitted, defaults to NanoChat's pretrain data dir.\n"
        "\n"
        "Required:\n"
        "  --lexicon FILE    TSV produced by 'ngram build_vocab'\n"
        "\n"
        "Optional:\n"
        "  --py-script PATH  Path to parquet_to_bin.py (auto-detected if omitted)\n"
        "  --python BIN      Python executable for parquet tokenization (default: python3)\n"
        "  --text-column C   Passed to parquet_to_bin.py (default: text)\n"
        "  --threads T       Passed to parquet_to_bin.py tokenizer threads\n"
        "  --batch-size N    Passed to parquet_to_bin.py batch size\n"
        "  --tokenizer-dir D Passed to parquet_to_bin.py tokenizer override\n"
        "  --skip-existing-ngram\n"
        "                    Do not rewrite an existing .ngram.bin sidecar\n",
        stderr);
}

static bool path_exists(const std::string &path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0;
}

static bool is_dir(const std::string &path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool is_regular(const std::string &path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool has_suffix(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string strip_extension(const std::string &path) {
    size_t dot = path.rfind('.');
    if (dot == std::string::npos) return path;
    return path.substr(0, dot);
}

static std::string sidecar_path_for_bin(const std::string &bin_path) {
    if (has_suffix(bin_path, ".bin")) return bin_path.substr(0, bin_path.size() - 4) + ".ngram.bin";
    return bin_path + ".ngram.bin";
}

static std::string trim_ascii(const std::string &s) {
    size_t lo = 0, hi = s.size();
    while (lo < hi && std::isspace(static_cast<unsigned char>(s[lo]))) lo++;
    while (hi > lo && std::isspace(static_cast<unsigned char>(s[hi - 1]))) hi--;
    return s.substr(lo, hi - lo);
}

static std::string default_data_dir() {
    const char *base_env = getenv("NANOCHAT_BASE_DIR");
    if (base_env && base_env[0]) return std::string(base_env) + "/base_data_climbmix";
    const char *home = getenv("HOME");
    if (!home || !home[0]) die("ngram_input_gen: HOME is not set and NANOCHAT_BASE_DIR is not set");
    return std::string(home) + "/.cache/nanochat/base_data_climbmix";
}

static std::string self_exe() {
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len < 0) die("ngram_input_gen: readlink /proc/self/exe: %s", strerror(errno));
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
    const std::vector<std::string> candidates = {
        root + "/parquet_to_bin.py",
        root + "/ngram/parquet_to_bin.py",
    };
    for (const auto &candidate : candidates) {
        if (is_regular(candidate)) return candidate;
    }
    return "";
}

static void append_arg(std::vector<std::string> &argv, const char *flag, const std::string &value) {
    argv.push_back(flag);
    argv.push_back(value);
}

static void run_command_sync(const std::vector<std::string> &argv, const std::string &label) {
    std::vector<char *> args;
    args.reserve(argv.size() + 1);
    for (const auto &s : argv) args.push_back(const_cast<char *>(s.c_str()));
    args.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) die("ngram_input_gen: fork failed for %s: %s", label.c_str(), strerror(errno));
    if (pid == 0) {
        execvp(args[0], args.data());
        fprintf(stderr, "execvp(%s) failed: %s\n", args[0], strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
        die("ngram_input_gen: waitpid failed for %s: %s", label.c_str(), strerror(errno));
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        die("ngram_input_gen: command failed for %s", label.c_str());
}

static std::string resolve_input_path(const std::string &input_path) {
    if (input_path.empty()) return default_data_dir();
    if (input_path[0] == '/' || path_exists(input_path)) return input_path;
    std::string candidate = default_data_dir() + "/" + input_path;
    if (path_exists(candidate)) return candidate;
    return input_path;
}

static std::vector<SourceFile> collect_sources_from_dir(const std::string &dir_path) {
    std::map<std::string, SourceFile> by_bin;
    DIR *dir = opendir(dir_path.c_str());
    if (!dir) die("ngram_input_gen: cannot open directory %s: %s", dir_path.c_str(), strerror(errno));
    for (;;) {
        dirent *ent = readdir(dir);
        if (!ent) break;
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        std::string full = dir_path + "/" + name;
        if (!is_regular(full)) continue;
        if (has_suffix(name, ".parquet")) {
            std::string bin_path = strip_extension(full) + ".bin";
            auto &src = by_bin[bin_path];
            src.bin_path = bin_path;
            src.parquet_path = full;
        } else if (has_suffix(name, ".bin") && !has_suffix(name, ".ngram.bin")) {
            auto &src = by_bin[full];
            src.bin_path = full;
        }
    }
    closedir(dir);

    std::vector<SourceFile> out;
    for (auto &kv : by_bin) out.push_back(kv.second);
    return out;
}

static std::vector<SourceFile> resolve_sources(const std::string &input_arg) {
    std::string input_path = resolve_input_path(input_arg);
    if (is_dir(input_path)) {
        std::vector<SourceFile> out = collect_sources_from_dir(input_path);
        if (out.empty()) die("ngram_input_gen: no .parquet or token .bin files found in %s", input_path.c_str());
        return out;
    }
    if (has_suffix(input_path, ".parquet")) {
        if (!is_regular(input_path)) die("ngram_input_gen: parquet file not found: %s", input_path.c_str());
        return {SourceFile{strip_extension(input_path) + ".bin", input_path}};
    }
    if (has_suffix(input_path, ".bin") && !has_suffix(input_path, ".ngram.bin")) {
        if (!is_regular(input_path)) die("ngram_input_gen: token bin not found: %s", input_path.c_str());
        return {SourceFile{input_path, ""}};
    }
    die("ngram_input_gen: unsupported input path: %s", input_path.c_str());
}

static std::vector<TrieNode> load_lexicon(const std::string &lexicon_path, int *max_order_out, uint32_t *max_id_out) {
    FILE *f = fopen(lexicon_path.c_str(), "r");
    if (!f) die("ngram_input_gen: cannot open lexicon %s: %s", lexicon_path.c_str(), strerror(errno));

    std::vector<TrieNode> nodes(1);
    int max_order = 0;
    uint32_t max_id = 0;
    char line[1 << 16];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        std::string raw = trim_ascii(line);
        if (raw.empty() || raw[0] == '#') continue;

        std::vector<std::string> fields;
        size_t start = 0;
        for (;;) {
            size_t tab = raw.find('\t', start);
            fields.push_back(raw.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
            if (tab == std::string::npos) break;
            start = tab + 1;
        }
        if (fields.empty()) continue;

        bool numeric_id = !fields[0].empty();
        for (char c : fields[0]) {
            if (!std::isdigit(static_cast<unsigned char>(c))) { numeric_id = false; break; }
        }
        if (!numeric_id) continue;

        uint32_t ngram_id = (uint32_t)strtoul(fields[0].c_str(), nullptr, 10);
        if (ngram_id == 0) die("ngram_input_gen: invalid ngram_id=0 at line %d", lineno);

        std::string token_field;
        int n_expected = -1;
        if (fields.size() == 2) {
            token_field = fields[1];
        } else if (fields.size() >= 3) {
            n_expected = atoi(fields[1].c_str());
            token_field = fields[2];
        } else {
            die("ngram_input_gen: malformed line %d in %s", lineno, lexicon_path.c_str());
        }

        std::vector<Token> toks;
        size_t tok_start = 0;
        while (tok_start < token_field.size()) {
            while (tok_start < token_field.size() && token_field[tok_start] == ' ') tok_start++;
            if (tok_start >= token_field.size()) break;
            size_t tok_end = token_field.find(' ', tok_start);
            std::string tok_str = token_field.substr(tok_start, tok_end == std::string::npos ? std::string::npos : tok_end - tok_start);
            toks.push_back((Token)strtoul(tok_str.c_str(), nullptr, 10));
            if (tok_end == std::string::npos) break;
            tok_start = tok_end + 1;
        }
        if (toks.empty()) die("ngram_input_gen: empty token sequence at line %d", lineno);
        if (n_expected >= 0 && (int)toks.size() != n_expected)
            die("ngram_input_gen: n mismatch at line %d: declared %d got %zu", lineno, n_expected, toks.size());
        if (toks.size() > MAX_N)
            die("ngram_input_gen: n-gram length %zu exceeds MAX_N=%zu at line %d", toks.size(), MAX_N, lineno);

        int node_idx = 0;
        for (size_t i = toks.size(); i > 0; i--) {
            Token tok = toks[i - 1];
            auto it = nodes[node_idx].children.find(tok);
            if (it == nodes[node_idx].children.end()) {
                int next_idx = (int)nodes.size();
                nodes[node_idx].children[tok] = next_idx;
                nodes.push_back(TrieNode{});
                node_idx = next_idx;
            } else {
                node_idx = it->second;
            }
        }
        if (nodes[node_idx].ngram_id != 0 && nodes[node_idx].ngram_id != ngram_id)
            die("ngram_input_gen: duplicate token sequence with conflicting IDs at line %d", lineno);
        nodes[node_idx].ngram_id = ngram_id;
        if ((int)toks.size() > max_order) max_order = (int)toks.size();
        if (ngram_id > max_id) max_id = ngram_id;
    }
    fclose(f);
    if (max_order == 0) die("ngram_input_gen: lexicon is empty: %s", lexicon_path.c_str());
    *max_order_out = max_order;
    *max_id_out = max_id;
    return nodes;
}

static void ensure_token_bin(const SourceFile &src,
                             const std::string &py_script,
                             const std::string &python_bin,
                             const std::string &text_column,
                             int threads,
                             int batch_size,
                             const std::string &tokenizer_dir) {
    if (path_exists(src.bin_path)) return;
    if (src.parquet_path.empty())
        die("ngram_input_gen: missing token bin %s and no source parquet is available", src.bin_path.c_str());
    if (py_script.empty())
        die("ngram_input_gen: token bin %s is missing and parquet_to_bin.py was not found", src.bin_path.c_str());

    char ts[16];
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] tokenizing %s -> %s\n", ts, src.parquet_path.c_str(), src.bin_path.c_str());
    fflush(stdout);

    std::vector<std::string> argv = {python_bin, py_script, src.parquet_path, "--out", src.bin_path};
    append_arg(argv, "--text-column", text_column);
    append_arg(argv, "--threads", std::to_string(threads));
    append_arg(argv, "--batch-size", std::to_string(batch_size));
    if (!tokenizer_dir.empty()) append_arg(argv, "--tokenizer-dir", tokenizer_dir);
    run_command_sync(argv, src.parquet_path);
}

static void generate_sidecar_for_bin(const std::string &bin_path,
                                     const std::string &ngram_path,
                                     const std::vector<TrieNode> &trie,
                                     int max_order) {
    FILE *in = fopen(bin_path.c_str(), "rb");
    if (!in) die("ngram_input_gen: cannot open token bin %s: %s", bin_path.c_str(), strerror(errno));
    FILE *out = fopen(ngram_path.c_str(), "wb");
    if (!out) {
        fclose(in);
        die("ngram_input_gen: cannot open output %s: %s", ngram_path.c_str(), strerror(errno));
    }

    const size_t CHUNK_TOKENS = 1 << 20;
    std::vector<Token> in_buf(CHUNK_TOKENS);
    std::vector<uint32_t> out_buf(CHUNK_TOKENS);
    std::vector<Token> history;
    history.reserve((size_t)max_order);
    uint64_t total_tokens = 0;
    Progress prog;

    while (true) {
        size_t n_read = fread(in_buf.data(), sizeof(Token), CHUNK_TOKENS, in);
        if (n_read == 0) break;
        for (size_t i = 0; i < n_read; i++) {
            Token tok = in_buf[i];
            if (history.size() == (size_t)max_order) history.erase(history.begin());
            history.push_back(tok);

            int node_idx = 0;
            uint32_t best = 0;
            for (size_t h = history.size(); h > 0; h--) {
                Token cur = history[h - 1];
                auto it = trie[node_idx].children.find(cur);
                if (it == trie[node_idx].children.end()) break;
                node_idx = it->second;
                if (trie[node_idx].ngram_id != 0) best = trie[node_idx].ngram_id;
            }
            out_buf[i] = best;
        }
        if (fwrite(out_buf.data(), sizeof(uint32_t), n_read, out) != n_read) {
            fclose(in);
            fclose(out);
            die("ngram_input_gen: write failed for %s", ngram_path.c_str());
        }
        total_tokens += n_read;
        if (prog.tick()) {
            char ts[16]; char tc[32];
            Progress::timestamp(ts, sizeof(ts));
            fmt_count(total_tokens, tc, sizeof(tc));
            printf("[%s] ngram_input_gen: %s tokens processed for %s\n", ts, tc, bin_path.c_str());
            fflush(stdout);
        }
        if (n_read < CHUNK_TOKENS) break;
    }

    if (ferror(in)) {
        fclose(in);
        fclose(out);
        die("ngram_input_gen: read failed for %s", bin_path.c_str());
    }
    fclose(in);
    fclose(out);
}

}  // namespace

int cmd_ngram_input_gen(int argc, char **argv) {
    if (argc == 0) {
        usage_ngram_input_gen();
        return 1;
    }

    std::string lexicon_path;
    std::string input_path;
    std::string py_script;
    std::string python_bin = "python3";
    std::string text_column = "text";
    std::string tokenizer_dir;
    int threads = std::max(8, (int)sysconf(_SC_NPROCESSORS_ONLN));
    int batch_size = 256;
    bool skip_existing_ngram = false;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_ngram_input_gen(); return 0; }
        else if (!strcmp(argv[i], "--lexicon") && i + 1 < argc) lexicon_path = argv[++i];
        else if (!strcmp(argv[i], "--py-script") && i + 1 < argc) py_script = argv[++i];
        else if (!strcmp(argv[i], "--python") && i + 1 < argc) python_bin = argv[++i];
        else if (!strcmp(argv[i], "--text-column") && i + 1 < argc) text_column = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-size") && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokenizer-dir") && i + 1 < argc) tokenizer_dir = argv[++i];
        else if (!strcmp(argv[i], "--skip-existing-ngram")) skip_existing_ngram = true;
        else if (argv[i][0] != '-') input_path = argv[i];
        else die("ngram_input_gen: unknown argument '%s'", argv[i]);
    }

    if (lexicon_path.empty()) {
        usage_ngram_input_gen();
        die("ngram_input_gen: --lexicon FILE required");
    }
    if (threads < 1) threads = 1;
    if (batch_size < 1) batch_size = 1;
    if (py_script.empty()) py_script = auto_find_py_script();

    int max_order = 0;
    uint32_t max_id = 0;
    std::vector<TrieNode> trie = load_lexicon(lexicon_path, &max_order, &max_id);
    std::vector<SourceFile> sources = resolve_sources(input_path);

    char ts[16];
    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] ngram_input_gen: lexicon=%s  max_order=%d  max_id=%u  files=%zu\n",
           ts, lexicon_path.c_str(), max_order, max_id, sources.size());
    fflush(stdout);

    uint64_t generated = 0;
    for (const auto &src : sources) {
        ensure_token_bin(src, py_script, python_bin, text_column, threads, batch_size, tokenizer_dir);
        std::string ngram_path = sidecar_path_for_bin(src.bin_path);
        if (skip_existing_ngram && path_exists(ngram_path)) {
            Progress::timestamp(ts, sizeof(ts));
            printf("[%s] skipping existing %s\n", ts, ngram_path.c_str());
            fflush(stdout);
            continue;
        }

        Progress::timestamp(ts, sizeof(ts));
        printf("[%s] generating %s from %s\n", ts, ngram_path.c_str(), src.bin_path.c_str());
        fflush(stdout);
        generate_sidecar_for_bin(src.bin_path, ngram_path, trie, max_order);
        generated++;
    }

    Progress::timestamp(ts, sizeof(ts));
    printf("[%s] ngram_input_gen: generated %llu sidecar file(s)\n",
           ts, (unsigned long long)generated);
    fflush(stdout);
    return 0;
}
