// ngram batch — orchestrate per-file n-gram counting with rolling pre-merge.
//
// For each gram size n in [2..N] sequentially:
//   For each batch of K corpus files:
//     1. Count n-grams in parallel (K forks of "ngram count")
//     2. Pre-merge K count files -> 1 premerg file  (--pre-min filter)
//     3. Delete the K per-file count files
//   Final merge: all premerg files -> OUTDIR/{n}gram.count.bin  (--final-min filter)
//
// Parquet files are tokenized upfront in parallel via python --py-script.
// Self-invocation: uses /proc/self/exe so the same binary handles count/process.

#include "common.h"

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <limits.h>

// ── Utilities ─────────────────────────────────────────────────────────────────

static std::string self_exe() {
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len < 0) die("batch: readlink /proc/self/exe: %s", strerror(errno));
    buf[len] = '\0';
    return buf;
}

static std::string file_stem(const std::string &path) {
    size_t slash = path.rfind('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.rfind('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

static bool has_suffix(const std::string &s, const std::string &suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static void ensure_dir(const std::string &dir) {
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST)
        die("batch: cannot create directory %s: %s", dir.c_str(), strerror(errno));
}

// Try to find parquet_to_bin.py relative to the binary location.
// Common layouts:
//   <repo>/ngram/bin/ngram_proc      -> <repo>/ngram/parquet_to_bin.py
//   <repo>/bin/ngram_proc            -> <repo>/ngram/parquet_to_bin.py
static std::string auto_find_py_script() {
    std::string exe = self_exe();
    size_t slash = exe.rfind('/');
    if (slash == std::string::npos) return "";
    std::string bin_dir = exe.substr(0, slash);          // e.g. /path/ngram/bin
    size_t slash2 = bin_dir.rfind('/');
    std::string root = (slash2 == std::string::npos)
                       ? bin_dir : bin_dir.substr(0, slash2);  // e.g. /path/ngram
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

// ── Job pool ──────────────────────────────────────────────────────────────────

struct Job {
    std::string              label;     // human-readable for progress output
    std::vector<std::string> argv;      // command + arguments
    std::string              log_path;  // subprocess stdout+stderr destination
};

static pid_t start_job(const Job &j) {
    char ts[16]; Progress::timestamp(ts, sizeof(ts));
    printf("[%s] START  %s\n", ts, j.label.c_str()); fflush(stdout);

    pid_t pid = fork();
    if (pid < 0) die("batch: fork: %s", strerror(errno));
    if (pid == 0) {
        FILE *lf = fopen(j.log_path.c_str(), "w");
        if (lf) {
            dup2(fileno(lf), STDOUT_FILENO);
            dup2(fileno(lf), STDERR_FILENO);
            fclose(lf);
        }
        std::vector<char *> args;
        for (auto &s : j.argv) args.push_back(const_cast<char *>(s.c_str()));
        args.push_back(nullptr);
        execvp(args[0], args.data());
        fprintf(stderr, "execvp(%s): %s\n", args[0], strerror(errno));
        _exit(127);
    }
    return pid;
}

// Called after waitpid returns for this job. Prints DONE/FAILED; dies on error.
static void finish_job(const Job &j, int wait_status) {
    bool ok = WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0;
    char ts[16]; Progress::timestamp(ts, sizeof(ts));
    if (ok) {
        printf("[%s] DONE   %s\n", ts, j.label.c_str());
    } else {
        int exit_code = WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : -1;
        printf("[%s] FAILED %s (exit %d) — log: %s\n",
               ts, j.label.c_str(), exit_code, j.log_path.c_str());
        // Dump log to stderr so the user sees what went wrong.
        FILE *lf = fopen(j.log_path.c_str(), "r");
        if (lf) {
            char line[1024];
            while (fgets(line, sizeof(line), lf)) fputs(line, stderr);
            fclose(lf);
        }
        die("batch: aborting due to failed job");
    }
    fflush(stdout);
    remove(j.log_path.c_str());
}

// Run all jobs with at most max_parallel simultaneous processes.
static void run_pool(const std::vector<Job> &jobs, int max_parallel) {
    std::map<pid_t, size_t> running;  // pid -> index into jobs
    size_t next = 0;

    while (next < jobs.size() || !running.empty()) {
        // Fill up to max_parallel.
        while ((int)running.size() < max_parallel && next < jobs.size()) {
            pid_t pid = start_job(jobs[next]);
            running[pid] = next++;
        }
        // Wait for one child.
        if (!running.empty()) {
            int status;
            pid_t pid = waitpid(-1, &status, 0);
            if (pid > 0 && running.count(pid)) {
                size_t idx = running[pid];
                running.erase(pid);
                finish_job(jobs[idx], status);
            }
        }
    }
}

static void run_sync(const Job &j) {
    run_pool({j}, 1);
}

// ── Subcommand ────────────────────────────────────────────────────────────────

static void usage_batch() {
    fputs(
        "usage: ngram batch -n N -o OUTDIR [options] FILE...\n"
        "\n"
        "Orchestrate per-file n-gram counting with rolling pre-merge.\n"
        "For each gram size n in [2..N] sequentially:\n"
        "  Count each file in parallel, pre-merge every K files, then\n"
        "  final-merge all batches into OUTDIR/{n}gram.count.bin.\n"
        "\n"
        "Required:\n"
        "  -n N              Max gram size (counts 2-gram through N-gram)\n"
        "  -o DIR            Output directory for final frequency tables\n"
        "  FILE...           Input .bin and/or .parquet corpus files\n"
        "\n"
        "Options:\n"
        "  --pre-merge K     Files per batch before pre-merge (default: 10)\n"
        "  --pre-min M       Min-count filter at pre-merge (default: 1 = keep all)\n"
        "  --final-min M     Min-count filter at final merge (default: 1 = keep all)\n"
        "  --workers T       Max parallel count subprocesses (default: nproc)\n"
        "  --threads T       Alias for --workers\n"
        "  --tmp DIR         Temp directory for intermediate files (default: /tmp)\n"
        "  --mem-gb GB       RAM per 'ngram count' subprocess in GB (default: 4)\n"
        "  --py-script PATH  Path to parquet_to_bin.py (auto-detected if omitted;\n"
        "                    required when .parquet files are given)\n",
        stderr);
}

int cmd_batch(int argc, char **argv) {
    if (argc == 0) { usage_batch(); return 1; }

    int         n_max     = 0;
    int         pre_merge = 10;
    uint32_t    pre_min   = 1;
    uint32_t    final_min = 1;
    int         workers   = (int)sysconf(_SC_NPROCESSORS_ONLN);
    size_t      mem_gb    = 4;
    std::string out_dir;
    std::string tmp_dir   = "/tmp";
    std::string py_script;
    std::vector<std::string> files;
    std::vector<std::string> temp_token_bins;

    for (int i = 0; i < argc; i++) {
        if      (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage_batch(); return 0; }
        else if (!strcmp(argv[i], "-n")           && i+1 < argc) n_max     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o")           && i+1 < argc) out_dir   = argv[++i];
        else if (!strcmp(argv[i], "--pre-merge")  && i+1 < argc) pre_merge = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pre-min")    && i+1 < argc) pre_min   = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--final-min")  && i+1 < argc) final_min = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--workers")    && i+1 < argc) workers   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")    && i+1 < argc) workers   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tmp")        && i+1 < argc) tmp_dir   = argv[++i];
        else if (!strcmp(argv[i], "--mem-gb")     && i+1 < argc) mem_gb    = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--py-script")  && i+1 < argc) py_script = argv[++i];
        else if (argv[i][0] != '-')                               files.push_back(argv[i]);
    }

    if (n_max < 2 || n_max > (int)MAX_N) { usage_batch(); die("batch: -n must be 2..%zu (got %d)", MAX_N, n_max); }
    if (out_dir.empty()) { usage_batch(); die("batch: -o OUTDIR required"); }
    if (files.empty())   { usage_batch(); die("batch: no input files specified"); }
    if (workers < 1)     workers = 1;
    if (pre_merge < 1)   pre_merge = 1;

    ensure_dir(out_dir);
    ensure_dir(tmp_dir);

    std::string self       = self_exe();
    std::string pid_pfx    = tmp_dir + "/ngbatch_" + std::to_string(getpid());
    int         total_n    = n_max - 1;   // number of gram sizes: 2..n_max
    int         total_batches_per_n =
        (int)((files.size() + (size_t)pre_merge - 1) / pre_merge);

    // ── Print config ──────────────────────────────────────────────────────────
    {
        char ts[16]; Progress::timestamp(ts, sizeof(ts));
        printf("[%s] ngram batch: n=2..%d  files=%zu  pre-merge=%d  "
               "pre-min=%u  final-min=%u  workers=%d  mem-gb=%zu\n",
               ts, n_max, files.size(), pre_merge,
               pre_min, final_min, workers, mem_gb);
        printf("[%s]   out=%s  tmp=%s\n", ts, out_dir.c_str(), tmp_dir.c_str());
        printf("[%s]   estimated peak RAM: %zu GB  (%d workers x %zu GB/count)\n",
               ts, (size_t)workers * mem_gb, workers, mem_gb);
        fflush(stdout);
    }

    // ── Step 1: Tokenize parquet files upfront ────────────────────────────────
    {
        std::vector<size_t> parquet_idxs;
        for (size_t i = 0; i < files.size(); i++)
            if (has_suffix(files[i], ".parquet")) parquet_idxs.push_back(i);

        if (!parquet_idxs.empty()) {
            if (py_script.empty()) py_script = auto_find_py_script();
            if (py_script.empty())
                die("batch: --py-script required: .parquet files present "
                    "but parquet_to_bin.py not found automatically");

            char ts[16]; Progress::timestamp(ts, sizeof(ts));
            printf("[%s] === Step 1: Tokenizing %zu parquet file(s) "
                   "(workers=%d) ===\n",
                   ts, parquet_idxs.size(), workers);
            fflush(stdout);

            std::vector<Job> tok_jobs;
            for (size_t idx : parquet_idxs) {
                std::string stem    = file_stem(files[idx]);
                std::string out_bin = pid_pfx + "_tok_" + stem + ".bin";
                std::string log     = pid_pfx + "_tok_" + stem + ".log";
                Job j;
                j.label    = "tokenize: " + files[idx];
                j.log_path = log;
                j.argv     = {"python3", py_script,
                              files[idx],
                              "--out",     out_bin};
                tok_jobs.push_back(j);
                temp_token_bins.push_back(out_bin);
                files[idx] = out_bin;   // replace .parquet with produced .bin
            }
            run_pool(tok_jobs, workers);

            Progress::timestamp(ts, sizeof(ts));
            printf("[%s] === Step 1 done ===\n\n", ts); fflush(stdout);
        } else {
            char ts[16]; Progress::timestamp(ts, sizeof(ts));
            printf("[%s] === Step 1: No parquet files, skipping ===\n\n", ts);
            fflush(stdout);
        }
    }

    // ── Step 2: For each gram size ────────────────────────────────────────────
    for (int n = 2; n <= n_max; n++) {
        char ts[16]; Progress::timestamp(ts, sizeof(ts));
        printf("[%s] === Step 2: %d-gram  (%d/%d) ===\n",
               ts, n, n - 1, total_n); fflush(stdout);

        std::vector<std::string> premerg_files;
        int batch_idx = 0;

        for (size_t bi = 0; bi < files.size(); bi += (size_t)pre_merge) {
            size_t bend = std::min(bi + (size_t)pre_merge, files.size());
            size_t batch_size = bend - bi;
            batch_idx++;

            Progress::timestamp(ts, sizeof(ts));
            printf("[%s] [batch %d/%d] counting %d-grams: %zu file(s)\n",
                   ts, batch_idx, total_batches_per_n, n, batch_size);
            fflush(stdout);

            // Build parallel count jobs for this batch.
            std::vector<std::string> count_outs;
            std::vector<Job> count_jobs;
            for (size_t fi = bi; fi < bend; fi++) {
                std::string tag = std::to_string(batch_idx) + "_" +
                                  std::to_string(fi) + "_" +
                                  std::to_string(n) + "gram";
                std::string out = pid_pfx + "_cnt_" + tag + ".bin";
                std::string log = pid_pfx + "_cnt_" + tag + ".log";
                count_outs.push_back(out);
                Job j;
                j.label    = "count " + std::to_string(n) + "-gram: " +
                              file_stem(files[fi]);
                j.log_path = log;
                j.argv     = {self, "count",
                              "-n", std::to_string(n),
                              "-m", std::to_string(mem_gb),
                              "-o", out,
                              files[fi]};
                count_jobs.push_back(j);
            }
            run_pool(count_jobs, workers);

            // Pre-merge this batch.
            Progress::timestamp(ts, sizeof(ts));
            printf("[%s] [batch %d/%d] pre-merging %zu count file(s) -> 1\n",
                   ts, batch_idx, total_batches_per_n, batch_size);
            fflush(stdout);

            std::string pmtag  = std::to_string(batch_idx) + "_" +
                                 std::to_string(n) + "gram";
            std::string pm_out = pid_pfx + "_premerg_" + pmtag + ".bin";
            std::string pm_log = pid_pfx + "_premerg_" + pmtag + ".log";

            Job pm_job;
            pm_job.label    = "pre-merge " + std::to_string(n) + "-gram batch " +
                              std::to_string(batch_idx);
            pm_job.log_path = pm_log;
            pm_job.argv     = {self, "process", "merge",
                               "-o", pm_out,
                               "--min-count", std::to_string(pre_min)};
            for (auto &cf : count_outs)
                pm_job.argv.insert(pm_job.argv.end(), {"-i", cf});
            run_sync(pm_job);

            // Clean up per-file count files.
            for (auto &cf : count_outs) remove(cf.c_str());
            premerg_files.push_back(pm_out);

            Progress::timestamp(ts, sizeof(ts));
            printf("[%s] [batch %d/%d] done\n\n",
                   ts, batch_idx, total_batches_per_n);
            fflush(stdout);
        }

        // Final merge for this gram size.
        std::string final_out = out_dir + "/" + std::to_string(n) + "gram.count.bin";
        Progress::timestamp(ts, sizeof(ts));
        printf("[%s] [final] merging %zu premerg file(s) -> %s\n",
               ts, premerg_files.size(), final_out.c_str());
        fflush(stdout);

        std::string fl_log = pid_pfx + "_final_" + std::to_string(n) + "gram.log";
        Job fl_job;
        fl_job.label    = "final merge " + std::to_string(n) + "-gram";
        fl_job.log_path = fl_log;
        fl_job.argv     = {self, "process", "merge",
                           "-o", final_out,
                           "--min-count", std::to_string(final_min)};
        for (auto &pf : premerg_files)
            fl_job.argv.insert(fl_job.argv.end(), {"-i", pf});
        run_sync(fl_job);

        for (auto &pf : premerg_files) remove(pf.c_str());

        Progress::timestamp(ts, sizeof(ts));
        printf("[%s] [final] done -> %s\n\n", ts, final_out.c_str());
        fflush(stdout);
    }

    for (const auto &path : temp_token_bins)
        remove(path.c_str());

    {
        char ts[16]; Progress::timestamp(ts, sizeof(ts));
        printf("[%s] batch complete. Output in: %s\n", ts, out_dir.c_str());
        fflush(stdout);
    }
    return 0;
}
