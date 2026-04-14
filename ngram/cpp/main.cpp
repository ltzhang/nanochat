// ngram — n-gram corpus analysis tool
//
// Subcommands:
//   ngram count  -n N  -o FREQ_FILE  [options]  FILE...
//   ngram process <op> [options]
//   ngram batch  -n N  -o OUTDIR  [options]  FILE...
//
// count options:
//   -n N              n-gram size (1–8, required)
//   -o FILE           output frequency table (required)
//   -m GB             hash table memory in GB (default 4)
//   -t T              flush-prune threshold: drop entries with per-file
//                     count <= T at flush time (default 1)
//   -s DIR            scratch directory for sorted runs (default /tmp)
//   --min-count M     only emit n-grams with count >= M (default 1)
//   --top-k K         only emit the K most frequent n-grams
//
// process ops (-o is optional for all; a descriptive default name is derived
//              from the input path when omitted):
//   filter-min  -i INPUT [-o OUTPUT] -t THRESHOLD
//   sort-count  -i INPUT [-o OUTPUT]
//   to-text     -i INPUT [-o OUTPUT] [--vocab VOCAB_FILE]
//   merge       -i INPUT [-i INPUT ...] [-o OUTPUT] [--min-count M]
//   split       -i INPUT -t THRESHOLD

#include <cstring>
#include <cstdio>

int cmd_count(int argc, char **argv);
int cmd_process(int argc, char **argv);
int cmd_batch(int argc, char **argv);
// int cmd_index(int argc, char **argv);
// int cmd_sample(int argc, char **argv);
// int cmd_validate(int argc, char **argv);

static void usage() {
    fputs(
        "usage: ngram <subcommand> [options] ...\n"
        "\n"
        "Subcommands:\n"
        "  count   Count n-gram frequencies across corpus files\n"
        "  process Post-process frequency tables\n"
        "  batch   Orchestrate per-file counting + rolling pre-merge\n"
        "\n"
        "Run 'ngram <subcommand>' with no arguments for per-command help.\n"
        "\n"
        "Typical workflow:\n"
        "  # Step 1: count 2-gram frequencies\n"
        "  ngram count -n 2 -o data/2gram.count.bin data/*.bin\n"
        "\n"
        "  # Step 2 (optional): sort by frequency, print top-k cutoff table\n"
        "  ngram process sort-count -i data/2gram.count.bin\n"
        "  # -> writes data/2gram.count.sorted.bin + data/2gram.count_sort_hist.txt\n"
        "\n"
        "  # Step 3 (optional): split into high/low frequency halves\n"
        "  ngram process split -i data/2gram.count.sorted.bin -t 5\n"
        "  # -> writes 2gram.count.sorted.high_5.bin and .low_5.bin\n"
        "\n"
        "  # Step 4 (optional): dump frequency table as text\n"
        "  ngram process to-text -i data/2gram.count.bin\n"
        "  # -> writes data/2gram.count.txt\n"
        "\n"
        "  # Step 5 (optional): merge multiple count tables\n"
        "  ngram process merge -i shard0.count.bin -i shard1.count.bin\n"
        "  # -> writes shard0.count.merged.bin\n",
        stderr);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    if (!strcmp(argv[1], "count"))  return cmd_count(argc - 2, argv + 2);
    if (!strcmp(argv[1], "process")) return cmd_process(argc - 2, argv + 2);
    if (!strcmp(argv[1], "batch"))  return cmd_batch(argc - 2, argv + 2);
    // if (!strcmp(argv[1], "index"))  return cmd_index(argc - 2, argv + 2);
    // if (!strcmp(argv[1], "sample")) return cmd_sample(argc - 2, argv + 2);
    // if (!strcmp(argv[1], "validate")) return cmd_validate(argc - 2, argv + 2);
    usage();
    return 1;
}
