#!/usr/bin/env python3
"""
Tokenize parquet files into flat uint32 binary token streams.

Output format: raw little-endian uint32 tokens, 4 bytes per token, no header.
Each document gets a BOS token prepended (matching pretraining dataloader behavior).

Usage:
    python ngram/parquet_to_bin.py shard_00000.parquet
    python ngram/parquet_to_bin.py /path/to/shard_00000.parquet --out /tmp/custom.bin
    python ngram/parquet_to_bin.py /path/to/parquet_dir --workers 8
    python ngram/parquet_to_bin.py /path/to/parquet_dir --out /path/to/bin_dir
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from array import array

import pyarrow.parquet as pq

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nanochat.dataset import DATA_DIR

LOG_INTERVAL = 3.0  # seconds between progress prints


def log(msg):
    print(msg, flush=True)


def worker_log(label, msg):
    log(f"[{label}] {msg}")


def resolve_parquet_path(parquet_arg):
    if not os.path.isabs(parquet_arg) and not os.path.exists(parquet_arg):
        parquet_arg = os.path.join(DATA_DIR, parquet_arg)
    if not os.path.exists(parquet_arg):
        raise FileNotFoundError(f"parquet not found: {parquet_arg}")
    return parquet_arg


def resolve_directory_path(dir_arg):
    if not os.path.isabs(dir_arg) and not os.path.exists(dir_arg):
        dir_arg = os.path.join(DATA_DIR, dir_arg)
    if not os.path.isdir(dir_arg):
        raise FileNotFoundError(f"directory not found: {dir_arg}")
    return dir_arg


def list_parquet_files_in_dir(dir_path):
    parquet_paths = sorted(
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if name.endswith(".parquet") and not name.endswith(".tmp")
    )
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files found in directory: {dir_path}")
    return parquet_paths


def default_output_path(parquet_path):
    stem, _ = os.path.splitext(parquet_path)
    return stem + ".bin"


def resolve_output_path(parquet_path, out_arg, input_is_dir):
    if out_arg is None:
        return default_output_path(parquet_path)

    if input_is_dir:
        os.makedirs(out_arg, exist_ok=True)
        return os.path.join(out_arg, os.path.basename(default_output_path(parquet_path)))

    if os.path.isdir(out_arg):
        return os.path.join(out_arg, os.path.basename(default_output_path(parquet_path)))
    return out_arg


def load_tokenizer(tokenizer_dir):
    if tokenizer_dir is None:
        from nanochat.tokenizer import get_tokenizer

        return get_tokenizer()

    from nanochat.tokenizer import RustBPETokenizer

    return RustBPETokenizer.from_directory(tokenizer_dir)


def convert_one_file(parquet_path, out_path, text_column, threads, batch_size, tokenizer_dir):

    label = os.path.basename(parquet_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    worker_log(label, f"parquet : {parquet_path}")
    worker_log(label, f"output  : {out_path}")
    worker_log(label, "loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_dir)
    bos_id = tokenizer.get_bos_token_id()
    worker_log(label, f"tokenizer loaded  bos_id={bos_id}")

    pf = pq.ParquetFile(parquet_path)
    num_rg = pf.num_row_groups
    worker_log(label, f"row_groups: {num_rg}")

    total_docs = 0
    total_tokens = 0
    t0 = time.time()
    last_log = t0

    with open(out_path, "wb") as out_f:
        for rg_idx in range(num_rg):
            rg = pf.read_row_group(rg_idx, columns=[text_column])
            docs = rg.column(text_column).to_pylist()

            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                token_lists = tokenizer.encode(
                    batch,
                    prepend=bos_id,
                    num_threads=threads,
                )

                buf = array("I")
                for ids in token_lists:
                    buf.extend(ids)
                if sys.byteorder != "little":
                    buf.byteswap()
                out_f.write(buf.tobytes())

                total_docs += len(token_lists)
                total_tokens += len(buf)

                now = time.time()
                if now - last_log >= LOG_INTERVAL:
                    elapsed = now - t0
                    rg_progress = (i + len(batch)) / len(docs) if docs else 1.0
                    pct = ((rg_idx + rg_progress) / num_rg * 100) if num_rg else 100.0
                    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    worker_log(
                        label,
                        f"[{elapsed:6.1f}s] {pct:5.1f}%  rg {rg_idx+1}/{num_rg}  "
                        f"docs={total_docs}  tokens={total_tokens}  "
                        f"tok/s={tok_per_sec:.0f}",
                    )
                    last_log = now

    elapsed = time.time() - t0
    worker_log(label, f"done in {elapsed:.1f}s")
    worker_log(label, f"documents : {total_docs}")
    worker_log(label, f"tokens    : {total_tokens}")
    worker_log(label, f"bytes     : {total_tokens * 4}")
    worker_log(label, f"output    : {out_path}")
    return {
        "parquet": parquet_path,
        "output": out_path,
        "documents": total_docs,
        "tokens": total_tokens,
        "bytes": total_tokens * 4,
        "elapsed": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize parquet files to flat uint32 binary files (4 bytes/token)."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help=(
            "Input parquet file or directory of parquet files. "
            "Relative paths are resolved under NanoChat's default data directory if needed."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Optional output file for single-file input, or output directory for directory input. "
            "Defaults to replacing .parquet with .bin next to each source file."
        ),
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Parquet column containing document text (default: text).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(8, (os.cpu_count() or 8)),
        help="Tokenizer threads for batch encoding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Documents per tokenizer batch (default: 256).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of files to process in parallel (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Optional tokenizer directory. Defaults to NanoChat's standard tokenizer location.",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.workers < 1:
        log("error: --workers must be >= 1")
        sys.exit(1)

    try:
        if args.input_path is None:
            log("error: missing input path")
            sys.exit(1)

        input_path = args.input_path
        if not os.path.isabs(input_path) and not os.path.exists(input_path):
            input_path = os.path.join(DATA_DIR, input_path)

        if os.path.isdir(input_path):
            dir_path = resolve_directory_path(input_path)
            parquet_paths = list_parquet_files_in_dir(dir_path)
            input_is_dir = True
        else:
            parquet_paths = [resolve_parquet_path(input_path)]
            input_is_dir = False
    except FileNotFoundError as exc:
        log(f"error: {exc}")
        sys.exit(1)

    if args.out is not None and input_is_dir and os.path.isfile(args.out):
        log("error: --out must be a directory when the input path is a directory")
        sys.exit(1)

    jobs = []
    for parquet_path in parquet_paths:
        out_path = resolve_output_path(parquet_path, args.out, input_is_dir)
        jobs.append((
            parquet_path,
            out_path,
            args.text_column,
            args.threads,
            args.batch_size,
            args.tokenizer_dir,
        ))

    log(f"files   : {len(jobs)}")
    log(f"workers : {min(args.workers, len(jobs))}")

    try:
        if len(jobs) == 1 or args.workers == 1:
            results = [convert_one_file(*jobs[0])] if jobs else []
        else:
            with mp.get_context("spawn").Pool(processes=min(args.workers, len(jobs))) as pool:
                results = pool.starmap(convert_one_file, jobs)
    except KeyboardInterrupt:
        log("error: interrupted")
        sys.exit(130)

    if len(results) > 1:
        total_docs = sum(result["documents"] for result in results)
        total_tokens = sum(result["tokens"] for result in results)
        total_bytes = sum(result["bytes"] for result in results)
        total_elapsed = sum(result["elapsed"] for result in results)
        log("")
        log("all files complete")
        log(f"documents : {total_docs}")
        log(f"tokens    : {total_tokens}")
        log(f"bytes     : {total_bytes}")
        log(f"worker_s  : {total_elapsed:.1f}")


if __name__ == "__main__":
    main()
