#!/usr/bin/env python3
"""
Tokenize a parquet file from base_data into a flat uint16 binary token stream.

Output format: raw little-endian uint16 tokens, 2 bytes per token, no header.
Each document gets a BOS token prepended (matching pretraining dataloader behavior).
Aborts with an error if any token ID exceeds 65535 (uint16 overflow).

Usage:
    python process/parquet_to_bin.py --parquet shard_00000.parquet
    python process/parquet_to_bin.py --parquet shard_00000.parquet --out process/output/custom.bin
"""

import argparse
import os
import sys
import time
from array import array

import pyarrow.parquet as pq

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
OUTPUT_DIR = os.path.join(THIS_DIR, "output")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nanochat.dataset import DATA_DIR
from nanochat.tokenizer import get_tokenizer

LOG_INTERVAL = 3.0  # seconds between progress prints


def log(msg):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a parquet file to a flat uint16 binary file (2 bytes/token)."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Parquet filename (basename only) inside base_data, or a full path.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .bin path. Defaults to process/output/<parquet_stem>.bin.",
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
    args = parser.parse_args()

    # Resolve parquet path: if just a filename, look in base_data
    parquet_path = args.parquet
    if not os.path.isabs(parquet_path) and not os.path.exists(parquet_path):
        parquet_path = os.path.join(DATA_DIR, args.parquet)
    if not os.path.exists(parquet_path):
        log(f"error: parquet not found: {parquet_path}")
        sys.exit(1)

    # Resolve output path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.out is None:
        stem = os.path.splitext(os.path.basename(parquet_path))[0]
        out_path = os.path.join(OUTPUT_DIR, stem + ".bin")
    else:
        out_path = args.out

    log(f"parquet : {parquet_path}")
    log(f"output  : {out_path}")

    log("loading tokenizer...")
    tokenizer = get_tokenizer()
    bos_id = tokenizer.get_bos_token_id()
    log(f"tokenizer loaded  bos_id={bos_id}")

    pf = pq.ParquetFile(parquet_path)
    num_rg = pf.num_row_groups
    log(f"row_groups: {num_rg}")
    log("")

    total_docs = 0
    total_tokens = 0
    t0 = time.time()
    last_log = t0

    with open(out_path, "wb") as out_f:
        for rg_idx in range(num_rg):
            rg = pf.read_row_group(rg_idx, columns=[args.text_column])
            docs = rg.column(args.text_column).to_pylist()

            for i in range(0, len(docs), args.batch_size):
                batch = docs[i : i + args.batch_size]
                token_lists = tokenizer.encode(
                    batch,
                    prepend=bos_id,
                    num_threads=args.threads,
                )

                buf = array("H")
                for ids in token_lists:
                    for tok in ids:
                        if tok > 65535:
                            log(f"error: token id {tok} overflows uint16")
                            sys.exit(1)
                    buf.extend(ids)
                if sys.byteorder != "little":
                    buf.byteswap()
                out_f.write(buf.tobytes())

                total_docs += len(token_lists)
                total_tokens += len(buf)

                now = time.time()
                if now - last_log >= LOG_INTERVAL:
                    elapsed = now - t0
                    pct = (rg_idx + (i + len(batch)) / len(docs)) / num_rg * 100
                    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    log(
                        f"[{elapsed:6.1f}s] {pct:5.1f}%  rg {rg_idx+1}/{num_rg}  "
                        f"docs={total_docs}  tokens={total_tokens}  "
                        f"tok/s={tok_per_sec:.0f}"
                    )
                    last_log = now

    elapsed = time.time() - t0
    log("")
    log(f"done in {elapsed:.1f}s")
    log(f"documents : {total_docs}")
    log(f"tokens    : {total_tokens}")
    log(f"bytes     : {total_tokens * 2}")
    log(f"output    : {out_path}")


if __name__ == "__main__":
    main()
