# Output File Formats

This document lists only the file formats used by the current NanoChat n-gram implementation.

All binary integer values are 32-bit little-endian signed integers on disk because the current loader uses `torch.from_file(..., dtype=torch.int32)` and then casts to `torch.long` in memory.

## 1. Token shard: `*.bin`

Purpose:
- Flat token ID stream for training.

Naming:
- Any shard ending in `.bin`
- Must not end in `.ngram.bin`

On-disk layout:
```text
[token_0][token_1]...[token_N-1]
```

Properties:
- No header
- No per-record delimiter
- One integer per token position
- File size must be divisible by 4

Loader behavior:
- The training loader treats the file as one flat token stream
- Examples are taken as non-overlapping windows with start offsets `row_idx * T`
- For a window starting at `start`:
  - `input_ids = tokens[start : start + T]`
  - `targets = tokens[start + 1 : start + T + 1]`

Code reference:
- [nanochat/dataloader.py](/home/lintaoz/aiwork/nanochat-ngram/nanochat/dataloader.py)

## 2. N-gram sidecar shard: `*.ngram.bin`

Purpose:
- Flat n-gram ID stream aligned 1:1 with the token shard.

Naming:
- For token shard `foo.bin`, the sidecar must be `foo.ngram.bin`

On-disk layout:
```text
[ngram_id_0][ngram_id_1]...[ngram_id_N-1]
```

Properties:
- No header
- No per-record delimiter
- Exactly one integer per token position
- Must have the same number of entries as the corresponding token shard
- File size must be divisible by 4

ID semantics:
- `0` means "no n-gram"
- Positive IDs refer to the flattened global n-gram vocabulary
- Each position stores the longest matching n-gram ending at that token position

Loader behavior:
- For a window starting at `start`:
  - `ngram_ids = ngrams[start : start + T]`
- The loader validates that token shard length and n-gram shard length match exactly

Code reference:
- [nanochat/dataloader.py](/home/lintaoz/aiwork/nanochat-ngram/nanochat/dataloader.py)

## 3. N-gram lexicon file

Purpose:
- Runtime lookup table used at inference time to reconstruct n-gram IDs from token sequences.

On-disk format:
- UTF-8 text
- Blank lines are ignored
- Lines starting with `#` are ignored
- Each data line is one of:

```text
<global_id>\t<token0 token1 ...>
<global_id>\t<n>\t<token0 token1 ...>
<global_id>\t<n>\t<token0 token1 ...>\t<count>\t<display_text>
```

Examples:
```text
1	10 20
2	3	10 20 30
3	2	42 99	1234	hello world
```

Properties:
- `global_id` must be a positive integer
- `0` is reserved and must not appear as a real n-gram ID
- If the optional `n` column is present, it must match the number of token IDs on the line
- Additional columns after the token field are allowed and ignored by the current loader
- The same token sequence must not appear with conflicting IDs

Runtime semantics:
- The lexicon is loaded into a reversed trie
- Inference uses longest-suffix match over the current token history
- The resulting per-position IDs follow the same semantics as the aligned `.ngram.bin` files

Code references:
- [nanochat/ngram.py](/home/lintaoz/aiwork/nanochat-ngram/nanochat/ngram.py)
- [nanochat/engine.py](/home/lintaoz/aiwork/nanochat-ngram/nanochat/engine.py)
- [nanochat/gpt.py](/home/lintaoz/aiwork/nanochat-ngram/nanochat/gpt.py)

## Not Included

The current implementation does not define or require any of the following in this document:
- count-table binary formats
- posting-list binary formats
- sampling manifests
- `.meta` sidecar formats for n-gram outputs
- separate per-order embedding files

If those are added later, they should be documented separately once they exist in code.
