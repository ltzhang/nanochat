"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import numpy as np
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets


def sampled_ngram_data_loader_with_state(
    sample_bin_path, ngram_list_path, B, T, split,
    device="cuda", resume_state_dict=None, seed=1337,
):
    """
    Dataloader over precomputed fixed-size sampled records plus per-token ngram ids.

    sample_bin_path format:
      - uint16 records of shape (chunk_size + 1), where chunk_size must equal T.
    ngram_list_path format:
      - uint32 records of shape (chunk_size), one ngram id per input token position.

    This loader is intended for pretraining with ngram id supervision.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    use_cuda = device == "cuda"

    rec_tokens = T + 1
    token_data = np.memmap(sample_bin_path, dtype=np.uint16, mode="r")
    ngram_data = np.memmap(ngram_list_path, dtype=np.uint32, mode="r")
    assert token_data.size % rec_tokens == 0, f"{sample_bin_path} has invalid size for chunk_size={T}"
    n_records = token_data.size // rec_tokens
    assert ngram_data.size == n_records * T, (
        f"{ngram_list_path} size mismatch: got {ngram_data.size}, expected {n_records * T}"
    )
    local_records = (n_records - ddp_rank + ddp_world_size - 1) // ddp_world_size
    assert local_records > 0, "Not enough records for current world size"

    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    resume_local_pos = resume_state_dict.get("local_pos", 0) if resume_state_dict is not None else 0
    epoch = max(1, int(resume_epoch))
    local_pos = int(resume_local_pos)

    row_buffer = torch.empty((B, rec_tokens), dtype=torch.long)
    row_ngram_buffer = torch.empty((B, T), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_ngram = torch.empty(B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    gpu_ngram = torch.empty(B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    ngram_ids = gpu_ngram.view(B, T)

    while True:
        # Deterministic local permutation per epoch; train split shuffles, val is sequential.
        if split == "train":
            rng = torch.Generator(device="cpu")
            rng.manual_seed(seed + epoch)
            perm = torch.randperm(local_records, generator=rng)
        else:
            perm = torch.arange(local_records, dtype=torch.long)

        # Consume whole batches; drop tail for constant shape.
        max_pos = local_records - (local_records % B)
        if max_pos == 0:
            raise ValueError(f"Not enough local records ({local_records}) for device batch size {B}")
        local_pos = min(local_pos, max_pos)
        if local_pos >= max_pos:
            epoch += 1
            local_pos = 0
            continue

        batch_local_ids = perm[local_pos:local_pos + B]
        local_pos += B

        for row_idx, local_id_t in enumerate(batch_local_ids):
            local_id = int(local_id_t.item())
            global_id = ddp_rank + local_id * ddp_world_size
            tok_start = global_id * rec_tokens
            ng_start = global_id * T
            row_buffer[row_idx].copy_(torch.tensor(token_data[tok_start:tok_start + rec_tokens], dtype=torch.long))
            row_ngram_buffer[row_idx].copy_(torch.tensor(ngram_data[ng_start:ng_start + T], dtype=torch.long))

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        cpu_ngram.copy_(row_ngram_buffer.view(-1))
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        gpu_ngram.copy_(cpu_ngram, non_blocking=use_cuda)

        state_dict = {"epoch": epoch, "local_pos": local_pos, "local_records": local_records}
        yield inputs, targets, ngram_ids, state_dict


def sampled_ngram_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, ngram_ids, state_dict in sampled_ngram_data_loader_with_state(*args, **kwargs):
        yield inputs, targets, ngram_ids
