# NanoChat N-Gram Input Integration Design

## Goal

Extend NanoChat so pretraining can consume a second aligned input stream of n-gram IDs alongside token IDs, and let the model use learned n-gram embeddings as an internal memory pathway similar to the existing value embeddings.

This document now reflects confirmed v1 decisions.

---

## Current NanoChat Baseline

Today the base training path is:

- Data comes from parquet shards in [`nanochat/dataset.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/dataset.py)
- The pretraining dataloader tokenizes text online in [`nanochat/dataloader.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/dataloader.py)
- The model consumes only token IDs in [`nanochat/gpt.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/gpt.py)
- The model already has a large auxiliary embedding family, `value_embeds`, injected into inner layers

So your proposal is not a small tweak. It implies a parallel pretokenized dataset path and a second embedding pathway in the model/training loop.

---

## My Understanding Of Your Idea

I believe the intended system is:

### 1. Data format

For every existing token chunk file in the training data directory, there is a corresponding `ngram.bin` sidecar file.

- Token file: token IDs, as today
- N-gram file: one n-gram ID per token position, aligned with the token sequence

The n-gram vocabulary is global and flattened across gram orders:

- 2-grams occupy one contiguous ID range
- 3-grams occupy the next contiguous range
- ...
- K-grams occupy the final range

Example:

- if there are 1M valid 2-grams and 2M valid 3-grams
- then global n-gram IDs span `[0, 3_000_000)` or `[1, 3_000_000]` depending on whether `0` is reserved for "no n-gram"

The model never needs separate per-order embedding tables unless we choose that internally. From the dataset point of view, it receives one flattened n-gram ID tensor.

### 2. Batch output

Each training batch should now contain at least:

- `token_ids`
- `ngram_ids`
- and still produce next-token `targets`

Conceptually:

```python
batch = {
    "input_ids": ...,
    "ngram_ids": ...,
    "targets": ...,
}
```

### 3. Model-side usage

The token embedding path remains.

In addition, the model has an n-gram embedding table sized by `ngram_vocab_size`, potentially very large, and those embeddings are not limited to input fusion. Instead, they act more like value embeddings:

- looked up from `ngram_ids`
- injected into internal transformer layers
- combined with the hidden state using learned gates / mixing coefficients

So this is closer to "retrieved memory features conditioned on n-gram identity" than to "just add another input embedding once at the bottom."

### 4. Memory placement

Because token and n-gram embedding tables may both be large, either table may be allowed to live in host memory instead of GPU memory.

Training should support a pipeline pattern where:

- IDs originate on the training device or are staged toward it
- the relevant embedding layer may reside on CPU
- embedding outputs are transferred to the compute device
- downstream transformer layers continue on GPU

This is a command-line training option, not a separate model family.

### 5. Scope

This is primarily a training-time architecture/data-pipeline change, but v1 also includes inference-time n-gram ID reconstruction from a loaded lexicon file.

So the first implementation must cover:

- training with aligned `foo.bin` + `foo.ngram.bin`
- runtime loading of the n-gram lexicon
- autoregressive longest-match lookup to recover `ngram_ids` online

---

## Confirmed V1 Decisions

The following are now confirmed:

- `ngram_ids[t]` is aligned to `token_ids[t]`
- each position stores the selected n-gram ending at token position `t`
- longest match wins
- `0` means "no n-gram"; real IDs start at `1`
- token and n-gram files are aligned flat streams
- sidecar naming is `foo.bin` and `foo.ngram.bin`
- n-gram IDs use one flattened global vocabulary spanning 2-grams through K-grams
- the model uses one shared n-gram embedding table
- n-gram embeddings are injected into inner layers on the same pattern as `value_embeds`, with separate learned gates
- CPU offload means lookup on CPU and transfer of embedding activations to the compute device
- token embeddings and n-gram embeddings can each be placed independently on CPU or GPU
- v1 is training-first, but it must also support inference-time n-gram lookup from a loaded n-gram lexicon file
- this feature is a new binary-loader path and does not replace the current parquet-online loader

---

## New Inference Requirement

In addition to `foo.ngram.bin`, there is another file that explicitly lists n-grams and their assigned IDs.

This file is loaded by the process and used at inference time to recover n-gram IDs online from the generated token stream, using the same "longest wins" rule as training.

This means the n-gram ID space is not just a training-side annotation artifact. It is part of the model input contract and must be reconstructible at inference time.

---

## Implemented Design

This section describes the code that now exists in the repo, not just the original plan.

### 1. Data layer changes

Implemented in [`nanochat/dataloader.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/dataloader.py).

The repo now has a second pretraining loader path for aligned binary shards:

- token shards are all `*.bin` files that are not `*.ngram.bin`
- each token shard must have a same-prefix sidecar `*.ngram.bin`
- both streams are treated as flat aligned `int32` arrays of equal length
- each training row uses:
  - `input_ids = tokens[start:start+T]`
  - `targets = tokens[start+1:start+T+1]`
  - `ngram_ids = ngrams[start:start+T]`

The aligned loader returns:

```python
inputs, ngram_ids, targets, state_dict
```

where `state_dict` tracks `shard_idx`, `row_idx`, and `epoch` for resume.

### 2. Per-tensor batch placement

The binary loader and the original BOS-bestfit loader now both support separate output placement:

- `target_device`
- `token_device`
- `ngram_device` for the aligned n-gram loader

This is the key fix for mixed CPU/GPU embedding placement.

In the common configuration:

- `target_device` is the compute GPU
- `token_device` is also the compute GPU
- `ngram_device` is CPU

That means:

- `targets` are staged where the loss is computed
- `idx` is staged where token embedding lookup happens
- `ngram_ids` is staged where n-gram embedding lookup happens

This avoids the bad earlier pattern where `ngram_ids` could be copied to GPU only to be copied back to CPU for lookup.

### 3. Lexicon file and runtime lookup

Implemented in [`nanochat/ngram.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/ngram.py).

The runtime now loads a dedicated lexicon file containing n-gram token sequences and IDs. At inference time:

- token history is scanned with longest-match semantics
- `0` means "no n-gram"
- the emitted ID is the longest matching suffix ending at the current token

The model API exposes:

- `load_ngram_lexicon(path)`
- `set_ngram_lexicon(lexicon)`
- `encode_ngram_ids(token_ids, device=None)`

The same lexicon family must be used for:

- offline `*.ngram.bin` generation
- online inference-time n-gram reconstruction

otherwise training and inference semantics diverge.

---

## Model Modifications

### 1. Config and parameters

Implemented in [`nanochat/gpt.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/gpt.py).

`GPTConfig` now includes:

- `ngram_vocab_size`
- `use_ngram_embeds`
- `token_embed_device`
- `ngram_embed_device`

The model now owns:

- the original token embedding `transformer.wte`
- one shared n-gram embedding table `ngram_embeds`
- per-block n-gram gates `nge_gate` on the same block pattern as `value_embeds`

The n-gram embedding table is shared across all eligible layers. It is not duplicated per layer.

### 2. Forward signature

The forward path is now:

```python
forward(self, idx, targets=None, ngram_ids=None, kv_cache=None, loss_reduction="mean")
```

Rules:

- if `use_ngram_embeds=False`, `ngram_ids` may be omitted
- if `use_ngram_embeds=True`, `ngram_ids` are required
- `ngram_ids.shape == idx.shape`

### 3. Injection path

The n-gram lookup happens once per forward pass:

```python
nge = self._embedding_lookup(self.ngram_embeds, ngram_ids, compute_device, x.dtype)
```

Then `nge` is reused across the transformer blocks that already use `value_embeds`.

Inside attention, `nge` is mixed into the value stream with a separate learned gate, analogous to `ve`.

This matters for correctness and efficiency:

- correctness: all downstream uses share one autograd path back to the same embedding parameter
- efficiency: the CPU lookup and CPU->GPU activation copy happen once per forward, not once per layer

### 4. Current scope limit

`value_embeds` were not offloaded as part of this feature. They still follow the compute device path in the current implementation.

---

## Device Placement And Transfer Path

### 1. Embedding placement

Implemented by `place_embedding_modules()` in [`nanochat/gpt.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/gpt.py).

The model is first created on the compute device, then embedding modules are explicitly placed:

- token embedding on compute or CPU
- n-gram embedding on compute or CPU
- transformer trunk stays on the compute device

The embedding weights stay resident on their chosen device across training. They are not copied wholesale each batch.

### 2. Lookup path

Cross-device lookup is centralized in `_embedding_lookup()`:

1. inspect `embedding.weight.device`
2. if the incoming ID tensor is on the wrong device, move the IDs to the embedding device
3. run the lookup on the embedding device
4. cast output dtype if needed
5. move the looked-up dense activation to the compute device

So the thing that moves per forward is:

- integer ID tensors when needed
- dense embedding activations
- backward gradients flowing back through the activation copy

The thing that does not move per forward is:

- the embedding weight matrix itself

### 3. The important split between `idx`, `idx_compute`, and `ngram_ids`

The current forward path deliberately keeps different tensors on different devices:

- `idx` stays on the token-embedding device for token lookup
- `idx_compute` is a compute-device copy used by `value_embeds`
- `ngram_ids` stays on the n-gram-embedding device until the n-gram lookup runs
- `targets` are moved to the compute device for loss

This is why the loader now accepts separate output placement instead of a single `device` argument.

### 4. Training-time frequency of transfer

The activation transfer happens once per forward pass.

During training that means:

- once per micro-batch for token embeddings if token embedding is CPU-hosted
- once per micro-batch for n-gram embeddings if n-gram embedding is CPU-hosted

If `grad_accum_steps > 1`, this happens on every micro-step, not once per optimizer step.

### 5. `torch.compile`

In [`scripts/base_train.py`](/home/lintaoz/aiwork/nanochat-ngram/scripts/base_train.py), `torch.compile` is disabled when either token or n-gram embeddings are CPU-hosted.

That is intentional. The cross-device embedding path is not the graph shape we want to hand to `torch.compile` in this implementation.

---

## Training Script Modifications

### 1. CLI and loader selection

Implemented in [`scripts/base_train.py`](/home/lintaoz/aiwork/nanochat-ngram/scripts/base_train.py).

The training script now supports:

- `--use-ngram-input`
- `--train-bin-dir`
- `--val-bin-dir`
- `--ngram-vocab-size`
- `--ngram-lexicon-path`
- `--token-embed-device`
- `--ngram-embed-device`

Behavior:

- if `--use-ngram-input` is off, the old parquet-online loader path is used
- if it is on, the aligned binary token/ngram loader is used

### 2. Batch-device resolution

The script resolves three batch destinations:

- `target_batch_device`
- `token_batch_device`
- `ngram_batch_device`

These are derived from the compute device and the two embedding placement flags.

This keeps the loader contract aligned with the real execution path.

### 3. Optimizer grouping

The n-gram embedding parameters are now part of the AdamW optimizer groups in [`nanochat/gpt.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/gpt.py).

They currently share the same family of hyperparameters as the other embedding-like tables, with a `0.5x` multiplier relative to token embedding LR, matching `value_embeds`.

### 4. Resume metadata

Checkpoints now preserve:

- model config with n-gram fields
- `ngram_lexicon_path`
- aligned dataloader resume state

Old checkpoints remain loadable because missing config keys are patched in [`nanochat/checkpoint_manager.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/checkpoint_manager.py).

---

## Inference Modifications

### 1. Engine path

Implemented in [`nanochat/engine.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/engine.py).

Inference now does two different n-gram-related things:

- prefill: encode prompt tokens into `ngram_ids`
- decode: for each newly generated token, compute the longest matching suffix ID online

### 2. Device placement during inference

The inference path now constructs prompt/decode `ngram_ids` on the n-gram embedding device, not blindly on the compute device.

This mirrors the training fix:

- if n-gram embedding is on CPU, inference `ngram_ids` stay on CPU
- only the looked-up n-gram activation is copied to the compute device

### 3. Eval path

The same device-aware rule was added to core evaluation in [`nanochat/core_eval.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/core_eval.py).

---

## Optimizer And Checkpoint Gotchas

### 1. Optimizer state must live with the parameter

For CPU-hosted embeddings, the optimizer state for those embeddings must also live on CPU.

For fresh training runs this is already natural:

- AdamW state uses `zeros_like(parameter)`
- so state tensors are created on the parameter device

For resumed runs, the more subtle issue is checkpoint deserialization.

The checkpoint loader now loads optimizer shards onto CPU first in [`nanochat/checkpoint_manager.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/checkpoint_manager.py), then the optimizer restore path places state tensors to match parameter placement.

This avoids loading a mixed-device optimizer shard straight onto GPU memory.

### 2. Defensive optimizer-state placement

[`nanochat/optim.py`](/home/lintaoz/aiwork/nanochat-ngram/nanochat/optim.py) now exposes `place_optimizer_state(optimizer)`.

This is mostly defensive. PyTorch already tends to relocate loaded optimizer state to the owning parameter device during `load_state_dict()`, but the helper makes the invariant explicit and keeps restore behavior robust for this mixed-placement setup.

---

## Correctness Gotchas

### 1. Do not copy the embedding weights back and forth

The safe design is:

- keep one master embedding parameter on its chosen device
- run the lookup there
- move the looked-up activation to the compute device

The unsafe design is:

- make multiple transient GPU copies of a CPU embedding parameter and treat them as trainable replicas

That would split gradient ownership and require manual merge logic.

### 2. Reuse the looked-up activation if it is consumed multiple times

For n-grams, the current implementation does the right thing:

- lookup once
- produce one `nge` activation tensor
- reuse that tensor across all eligible layers

This preserves gradient accumulation correctly because all layer uses feed back into the same activation and therefore back into the same embedding parameter.

### 3. Repeated lookups are not wrong, but they are wasteful

If the same embedding parameter is looked up multiple times in one forward pass, gradients will still accumulate to the same parameter as long as the parameter itself is shared.

But:

- CPU lookup work is repeated
- CPU<->GPU transfer work is repeated
- autograd graph size increases

So the implemented rule is:

- share the parameter
- reuse the looked-up activation where possible

### 4. `ngram_ids` do not need to live on GPU when the n-gram table is on CPU

This was one of the subtle fixes after the initial implementation.

If `ngram_embeds` is on CPU, then `ngram_ids` should also stay on CPU in the loader and inference path. The GPU only needs the dense looked-up activation `nge`.

### 5. There is still backward traffic across the device boundary

Even with CPU-hosted embeddings, the backward pass is still correct because autograd tracks the device copy.

The cost is that training still pays for:

- forward activation transfer to the compute device
- backward gradient transfer back toward the CPU-hosted embedding

So CPU-hosted embeddings are a memory tradeoff, not a free optimization.

### 6. `value_embeds` remain a special case

The current implementation does not make `value_embeds` independently placeable. They still use `idx_compute` on the compute device.

So the current mixed-device design covers:

- token embedding
- n-gram embedding

but not:

- per-layer `value_embeds`

### 7. Dataloader/device coupling is now intentional

Earlier NanoChat loaders assumed a single destination device for the whole batch.

That assumption is no longer valid once embeddings can live on different devices.

The new loader contract is intentionally more explicit because placement of:

- `targets`
- token IDs
- n-gram IDs

are genuinely different concerns now.

---

## Residual Risks And Future Work

### 1. Throughput

CPU-hosted embeddings can still become the throughput bottleneck, especially for large `B x T` or very large host tables.

### 2. Overlap

The implementation is correct and pragmatic, but not yet an aggressively optimized pipeline-parallel design. There is still room to improve overlap between:

- host lookup
- host/device copy
- compute-device transformer execution

### 3. Validation of offline data

The loader validates shard pairing and equal stream length, but the full offline data-generation contract still depends on:

- correct longest-match generation
- correct ID assignment
- correct use of `0` as the reserved null value

### 4. Inference/train consistency

The lexicon file remains the source of truth. If training-side `*.ngram.bin` files are generated from one lexicon and inference uses another, the model will silently see inconsistent n-gram semantics.
