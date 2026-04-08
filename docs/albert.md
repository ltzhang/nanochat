# Partial Layer Weight Tying for Base Pretraining

## Goal

This branch adds an optional form of ALBERT-like weight sharing to nanochat base pretraining.

The goal is not to reproduce canonical ALBERT, where all transformer layers share the same weights.
Instead, this change allows sharing only over a selected middle section of the stack, and only within
fixed contiguous groups of layers.

This makes it possible to explore architectures such as:

- keep the early layers untied
- keep the late layers untied
- tie only the middle layers
- tie the middle in groups of size `k`

Example:

- depth = 20
- tie range = `[4, 16)`
- group size = `4`

Then the tying groups are:

- layers `4, 5, 6, 7` share weights
- layers `8, 9, 10, 11` share weights
- layers `12, 13, 14, 15` share weights

Layers `0-3` and `16-19` remain independent.

## Scope

This implementation currently targets **base pretraining** only.

The main entrypoint is [scripts/base_train.py](/home/lintaoz/aiwork/nanochat/scripts/base_train.py).

The changes are centered in:

- [scripts/base_train.py](/home/lintaoz/aiwork/nanochat/scripts/base_train.py)
- [nanochat/gpt.py](/home/lintaoz/aiwork/nanochat/nanochat/gpt.py)
- [nanochat/checkpoint_manager.py](/home/lintaoz/aiwork/nanochat/nanochat/checkpoint_manager.py)
- [tests/test_layer_tying.py](/home/lintaoz/aiwork/nanochat/tests/test_layer_tying.py)

## User-Facing Interface

Three new optional CLI flags were added to base pretraining:

- `--tie-layers-start`
- `--tie-layers-end`
- `--tie-layers-stride`

These map onto new `GPTConfig` fields:

- `tie_layers_start`
- `tie_layers_end`
- `tie_layers_stride`

The semantics are:

- the tied region is the half-open interval `[start, end)`
- that interval is partitioned into contiguous groups of size `stride`
- all layers inside each group share the same transformer matrix weights

Important:

- `stride` means **group size**
- `stride` does **not** mean “tie every k-th layer”

So for:

```bash
python -m scripts.base_train \
  --depth 20 \
  --tie-layers-start 4 \
  --tie-layers-end 16 \
  --tie-layers-stride 4
```

the groups are:

- `4, 5, 6, 7`
- `8, 9, 10, 11`
- `12, 13, 14, 15`

## Validation Rules

The config is treated as disabled if all three values are `None`.

If any one of the three is provided, then all three must be provided.

The implementation enforces:

- `0 <= start < end <= n_layer`
- `stride > 0`
- `(end - start) % stride == 0`

The last condition means the tied region must break exactly into groups of size `stride`.

Example:

- valid: `[4, 16)` with `stride=4`
- invalid: `[4, 15)` with `stride=4`
- invalid: `[1, 6)` with `stride=2`

## What Is Actually Shared

The implementation shares the **core transformer matrix weights** inside each tied group.

Specifically, for each destination layer in a tied group, the following parameters are aliased to the
group’s first layer:

- `attn.c_q.weight`
- `attn.c_k.weight`
- `attn.c_v.weight`
- `attn.c_proj.weight`
- `mlp.c_fc.weight`
- `mlp.c_proj.weight`

This is implemented in [nanochat/gpt.py](/home/lintaoz/aiwork/nanochat/nanochat/gpt.py) by:

- validating the tie config
- constructing a normal `ModuleList` of per-layer `Block`s
- re-pointing selected `Parameter` objects so tied layers reference the same underlying weights

## What Is Not Shared

Several pieces intentionally remain layer-local.

### 1. Layer object identity

Each layer still has its own `Block` instance.

This is important because nanochat’s layers are not pure interchangeable blocks. Some layer behavior
depends on the layer index.

### 2. Attention cache index

`CausalSelfAttention` stores `layer_idx`, which is used for KV cache lookup during inference.

If entire `Block` objects were reused directly, multiple logical layers would point at the same cache
slot, which would break inference and cache updates.

Because of this, we do **not** replace multiple layers with the exact same `Block` module object.
We only share selected parameters.

### 3. Per-layer scalar parameters

These remain distinct:

- `resid_lambdas[i]`
- `x0_lambdas[i]`

This means tied layers still have independent residual scaling and independent `x0` blending.

### 4. Per-layer sliding-window pattern placement

`window_sizes[i]` remains layer-specific, because the model still has a full logical depth even when
some weights are shared.

### 5. Value embedding gate module

The attention `ve_gate` is left layer-local.

This is deliberate because:

- not all layers have a value embedding gate
- gate presence depends on layer index via `has_ve(layer_idx, n_layer)`
- tying entire attention modules would entangle that layer-dependent structure

### 6. Value embeddings

The `value_embeds` table is also left untouched. This change only shares the main transformer block
matrices.

## Why This Is Not Canonical ALBERT

Canonical ALBERT typically shares a single transformer block across all layers.

This branch does something narrower and more controllable:

- sharing is optional
- sharing can be limited to a subrange
- sharing is by contiguous groups
- groups do not share with each other
- some layer-local behavior remains independent even inside a tied group

So this is better described as **partial grouped layer weight tying** than strict ALBERT.

The branch name is still `albert` because the conceptual inspiration is the same: reduce parameter count
by reusing block weights across multiple layers.

## Construction Flow

### In `scripts/base_train.py`

Base pretraining parses the new CLI flags and converts `-1` to `None` before building `GPTConfig`.

Only the actual target training model receives the tie config.

The auxiliary `d12` reference model used for scaling-law heuristics is intentionally kept as the baseline
untied architecture. This matters because:

- the reference is used only for heuristic scaling calculations
- reusing the tie config there could fail validation for unrelated depths
- it would also make the reference architecture drift away from the intended baseline

### In `nanochat/gpt.py`

`GPT.__init__` does the following relevant work:

1. validate the tie config
2. construct all logical layers normally
3. apply parameter aliasing across the requested groups

`init_weights()` then initializes the model and re-applies the tying once more.

That second application is important. See the gotchas section below.

## Checkpoint and Resume Behavior

Checkpoint loading is handled in [nanochat/checkpoint_manager.py](/home/lintaoz/aiwork/nanochat/nanochat/checkpoint_manager.py).

There are two pieces to be aware of.

### 1. Old checkpoints need config patching

Older checkpoints will not have the three new config keys. Those are patched to `None` on load:

- `tie_layers_start`
- `tie_layers_end`
- `tie_layers_stride`

This keeps older checkpoints loadable without changing their behavior.

### 2. Weight tying must be re-applied after `load_state_dict`

This implementation uses:

```python
model.load_state_dict(..., assign=True)
```

That call can replace parameter objects in a way that breaks the aliasing established during
construction.

Because of that, the code explicitly calls `_apply_layer_weight_tie()` again after loading state for:

- checkpoint restore in base training
- checkpoint-based model construction in `checkpoint_manager.py`

Without this step, a model that was supposed to have tied weights could silently become untied after load.

## Gotchas

### 1. `to_empty()` and initialization can break aliasing

The model is first built on the meta device, then materialized with `to_empty()`, then initialized.

In practice, these steps can disturb parameter aliasing. That is why `init_weights()` re-applies the tie
operation after initialization.

This is not cosmetic. The tests caught this failure mode directly.

### 2. Sharing modules is not the same as sharing parameters

Reusing one `Block` object for several layers would look simpler, but it is wrong here because:

- attention stores `layer_idx`
- inference KV cache is indexed by layer
- some layer features are conditional on logical layer number

The safe approach is:

- preserve one `Block` per logical layer
- share only the selected `Parameter` objects

### 3. Parameter counts change in the expected way

Because tied parameters are literally aliased, `model.parameters()` only counts unique parameter objects.
That means:

- optimizer groups naturally operate on unique shared parameters
- scaling parameter counts reflect the reduced parameterization
- FLOP estimates still use the full logical depth because compute still executes every logical layer

This distinction is important:

- **parameter count goes down**
- **forward compute does not collapse to the smaller number of unique blocks**

Every logical layer still runs.

### 4. The implementation ties contiguous groups, not periodic layers

This is easy to misunderstand.

The current code means:

- group size `4` => `4,5,6,7` tied together

It does **not** mean:

- `4,8,12,16` tied together

If someone wants periodic tying instead, the grouping logic in `_apply_layer_weight_tie()` would need to be
changed.

### 5. Resume compatibility depends on config consistency

If a user resumes from a checkpoint, the model config being reconstructed must match the checkpointed model.

In practice, that means the tie config should be treated as part of model architecture, not as a casual
runtime toggle.

If someone later adds more resume-path validation, the tie config should be included in that mental model.

## Testing

Tests were added in [tests/test_layer_tying.py](/home/lintaoz/aiwork/nanochat/tests/test_layer_tying.py).

They verify:

- tied layers in the same group share the same parameter objects
- different groups do not share with each other
- layer-local metadata such as `layer_idx` remains distinct
- invalid configs fail early
- partially specified tie configs fail early

Existing engine tests were also re-run to make sure this change did not disturb inference-side machinery.

## Example Mental Model

Suppose:

- `depth = 12`
- tied range = `[2, 10)`
- `stride = 4`

Then the logical stack still has 12 layers:

- `0`
- `1`
- `2`
- `3`
- `4`
- `5`
- `6`
- `7`
- `8`
- `9`
- `10`
- `11`

But the shared-weight groups are:

- group A: `2, 3, 4, 5`
- group B: `6, 7, 8, 9`

The execution still runs 12 layers in sequence.

The difference is that:

- layers `2-5` reuse the same core block matrices
- layers `6-9` reuse another shared set of core block matrices

Each logical layer still has:

- its own position in the stack
- its own residual scalar entries
- its own cache identity
- its own layer-dependent behavior where applicable

## If You Extend This Later

If someone continues this work, the first questions to answer should be:

- should `ve_gate` also be tied?
- should `value_embeds` also be tied?
- should grouped tying be exposed to SFT / RL training as well?
- should periodic tying be supported in addition to contiguous grouping?
- should the optimizer or logging surface explicitly report tie groups?

If any of those are implemented, update this document with:

- exact semantics
- checkpoint compatibility implications
- whether old checkpoints can still be loaded unchanged

## Summary

This branch adds a controlled, optional parameter-sharing mechanism for base pretraining.

The implementation is intentionally conservative:

- preserve logical layer structure
- preserve inference correctness
- share only selected matrix weights
- re-apply tying after init and after load
- reject malformed configs early

That keeps the default model behavior unchanged while making grouped middle-layer sharing available for
experiments.
