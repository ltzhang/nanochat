import pytest
import torch
import tempfile
from pathlib import Path

from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint


def build_model(**kwargs):
    config_kwargs = dict(
        sequence_len=32,
        vocab_size=128,
        n_layer=8,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="L",
    )
    config_kwargs.update(kwargs)
    config = GPTConfig(**config_kwargs)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def test_tied_layers_share_group_weights():
    model = build_model(tie_layers_start=2, tie_layers_end=6, tie_layers_stride=2)
    layers = model.transformer.h

    assert layers[2].attn.c_q.weight is layers[3].attn.c_q.weight
    assert layers[2].mlp.c_fc.weight is layers[3].mlp.c_fc.weight
    assert layers[4].attn.c_q.weight is layers[5].attn.c_q.weight
    assert layers[4].mlp.c_proj.weight is layers[5].mlp.c_proj.weight

    assert layers[1].attn.c_q.weight is not layers[2].attn.c_q.weight
    assert layers[3].attn.c_q.weight is not layers[4].attn.c_q.weight
    assert layers[5].mlp.c_fc.weight is not layers[6].mlp.c_fc.weight


def test_tied_layers_keep_layer_local_attention_metadata():
    model = build_model(tie_layers_start=2, tie_layers_end=6, tie_layers_stride=2)
    layers = model.transformer.h

    assert layers[2].attn.layer_idx == 2
    assert layers[3].attn.layer_idx == 3
    assert layers[2].attn is not layers[3].attn


def test_invalid_tied_layer_span_raises():
    with pytest.raises(AssertionError, match="divisible"):
        build_model(tie_layers_start=1, tie_layers_end=6, tie_layers_stride=2)


def test_partial_tie_args_must_be_complete():
    with pytest.raises(AssertionError, match="must all be set together"):
        build_model(tie_layers_start=2)


def test_stride_equal_to_range_is_single_tied_group():
    model = build_model(tie_layers_start=2, tie_layers_end=6, tie_layers_stride=4)
    layers = model.transformer.h

    assert layers[2].attn.c_q.weight is layers[3].attn.c_q.weight
    assert layers[2].attn.c_q.weight is layers[4].attn.c_q.weight
    assert layers[2].attn.c_q.weight is layers[5].attn.c_q.weight


def test_stride_one_produces_no_sharing():
    model = build_model(tie_layers_start=2, tie_layers_end=6, tie_layers_stride=1)
    layers = model.transformer.h

    assert layers[2].attn.c_q.weight is not layers[3].attn.c_q.weight
    assert layers[3].attn.c_q.weight is not layers[4].attn.c_q.weight
    assert layers[4].attn.c_q.weight is not layers[5].attn.c_q.weight


def test_estimate_flops_counts_logical_tied_layers():
    untied = build_model()
    tied = build_model(tie_layers_start=2, tie_layers_end=6, tie_layers_stride=2)

    assert tied.num_scaling_params()["transformer_matrices"] < untied.num_scaling_params()["transformer_matrices"]
    assert tied.estimate_flops() == untied.estimate_flops()


def test_load_state_dict_requires_retie_after_assign():
    src = build_model(tie_layers_start=2, tie_layers_end=6, tie_layers_stride=2)
    dst = build_model(tie_layers_start=2, tie_layers_end=6, tie_layers_stride=2)

    assert dst.transformer.h[2].attn.c_q.weight is dst.transformer.h[3].attn.c_q.weight

    dst.load_state_dict(src.state_dict(), strict=True, assign=True)
    assert dst.transformer.h[2].attn.c_q.weight is not dst.transformer.h[3].attn.c_q.weight

    dst._apply_layer_weight_tie()
    assert dst.transformer.h[2].attn.c_q.weight is dst.transformer.h[3].attn.c_q.weight
    assert torch.equal(dst.transformer.h[2].attn.c_q.weight, src.transformer.h[2].attn.c_q.weight)


def test_checkpoint_round_trip_restores_aliasing():
    model = build_model(tie_layers_start=2, tie_layers_end=6, tie_layers_stride=2)
    model_config = model.config.__dict__.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        save_checkpoint(
            str(checkpoint_dir),
            7,
            model.state_dict(),
            optimizer_data=None,
            meta_data={"model_config": model_config},
            rank=0,
        )
        model_data, _, meta_data = load_checkpoint(str(checkpoint_dir), 7, device="cpu", load_optimizer=False)
        restored = build_model(**meta_data["model_config"])
        restored.load_state_dict(model_data, strict=True, assign=True)
        restored._apply_layer_weight_tie()

    assert restored.transformer.h[2].attn.c_q.weight is restored.transformer.h[3].attn.c_q.weight
    assert torch.equal(restored.transformer.h[2].attn.c_q.weight, model.transformer.h[2].attn.c_q.weight)
