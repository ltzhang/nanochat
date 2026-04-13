import torch

import nanochat.dataloader as dataloader_mod
from nanochat.checkpoint_manager import _patch_missing_keys
from nanochat.dataloader import aligned_binary_ngram_data_loader_with_state
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.ngram import NgramLexicon


def test_ngram_lexicon_longest_match_and_encoding(tmp_path):
    lexicon_path = tmp_path / "ngrams.txt"
    lexicon_path.write_text(
        "\n".join(
            [
                "# id then token sequence",
                "1\t10 20",
                "2\t3\t10 20 30",
                "3\t20 30",
            ]
        ),
        encoding="utf-8",
    )

    lexicon = NgramLexicon.from_file(lexicon_path)

    assert lexicon.longest_suffix_id([10, 20]) == 1
    assert lexicon.longest_suffix_id([5, 10, 20, 30]) == 2
    assert lexicon.encode_sequence([5, 10, 20, 30]) == [0, 0, 1, 2]

    tensor_ids = lexicon.encode_tensor(torch.tensor([[5, 10, 20, 30]], dtype=torch.long))
    assert tensor_ids.tolist() == [[0, 0, 1, 2]]


def test_aligned_binary_ngram_loader_reads_matching_streams(tmp_path):
    token_path = tmp_path / "shard0.bin"
    ngram_path = tmp_path / "shard0.ngram.bin"

    torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32).numpy().tofile(token_path)
    torch.tensor([0, 0, 1, 2, 3, 4, 5], dtype=torch.int32).numpy().tofile(ngram_path)

    loader = aligned_binary_ngram_data_loader_with_state(
        B=2,
        T=3,
        split="train",
        bin_dir=str(tmp_path),
        device="cpu",
    )

    inputs, ngram_ids, targets, state = next(loader)

    assert inputs.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert ngram_ids.tolist() == [[0, 0, 1], [2, 3, 4]]
    assert targets.tolist() == [[2, 3, 4], [5, 6, 7]]
    assert state == {"shard_idx": 0, "row_idx": 2, "epoch": 1}


def test_aligned_binary_ngram_loader_supports_separate_output_devices(tmp_path):
    token_path = tmp_path / "shard0.bin"
    ngram_path = tmp_path / "shard0.ngram.bin"

    torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).numpy().tofile(token_path)
    torch.tensor([0, 7, 8, 9, 10], dtype=torch.int32).numpy().tofile(ngram_path)

    loader = aligned_binary_ngram_data_loader_with_state(
        B=1,
        T=4,
        split="train",
        bin_dir=str(tmp_path),
        target_device="cpu",
        token_device="cpu",
        ngram_device="cpu",
    )

    inputs, ngram_ids, targets, _ = next(loader)

    assert inputs.device.type == "cpu"
    assert ngram_ids.device.type == "cpu"
    assert targets.device.type == "cpu"
    assert inputs.tolist() == [[1, 2, 3, 4]]
    assert ngram_ids.tolist() == [[0, 7, 8, 9]]
    assert targets.tolist() == [[2, 3, 4, 5]]


def test_aligned_binary_ngram_loader_requires_explicit_ngram_device_without_legacy_device(tmp_path):
    token_path = tmp_path / "shard0.bin"
    ngram_path = tmp_path / "shard0.ngram.bin"

    torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).numpy().tofile(token_path)
    torch.tensor([0, 7, 8, 9, 10], dtype=torch.int32).numpy().tofile(ngram_path)

    try:
        next(
            aligned_binary_ngram_data_loader_with_state(
                B=1,
                T=4,
                split="train",
                bin_dir=str(tmp_path),
                target_device="cpu",
                token_device="cpu",
            )
        )
    except ValueError as exc:
        assert "explicit ngram_device" in str(exc)
    else:
        raise AssertionError("Expected missing ngram_device to raise")


def test_aligned_binary_ngram_loader_offsets_resume_row_by_rank(tmp_path):
    token_path = tmp_path / "shard0.bin"
    ngram_path = tmp_path / "shard0.ngram.bin"

    torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int32).numpy().tofile(token_path)
    torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=torch.int32).numpy().tofile(ngram_path)

    original_get_dist_info = dataloader_mod.get_dist_info
    dataloader_mod.get_dist_info = lambda: (True, 1, 1, 2)
    try:
        loader = aligned_binary_ngram_data_loader_with_state(
            B=1,
            T=2,
            split="train",
            bin_dir=str(tmp_path),
            device="cpu",
            resume_state_dict={"shard_idx": 0, "row_idx": 2, "epoch": 1},
        )
        inputs, ngram_ids, targets, state = next(loader)
    finally:
        dataloader_mod.get_dist_info = original_get_dist_info

    assert inputs.tolist() == [[7, 8]]
    assert ngram_ids.tolist() == [[3, 3]]
    assert targets.tolist() == [[8, 9]]
    assert state == {"shard_idx": 0, "row_idx": 5, "epoch": 1}


def test_gpt_forward_accepts_ngram_ids():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        ngram_vocab_size=8,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        use_ngram_embeds=True,
    )
    model = GPT(config)
    model.init_weights()
    model.place_embedding_modules("cpu")

    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ngram_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

    logits = model(idx, ngram_ids=ngram_ids)
    loss = model(idx, targets=targets, ngram_ids=ngram_ids)

    assert logits.shape == (1, 4, 32)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gpt_forward_accepts_cpu_ngram_ids_with_gpu_compute():
    if not torch.cuda.is_available():
        return

    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        ngram_vocab_size=8,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        use_ngram_embeds=True,
        token_embed_device="compute",
        ngram_embed_device="cpu",
    )
    model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    model.place_embedding_modules("cuda")

    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")
    ngram_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long, device="cpu")
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long, device="cuda")

    loss = model(idx, targets=targets, ngram_ids=ngram_ids)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ngram_model_requires_lexicon_for_dynamic_lookup():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        ngram_vocab_size=8,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        use_ngram_embeds=True,
    )
    model = GPT(config)
    model.init_weights()
    model.place_embedding_modules("cpu")

    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    try:
        model.encode_ngram_ids(idx)
    except RuntimeError as exc:
        assert "requires a loaded n-gram lexicon" in str(exc)
    else:
        raise AssertionError("Expected missing lexicon to raise")


def test_patch_missing_keys_rejects_non_ngram_checkpoint_upgrade():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        ngram_vocab_size=8,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        use_ngram_embeds=True,
    )

    try:
        _patch_missing_keys({}, config)
    except ValueError as exc:
        assert "Upgrading a non-n-gram checkpoint" in str(exc)
        assert "ngram_embeds.weight" in str(exc)
    else:
        raise AssertionError("Expected unsupported checkpoint upgrade to raise")


def test_evaluate_bpb_dynamic_ngram_lookup_from_lexicon(tmp_path):
    lexicon_path = tmp_path / "ngrams.txt"
    lexicon_path.write_text(
        "\n".join(
            [
                "1\t10 20",
                "2\t20 30",
            ]
        ),
        encoding="utf-8",
    )

    config = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        ngram_vocab_size=8,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        use_ngram_embeds=True,
    )
    model = GPT(config)
    model.init_weights()
    model.place_embedding_modules("cpu")
    model.load_ngram_lexicon(lexicon_path)

    x = torch.tensor([[10, 20, 30, 4]], dtype=torch.long)
    y = torch.tensor([[20, 30, 4, 5]], dtype=torch.long)
    token_bytes = torch.ones(config.vocab_size, dtype=torch.int64)

    bpb = evaluate_bpb(model, iter([(x, y)]), steps=1, token_bytes=token_bytes)

    assert isinstance(bpb, float)
    assert bpb == bpb


def test_gpt_generate_uses_ngram_embedding_device(tmp_path):
    lexicon_path = tmp_path / "ngrams.txt"
    lexicon_path.write_text("1\t1 2\n", encoding="utf-8")

    config = GPTConfig(
        sequence_len=8,
        vocab_size=16,
        ngram_vocab_size=4,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        use_ngram_embeds=True,
        ngram_embed_device="cpu",
    )
    model = GPT(config)
    model.init_weights()
    model.place_embedding_modules("cpu")
    model.load_ngram_lexicon(lexicon_path)

    seen_devices = []
    original_encode = model.encode_ngram_ids

    def wrapped_encode(token_ids, device=None):
        seen_devices.append(torch.device(device).type if device is not None else token_ids.device.type)
        return original_encode(token_ids, device=device)

    model.encode_ngram_ids = wrapped_encode
    try:
        token = next(model.generate([1, 2], max_tokens=1, temperature=0.0))
    finally:
        model.encode_ngram_ids = original_encode

    assert isinstance(token, int)
    assert seen_devices == ["cpu"]


def test_gpt_generate_accepts_single_token_prompt():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=16,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    model.place_embedding_modules("cpu")

    token = next(model.generate([1], max_tokens=1, temperature=0.0))

    assert isinstance(token, int)
