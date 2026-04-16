"""
Compare learning curves for three variants on identical real-data batches:

1. ngram embeddings on GPU
2. ngram embeddings on CPU
3. no ngram embeddings

The batch stream comes from the aligned binary shard directory in the default
nanochat cache, so all variants see the same token targets. The ngram-free
variant ignores the ngram IDs but otherwise uses the same token batches.
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from nanochat.common import get_base_dir, COMPUTE_DTYPE, print0
from nanochat.dataloader import aligned_binary_ngram_data_loader_with_state
from nanochat.gpt import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ngram learning curves on real data")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-pattern", type=str, default="L")
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--ngram-vocab-size", type=int, default=5000)
    parser.add_argument(
        "--ngram-lexicon-path",
        type=str,
        default="/home/lintaoz/aiwork/nanochat/ngram/output/climbmix_2to6_pre10_final100/ngram_vocab_top1000_each.tsv",
    )
    parser.add_argument(
        "--bin-dir",
        type=str,
        default=str(Path(get_base_dir()) / "base_data_climbmix"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/tmp/ngram_learning_curve_compare",
    )
    return parser.parse_args()


def build_config(args, use_ngram_embeds, ngram_embed_device):
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    return GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=32768,
        ngram_vocab_size=args.ngram_vocab_size if use_ngram_embeds else 0,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
        use_ngram_embeds=use_ngram_embeds,
        token_embed_device="compute",
        ngram_embed_device=ngram_embed_device,
    )


def get_lr_multiplier(step, num_steps, warmup_steps, warmdown_ratio, final_lr_frac):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    warmdown_iters = round(warmdown_ratio * num_steps)
    if warmdown_iters <= 0:
        return 1.0
    if step <= num_steps - warmdown_iters:
        return 1.0
    progress = (num_steps - step) / warmdown_iters
    return final_lr_frac + (1 - final_lr_frac) * 0.5 * (1 + math.cos(math.pi * (1 - progress)))


def get_weight_decay(step, num_steps, weight_decay):
    return weight_decay * 0.5 * (1 + math.cos(math.pi * step / num_steps))


def preload_batches(args):
    loader = aligned_binary_ngram_data_loader_with_state(
        B=args.device_batch_size,
        T=args.max_seq_len,
        split="train",
        bin_dir=args.bin_dir,
        target_device="cpu",
        token_device="cpu",
        ngram_device="cpu",
    )
    batches = []
    for _ in range(args.steps):
        x, ng, y, _ = next(loader)
        batches.append((x.clone(), ng.clone(), y.clone()))
    return batches


def make_model(config, device, lexicon_path=None):
    model = GPT(config)
    model.materialize_on_final_devices(device)
    model.init_weights()
    if config.use_ngram_embeds:
        model.load_ngram_lexicon(lexicon_path)
    return model


def optimizer_kwargs(args):
    batch_scale = math.sqrt((args.device_batch_size * args.max_seq_len) / (2**19))
    return dict(
        unembedding_lr=args.unembedding_lr * batch_scale,
        embedding_lr=args.embedding_lr * batch_scale,
        matrix_lr=args.matrix_lr * batch_scale,
        weight_decay=args.weight_decay,
        scalar_lr=args.scalar_lr * batch_scale,
    )


def train_variant(name, model, optimizer, batches, args, use_ngram, log_path):
    log_lines = [
        f"variant={name}",
        f"steps={args.steps}",
        f"depth={args.depth}",
        f"max_seq_len={args.max_seq_len}",
        f"device_batch_size={args.device_batch_size}",
        f"use_ngram={use_ngram}",
    ]

    def log(msg):
        print0(msg)
        log_lines.append(msg)

    losses = []
    start = time.time()
    for step, (x_cpu, ng_cpu, y_cpu) in enumerate(batches):
        x = x_cpu.to("cuda", non_blocking=False)
        y = y_cpu.to("cuda", non_blocking=False)
        ng = ng_cpu.to(model.ngram_embeds.weight.device, non_blocking=False) if use_ngram else None

        lrm = get_lr_multiplier(step, args.steps, args.warmup_steps, args.warmdown_ratio, args.final_lr_frac)
        wd = get_weight_decay(step, args.steps, args.weight_decay)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["weight_decay"] = wd

        optimizer.zero_grad(set_to_none=True)
        loss = model(x, targets=y, ngram_ids=ng)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))

        if step in {0, 9, 99, args.steps - 1}:
            elapsed = time.time() - start
            log(f"{name}: step {step+1}/{args.steps} loss={losses[-1]:.6f} elapsed={elapsed:.1f}s")
    elapsed = time.time() - start
    log(f"{name}: total_elapsed={elapsed:.3f}s seconds_per_step={elapsed / args.steps:.6f}")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
    return losses, elapsed


def write_csv(out_path, curves):
    steps = len(next(iter(curves.values())))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + list(curves.keys()))
        for i in range(steps):
            writer.writerow([i + 1] + [curves[name][i] for name in curves])


def write_plot(out_path, curves, title):
    plt.figure(figsize=(10, 6))
    for name, losses in curves.items():
        plt.plot(range(1, len(losses) + 1), losses, label=name, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print0(f"COMPUTE_DTYPE={COMPUTE_DTYPE}")
    print0(f"Preloading {args.steps} batches from {args.bin_dir}")
    batches = preload_batches(args)

    gpu_ngram_config = build_config(args, use_ngram_embeds=True, ngram_embed_device="compute")
    cpu_ngram_config = build_config(args, use_ngram_embeds=True, ngram_embed_device="cpu")
    no_ngram_config = build_config(args, use_ngram_embeds=False, ngram_embed_device="compute")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    gpu_ngram = make_model(gpu_ngram_config, "cuda", args.ngram_lexicon_path)

    cpu_ngram = make_model(cpu_ngram_config, "cuda", args.ngram_lexicon_path)
    cpu_ngram.load_state_dict(gpu_ngram.state_dict())

    no_ngram = make_model(no_ngram_config, "cuda")
    missing, unexpected = no_ngram.load_state_dict(gpu_ngram.state_dict(), strict=False)
    allowed_missing = sorted([k for k in no_ngram.state_dict().keys() if "nge_gate" in k])
    assert sorted(missing) == allowed_missing, (missing, allowed_missing)
    assert sorted(unexpected) == sorted(
        [k for k in gpu_ngram.state_dict().keys() if k == "ngram_embeds.weight" or "nge_gate" in k]
    ), unexpected

    opt_kwargs = optimizer_kwargs(args)
    gpu_ngram_optim = gpu_ngram.setup_optimizer(**opt_kwargs)
    cpu_ngram_optim = cpu_ngram.setup_optimizer(**opt_kwargs)
    no_ngram_optim = no_ngram.setup_optimizer(**opt_kwargs)

    curves = {}
    timings = {}
    for name, model, optimizer, use_ngram in [
        ("ngram_gpu", gpu_ngram, gpu_ngram_optim, True),
        ("ngram_cpu", cpu_ngram, cpu_ngram_optim, True),
        ("no_ngram_gpu", no_ngram, no_ngram_optim, False),
    ]:
        log_path = Path(args.out_dir) / f"{name}.log"
        losses, elapsed = train_variant(name, model, optimizer, batches, args, use_ngram, log_path)
        curves[name] = losses
        timings[name] = elapsed

    csv_path = Path(args.out_dir) / "learning_curves.csv"
    plot_path = Path(args.out_dir) / "learning_curves.png"
    summary_path = Path(args.out_dir) / "summary.txt"
    write_csv(csv_path, curves)
    write_plot(
        plot_path,
        curves,
        title=f"Ngram Comparison on Real Data ({args.steps} steps, depth={args.depth}, seq={args.max_seq_len}, batch={args.device_batch_size})",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        for name, elapsed in timings.items():
            f.write(f"{name}\t{elapsed:.3f}\n")
    print0(f"Wrote {csv_path}")
    print0(f"Wrote {plot_path}")
    print0(f"Wrote {summary_path}")
    for name, elapsed in timings.items():
        print0(f"{name}: {elapsed:.1f}s total, {elapsed / args.steps:.4f}s/step")


if __name__ == "__main__":
    main()
