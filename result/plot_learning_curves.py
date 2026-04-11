#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"step\s+(\d+)/(\d+).*?\|\s+loss:\s+([0-9]*\.?[0-9]+)")


@dataclass
class RunData:
    name: str
    path: Path
    steps: list[int]
    losses: list[float]
    total_steps: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning-curve comparisons for all training logs in a directory."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing .log files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs. Defaults to the input directory.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the input directory name.",
    )
    parser.add_argument(
        "--tail-steps",
        type=int,
        default=300,
        help="Number of final steps to show in the zoomed-in plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution.",
    )
    return parser.parse_args()


def prettify_name(stem: str) -> str:
    return stem.replace("_defaulttokens", "").replace("_", " ")


def load_run(path: Path) -> RunData | None:
    steps: list[int] = []
    losses: list[float] = []
    total_steps: int | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            total_steps = int(match.group(2))
            loss = float(match.group(3))
            steps.append(step)
            losses.append(loss)

    if not steps:
        return None

    return RunData(
        name=prettify_name(path.stem),
        path=path,
        steps=steps,
        losses=losses,
        total_steps=total_steps,
    )


def discover_runs(input_dir: Path) -> list[RunData]:
    runs: list[RunData] = []
    for path in sorted(input_dir.glob("*.log")):
        run = load_run(path)
        if run is not None:
            runs.append(run)
    return runs


def style_axis(ax: plt.Axes, title: str, xmin: int, xmax: int) -> None:
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_xlim(xmin, xmax)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)


def add_summary(ax: plt.Axes, run: RunData) -> None:
    final_step = run.steps[-1]
    final_loss = run.losses[-1]
    best_loss = min(run.losses)
    label = f"{run.name}  final={final_loss:.4f}  best={best_loss:.4f}"
    ax.plot(run.steps, run.losses, linewidth=1.8, label=label)
    ax.scatter([final_step], [final_loss], s=14)


def plot_runs(runs: list[RunData], output_path: Path, title: str, tail_steps: int | None, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for run in runs:
        add_summary(ax, run)

    max_step = max(run.steps[-1] for run in runs)
    if tail_steps is None:
        xmin = 0
    else:
        xmin = max(0, max_step - tail_steps)
    style_axis(ax, title, xmin, max_step)

    if tail_steps is not None:
        tail_losses = [
            loss
            for run in runs
            for step, loss in zip(run.steps, run.losses)
            if step >= xmin
        ]
        if tail_losses:
            padding = max(0.02, (max(tail_losses) - min(tail_losses)) * 0.12)
            ax.set_ylim(min(tail_losses) - padding, max(tail_losses) + padding)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir).expanduser().resolve()
    prefix = args.prefix or input_dir.name

    if not input_dir.is_dir():
        print(f"error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    runs = discover_runs(input_dir)
    if not runs:
        print(f"error: no parseable .log files found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"{prefix}_learning_curves.png"
    tail_path = output_dir / f"{prefix}_learning_curves_tail.png"

    plot_runs(
        runs=runs,
        output_path=full_path,
        title=f"Learning Curves: {input_dir.name}",
        tail_steps=None,
        dpi=args.dpi,
    )
    plot_runs(
        runs=runs,
        output_path=tail_path,
        title=f"Learning Curves Tail ({args.tail_steps} steps): {input_dir.name}",
        tail_steps=args.tail_steps,
        dpi=args.dpi,
    )

    print(f"Wrote {full_path}")
    print(f"Wrote {tail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
