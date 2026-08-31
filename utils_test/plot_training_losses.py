#!/usr/bin/env python3
"""Plot OSVI training and validation losses from a JSONL training log."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_events(log_path):
    events = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {log_path}:{line_no}: {exc}") from exc
    return events


def collect_series(events):
    train_steps = []
    train_losses = []
    small_val_steps = []
    small_val_losses = []
    epoch_val_steps = []
    epoch_val_losses = []

    for event in events:
        event_type = event.get("event")
        global_step = event.get("global_step")
        if global_step is None:
            continue

        if event_type == "train_step" and "loss" in event:
            train_steps.append(global_step)
            train_losses.append(event["loss"])
        elif event_type == "small_val" and "val_loss" in event:
            small_val_steps.append(global_step)
            small_val_losses.append(event["val_loss"])
        elif event_type == "epoch" and "val_loss" in event:
            epoch_val_steps.append(global_step)
            epoch_val_losses.append(event["val_loss"])

    return {
        "train": (train_steps, train_losses),
        "small_val": (small_val_steps, small_val_losses),
        "epoch_val": (epoch_val_steps, epoch_val_losses),
    }


def plot_losses(series, output_path, title):
    train_steps, train_losses = series["train"]
    small_val_steps, small_val_losses = series["small_val"]
    epoch_val_steps, epoch_val_losses = series["epoch_val"]

    if not train_steps:
        raise ValueError("No train_step/loss entries found in the log.")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(train_steps, train_losses, color="#1f77b4", linewidth=1.6, label="train loss")

    if small_val_steps:
        ax.plot(
            small_val_steps,
            small_val_losses,
            color="#ff7f0e",
            linewidth=1.4,
            marker="o",
            markersize=3,
            label="small val loss",
        )

    if epoch_val_steps:
        ax.plot(
            epoch_val_steps,
            epoch_val_losses,
            color="#d62728",
            linewidth=2.0,
            marker="s",
            markersize=5,
            label="epoch val loss",
        )
        best_idx = min(range(len(epoch_val_losses)), key=epoch_val_losses.__getitem__)
        best_step = epoch_val_steps[best_idx]
        best_loss = epoch_val_losses[best_idx]
        ax.scatter([best_step], [best_loss], color="#2ca02c", s=70, zorder=5, label="best epoch val")
        ax.annotate(
            f"best: step {best_step}\nloss {best_loss:.4f}",
            xy=(best_step, best_loss),
            xytext=(10, 16),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#2ca02c", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
        )

    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def default_output_path(log_path):
    return log_path.with_name(f"{log_path.stem}_loss_curve.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot OSVI train/validation loss curves from a JSONL log."
    )
    parser.add_argument("log_path", type=Path, help="Path to the JSONL training log.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <log_stem>_loss_curve.png next to the log.",
    )
    parser.add_argument(
        "--title",
        default="OSVI training and validation loss",
        help="Plot title.",
    )
    args = parser.parse_args()

    log_path = args.log_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else default_output_path(log_path)

    events = read_events(log_path)
    series = collect_series(events)
    plot_losses(series, output_path, args.title)

    train_count = len(series["train"][0])
    small_val_count = len(series["small_val"][0])
    epoch_val_count = len(series["epoch_val"][0])
    print(f"Saved loss plot to: {output_path}")
    print(f"Points: train={train_count}, small_val={small_val_count}, epoch_val={epoch_val_count}")


if __name__ == "__main__":
    main()
