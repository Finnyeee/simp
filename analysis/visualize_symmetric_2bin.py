#!/usr/bin/env python3
"""
Visualize symmetric 2-bin memory results.

Outputs:
- One profit bar chart over thresholds:  figures/symmetric_2bin_profits.png
- One BR state-report figure per threshold:
    figures/symmetric_2bin_th{k}_br.png

Usage example (from repo root):
    python analysis/visualize_symmetric_2bin.py \
        --data-dir data_full_memory \
        --config simulations/configs/symmetric_2bin.json \
        --output-dir figures \
        --prefix data_multithreaded_symmetric_2bin_
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from visualize_batches import (
    translate_memories,
    constructor,
    profit_computer_with_errors,
    plot_br_state_report,
)


def load_symmetric_data(data_dir: Path, config_path: Path, prefix: str):
    """Load 14 symmetric 2-bin configs and their data."""
    raw = json.load(open(config_path))
    config = json.loads(raw) if isinstance(raw, str) else raw
    memories = translate_memories(config)

    items = []  # (idx, threshold_k, mem_cfg, mem_dict, data_or_None)
    for idx, mem_cfg in enumerate(memories, start=1):
        mem_dict = constructor(mem_cfg)
        # symmetric: take P1's own-threshold first element as label
        k = mem_cfg[1][0][0]

        data_path = data_dir / f"{prefix}{idx}.json"
        if not data_path.exists():
            items.append((idx, k, mem_cfg, mem_dict, None))
            continue

        with open(data_path) as fp:
            data_raw = json.load(fp)
        data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
        items.append((idx, k, mem_cfg, mem_dict, data))

    return items


def plot_profits(items, output_dir: Path):
    """One bar chart: profit vs threshold k."""
    labels, means, errs = [], [], []
    for idx, k, mem_cfg, mem_dict, data in items:
        labels.append(str(k))
        if data is None:
            means.append(np.nan)
            errs.append(0.0)
        else:
            m, e = profit_computer_with_errors(mem_dict, data, len(data))
            means.append(m)
            errs.append(e)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=4, color="#FF8000", edgecolor="darkorange")
    ax.errorbar(x, means, yerr=errs, fmt="o", color="r")
    ax.axhline(0.25, color="gray", linestyle="--", label="Competitive (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Threshold index k (2-bin symmetric memory)")
    ax.set_ylabel("Profit")
    ax.set_title("Profits for symmetric_2bin (thresholds 1–14)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = output_dir / "symmetric_2bin_profits.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_best_responses(items, output_dir: Path):
    """One BR state-report figure per threshold with data."""
    for idx, k, mem_cfg, mem_dict, data in items:
        if data is None:
            continue
        label = f"th={k}"
        out = plot_br_state_report(
            batch_id="symmetric_2bin",
            mem_idx=idx - 1,
            label=label,
            mem_cfg=mem_cfg,
            mem_dict=mem_dict,
            data=data,
            output_dir=output_dir,
            prefix="symmetric_2bin_",
            trial=0,  # use first trial
        )
        print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize symmetric 2-bin results: profits + best responses"
    )
    parser.add_argument(
        "--data-dir", default="data_full_memory", help="Directory with data JSONs"
    )
    parser.add_argument(
        "--config",
        default="simulations/configs/symmetric_2bin.json",
        help="Config JSON for symmetric_2bin",
    )
    parser.add_argument(
        "--output-dir", default="figures", help="Output directory for figures"
    )
    parser.add_argument(
        "--prefix",
        default="data_multithreaded_symmetric_2bin_",
        help="Data file prefix (before 1..14).",
    )
    args = parser.parse_args()

    base = Path(".").resolve()
    data_dir = base / args.data_dir
    config_path = base / args.config
    output_dir = base / args.output_dir
    prefix = args.prefix

    items = load_symmetric_data(data_dir, config_path, prefix)
    if not items:
        print("No symmetric_2bin configs or data found.")
        return

    plot_profits(items, output_dir)
    plot_best_responses(items, output_dir)


if __name__ == "__main__":
    main()