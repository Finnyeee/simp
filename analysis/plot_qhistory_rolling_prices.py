#!/usr/bin/env python3
"""
Plot rolling average of optimal (best-response) prices over the last 5000 iterations
from trimmed Q-history files (qhistory_*_trimmed.json).

For each snapshot in Q_history we compute, per player, the mean optimal price across
all states (argmax of Q-values -> price grid). Then we plot rolling average over time.

Usage:
    python analysis/plot_qhistory_rolling_prices.py [--data-dir data_full_memory] [--output-dir figures] [--window 50]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PRICE_GRID = np.array([1.43 + i * (1.97 - 1.43) / 14 for i in range(15)])


def load_trimmed(path):
    with open(path) as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(json.loads(raw))  # double-encoded
    return data


def optimal_prices_per_snapshot(snapshot):
    """For one Q snapshot: return (mean optimal price P1, mean optimal price P2)."""
    means = {}
    for player_id, state_dict in snapshot.items():
        prices = []
        for state_key, qvec in state_dict.items():
            qvec = np.asarray(qvec)
            best_idx = int(np.argmax(qvec))  # 0-based
            prices.append(PRICE_GRID[best_idx])
        means[player_id] = np.mean(prices) if prices else np.nan
    return means.get("1", np.nan), means.get("2", np.nan)


def rolling_mean(x, window):
    if window >= len(x):
        return np.full_like(x, np.nanmean(x))
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.nanmean(x[start : i + 1])
    return out


def plot_one_file(data_path, output_dir, window=50):
    data = load_trimmed(data_path)
    qhistory = data.get("Q_history", [])
    if not qhistory:
        print(f"No Q_history in {data_path}")
        return None

    n = len(qhistory)
    p1 = np.zeros(n)
    p2 = np.zeros(n)
    for t, snap in enumerate(qhistory):
        a1, a2 = optimal_prices_per_snapshot(snap)
        p1[t] = a1
        p2[t] = a2

    p1_roll = rolling_mean(p1, window)
    p2_roll = rolling_mean(p2, window)

    fig, ax = plt.subplots(figsize=(10, 5))
    iterations = np.arange(1, n + 1)  # period index in the trimmed range
    ax.plot(iterations, p1_roll, label="P1 (rolling)", color="C0")
    ax.plot(iterations, p2_roll, label="P2 (rolling)", color="C1")
    ax.axhline(1.43, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(1.97, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Period (last 5000 iterations)")
    ax.set_ylabel("Rolling average optimal price")
    ax.set_title(f"{data_path.name} (window={window})")
    ax.legend()
    ax.set_ylim(1.3, 2.05)
    fig.tight_layout()
    out_path = output_dir / f"{data_path.stem.replace('_trimmed', '')}_rolling_prices.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot rolling average optimal prices from trimmed Q-history")
    parser.add_argument("--data-dir", default="data_full_memory", help="Directory containing *_trimmed.json")
    parser.add_argument("--output-dir", default="figures", help="Where to save plots")
    parser.add_argument("--window", type=int, default=50, help="Rolling average window size")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / args.data_dir
    output_dir = base / args.output_dir

    files = sorted(data_dir.glob("*_trimmed.json"))
    if not files:
        print(f"No *_trimmed.json files in {data_dir}")
        return

    for path in files:
        out = plot_one_file(path, output_dir, window=args.window)
        if out:
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
