#!/usr/bin/env python3
"""
Visualization tool for batch simulation results.

Two outputs:
1. Profit grinds — bar chart of mean profit for the 4 memory types in each batch
2. Best-response matrices — heatmaps of learned best responses per batch

Usage:
    python visualize_batches.py [--data-dir data_full_memory] [--config-dir simulations/configs] [--output-dir figures]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PRICE_GRID = np.array([1.43 + i * (1.97 - 1.43) / 14 for i in range(15)])


def translate_memories(temp):
    memories = []
    for i in range(len(temp)):
        inner_dict = {}
        for k, v in temp[i].items():
            int_key = int(str(k))
            new_value = [[int(x) for x in v[0]], [int(y) for y in v[1]]]
            inner_dict[int_key] = new_value
        memories.append(inner_dict)
    return memories


def _n_bins(threshold_list):
    if not threshold_list or (len(threshold_list) >= 15 and 15 in threshold_list):
        return 15
    return len(threshold_list) + 1


def memory_label(memory_config):
    parts = []
    for pid in sorted(memory_config.keys()):
        own, opp = memory_config[pid]
        n_own = _n_bins(own)
        n_opp = _n_bins(opp)
        parts.append(f"P{pid}:{n_own}×{n_opp}")
    return ", ".join(parts)


def __pooling(action, thresholds):
    if not thresholds:
        return action
    bin_ = 1
    for t in thresholds:
        if action > t:
            bin_ += 1
        else:
            break
    return bin_


def __full_monitoring(last_prices, identity, thresholds=None):
    if thresholds is None or not thresholds.get(identity):
        return np.concatenate([last_prices, [identity]])
    ti = thresholds[identity]
    pooled = [__pooling(p, ti[i]) for i, p in enumerate(last_prices)]
    return np.concatenate([pooled, [identity]])


def constructor(monitoring):
    d = {}
    for i in range(1, 16):
        for j in range(1, 16):
            for identity in [1, 2]:
                d[((i, j), identity)] = __full_monitoring([i, j], identity, thresholds=monitoring)
    return d


def profits(prices):
    mu = np.ones(2) * 0.25
    q = np.ones(2) * 2.0
    costs = np.ones(2) * 1.0
    exp_q = np.exp((q - prices) / mu)
    quantities = exp_q / exp_q.sum()
    return (prices - costs) * quantities


def best_response(memory_dict, data, trial):
    brf = {}
    for (state, repr_arr) in memory_dict.items():
        repr_list = repr_arr.tolist()
        pid = str(int(repr_list[2]))
        state_key = str(repr_list[0:2])
        if pid not in data[trial] or state_key not in data[trial][pid]:
            continue
        qvals = np.array(data[trial][pid][state_key])
        brf[str(repr_list)] = int(np.argmax(qvals)) + 1
    return brf


def follow_the_cycles(memory_dict, data, trial, start=(15, 15)):
    brf = best_response(memory_dict, data, trial)
    if not brf:
        return None, None
    loop = []
    a, b = start[0], start[1]
    while [a, b] not in loop:
        loop.append([a, b])
        r1 = memory_dict[((a, b), 1)].tolist()
        r2 = memory_dict[((a, b), 2)].tolist()
        a = brf.get(str(r1), 15)
        b = brf.get(str(r2), 15)
    loop.append([a, b])
    idx = loop.index([a, b])
    cycle = loop[idx:-1]
    if not cycle:
        cycle = [[a, b]]
    p1 = [x[0] for x in cycle]
    p2 = [x[1] for x in cycle]
    return p1, p2


def profit_computer_with_errors(memory_dict, data, n_trials):
    profit_list = []
    for t in range(n_trials):
        p1, p2 = follow_the_cycles(memory_dict, data, t)
        if p1 is None:
            continue
        profs = []
        for i in range(len(p1)):
            prices = np.array([PRICE_GRID[p1[i] - 1], PRICE_GRID[p2[i] - 1]])
            profs.append(profits(prices))
        avg_profit = (np.mean([x[0] for x in profs]) + np.mean([x[1] for x in profs])) / 2
        profit_list.append(avg_profit)
    if not profit_list:
        return np.nan, np.nan
    mean = np.mean(profit_list)
    ste = np.std(profit_list, ddof=1) / np.sqrt(len(profit_list)) if len(profit_list) > 1 else 0.0
    return mean, ste


def best_response_matrix(mem_cfg, memory_dict, identity, data, trial):
    brf = best_response(memory_dict, data, trial)
    if not brf:
        return None, None, None
    own_bins = mem_cfg[identity][0]
    opp_bins = mem_cfg[identity][1]
    n_own = _n_bins(own_bins)
    n_opp = _n_bins(opp_bins)
    mat = np.full((n_own, n_opp), np.nan)
    for (state, repr_arr) in memory_dict.items():
        r = repr_arr.tolist()
        if int(r[2]) != identity:
            continue
        obin, opp_bin = int(r[0]), int(r[1])
        if 1 <= obin <= n_own and 1 <= opp_bin <= n_opp:
            action = brf.get(str(r), np.nan)
            if action is not None and not (isinstance(action, float) and np.isnan(action)):
                mat[obin - 1, opp_bin - 1] = PRICE_GRID[int(action) - 1]
    row_labs = [f"{i}" for i in range(1, n_own + 1)]
    col_labs = [f"{j}" for j in range(1, n_opp + 1)]
    return mat, row_labs, col_labs


def load_batch_data(data_dir, config_dir, prefix="data_sync"):
    config_dir = Path(config_dir)
    data_dir = Path(data_dir)
    results = []
    for n in range(1, 17):
        batch_id = f"batch_{n:02d}"
        config_path = config_dir / f"{batch_id}.json"
        if not config_path.exists():
            continue
        raw = json.load(open(config_path))
        config = json.loads(raw) if isinstance(raw, str) else raw
        memories = translate_memories(config)
        batch_results = []
        for f, mem_cfg in enumerate(memories):
            mem_dict = constructor(mem_cfg)
            label = memory_label(mem_cfg)
            data_path = data_dir / f"{prefix}_{batch_id}_{f + 1}.json"
            if not data_path.exists():
                batch_results.append((label, mem_cfg, mem_dict, None))
                continue
            with open(data_path) as fp:
                data_raw = json.load(fp)
            data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
            batch_results.append((label, mem_cfg, mem_dict, data))
        results.append((batch_id, batch_results))
    return results


def plot_profit_grinds(batch_id, items, output_dir, prefix=""):
    """Bar chart of mean profits for the 4 memory types in a batch."""
    labels = []
    means = []
    errs = []
    for label, _, mem_dict, data in items:
        labels.append(label)
        if data is None:
            means.append(np.nan)
            errs.append(0)
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
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Profit")
    ax.set_title(f"Profits for {batch_id}")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = Path(output_dir) / f"{prefix}{batch_id}_profits.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close()
    return out


def plot_br_matrices(batch_id, items, output_dir, prefix="", trial=0):
    """Best-response matrices for each memory type (one batch)."""
    valid = [(i, it) for i, it in enumerate(items) if it[3] is not None]
    n_mem = len(valid)
    if n_mem == 0:
        return None
    fig, axes = plt.subplots(2, n_mem, figsize=(4 * n_mem, 8))
    if n_mem == 1:
        axes = axes.reshape(-1, 1)
    for col, (_, (label, mem_cfg, mem_dict, data)) in enumerate(valid):
        for player, ax_row in enumerate([0, 1]):
            ax = axes[ax_row, col] if n_mem > 1 else axes[ax_row, 0]
            mat, row_labs, col_labs = best_response_matrix(mem_cfg, mem_dict, player + 1, data, trial)
            if mat is None or np.all(np.isnan(mat)):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=PRICE_GRID.min(), vmax=PRICE_GRID.max())
            ax.set_xticks(np.arange(len(col_labs)))
            ax.set_xticklabels(col_labs)
            ax.set_yticks(np.arange(len(row_labs)))
            ax.set_yticklabels(row_labs)
            ax.set_xlabel("Opponent bin")
            ax.set_ylabel("Own bin")
            ax.set_title(f"P{player + 1}")
            plt.colorbar(im, ax=ax, label="Best-response price")
        axes[0, col].set_title(f"{label}\nP1 BR", fontsize=9)
        if n_mem > 1:
            axes[1, col].set_title(f"P2 BR", fontsize=9)
    fig.suptitle(f"{batch_id}: Best-response matrices (trial {trial})", fontsize=12)
    fig.tight_layout()
    out = Path(output_dir) / f"{prefix}{batch_id}_br_matrices.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize batch results: profit grinds + best-response matrices")
    parser.add_argument("--data-dir", default="data_full_memory", help="Directory with data JSONs")
    parser.add_argument("--config-dir", default="simulations/configs", help="Directory with batch config JSONs")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--prefix", default="data_sync_", help="Data file prefix (e.g. data_sync_, data_multithreaded_)")
    parser.add_argument("--br-batches", default="batch_01,batch_10", help="Batches for BR matrices (default: all)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / args.data_dir
    config_dir = base / args.config_dir
    output_dir = base / args.output_dir
    prefix = args.prefix.rstrip("_") + "_"

    all_batches = load_batch_data(data_dir, config_dir, prefix=prefix.rstrip("_"))
    if not all_batches:
        print("No batch data found. Check --data-dir and --config-dir.")
        return

    br_batch_ids = [b.strip() for b in args.br_batches.split(",") if b.strip()]
    batch_map = {b[0]: b[1] for b in all_batches}
    fig_prefix = args.prefix.replace("data_", "").replace("_", "") + "_"

    print(f"Loaded {len(all_batches)} batches.")
    for batch_id, items in all_batches:
        # 1. Profit grinds (4 memory types per batch)
        out = plot_profit_grinds(batch_id, items, output_dir, prefix=fig_prefix)
        print(f"  Saved {out}")
        # 2. Best-response matrices (for selected batches)
        if batch_id in br_batch_ids:
            out = plot_br_matrices(batch_id, items, output_dir, prefix=fig_prefix)
            if out:
                print(f"  Saved {out}")


if __name__ == "__main__":
    main()
