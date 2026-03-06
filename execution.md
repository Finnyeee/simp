# Execution Guide

This document describes how to run the simplicity simulations from a fresh setup.

**Repository:** https://github.com/martinobanchio/simplicity/tree/asym_memories

## Prerequisites

- **Julia** (1.x recommended; check compatibility with `Distributions`, `NNlib`, etc.)
- **Python 3** (for visualization tools, with `numpy`, `matplotlib`, `json`)

## Environment Setup

### Julia Project

The project uses a Julia environment defined by `Project.toml` in the repo root. Activate it before running:

```bash
cd /path/to/simplicity
julia --project=.
```

Or from the Julia REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

This installs dependencies: `DelimitedFiles`, `Distributed`, `Distributions`, `JSON`, `LinearAlgebra`, `NNlib`, `ProgressMeter`, `Random`, `Statistics`, `StatsBase`.

### Python (for visualization)

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Or with a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The visualization script requires `numpy` and `matplotlib`.

---

## Running Simulations

### Option 1: Synchronous generator (single batch)

Runs one batch at a time. Use this for smaller runs or when you want to target specific batches.

**Command:**

```bash
julia --project=. simulations/generator_sync.jl BATCH K ITERATIONS
```

**Arguments:**

| Argument | Meaning | Example |
|----------|---------|---------|
| `BATCH` | Config name (e.g. `batch_01`, `batch_02`) | `batch_01` |
| `K` | Number of simulation runs per memory type | `4` |
| `ITERATIONS` | Training iterations per run | `1000000` |

**Example:**

```bash
julia --project=. simulations/generator_sync.jl batch_01 4 1000000
```

Output files: `data_full_memory/data_sync_batch_01_1.json`, `_2.json`, `_3.json`, `_4.json` (one per memory type in the batch).

### Option 2: Multithreaded generator (all batches)

Runs all 16 batches in sequence, using multiple Julia processes for parallelism within each batch.

**Command:**

```bash
julia --project=. generator_multithreaded.jl THREADS K ITERATIONS
```

**Arguments:**

| Argument | Meaning | Example |
|----------|---------|---------|
| `THREADS` | Number of worker processes | `4` |
| `K` | Number of simulation runs per memory type | `4` |
| `ITERATIONS` | Training iterations per run | `1000000` |

**Example:**

```bash
julia --project=. generator_multithreaded.jl 4 4 1000000
```

Output files: `data_full_memory/data_multithreaded_batch_XX_f.json` for batches 01–16 and memory indices 1–4.

**Optional 4th argument — single config:** To run one config file (e.g. symmetric 2-bin) instead of all 16 batches:

```bash
julia --project=. generator_multithreaded.jl 4 4 2000000 symmetric_2bin
```

Output: `data_full_memory/data_multithreaded_symmetric_2bin_1.json` … `_14.json`.

### Option 3: Asynchronous generator

Same interface as the sync generator, but with asynchronous updates. Writes to `data_full_memory_long/`.

```bash
julia --project=. simulations/generator_async.jl batch_01 4 1000000
```

### Option 4: Symmetric 2-bin (14 memory types)

Use the multithreaded generator with config name `symmetric_2bin`. Config: `simulations/configs/symmetric_2bin.json` (thresholds 1–14; both players 2 bins each).

```bash
julia --project=. generator_multithreaded.jl 4 4 2000000 symmetric_2bin
```

Output: `data_full_memory/data_multithreaded_symmetric_2bin_1.json` … `_14.json`.

### Option 5: Q-history (2-bin, 2M iterations, full Q saved)

Run **one** simulation per symmetric 2-bin memory type for 2 million iterations and save the **entire Q matrix at every iteration**. Use this when you need Q-history for analysis (e.g. rolling optimal prices). Output files are very large; a trim step keeps only the last 5000 iterations for plotting.

**1. Run (saves full Q every iteration):**

```bash
julia --project=. run_2bin_qhistory_2m.jl
```

Output: `data_full_memory/qhistory_2m_symmetric_2bin_1.json` … `_14.json`. Each file contains `final_Q`, `Q_history` (length 2e6), and metadata.

**2. Trim to last 5000 iterations (optional, to reduce size for analysis):**

```bash
julia --project=. scripts/trim_qhistory_to_last_500.jl [data_dir]
```

Default `data_dir` is `data_full_memory/`. Reads all `qhistory_2m_*.json` (excluding already trimmed), keeps the last 5000 snapshots, and writes `qhistory_2m_symmetric_2bin_<k>_trimmed.json`.

**3. Plot rolling optimal prices from trimmed Q-history:**

```bash
python analysis/plot_qhistory_rolling_prices.py [--data-dir data_full_memory] [--output-dir figures] [--window 50]
```

Requires `*_trimmed.json` in the data directory. Produces `figures/qhistory_2m_symmetric_2bin_<k>_rolling_prices.png` for each trimmed file (rolling average of best-response prices over the last 5000 iterations).

---

## Data Layout

- **Batches:** Configs live in `simulations/configs/batch_01.json` through `batch_16.json`.
- **Memory types:** Each batch has 4 memory configurations (indexed 1–4).
- **Runs:** Each memory type is run `K` times (e.g. 4 simulations per memory type).

File naming:

- Sync: `data_full_memory/data_sync_{BATCH}_{MEMORY_INDEX}.json`
- Multithreaded: `data_full_memory/data_multithreaded_{BATCH}_{MEMORY_INDEX}.json`
- Async: `data_full_memory_long/data_async_{BATCH}_{MEMORY_INDEX}.json`

---

## Visualization

The visualization script produces two types of figures from simulation data.

### Prerequisites

1. Simulation data must exist in `data_full_memory/` (or the directory given by `--data-dir`).
2. Install Python dependencies: `pip install -r requirements.txt`

### Making the plots

**1. Profit grinds**

Run from the project root:

```bash
python analysis/visualize_batches.py
```

This generates profit grind bar charts for **every batch**: `figures/sync_batch_01_profits.png` through `sync_batch_16_profits.png`. Each plot shows mean profit (with error bars) for the 4 memory types in that batch.

**2. Best-response state reports**

Best-response figures are generated only for batches listed in `--br-batches`. By default, only `batch_01` and `batch_10`. Each figure shows:
- State representation matrix (pooled prices labeled with letters A, B, C, D, etc.)
- Best-response price pairs for player 1
- Best-response price pairs for player 2

```bash
python analysis/visualize_batches.py
```

Output: `figures/sync_batch_01_mem1_br.png` through `mem4_br.png` (one per memory type) for each selected batch.

To generate for more batches:

```bash
python analysis/visualize_batches.py --br-batches batch_01,batch_05,batch_10,batch_16
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `data_full_memory` | Directory containing data JSONs |
| `--config-dir` | `simulations/configs` | Directory containing batch config JSONs |
| `--output-dir` | `figures` | Where to save the plots |
| `--prefix` | `data_sync_` | Data file prefix (`data_sync_` or `data_multithreaded_`) |
| `--br-batches` | `batch_01,batch_10` | Comma-separated list of batches to generate BR state reports for |

### Using multithreaded data

If you ran the multithreaded generator instead of the sync generator:

```bash
python analysis/visualize_batches.py --prefix data_multithreaded_
```

### Q-history rolling prices (from trimmed Q-history)

If you have run the 2-bin Q-history pipeline (Option 5) and trimmed to the last 5000 iterations:

```bash
python analysis/plot_qhistory_rolling_prices.py
```

Output: `figures/qhistory_2m_symmetric_2bin_<k>_rolling_prices.png` for each trimmed file. Optional: `--window 50` (default), `--data-dir`, `--output-dir`.

### Summary

| Step | Command | Output |
|------|---------|--------|
| Profit grinds (all batches) | `python analysis/visualize_batches.py` | `figures/sync_batch_XX_profits.png` for each batch |
| Best-response state reports | `python analysis/visualize_batches.py --br-batches batch_01,batch_10` | `figures/sync_batch_XX_memN_br.png` (state matrix + BR pairs) for selected batches |
| Q-history rolling prices | `python analysis/plot_qhistory_rolling_prices.py` | `figures/qhistory_2m_symmetric_2bin_<k>_rolling_prices.png` (from `*_trimmed.json`) |

### Quick Start (visualization)

After simulations have produced data in `data_full_memory/`:

```bash
pip install -r requirements.txt
python analysis/visualize_batches.py
```

This generates all profit grinds and BR state reports for `batch_01` and `batch_10`. Add `--br-batches batch_01,batch_02,...,batch_16` to generate BR reports for all batches.

---

## Quick Start (minimal run)

```bash
cd /path/to/simplicity
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. simulations/generator_sync.jl batch_01 2 100000
```

This produces 4 data files in `data_full_memory/` (2 runs × 4 memory types for `batch_01`).
