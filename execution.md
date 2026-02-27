# Execution Guide

This document describes how to run the simplicity simulations from a fresh setup.

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

### Option 3: Asynchronous generator

Same interface as the sync generator, but with asynchronous updates. Writes to `data_full_memory_long/`.

```bash
julia --project=. simulations/generator_async.jl batch_01 4 1000000
```

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

**2. Best-response matrices**

Best-response matrices are generated only for batches listed in `--br-batches`. By default, only `batch_01` and `batch_10`:

```bash
python analysis/visualize_batches.py
```

Output: `figures/sync_batch_01_br_matrices.png` and `figures/sync_batch_10_br_matrices.png`.

To generate BR matrices for more batches:

```bash
python analysis/visualize_batches.py --br-batches batch_01,batch_05,batch_10,batch_16
```

To generate BR matrices for all 16 batches:

```bash
python analysis/visualize_batches.py --br-batches batch_01,batch_02,batch_03,batch_04,batch_05,batch_06,batch_07,batch_08,batch_09,batch_10,batch_11,batch_12,batch_13,batch_14,batch_15,batch_16
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `data_full_memory` | Directory containing data JSONs |
| `--config-dir` | `simulations/configs` | Directory containing batch config JSONs |
| `--output-dir` | `figures` | Where to save the plots |
| `--prefix` | `data_sync_` | Data file prefix (`data_sync_` or `data_multithreaded_`) |
| `--br-batches` | `batch_01,batch_10` | Comma-separated list of batches to generate BR matrices for |

### Using multithreaded data

If you ran the multithreaded generator instead of the sync generator:

```bash
python analysis/visualize_batches.py --prefix data_multithreaded_
```

### Summary

| Step | Command | Output |
|------|---------|--------|
| Profit grinds (all batches) | `python analysis/visualize_batches.py` | `figures/sync_batch_XX_profits.png` for each batch |
| Best-response matrices | `python analysis/visualize_batches.py --br-batches batch_01,batch_10` | `figures/sync_batch_XX_br_matrices.png` for selected batches |

---

## Quick Start (minimal run)

```bash
cd /path/to/simplicity
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. simulations/generator_sync.jl batch_01 2 100000
```

This produces 4 data files in `data_full_memory/` (2 runs × 4 memory types for `batch_01`).
