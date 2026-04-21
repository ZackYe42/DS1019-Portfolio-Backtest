# Optimized Portfolio Backtest Engine

DS-GA 1019 Advanced Python for Data Science — Spring 2026 Final Project

**Team:** Shawn Xia, Zack Ye

## Overview

This project implements a rolling-window portfolio backtesting framework with
bootstrap confidence intervals, and benchmarks five progressively optimized
implementations: pure Python → NumPy → Numba → parallel CPU (multiprocessing +
MPI) → GPU (Numba CUDA).

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Download price data (cached after first run)
python -m src.data_loader

# 2. Run a single version
python -m src.versions.v1_baseline

# 3. Run the full benchmark suite
python -m benchmarks.run_all
```

## Project Structure

- `src/` — shared modules (data loading, estimators, optimizer, metrics, bootstrap)
- `src/versions/` — five implementations of the same backtest pipeline
- `benchmarks/` — timing harness and scaling studies
- `tests/` — correctness tests (all versions must agree within 1e-6)
- `results/` — benchmark CSVs and figures
- `report/` — final 4-page writeup
- `slides/` — 5-minute final presentation

## Results Summary

_(to be filled in after Phase 6)_

| Version | Speedup vs Baseline |
|---------|---------------------|
| v1 Baseline (pure Python) | 1× |
| v2 NumPy vectorized | TBD |
| v3 Numba | TBD |
| v4 Multiprocessing / MPI | TBD |
| v5 GPU (Numba CUDA) | TBD |