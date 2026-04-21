"""
v5: GPU-accelerated backtest via Numba CUDA kernels.

Strategy:
    - min_variance configs  -> GPU batched full-backtest kernel
      (src.versions.v5_gpu_kernels.run_full_backtest_gpu)
      One CUDA block per (config, bootstrap) pair, running all rebalances
      on-device; zero CPU-GPU roundtrips during the rebalance loop.

    - max_sharpe configs    -> CPU multiprocessing fallback (v4-style pool)
      The GPU max_sharpe kernel exists and is correct, but hasn't been
      integrated into the full-backtest kernel yet; we route these to
      the CPU path for now.

Pipeline optimizations (v5.1):
    - Vectorized portfolio-return reconstruction using np.einsum
      (replaces a Python loop over ~2500 days, ~100x faster)
    - Parallel bootstrap sampling via ThreadPoolExecutor (bootstrap is
      numpy-heavy; the GIL is released during np operations)

Public API matches v1/v2/v3/v4:
    run(returns, configs, n_bootstrap, block_size, seed, gpu_batch_size,
        show_progress, verbose_timing) -> BootstrapResult
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba.core.errors import NumbaPerformanceWarning

from src.backtest import BacktestConfig
from src.bootstrap import fixed_block_bootstrap
from src.metrics import summary as metric_summary
from src.versions import v3_numba
from src.versions.v5_gpu_kernels import run_full_backtest_gpu, N_ASSETS

# The GPU kernel issues these when batch is small; informational only.
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


@dataclass
class BootstrapResult:
    metrics_per_config: pd.DataFrame
    raw_metrics: pd.DataFrame
    wall_time: float
    gpu_fraction: float


# =============================================================================
# Vectorized portfolio-return reconstruction
# =============================================================================

def _reconstruct_port_returns_vectorized(
    weights_history: np.ndarray,       # (n_rebalance, N)
    boot_arith: np.ndarray,            # (T, N)
    lookback: int,
    rebalance_indices: np.ndarray,     # (n_rebalance,) day indices
) -> np.ndarray:
    """
    Reconstruct daily portfolio returns given the weight history and
    bootstrapped arithmetic returns, fully vectorized.

    Between rebalance dates, weights are constant.  We build a
    (T - lookback, N) weights array by broadcasting each rebalance's
    weights to every day until the next rebalance, then take the
    element-wise dot product with the bootstrapped returns.

    Returns
    -------
    port_returns : (T - lookback,) array of daily portfolio returns
    """
    T, N = boot_arith.shape
    n_rebalance = weights_history.shape[0]
    n_days = T - lookback

    # Build (n_days, N) weights matrix
    weights_per_day = np.empty((n_days, N), dtype=np.float64)

    # Days before the first rebalance get equal weights (v3 behavior)
    # rebalance_indices[0] is the first rebalance day; everything before
    # it uses equal weights.  In practice rebalance_indices[0] == lookback,
    # so this slice is empty.  Keep the code for generality.
    first_reb_offset = rebalance_indices[0] - lookback
    if first_reb_offset > 0:
        weights_per_day[:first_reb_offset] = 1.0 / N

    # For each rebalance r, fill [rebalance_indices[r]-lookback, next-rebalance-day)
    for r in range(n_rebalance):
        t_start = rebalance_indices[r] - lookback
        if r + 1 < n_rebalance:
            t_end = rebalance_indices[r + 1] - lookback
        else:
            t_end = n_days
        weights_per_day[t_start:t_end] = weights_history[r]

    # Daily portfolio return = w[d] . r[d]  for each day d
    # einsum computes this in one pass
    return np.einsum("dn,dn->d", weights_per_day, boot_arith[lookback:])


# =============================================================================
# Parallel bootstrap sampling
# =============================================================================

def _build_bootstrap_batch_parallel(
    returns: np.ndarray,
    seed: int,
    cfg_idx: int,
    boot_indices: Sequence[int],
    block_size: int,
    n_threads: int = 4,
) -> np.ndarray:
    """
    Build a batch of bootstrap samples in parallel threads.

    Bootstrap is numpy-heavy (index fancy-indexing + concatenation) and
    releases the GIL during those ops, so threads parallelize well.

    Returns
    -------
    boot_returns : (len(boot_indices), T, N) float64
    """
    T, N = returns.shape
    B = len(boot_indices)
    out = np.empty((B, T, N), dtype=np.float64)

    def _one(i: int) -> None:
        boot_idx = boot_indices[i]
        sub_seed = seed * 1_000_003 + cfg_idx * 997 + boot_idx
        rng = np.random.default_rng(sub_seed)
        out[i] = fixed_block_bootstrap(returns, block_size=block_size, rng=rng)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(_one, range(B)))

    return out


# =============================================================================
# GPU path: one config's worth of bootstraps in batches
# =============================================================================

def _run_config_on_gpu(
    returns: np.ndarray,
    config: BacktestConfig,
    cfg_idx: int,
    n_bootstrap: int,
    block_size: int,
    seed: int,
    gpu_batch_size: int,
    verbose_timing: bool = False,
) -> list[dict]:
    """
    Run all bootstrap samples for a single min_variance config on GPU.

    Returns a list of per-bootstrap metric dicts.
    """
    T, N = returns.shape
    lb = config.lookback
    rb = config.rebalance_every

    n_rebalance = 0
    t = lb
    while t < T:
        n_rebalance += 1
        t += rb
    rebalance_indices = np.arange(lb, T, rb)[:n_rebalance]

    rows: list[dict] = []
    t_boot_total = 0.0
    t_gpu_total = 0.0
    t_recon_total = 0.0

    for batch_start in range(0, n_bootstrap, gpu_batch_size):
        batch_end = min(batch_start + gpu_batch_size, n_bootstrap)
        batch_indices = list(range(batch_start, batch_end))
        B = len(batch_indices)

        # 1) Build bootstraps in parallel
        t0 = time.perf_counter()
        boot_returns = _build_bootstrap_batch_parallel(
            returns, seed, cfg_idx, batch_indices, block_size
        )
        t_boot_total += time.perf_counter() - t0

        # 2) Run GPU kernel
        t0 = time.perf_counter()
        gpu_weights = run_full_backtest_gpu(
            boot_returns, lb, rb,
            estimator=config.estimator,
            ewma_lam=0.94,
        )
        t_gpu_total += time.perf_counter() - t0

        # 3) Vectorized reconstruction on CPU (fast)
        t0 = time.perf_counter()
        for i, boot_idx in enumerate(batch_indices):
            weights_history = gpu_weights[i].astype(np.float64)
            boot_arith = np.exp(boot_returns[i]) - 1.0

            port_returns = _reconstruct_port_returns_vectorized(
                weights_history, boot_arith, lb, rebalance_indices
            )

            metrics = metric_summary(port_returns, rf_annual=config.rf_annual,
                                      log_returns=False)

            row = {
                "config_idx": cfg_idx,
                "bootstrap_idx": boot_idx,
                "estimator": config.estimator,
                "objective": config.objective,
                "lookback": config.lookback,
                "rebalance_every": config.rebalance_every,
                "label": config.label(),
            }
            row.update(metrics)
            rows.append(row)
        t_recon_total += time.perf_counter() - t0

    if verbose_timing:
        print(f"    cfg{cfg_idx} [{config.label()[:40]}]: "
              f"boot={t_boot_total*1000:.0f}ms, "
              f"gpu={t_gpu_total*1000:.0f}ms, "
              f"recon={t_recon_total*1000:.0f}ms")

    return rows


# =============================================================================
# Public API
# =============================================================================

def run(
    returns: np.ndarray,
    configs: Sequence[BacktestConfig],
    n_bootstrap: int = 50,
    block_size: int = 20,
    seed: int = 0,
    gpu_batch_size: int = 200,
    show_progress: bool = True,
    verbose_timing: bool = False,
) -> BootstrapResult:
    """
    v5 GPU run.  Same API as v1/v2/v3/v4.

    Parameters
    ----------
    gpu_batch_size : int
        Maximum number of backtests per GPU kernel launch.  Each problem
        needs ~1 MB of VRAM for returns storage; 200 fits easily.
    verbose_timing : bool
        If True, print per-config timing breakdown (bootstrap, GPU, reconstruction).
    """
    t0 = time.perf_counter()

    gpu_configs_idx = [i for i, c in enumerate(configs) if c.objective == "min_variance"]
    cpu_configs_idx = [i for i, c in enumerate(configs) if c.objective == "max_sharpe"]

    raw_rows: list[dict] = []

    # --- GPU path: min_variance configs ---
    if gpu_configs_idx:
        iterator = gpu_configs_idx
        if show_progress:
            iterator = tqdm(gpu_configs_idx, desc="v5 GPU min_var", unit="cfg")
        for cfg_idx in iterator:
            cfg = configs[cfg_idx]
            rows = _run_config_on_gpu(
                returns, cfg, cfg_idx, n_bootstrap, block_size,
                seed, gpu_batch_size, verbose_timing=verbose_timing,
            )
            raw_rows.extend(rows)

    # --- CPU fallback: max_sharpe configs via multiprocessing pool ---
    # We replicate v4's pool-based dispatch here so bootstrap seeds use
    # the ORIGINAL cfg_idx (workers look up configs[cfg_idx] with the full
    # config list).
    if cpu_configs_idx:
        from src.versions.v4_multiproc import _worker_init, _worker_task

        tasks = [
            (cfg_idx, boot_idx, seed, block_size)
            for cfg_idx in cpu_configs_idx
            for boot_idx in range(n_bootstrap)
        ]

        n_workers = min(8, max(1, len(tasks) // 10))
        ctx = get_context("spawn")

        cpu_rows: list[dict] = []
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(returns, configs),
            maxtasksperchild=100,
        ) as pool:
            chunksize = max(1, len(tasks) // (n_workers * 8))
            iterator = pool.imap_unordered(_worker_task, tasks, chunksize=chunksize)
            if show_progress:
                iterator = tqdm(iterator, total=len(tasks),
                                desc="v5 CPU fallback", unit="bt")
            for row in iterator:
                cpu_rows.append(row)

        raw_rows.extend(cpu_rows)

    wall_time = time.perf_counter() - t0

    raw_df = pd.DataFrame(raw_rows).sort_values(
        ["config_idx", "bootstrap_idx"]
    ).reset_index(drop=True)

    metric_cols = ["sharpe", "annual_vol", "annual_return", "max_drawdown", "cagr"]
    agg_rows = []
    for cfg_idx, cfg in enumerate(configs):
        sub = raw_df[raw_df["config_idx"] == cfg_idx]
        row = {
            "config_idx": cfg_idx,
            "estimator": cfg.estimator,
            "objective": cfg.objective,
            "lookback": cfg.lookback,
            "rebalance_every": cfg.rebalance_every,
            "label": cfg.label(),
        }
        for m in metric_cols:
            vals = sub[m].to_numpy()
            row[f"{m}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            row[f"{m}_p05"] = float(np.nanquantile(vals, 0.05)) if len(vals) else float("nan")
            row[f"{m}_p95"] = float(np.nanquantile(vals, 0.95)) if len(vals) else float("nan")
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    n_gpu = len(gpu_configs_idx) * n_bootstrap
    n_total = len(configs) * n_bootstrap
    gpu_fraction = n_gpu / n_total if n_total > 0 else 0.0

    return BootstrapResult(
        metrics_per_config=agg_df,
        raw_metrics=raw_df,
        wall_time=wall_time,
        gpu_fraction=gpu_fraction,
    )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v1_baseline

    print("v5 GPU smoke test\n" + "=" * 70)

    returns = load_returns().to_numpy()
    configs = default_config_grid()[:6]
    n_bootstrap = 10

    print(f"\nWorkload: {len(configs)} configs x {n_bootstrap} bootstrap "
          f"= {len(configs) * n_bootstrap} backtests")

    gpu_cfgs = [c for c in configs if c.objective == "min_variance"]
    cpu_cfgs = [c for c in configs if c.objective == "max_sharpe"]
    print(f"  GPU path: {len(gpu_cfgs)} configs")
    print(f"  CPU path: {len(cpu_cfgs)} configs (max_sharpe)\n")

    print("Running v1 baseline...")
    v1r = v1_baseline.run(returns, configs, n_bootstrap=n_bootstrap,
                          seed=42, show_progress=False)
    print(f"  v1 wall time: {v1r.wall_time:.2f}s\n")

    print("Running v5 GPU (with timing breakdown)...")
    v5r = run(returns, configs, n_bootstrap=n_bootstrap,
              seed=42, show_progress=False, verbose_timing=True)
    print(f"  v5 wall time: {v5r.wall_time:.2f}s")
    print(f"  GPU fraction: {v5r.gpu_fraction:.0%}")

    speedup = v1r.wall_time / v5r.wall_time
    print(f"\nSpeedup vs v1: {speedup:.1f}x\n")

    # Correctness
    df1 = v1r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
    df5 = v5r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
    max_diff = np.abs(df1["sharpe"].to_numpy() - df5["sharpe"].to_numpy()).max()
    print(f"max |Sharpe diff|: {max_diff:.2e}  "
          f"{'PASS' if max_diff < 1e-4 else 'FAIL'}")