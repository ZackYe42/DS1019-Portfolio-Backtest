"""
v4: Multiprocessing parallelism.

Same computation as v3, but each backtest runs in a worker process.
With N physical cores and no inter-task communication, expected
speedup is ~0.6-0.9 * n_workers * (v3 speedup).

On a 24-physical-core i9-14900F with 8 default workers, expect:
    v3 single-thread: ~7x over v1
    v4 8 workers:     ~35-55x over v1

Design:
    - Pool of persistent worker processes (spawn overhead paid once)
    - imap_unordered for per-(config, sample) task dispatch:
        * Tasks of ~200ms each
        * ~1ms of IPC overhead per task (negligible)
        * Perfect load balancing regardless of task variance
    - Default workers capped at 8 to avoid memory blowup on high-core
      machines (each worker consumes ~500 MB of Python/Numba state).
      For scaling studies, pass n_workers explicitly.
    - Each worker loads Numba cache on startup (~100ms), no re-JIT.
    - maxtasksperchild recycles workers every 100 tasks to bound memory.

Public API matches v1/v2/v3 with the addition of n_workers:
    run(returns, configs, n_bootstrap, block_size, seed, n_workers, show_progress)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtest import BacktestConfig
from src.bootstrap import fixed_block_bootstrap


# =============================================================================
# Worker process: pickled initialization + single-task function
# =============================================================================
#
# Each worker needs access to:
#   - the returns matrix (passed via Pool initializer; ~2 MB, copied once)
#   - the list of configs (passed in the same way)
#   - the v3_numba module (imported on first task; triggers JIT warmup from
#     on-disk cache, about 100ms per worker)
#
# We use `get_context("spawn")` explicitly because it's required on Windows
# and safe on macOS/Linux.  Spawn forks fresh Python interpreters that
# re-import modules, so each worker compiles/loads Numba functions on
# first touch.
# =============================================================================

# Module-level variables populated in each worker by _worker_init
_WORKER_RETURNS: Optional[np.ndarray] = None
_WORKER_CONFIGS: Optional[Sequence[BacktestConfig]] = None


def _worker_init(returns: np.ndarray, configs: Sequence[BacktestConfig]) -> None:
    """
    Runs once per worker process.  Loads data, caps BLAS threads, and
    triggers JIT warmup.

    Critical: we limit BLAS (MKL / OpenBLAS) and OpenMP to ONE thread per
    worker.  Without this cap, each of our worker processes would spawn
    N_logical_cores threads for every matrix multiplication, leading to
    massive oversubscription (n_workers x 32 = 256 threads fighting for 32
    cores).  With the cap, total thread count = n_workers, and parallel
    scaling works as expected.

    These environment variables must be set BEFORE importing numpy/numba,
    which is why they're at the top of the init function.
    """
    import os
    # BLAS backends — set all of them since we don't know which numpy was
    # compiled against.  conda-forge numpy uses MKL, pip wheels use OpenBLAS.
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    # Also set Numba's own thread count
    os.environ["NUMBA_NUM_THREADS"] = "1"

    global _WORKER_RETURNS, _WORKER_CONFIGS
    _WORKER_RETURNS = returns
    _WORKER_CONFIGS = configs

    # Import v3 AFTER setting env vars.  Each worker has its own Numba cache;
    # the import triggers _warmup_jit().
    import src.versions.v3_numba  # noqa: F401


def _worker_task(task: tuple[int, int, int, int]) -> dict:
    """
    Run one (config, bootstrap_sample) backtest.

    Task tuple: (cfg_idx, boot_idx, seed, block_size)
    """
    from src.versions.v3_numba import _run_backtest_v3

    cfg_idx, boot_idx, seed, block_size = task

    cfg = _WORKER_CONFIGS[cfg_idx]
    returns = _WORKER_RETURNS

    sub_seed = seed * 1_000_003 + cfg_idx * 997 + boot_idx
    rng = np.random.default_rng(sub_seed)
    boot_returns = fixed_block_bootstrap(returns, block_size=block_size, rng=rng)

    result = _run_backtest_v3(boot_returns, cfg)

    row = {
        "config_idx": cfg_idx,
        "bootstrap_idx": boot_idx,
        "estimator": cfg.estimator,
        "objective": cfg.objective,
        "lookback": cfg.lookback,
        "rebalance_every": cfg.rebalance_every,
        "label": cfg.label(),
    }
    row.update(result.metrics)
    return row


# =============================================================================
# Public API
# =============================================================================

@dataclass
class BootstrapResult:
    metrics_per_config: pd.DataFrame
    raw_metrics: pd.DataFrame
    wall_time: float
    n_workers: int


def _default_workers() -> int:
    """
    Default worker count, capped at a safe value.

    For CPU-bound numerical work, the speedup curve flattens around
    8 workers regardless of whether you have 8 or 32 cores, because:
      - Each worker duplicates ~500 MB of Python/Numba/library state
      - Numba JIT compilation contention increases with worker count
      - Our workload per backtest (~200ms) is short enough that
        coordination overhead starts to dominate past ~8 workers

    On a high-core-count machine, you can still set n_workers=N explicitly
    to benchmark scaling (benchmarks/scaling_study.py does this).
    """
    MAX_DEFAULT_WORKERS = 8

    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n is not None and n > 0:
            return min(n, MAX_DEFAULT_WORKERS)
    except ImportError:
        pass

    n_logical = os.cpu_count() or 2
    return min(max(1, n_logical // 2), MAX_DEFAULT_WORKERS)


def run(
    returns: np.ndarray,
    configs: Sequence[BacktestConfig],
    n_bootstrap: int = 50,
    block_size: int = 20,
    seed: int = 0,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
) -> BootstrapResult:
    """
    v4 multiprocessing run.  Same API as v1/v2/v3 plus n_workers.

    Parameters
    ----------
    n_workers : int or None
        Number of worker processes.  None = auto (physical cores, capped at 8).
        Pass an explicit integer to benchmark scaling.
    """
    if n_workers is None:
        n_workers = _default_workers()

    t0 = time.perf_counter()

    n_configs = len(configs)
    total_tasks = n_configs * n_bootstrap

    # Build the task list.  Each task is a tiny tuple — the bulk of
    # data (the returns matrix) is sent once per worker via init.
    tasks = [
        (cfg_idx, boot_idx, seed, block_size)
        for cfg_idx in range(n_configs)
        for boot_idx in range(n_bootstrap)
    ]

    # spawn context - required on Windows, safe on macOS/Linux
    ctx = get_context("spawn")

    raw_rows: list[dict] = []
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(returns, configs),
        maxtasksperchild=100,  # recycle workers periodically to bound memory
    ) as pool:
        # imap_unordered: return results as they complete.
        # chunksize controls how many tasks a worker grabs at once;
        # larger chunks reduce IPC overhead but worsen load balance.
        chunksize = max(1, total_tasks // (n_workers * 8))
        iterator = pool.imap_unordered(_worker_task, tasks, chunksize=chunksize)

        if show_progress:
            iterator = tqdm(iterator, total=total_tasks,
                            desc=f"v4 multiproc ({n_workers} workers)", unit="bt")

        for row in iterator:
            raw_rows.append(row)

    wall_time = time.perf_counter() - t0

    # Build DataFrames (same structure as v1/v2/v3)
    raw_df = pd.DataFrame(raw_rows)

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
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_p05"] = float(np.nanquantile(vals, 0.05))
            row[f"{m}_p95"] = float(np.nanquantile(vals, 0.95))
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    return BootstrapResult(
        metrics_per_config=agg_df,
        raw_metrics=raw_df,
        wall_time=wall_time,
        n_workers=n_workers,
    )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v1_baseline, v3_numba

    print("v4 multiprocessing smoke test\n" + "=" * 70)

    returns = load_returns().to_numpy()

    configs = default_config_grid()[:6]
    n_bootstrap = 10

    print(f"\nHardware:")
    print(f"  Logical cores:  {os.cpu_count()}")
    try:
        import psutil
        print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    except ImportError:
        print(f"  (install psutil for physical core detection)")
    print(f"  Default workers (capped): {_default_workers()}")

    print(f"\nWorkload: {len(configs)} configs x {n_bootstrap} bootstrap "
          f"= {len(configs) * n_bootstrap} backtests\n")

    # Run v1 for baseline timing
    print("Running v1 baseline (sequential, pure Python)...")
    v1_result = v1_baseline.run(returns, configs, n_bootstrap=n_bootstrap,
                                seed=42, show_progress=False)
    print(f"  v1 wall time: {v1_result.wall_time:.2f}s")

    # Run v3 (single-thread numba) for relative comparison
    print("\nRunning v3 numba (single-thread)...")
    v3_result = v3_numba.run(returns, configs, n_bootstrap=n_bootstrap,
                             seed=42, show_progress=False)
    print(f"  v3 wall time: {v3_result.wall_time:.2f}s")

    # Run v4 with default workers
    print("\nRunning v4 multiproc (auto workers)...")
    v4_result = run(returns, configs, n_bootstrap=n_bootstrap,
                    seed=42, show_progress=False)
    print(f"  v4 wall time: {v4_result.wall_time:.2f}s  ({v4_result.n_workers} workers)")

    speedup_v1 = v1_result.wall_time / v4_result.wall_time
    speedup_v3 = v3_result.wall_time / v4_result.wall_time
    efficiency = 100 * speedup_v3 / v4_result.n_workers
    print(f"\nSpeedup vs v1: {speedup_v1:.1f}x")
    print(f"Speedup vs v3: {speedup_v3:.2f}x  (parallelism efficiency: {efficiency:.0f}%)")

    # Correctness check
    print("\n" + "=" * 70)
    print("Correctness check: v4 vs v1 metrics")
    print("=" * 70)

    df1 = v1_result.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
    df4 = v4_result.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)

    TOL = 1e-6
    all_pass = True
    for col in ["sharpe", "annual_vol", "max_drawdown", "cagr"]:
        diff = np.abs(df1[col].to_numpy() - df4[col].to_numpy()).max()
        status = "PASS" if diff < TOL else "FAIL"
        if diff >= TOL:
            all_pass = False
        print(f"  {col:15s} max |v1 - v4| = {diff:.2e}  [{status}]")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")