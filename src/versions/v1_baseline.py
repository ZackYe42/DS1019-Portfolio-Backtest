"""
v1: Baseline implementation.

The reference implementation of the full bootstrap-backtest workload.
Deliberately written as a naive double loop for clarity and for use as
a ground-truth target.  No vectorization, no compilation, no
parallelism.  All subsequent versions (v2, v3, v4, v5) must produce
identical results within floating-point tolerance.

Public API:
    run(returns, configs, n_bootstrap, block_size, seed) -> BootstrapResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtest import BacktestConfig, run_backtest
from src.bootstrap import fixed_block_bootstrap
from src.metrics import summary as metric_summary


@dataclass
class BootstrapResult:
    """
    Output of a full bootstrap-backtest run.

    metrics_per_config: DataFrame of shape (n_configs, ~5 metric cols)
        Mean of each metric across bootstrap samples, plus the 5%/95% quantiles.
        This is the table that goes into the final report.

    raw_metrics: DataFrame of shape (n_configs * n_bootstrap, ~5 cols)
        Every single bootstrap sample's metrics.  Useful for diagnostic plots.

    wall_time: float
        Total runtime in seconds.  This is the number we'll compare across
        versions.
    """
    metrics_per_config: pd.DataFrame
    raw_metrics: pd.DataFrame
    wall_time: float


def run(
    returns: np.ndarray,
    configs: Sequence[BacktestConfig],
    n_bootstrap: int = 50,
    block_size: int = 20,
    seed: int = 0,
    show_progress: bool = True,
) -> BootstrapResult:
    """
    Run the full bootstrap-backtest workload.

    For each config and each bootstrap resample, run a backtest and record
    its metrics.  Return an aggregated DataFrame.

    This is the naive baseline: a pure Python double loop over configs and
    bootstrap samples, with each backtest computed serially.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Log returns.
    configs : Sequence[BacktestConfig]
        List of configs to evaluate.
    n_bootstrap : int
        Number of bootstrap resamples per config.
    block_size : int
        Block size for the fixed-block bootstrap.
    seed : int
        Base seed.  Each (config, bootstrap) pair uses a derived sub-seed
        so the full run is reproducible.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    BootstrapResult
    """
    import time

    t0 = time.perf_counter()

    n_configs = len(configs)
    total_iters = n_configs * n_bootstrap

    # Collect raw per-(config, sample) metrics
    raw_rows = []

    iterator = range(total_iters)
    if show_progress:
        iterator = tqdm(iterator, desc="v1 baseline", unit="bt")

    for idx in iterator:
        cfg_idx = idx // n_bootstrap
        boot_idx = idx % n_bootstrap
        cfg = configs[cfg_idx]

        # Derived seed so each (config, bootstrap) pair is reproducible
        sub_seed = seed * 1_000_003 + cfg_idx * 997 + boot_idx
        rng = np.random.default_rng(sub_seed)

        # Generate one bootstrap resample of the returns
        boot_returns = fixed_block_bootstrap(returns, block_size=block_size, rng=rng)

        # Run a full backtest on the resampled returns
        result = run_backtest(boot_returns, cfg)

        # Record: config label + bootstrap index + metrics
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
        raw_rows.append(row)

    wall_time = time.perf_counter() - t0

    # Build DataFrames
    raw_df = pd.DataFrame(raw_rows)

    # Per-config summary: mean + 5%/95% quantiles across bootstrap samples
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
    )


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import load_returns
    from src.backtest import default_config_grid

    print("v1_baseline smoke test\n" + "=" * 70)

    returns = load_returns().to_numpy()
    T, N = returns.shape
    print(f"Input: returns shape = ({T}, {N})")

    # SMALL problem size for the smoke test: 6 configs, 10 bootstrap samples
    # This keeps runtime under ~2 minutes
    configs = default_config_grid()[:6]
    n_bootstrap = 10

    print(f"\nWorkload: {len(configs)} configs x {n_bootstrap} bootstrap samples "
          f"= {len(configs) * n_bootstrap} backtests")
    print(f"Configs:")
    for i, c in enumerate(configs):
        print(f"  [{i}] {c.label()}")

    print()
    result = run(returns, configs, n_bootstrap=n_bootstrap, seed=42)

    print(f"\nTotal wall time: {result.wall_time:.1f}s "
          f"({result.wall_time / (len(configs) * n_bootstrap):.2f}s per backtest)")

    print("\nPer-config summary:\n")
    # Show just the key metrics
    show_cols = ["label", "sharpe_mean", "sharpe_p05", "sharpe_p95",
                 "max_drawdown_mean", "cagr_mean"]
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(result.metrics_per_config[show_cols].to_string(index=False))