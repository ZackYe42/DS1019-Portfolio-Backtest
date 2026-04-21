"""
Rolling-window portfolio backtest engine.

Given historical returns and a configuration (estimator / objective /
lookback / rebalance frequency), simulate a trading strategy that:
    1. estimates covariance on a rolling window,
    2. solves for optimal portfolio weights,
    3. holds those weights until the next rebalance date.

The output is a time series of daily portfolio returns, plus the
performance metrics computed from that series.

This is the REFERENCE IMPLEMENTATION used by src/versions/v1_baseline.py.
All optimized versions (NumPy vectorized, Numba, GPU) reimplement the
same function with different performance strategies but must produce
identical results (within floating-point tolerance).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np
import pandas as pd

from src.estimators import estimate_cov
from src.optimizer import min_variance, max_sharpe
from src.metrics import summary as metric_summary


EstimatorName = Literal["sample", "ledoit_wolf", "ewma"]
ObjectiveName = Literal["min_variance", "max_sharpe"]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    """One set of parameters defining a backtest run."""
    estimator: EstimatorName = "ledoit_wolf"
    objective: ObjectiveName = "min_variance"
    lookback: int = 252                # trading days
    rebalance_every: int = 21          # trading days (monthly ~= 21)
    long_only: bool = True
    rf_annual: float = 0.0             # risk-free rate (annual)

    def label(self) -> str:
        return (f"{self.estimator}|{self.objective}"
                f"|lb={self.lookback}|rb={self.rebalance_every}")


@dataclass
class BacktestResult:
    """Output of one backtest run."""
    config: BacktestConfig
    portfolio_returns: np.ndarray             # shape (T_eff,), daily arithmetic returns
    weights_history: np.ndarray               # shape (n_rebalance, N)
    rebalance_indices: np.ndarray             # shape (n_rebalance,), idx into returns
    metrics: dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Core backtest function (reference implementation)
# -----------------------------------------------------------------------------

def run_backtest(
    returns: np.ndarray,
    config: BacktestConfig,
) -> BacktestResult:
    """
    Run one backtest and return a BacktestResult.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Log returns.  The first `config.lookback` days are used as burn-in
        (the portfolio does not trade until we have a full lookback window).
    config : BacktestConfig

    Returns
    -------
    BacktestResult
        Contains the daily portfolio return series and summary metrics.
    """
    T, N = returns.shape
    lb = config.lookback
    rb = config.rebalance_every

    if lb >= T:
        raise ValueError(f"lookback {lb} >= T {T}; no data to trade on")
    if rb < 1:
        raise ValueError(f"rebalance_every must be >= 1, got {rb}")

    # Convert log returns to arithmetic returns for portfolio compounding.
    # w @ arithmetic_returns is the portfolio's arithmetic return for that day.
    arith_returns = np.exp(returns) - 1.0

    # Rebalance at days lb, lb+rb, lb+2*rb, ...
    rebalance_indices = np.arange(lb, T, rb)
    n_rebalance = len(rebalance_indices)

    # Initial weights: equal-weighted.  Will be overwritten at first rebalance.
    weights = np.full(N, 1.0 / N)
    weights_history = np.zeros((n_rebalance, N))

    # Portfolio return series: length T - lb, starting from day lb onward
    port_returns = np.zeros(T - lb)

    rb_ptr = 0
    for t in range(lb, T):
        # Did we hit a rebalance date?  Solve a new weight vector.
        if rb_ptr < n_rebalance and t == rebalance_indices[rb_ptr]:
            window = returns[t - lb : t]                  # (lb, N) of past returns
            cov = estimate_cov(window, method=config.estimator)

            if config.objective == "min_variance":
                weights = min_variance(cov, method="numpy",
                                       long_only=config.long_only)
            elif config.objective == "max_sharpe":
                mu = window.mean(axis=0)                   # historical mean proxy
                weights = max_sharpe(cov, mu, rf=0.0, method="numpy",
                                     long_only=config.long_only)
            else:
                raise ValueError(f"unknown objective '{config.objective}'")

            weights_history[rb_ptr] = weights
            rb_ptr += 1

        # Day t's portfolio return is weights dotted with today's asset returns
        port_returns[t - lb] = weights @ arith_returns[t]

    # Compute summary metrics
    metrics = metric_summary(
        port_returns,
        rf_annual=config.rf_annual,
        log_returns=False,
    )

    return BacktestResult(
        config=config,
        portfolio_returns=port_returns,
        weights_history=weights_history,
        rebalance_indices=rebalance_indices,
        metrics=metrics,
    )


# -----------------------------------------------------------------------------
# Configuration grid
# -----------------------------------------------------------------------------

def default_config_grid() -> list[BacktestConfig]:
    """
    The parameter sweep we'll benchmark across all versions.

    Cross-product of:
        - 3 estimators
        - 2 objectives
        - 3 lookback windows (63 / 252 / 504 days)
        - 3 rebalance frequencies (5 / 21 / 63 days)

    = 54 configs, enough to demonstrate coarse-grained parallelism.
    """
    estimators: list[EstimatorName] = ["sample", "ledoit_wolf", "ewma"]
    objectives: list[ObjectiveName] = ["min_variance", "max_sharpe"]
    lookbacks = [63, 252, 504]
    rebalances = [5, 21, 63]

    configs = []
    for est in estimators:
        for obj in objectives:
            for lb in lookbacks:
                for rb in rebalances:
                    configs.append(BacktestConfig(
                        estimator=est,
                        objective=obj,
                        lookback=lb,
                        rebalance_every=rb,
                    ))
    return configs


def summarize_results(results: list[BacktestResult]) -> pd.DataFrame:
    """
    Collapse a list of BacktestResults into a DataFrame of metrics.

    Each row = one config; columns = config params + metric values.
    """
    rows = []
    for r in results:
        row = asdict(r.config)
        row.update(r.metrics)
        row["label"] = r.config.label()
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from src.data_loader import load_returns

    print("Backtest smoke test\n" + "=" * 70)

    returns_df = load_returns()
    returns = returns_df.to_numpy()
    T, N = returns.shape
    print(f"Input: returns shape = ({T}, {N})\n")

    # --- Test 1: one config ---
    print("[Test 1: single backtest, Ledoit-Wolf + min-variance, monthly rebalance]")
    cfg = BacktestConfig(
        estimator="ledoit_wolf",
        objective="min_variance",
        lookback=252,
        rebalance_every=21,
    )
    t0 = time.perf_counter()
    result = run_backtest(returns, cfg)
    dt = time.perf_counter() - t0
    print(f"  runtime = {dt:.2f}s")
    print(f"  n_rebalances = {len(result.rebalance_indices)}")
    print(f"  portfolio_returns shape = {result.portfolio_returns.shape}")
    print(f"  metrics:")
    for k, v in result.metrics.items():
        print(f"    {k:15s} = {v:+.4f}")

    # --- Test 2: compare three estimators (all min-variance) ---
    print("\n[Test 2: compare 3 estimators, monthly min-variance rebalance]")
    results = []
    t0 = time.perf_counter()
    for est in ["sample", "ledoit_wolf", "ewma"]:
        cfg = BacktestConfig(estimator=est, objective="min_variance",
                             lookback=252, rebalance_every=21)
        results.append(run_backtest(returns, cfg))
    dt = time.perf_counter() - t0
    print(f"  runtime for 3 configs = {dt:.2f}s")
    df = summarize_results(results)
    print(df[["estimator", "objective", "sharpe", "annual_vol",
              "max_drawdown", "cagr"]].to_string(index=False))

    # --- Test 3: max-Sharpe comparison ---
    print("\n[Test 3: compare 3 estimators, monthly max-Sharpe rebalance]")
    results = []
    t0 = time.perf_counter()
    for est in ["sample", "ledoit_wolf", "ewma"]:
        cfg = BacktestConfig(estimator=est, objective="max_sharpe",
                             lookback=252, rebalance_every=21)
        results.append(run_backtest(returns, cfg))
    dt = time.perf_counter() - t0
    print(f"  runtime for 3 configs = {dt:.2f}s")
    df = summarize_results(results)
    print(df[["estimator", "objective", "sharpe", "annual_vol",
              "max_drawdown", "cagr"]].to_string(index=False))

    # --- Test 4: the full grid (54 configs) — scale check ---
    print(f"\n[Test 4: full config grid ({len(default_config_grid())} configs)]")
    grid = default_config_grid()
    t0 = time.perf_counter()
    grid_results = [run_backtest(returns, c) for c in grid]
    dt = time.perf_counter() - t0
    print(f"  runtime = {dt:.1f}s  ({dt/len(grid)*1000:.1f}ms per config)")
    df = summarize_results(grid_results)

    # Show the top-5 and bottom-5 by Sharpe
    df_sorted = df.sort_values("sharpe", ascending=False)
    print("\n  Top 5 by Sharpe:")
    print(df_sorted.head(5)[["label", "sharpe", "max_drawdown", "cagr"]]
          .to_string(index=False))
    print("\n  Bottom 5 by Sharpe:")
    print(df_sorted.tail(5)[["label", "sharpe", "max_drawdown", "cagr"]]
          .to_string(index=False))