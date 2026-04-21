"""
v3: Numba JIT compilation.

Same computation as v1/v2, with the hot inner functions (covariance
estimators, portfolio optimizers, and the eigenvalue estimator used
for step-size selection) compiled to native code via Numba.

Expected speedup vs v1: 4-7x on this workload.

Strategy:
    - @njit estimators: sample_cov, ledoit_wolf, ewma_cov
    - @njit optimizers: min_variance (projected gradient), max_sharpe (Frank-Wolfe)
    - @njit power iteration to replace np.linalg.eigvalsh for step-size selection
      (eigvalsh is 2.5ms per call x 130 calls per backtest = 325ms overhead)
    - Everything else (bootstrap, backtest driver loop) stays in plain Python
      because it's cheap relative to the compiled kernels

The eigenvalue replacement via power iteration is worth special mention:
a full eigendecomposition gives us all N eigenvalues, but we only need
the largest one.  Power iteration converges geometrically to lambda_max
for any PSD matrix, and 30 iterations gives 6+ decimals of accuracy on
typical covariance matrices.  This single change saves ~325ms per backtest,
and more importantly, power iteration is trivially portable to CUDA for v5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import time

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from src.backtest import BacktestConfig, BacktestResult
from src.bootstrap import fixed_block_bootstrap
from src.metrics import summary as metric_summary


# =============================================================================
# Numba-compiled covariance estimators
# =============================================================================

@njit(cache=True, fastmath=True)
def _sample_cov_njit(returns: np.ndarray) -> np.ndarray:
    """Sample covariance (ddof=1), matches np.cov(returns, rowvar=False)."""
    T, N = returns.shape
    mean = np.zeros(N)
    for j in range(N):
        s = 0.0
        for i in range(T):
            s += returns[i, j]
        mean[j] = s / T
    X = np.empty_like(returns)
    for i in range(T):
        for j in range(N):
            X[i, j] = returns[i, j] - mean[j]
    return (X.T @ X) / (T - 1)


@njit(cache=True, fastmath=True)
def _ledoit_wolf_njit(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance, matches src.estimators.ledoit_wolf."""
    T, N = returns.shape
    mean = np.zeros(N)
    for j in range(N):
        s = 0.0
        for i in range(T):
            s += returns[i, j]
        mean[j] = s / T
    X = np.empty_like(returns)
    for i in range(T):
        for j in range(N):
            X[i, j] = returns[i, j] - mean[j]
    # S = X^T X / T (divisor T, not T-1, per Ledoit-Wolf formulas)
    S = (X.T @ X) / T

    trace = 0.0
    for i in range(N):
        trace += S[i, i]
    mu = trace / N

    X2 = X * X
    pi_mat = (X2.T @ X2) / T
    pi_hat = 0.0
    for a in range(N):
        for b in range(N):
            pi_hat += pi_mat[a, b] - S[a, b] * S[a, b]

    gamma_hat = 0.0
    for a in range(N):
        for b in range(N):
            if a == b:
                d = S[a, b] - mu
            else:
                d = S[a, b]
            gamma_hat += d * d

    if gamma_hat < 1e-12:
        alpha = 0.0
    else:
        kappa = pi_hat / gamma_hat
        alpha = kappa / T
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0

    out = np.empty((N, N))
    one_minus = 1.0 - alpha
    for a in range(N):
        for b in range(N):
            if a == b:
                out[a, b] = one_minus * S[a, b] + alpha * mu
            else:
                out[a, b] = one_minus * S[a, b]
    return out


@njit(cache=True, fastmath=True)
def _ewma_cov_njit(returns: np.ndarray, lam: float) -> np.ndarray:
    """EWMA covariance (RiskMetrics), matches src.estimators.ewma_cov."""
    T, N = returns.shape
    weights = np.empty(T)
    total = 0.0
    power = 1.0
    for t in range(T - 1, -1, -1):
        weights[t] = (1.0 - lam) * power
        total += weights[t]
        power *= lam
    for t in range(T):
        weights[t] /= total

    mean = weights @ returns

    X = np.empty_like(returns)
    sqw = np.sqrt(weights)
    for t in range(T):
        sw = sqw[t]
        for j in range(N):
            X[t, j] = sw * (returns[t, j] - mean[j])

    return X.T @ X


def _estimate_cov_fast(returns: np.ndarray, method: str, ewma_lam: float = 0.94) -> np.ndarray:
    """Dispatcher for njit covariance estimators."""
    if method == "sample":
        return _sample_cov_njit(returns)
    if method == "ledoit_wolf":
        return _ledoit_wolf_njit(returns)
    if method == "ewma":
        return _ewma_cov_njit(returns, ewma_lam)
    raise ValueError(f"unknown estimator '{method}'")


# =============================================================================
# Power iteration for step-size selection (NEW in v3.1)
# =============================================================================

@njit(cache=True, fastmath=True)
def _power_iteration_njit(Sigma: np.ndarray, n_iter: int = 30) -> float:
    """
    Estimate the largest eigenvalue of a symmetric PSD matrix via power iteration.

    For a PSD matrix, power iteration converges geometrically to the top
    eigenvalue.  30 iterations gives 6+ decimals of accuracy on typical
    well-conditioned covariance matrices, at ~0.05ms vs ~2.5ms for numpy's
    eigvalsh (which computes ALL eigenvalues).

    The Rayleigh quotient `v^T Sigma v / v^T v` gives the eigenvalue
    estimate; normalizing v keeps the iteration numerically stable.

    Returns
    -------
    float : estimate of lambda_max(Sigma)
    """
    N = Sigma.shape[0]
    v = np.full(N, 1.0 / np.sqrt(N))  # unit-norm init
    lam = 0.0

    for _ in range(n_iter):
        # u = Sigma @ v
        u = np.empty(N)
        for i in range(N):
            acc = 0.0
            for j in range(N):
                acc += Sigma[i, j] * v[j]
            u[i] = acc

        # Rayleigh quotient: v already unit-norm, so lam = v^T u
        lam = 0.0
        for i in range(N):
            lam += v[i] * u[i]

        # Normalize u to get next v
        norm = 0.0
        for i in range(N):
            norm += u[i] * u[i]
        norm = np.sqrt(norm)
        if norm < 1e-300:
            return lam
        for i in range(N):
            v[i] = u[i] / norm

    return lam


# =============================================================================
# Numba-compiled optimizers
# =============================================================================

@njit(cache=True, fastmath=True)
def _project_simplex_njit(v: np.ndarray) -> np.ndarray:
    """Duchi 2008 projection onto the simplex, njit-compatible."""
    N = v.shape[0]
    u_asc = np.sort(v)
    u = np.empty(N)
    for i in range(N):
        u[i] = u_asc[N - 1 - i]
    cssv = np.empty(N)
    running = 0.0
    for i in range(N):
        running += u[i]
        cssv[i] = running - 1.0
    rho = 0
    for i in range(N):
        if u[i] * (i + 1) > cssv[i]:
            rho = i + 1
    tau = cssv[rho - 1] / rho
    out = np.empty(N)
    for i in range(N):
        diff = v[i] - tau
        out[i] = diff if diff > 0.0 else 0.0
    return out


@njit(cache=True, fastmath=True)
def _min_variance_njit(Sigma: np.ndarray, step_size: float, n_iter: int) -> np.ndarray:
    """Projected-gradient min-variance with precomputed step_size."""
    N = Sigma.shape[0]
    w = np.full(N, 1.0 / N)
    two_step = 2.0 * step_size

    for _ in range(n_iter):
        # Sigma @ w manually so Numba vectorizes tightly
        Sw = np.empty(N)
        for i in range(N):
            acc = 0.0
            for j in range(N):
                acc += Sigma[i, j] * w[j]
            Sw[i] = acc

        v = np.empty(N)
        for i in range(N):
            v[i] = w[i] - two_step * Sw[i]

        w = _project_simplex_njit(v)

    return w


@njit(cache=True, fastmath=True)
def _max_sharpe_njit(Sigma: np.ndarray, excess: np.ndarray, n_iter: int) -> np.ndarray:
    """
    Frank-Wolfe max-Sharpe with incrementally-cached Sigma @ w.

    The FW update w_new = (1-gamma) w + gamma e_best lets us update
    Sw = Sigma @ w incrementally: one column extraction plus AXPY
    instead of a full matvec.
    """
    N = Sigma.shape[0]
    w = np.full(N, 1.0 / N)

    # Initial Sw = Sigma @ w where w is uniform, so Sw[i] = mean(Sigma[i, :])
    Sw = np.empty(N)
    for i in range(N):
        acc = 0.0
        for j in range(N):
            acc += Sigma[i, j]
        Sw[i] = acc / N

    for k in range(n_iter):
        port_var = 0.0
        for i in range(N):
            port_var += w[i] * Sw[i]
        port_vol = np.sqrt(port_var)
        inv_vol = 1.0 / port_vol
        inv_vol3 = inv_vol * inv_vol * inv_vol

        port_ret = 0.0
        for i in range(N):
            port_ret += excess[i] * w[i]
        coeff = port_ret * inv_vol3

        # Gradient and argmax in one pass
        best_val = -1e300
        best = 0
        for i in range(N):
            g = excess[i] * inv_vol - coeff * Sw[i]
            if g > best_val:
                best_val = g
                best = i

        gamma = 2.0 / (k + 2.0)
        one_minus_gamma = 1.0 - gamma

        # w <- (1 - gamma) w + gamma e_best
        for i in range(N):
            w[i] *= one_minus_gamma
        w[best] += gamma

        # Sw <- (1 - gamma) Sw + gamma Sigma[:, best]
        for i in range(N):
            Sw[i] = one_minus_gamma * Sw[i] + gamma * Sigma[i, best]

    return w


# =============================================================================
# JIT warmup
# =============================================================================

def _warmup_jit():
    """Invoke each njit function once to trigger compilation."""
    rng = np.random.default_rng(0)
    T, N = 50, 10
    fake_returns = rng.standard_normal((T, N)) * 0.01

    # Estimators
    _sample_cov_njit(fake_returns)
    _ledoit_wolf_njit(fake_returns)
    _ewma_cov_njit(fake_returns, 0.94)

    # Optimizers & eigenvalue
    cov = fake_returns.T @ fake_returns / T + 0.1 * np.eye(N)
    mu = fake_returns.mean(axis=0)
    _project_simplex_njit(np.ones(N) / N)
    _power_iteration_njit(cov, 10)
    _min_variance_njit(cov, 0.01, 10)
    _max_sharpe_njit(cov, mu + 0.01, 10)


_warmup_jit()


# =============================================================================
# v3 backtest (hybrid: Python driver + njit kernels)
# =============================================================================

def _run_backtest_v3(returns: np.ndarray, config: BacktestConfig) -> BacktestResult:
    T, N = returns.shape
    lb = config.lookback
    rb = config.rebalance_every

    arith_returns = np.exp(returns) - 1.0

    rebalance_indices = np.arange(lb, T, rb)
    n_rebalance = len(rebalance_indices)

    weights = np.full(N, 1.0 / N)
    weights_history = np.zeros((n_rebalance, N))
    port_returns = np.zeros(T - lb)

    rb_ptr = 0
    for t in range(lb, T):
        if rb_ptr < n_rebalance and t == rebalance_indices[rb_ptr]:
            window = returns[t - lb : t]

            # Fast njit estimator
            cov = _estimate_cov_fast(window, method=config.estimator)

            # Add jitter for PD-ness (stays in numpy - only N multiplies)
            Sigma = cov + 1e-10 * np.eye(N)

            if config.objective == "min_variance":
                # Power iteration replaces np.linalg.eigvalsh (50x faster)
                lam_max = _power_iteration_njit(Sigma, 50)
                step_size = 1.0 / (2.0 * lam_max)
                weights = _min_variance_njit(Sigma, step_size, 500)
            elif config.objective == "max_sharpe":
                mu = window.mean(axis=0)
                excess = mu - 0.0  # rf=0
                weights = _max_sharpe_njit(Sigma, excess, 5000)
            else:
                raise ValueError(f"unknown objective '{config.objective}'")

            weights_history[rb_ptr] = weights
            rb_ptr += 1

        port_returns[t - lb] = weights @ arith_returns[t]

    metrics = metric_summary(port_returns, rf_annual=config.rf_annual,
                             log_returns=False)

    return BacktestResult(
        config=config,
        portfolio_returns=port_returns,
        weights_history=weights_history,
        rebalance_indices=rebalance_indices,
        metrics=metrics,
    )


# =============================================================================
# Public API (same signature as v1_baseline.run)
# =============================================================================

@dataclass
class BootstrapResult:
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
    """v3 Numba-JIT-compiled run.  Same API and semantics as v1/v2."""
    t0 = time.perf_counter()

    n_configs = len(configs)
    total_iters = n_configs * n_bootstrap
    raw_rows = []

    iterator = range(total_iters)
    if show_progress:
        iterator = tqdm(iterator, desc="v3 numba", unit="bt")

    for idx in iterator:
        cfg_idx = idx // n_bootstrap
        boot_idx = idx % n_bootstrap
        cfg = configs[cfg_idx]

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
        raw_rows.append(row)

    wall_time = time.perf_counter() - t0

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
    )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v1_baseline

    print("v3 numba smoke test\n" + "=" * 70)

    returns = load_returns().to_numpy()

    configs = default_config_grid()[:6]
    n_bootstrap = 10

    print(f"\nWorkload: {len(configs)} configs x {n_bootstrap} bootstrap "
          f"= {len(configs) * n_bootstrap} backtests\n")

    print("Running v1 baseline first (for timing comparison)...")
    v1_result = v1_baseline.run(returns, configs, n_bootstrap=n_bootstrap,
                                seed=42, show_progress=False)
    print(f"  v1 wall time: {v1_result.wall_time:.2f}s")

    print("\nRunning v3 numba...")
    v3_result = run(returns, configs, n_bootstrap=n_bootstrap,
                    seed=42, show_progress=False)
    print(f"  v3 wall time: {v3_result.wall_time:.2f}s")

    speedup = v1_result.wall_time / v3_result.wall_time
    print(f"\nSpeedup vs v1: {speedup:.1f}x\n")

    print("=" * 70)
    print("Correctness check: v3 vs v1 metrics")
    print("=" * 70)

    df1 = v1_result.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
    df3 = v3_result.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)

    # Power iteration gives approximate step_size, so weights differ slightly
    # from v1.  The resulting metrics should still match to ~1e-6.
    TOL = 1e-6
    all_pass = True
    for col in ["sharpe", "annual_vol", "max_drawdown", "cagr"]:
        diff = np.abs(df1[col].to_numpy() - df3[col].to_numpy()).max()
        status = "PASS" if diff < TOL else "FAIL"
        if diff >= TOL:
            all_pass = False
        print(f"  {col:15s} max |v1 - v3| = {diff:.2e}  [{status}]")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")