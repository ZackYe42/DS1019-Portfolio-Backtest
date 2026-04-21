"""
Validate the B.3 full-backtest kernel against v3 CPU.

Runs B identical backtests (min_variance + sample_cov) on both and compares
the resulting weight histories.
"""
import time
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.versions.v3_numba import (
        _sample_cov_njit, _min_variance_njit, _power_iteration_njit,
    )
    from src.versions.v5_gpu_kernels import (
        run_full_backtest_gpu, N_ASSETS,
    )
    from src.bootstrap import fixed_block_bootstrap

    returns = load_returns().to_numpy()[:, :N_ASSETS]
    T, N = returns.shape
    print(f"Loaded returns: ({T}, {N})")

    # --- Build test batch: B bootstrap samples ---
    B = 50
    lookback = 252
    rebalance_every = 21
    n_rebalance = 0
    t = lookback
    while t < T:
        n_rebalance += 1
        t += rebalance_every
    print(f"Batch: B={B}, lookback={lookback}, rebalance_every={rebalance_every}")
    print(f"Per backtest: {n_rebalance} rebalances")

    print(f"\nBuilding {B} bootstrap samples...")
    returns_batch = np.empty((B, T, N), dtype=np.float64)
    for b in range(B):
        rng = np.random.default_rng(42 + b)
        returns_batch[b] = fixed_block_bootstrap(returns, block_size=20, rng=rng)

    # --- CPU reference: run full backtest on each sample ---
    print(f"\nRunning v3 CPU reference ({B} backtests)...")
    cpu_weights = np.zeros((B, n_rebalance, N), dtype=np.float64)
    t_cpu = time.perf_counter()
    for b in range(B):
        t_now = lookback
        rb_idx = 0
        while t_now < T:
            window = returns_batch[b, t_now - lookback:t_now]
            cov = _sample_cov_njit(window) + 1e-10 * np.eye(N)
            lam = _power_iteration_njit(cov, 50)
            step = 1.0 / (2.0 * lam)
            cpu_weights[b, rb_idx] = _min_variance_njit(cov, step, 500)
            t_now += rebalance_every
            rb_idx += 1
    cpu_time = time.perf_counter() - t_cpu
    print(f"  CPU time: {cpu_time:.2f}s  ({cpu_time/B*1000:.1f} ms/backtest)")

    # --- GPU run ---
    print(f"\nRunning GPU B.3 kernel ({B} backtests in parallel)...")
    print("  (warming up kernel JIT — may take 30-60s on first run)...")
    _ = run_full_backtest_gpu(returns_batch[:2], lookback, rebalance_every)

    t_gpu = time.perf_counter()
    gpu_weights = run_full_backtest_gpu(returns_batch, lookback, rebalance_every)
    gpu_time = time.perf_counter() - t_gpu
    print(f"  GPU time: {gpu_time*1000:.1f} ms  ({gpu_time/B*1000:.2f} ms/backtest)")
    print(f"  Speedup:  {cpu_time/gpu_time:.1f}x")

    # --- Correctness check (per-rebalance variance match) ---
    print("\n" + "=" * 70)
    print("Correctness: variance (objective) at each rebalance")
    print("=" * 70)

    max_var_rel = 0.0
    max_sum_err = 0.0
    min_weight_gpu = 0.0

    for b in range(B):
        for r in range(n_rebalance):
            wc = cpu_weights[b, r]
            wg = gpu_weights[b, r].astype(np.float64)
            t_now = lookback + r * rebalance_every
            window = returns_batch[b, t_now - lookback:t_now]
            Sigma = _sample_cov_njit(window) + 1e-10 * np.eye(N)
            var_c = wc @ Sigma @ wc
            var_g = wg @ Sigma @ wg
            rel = abs(var_c - var_g) / max(var_c, 1e-300)
            max_var_rel = max(max_var_rel, rel)
            max_sum_err = max(max_sum_err, abs(wg.sum() - 1.0))
            min_weight_gpu = min(min_weight_gpu, wg.min())

    print(f"  max relative |var_cpu - var_gpu|: {max_var_rel:.2e}")
    print(f"  max |sum(w_gpu) - 1|:             {max_sum_err:.2e}")
    print(f"  min w_gpu (over all rebalances):  {min_weight_gpu:.2e}")

    var_ok = max_var_rel < 1e-3
    sum_ok = max_sum_err < 1e-4
    nonneg_ok = min_weight_gpu >= -1e-4

    print(f"\n  Variance match:      {'PASS' if var_ok else 'FAIL'}")
    print(f"  Weights sum to 1:    {'PASS' if sum_ok else 'FAIL'}")
    print(f"  Weights nonnegative: {'PASS' if nonneg_ok else 'FAIL'}")


if __name__ == "__main__":
    main()