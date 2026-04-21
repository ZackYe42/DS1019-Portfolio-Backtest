"""Scale-up test for the B.3 full-backtest kernel."""
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
    from src.versions.v5_gpu_kernels import run_full_backtest_gpu, N_ASSETS
    from src.bootstrap import fixed_block_bootstrap

    returns = load_returns().to_numpy()[:, :N_ASSETS]
    T, N = returns.shape

    lookback = 252
    rebalance_every = 21
    n_rebalance = 0
    t = lookback
    while t < T:
        n_rebalance += 1
        t += rebalance_every

    print(f"{'B':>5s} {'CPU est':>10s} {'GPU ms':>10s} {'speedup':>10s} "
          f"{'GPU ms/bt':>10s}")
    print("-" * 55)

    # Pre-build a pool of bootstrap samples (largest size once, slice for smaller tests)
    MAX_B = 500
    print(f"(building {MAX_B} bootstrap samples ~20s)")
    returns_all = np.empty((MAX_B, T, N), dtype=np.float64)
    for b in range(MAX_B):
        rng = np.random.default_rng(42 + b)
        returns_all[b] = fixed_block_bootstrap(returns, block_size=20, rng=rng)

    # Warmup GPU
    _ = run_full_backtest_gpu(returns_all[:4], lookback, rebalance_every)

    # Measure single-backtest CPU time on one sample
    t0 = time.perf_counter()
    for _ in range(3):
        t_now = lookback
        while t_now < T:
            window = returns_all[0, t_now - lookback:t_now]
            cov = _sample_cov_njit(window) + 1e-10 * np.eye(N)
            lam = _power_iteration_njit(cov, 50)
            step = 1.0 / (2.0 * lam)
            _ = _min_variance_njit(cov, step, 500)
            t_now += rebalance_every
    cpu_ms_per_backtest = (time.perf_counter() - t0) / 3 * 1000
    print(f"(CPU per-backtest baseline: {cpu_ms_per_backtest:.1f} ms)")
    print()

    for B in [10, 50, 100, 200, 500]:
        # GPU
        t0 = time.perf_counter()
        _ = run_full_backtest_gpu(returns_all[:B], lookback, rebalance_every)
        gpu_ms = (time.perf_counter() - t0) * 1000
        cpu_est_ms = cpu_ms_per_backtest * B
        speedup = cpu_est_ms / gpu_ms
        print(f"{B:>5d} {cpu_est_ms:>10.0f} {gpu_ms:>10.1f} "
              f"{speedup:>9.1f}x {gpu_ms/B:>10.2f}")


if __name__ == "__main__":
    main()