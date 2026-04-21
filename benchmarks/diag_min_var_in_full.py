"""Isolate: does _project_simplex_bisection work correctly inside full-backtest kernel?

Compare the standalone min_variance_batch_kernel (which passes) to
backtest_full_kernel_minvar_sample (which fails) on the same Sigma matrices.
"""
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.versions.v5_gpu_kernels import (
        solve_min_variance_batch_gpu, run_full_backtest_gpu,
        N_ASSETS,
    )
    from src.versions.v3_numba import (
        _sample_cov_njit, _power_iteration_njit, _min_variance_njit,
    )

    returns = load_returns().to_numpy()[:, :N_ASSETS]
    T, N = returns.shape
    lookback = 252
    rebalance_every = 21

    # n_rebalance
    n_rebalance = 0
    t = lookback
    while t < T:
        n_rebalance += 1
        t += rebalance_every
    print(f"n_rebalance per backtest: {n_rebalance}")

    # Build ONE bootstrap sample
    rng = np.random.default_rng(42)
    from src.bootstrap import fixed_block_bootstrap
    boot = fixed_block_bootstrap(returns, block_size=20, rng=rng)

    # Method A: full-backtest kernel, one block
    print("\n[Method A] Full-backtest kernel (1 block, 1 backtest)")
    weights_A = run_full_backtest_gpu(
        boot[None, :, :], lookback, rebalance_every, estimator="sample"
    )  # shape (1, n_rebalance, N)

    # Method B: CPU computes Sigma per rebalance, then batch min_variance kernel
    print("[Method B] CPU Sigmas + batched min_variance kernel")
    Sigmas = np.empty((n_rebalance, N, N), dtype=np.float64)
    for r in range(n_rebalance):
        t_now = lookback + r * rebalance_every
        window = boot[t_now - lookback:t_now]
        cov = _sample_cov_njit(window) + 1e-10 * np.eye(N)
        Sigmas[r] = cov
    weights_B = solve_min_variance_batch_gpu(Sigmas)  # (n_rebalance, N)

    # Method C: pure CPU
    print("[Method C] Pure CPU (v3)")
    weights_C = np.zeros((n_rebalance, N))
    for r in range(n_rebalance):
        Sigma = Sigmas[r]
        lam = _power_iteration_njit(Sigma, 50)
        weights_C[r] = _min_variance_njit(Sigma, 1.0/(2.0*lam), 500)

    # Compare weight by weight, rebalance by rebalance
    print(f"\n{'rebal':>6s} {'var_A':>14s} {'var_B':>14s} {'var_C':>14s} "
          f"{'rel A-C':>10s} {'rel B-C':>10s}")
    print("-" * 72)

    max_A_C = 0.0
    max_B_C = 0.0
    for r in range(min(n_rebalance, 20)):
        wA = weights_A[0, r].astype(np.float64)
        wB = weights_B[r].astype(np.float64)
        wC = weights_C[r]
        vA = wA @ Sigmas[r] @ wA
        vB = wB @ Sigmas[r] @ wB
        vC = wC @ Sigmas[r] @ wC

        relAC = abs(vA - vC) / vC
        relBC = abs(vB - vC) / vC
        max_A_C = max(max_A_C, relAC)
        max_B_C = max(max_B_C, relBC)
        flag = " <--" if relAC > 1e-3 else ""
        print(f"{r:>6d} {vA:>14.6e} {vB:>14.6e} {vC:>14.6e} "
              f"{relAC:>10.2e} {relBC:>10.2e}{flag}")

    print(f"\nMax rel diff A vs C (full-backtest kernel vs CPU): {max_A_C:.2e}")
    print(f"Max rel diff B vs C (batched min_var vs CPU):       {max_B_C:.2e}")

    if max_A_C > 1e-3 and max_B_C < 1e-3:
        print("\n=> The projection works in the standalone kernel but breaks "
              "in the full-backtest kernel.")
        print("   The bug is in how the full-backtest kernel uses _project_simplex_bisection.")


if __name__ == "__main__":
    main()