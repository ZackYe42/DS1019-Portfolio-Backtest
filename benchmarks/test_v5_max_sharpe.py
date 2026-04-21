"""
Validate v5 GPU max-Sharpe kernel against v3 Numba CPU reference.

Same pattern as test_v5_kernels.py but for max-Sharpe.
"""
import time
import warnings
import numpy as np

# Hide the occupancy warnings; they're informational not errors.
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.versions.v3_numba import _max_sharpe_njit
    from src.versions.v5_gpu_kernels import (
        solve_max_sharpe_batch_gpu, N_ASSETS,
    )

    print("=" * 70)
    print("v5 GPU kernel validation: max-Sharpe")
    print("=" * 70)

    returns = load_returns().to_numpy()
    T, N = returns.shape
    print(f"Loaded returns shape: ({T}, {N})")
    if N != N_ASSETS:
        returns = returns[:, :N_ASSETS]
        N = N_ASSETS

    rng = np.random.default_rng(42)
    B = 100
    print(f"\nGenerating {B} test (Sigma, excess) pairs...")

    Sigmas = np.empty((B, N, N), dtype=np.float64)
    excess_all = np.empty((B, N), dtype=np.float64)
    for b in range(B):
        start = rng.integers(0, T - 252)
        window = returns[start:start + 252]
        Sigmas[b] = np.cov(window, rowvar=False) + 1e-10 * np.eye(N)
        # Excess returns = historical mean (simple proxy, same as CPU path)
        mu = window.mean(axis=0)
        excess_all[b] = mu  # rf = 0

    # --- CPU reference ---
    print(f"\nSolving on CPU (v3 Numba, {B} problems serially)...")
    cpu_weights = np.zeros((B, N), dtype=np.float64)
    t0 = time.perf_counter()
    for b in range(B):
        cpu_weights[b] = _max_sharpe_njit(Sigmas[b].copy(), excess_all[b], 5000)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU time: {cpu_time*1000:.1f} ms  ({cpu_time/B*1000:.2f} ms/problem)")

    # --- GPU ---
    print(f"\nSolving on GPU (v5 kernel, {B} problems in parallel)...")
    print("  (warming up kernel JIT...)")
    _ = solve_max_sharpe_batch_gpu(Sigmas[:2], excess_all[:2])

    t0 = time.perf_counter()
    gpu_weights = solve_max_sharpe_batch_gpu(Sigmas, excess_all)
    gpu_time = time.perf_counter() - t0
    print(f"  GPU time: {gpu_time*1000:.1f} ms  ({gpu_time/B*1000:.3f} ms/problem)")
    print(f"  Speedup:  {cpu_time/gpu_time:.1f}x")

    # --- Correctness: Sharpe values should match ---
    print("\n" + "=" * 70)
    print("Correctness: Sharpe ratio (objective) should match")
    print("=" * 70)

    def sharpe(w, mu, S):
        return float((w @ mu) / np.sqrt(w @ S @ w))

    max_sharpe_rel_diff = 0.0
    max_weight_diff = 0.0
    max_sum_err = 0.0
    min_weight = 0.0

    for b in range(B):
        wc = cpu_weights[b]
        wg = gpu_weights[b].astype(np.float64)
        S = Sigmas[b]
        mu = excess_all[b]

        s_cpu = sharpe(wc, mu, S)
        s_gpu = sharpe(wg, mu, S)
        rel = abs(s_cpu - s_gpu) / max(abs(s_cpu), 1e-300)

        max_sharpe_rel_diff = max(max_sharpe_rel_diff, rel)
        max_weight_diff = max(max_weight_diff, np.abs(wc - wg).max())
        max_sum_err = max(max_sum_err, abs(wg.sum() - 1.0))
        min_weight = min(min_weight, wg.min())

    print(f"  max relative |sharpe_cpu - sharpe_gpu|: {max_sharpe_rel_diff:.2e}")
    print(f"  max |w_cpu - w_gpu|_inf:                {max_weight_diff:.2e}")
    print(f"  max |sum(w_gpu) - 1|:                   {max_sum_err:.2e}")
    print(f"  min w_gpu over all batch:               {min_weight:.2e}")

    sharpe_ok = max_sharpe_rel_diff < 1e-3
    sum_ok = max_sum_err < 1e-3
    nonneg_ok = min_weight >= -1e-4

    print(f"\n  Sharpe match:         {'PASS' if sharpe_ok else 'FAIL'}")
    print(f"  Weights sum to 1:     {'PASS' if sum_ok else 'FAIL'}")
    print(f"  Weights nonnegative:  {'PASS' if nonneg_ok else 'FAIL'}")


if __name__ == "__main__":
    main()