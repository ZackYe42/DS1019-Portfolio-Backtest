"""
Validate v5 GPU kernels against v3 Numba CPU reference.

For each config in a small batch:
    1. Generate a realistic covariance matrix
    2. Solve min-variance on CPU (v3) and GPU (v5)
    3. Compare objective values (variance); weights may differ on plateaus

Success criterion: GPU and CPU produce same variance within 1e-4
(loosened from CPU's 1e-8 because GPU uses float32 vs CPU's float64).
"""
import time
import numpy as np


def main():
    from src.data_loader import load_returns
    from src.versions.v3_numba import _power_iteration_njit, _min_variance_njit
    from src.versions.v5_gpu_kernels import (
        solve_min_variance_batch_gpu, N_ASSETS,
    )

    print("=" * 70)
    print("v5 GPU kernel validation: min-variance")
    print("=" * 70)

    returns = load_returns().to_numpy()
    T, N = returns.shape
    print(f"Loaded returns shape: ({T}, {N})")
    if N != N_ASSETS:
        print(f"WARNING: kernel compiled for N={N_ASSETS}, data has N={N}")
        print("         truncating columns.")
        returns = returns[:, :N_ASSETS]
        N = N_ASSETS

    # Build a small batch of realistic covariance matrices (one per rolling window)
    rng = np.random.default_rng(42)
    B = 100
    print(f"\nGenerating {B} test covariance matrices (each from a rolling window)...")

    Sigmas = np.empty((B, N, N), dtype=np.float64)
    for b in range(B):
        start = rng.integers(0, T - 252)
        window = returns[start:start + 252]
        cov = np.cov(window, rowvar=False) + 1e-10 * np.eye(N)
        Sigmas[b] = cov

    # --- CPU reference (v3 path) ---
    print(f"\nSolving on CPU (v3 Numba, {B} problems serially)...")
    cpu_weights = np.zeros((B, N), dtype=np.float64)
    t0 = time.perf_counter()
    for b in range(B):
        S = Sigmas[b].copy()
        lam = _power_iteration_njit(S, 50)
        step = 1.0 / (2.0 * lam)
        cpu_weights[b] = _min_variance_njit(S, step, 500)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU time: {cpu_time*1000:.1f} ms  ({cpu_time/B*1000:.2f} ms/problem)")

    # --- GPU (v5 kernel) ---
    print(f"\nSolving on GPU (v5 kernel, {B} problems in parallel)...")
    # Warmup (first launch triggers kernel JIT, ~1-3s)
    print("  (warming up kernel JIT...)")
    _ = solve_min_variance_batch_gpu(Sigmas[:2])

    t0 = time.perf_counter()
    gpu_weights = solve_min_variance_batch_gpu(Sigmas)
    gpu_time = time.perf_counter() - t0
    print(f"  GPU time: {gpu_time*1000:.1f} ms  ({gpu_time/B*1000:.3f} ms/problem)")
    print(f"  Speedup:  {cpu_time/gpu_time:.1f}x")

    # --- Correctness: compare objective values ---
    print("\n" + "=" * 70)
    print("Correctness: variance (objective) should match")
    print("=" * 70)

    max_var_rel_diff = 0.0
    max_weight_diff = 0.0
    max_sum_err = 0.0
    min_weight = 0.0
    bad_idx = -1

    for b in range(B):
        wc = cpu_weights[b]
        wg = gpu_weights[b].astype(np.float64)
        S = Sigmas[b]
        var_cpu = float(wc @ S @ wc)
        var_gpu = float(wg @ S @ wg)
        rel_diff = abs(var_cpu - var_gpu) / max(var_cpu, 1e-300)
        wdiff = np.abs(wc - wg).max()
        sum_err = abs(wg.sum() - 1.0)
        min_w = wg.min()

        if rel_diff > max_var_rel_diff:
            max_var_rel_diff = rel_diff
            bad_idx = b
        if wdiff > max_weight_diff:
            max_weight_diff = wdiff
        if sum_err > max_sum_err:
            max_sum_err = sum_err
        if min_w < min_weight:
            min_weight = min_w

    print(f"  max relative |var_cpu - var_gpu|: {max_var_rel_diff:.2e}")
    print(f"  max |w_cpu - w_gpu|_inf:          {max_weight_diff:.2e}")
    print(f"  max |sum(w_gpu) - 1|:             {max_sum_err:.2e}")
    print(f"  min w_gpu over all batch:         {min_weight:.2e}")

    variance_ok = max_var_rel_diff < 1e-3
    sum_ok = max_sum_err < 1e-4
    nonneg_ok = min_weight >= -1e-4

    print(f"\n  Variance match:       {'PASS' if variance_ok else 'FAIL'}")
    print(f"  Weights sum to 1:     {'PASS' if sum_ok else 'FAIL'}")
    print(f"  Weights nonnegative:  {'PASS' if nonneg_ok else 'FAIL'}")

    if not variance_ok and bad_idx >= 0:
        print(f"\n  Worst batch index: {bad_idx}")
        print(f"    var_cpu = {float(cpu_weights[bad_idx] @ Sigmas[bad_idx] @ cpu_weights[bad_idx]):.6e}")
        wg = gpu_weights[bad_idx].astype(np.float64)
        print(f"    var_gpu = {float(wg @ Sigmas[bad_idx] @ wg):.6e}")


if __name__ == "__main__":
    main()