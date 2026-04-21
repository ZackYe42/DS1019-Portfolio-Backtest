"""
Validate v5 GPU covariance estimators against v3 Numba CPU versions.

Each test builds a batch of synthetic windows, runs a small "just the
estimator" kernel on the GPU, and compares to v3's CPU output.
"""
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    import math
    from numba import cuda, float32
    from src.versions.v3_numba import (
        _sample_cov_njit, _ledoit_wolf_njit, _ewma_cov_njit,
    )
    from src.versions.v5_gpu_kernels import (
        _sample_cov_cooperative, _ledoit_wolf_cooperative,
        _ewma_cov_cooperative, N_ASSETS,
    )

    # -- Test kernels that wrap the device functions --

    @cuda.jit
    def sample_cov_test_kernel(windows, covs):
        bid = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        B = windows.shape[0]
        if bid >= B:
            return
        T = windows.shape[1]
        N = N_ASSETS

        s_mean = cuda.shared.array(shape=N_ASSETS, dtype=float32)
        s_cov  = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)

        # Run the estimator on block bid's window
        _sample_cov_cooperative(windows[bid], s_mean, s_cov, tid, N, T)

        # Write s_cov to global
        if tid < N:
            for j in range(N):
                covs[bid, tid, j] = s_cov[tid, j]

    @cuda.jit
    def ledoit_wolf_test_kernel(windows, covs):
        bid = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        B = windows.shape[0]
        if bid >= B:
            return
        T = windows.shape[1]
        N = N_ASSETS

        s_mean    = cuda.shared.array(shape=N_ASSETS, dtype=float32)
        s_cov     = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)
        s_scratch = cuda.shared.array(shape=8, dtype=float32)

        _ledoit_wolf_cooperative(windows[bid], s_mean, s_cov, s_scratch, tid, N, T)

        if tid < N:
            for j in range(N):
                covs[bid, tid, j] = s_cov[tid, j]

    @cuda.jit
    def ewma_cov_test_kernel(windows, covs, lam):
        bid = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        B = windows.shape[0]
        if bid >= B:
            return
        T = windows.shape[1]
        N = N_ASSETS

        s_mean    = cuda.shared.array(shape=N_ASSETS, dtype=float32)
        # T_max = 252 (max lookback we support)
        s_weights = cuda.shared.array(shape=252, dtype=float32)
        s_cov     = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)

        _ewma_cov_cooperative(windows[bid], s_mean, s_weights, s_cov,
                              tid, N, T, lam)

        if tid < N:
            for j in range(N):
                covs[bid, tid, j] = s_cov[tid, j]

    def run_gpu(kernel, windows, *extra_args):
        B, T, N = windows.shape
        d_windows = cuda.to_device(np.ascontiguousarray(windows, dtype=np.float32))
        d_covs = cuda.device_array((B, N, N), dtype=np.float32)
        kernel[B, N](d_windows, d_covs, *extra_args)
        cuda.synchronize()
        return d_covs.copy_to_host()

    # --- Build test windows ---
    np.random.seed(42)
    B = 50
    T = 252
    N = N_ASSETS
    print(f"Generating {B} synthetic windows of shape ({T}, {N})...")
    windows = np.random.randn(B, T, N).astype(np.float64) * 0.01

    # ========== sample_cov ==========
    print("\n" + "=" * 70)
    print("Test: sample_cov")
    print("=" * 70)
    cpu_covs = np.empty((B, N, N), dtype=np.float64)
    for b in range(B):
        cpu_covs[b] = _sample_cov_njit(windows[b])

    print("  Warming GPU...")
    _ = run_gpu(sample_cov_test_kernel, windows[:2])

    gpu_covs = run_gpu(sample_cov_test_kernel, windows)

    max_abs = np.abs(cpu_covs - gpu_covs).max()
    rel = max_abs / np.abs(cpu_covs).max()
    print(f"  max |cpu - gpu|:      {max_abs:.2e}")
    print(f"  relative max diff:    {rel:.2e}")
    print(f"  sample_cov: {'PASS' if rel < 1e-4 else 'FAIL'}")

    # ========== ledoit_wolf ==========
    print("\n" + "=" * 70)
    print("Test: ledoit_wolf")
    print("=" * 70)
    cpu_covs = np.empty((B, N, N), dtype=np.float64)
    for b in range(B):
        cpu_covs[b] = _ledoit_wolf_njit(windows[b])

    print("  Warming GPU...")
    _ = run_gpu(ledoit_wolf_test_kernel, windows[:2])

    gpu_covs = run_gpu(ledoit_wolf_test_kernel, windows)

    max_abs = np.abs(cpu_covs - gpu_covs).max()
    rel = max_abs / np.abs(cpu_covs).max()
    print(f"  max |cpu - gpu|:      {max_abs:.2e}")
    print(f"  relative max diff:    {rel:.2e}")
    print(f"  ledoit_wolf: {'PASS' if rel < 1e-4 else 'FAIL'}")

    # ========== ewma ==========
    print("\n" + "=" * 70)
    print("Test: ewma_cov (lam=0.94)")
    print("=" * 70)
    cpu_covs = np.empty((B, N, N), dtype=np.float64)
    for b in range(B):
        cpu_covs[b] = _ewma_cov_njit(windows[b], 0.94)

    print("  Warming GPU...")
    _ = run_gpu(ewma_cov_test_kernel, windows[:2], np.float32(0.94))

    gpu_covs = run_gpu(ewma_cov_test_kernel, windows, np.float32(0.94))

    max_abs = np.abs(cpu_covs - gpu_covs).max()
    rel = max_abs / np.abs(cpu_covs).max()
    print(f"  max |cpu - gpu|:      {max_abs:.2e}")
    print(f"  relative max diff:    {rel:.2e}")
    print(f"  ewma: {'PASS' if rel < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()