"""Test v5 GPU scaling across different batch sizes."""
import time
import numpy as np


def main():
    from src.data_loader import load_returns
    from src.versions.v3_numba import _power_iteration_njit, _min_variance_njit
    from src.versions.v5_gpu_kernels import solve_min_variance_batch_gpu

    returns = load_returns().to_numpy()
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("v5 GPU scaling test: how does speedup grow with batch size?")
    print("=" * 70)
    print(f"{'B':>8s} {'CPU ms':>12s} {'GPU ms':>12s} {'speedup':>10s} "
          f"{'GPU us/prob':>12s}")
    print("-" * 62)

    # Build ONE big pool of covariance matrices, then slice it
    MAX_B = 5000
    print(f"(building {MAX_B} test covariances, takes ~10s)")
    Sigmas_all = np.empty((MAX_B, 96, 96), dtype=np.float64)
    for b in range(MAX_B):
        start = rng.integers(0, 2766 - 252)
        window = returns[start:start + 252, :96]
        Sigmas_all[b] = np.cov(window, rowvar=False) + 1e-10 * np.eye(96)

    # Warmup GPU
    _ = solve_min_variance_batch_gpu(Sigmas_all[:4])

    for B in [10, 50, 100, 500, 1000, 2000, 5000]:
        S_batch = Sigmas_all[:B]

        # CPU (sample to avoid absurd waits at large B)
        sample_size = min(B, 50)
        t0 = time.perf_counter()
        for b in range(sample_size):
            S = S_batch[b].copy()
            lam = _power_iteration_njit(S, 50)
            _ = _min_variance_njit(S, 1.0 / (2.0 * lam), 500)
        cpu_time_per_problem = (time.perf_counter() - t0) / sample_size
        cpu_total_est = cpu_time_per_problem * B

        # GPU (actual full batch)
        t0 = time.perf_counter()
        _ = solve_min_variance_batch_gpu(S_batch)
        gpu_time = time.perf_counter() - t0
        gpu_us_per_problem = gpu_time / B * 1e6

        print(f"{B:>8d} {cpu_total_est*1000:>10.1f}   {gpu_time*1000:>10.1f}   "
              f"{cpu_total_est/gpu_time:>8.1f}x   {gpu_us_per_problem:>10.1f}")


if __name__ == "__main__":
    main()