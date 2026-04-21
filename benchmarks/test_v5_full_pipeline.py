"""
Full v5 pipeline test on a realistic workload.

Validates:
    1. The reduction fix doesn't break min_variance correctness
    2. Full config grid (all 9 estimator/objective combos) runs correctly
    3. v5 beats v4 on a realistic workload
    4. Timing baseline for deciding whether to add GPU max_sharpe kernels

Workload: full 54-config grid x 30 bootstrap = 1620 backtests.
Expected runtime: ~3-5 minutes.
"""
import time
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v1_baseline, v3_numba, v4_multiproc, v5_gpu

    returns = load_returns().to_numpy()
    configs = default_config_grid()
    n_bootstrap = 30

    n_configs = len(configs)
    n_minvar = sum(1 for c in configs if c.objective == "min_variance")
    n_maxsharpe = n_configs - n_minvar
    total = n_configs * n_bootstrap

    print("=" * 70)
    print("v5 full-pipeline test")
    print("=" * 70)
    print(f"Configs: {n_configs} ({n_minvar} min_variance, {n_maxsharpe} max_sharpe)")
    print(f"Bootstrap: {n_bootstrap}")
    print(f"Total: {total} backtests")
    print()

    # --- Estimate v1 baseline time (don't actually run it — would take ~50 min) ---
    print("Calibrating v1 baseline (short run to estimate)...")
    calibration_configs = configs[:3]
    calibration_boot = 3
    t0 = time.perf_counter()
    _ = v1_baseline.run(returns, calibration_configs, n_bootstrap=calibration_boot,
                         seed=42, show_progress=False)
    cal_time = time.perf_counter() - t0
    v1_per_bt = cal_time / (len(calibration_configs) * calibration_boot)
    v1_projected = v1_per_bt * total
    print(f"  v1 per-backtest: {v1_per_bt*1000:.0f}ms")
    print(f"  v1 projected for {total} backtests: {v1_projected:.0f}s "
          f"({v1_projected/60:.1f} min)")

    # --- v3 single-thread ---
    print("\nRunning v3 (Numba single-thread)...")
    t0 = time.perf_counter()
    v3r = v3_numba.run(returns, configs, n_bootstrap=n_bootstrap,
                        seed=42, show_progress=True)
    v3_time = time.perf_counter() - t0
    print(f"  v3 wall time: {v3_time:.1f}s  ({v3_time/total*1000:.0f} ms/backtest)")

    # --- v4 multiprocessing ---
    print("\nRunning v4 (8 workers)...")
    t0 = time.perf_counter()
    v4r = v4_multiproc.run(returns, configs, n_bootstrap=n_bootstrap,
                             n_workers=8, seed=42, show_progress=True)
    v4_time = time.perf_counter() - t0
    print(f"  v4 wall time: {v4_time:.1f}s  ({v4_time/total*1000:.0f} ms/backtest)")

    # --- v5 GPU (min_var on GPU, max_sharpe on CPU fallback) ---
    print("\nRunning v5 (GPU min_var + multiproc max_sharpe)...")
    t0 = time.perf_counter()
    v5r = v5_gpu.run(returns, configs, n_bootstrap=n_bootstrap,
                      seed=42, show_progress=True)
    v5_time = time.perf_counter() - t0
    print(f"  v5 wall time: {v5_time:.1f}s  ({v5_time/total*1000:.0f} ms/backtest)")

    # --- Correctness check vs v3 ---
    print("\n" + "=" * 70)
    print("Correctness: v5 vs v3")
    print("=" * 70)
    df3 = v3r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
    df5 = v5r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)

    max_gpu_diff = 0.0
    max_cpu_diff = 0.0
    for cfg_idx in range(n_configs):
        s3 = df3[df3["config_idx"] == cfg_idx]["sharpe"].to_numpy()
        s5 = df5[df5["config_idx"] == cfg_idx]["sharpe"].to_numpy()
        d = np.abs(s3 - s5).max() if len(s3) == len(s5) else np.inf
        if configs[cfg_idx].objective == "min_variance":
            max_gpu_diff = max(max_gpu_diff, d)
        else:
            max_cpu_diff = max(max_cpu_diff, d)

    print(f"  GPU path (min_var):    max Sharpe diff = {max_gpu_diff:.2e}")
    print(f"  CPU path (max_sharpe): max Sharpe diff = {max_cpu_diff:.2e}")
    TOL_GPU, TOL_CPU = 1e-4, 1e-6
    gpu_ok = max_gpu_diff < TOL_GPU
    cpu_ok = max_cpu_diff < TOL_CPU
    print(f"  GPU path: {'PASS' if gpu_ok else 'FAIL'}")
    print(f"  CPU path: {'PASS' if cpu_ok else 'FAIL'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Version':<30s} {'Wall time':>12s} {'vs v1':>10s} {'ms/bt':>10s}")
    print("-" * 66)
    print(f"{'v1 baseline (projected)':<30s} {v1_projected:>10.0f}s {1:>9.1f}x {v1_per_bt*1000:>10.0f}")
    print(f"{'v3 Numba':<30s} {v3_time:>10.1f}s "
          f"{v1_projected/v3_time:>9.1f}x {v3_time/total*1000:>10.0f}")
    print(f"{'v4 multiproc (8 workers)':<30s} {v4_time:>10.1f}s "
          f"{v1_projected/v4_time:>9.1f}x {v4_time/total*1000:>10.0f}")
    print(f"{'v5 GPU (partial)':<30s} {v5_time:>10.1f}s "
          f"{v1_projected/v5_time:>9.1f}x {v5_time/total*1000:>10.0f}")

    print()
    print(f"v5 / v4 ratio: {v4_time/v5_time:.2f}x "
          f"({'v5 wins' if v5_time < v4_time else 'v4 wins'})")


if __name__ == "__main__":
    main()