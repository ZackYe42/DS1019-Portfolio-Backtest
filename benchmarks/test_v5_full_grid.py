"""
Comprehensive v5 test: all estimators + both objectives + realistic workload.
Verifies both correctness (vs v1) and performance (vs v4).
"""
import time
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v1_baseline, v4_multiproc, v5_gpu

    print("=" * 70)
    print("v5 comprehensive test: all estimators + objectives")
    print("=" * 70)

    returns = load_returns().to_numpy()
    all_configs = default_config_grid()

    # --- Correctness test: small workload covering all config types ---
    # Pick configs that span the grid: sample/ledoit/ewma × min_var/max_sharpe
    # at 2 lookbacks × 1 rebalance each = up to 12 configs
    probe_configs = []
    for cfg in all_configs:
        if cfg.lookback == 252 and cfg.rebalance_every == 21:
            probe_configs.append(cfg)
    print(f"\nProbe configs (lb=252, rb=21): {len(probe_configs)}")
    for i, c in enumerate(probe_configs):
        print(f"  cfg {i}: {c.label()}")

    n_bootstrap = 10
    print(f"\nCorrectness run: {len(probe_configs)} configs x {n_bootstrap} bootstrap")

    print("\nRunning v1...")
    t0 = time.perf_counter()
    v1r = v1_baseline.run(returns, probe_configs, n_bootstrap=n_bootstrap,
                           seed=42, show_progress=False)
    print(f"  v1: {v1r.wall_time:.1f}s")

    print("Running v5...")
    t0 = time.perf_counter()
    v5r = v5_gpu.run(returns, probe_configs, n_bootstrap=n_bootstrap,
                       seed=42, show_progress=False)
    print(f"  v5: {v5r.wall_time:.1f}s  ({v1r.wall_time/v5r.wall_time:.1f}x)")
    print(f"  GPU fraction: {v5r.gpu_fraction:.0%}")

    # Per-config comparison
    df1 = v1r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
    df5 = v5r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)

    print("\n  Per-config Sharpe diff:")
    max_diff_gpu = 0.0
    max_diff_cpu = 0.0
    for cfg_idx, cfg in enumerate(probe_configs):
        s1 = df1[df1["config_idx"] == cfg_idx]["sharpe"].to_numpy()
        s5 = df5[df5["config_idx"] == cfg_idx]["sharpe"].to_numpy()
        d = np.abs(s1 - s5).max()
        path = "GPU" if cfg.objective == "min_variance" else "CPU"
        if path == "GPU":
            max_diff_gpu = max(max_diff_gpu, d)
        else:
            max_diff_cpu = max(max_diff_cpu, d)
        print(f"    cfg{cfg_idx} [{path}] {cfg.label():50s}: {d:.2e}")

    TOL_GPU = 1e-4    # float32 has ~1e-7 epsilon but Sharpe can amplify
    TOL_CPU = 1e-6    # CPU path is v3 which matches v1 to 1e-6
    gpu_ok = max_diff_gpu < TOL_GPU
    cpu_ok = max_diff_cpu < TOL_CPU
    print(f"\n  GPU path (min_variance):  max diff {max_diff_gpu:.2e}  {'PASS' if gpu_ok else 'FAIL'}")
    print(f"  CPU path (max_sharpe):    max diff {max_diff_cpu:.2e}  {'PASS' if cpu_ok else 'FAIL'}")

    # --- Performance: realistic workload ---
    print("\n" + "=" * 70)
    print("Performance test: larger workload")
    print("=" * 70)

    perf_configs = all_configs[:18]  # half the grid
    n_bootstrap_perf = 20
    total = len(perf_configs) * n_bootstrap_perf
    print(f"Workload: {len(perf_configs)} configs x {n_bootstrap_perf} bootstrap "
          f"= {total} backtests")
    n_gpu_cfgs = sum(1 for c in perf_configs if c.objective == "min_variance")
    n_cpu_cfgs = len(perf_configs) - n_gpu_cfgs
    print(f"  GPU path: {n_gpu_cfgs} configs, CPU path: {n_cpu_cfgs} configs")

    print("\nRunning v4 (multiprocessing, 8 workers)...")
    v4r = v4_multiproc.run(returns, perf_configs, n_bootstrap=n_bootstrap_perf,
                            n_workers=8, seed=42, show_progress=False)
    print(f"  v4 wall time: {v4r.wall_time:.1f}s  ({v4r.wall_time/total*1000:.0f} ms/backtest)")

    print("\nRunning v5 GPU...")
    v5r = v5_gpu.run(returns, perf_configs, n_bootstrap=n_bootstrap_perf,
                      seed=42, show_progress=False)
    print(f"  v5 wall time: {v5r.wall_time:.1f}s  ({v5r.wall_time/total*1000:.0f} ms/backtest)")
    print(f"  v5 / v4:      {v4r.wall_time/v5r.wall_time:.2f}x")

    # Estimate v1 equivalent from first test
    v1_per_bt_ms = v1r.wall_time / (len(probe_configs) * n_bootstrap) * 1000
    v1_est_total = v1_per_bt_ms * total / 1000
    print(f"\n  v1 projected: {v1_est_total:.0f}s (est from smaller run)")
    print(f"  v5 / v1 projected: {v1_est_total/v5r.wall_time:.0f}x")


if __name__ == "__main__":
    main()