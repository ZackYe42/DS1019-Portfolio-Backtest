"""
Compare v5 vs v4 on min_variance-only workload.

This is v5's best case: no CPU fallback, pure GPU path.  Demonstrates
that when GPU is actually used end-to-end, it decisively beats 8-worker
CPU parallelism.  For the report, this is the "GPU is faster" evidence
point, contrasted with the mixed workload where CPU fallback levels things.
"""
import time
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v3_numba, v4_multiproc, v5_gpu

    returns = load_returns().to_numpy()
    configs = default_config_grid()
    minvar_configs = [c for c in configs if c.objective == "min_variance"]
    print(f"Workload: {len(minvar_configs)} min_variance configs")

    # Run at two bootstrap sizes to see scaling:
    # - small (20): startup overhead matters more
    # - large (50): GPU saturates better
    for n_bootstrap in [20, 50]:
        total = len(minvar_configs) * n_bootstrap
        print(f"\n{'=' * 70}")
        print(f"Bootstrap={n_bootstrap}  (total {total} backtests)")
        print('=' * 70)

        # v3 single-thread (as baseline)
        print("Running v3 (Numba single-thread)...")
        v3r = v3_numba.run(returns, minvar_configs, n_bootstrap=n_bootstrap,
                            seed=42, show_progress=True)
        print(f"  v3: {v3r.wall_time:.1f}s  ({v3r.wall_time/total*1000:.0f} ms/bt)")

        print("\nRunning v4 (8 workers)...")
        v4r = v4_multiproc.run(returns, minvar_configs, n_bootstrap=n_bootstrap,
                                 n_workers=8, seed=42, show_progress=True)
        print(f"  v4: {v4r.wall_time:.1f}s  ({v4r.wall_time/total*1000:.0f} ms/bt)")

        print("\nRunning v5 GPU...")
        v5r = v5_gpu.run(returns, minvar_configs, n_bootstrap=n_bootstrap,
                          seed=42, show_progress=True)
        print(f"  v5: {v5r.wall_time:.1f}s  ({v5r.wall_time/total*1000:.0f} ms/bt)")

        # Speedups
        print(f"\nSpeedups:")
        print(f"  v5 / v4 (GPU vs 8-worker CPU): {v4r.wall_time/v5r.wall_time:.2f}x")
        print(f"  v5 / v3 (GPU vs single-thread): {v3r.wall_time/v5r.wall_time:.2f}x")

        # Correctness
        df3 = v3r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
        df5 = v5r.raw_metrics.sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
        max_diff = np.abs(df3["sharpe"].to_numpy() - df5["sharpe"].to_numpy()).max()
        print(f"  max |Sharpe v3-v5|: {max_diff:.2e}  "
              f"{'PASS' if max_diff < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()