"""
v5 vs v4 scaling curve on min_variance workload.

Three data points (B=20, B=50, B=100) demonstrate the key insight:
v5 has fixed GPU overhead (~60s) but near-zero marginal cost per
backtest at saturation, while v4 scales linearly with workload.

This is the data plot for the report's GPU section.
"""
import time
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v4_multiproc, v5_gpu

    returns = load_returns().to_numpy()
    configs = default_config_grid()
    minvar_configs = [c for c in configs if c.objective == "min_variance"]
    print(f"Workload: {len(minvar_configs)} min_variance configs")

    results = []  # list of (B, total_bt, v4_time, v5_time, v4_per_bt, v5_per_bt)

    for n_bootstrap in [20, 50, 100]:
        total = len(minvar_configs) * n_bootstrap
        print(f"\n{'=' * 70}")
        print(f"B={n_bootstrap}  ({total} backtests)")
        print('=' * 70)

        print("Running v4 (8 workers)...")
        v4r = v4_multiproc.run(returns, minvar_configs, n_bootstrap=n_bootstrap,
                                 n_workers=8, seed=42, show_progress=False)
        print(f"  v4: {v4r.wall_time:.1f}s  ({v4r.wall_time/total*1000:.1f} ms/bt)")

        print("Running v5 GPU...")
        v5r = v5_gpu.run(returns, minvar_configs, n_bootstrap=n_bootstrap,
                          seed=42, show_progress=False)
        print(f"  v5: {v5r.wall_time:.1f}s  ({v5r.wall_time/total*1000:.1f} ms/bt)")

        print(f"  v5 / v4: {v4r.wall_time/v5r.wall_time:.2f}x")

        results.append((
            n_bootstrap, total,
            v4r.wall_time, v5r.wall_time,
            v4r.wall_time / total * 1000,
            v5r.wall_time / total * 1000,
        ))

    # Summary table
    print(f"\n{'=' * 70}")
    print("SCALING CURVE SUMMARY")
    print('=' * 70)
    print(f"{'B':>4s} {'total_bt':>10s} {'v4 wall':>10s} {'v5 wall':>10s} "
          f"{'v4 ms/bt':>10s} {'v5 ms/bt':>10s} {'v5/v4':>8s}")
    print("-" * 70)
    for B, total, v4t, v5t, v4ms, v5ms in results:
        print(f"{B:>4d} {total:>10d} {v4t:>9.1f}s {v5t:>9.1f}s "
              f"{v4ms:>10.1f} {v5ms:>10.1f} {v4t/v5t:>7.2f}x")

    print(f"\nNote: v5 wall time should be roughly constant across B's "
          f"(GPU saturated).")
    print(f"v4 wall time scales linearly with B.")


if __name__ == "__main__":
    main()