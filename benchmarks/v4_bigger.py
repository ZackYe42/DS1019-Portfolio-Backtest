"""Test v4 on a realistically-sized workload to see actual scaling."""
import time


def main():
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v3_numba, v4_multiproc

    returns = load_returns().to_numpy()
    configs = default_config_grid()  # all 54 configs
    n_bootstrap = 50

    total = len(configs) * n_bootstrap
    print("=" * 60)
    print(f"Larger workload: {len(configs)} configs x {n_bootstrap} bootstrap = {total} backtests")
    print("=" * 60)
    print("(Each backtest ~0.25s, so total serial work ~= "
          f"{total * 0.25 / 60:.1f} minutes)")

    # v3 single-thread
    print("\nRunning v3 numba (single-thread)...")
    t0 = time.perf_counter()
    v3 = v3_numba.run(returns, configs, n_bootstrap=n_bootstrap,
                      seed=42, show_progress=True)
    dt_v3 = time.perf_counter() - t0
    print(f"  v3 wall time: {dt_v3:.1f}s  ({dt_v3/total*1000:.0f}ms per backtest)")

    # v4 at different worker counts
    print("\nRunning v4 at 2, 4, 8 workers...")
    times = {}
    for nw in [2, 4, 8]:
        t0 = time.perf_counter()
        v4 = v4_multiproc.run(returns, configs, n_bootstrap=n_bootstrap,
                              seed=42, n_workers=nw, show_progress=True)
        times[nw] = time.perf_counter() - t0
        print(f"  {nw} workers: {times[nw]:.1f}s  "
              f"(v3 speedup: {dt_v3/times[nw]:.2f}x, efficiency: {100*dt_v3/times[nw]/nw:.0f}%)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"v3 baseline:       {dt_v3:.1f}s")
    for nw in [2, 4, 8]:
        print(f"v4, {nw} workers:     {times[nw]:.1f}s  ({dt_v3/times[nw]:.2f}x vs v3)")


if __name__ == "__main__":
    main()