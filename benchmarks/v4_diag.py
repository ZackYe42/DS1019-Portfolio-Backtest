"""Diagnose v4 overhead: measure startup vs work time separately."""
import time
import os
from multiprocessing import get_context


def noop_task(x):
    return x * 2


def noop_init(*args):
    import src.versions.v3_numba  # noqa
    return


def main():
    from src.data_loader import load_returns
    from src.backtest import default_config_grid
    from src.versions import v4_multiproc

    returns = load_returns().to_numpy()
    configs = default_config_grid()[:6]

    print("=" * 60)
    print("Overhead diagnostic")
    print("=" * 60)

    # 1. Just spawn 8 workers and do nothing
    print("\n[Test 1] Spawn 8 workers, no module import")
    ctx = get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(processes=8) as pool:
        pool.map(noop_task, range(8))
    dt_spawn = time.perf_counter() - t0
    print(f"  Time: {dt_spawn:.2f}s  ({dt_spawn/8:.2f}s per worker)")

    # 2. Spawn 8 workers and have each import v3_numba
    print("\n[Test 2] Spawn 8 workers + import v3_numba in each")
    t0 = time.perf_counter()
    with ctx.Pool(processes=8, initializer=noop_init) as pool:
        pool.map(noop_task, range(8))
    dt_with_import = time.perf_counter() - t0
    print(f"  Time: {dt_with_import:.2f}s  ({dt_with_import/8:.2f}s per worker)")
    print(f"  Import overhead per worker: {(dt_with_import - dt_spawn)/8:.2f}s")

    # 3. Full v4 with just 1 worker (isolates worker overhead vs v3 single thread)
    print("\n[Test 3] v4 with 1 worker")
    t0 = time.perf_counter()
    v4_1 = v4_multiproc.run(returns, configs, n_bootstrap=10,
                            seed=42, n_workers=1, show_progress=False)
    dt_v4_1 = time.perf_counter() - t0
    print(f"  Time: {dt_v4_1:.2f}s")

    # 4. Full v4 with 2 workers
    print("\n[Test 4] v4 with 2 workers")
    t0 = time.perf_counter()
    v4_2 = v4_multiproc.run(returns, configs, n_bootstrap=10,
                            seed=42, n_workers=2, show_progress=False)
    dt_v4_2 = time.perf_counter() - t0
    print(f"  Time: {dt_v4_2:.2f}s")

    # 5. Full v4 with 4 workers
    print("\n[Test 5] v4 with 4 workers")
    t0 = time.perf_counter()
    v4_4 = v4_multiproc.run(returns, configs, n_bootstrap=10,
                            seed=42, n_workers=4, show_progress=False)
    dt_v4_4 = time.perf_counter() - t0
    print(f"  Time: {dt_v4_4:.2f}s")

    # 6. Full v4 with 8 workers
    print("\n[Test 6] v4 with 8 workers")
    t0 = time.perf_counter()
    v4_8 = v4_multiproc.run(returns, configs, n_bootstrap=10,
                            seed=42, n_workers=8, show_progress=False)
    dt_v4_8 = time.perf_counter() - t0
    print(f"  Time: {dt_v4_8:.2f}s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pure spawn (no imports):        {dt_spawn:.2f}s")
    print(f"Spawn + v3 import per worker:   {dt_with_import:.2f}s "
          f"(per-worker cost: {(dt_with_import - dt_spawn)/8:.2f}s)")
    print(f"v4 n_workers=1:                 {dt_v4_1:.2f}s")
    print(f"v4 n_workers=2:                 {dt_v4_2:.2f}s")
    print(f"v4 n_workers=4:                 {dt_v4_4:.2f}s")
    print(f"v4 n_workers=8:                 {dt_v4_8:.2f}s")

    # Scaling analysis
    print(f"\nWith 1 worker:  {dt_v4_1:.2f}s  (baseline for v4 overhead)")
    if dt_v4_2 > 0:
        print(f"With 2 workers: {dt_v4_2:.2f}s  speedup={dt_v4_1/dt_v4_2:.2f}x  efficiency={100*(dt_v4_1/dt_v4_2)/2:.0f}%")
    if dt_v4_4 > 0:
        print(f"With 4 workers: {dt_v4_4:.2f}s  speedup={dt_v4_1/dt_v4_4:.2f}x  efficiency={100*(dt_v4_1/dt_v4_4)/4:.0f}%")
    if dt_v4_8 > 0:
        print(f"With 8 workers: {dt_v4_8:.2f}s  speedup={dt_v4_1/dt_v4_8:.2f}x  efficiency={100*(dt_v4_1/dt_v4_8)/8:.0f}%")


if __name__ == "__main__":
    main()