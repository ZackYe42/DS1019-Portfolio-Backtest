"""
benchmarks/run_all.py

Single-source-of-truth benchmark sweep for the final report.

Generates:
    benchmarks/results/headline.csv          v1-v5 on canonical workload
    benchmarks/results/v4_worker_scaling.csv v4 at 1,2,4,8,16 workers
    benchmarks/results/v5_batch_scaling.csv  v5 at B=20,50,100,200
    benchmarks/results/correctness.csv       max Sharpe diff vs v1
    benchmarks/results/per_config.csv        mean metrics per strategy

Run:
    python -m benchmarks.run_all              # default: 54 cfg x 100 bt
    python -m benchmarks.run_all --small      # 27 min_var cfg x 20 bt (fast)
    python -m benchmarks.run_all --full       # 54 cfg x 500 bt (production)

All results are written as CSV so the report can pull numbers directly.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def hardware_info() -> dict:
    """Record hardware so readers know what numbers apply to."""
    import os
    info = {
        "cpu": platform.processor() or "unknown",
        "n_logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        info["n_physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    try:
        from numba import cuda
        devs = cuda.list_devices()
        if devs:
            d = devs[0]
            info["gpu"] = d.name.decode() if isinstance(d.name, bytes) else str(d.name)
    except Exception:
        info["gpu"] = "none"
    return info


def estimate_v1_time(returns, configs, n_bootstrap, seed=42):
    """Run a small calibration to estimate v1 per-backtest time, then extrapolate."""
    from src.versions import v1_baseline
    cal_configs = configs[:3]
    cal_n_bs = 3
    t0 = time.perf_counter()
    v1_baseline.run(returns, cal_configs, n_bootstrap=cal_n_bs,
                     seed=seed, show_progress=False)
    cal_time = time.perf_counter() - t0
    per_bt = cal_time / (len(cal_configs) * cal_n_bs)
    total = per_bt * len(configs) * n_bootstrap
    return total, per_bt


def sweep_headline(returns, configs, n_bootstrap, seed=42):
    """Run v1 (calibrated) + v2, v3, v4, v5 on the canonical workload.

    Returns a DataFrame with per-version timing and a dict of raw metrics
    (for correctness comparison).
    """
    from src.versions import v1_baseline, v2_numpy, v3_numba, v4_multiproc, v5_gpu

    n_cfg = len(configs)
    n_total = n_cfg * n_bootstrap

    rows = []
    raws = {}  # version -> raw_metrics DataFrame

    # v1: calibrate only (running full v1 at 5400 bt would take hours)
    print("\n[v1] Calibrating...")
    v1_total, v1_per_bt = estimate_v1_time(returns, configs, n_bootstrap, seed=seed)
    print(f"  v1 projected: {v1_total:.0f}s ({v1_per_bt*1000:.0f} ms/bt)")
    rows.append({
        "version": "v1_baseline",
        "description": "Pure Python, projected from 9-backtest calibration",
        "wall_time_s": v1_total,
        "per_backtest_ms": v1_per_bt * 1000,
        "speedup_vs_v1": 1.0,
        "projected": True,
    })

    # v2
    print("\n[v2] NumPy vectorized...")
    t0 = time.perf_counter()
    v2r = v2_numpy.run(returns, configs, n_bootstrap=n_bootstrap,
                        seed=seed, show_progress=True)
    print(f"  v2: {v2r.wall_time:.1f}s")
    raws["v2"] = v2r.raw_metrics
    rows.append({
        "version": "v2_numpy",
        "description": "NumPy-vectorized",
        "wall_time_s": v2r.wall_time,
        "per_backtest_ms": v2r.wall_time / n_total * 1000,
        "speedup_vs_v1": v1_total / v2r.wall_time,
        "projected": False,
    })

    # v3
    print("\n[v3] Numba JIT...")
    t0 = time.perf_counter()
    v3r = v3_numba.run(returns, configs, n_bootstrap=n_bootstrap,
                        seed=seed, show_progress=True)
    print(f"  v3: {v3r.wall_time:.1f}s")
    raws["v3"] = v3r.raw_metrics
    rows.append({
        "version": "v3_numba",
        "description": "Numba JIT + power iteration",
        "wall_time_s": v3r.wall_time,
        "per_backtest_ms": v3r.wall_time / n_total * 1000,
        "speedup_vs_v1": v1_total / v3r.wall_time,
        "projected": False,
    })

    # v4
    print("\n[v4] Multiprocessing (8 workers)...")
    v4r = v4_multiproc.run(returns, configs, n_bootstrap=n_bootstrap,
                             n_workers=8, seed=seed, show_progress=True)
    print(f"  v4: {v4r.wall_time:.1f}s")
    raws["v4"] = v4r.raw_metrics
    rows.append({
        "version": "v4_multiproc_8w",
        "description": "v3 under 8-worker multiprocessing pool",
        "wall_time_s": v4r.wall_time,
        "per_backtest_ms": v4r.wall_time / n_total * 1000,
        "speedup_vs_v1": v1_total / v4r.wall_time,
        "projected": False,
    })

    # v5
    print("\n[v5] GPU...")
    v5r = v5_gpu.run(returns, configs, n_bootstrap=n_bootstrap,
                      seed=seed, show_progress=True)
    print(f"  v5: {v5r.wall_time:.1f}s (GPU fraction {v5r.gpu_fraction:.0%})")
    raws["v5"] = v5r.raw_metrics
    rows.append({
        "version": "v5_gpu",
        "description": f"GPU min_var + CPU max_sharpe (gpu_fraction={v5r.gpu_fraction:.0%})",
        "wall_time_s": v5r.wall_time,
        "per_backtest_ms": v5r.wall_time / n_total * 1000,
        "speedup_vs_v1": v1_total / v5r.wall_time,
        "projected": False,
    })

    return pd.DataFrame(rows), raws


def sweep_v4_workers(returns, configs, n_bootstrap, seed=42):
    """Scaling study: v4 at 1, 2, 4, 8 workers.

    We stop at 8 because:
      - Past 8, the Amdahl speedup curve flattens (we're memory-bandwidth bound).
      - 16+ workers on a high-core-count desktop can trigger thermal/power
        limits that crash the machine under sustained load.  8 workers is
        our production recommendation anyway.
    """
    from src.versions import v4_multiproc
    rows = []
    worker_counts = [1, 2, 4, 8]
    for nw in worker_counts:
        print(f"\n[v4 scaling] n_workers={nw}")
        v4r = v4_multiproc.run(returns, configs, n_bootstrap=n_bootstrap,
                                 n_workers=nw, seed=seed, show_progress=False)
        n_total = len(configs) * n_bootstrap
        rows.append({
            "n_workers": nw,
            "wall_time_s": v4r.wall_time,
            "per_backtest_ms": v4r.wall_time / n_total * 1000,
        })
        print(f"  wall: {v4r.wall_time:.1f}s  ({v4r.wall_time/n_total*1000:.0f} ms/bt)")

        # Cooldown pause between runs to let CPU temps settle.
        # Without this, back-to-back sustained runs can heat-soak the VRM.
        if nw != worker_counts[-1]:
            print(f"  (cooldown 10s...)")
            time.sleep(10)

    df = pd.DataFrame(rows)
    df["speedup_vs_1w"] = df.loc[0, "wall_time_s"] / df["wall_time_s"]
    df["efficiency"] = df["speedup_vs_1w"] / df["n_workers"]
    return df


def sweep_v5_batches(returns, configs, seed=42):
    """Scaling study: v5 at B=20, 50, 100, 200 (min_variance only)."""
    from src.versions import v5_gpu
    rows = []
    minvar = [c for c in configs if c.objective == "min_variance"]
    batch_sizes = [20, 50, 100, 200]
    for B in batch_sizes:
        print(f"\n[v5 scaling] B={B}")
        v5r = v5_gpu.run(returns, minvar, n_bootstrap=B,
                          seed=seed, show_progress=False)
        n_total = len(minvar) * B
        rows.append({
            "n_bootstrap": B,
            "n_total_backtests": n_total,
            "wall_time_s": v5r.wall_time,
            "per_backtest_ms": v5r.wall_time / n_total * 1000,
        })
        print(f"  wall: {v5r.wall_time:.1f}s  ({v5r.wall_time/n_total*1000:.1f} ms/bt)")
    return pd.DataFrame(rows)


def compute_correctness(raws: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Max absolute diff in each metric between v2-v5 and v1 (reference).

    We use v3 as the reference since it matches v1 bit-for-bit and is
    much faster to rerun at full workload.
    """
    # Use v3 as canonical reference (we validated v3 == v1 in tests)
    if "v3" not in raws:
        return pd.DataFrame()
    ref = raws["v3"].sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)

    rows = []
    for ver in ["v2", "v3", "v4", "v5"]:
        if ver not in raws:
            continue
        df = raws[ver].sort_values(["config_idx", "bootstrap_idx"]).reset_index(drop=True)
        row = {"version": ver}
        for col in ["sharpe", "annual_vol", "max_drawdown", "cagr"]:
            diff = np.abs(ref[col].to_numpy() - df[col].to_numpy()).max()
            row[f"max_{col}_diff"] = diff
        rows.append(row)
    return pd.DataFrame(rows)


def per_config_aggregates(raws: dict) -> pd.DataFrame:
    """Per-strategy mean metrics, from the v5 run (same as other versions bit-for-bit)."""
    if "v5" not in raws:
        return pd.DataFrame()
    df = raws["v5"]
    agg = df.groupby(["label", "estimator", "objective",
                       "lookback", "rebalance_every"]).agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_p05=("sharpe", lambda x: np.quantile(x, 0.05)),
        sharpe_p95=("sharpe", lambda x: np.quantile(x, 0.95)),
        annual_vol_mean=("annual_vol", "mean"),
        max_dd_mean=("max_drawdown", "mean"),
        cagr_mean=("cagr", "mean"),
    ).reset_index()
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip-v4-scaling", action="store_true")
    parser.add_argument("--skip-v5-scaling", action="store_true")
    parser.add_argument("--force", action="store_true",
                         help="Re-run stages even if CSVs exist")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.data_loader import load_returns
    from src.backtest import default_config_grid

    if args.small:
        configs = [c for c in default_config_grid() if c.objective == "min_variance"]
        n_bootstrap = 20
        mode = "small"
    elif args.full:
        configs = default_config_grid()
        n_bootstrap = 500
        mode = "full"
    else:
        configs = default_config_grid()
        n_bootstrap = 100
        mode = "default"

    returns = load_returns().to_numpy()
    print(f"\n{'='*70}\nbenchmarks/run_all  mode={mode}\n{'='*70}")
    print(f"Workload: {len(configs)} configs x {n_bootstrap} bootstrap "
          f"= {len(configs)*n_bootstrap} backtests")

    hw = hardware_info()
    print(f"\nHardware:")
    for k, v in hw.items():
        print(f"  {k}: {v}")
    (RESULTS_DIR / "hardware.json").write_text(json.dumps(hw, indent=2))

    t_start = time.perf_counter()

    # 1. Headline
    headline_path = RESULTS_DIR / "headline.csv"
    raws_path = RESULTS_DIR / "raw_metrics.pkl"
    if headline_path.exists() and raws_path.exists() and not args.force:
        print(f"\n[SKIP] {headline_path.name} already exists (use --force to rerun)")
        headline_df = pd.read_csv(headline_path)
        import pickle
        with open(raws_path, "rb") as f:
            raws = pickle.load(f)
    else:
        print(f"\n{'='*70}\n1. HEADLINE: v1-v5 on canonical workload\n{'='*70}")
        headline_df, raws = sweep_headline(returns, configs, n_bootstrap, seed=args.seed)
        headline_df.to_csv(headline_path, index=False)
        import pickle
        with open(raws_path, "wb") as f:
            pickle.dump(raws, f)
        print("\n", headline_df.to_string(index=False))

    # 2. v4 worker scaling
    if not args.skip_v4_scaling:
        out_path = RESULTS_DIR / "v4_worker_scaling.csv"
        if out_path.exists() and not args.force:
            print(f"\n[SKIP] {out_path.name} already exists")
        else:
            print(f"\n{'='*70}\n2. v4 WORKER SCALING (1,2,4,8 workers)\n{'='*70}")
            scale_configs = configs[:18] if len(configs) > 18 else configs
            scale_bs = min(n_bootstrap, 50)
            print(f"Using reduced workload: {len(scale_configs)} configs x {scale_bs} bt")
            v4_df = sweep_v4_workers(returns, scale_configs, scale_bs, seed=args.seed)
            v4_df.to_csv(out_path, index=False)
            print("\n", v4_df.to_string(index=False))

    # 3. v5 batch scaling
    if not args.skip_v5_scaling:
        out_path = RESULTS_DIR / "v5_batch_scaling.csv"
        if out_path.exists() and not args.force:
            print(f"\n[SKIP] {out_path.name} already exists")
        else:
            print(f"\n{'='*70}\n3. v5 BATCH SCALING (B=20,50,100,200)\n{'='*70}")
            v5_df = sweep_v5_batches(returns, configs, seed=args.seed)
            v5_df.to_csv(out_path, index=False)
            print("\n", v5_df.to_string(index=False))

    # 4. Correctness
    corr_path = RESULTS_DIR / "correctness.csv"
    if corr_path.exists() and not args.force:
        print(f"\n[SKIP] {corr_path.name} already exists")
    else:
        print(f"\n{'='*70}\n4. CORRECTNESS (vs v3 reference)\n{'='*70}")
        corr_df = compute_correctness(raws)
        corr_df.to_csv(corr_path, index=False)
        print("\n", corr_df.to_string(index=False))

    # 5. Per-config aggregates
    perc_path = RESULTS_DIR / "per_config.csv"
    if perc_path.exists() and not args.force:
        print(f"\n[SKIP] {perc_path.name} already exists")
    else:
        print(f"\n{'='*70}\n5. PER-CONFIG METRICS\n{'='*70}")
        per_cfg_df = per_config_aggregates(raws)
        per_cfg_df.to_csv(perc_path, index=False)
        print(f"  Written {len(per_cfg_df)} strategies to per_config.csv")

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*70}")
    print(f"Total sweep time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Results in: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()