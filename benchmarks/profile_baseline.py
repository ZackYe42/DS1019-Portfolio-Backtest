"""
Profile v1_baseline to identify bottlenecks.

Runs two profilers:
    1. cProfile    - function-level: which functions take the most time
    2. line_profiler - line-level: for each hot function, which specific
                     lines are slow

Output files (in results/):
    profile_cprofile.txt      - cProfile report, top 30 functions by cumulative time
    profile_lineprofile.txt   - line_profiler annotations on the 3 hottest functions

This report is Figure 1 / Table 1 of the final report: it justifies
which parts of the pipeline we chose to optimize in v2-v5.
"""

from __future__ import annotations

import cProfile
import io
import pstats
from pathlib import Path

import numpy as np

from src.backtest import BacktestConfig
from src.data_loader import load_returns
from src.versions import v1_baseline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def small_workload(returns: np.ndarray):
    """
    A small, representative workload for profiling.

    4 configs x 5 bootstrap samples = 20 backtests.
    Covers both objectives (min_variance and max_sharpe) so the profile
    reflects both code paths.  Runs in ~60-90 seconds.
    """
    configs = [
        BacktestConfig(estimator="sample",       objective="min_variance",
                       lookback=252, rebalance_every=21),
        BacktestConfig(estimator="ledoit_wolf",  objective="min_variance",
                       lookback=252, rebalance_every=21),
        BacktestConfig(estimator="ewma",         objective="min_variance",
                       lookback=252, rebalance_every=21),
        BacktestConfig(estimator="ledoit_wolf",  objective="max_sharpe",
                       lookback=252, rebalance_every=21),
    ]
    return configs, 5  # n_bootstrap=5


# -----------------------------------------------------------------------------
# cProfile: function-level profiling
# -----------------------------------------------------------------------------

def run_cprofile(returns, configs, n_bootstrap):
    print("Running cProfile...")
    profiler = cProfile.Profile()
    profiler.enable()
    v1_baseline.run(returns, configs, n_bootstrap=n_bootstrap, seed=42,
                    show_progress=False)
    profiler.disable()

    # Generate report sorted by cumulative time
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("cumulative")

    # Header
    stream.write("=" * 80 + "\n")
    stream.write(f"cProfile report: {len(configs)} configs x "
                 f"{n_bootstrap} bootstrap samples = "
                 f"{len(configs) * n_bootstrap} backtests\n")
    stream.write("=" * 80 + "\n\n")

    # Top 30 by cumulative time
    stream.write(">>> TOP 30 FUNCTIONS BY CUMULATIVE TIME <<<\n\n")
    stats.print_stats(30)

    # Also top 30 by internal (tottime) - excludes sub-function time
    stream.write("\n\n>>> TOP 30 FUNCTIONS BY INTERNAL TIME (excl. callees) <<<\n\n")
    stats.sort_stats("tottime")
    stats.print_stats(30)

    report = stream.getvalue()
    out_path = RESULTS_DIR / "profile_cprofile.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"  saved {out_path}")

    # Also print the top-10 to stdout so the user sees it immediately
    print("\n" + "-" * 80)
    print("TOP 10 BY CUMULATIVE TIME (functions that dominate wall time):")
    print("-" * 80)
    stats.sort_stats("cumulative")
    stats.print_stats(10)


# -----------------------------------------------------------------------------
# line_profiler: line-level profiling of hot functions
# -----------------------------------------------------------------------------

def run_line_profiler(returns, configs, n_bootstrap):
    print("\nRunning line_profiler on hot functions...")
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("  line_profiler not installed; skipping.  "
              "Install with: pip install line_profiler")
        return

    # Import the functions we want to profile at the line level.
    # These are our top candidates based on the pipeline structure.
    from src import estimators, optimizer, backtest

    lp = LineProfiler()
    # Add the five functions most likely to dominate
    lp.add_function(backtest.run_backtest)
    lp.add_function(estimators.ledoit_wolf)
    lp.add_function(estimators.sample_cov)
    lp.add_function(estimators.ewma_cov)
    lp.add_function(optimizer.min_variance_numpy)
    lp.add_function(optimizer.max_sharpe_numpy)

    wrapped = lp(v1_baseline.run)
    wrapped(returns, configs, n_bootstrap=n_bootstrap, seed=42,
            show_progress=False)

    # Capture the report
    stream = io.StringIO()
    lp.print_stats(stream=stream, output_unit=1e-3)  # report in ms

    report = stream.getvalue()
    out_path = RESULTS_DIR / "profile_lineprofile.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"  saved {out_path}")

    # Show a short summary to stdout
    print("\n" + "-" * 80)
    print("LINE PROFILER SUMMARY (see results/profile_lineprofile.txt for full)")
    print("-" * 80)
    # Print just the first ~40 lines of the report so stdout isn't overwhelming
    lines = report.split("\n")
    for line in lines[:40]:
        print(line)
    if len(lines) > 40:
        print(f"... [{len(lines) - 40} more lines in the file]")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("PROFILING v1_baseline")
    print("=" * 80)

    returns = load_returns().to_numpy()
    configs, n_bootstrap = small_workload(returns)

    print(f"\nWorkload: {len(configs)} configs x {n_bootstrap} bootstrap "
          f"= {len(configs) * n_bootstrap} backtests")
    print("Configs:")
    for i, c in enumerate(configs):
        print(f"  [{i}] {c.label()}")
    print()

    run_cprofile(returns, configs, n_bootstrap)
    run_line_profiler(returns, configs, n_bootstrap)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print(f"See:  {RESULTS_DIR / 'profile_cprofile.txt'}")
    print(f"And:  {RESULTS_DIR / 'profile_lineprofile.txt'}")
    print("=" * 80)