"""
Correctness tests for the bootstrap-backtest pipeline.

Every version in src/versions/ must pass these tests.  The tests run
v1_baseline as the ground truth, then (optionally) compare other
versions' outputs against it.

Invariants we check:
    1. Portfolio returns are finite and in a plausible range
    2. Weights sum to 1 at every rebalance date
    3. Weights are non-negative under long_only=True
    4. Metrics are reproducible with a fixed seed
    5. Two consecutive runs with the same seed produce identical results
    6. Cross-version agreement (when multiple versions are available)

Run with:  pytest tests/test_correctness.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_backtest, default_config_grid
from src.data_loader import load_returns
from src.versions import v1_baseline


# Tolerance for cross-version numerical agreement.
# v1 uses float64 throughout.  v3 Numba will also use float64 (can match
# to 1e-10).  v5 GPU uses float32 on RTX 4070 (matches to ~1e-4 at best).
TOL_FLOAT64 = 1e-10
TOL_FLOAT32 = 1e-4


# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def returns():
    """Load real S&P 100 returns once per test module."""
    return load_returns().to_numpy()


@pytest.fixture(scope="module")
def small_configs():
    """A small set of configs for quick testing."""
    return [
        BacktestConfig(estimator="sample", objective="min_variance",
                       lookback=252, rebalance_every=21),
        BacktestConfig(estimator="ledoit_wolf", objective="min_variance",
                       lookback=252, rebalance_every=21),
        BacktestConfig(estimator="ledoit_wolf", objective="max_sharpe",
                       lookback=252, rebalance_every=21),
    ]


# -----------------------------------------------------------------------------
# Single-backtest invariants
# -----------------------------------------------------------------------------

def test_weights_sum_to_one(returns, small_configs):
    """At every rebalance date, weights must sum to 1."""
    for cfg in small_configs:
        result = run_backtest(returns, cfg)
        sums = result.weights_history.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6), \
            f"{cfg.label()}: weight sums = {sums}"


def test_weights_nonneg_long_only(returns, small_configs):
    """Under long_only=True, no weight may be negative (beyond tolerance)."""
    for cfg in small_configs:
        result = run_backtest(returns, cfg)
        min_w = result.weights_history.min()
        assert min_w >= -1e-8, f"{cfg.label()}: min weight = {min_w}"


def test_portfolio_returns_finite(returns, small_configs):
    """Portfolio returns must be finite and in a plausible daily range."""
    for cfg in small_configs:
        result = run_backtest(returns, cfg)
        r = result.portfolio_returns
        assert np.isfinite(r).all(), f"{cfg.label()}: non-finite return"
        # Largest one-day equity-portfolio return in our data period is
        # +/-15%; any value beyond +/-50% is almost certainly a bug.
        assert np.abs(r).max() < 0.5, \
            f"{cfg.label()}: extreme return {np.abs(r).max()}"


def test_metrics_plausible(returns, small_configs):
    """Sharpe, vol, and drawdown should fall in financially plausible ranges."""
    for cfg in small_configs:
        result = run_backtest(returns, cfg)
        m = result.metrics
        assert -3 < m["sharpe"] < 5, f"{cfg.label()}: sharpe = {m['sharpe']}"
        assert 0 < m["annual_vol"] < 1.0, \
            f"{cfg.label()}: annual_vol = {m['annual_vol']}"
        assert -1.0 < m["max_drawdown"] <= 0, \
            f"{cfg.label()}: max_drawdown = {m['max_drawdown']}"


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def test_single_backtest_deterministic(returns, small_configs):
    """Same returns + same config = identical output."""
    for cfg in small_configs:
        r1 = run_backtest(returns, cfg)
        r2 = run_backtest(returns, cfg)
        np.testing.assert_allclose(r1.portfolio_returns, r2.portfolio_returns,
                                   atol=TOL_FLOAT64)
        np.testing.assert_allclose(r1.weights_history, r2.weights_history,
                                   atol=TOL_FLOAT64)


def test_v1_baseline_deterministic(returns, small_configs):
    """Same seed = identical bootstrap output."""
    result_a = v1_baseline.run(returns, small_configs,
                               n_bootstrap=3, seed=42, show_progress=False)
    result_b = v1_baseline.run(returns, small_configs,
                               n_bootstrap=3, seed=42, show_progress=False)

    # Sort by (config_idx, bootstrap_idx) to ensure row alignment
    df_a = result_a.raw_metrics.sort_values(["config_idx", "bootstrap_idx"])
    df_b = result_b.raw_metrics.sort_values(["config_idx", "bootstrap_idx"])

    for col in ["sharpe", "annual_vol", "max_drawdown", "cagr"]:
        np.testing.assert_allclose(
            df_a[col].to_numpy(), df_b[col].to_numpy(),
            atol=TOL_FLOAT64,
            err_msg=f"mismatch on column '{col}' between two seeded runs",
        )


def test_different_seeds_produce_different_results(returns, small_configs):
    """Different seeds must produce non-identical bootstrap metrics."""
    result_a = v1_baseline.run(returns, small_configs,
                               n_bootstrap=3, seed=1, show_progress=False)
    result_b = v1_baseline.run(returns, small_configs,
                               n_bootstrap=3, seed=2, show_progress=False)

    df_a = result_a.raw_metrics.sort_values(["config_idx", "bootstrap_idx"])
    df_b = result_b.raw_metrics.sort_values(["config_idx", "bootstrap_idx"])

    # Sharpes should differ across seeds (not bit-identical)
    assert not np.allclose(df_a["sharpe"].to_numpy(),
                           df_b["sharpe"].to_numpy(),
                           atol=1e-6)


# -----------------------------------------------------------------------------
# Cross-version agreement (skipped until v2+ exists)
# -----------------------------------------------------------------------------

def _available_versions():
    """Return a list of (name, module) pairs for every version module present."""
    from src.versions import v1_baseline
    versions = [("v1_baseline", v1_baseline)]

    try:
        from src.versions import v2_numpy
        versions.append(("v2_numpy", v2_numpy))
    except ImportError:
        pass
    try:
        from src.versions import v3_numba
        versions.append(("v3_numba", v3_numba))
    except ImportError:
        pass
    try:
        from src.versions import v4_multiproc
        versions.append(("v4_multiproc", v4_multiproc))
    except ImportError:
        pass
    try:
        from src.versions import v5_gpu
        versions.append(("v5_gpu", v5_gpu))
    except ImportError:
        pass

    return versions


@pytest.mark.parametrize("target_name,target", _available_versions()[1:])
def test_version_matches_baseline(target_name, target, returns, small_configs):
    """
    Every version must match v1_baseline within tolerance.

    v2 NumPy and v3 Numba must match to float64 precision.
    v5 GPU (float32) only matches to float32 precision.
    """
    baseline_result = v1_baseline.run(returns, small_configs,
                                      n_bootstrap=3, seed=42,
                                      show_progress=False)
    target_result = target.run(returns, small_configs,
                               n_bootstrap=3, seed=42,
                               show_progress=False)

    df_base = baseline_result.raw_metrics.sort_values(
        ["config_idx", "bootstrap_idx"]).reset_index(drop=True)
    df_targ = target_result.raw_metrics.sort_values(
        ["config_idx", "bootstrap_idx"]).reset_index(drop=True)

    # GPU version uses float32
    tol = TOL_FLOAT32 if "gpu" in target_name else 1e-6

    for col in ["sharpe", "annual_vol", "max_drawdown", "cagr"]:
        np.testing.assert_allclose(
            df_base[col].to_numpy(),
            df_targ[col].to_numpy(),
            atol=tol,
            err_msg=f"{target_name}: mismatch on '{col}' vs v1_baseline",
        )


# -----------------------------------------------------------------------------
# Allow running as a script (without pytest)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], check=False)