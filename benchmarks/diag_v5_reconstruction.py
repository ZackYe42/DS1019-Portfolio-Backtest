"""
Diagnose v5 wrapper's portfolio-return reconstruction.

Strategy: run ONE backtest through three paths, comparing at each stage:
    1. v3 (reference)
    2. v5 GPU kernel -> v5 wrapper's portfolio reconstruction
    3. v5 GPU kernel -> v3's EXACT reconstruction code

If (1) == (3) but (2) diverges, the bug is in the v5 wrapper's reconstruction.
If (1) diverges from (3), the GPU weights are actually wrong at some rebalance.
"""
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.bootstrap import fixed_block_bootstrap
    from src.versions.v3_numba import (
        _sample_cov_njit, _power_iteration_njit, _min_variance_njit,
    )
    from src.versions.v5_gpu_kernels import run_full_backtest_gpu, N_ASSETS
    from src.metrics import summary as metric_summary

    returns = load_returns().to_numpy()[:, :N_ASSETS]
    T, N = returns.shape
    lookback = 252
    rebalance_every = 21

    # Use the SAME bootstrap seed formula as v5_gpu wrapper, for cfg_idx=0, boot_idx=0
    seed = 42
    cfg_idx = 0
    boot_idx = 0
    sub_seed = seed * 1_000_003 + cfg_idx * 997 + boot_idx
    rng = np.random.default_rng(sub_seed)
    boot = fixed_block_bootstrap(returns, block_size=20, rng=rng)

    # Rebalance indices (both methods should agree)
    rebalance_indices = np.arange(lookback, T, rebalance_every)
    n_rebalance = len(rebalance_indices)
    print(f"T={T}, lookback={lookback}, rebalance_every={rebalance_every}")
    print(f"n_rebalance={n_rebalance}")
    print(f"First rebal at t={rebalance_indices[0]}, last at t={rebalance_indices[-1]}")

    # ====================================================================
    # PATH 1: Pure v3-style reference
    # ====================================================================
    print("\n[Path 1] Pure v3-style reference")
    arith_returns = np.exp(boot) - 1.0
    weights_v3 = np.zeros((n_rebalance, N))
    port_v3 = np.zeros(T - lookback)
    weights_current = np.full(N, 1.0 / N)
    rb_ptr = 0
    for t in range(lookback, T):
        if rb_ptr < n_rebalance and t == rebalance_indices[rb_ptr]:
            window = boot[t - lookback:t]
            cov = _sample_cov_njit(window) + 1e-10 * np.eye(N)
            lam = _power_iteration_njit(cov, 50)
            weights_current = _min_variance_njit(cov, 1.0 / (2.0 * lam), 500)
            weights_v3[rb_ptr] = weights_current
            rb_ptr += 1
        port_v3[t - lookback] = weights_current @ arith_returns[t]

    m_v3 = metric_summary(port_v3, rf_annual=0.0, log_returns=False)
    print(f"  Sharpe: {m_v3['sharpe']:.6f}")

    # ====================================================================
    # PATH 2: GPU kernel weights -> v5_gpu's wrapper reconstruction
    # ====================================================================
    print("\n[Path 2] GPU kernel -> v5 wrapper reconstruction")
    gpu_weights_raw = run_full_backtest_gpu(
        boot[None], lookback, rebalance_every, estimator="sample"
    )[0]  # (n_rebalance, N), float32

    # This is v5_gpu's exact reconstruction code:
    weights_history_v5 = gpu_weights_raw.astype(np.float64)
    boot_arith = np.exp(boot) - 1.0
    port_v5wrap = np.zeros(T - lookback)
    current_w = np.full(N, 1.0 / N)
    rb_ptr = 0
    for t_ in range(lookback, T):
        if rb_ptr < n_rebalance and t_ == rebalance_indices[rb_ptr]:
            current_w = weights_history_v5[rb_ptr]
            rb_ptr += 1
        port_v5wrap[t_ - lookback] = current_w @ boot_arith[t_]

    m_v5wrap = metric_summary(port_v5wrap, rf_annual=0.0, log_returns=False)
    print(f"  Sharpe: {m_v5wrap['sharpe']:.6f}")

    # ====================================================================
    # PATH 3: GPU kernel weights -> v3's EXACT reconstruction code
    # ====================================================================
    print("\n[Path 3] GPU kernel weights -> v3's exact reconstruction")
    # Feed GPU weights into v3's loop structure, NOT v3's solver
    port_v5inv3 = np.zeros(T - lookback)
    weights_current = np.full(N, 1.0 / N)
    rb_ptr = 0
    for t in range(lookback, T):
        if rb_ptr < n_rebalance and t == rebalance_indices[rb_ptr]:
            weights_current = weights_history_v5[rb_ptr]  # use GPU weights
            rb_ptr += 1
        port_v5inv3[t - lookback] = weights_current @ arith_returns[t]

    m_v5inv3 = metric_summary(port_v5inv3, rf_annual=0.0, log_returns=False)
    print(f"  Sharpe: {m_v5inv3['sharpe']:.6f}")

    # ====================================================================
    # Analysis
    # ====================================================================
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    # Compare weight histories per rebalance
    weight_diffs = np.abs(weights_v3 - weights_history_v5).max(axis=1)
    print(f"\nWeight diffs per rebalance (max |v3 - v5|_inf):")
    worst_rebs = np.argsort(-weight_diffs)[:5]
    for r in worst_rebs:
        print(f"  rebal {r}: {weight_diffs[r]:.2e}")
    print(f"  overall max: {weight_diffs.max():.2e}")

    # Compare port_returns
    port_diff_2_1 = np.abs(port_v5wrap - port_v3).max()
    port_diff_3_1 = np.abs(port_v5inv3 - port_v3).max()
    port_diff_2_3 = np.abs(port_v5wrap - port_v5inv3).max()

    print(f"\nPortfolio-return diffs:")
    print(f"  Path 2 (wrap)   vs Path 1 (v3): {port_diff_2_1:.2e}")
    print(f"  Path 3 (v3loop) vs Path 1 (v3): {port_diff_3_1:.2e}")
    print(f"  Path 2 (wrap)   vs Path 3 (v3loop): {port_diff_2_3:.2e}")

    print(f"\nSharpe values:")
    print(f"  Path 1 (pure v3):          {m_v3['sharpe']:.6f}")
    print(f"  Path 2 (v5 wrapper):       {m_v5wrap['sharpe']:.6f}")
    print(f"  Path 3 (GPU+v3 reconstr):  {m_v5inv3['sharpe']:.6f}")

    print()
    if abs(m_v5inv3['sharpe'] - m_v3['sharpe']) < 1e-4:
        print("=> GPU weights feed into v3 reconstruction cleanly.")
        if abs(m_v5wrap['sharpe'] - m_v3['sharpe']) > 1e-4:
            print("=> BUG IS IN THE V5 WRAPPER's reconstruction loop.")
        else:
            print("=> No bug detected on this single backtest.  Issue may be "
                  "config-specific or cross-config (cfg_idx seeding).")
    else:
        print("=> GPU weights are incorrect at some rebalance.  "
              "Bug is in the full-backtest kernel.")


if __name__ == "__main__":
    main()