"""
Block bootstrap for time-series resampling.

Given a (T, N) matrix of returns, generate bootstrap resamples that
preserve local time-series structure (autocorrelation, volatility
clustering) by drawing contiguous blocks rather than individual
observations.

Two variants are provided:
    - fixed_block_bootstrap:     blocks of a fixed size k (Kuensch 1989)
    - stationary_bootstrap:      blocks of random geometric length
                                 (Politis & Romano 1994)

The stationary variant is preferred in practice because the resampled
series is itself stationary under mild conditions; the fixed-block
variant is simpler and faster.  Our benchmarks use the fixed variant
(it's the natural choice for vectorization and GPU parallelism).

Each call produces a new (T, N) array.  To produce B bootstrap samples,
call the function B times with different seeds, or use the `batch_*`
variants which return a (B, T, N) array in one call.
"""

from __future__ import annotations

import numpy as np


def fixed_block_bootstrap(
    returns: np.ndarray,
    block_size: int = 20,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate one block-bootstrap resample of a (T, N) returns matrix.

    Draws ceil(T / block_size) blocks of contiguous rows uniformly at
    random (with replacement), concatenates them, and truncates to T.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Source returns.
    block_size : int
        Length of each contiguous block in days.  Typical values are 5
        (one trading week) or 20 (one trading month).
    n_samples : int or None
        Length of the output series.  Defaults to T (same length as input).
    rng : np.random.Generator or None
        Random number generator.  If None, a fresh one is created.

    Returns
    -------
    np.ndarray, shape (n_samples, N)
        The bootstrapped returns.
    """
    T, N = returns.shape
    if n_samples is None:
        n_samples = T
    if block_size < 1 or block_size > T:
        raise ValueError(f"block_size must be in [1, T], got {block_size} for T={T}")
    if rng is None:
        rng = np.random.default_rng()

    # Number of blocks we need to produce (overshoot, then truncate)
    n_blocks = (n_samples + block_size - 1) // block_size

    # Valid starting indices: a block starting at s covers rows [s, s+block_size).
    # We require s + block_size <= T, so s in [0, T - block_size].
    max_start = T - block_size
    starts = rng.integers(low=0, high=max_start + 1, size=n_blocks)

    # Build output using fancy indexing: for each start s, we want rows
    # s, s+1, ..., s+block_size-1.  Vectorize this with broadcasting.
    offsets = np.arange(block_size)                    # shape (block_size,)
    idx = starts[:, None] + offsets[None, :]            # shape (n_blocks, block_size)
    idx = idx.reshape(-1)                              # shape (n_blocks * block_size,)

    sampled = returns[idx]                             # shape (n_blocks * block_size, N)
    return sampled[:n_samples]


def batch_fixed_block_bootstrap(
    returns: np.ndarray,
    n_bootstrap: int,
    block_size: int = 20,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate B block-bootstrap resamples at once.

    Returns a (B, T, N) array.  This is the format the GPU version will
    consume: all B resamples are generated on the host, then the whole
    batch is transferred to the device once.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    n_bootstrap : int
        Number of bootstrap replicates to generate.
    block_size : int
    n_samples : int or None
        Length of each resample.  Defaults to T.
    seed : int or None
        Seed for reproducibility.  If None, uses OS randomness.

    Returns
    -------
    np.ndarray, shape (n_bootstrap, n_samples, N)
    """
    T, N = returns.shape
    if n_samples is None:
        n_samples = T

    rng = np.random.default_rng(seed)
    n_blocks = (n_samples + block_size - 1) // block_size
    max_start = T - block_size

    # Draw all start indices at once: shape (B, n_blocks)
    all_starts = rng.integers(low=0, high=max_start + 1, size=(n_bootstrap, n_blocks))

    # Build all index matrices at once via broadcasting
    offsets = np.arange(block_size)                            # (block_size,)
    # all_idx shape: (B, n_blocks, block_size) -> (B, n_blocks * block_size)
    all_idx = all_starts[:, :, None] + offsets[None, None, :]
    all_idx = all_idx.reshape(n_bootstrap, -1)[:, :n_samples]  # (B, n_samples)

    # Advanced indexing: returns[all_idx] yields shape (B, n_samples, N)
    return returns[all_idx]


def stationary_bootstrap(
    returns: np.ndarray,
    avg_block_size: float = 20.0,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Politis-Romano stationary block bootstrap.

    At each step, with probability p = 1/avg_block_size, start a new
    random block; otherwise continue the current block by one day,
    wrapping around the end of the series if needed.

    The resulting resample is stationary (its distribution doesn't
    depend on the starting point), which is a useful property for
    statistical inference.  Slower than fixed-block because of the
    random block-length Bernoulli trials.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    avg_block_size : float
        Expected length of each block (the mean of the geometric
        distribution over block lengths).
    n_samples : int or None
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray, shape (n_samples, N)
    """
    T, N = returns.shape
    if n_samples is None:
        n_samples = T
    if avg_block_size < 1.0 or avg_block_size > T:
        raise ValueError(f"avg_block_size must be in [1, T], got {avg_block_size}")
    if rng is None:
        rng = np.random.default_rng()

    p = 1.0 / avg_block_size

    # Decide at each step whether to restart.  Vectorize the Bernoulli draws.
    restart = rng.random(n_samples) < p
    restart[0] = True  # always "restart" at the beginning

    # First, draw all starting positions we might need (at most n_samples of them)
    # Only len(restart.sum()) will actually be used, but draw extra for simplicity
    starts = rng.integers(low=0, high=T, size=n_samples)

    # Build index array: on restart days, take the next start; otherwise
    # advance the previous index by 1 (with wraparound modulo T).
    idx = np.empty(n_samples, dtype=np.int64)
    idx[0] = starts[0]
    start_ptr = 1
    for t in range(1, n_samples):
        if restart[t]:
            idx[t] = starts[start_ptr]
            start_ptr += 1
        else:
            idx[t] = (idx[t - 1] + 1) % T

    return returns[idx]


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Bootstrap smoke test\n" + "=" * 60)

    from src.data_loader import load_returns
    returns_df = load_returns()
    returns = returns_df.to_numpy()
    T, N = returns.shape
    print(f"Input: returns shape = ({T}, {N})")

    # --- Test 1: shape and basic sanity ---
    print("\n[Test 1: single fixed-block bootstrap]")
    boot = fixed_block_bootstrap(returns, block_size=20, rng=np.random.default_rng(42))
    print(f"  output shape = {boot.shape}  (expected ({T}, {N}))")
    assert boot.shape == (T, N)

    # Every row of boot must equal some row of the original returns
    # (we sample rows, we don't interpolate).  Check a few spot rows.
    matches = (returns == boot[0]).all(axis=1).any()
    print(f"  row 0 of bootstrap exists in original: {matches}  (expected True)")
    assert matches

    # --- Test 2: batch bootstrap ---
    print("\n[Test 2: batch of 50 bootstraps]")
    batch = batch_fixed_block_bootstrap(returns, n_bootstrap=50, block_size=20, seed=42)
    print(f"  output shape = {batch.shape}  (expected (50, {T}, {N}))")
    assert batch.shape == (50, T, N)

    # --- Test 3: mean-preserving property (moments should be close to original) ---
    print("\n[Test 3: distribution preservation over 200 resamples]")
    batch = batch_fixed_block_bootstrap(returns, n_bootstrap=200, block_size=20, seed=0)
    # Across all (B=200, T) observations we have 200*T draws; by bootstrap theory
    # the empirical mean and std should be very close to the source's.
    boot_mean = batch.mean(axis=(0, 1))   # average per asset across all samples
    orig_mean = returns.mean(axis=0)
    mean_diff = np.abs(boot_mean - orig_mean).max()
    boot_std = batch.std(axis=(0, 1))
    orig_std = returns.std(axis=0)
    std_diff = np.abs(boot_std - orig_std).max()
    print(f"  max |mean_boot - mean_orig| = {mean_diff:.2e}  (expected < 1e-3)")
    print(f"  max |std_boot  - std_orig|  = {std_diff:.2e}  (expected < 1e-3)")

    # --- Test 4: autocorrelation preservation ---
    # Block bootstrap should preserve short-lag autocorrelation much better
    # than iid sampling.  Compare lag-1 autocorrelation of the first asset.
    print("\n[Test 4: lag-1 autocorrelation preservation]")
    asset_0 = returns[:, 0]
    acf_orig = np.corrcoef(asset_0[:-1], asset_0[1:])[0, 1]

    # Block bootstrap
    boot_block = fixed_block_bootstrap(returns, block_size=20, rng=np.random.default_rng(7))
    acf_block = np.corrcoef(boot_block[:-1, 0], boot_block[1:, 0])[0, 1]

    # iid (block_size=1) — should destroy autocorrelation
    boot_iid = fixed_block_bootstrap(returns, block_size=1, rng=np.random.default_rng(7))
    acf_iid = np.corrcoef(boot_iid[:-1, 0], boot_iid[1:, 0])[0, 1]

    print(f"  original lag-1 ACF (AAPL): {acf_orig:+.4f}")
    print(f"  block (k=20)   lag-1 ACF : {acf_block:+.4f}  (expected close to original)")
    print(f"  iid   (k=1)    lag-1 ACF : {acf_iid:+.4f}  (expected ~0)")

    # --- Test 5: stationary variant ---
    print("\n[Test 5: stationary bootstrap shape]")
    boot_stat = stationary_bootstrap(returns, avg_block_size=20.0,
                                     rng=np.random.default_rng(42))
    print(f"  output shape = {boot_stat.shape}  (expected ({T}, {N}))")
    assert boot_stat.shape == (T, N)

    # --- Test 6: different seeds produce different samples ---
    print("\n[Test 6: reproducibility and independence]")
    b_a = batch_fixed_block_bootstrap(returns, n_bootstrap=5, seed=1)
    b_b = batch_fixed_block_bootstrap(returns, n_bootstrap=5, seed=1)
    b_c = batch_fixed_block_bootstrap(returns, n_bootstrap=5, seed=2)
    print(f"  seed=1 twice identical: {np.array_equal(b_a, b_b)}  (expected True)")
    print(f"  seed=1 vs seed=2 differ: {not np.array_equal(b_a, b_c)}  (expected True)")