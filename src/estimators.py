"""
Covariance matrix estimators for portfolio construction.

Three estimators are provided:
    - sample_cov:   classical sample covariance (MLE)
    - ledoit_wolf:  linear shrinkage toward scaled identity
    - ewma_cov:     exponentially-weighted covariance (RiskMetrics style)

Each takes a (T, N) array of log returns and returns an (N, N) covariance
matrix.  All implementations here are written in plain NumPy for correctness;
optimized versions (Numba, GPU) will live in src/versions/.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# 1. Sample covariance
# -----------------------------------------------------------------------------

def sample_cov(returns: np.ndarray) -> np.ndarray:
    """
    Classical sample covariance matrix.

    Given T observations of N asset returns, returns the (N, N) covariance
    matrix estimated as
        S = (1 / (T - 1)) * (R - mean)^T (R - mean)

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Log returns.  Each row is one time step, each column is one asset.

    Returns
    -------
    np.ndarray, shape (N, N)
        Sample covariance matrix.
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D, got shape {returns.shape}")
    T, N = returns.shape
    if T < 2:
        raise ValueError(f"need at least 2 observations, got T={T}")

    # np.cov with rowvar=False treats columns as variables (what we want)
    return np.cov(returns, rowvar=False, ddof=1)


# -----------------------------------------------------------------------------
# 2. Ledoit-Wolf shrinkage
# -----------------------------------------------------------------------------

def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf linear shrinkage estimator toward the scaled-identity target.

    Returns
        Sigma_hat = (1 - alpha) * S + alpha * mu * I
    where
        S    = sample covariance
        mu   = trace(S) / N  (average variance)
        I    = identity matrix of size N
        alpha = optimal shrinkage intensity in [0, 1], computed from data

    Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices."

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)

    Returns
    -------
    np.ndarray, shape (N, N)
        Shrinkage covariance estimate.
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D, got shape {returns.shape}")
    T, N = returns.shape
    if T < 2:
        raise ValueError(f"need at least 2 observations, got T={T}")

    # Center the returns
    X = returns - returns.mean(axis=0, keepdims=True)

    # Sample covariance with divisor T (not T-1) to match Ledoit-Wolf formulas
    S = (X.T @ X) / T

    # Shrinkage target: scaled identity with mu = tr(S)/N
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # Compute optimal shrinkage intensity (see Ledoit-Wolf 2004, eq. 14)
    #   pi_hat: sum of asymptotic variances of entries of S
    #   gamma_hat: squared Frobenius distance between S and target
    #   alpha   = pi_hat / (gamma_hat * T), clipped to [0, 1]
    X2 = X ** 2
    pi_mat = (X2.T @ X2) / T - S ** 2
    pi_hat = pi_mat.sum()

    gamma_hat = np.sum((S - F) ** 2)

    # Kappa estimate (shrinkage intensity, before clipping and /T)
    if gamma_hat < 1e-12:
        # Degenerate case: S already equals target, no shrinkage needed
        alpha = 0.0
    else:
        kappa = pi_hat / gamma_hat
        alpha = max(0.0, min(1.0, kappa / T))

    return (1.0 - alpha) * S + alpha * F


# -----------------------------------------------------------------------------
# 3. Exponentially-weighted moving average covariance
# -----------------------------------------------------------------------------

def ewma_cov(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    Exponentially-weighted covariance (RiskMetrics convention).

    Weights decay geometrically: observation at lag k receives weight
    proportional to lam^k.  Default lam=0.94 matches J.P. Morgan's
    RiskMetrics technical document for daily data.

    Formally,
        Sigma = sum_{k=0}^{T-1} w_k (r_{T-k} - r_bar) (r_{T-k} - r_bar)^T
    with
        w_k = (1 - lam) * lam^k   (normalized so sum_k w_k = 1 - lam^T)

    For stability we normalize the final weights so they sum to exactly 1.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    lam : float, in (0, 1)
        Decay factor.  Higher values put more weight on the distant past.

    Returns
    -------
    np.ndarray, shape (N, N)
        Exponentially-weighted covariance.
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D, got shape {returns.shape}")
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    T, N = returns.shape
    if T < 2:
        raise ValueError(f"need at least 2 observations, got T={T}")

    # Weights: most recent observation has index T-1 and gets the largest weight.
    # w_k = (1 - lam) * lam^k for k = 0, 1, ..., T-1 (k measures lag from present)
    k = np.arange(T - 1, -1, -1, dtype=np.float64)  # lags: T-1, T-2, ..., 1, 0
    weights = (1.0 - lam) * (lam ** k)
    weights /= weights.sum()  # renormalize so weights sum to 1

    # Weighted mean
    mu = (weights[:, None] * returns).sum(axis=0, keepdims=True)

    # Weighted centered returns
    X = returns - mu

    # Cov = X^T diag(w) X
    # Efficient form: (X * sqrt(w))^T (X * sqrt(w))
    Xw = X * np.sqrt(weights)[:, None]
    return Xw.T @ Xw


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

ESTIMATORS = {
    "sample": sample_cov,
    "ledoit_wolf": ledoit_wolf,
    "ewma": ewma_cov,
}


def estimate_cov(returns: np.ndarray, method: str = "sample", **kwargs) -> np.ndarray:
    """
    Dispatch to one of the three estimators by name.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    method : {"sample", "ledoit_wolf", "ewma"}
    **kwargs : passed through to the underlying estimator (e.g. lam for EWMA)
    """
    if method not in ESTIMATORS:
        raise ValueError(f"unknown estimator '{method}', choose from {list(ESTIMATORS)}")
    return ESTIMATORS[method](returns, **kwargs)


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check: generate fake returns and run all three estimators.
    rng = np.random.default_rng(42)
    T, N = 252, 10
    fake_returns = rng.standard_normal((T, N)) * 0.01

    print(f"Testing estimators on fake data with T={T}, N={N}\n")

    for name in ESTIMATORS:
        cov = estimate_cov(fake_returns, method=name)
        eig = np.linalg.eigvalsh(cov)
        print(f"[{name:12s}]  shape={cov.shape}  "
              f"eig_range=({eig.min():+.2e}, {eig.max():+.2e})  "
              f"pd={bool((eig > -1e-10).all())}")

    # Real data check
    print("\nTesting on real S&P 100 returns...\n")
    from src.data_loader import load_returns
    returns = load_returns().to_numpy()
    print(f"Real returns shape: {returns.shape}\n")

    # Use last 252 days (one trading year)
    recent = returns[-252:]
    for name in ESTIMATORS:
        cov = estimate_cov(recent, method=name)
        eig = np.linalg.eigvalsh(cov)
        print(f"[{name:12s}]  shape={cov.shape}  "
              f"eig_range=({eig.min():+.2e}, {eig.max():+.2e})  "
              f"condition={eig.max()/max(eig.min(), 1e-20):.2e}")