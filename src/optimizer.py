"""
Portfolio weight optimizers.

Given a covariance matrix (and optionally expected returns), produce a
vector of portfolio weights that satisfy the classical optimization problems
of modern portfolio theory.

Two objectives are provided:
    - min_variance:  minimize  w^T Sigma w     s.t. sum(w)=1, w>=0
    - max_sharpe:    maximize (mu^T w - rf) / sqrt(w^T Sigma w)
                                              s.t. sum(w)=1, w>=0

Two implementations are provided for each:
    - *_cvxpy:    convex-optimization reference (clean, trusted, slow)
    - *_numpy:    projected-gradient solver (fast, predictable work;
                  used as the baseline for Numba/GPU optimization)

The projected-gradient solver works by taking gradient steps and
projecting onto the long-only simplex {w : sum(w)=1, w>=0} after each
step.  It converges to within 1e-6 of the cvxpy solution in ~500 iters
and does a FIXED amount of work per call, which is ideal for compilation
and GPU vectorization.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_psd(cov: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Add a tiny multiple of the identity to guarantee positive-definiteness."""
    N = cov.shape[0]
    return cov + jitter * np.eye(N)


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection of a vector onto the probability simplex
        { w : sum(w) = 1, w >= 0 }

    Algorithm: Duchi, Shalev-Shwartz, Singer, Chandra (ICML 2008),
    "Efficient projections onto the l1-ball for learning in high dimensions."

    Complexity: O(N log N) dominated by the sort.

    Parameters
    ----------
    v : np.ndarray, shape (N,)

    Returns
    -------
    np.ndarray, shape (N,)
        The projection of v onto the simplex.
    """
    N = v.shape[0]
    # Sort v in descending order
    u = np.sort(v)[::-1]
    # Cumulative sum of the sorted values
    cssv = np.cumsum(u) - 1.0
    # Find the largest index where u[i] - cssv[i]/(i+1) > 0
    ind = np.arange(1, N + 1)
    cond = u - cssv / ind > 0
    rho = np.where(cond)[0].max() + 1  # +1 because we want count, not index
    # The threshold tau
    tau = cssv[rho - 1] / rho
    # Project
    return np.maximum(v - tau, 0.0)


# -----------------------------------------------------------------------------
# Minimum-variance portfolio
# -----------------------------------------------------------------------------

def min_variance_cvxpy(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """Reference minimum-variance solver using cvxpy."""
    N = cov.shape[0]
    Sigma = _ensure_psd(cov)

    w = cp.Variable(N)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
    constraints = [cp.sum(w) == 1.0]
    if long_only:
        constraints.append(w >= 0.0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if w.value is None:
        raise RuntimeError(f"cvxpy failed to solve min-variance: status={prob.status}")
    return np.asarray(w.value, dtype=np.float64)


def min_variance_numpy(
    cov: np.ndarray,
    long_only: bool = True,
    n_iter: int = 500,
    step_size: float | None = None,
) -> np.ndarray:
    """
    Projected-gradient minimum-variance solver.

    We minimize f(w) = w^T Sigma w.  Gradient is 2 Sigma w.  Each iteration:
        1. w <- w - step * gradient
        2. w <- project_simplex(w)    (if long_only)

    step_size defaults to 1 / (2 * lambda_max(Sigma)) for guaranteed
    convergence.  Fixed iteration count => fixed amount of work => easy
    to compile with Numba and vectorize on GPU.

    For long_only=False, the problem has a closed form:
        w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)
    which we use directly (no iteration).
    """
    N = cov.shape[0]
    Sigma = _ensure_psd(cov)

    if not long_only:
        # Closed-form unconstrained (except sum=1) solution
        ones = np.ones(N)
        x = np.linalg.solve(Sigma, ones)
        return x / x.sum()

    # Choose step size based on largest eigenvalue of Sigma
    if step_size is None:
        lam_max = np.linalg.eigvalsh(Sigma).max()
        step_size = 1.0 / (2.0 * lam_max)

    # Initialize at equal weights
    w = np.full(N, 1.0 / N)

    for _ in range(n_iter):
        grad = 2.0 * Sigma @ w
        w = project_simplex(w - step_size * grad)

    return w


# -----------------------------------------------------------------------------
# Maximum-Sharpe portfolio
# -----------------------------------------------------------------------------

def max_sharpe_cvxpy(
    cov: np.ndarray,
    mu: np.ndarray,
    rf: float = 0.0,
    long_only: bool = True,
) -> np.ndarray:
    """
    Reference maximum-Sharpe solver using cvxpy.

    Uses the classical reformulation: substitute y = kappa * w with the
    constraint (mu - rf)^T y = 1.  Then min y^T Sigma y is a convex QP,
    and w = y / sum(y).
    """
    N = cov.shape[0]
    Sigma = _ensure_psd(cov)
    excess = mu - rf

    if long_only and (excess <= 0).all():
        raise ValueError("max_sharpe: no asset has positive excess return (long-only)")

    y = cp.Variable(N)
    objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma)))
    constraints = [excess @ y == 1.0]
    if long_only:
        constraints.append(y >= 0.0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if y.value is None:
        raise RuntimeError(f"cvxpy failed to solve max-Sharpe: status={prob.status}")

    y_val = np.asarray(y.value, dtype=np.float64)
    return y_val / y_val.sum()


def max_sharpe_numpy(
    cov: np.ndarray,
    mu: np.ndarray,
    rf: float = 0.0,
    long_only: bool = True,
    n_iter: int = 5000,
) -> np.ndarray:
    """
    Frank-Wolfe maximum-Sharpe solver.

    The long-only max-Sharpe portfolio lives on the probability simplex
    { w : sum(w) = 1, w >= 0 }.  Frank-Wolfe is a beautifully simple
    algorithm for convex problems on a simplex:

        repeat:
            1. compute gradient of Sharpe at current w
            2. find the corner (single asset) with max gradient component
            3. move a fraction gamma = 2/(k+2) toward that corner

    Each iteration does ONE matrix-vector product, one argmax, and a
    convex combination.  There is no step-size to tune, no projection
    to compute, and the iterate stays on the simplex by construction.
    This structure makes the algorithm trivial to compile with Numba
    and trivial to vectorize on GPU (we can batch thousands of these
    solvers running in parallel on different covariance matrices).

    For long_only=False, the unconstrained tangency portfolio has a
    closed form:
        y* = Sigma^{-1} (mu - rf)
        w* = y* / sum(y*)
    """
    N = cov.shape[0]
    Sigma = _ensure_psd(cov)
    excess = mu - rf

    if long_only and (excess <= 0).all():
        raise ValueError("max_sharpe: no asset has positive excess return (long-only)")

    if not long_only:
        y = np.linalg.solve(Sigma, excess)
        return y / y.sum()

    # Frank-Wolfe on the simplex, starting from equal weights
    w = np.ones(N) / N
    for k in range(n_iter):
        Sw = Sigma @ w
        port_var = w @ Sw
        port_vol = np.sqrt(port_var)
        port_ret = excess @ w

        # Gradient of Sharpe(w) = (mu^T w) / sqrt(w^T Sigma w)
        grad = excess / port_vol - (port_ret / (port_vol ** 3)) * Sw

        # The best "atom" on the simplex is the standard basis vector e_i
        # where i = argmax(grad)
        best = int(np.argmax(grad))

        # Frank-Wolfe step: w <- (1 - gamma) * w + gamma * e_best
        gamma = 2.0 / (k + 2.0)
        w *= 1.0 - gamma
        w[best] += gamma

    return w


# -----------------------------------------------------------------------------
# Dispatchers
# -----------------------------------------------------------------------------

def min_variance(
    cov: np.ndarray,
    method: str = "numpy",
    long_only: bool = True,
) -> np.ndarray:
    """Dispatch to a minimum-variance implementation by name."""
    if method == "numpy":
        return min_variance_numpy(cov, long_only=long_only)
    if method == "cvxpy":
        return min_variance_cvxpy(cov, long_only=long_only)
    raise ValueError(f"unknown method '{method}', choose from ('numpy', 'cvxpy')")


def max_sharpe(
    cov: np.ndarray,
    mu: np.ndarray,
    rf: float = 0.0,
    method: str = "numpy",
    long_only: bool = True,
) -> np.ndarray:
    """Dispatch to a maximum-Sharpe implementation by name."""
    if method == "numpy":
        return max_sharpe_numpy(cov, mu, rf=rf, long_only=long_only)
    if method == "cvxpy":
        return max_sharpe_cvxpy(cov, mu, rf=rf, long_only=long_only)
    raise ValueError(f"unknown method '{method}', choose from ('numpy', 'cvxpy')")


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import load_returns
    from src.estimators import estimate_cov

    print("Optimizer smoke test\n" + "=" * 60)

    def sharpe_of(w, mu, cov, rf=0.0):
        return float((w @ (mu - rf)) / np.sqrt(w @ cov @ w))

    def variance_of(w, cov):
        return float(w @ cov @ w)

    # --- Fake data ---
    rng = np.random.default_rng(0)
    N = 10
    A = rng.standard_normal((N, N))
    fake_cov = A @ A.T + 0.1 * np.eye(N)
    fake_mu = rng.standard_normal(N) * 0.001 + 0.0005

    print(f"\n[Fake data, N={N}]")
    w_mv_np = min_variance_numpy(fake_cov)
    w_mv_cx = min_variance_cvxpy(fake_cov)
    print(f"  min-variance  |w diff|_inf = {np.abs(w_mv_np - w_mv_cx).max():.2e}")
    print(f"                variance     numpy={variance_of(w_mv_np, fake_cov):.6e}  "
          f"cvxpy={variance_of(w_mv_cx, fake_cov):.6e}")

    w_ms_np = max_sharpe_numpy(fake_cov, fake_mu)
    w_ms_cx = max_sharpe_cvxpy(fake_cov, fake_mu)
    print(f"  max-Sharpe    |w diff|_inf = {np.abs(w_ms_np - w_ms_cx).max():.2e}")
    print(f"                Sharpe       numpy={sharpe_of(w_ms_np, fake_mu, fake_cov):.6f}  "
          f"cvxpy={sharpe_of(w_ms_cx, fake_mu, fake_cov):.6f}")

    # --- Real data ---
    print(f"\n[Real S&P 100 data, last 252 days]")
    returns = load_returns().to_numpy()
    window = returns[-252:]
    cov = estimate_cov(window, method="ledoit_wolf")
    mu = window.mean(axis=0)

    w_mv_np = min_variance_numpy(cov)
    w_mv_cx = min_variance_cvxpy(cov)
    print(f"  min-variance  |w diff|_inf = {np.abs(w_mv_np - w_mv_cx).max():.2e}")
    print(f"                variance     numpy={variance_of(w_mv_np, cov):.6e}  "
          f"cvxpy={variance_of(w_mv_cx, cov):.6e}")
    print(f"                n_nonzero    numpy={(w_mv_np > 1e-6).sum()}  "
          f"cvxpy={(w_mv_cx > 1e-6).sum()}")

    w_ms_np = max_sharpe_numpy(cov, mu)
    w_ms_cx = max_sharpe_cvxpy(cov, mu)
    print(f"  max-Sharpe    |w diff|_inf = {np.abs(w_ms_np - w_ms_cx).max():.2e}")
    print(f"                Sharpe       numpy={sharpe_of(w_ms_np, mu, cov):.6f}  "
          f"cvxpy={sharpe_of(w_ms_cx, mu, cov):.6f}")
    print(f"                n_nonzero    numpy={(w_ms_np > 1e-6).sum()}  "
          f"cvxpy={(w_ms_cx > 1e-6).sum()}")