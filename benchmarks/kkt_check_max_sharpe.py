"""Check KKT optimality of GPU vs CPU solutions for the failing problems."""
import numpy as np
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main():
    from src.data_loader import load_returns
    from src.versions.v3_numba import _max_sharpe_njit
    from src.versions.v5_gpu_kernels import solve_max_sharpe_batch_gpu

    returns = load_returns().to_numpy()[:, :96]
    T, N = returns.shape

    # Reproduce the same problems as before
    rng = np.random.default_rng(42)
    B = 100
    Sigmas = np.empty((B, N, N), dtype=np.float64)
    excess_all = np.empty((B, N), dtype=np.float64)
    for b in range(B):
        start = rng.integers(0, T - 252)
        window = returns[start:start + 252]
        Sigmas[b] = np.cov(window, rowvar=False) + 1e-10 * np.eye(N)
        excess_all[b] = window.mean(axis=0)

    _ = solve_max_sharpe_batch_gpu(Sigmas[:2], excess_all[:2])

    cpu_weights = np.zeros((B, N), dtype=np.float64)
    for b in range(B):
        cpu_weights[b] = _max_sharpe_njit(Sigmas[b].copy(), excess_all[b], 5000)
    gpu_weights = solve_max_sharpe_batch_gpu(Sigmas, excess_all)

    def sharpe_grad(w, Sigma, excess):
        """Gradient of Sharpe(w) = (excess^T w) / sqrt(w^T Sigma w)."""
        Sw = Sigma @ w
        vol_sq = w @ Sw
        vol = np.sqrt(vol_sq)
        ret = excess @ w
        # d/dw [ret/vol] = excess/vol - ret/(vol^3) * Sw
        return excess / vol - (ret / vol**3) * Sw

    def kkt_check(w, Sigma, excess, tol=1e-5):
        """
        Check KKT conditions for max Sharpe on the simplex.
        For max Sharpe (equivalent to: max excess^T w s.t. w^T Sigma w = c, w in simplex),
        the gradient at the optimum should satisfy:
            grad[i] == lambda  if w[i] > 0  (active)
            grad[i] <= lambda  if w[i] == 0 (inactive)
        where lambda is common to all active assets.
        """
        grad = sharpe_grad(w, Sigma, excess)
        active = w > 1e-6
        if not active.any():
            return np.nan, np.nan, np.nan  # degenerate

        grad_active = grad[active]
        grad_inactive = grad[~active] if (~active).any() else np.array([-np.inf])

        # Lambda = mean of active gradients (should all be equal at optimum)
        lam = grad_active.mean()

        # Measures of KKT violation
        # 1. Spread among active gradients (should be 0 at optimum)
        active_spread = grad_active.max() - grad_active.min()
        # 2. Max inactive gradient above lambda (should be <=0)
        inactive_excess = (grad_inactive.max() - lam) if grad_inactive.size else -np.inf

        return lam, active_spread, inactive_excess

    # Check CPU vs GPU KKT for the 10 failing cases
    def sharpe(w, mu, S):
        v = w @ S @ w
        if v <= 0:
            return 0.0
        return float((w @ mu) / np.sqrt(v))

    # Find the bad cases
    bad_indices = []
    for b in range(B):
        sc = sharpe(cpu_weights[b], excess_all[b], Sigmas[b])
        sg = sharpe(gpu_weights[b].astype(np.float64), excess_all[b], Sigmas[b])
        rel = abs(sc - sg) / max(abs(sc), 1e-20)
        if rel > 1e-3:
            bad_indices.append((b, rel, sc, sg))

    bad_indices.sort(key=lambda x: -x[1])

    print("KKT analysis of failing cases:")
    print("=" * 100)
    print(f"{'b':>4s} {'Scpu':>8s} {'Sgpu':>8s} "
          f"{'CPU_spread':>12s} {'CPU_inact':>12s} "
          f"{'GPU_spread':>12s} {'GPU_inact':>12s} {'verdict':>20s}")
    print("-" * 100)

    for b, rel, sc, sg in bad_indices[:10]:
        S = Sigmas[b]
        mu = excess_all[b]
        wc = cpu_weights[b]
        wg = gpu_weights[b].astype(np.float64)

        lam_c, spread_c, inact_c = kkt_check(wc, S, mu)
        lam_g, spread_g, inact_g = kkt_check(wg, S, mu)

        # Both optimal? Then it's plateau ambiguity.
        # GPU clearly suboptimal? Then inact_g > 0 significantly.
        # A small positive inact_excess means "an inactive asset's gradient beats
        # the active mean" - the GPU could have improved by adding that asset.
        if inact_g > 1e-2:
            verdict = "GPU_SUBOPTIMAL"
        elif spread_g > 0.1 * abs(lam_g):
            verdict = "GPU_NOT_CONVERGED"
        elif sg < sc - 1e-4:
            verdict = "plateau_diff"
        else:
            verdict = "both_ok"

        print(f"{b:>4d} {sc:>8.4f} {sg:>8.4f} "
              f"{spread_c:>12.4e} {inact_c:>+12.4e} "
              f"{spread_g:>12.4e} {inact_g:>+12.4e} {verdict:>20s}")

    print()
    print("Legend:")
    print("  spread   = max grad active - min grad active  (should be 0 at optimum)")
    print("  inact    = max(inactive grad) - lambda        (should be <= 0)")
    print("  GPU_SUBOPTIMAL = GPU could improve by adding an inactive asset -> real bug")
    print("  plateau_diff   = Both are locally optimal, just different corners")


if __name__ == "__main__":
    main()