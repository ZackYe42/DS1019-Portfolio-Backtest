"""Find WHICH problems are failing on the GPU max_sharpe."""
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

    rng = np.random.default_rng(42)
    B = 100
    print(f"Generating {B} test problems (same as failing test)...")
    Sigmas = np.empty((B, N, N), dtype=np.float64)
    excess_all = np.empty((B, N), dtype=np.float64)
    for b in range(B):
        start = rng.integers(0, T - 252)
        window = returns[start:start + 252]
        Sigmas[b] = np.cov(window, rowvar=False) + 1e-10 * np.eye(N)
        excess_all[b] = window.mean(axis=0)

    # Per-problem diagnostics
    print("\nExcess return stats per problem:")
    excess_min_per_b = excess_all.min(axis=1)
    excess_max_per_b = excess_all.max(axis=1)
    excess_pos_count = (excess_all > 0).sum(axis=1)
    print(f"  across all {B}: excess.min range = [{excess_min_per_b.min():.2e}, {excess_min_per_b.max():.2e}]")
    print(f"                  excess.max range = [{excess_max_per_b.min():.2e}, {excess_max_per_b.max():.2e}]")
    print(f"                  #positive range  = [{excess_pos_count.min()}, {excess_pos_count.max()}]")

    # All-negative excess?
    all_neg = np.where(excess_max_per_b <= 0)[0]
    print(f"  problems with ALL non-positive excess: {len(all_neg)}")
    if len(all_neg) > 0:
        print(f"    indices: {all_neg[:10].tolist()}")

    # Warm up GPU
    print("\nWarming GPU...")
    _ = solve_max_sharpe_batch_gpu(Sigmas[:2], excess_all[:2])

    # CPU
    print("Running CPU...")
    cpu_weights = np.zeros((B, N), dtype=np.float64)
    for b in range(B):
        cpu_weights[b] = _max_sharpe_njit(Sigmas[b].copy(), excess_all[b], 5000)

    # GPU
    print("Running GPU...")
    gpu_weights = solve_max_sharpe_batch_gpu(Sigmas, excess_all)

    # Per-problem Sharpe comparison
    print("\n" + "=" * 70)
    print("Per-problem Sharpe diff (top 10 worst)")
    print("=" * 70)
    diffs = []
    for b in range(B):
        wc = cpu_weights[b]
        wg = gpu_weights[b].astype(np.float64)
        S = Sigmas[b]
        mu = excess_all[b]
        vc = wc @ S @ wc
        vg = wg @ S @ wg
        sc = (wc @ mu) / np.sqrt(vc) if vc > 0 else 0
        sg = (wg @ mu) / np.sqrt(vg) if vg > 0 else 0
        rel = abs(sc - sg) / max(abs(sc), 1e-20)
        diffs.append((b, rel, sc, sg,
                      excess_all[b].min(), excess_all[b].max(),
                      (excess_all[b] > 0).sum(),
                      wg.sum(), wg.min(), np.isnan(wg).any()))

    diffs.sort(key=lambda x: -x[1])
    print(f"{'idx':>4s} {'rel_diff':>12s} {'S_cpu':>10s} {'S_gpu':>10s} "
          f"{'ex.min':>10s} {'ex.max':>10s} {'#pos':>4s} "
          f"{'sum(wg)':>10s} {'min(wg)':>10s} {'NaN?':>5s}")
    for b, rel, sc, sg, emn, emx, npos, swg, mwg, nan in diffs[:10]:
        print(f"{b:>4d} {rel:>12.4e} {sc:>10.4f} {sg:>10.4f} "
              f"{emn:>+10.2e} {emx:>+10.2e} {npos:>4d} "
              f"{swg:>10.4f} {mwg:>+10.2e} {str(nan):>5s}")

    # Count PASS/FAIL
    passes = sum(1 for (_, r, *_) in diffs if r < 1e-3)
    print(f"\n{passes}/{B} problems within 1e-3 relative Sharpe tolerance")


if __name__ == "__main__":
    main()