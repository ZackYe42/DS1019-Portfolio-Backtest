"""Diagnostic: measure each v3 component in isolation."""
import time
import numpy as np
from src.versions.v3_numba import (
    _sample_cov_njit, _ledoit_wolf_njit, _ewma_cov_njit,
    _min_variance_njit, _max_sharpe_njit,
)
from src.estimators import sample_cov, ledoit_wolf, ewma_cov
from src.optimizer import min_variance_numpy, max_sharpe_numpy

np.random.seed(42)
T, N = 252, 96
window = np.random.randn(T, N) * 0.01
cov = np.cov(window, rowvar=False) + 1e-10 * np.eye(N)
mu = window.mean(axis=0)
lam = float(np.linalg.eigvalsh(cov).max())
step = 1.0 / (2.0 * lam)

# Warmup
_sample_cov_njit(window); _ledoit_wolf_njit(window); _ewma_cov_njit(window, 0.94)
_min_variance_njit(cov, step, 500); _max_sharpe_njit(cov, mu, 5000)

REPS = 100

def bench(fn, args, reps=REPS):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    return (time.perf_counter() - t0) / reps * 1000

print(f"{'Function':<35s} {'numpy (ms)':>12s} {'numba (ms)':>12s} {'speedup':>10s}")
print("-" * 75)

# Estimators
r = [
    ("sample_cov",       sample_cov,        _sample_cov_njit,      (window,)),
    ("ledoit_wolf",      ledoit_wolf,       _ledoit_wolf_njit,     (window,)),
    ("ewma_cov",         ewma_cov,          lambda w: _ewma_cov_njit(w, 0.94),    (window,)),
]
for name, f_np, f_nb, args in r:
    t_np = bench(f_np, args)
    t_nb = bench(f_nb, args)
    print(f"{name:<35s} {t_np:>12.3f} {t_nb:>12.3f} {t_np/t_nb:>9.1f}x")

# Optimizers (lower reps since they're slower)
t_mv_np = bench(min_variance_numpy, (cov,), reps=20)
t_mv_nb = bench(_min_variance_njit, (cov, step, 500), reps=20)
print(f"{'min_variance 500 iter':<35s} {t_mv_np:>12.3f} {t_mv_nb:>12.3f} {t_mv_np/t_mv_nb:>9.1f}x")

t_ms_np = bench(max_sharpe_numpy, (cov, mu), reps=10)
t_ms_nb = bench(_max_sharpe_njit, (cov, mu, 5000), reps=10)
print(f"{'max_sharpe 5000 iter':<35s} {t_ms_np:>12.3f} {t_ms_nb:>12.3f} {t_ms_np/t_ms_nb:>9.1f}x")

# eigvalsh
t_eig = bench(lambda c: float(np.linalg.eigvalsh(c).max()), (cov,))
print(f"{'eigvalsh (always numpy)':<35s} {t_eig:>12.3f} {'-':>12s} {'-':>10s}")

# Simulate ONE full backtest of sample|min_variance to match your smoke test
print("\n--- Full sample|min_variance single backtest breakdown ---")
# 130 rebalances, 2514 trading days, N=96
n_rebs = 130
t_est = n_rebs * bench(_sample_cov_njit, (window,))
t_eig_tot = n_rebs * t_eig
t_opt = n_rebs * t_mv_nb
t_daily = 25  # rough, from earlier diagnostic
print(f"  sample_cov_njit x {n_rebs}:       {t_est:.0f} ms")
print(f"  eigvalsh x {n_rebs}:              {t_eig_tot:.0f} ms")
print(f"  min_variance_njit x {n_rebs}:     {t_opt:.0f} ms")
print(f"  daily loop (estimate):           {t_daily:.0f} ms")
print(f"  TOTAL predicted:                 {t_est + t_eig_tot + t_opt + t_daily:.0f} ms")
print(f"  Your observed ~47s/60 backtests = {47000/60:.0f} ms/backtest")