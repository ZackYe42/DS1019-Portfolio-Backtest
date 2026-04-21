"""
v5 GPU kernels for batched portfolio optimization.

Each CUDA thread block solves ONE portfolio problem (min-variance or
max-Sharpe).  Inside a block, the N threads (one per asset) cooperate
to do matrix-vector products, reductions, projections, etc.

Thread-block layout:
    blockIdx.x  = which problem in the batch (0..B-1)
    threadIdx.x = which asset index          (0..N-1)

Shared memory per block (min-variance):
    s_Sigma:  (N, N)    covariance matrix
    s_w:      (N,)      current portfolio weights
    s_Sw:     (N,)      Sigma @ w cache
    s_v:      (N,)      workspace
    s_red:    (128,)    parallel-reduction scratch (padded to next pow of 2)
    s_mini:   (8,)      scalar broadcast scratch

Shared memory per block (max-Sharpe) adds:
    s_excess: (N,)      excess returns per asset
    s_grad:   (N,)      gradient workspace
    s_idx:    (128,)    argmax index scratch
    s_best:   (1,)      broadcast argmax result

Key design choices:
    - Simplex projection uses bisection + parallel reduction (NOT sort),
      which was 10x faster than the Duchi 2008 sort-based projection in
      our benchmarks.
    - All reductions are tree-style (O(log N)) using the standard CUDA
      pattern with RED_SIZE=128 (next pow 2 >= N_ASSETS=96) and padding.
    - Max-Sharpe uses Frank-Wolfe (no projection needed, just argmax +
      AXPY), with incremental Sigma @ w updates.
"""

from __future__ import annotations

import math
import numpy as np
from numba import cuda, float32, int32


# =============================================================================
# CONSTANTS
# =============================================================================
N_ASSETS = 96              # must match our problem size; used for shared-mem sizing
MV_ITERS = 500             # projected-gradient iterations (matches v3 CPU)
MS_ITERS = 5000            # Frank-Wolfe iterations (matches v3 CPU)
POWER_ITERS = 50           # power iterations for lambda_max (matches v3 CPU)
PROJECTION_BISECTION_ITERS = 30   # bisection steps per projection

# Padded size for reductions (next power of 2 >= N_ASSETS).  We operate on
# 128 elements with the last 32 padded to identity values (0 for sum,
# +/-inf for min/max, -inf for argmax).
RED_SIZE = 128


# =============================================================================
# Device helpers shared across kernels
# =============================================================================

@cuda.jit(device=True, inline=True)
def _matvec_cooperative(Sigma, w, Sw, tid, N):
    """
    Sw[tid] = sum_j Sigma[tid, j] * w[j].  One row per thread.

    Caller must syncthreads() after this, before reading Sw.
    """
    if tid < N:
        acc = float32(0.0)
        for j in range(N):
            acc += Sigma[tid, j] * w[j]
        Sw[tid] = acc


@cuda.jit(device=True, inline=True)
def _parallel_dot(x, y, red, tid, N):
    """Parallel dot product via tree reduction.  On exit: red[0] has sum."""
    if tid < N:
        red[tid] = x[tid] * y[tid]
    if tid < RED_SIZE - N:
        red[N + tid] = float32(0.0)
    cuda.syncthreads()

    stride = RED_SIZE // 2
    while stride > 0:
        # Phase 1: read into local
        val = float32(0.0)
        if tid < stride:
            val = red[tid] + red[tid + stride]
        cuda.syncthreads()
        # Phase 2: write back
        if tid < stride:
            red[tid] = val
        cuda.syncthreads()
        stride //= 2


@cuda.jit(device=True, inline=True)
def _parallel_argmax(grad, red_val, red_idx, tid, N):
    """Parallel argmax via tree reduction.  On exit: red_val[0]=max, red_idx[0]=argmax."""
    if tid < N:
        red_val[tid] = grad[tid]
        red_idx[tid] = tid
    if tid < RED_SIZE - N:
        red_val[N + tid] = -float32(1e30)
        red_idx[N + tid] = 0
    cuda.syncthreads()

    stride = RED_SIZE // 2
    while stride > 0:
        chosen_val = float32(0.0)
        chosen_idx = int32(0)
        if tid < stride:
            a_val = red_val[tid]
            b_val = red_val[tid + stride]
            if b_val > a_val:
                chosen_val = b_val
                chosen_idx = red_idx[tid + stride]
            else:
                chosen_val = a_val
                chosen_idx = red_idx[tid]
        cuda.syncthreads()
        if tid < stride:
            red_val[tid] = chosen_val
            red_idx[tid] = chosen_idx
        cuda.syncthreads()
        stride //= 2

@cuda.jit(device=True, inline=True)
def _power_iteration_cooperative(Sigma, v, u, scratch, tid, N, n_iter):
    """
    Estimate lambda_max(Sigma) via power iteration.

    On entry: v[tid] = 1/sqrt(N) for all threads (caller sets this up).
    On exit:  scratch[0] = lambda_max estimate.

    Uses thread-0 serial reductions for the Rayleigh quotient and norm.
    Since power iteration only runs 50 times (not 500 like projected
    gradient), the serial cost is negligible here.
    """
    for _ in range(n_iter):
        _matvec_cooperative(Sigma, v, u, tid, N)
        cuda.syncthreads()

        if tid == 0:
            lam = float32(0.0)
            for i in range(N):
                lam += v[i] * u[i]
            scratch[0] = lam

        if tid == 0:
            norm_sq = float32(0.0)
            for i in range(N):
                norm_sq += u[i] * u[i]
            scratch[1] = math.sqrt(norm_sq)
        cuda.syncthreads()

        norm = scratch[1]
        if tid < N:
            if norm > float32(1e-30):
                v[tid] = u[tid] / norm
            else:
                v[tid] = float32(0.0)
        cuda.syncthreads()


@cuda.jit(device=True, inline=True)
def _project_simplex_bisection(v, red, mini, tid, N):
    """
    Projection onto { w : sum(w)=1, w>=0 } via bisection on threshold tau.

    Each bisection step uses a parallel tree reduction to compute
        f(tau) = sum_i max(v_i - tau, 0)
    Bisection shrinks [tau_lo, tau_hi] until f(tau) = 1.

    All tree reductions use split-phase reads/writes with an explicit
    syncthreads between the read and write to avoid warp-synchronous
    read-write races.
    """
    # --- min(v) -> mini[0] ---
    if tid < N:
        red[tid] = v[tid]
    if tid < RED_SIZE - N:
        red[N + tid] = float32(1e30)
    cuda.syncthreads()

    stride = RED_SIZE // 2
    while stride > 0:
        val = float32(1e30)
        if tid < stride:
            a = red[tid]
            b = red[tid + stride]
            val = a if a < b else b
        cuda.syncthreads()
        if tid < stride:
            red[tid] = val
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        mini[0] = red[0]
    cuda.syncthreads()

    # --- max(v) -> mini[1] ---
    if tid < N:
        red[tid] = v[tid]
    if tid < RED_SIZE - N:
        red[N + tid] = -float32(1e30)
    cuda.syncthreads()

    stride = RED_SIZE // 2
    while stride > 0:
        val = -float32(1e30)
        if tid < stride:
            a = red[tid]
            b = red[tid + stride]
            val = a if a > b else b
        cuda.syncthreads()
        if tid < stride:
            red[tid] = val
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        mini[1] = red[0]
        mini[0] = mini[0] - float32(1.0)
    cuda.syncthreads()

    # --- Bisection loop ---
    for _ in range(PROJECTION_BISECTION_ITERS):
        if tid == 0:
            mini[2] = float32(0.5) * (mini[0] + mini[1])
        cuda.syncthreads()
        tau_mid = mini[2]

        # Sum of max(v - tau_mid, 0) via parallel reduction
        if tid < N:
            diff = v[tid] - tau_mid
            red[tid] = diff if diff > float32(0.0) else float32(0.0)
        if tid < RED_SIZE - N:
            red[N + tid] = float32(0.0)
        cuda.syncthreads()

        stride = RED_SIZE // 2
        while stride > 0:
            val = float32(0.0)
            if tid < stride:
                val = red[tid] + red[tid + stride]
            cuda.syncthreads()
            if tid < stride:
                red[tid] = val
            cuda.syncthreads()
            stride //= 2

        s = red[0]

        if tid == 0:
            if s > float32(1.0):
                mini[0] = tau_mid
            else:
                mini[1] = tau_mid
        cuda.syncthreads()

    tau_final = float32(0.5) * (mini[0] + mini[1])
    if tid < N:
        diff = v[tid] - tau_final
        v[tid] = diff if diff > float32(0.0) else float32(0.0)
    cuda.syncthreads()


# =============================================================================
# Min-variance batch kernel
# =============================================================================

@cuda.jit
def min_variance_batch_kernel(Sigmas, weights_out):
    """
    Solve a batch of B min-variance problems on GPU.

    Launch config:
        Grid:  (B,)        one block per problem
        Block: (N_ASSETS,) one thread per asset

    Inputs:
        Sigmas       : (B, N_ASSETS, N_ASSETS) float32, pre-regularized
    Outputs:
        weights_out  : (B, N_ASSETS) float32
    """
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    B = Sigmas.shape[0]
    if bid >= B:
        return

    # Shared memory: ~38 KB total per block
    s_Sigma = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)
    s_w     = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_Sw    = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_v     = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_red   = cuda.shared.array(shape=RED_SIZE, dtype=float32)
    s_mini  = cuda.shared.array(shape=8, dtype=float32)

    N = N_ASSETS

    # --- Load Sigma; initialize w and v ---
    if tid < N:
        for j in range(N):
            s_Sigma[tid, j] = Sigmas[bid, tid, j]
        s_w[tid] = float32(1.0) / float32(N)
        s_v[tid] = float32(1.0) / math.sqrt(float32(N))  # init for power iter
    cuda.syncthreads()

    # --- Step 1: lambda_max via power iteration ---
    _power_iteration_cooperative(s_Sigma, s_v, s_Sw, s_mini,
                                  tid, N, POWER_ITERS)

    lam_max = s_mini[0]
    two_step = float32(1.0) / lam_max

    # --- Step 2: projected gradient descent ---
    for _ in range(MV_ITERS):
        _matvec_cooperative(s_Sigma, s_w, s_Sw, tid, N)
        cuda.syncthreads()

        if tid < N:
            s_v[tid] = s_w[tid] - two_step * s_Sw[tid]
        cuda.syncthreads()

        _project_simplex_bisection(s_v, s_red, s_mini, tid, N)

        if tid < N:
            s_w[tid] = s_v[tid]
        cuda.syncthreads()

    # --- Write result ---
    if tid < N:
        weights_out[bid, tid] = s_w[tid]


# =============================================================================
# Max-Sharpe batch kernel
# =============================================================================

@cuda.jit
def max_sharpe_batch_kernel(Sigmas, excess_all, weights_out):
    """
    Solve a batch of B max-Sharpe problems on GPU via Frank-Wolfe.

    Launch config:
        Grid:  (B,)        one block per problem
        Block: (N_ASSETS,) one thread per asset

    Inputs:
        Sigmas       : (B, N_ASSETS, N_ASSETS) float32, pre-regularized
        excess_all   : (B, N_ASSETS) float32, excess returns (mu - rf) per asset
    Outputs:
        weights_out  : (B, N_ASSETS) float32
    """
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    B = Sigmas.shape[0]
    if bid >= B:
        return

    # Shared memory: ~39 KB total per block
    s_Sigma  = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)
    s_excess = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_w      = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_Sw     = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_grad   = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_red    = cuda.shared.array(shape=RED_SIZE, dtype=float32)
    s_idx    = cuda.shared.array(shape=RED_SIZE, dtype=int32)
    s_best   = cuda.shared.array(shape=1, dtype=int32)

    N = N_ASSETS

    # --- Load Sigma, excess; initialize w ---
    if tid < N:
        for j in range(N):
            s_Sigma[tid, j] = Sigmas[bid, tid, j]
        s_excess[tid] = excess_all[bid, tid]
        s_w[tid] = float32(1.0) / float32(N)
    cuda.syncthreads()

    # Initial Sw = Sigma @ w where w is uniform 1/N:
    # Sw[i] = (1/N) * sum_j Sigma[i, j]
    if tid < N:
        acc = float32(0.0)
        for j in range(N):
            acc += s_Sigma[tid, j]
        s_Sw[tid] = acc / float32(N)
    cuda.syncthreads()

    # --- Frank-Wolfe iterations ---
    for k in range(MS_ITERS):
        # 1. port_var = w^T Sw
        _parallel_dot(s_w, s_Sw, s_red, tid, N)

        port_var = s_red[0]
        port_vol = math.sqrt(port_var)
        inv_vol = float32(1.0) / port_vol
        inv_vol3 = inv_vol * inv_vol * inv_vol

        # 2. port_ret = excess^T w
        _parallel_dot(s_excess, s_w, s_red, tid, N)
        port_ret = s_red[0]
        coeff = port_ret * inv_vol3

        # 3. grad[i] = excess[i] * inv_vol - coeff * Sw[i]
        if tid < N:
            s_grad[tid] = s_excess[tid] * inv_vol - coeff * s_Sw[tid]
        cuda.syncthreads()

        # 4. best = argmax(grad)
        _parallel_argmax(s_grad, s_red, s_idx, tid, N)
        if tid == 0:
            s_best[0] = s_idx[0]
        cuda.syncthreads()
        best = s_best[0]

        # 5. gamma = 2 / (k + 2)
        gamma = float32(2.0) / float32(k + 2)
        one_minus_gamma = float32(1.0) - gamma

        # 6. w <- (1 - gamma) w + gamma e_best
        if tid < N:
            s_w[tid] *= one_minus_gamma
        cuda.syncthreads()
        if tid == best:
            s_w[best] += gamma
        cuda.syncthreads()

        # 7. Sw <- (1 - gamma) Sw + gamma Sigma[:, best]
        if tid < N:
            s_Sw[tid] = one_minus_gamma * s_Sw[tid] + gamma * s_Sigma[tid, best]
        cuda.syncthreads()

    # --- Write result ---
    if tid < N:
        weights_out[bid, tid] = s_w[tid]


# =============================================================================
# Python-side launchers
# =============================================================================

def solve_min_variance_batch_gpu(Sigmas: np.ndarray) -> np.ndarray:
    """
    Solve a batch of min-variance problems on the GPU.

    Parameters
    ----------
    Sigmas : (B, N, N) array, float32 or float64
        Pre-regularized covariance matrices.  N must equal N_ASSETS (96).
        Adding jitter (Sigma += eps * I) is the caller's responsibility.

    Returns
    -------
    (B, N) array of weights, float32.
    """
    if Sigmas.ndim != 3:
        raise ValueError(f"expected 3D input, got shape {Sigmas.shape}")
    B, N, N2 = Sigmas.shape
    if N != N2:
        raise ValueError(f"Sigmas must be square, got ({N}, {N2})")
    if N != N_ASSETS:
        raise ValueError(f"kernel compiled for N={N_ASSETS}, got N={N}")

    Sigmas_f32 = np.ascontiguousarray(Sigmas, dtype=np.float32)

    d_Sigmas = cuda.to_device(Sigmas_f32)
    d_weights = cuda.device_array((B, N), dtype=np.float32)

    min_variance_batch_kernel[B, N](d_Sigmas, d_weights)
    cuda.synchronize()

    return d_weights.copy_to_host()


def solve_max_sharpe_batch_gpu(Sigmas: np.ndarray, excess: np.ndarray) -> np.ndarray:
    """
    Solve a batch of max-Sharpe problems on the GPU via Frank-Wolfe.

    Parameters
    ----------
    Sigmas : (B, N, N) array
        Pre-regularized covariance matrices.  N must equal N_ASSETS.
    excess : (B, N) array
        Excess returns (mu - rf) per asset, per batch.

    Returns
    -------
    (B, N) array of weights, float32.
    """
    if Sigmas.ndim != 3:
        raise ValueError(f"expected 3D Sigmas, got shape {Sigmas.shape}")
    B, N, N2 = Sigmas.shape
    if N != N2 or N != N_ASSETS:
        raise ValueError(f"Sigmas must be ({N_ASSETS}, {N_ASSETS}), got ({N}, {N2})")
    if excess.shape != (B, N):
        raise ValueError(f"excess must be (B={B}, N={N}), got {excess.shape}")

    Sigmas_f32 = np.ascontiguousarray(Sigmas, dtype=np.float32)
    excess_f32 = np.ascontiguousarray(excess, dtype=np.float32)

    d_Sigmas = cuda.to_device(Sigmas_f32)
    d_excess = cuda.to_device(excess_f32)
    d_weights = cuda.device_array((B, N), dtype=np.float32)

    max_sharpe_batch_kernel[B, N](d_Sigmas, d_excess, d_weights)
    cuda.synchronize()

    return d_weights.copy_to_host()


# =============================================================================
# Covariance estimators (device functions for use inside batched kernels)
# =============================================================================
#
# These compute a covariance matrix from a (T, N) window of log returns.
# Each thread block has N threads; each thread computes row `tid` of the
# output covariance matrix.  Inputs and outputs are in shared memory.
#
# Memory layout convention:
#   windows_gmem[b, t, i] = global-memory window for batch b
#   s_cov[i, j]           = shared-memory output covariance
#
# The window stays in global memory (not cached in shared) because at
# T=252, N=96, the window alone is 96 KB — too big to fit alongside
# everything else in shared memory.  Each thread reads window[:, tid]
# when computing means and window[:, tid]/window[:, j] when computing
# covariance.  Memory access is not perfectly coalesced, but for T=252
# this is acceptable (each thread reads T floats sequentially).
# =============================================================================

@cuda.jit(device=True, inline=True)
def _sample_cov_cooperative(window, s_mean, s_cov, tid, N, T):
    """
    Sample covariance (ddof=1), matches np.cov(window, rowvar=False).

    Cooperative: each thread computes mean[tid] and row tid of cov.

    window : (T, N) array (global or shared memory)
    s_mean : (N,) shared memory, overwritten with column means
    s_cov  : (N, N) shared memory, overwritten with covariance
    tid    : threadIdx.x
    N, T   : dimensions

    Complexity: O(T*N) per thread = O(T*N^2) total per block.
    For T=252, N=96: ~2.4M FLOPs per block (fast).
    """
    # Step 1: each thread computes one column mean
    if tid < N:
        acc = float32(0.0)
        for t in range(T):
            acc += window[t, tid]
        s_mean[tid] = acc / float32(T)
    cuda.syncthreads()

    # Step 2: each thread computes one row of cov
    # cov[i, j] = sum_t (X[t, i] - mean[i]) * (X[t, j] - mean[j]) / (T-1)
    # Thread `tid` computes cov[tid, :]
    if tid < N:
        mu_i = s_mean[tid]
        inv = float32(1.0) / float32(T - 1)
        for j in range(N):
            mu_j = s_mean[j]
            acc = float32(0.0)
            for t in range(T):
                acc += (window[t, tid] - mu_i) * (window[t, j] - mu_j)
            s_cov[tid, j] = acc * inv
    cuda.syncthreads()


@cuda.jit(device=True, inline=True)
def _ledoit_wolf_cooperative(window, s_mean, s_acc, s_cov, s_scratch, tid, N, T):
    """
    Ledoit-Wolf shrinkage covariance, matches src.estimators.ledoit_wolf
    and _ledoit_wolf_njit.

    We compute:
        S = X^T X / T (divisor T, not T-1)
        mu = trace(S) / N
        pi_hat = sum_ab [(X^2)^T (X^2) / T - S^2]_{ab}
        gamma_hat = ||S - mu*I||_F^2
        alpha = clip(pi_hat / (T * gamma_hat), 0, 1)
        out = (1 - alpha) * S + alpha * mu * I

    Parameters
    ----------
    window    : (T, N) input returns window
    s_mean    : shared (N,), column means (read during steps 4 and 5)
    s_acc     : shared (N,), scratch for per-row pi_hat and gamma_hat
                accumulators.  MUST NOT ALIAS s_mean (race condition).
    s_cov     : shared (N, N), output covariance
    s_scratch : shared (>=8,), scalar broadcast slots:
        s_scratch[0] = mu
        s_scratch[1] = pi_hat
        s_scratch[2] = gamma_hat
        s_scratch[3] = alpha
    """
    # Step 1: column means
    if tid < N:
        acc = float32(0.0)
        for t in range(T):
            acc += window[t, tid]
        s_mean[tid] = acc / float32(T)
    cuda.syncthreads()

    # Step 2: S = X^T X / T, each thread does one row
    if tid < N:
        mu_i = s_mean[tid]
        inv = float32(1.0) / float32(T)
        for j in range(N):
            mu_j = s_mean[j]
            acc = float32(0.0)
            for t in range(T):
                acc += (window[t, tid] - mu_i) * (window[t, j] - mu_j)
            s_cov[tid, j] = acc * inv
    cuda.syncthreads()

    # Step 3: mu = trace(S) / N  (thread 0 serial; N adds)
    if tid == 0:
        tr = float32(0.0)
        for i in range(N):
            tr += s_cov[i, i]
        s_scratch[0] = tr / float32(N)
    cuda.syncthreads()
    mu = s_scratch[0]

    # Step 4: pi_hat = sum_ab [pi_mat[a,b] - S[a,b]^2]
    # Each thread computes its row contribution to pi_hat.  We write
    # row_pi into s_acc (NOT s_mean) because other threads are still
    # reading s_mean[j] inside their inner loop — writing to s_mean
    # here would race.
    if tid < N:
        mu_i = s_mean[tid]
        row_pi = float32(0.0)
        inv_T = float32(1.0) / float32(T)
        for j in range(N):
            mu_j = s_mean[j]
            pm = float32(0.0)
            for t in range(T):
                di = window[t, tid] - mu_i
                dj = window[t, j] - mu_j
                pm += (di * di) * (dj * dj)
            pm *= inv_T
            Sij = s_cov[tid, j]
            row_pi += pm - Sij * Sij
        s_acc[tid] = row_pi
    cuda.syncthreads()
    if tid == 0:
        pi = float32(0.0)
        for i in range(N):
            pi += s_acc[i]
        s_scratch[1] = pi
    cuda.syncthreads()
    pi_hat = s_scratch[1]

    # Step 5: gamma_hat = ||S - mu*I||_F^2.  Row contribution again in s_acc.
    if tid < N:
        row_gamma = float32(0.0)
        for j in range(N):
            if tid == j:
                d = s_cov[tid, j] - mu
            else:
                d = s_cov[tid, j]
            row_gamma += d * d
        s_acc[tid] = row_gamma
    cuda.syncthreads()
    if tid == 0:
        g = float32(0.0)
        for i in range(N):
            g += s_acc[i]
        s_scratch[2] = g
    cuda.syncthreads()
    gamma_hat = s_scratch[2]

    # Step 6: alpha = clip(pi_hat / (T * gamma_hat), 0, 1)
    if tid == 0:
        if gamma_hat < float32(1e-12):
            a = float32(0.0)
        else:
            a = pi_hat / (float32(T) * gamma_hat)
            if a < float32(0.0):
                a = float32(0.0)
            if a > float32(1.0):
                a = float32(1.0)
        s_scratch[3] = a
    cuda.syncthreads()
    alpha = s_scratch[3]
    one_minus = float32(1.0) - alpha

    # Step 7: blend  out = (1 - alpha) * S + alpha * mu * I
    if tid < N:
        for j in range(N):
            if tid == j:
                s_cov[tid, j] = one_minus * s_cov[tid, j] + alpha * mu
            else:
                s_cov[tid, j] = one_minus * s_cov[tid, j]
    cuda.syncthreads()


@cuda.jit(device=True, inline=True)
def _ewma_cov_cooperative(window, s_mean, s_weights, s_cov, tid, N, T, lam):
    """
    EWMA covariance (RiskMetrics), matches src.estimators.ewma_cov.

    Weights: w_t = (1 - lam) * lam^(T-1-t), normalized so sum = 1.
    Weighted mean: mean[i] = sum_t w_t * X[t, i]
    Weighted cov:  cov[i, j] = sum_t w_t * (X[t, i] - mean[i]) * (X[t, j] - mean[j])

    s_weights must have size >= T.  Caller allocates (size 252 is enough
    for any lookback we support).
    """
    # Step 1: compute weights (thread 0 serial; T ops, T=252 max)
    if tid == 0:
        total = float32(0.0)
        p = float32(1.0)
        for t in range(T - 1, -1, -1):
            w = (float32(1.0) - lam) * p
            s_weights[t] = w
            total += w
            p *= lam
        for t in range(T):
            s_weights[t] = s_weights[t] / total
    cuda.syncthreads()

    # Step 2: weighted mean, each thread one column
    if tid < N:
        acc = float32(0.0)
        for t in range(T):
            acc += s_weights[t] * window[t, tid]
        s_mean[tid] = acc
    cuda.syncthreads()

    # Step 3: weighted covariance, each thread one row
    if tid < N:
        mu_i = s_mean[tid]
        for j in range(N):
            mu_j = s_mean[j]
            acc = float32(0.0)
            for t in range(T):
                di = window[t, tid] - mu_i
                dj = window[t, j] - mu_j
                acc += s_weights[t] * di * dj
            s_cov[tid, j] = acc
    cuda.syncthreads()


# =============================================================================
# Solver device function (wraps the min-variance inner loop)
# =============================================================================

@cuda.jit(device=True, inline=True)
def _min_variance_solve_block(
    s_Sigma, s_w, s_Sw, s_v, s_red, s_mini,
    tid, N, n_iter, power_iters,
):
    """
    Solve one min-variance problem given Sigma already in shared memory.

    Factored out of min_variance_batch_kernel so the full-backtest kernels
    (B.3) can reuse it without code duplication.

    On entry:
        s_Sigma[N, N] : covariance matrix, fully loaded
        tid           : threadIdx.x
    On exit:
        s_w[N] : optimal weights
        s_Sw, s_v, s_red, s_mini : clobbered
    """
    # Initialize w to uniform, v to unit-norm (for power iteration init)
    if tid < N:
        s_w[tid] = float32(1.0) / float32(N)
        s_v[tid] = float32(1.0) / math.sqrt(float32(N))
    cuda.syncthreads()

    # Power iteration for lambda_max
    _power_iteration_cooperative(s_Sigma, s_v, s_Sw, s_mini,
                                  tid, N, power_iters)
    lam_max = s_mini[0]
    two_step = float32(1.0) / lam_max

    # Projected gradient loop
    for _ in range(n_iter):
        _matvec_cooperative(s_Sigma, s_w, s_Sw, tid, N)
        cuda.syncthreads()

        if tid < N:
            s_v[tid] = s_w[tid] - two_step * s_Sw[tid]
        cuda.syncthreads()

        _project_simplex_bisection(s_v, s_red, s_mini, tid, N)

        if tid < N:
            s_w[tid] = s_v[tid]
        cuda.syncthreads()


# =============================================================================
# Full-backtest kernels (one per estimator)
# =============================================================================
#
# Each block runs one complete backtest (all rebalances) for one
# (config, bootstrap) pair, end-to-end on the GPU.  Zero CPU-GPU
# roundtrips during the rebalance loop.
#
# We have separate kernels per estimator rather than one kernel with a
# runtime branch, because Numba CUDA wouldn't reliably DCE the unused
# code paths — they'd still consume registers and shared memory.
# =============================================================================

@cuda.jit
def backtest_full_kernel_minvar_sample(
    returns_batch,        # (B, T, N) float32, bootstrapped log returns per problem
    weights_out,          # (B, n_rebalance, N) float32, one weight vec per rebalance
    lookback,             # int: lookback window size
    rebalance_every,      # int: rebalance period
):
    """Run B full min_variance + sample_cov backtests in parallel.

    Launch config: grid (B,), block (N_ASSETS,).
    """
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    B = returns_batch.shape[0]
    T = returns_batch.shape[1]
    if bid >= B:
        return

    s_Sigma = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)
    s_mean  = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_w     = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_Sw    = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_v     = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_red   = cuda.shared.array(shape=RED_SIZE, dtype=float32)
    s_mini  = cuda.shared.array(shape=8, dtype=float32)

    N = N_ASSETS

    t = lookback
    rb_idx = 0
    while t < T:
        # --- sample_cov(returns[t-lookback:t]) -> s_Sigma ---
        # Thread tid computes mean[tid] then row tid of the covariance.
        if tid < N:
            acc = float32(0.0)
            for tt in range(lookback):
                acc += returns_batch[bid, t - lookback + tt, tid]
            s_mean[tid] = acc / float32(lookback)
        cuda.syncthreads()

        if tid < N:
            mu_i = s_mean[tid]
            inv = float32(1.0) / float32(lookback - 1)
            for j in range(N):
                mu_j = s_mean[j]
                acc = float32(0.0)
                for tt in range(lookback):
                    ti = returns_batch[bid, t - lookback + tt, tid] - mu_i
                    tj = returns_batch[bid, t - lookback + tt, j] - mu_j
                    acc += ti * tj
                # Add PD-jitter on the diagonal in the same pass
                if tid == j:
                    s_Sigma[tid, j] = acc * inv + float32(1e-10)
                else:
                    s_Sigma[tid, j] = acc * inv
        cuda.syncthreads()

        # --- Solve min-variance ---
        _min_variance_solve_block(s_Sigma, s_w, s_Sw, s_v, s_red, s_mini,
                                    tid, N, MV_ITERS, POWER_ITERS)

        # --- Output ---
        if tid < N:
            weights_out[bid, rb_idx, tid] = s_w[tid]
        cuda.syncthreads()

        t += rebalance_every
        rb_idx += 1


@cuda.jit
def backtest_full_kernel_minvar_ledoit(
    returns_batch, weights_out, lookback, rebalance_every,
):
    """min_variance + Ledoit-Wolf.

    NOTE: the pi_hat / gamma_hat accumulators live in s_Sw (NOT s_mean),
    because the inner loops still need s_mean for mu_j; writing to s_mean
    while other warps are reading from it would race.  s_Sw is unused at
    that point in the iteration; the subsequent _min_variance_solve_block
    overwrites it cleanly.
    """
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    B = returns_batch.shape[0]
    T = returns_batch.shape[1]
    if bid >= B:
        return

    s_Sigma   = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)
    s_mean    = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_w       = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_Sw      = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_v       = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_red     = cuda.shared.array(shape=RED_SIZE, dtype=float32)
    s_mini    = cuda.shared.array(shape=8, dtype=float32)
    s_scratch = cuda.shared.array(shape=8, dtype=float32)

    N = N_ASSETS

    t = lookback
    rb_idx = 0
    while t < T:
        # Step 1: column means -> s_mean
        if tid < N:
            acc = float32(0.0)
            for tt in range(lookback):
                acc += returns_batch[bid, t - lookback + tt, tid]
            s_mean[tid] = acc / float32(lookback)
        cuda.syncthreads()

        # Step 2: S = X^T X / T (LW uses T, not T-1)
        if tid < N:
            mu_i = s_mean[tid]
            inv = float32(1.0) / float32(lookback)
            for j in range(N):
                mu_j = s_mean[j]
                acc = float32(0.0)
                for tt in range(lookback):
                    xi = returns_batch[bid, t - lookback + tt, tid] - mu_i
                    xj = returns_batch[bid, t - lookback + tt, j] - mu_j
                    acc += xi * xj
                s_Sigma[tid, j] = acc * inv
        cuda.syncthreads()

        # Step 3: mu = trace(S) / N
        if tid == 0:
            tr = float32(0.0)
            for i in range(N):
                tr += s_Sigma[i, i]
            s_scratch[0] = tr / float32(N)
        cuda.syncthreads()
        mu = s_scratch[0]

        # Step 4: pi_hat row accumulation in s_Sw (not s_mean!)
        if tid < N:
            mu_i = s_mean[tid]
            row_pi = float32(0.0)
            inv_T = float32(1.0) / float32(lookback)
            for j in range(N):
                mu_j = s_mean[j]
                pm = float32(0.0)
                for tt in range(lookback):
                    di = returns_batch[bid, t - lookback + tt, tid] - mu_i
                    dj = returns_batch[bid, t - lookback + tt, j] - mu_j
                    pm += (di * di) * (dj * dj)
                pm *= inv_T
                Sij = s_Sigma[tid, j]
                row_pi += pm - Sij * Sij
            s_Sw[tid] = row_pi
        cuda.syncthreads()
        if tid == 0:
            pi = float32(0.0)
            for i in range(N):
                pi += s_Sw[i]
            s_scratch[1] = pi
        cuda.syncthreads()
        pi_hat = s_scratch[1]

        # Step 5: gamma_hat row accumulation in s_Sw
        if tid < N:
            row_g = float32(0.0)
            for j in range(N):
                if tid == j:
                    d = s_Sigma[tid, j] - mu
                else:
                    d = s_Sigma[tid, j]
                row_g += d * d
            s_Sw[tid] = row_g
        cuda.syncthreads()
        if tid == 0:
            g = float32(0.0)
            for i in range(N):
                g += s_Sw[i]
            s_scratch[2] = g
        cuda.syncthreads()
        gamma_hat = s_scratch[2]

        # Step 6: alpha
        if tid == 0:
            if gamma_hat < float32(1e-12):
                a = float32(0.0)
            else:
                a = pi_hat / (float32(lookback) * gamma_hat)
                if a < float32(0.0):
                    a = float32(0.0)
                if a > float32(1.0):
                    a = float32(1.0)
            s_scratch[3] = a
        cuda.syncthreads()
        alpha = s_scratch[3]
        one_minus = float32(1.0) - alpha

        # Step 7: blend + add diagonal jitter in one pass
        if tid < N:
            for j in range(N):
                if tid == j:
                    s_Sigma[tid, j] = one_minus * s_Sigma[tid, j] + alpha * mu + float32(1e-10)
                else:
                    s_Sigma[tid, j] = one_minus * s_Sigma[tid, j]
        cuda.syncthreads()

        # Solve min-variance.  _min_variance_solve_block clobbers s_Sw
        # and s_v and s_w, which is fine now that we're done with the
        # Ledoit-Wolf stage.
        _min_variance_solve_block(s_Sigma, s_w, s_Sw, s_v, s_red, s_mini,
                                    tid, N, MV_ITERS, POWER_ITERS)

        if tid < N:
            weights_out[bid, rb_idx, tid] = s_w[tid]
        cuda.syncthreads()

        t += rebalance_every
        rb_idx += 1


@cuda.jit
def backtest_full_kernel_minvar_ewma(
    returns_batch, weights_out, lookback, rebalance_every, lam,
):
    """min_variance + EWMA."""
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    B = returns_batch.shape[0]
    T = returns_batch.shape[1]
    if bid >= B:
        return

    s_Sigma   = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)
    s_mean    = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    # s_weights sized for max lookback we support (252)
    s_weights = cuda.shared.array(shape=512, dtype=float32)
    s_w       = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_Sw      = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_v       = cuda.shared.array(shape=N_ASSETS, dtype=float32)
    s_red     = cuda.shared.array(shape=RED_SIZE, dtype=float32)
    s_mini    = cuda.shared.array(shape=8, dtype=float32)

    N = N_ASSETS

    t = lookback
    rb_idx = 0
    while t < T:
        # Step 1: EWMA weights (thread 0 serial, T ops max 252)
        if tid == 0:
            total = float32(0.0)
            p = float32(1.0)
            for tt in range(lookback - 1, -1, -1):
                w = (float32(1.0) - lam) * p
                s_weights[tt] = w
                total += w
                p *= lam
            for tt in range(lookback):
                s_weights[tt] = s_weights[tt] / total
        cuda.syncthreads()

        # Step 2: weighted means
        if tid < N:
            acc = float32(0.0)
            for tt in range(lookback):
                acc += s_weights[tt] * returns_batch[bid, t - lookback + tt, tid]
            s_mean[tid] = acc
        cuda.syncthreads()

        # Step 3: weighted covariance + diagonal jitter
        if tid < N:
            mu_i = s_mean[tid]
            for j in range(N):
                mu_j = s_mean[j]
                acc = float32(0.0)
                for tt in range(lookback):
                    di = returns_batch[bid, t - lookback + tt, tid] - mu_i
                    dj = returns_batch[bid, t - lookback + tt, j] - mu_j
                    acc += s_weights[tt] * di * dj
                if tid == j:
                    s_Sigma[tid, j] = acc + float32(1e-10)
                else:
                    s_Sigma[tid, j] = acc
        cuda.syncthreads()

        # Solve
        _min_variance_solve_block(s_Sigma, s_w, s_Sw, s_v, s_red, s_mini,
                                    tid, N, MV_ITERS, POWER_ITERS)

        if tid < N:
            weights_out[bid, rb_idx, tid] = s_w[tid]
        cuda.syncthreads()

        t += rebalance_every
        rb_idx += 1


# =============================================================================
# Python launcher for full-backtest kernels (dispatches by estimator)
# =============================================================================

def run_full_backtest_gpu(
    returns_batch: np.ndarray,
    lookback: int,
    rebalance_every: int,
    estimator: str = "sample",
    ewma_lam: float = 0.94,
) -> np.ndarray:
    """
    Run a batch of full min_variance backtests on GPU.

    Each block runs one complete backtest (all rebalances) for its assigned
    (config, bootstrap) slice.  Estimators are inlined in-kernel; no
    CPU-GPU roundtrips during the rebalance loop.

    Parameters
    ----------
    returns_batch : (B, T, N) array
        Bootstrapped log returns per problem.  All problems in the batch
        must share the same lookback, rebalance_every, and estimator.
    lookback : int
    rebalance_every : int
    estimator : {"sample", "ledoit_wolf", "ewma"}
    ewma_lam : float
        Smoothing factor, used only when estimator == "ewma".

    Returns
    -------
    weights : (B, n_rebalance, N) float32 array
    """
    if returns_batch.ndim != 3:
        raise ValueError(f"expected 3D returns, got shape {returns_batch.shape}")
    B, T, N = returns_batch.shape
    if N != N_ASSETS:
        raise ValueError(f"kernel compiled for N={N_ASSETS}, got N={N}")

    n_rebalance = 0
    t = lookback
    while t < T:
        n_rebalance += 1
        t += rebalance_every

    returns_f32 = np.ascontiguousarray(returns_batch, dtype=np.float32)
    d_returns = cuda.to_device(returns_f32)
    d_weights = cuda.device_array((B, n_rebalance, N), dtype=np.float32)

    if estimator == "sample":
        backtest_full_kernel_minvar_sample[B, N](
            d_returns, d_weights, lookback, rebalance_every
        )
    elif estimator == "ledoit_wolf":
        backtest_full_kernel_minvar_ledoit[B, N](
            d_returns, d_weights, lookback, rebalance_every
        )
    elif estimator == "ewma":
        backtest_full_kernel_minvar_ewma[B, N](
            d_returns, d_weights, lookback, rebalance_every, np.float32(ewma_lam)
        )
    else:
        raise ValueError(f"unknown estimator '{estimator}'")
    cuda.synchronize()

    return d_weights.copy_to_host()