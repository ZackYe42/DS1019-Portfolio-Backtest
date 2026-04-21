"""Diagnose the GPU max_sharpe kernel.

Strategy: Compare iteration-by-iteration state between CPU and GPU by
running ONE problem through each and dumping the 'best' index chosen
at each Frank-Wolfe iteration.  If they diverge early, the bug is
obvious at iteration 0 or 1.
"""
import numpy as np


def main():
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

    import math
    from numba import cuda, float32, int32
    from src.versions.v5_gpu_kernels import (
        _parallel_dot, _parallel_argmax, N_ASSETS, RED_SIZE,
    )

    N_DEBUG_ITERS = 20

    @cuda.jit
    def max_sharpe_debug_kernel(Sigmas, excess_all, weights_out, best_log,
                                 port_vol_log, port_ret_log, grad_max_log):
        bid = cuda.blockIdx.x
        tid = cuda.threadIdx.x

        s_Sigma  = cuda.shared.array(shape=(N_ASSETS, N_ASSETS), dtype=float32)
        s_excess = cuda.shared.array(shape=N_ASSETS, dtype=float32)
        s_w      = cuda.shared.array(shape=N_ASSETS, dtype=float32)
        s_Sw     = cuda.shared.array(shape=N_ASSETS, dtype=float32)
        s_grad   = cuda.shared.array(shape=N_ASSETS, dtype=float32)
        s_red    = cuda.shared.array(shape=RED_SIZE, dtype=float32)
        s_idx    = cuda.shared.array(shape=RED_SIZE, dtype=int32)
        s_best   = cuda.shared.array(shape=1, dtype=int32)

        N = N_ASSETS

        if tid < N:
            for j in range(N):
                s_Sigma[tid, j] = Sigmas[bid, tid, j]
            s_excess[tid] = excess_all[bid, tid]
            s_w[tid] = float32(1.0) / float32(N)
        cuda.syncthreads()

        if tid < N:
            acc = float32(0.0)
            for j in range(N):
                acc += s_Sigma[tid, j]
            s_Sw[tid] = acc / float32(N)
        cuda.syncthreads()

        for k in range(N_DEBUG_ITERS):
            _parallel_dot(s_w, s_Sw, s_red, tid, N)
            port_var = s_red[0]
            port_vol = math.sqrt(port_var)
            inv_vol = float32(1.0) / port_vol
            inv_vol3 = inv_vol * inv_vol * inv_vol

            _parallel_dot(s_excess, s_w, s_red, tid, N)
            port_ret = s_red[0]
            coeff = port_ret * inv_vol3

            if tid < N:
                s_grad[tid] = s_excess[tid] * inv_vol - coeff * s_Sw[tid]
            cuda.syncthreads()

            _parallel_argmax(s_grad, s_red, s_idx, tid, N)
            if tid == 0:
                s_best[0] = s_idx[0]
                best_log[bid, k] = s_idx[0]
                port_vol_log[bid, k] = port_vol
                port_ret_log[bid, k] = port_ret
                grad_max_log[bid, k] = s_red[0]
            cuda.syncthreads()
            best = s_best[0]

            gamma = float32(2.0) / float32(k + 2)
            one_minus_gamma = float32(1.0) - gamma

            if tid < N:
                s_w[tid] *= one_minus_gamma
            cuda.syncthreads()
            if tid == best:
                s_w[best] += gamma
            cuda.syncthreads()

            if tid < N:
                s_Sw[tid] = one_minus_gamma * s_Sw[tid] + gamma * s_Sigma[tid, best]
            cuda.syncthreads()

        if tid < N:
            weights_out[bid, tid] = s_w[tid]

    # --- CPU reference ---
    np.random.seed(42)
    N = 96
    A = np.random.randn(252, N) * 0.01
    Sigma = np.cov(A, rowvar=False) + 1e-10 * np.eye(N)
    excess = A.mean(axis=0)

    print("=" * 70)
    print("CPU reference (first 20 iterations)")
    print("=" * 70)
    w_cpu = np.full(N, 1.0 / N)
    Sw_cpu = Sigma @ w_cpu
    cpu_bests = []
    cpu_vols = []
    cpu_rets = []
    cpu_grads = []
    for k in range(N_DEBUG_ITERS):
        port_var = w_cpu @ Sw_cpu
        port_vol = np.sqrt(port_var)
        inv_vol = 1.0 / port_vol
        inv_vol3 = inv_vol ** 3
        port_ret = excess @ w_cpu
        coeff = port_ret * inv_vol3
        grad = excess * inv_vol - coeff * Sw_cpu
        best = int(np.argmax(grad))
        cpu_bests.append(best)
        cpu_vols.append(port_vol)
        cpu_rets.append(port_ret)
        cpu_grads.append(float(grad.max()))
        gamma = 2.0 / (k + 2.0)
        w_cpu *= (1.0 - gamma)
        w_cpu[best] += gamma
        Sw_cpu = (1.0 - gamma) * Sw_cpu + gamma * Sigma[:, best]

    for k in range(N_DEBUG_ITERS):
        print(f"  k={k:2d}: best={cpu_bests[k]:3d}  port_vol={cpu_vols[k]:.6e}  "
              f"port_ret={cpu_rets[k]:+.6e}  grad_max={cpu_grads[k]:+.6e}")

    # --- GPU debug ---
    print("\n" + "=" * 70)
    print("GPU debug kernel (first 20 iterations)")
    print("=" * 70)

    B = 1
    Sigmas_gpu = np.ascontiguousarray(Sigma[None].astype(np.float32))
    excess_gpu = np.ascontiguousarray(excess[None].astype(np.float32))

    d_Sigmas = cuda.to_device(Sigmas_gpu)
    d_excess = cuda.to_device(excess_gpu)
    d_weights = cuda.device_array((B, N), dtype=np.float32)
    d_best_log = cuda.device_array((B, N_DEBUG_ITERS), dtype=np.int32)
    d_vol_log = cuda.device_array((B, N_DEBUG_ITERS), dtype=np.float32)
    d_ret_log = cuda.device_array((B, N_DEBUG_ITERS), dtype=np.float32)
    d_grad_log = cuda.device_array((B, N_DEBUG_ITERS), dtype=np.float32)

    max_sharpe_debug_kernel[B, N](d_Sigmas, d_excess, d_weights,
                                    d_best_log, d_vol_log, d_ret_log, d_grad_log)
    cuda.synchronize()

    gpu_bests = d_best_log.copy_to_host()[0]
    gpu_vols = d_vol_log.copy_to_host()[0]
    gpu_rets = d_ret_log.copy_to_host()[0]
    gpu_grads = d_grad_log.copy_to_host()[0]

    print(f"{'k':>3s} {'CPU best':>10s} {'GPU best':>10s} "
          f"{'CPU vol':>12s} {'GPU vol':>12s} "
          f"{'CPU ret':>14s} {'GPU ret':>14s} {'match':>8s}")
    for k in range(N_DEBUG_ITERS):
        match = "OK" if gpu_bests[k] == cpu_bests[k] else "DIVERGE"
        print(f"{k:>3d} {cpu_bests[k]:>10d} {gpu_bests[k]:>10d} "
              f"{cpu_vols[k]:>12.6e} {gpu_vols[k]:>12.6e} "
              f"{cpu_rets[k]:>+14.6e} {gpu_rets[k]:>+14.6e} {match:>8s}")


if __name__ == "__main__":
    main()