"""
CUDA sanity check.

Before building v5, verify that Numba CUDA can compile and launch
kernels, transfer data correctly, and deliver GPU-like speedups on
a toy problem.  If any of these fail, v5 won't work — better to
find out now on 20 lines of code than on 200.
"""

import time
import numpy as np
from numba import cuda


def main():
    # --- Basic environment check ---
    print("=" * 70)
    print("CUDA environment")
    print("=" * 70)
    print(f"cuda.is_available(): {cuda.is_available()}")
    if not cuda.is_available():
        print("ERROR: CUDA not available.  Stop.")
        return

    device = cuda.get_current_device()
    print(f"Device name:         {device.name.decode() if isinstance(device.name, bytes) else device.name}")
    print(f"Compute capability:  {device.compute_capability}")
    print(f"Max threads/block:   {device.MAX_THREADS_PER_BLOCK}")
    print(f"Max block dim:       ({device.MAX_BLOCK_DIM_X}, {device.MAX_BLOCK_DIM_Y}, {device.MAX_BLOCK_DIM_Z})")
    print(f"Multiprocessors:     {device.MULTIPROCESSOR_COUNT}")

    # --- Test 1: Simple kernel (vector add) ---
    print("\n" + "=" * 70)
    print("Test 1: Vector add kernel")
    print("=" * 70)

    @cuda.jit
    def vector_add_kernel(a, b, out):
        i = cuda.grid(1)  # global thread index
        if i < a.size:
            out[i] = a[i] + b[i]

    N = 1_000_000
    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32) * 2
    expected = a + b

    # Allocate on device
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.device_array_like(a)

    # Launch: 256 threads per block, enough blocks to cover N
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block

    # Warmup (first launch is slow due to JIT)
    vector_add_kernel[blocks, threads_per_block](d_a, d_b, d_out)
    cuda.synchronize()

    # Time it
    t0 = time.perf_counter()
    for _ in range(100):
        vector_add_kernel[blocks, threads_per_block](d_a, d_b, d_out)
    cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / 100

    # Verify correctness
    result = d_out.copy_to_host()
    max_diff = np.abs(result - expected).max()
    print(f"  GPU time:    {gpu_time*1e6:.1f} us")
    print(f"  max |diff|:  {max_diff:.2e}  {'PASS' if max_diff < 1e-3 else 'FAIL'}")

    # Compare to NumPy
    t0 = time.perf_counter()
    for _ in range(100):
        _ = a + b
    cpu_time = (time.perf_counter() - t0) / 100
    print(f"  CPU time:    {cpu_time*1e6:.1f} us")
    print(f"  GPU speedup: {cpu_time / gpu_time:.1f}x")

    # --- Test 2: Matrix multiplication kernel (batched) ---
    print("\n" + "=" * 70)
    print("Test 2: Batched matmul (the pattern we'll use in v5)")
    print("=" * 70)

    @cuda.jit
    def batched_matvec_kernel(mats, vecs, outs):
        """Compute outs[b, i] = sum_j mats[b, i, j] * vecs[b, j].
        
        One thread per (batch, row)."""
        b, i = cuda.grid(2)
        if b < mats.shape[0] and i < mats.shape[1]:
            acc = 0.0
            N = mats.shape[2]
            for j in range(N):
                acc += mats[b, i, j] * vecs[b, j]
            outs[b, i] = acc

    B = 1000  # batch size (like our bootstrap samples)
    N = 96    # asset count (like our portfolio)

    rng = np.random.default_rng(42)
    mats = rng.standard_normal((B, N, N)).astype(np.float32)
    vecs = rng.standard_normal((B, N)).astype(np.float32)
    expected = np.einsum("bij,bj->bi", mats, vecs)

    d_mats = cuda.to_device(mats)
    d_vecs = cuda.to_device(vecs)
    d_outs = cuda.device_array((B, N), dtype=np.float32)

    # 2D grid: batch x row
    threads_per_block = (1, 96)  # threads in x=batch, y=row
    blocks = ((B + threads_per_block[0] - 1) // threads_per_block[0],
              (N + threads_per_block[1] - 1) // threads_per_block[1])

    # Warmup
    batched_matvec_kernel[blocks, threads_per_block](d_mats, d_vecs, d_outs)
    cuda.synchronize()

    # Time
    t0 = time.perf_counter()
    for _ in range(50):
        batched_matvec_kernel[blocks, threads_per_block](d_mats, d_vecs, d_outs)
    cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / 50

    result = d_outs.copy_to_host()
    max_diff = np.abs(result - expected).max()
    rel_diff = max_diff / np.abs(expected).max()

    print(f"  Batch size:  {B} matrices of shape ({N}, {N})")
    print(f"  GPU time:    {gpu_time*1e3:.2f} ms")
    print(f"  max |diff|:  {max_diff:.2e}  (relative: {rel_diff:.2e})  "
          f"{'PASS' if rel_diff < 1e-3 else 'FAIL'}")

    # Compare to NumPy loop (what a single-threaded CPU would do)
    t0 = time.perf_counter()
    for _ in range(5):
        _ = np.einsum("bij,bj->bi", mats, vecs)
    cpu_time = (time.perf_counter() - t0) / 5
    print(f"  CPU time:    {cpu_time*1e3:.2f} ms  (using np.einsum — already BLAS)")
    print(f"  GPU speedup: {cpu_time / gpu_time:.1f}x")

    # --- Test 3: The scale we'll actually need ---
    print("\n" + "=" * 70)
    print("Test 3: Memory budget for v5")
    print("=" * 70)

    # For v5: per batch, we store
    #   - returns_window: (batch, lookback=252, N=96) float32 = 96 KB per sample
    #   - cov:            (batch, N, N)          float32 = 36 KB per sample
    #   - weights:        (batch, N)            float32 =  0.4 KB per sample
    #   - Sw scratch:     (batch, N)            float32 =  0.4 KB per sample
    # Total: ~133 KB per (config, bootstrap, rebalance) triple.

    # Your GPU has 16 GB VRAM.  How many samples can we batch at once?
    per_sample_bytes = 252 * 96 * 4 + 96 * 96 * 4 + 96 * 4 + 96 * 4
    vram_gb = 12  # use ~75% of 16 GB, leaving room for Python and overhead
    max_batch = int(vram_gb * 1e9 / per_sample_bytes)
    print(f"  Per-sample memory: {per_sample_bytes / 1024:.1f} KB")
    print(f"  Safe batch size:   ~{max_batch} samples per GPU launch")
    print(f"  Full workload:     54 configs x 1000 bootstrap x 130 rebalances = 7.0M")
    print(f"  Launches needed:   ~{7_000_000 // max_batch + 1}")
    print(f"  (If this is >100, we'll batch per-rebalance instead of per-backtest)")

    print("\nAll CUDA sanity checks passed.")


if __name__ == "__main__":
    main()