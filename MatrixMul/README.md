# Matrix Multiplication Kernels

## matrix_mul_v1.cu

First tiled shared memory version.

Uses 4 by 4 blocks and 4 by 4 tiles.

Matrix sizes are M 100, N 5000, P 200.

Includes CPU reference computation and exact result check.

Good for learning correctness and basic tiling.

## matrix_mul_v2.cu

Larger benchmark version based on the same kernel layout.

Uses 16 by 16 blocks and 16 by 16 tiles.

Matrix sizes are M 4096, N 4096, P 4096.

CPU reference computation was removed.

Good for profiling the kernel at a realistic size.

Current main bottleneck is memory access pattern.

## matrix_mul_v3.cu

Coalesced row major access version.

Uses 16 by 16 blocks and 16 by 16 tiles.

Matrix sizes are M 4096, N 4096, P 4096.

Maps threadIdx x to matrix columns and threadIdx y to matrix rows.

Good for comparing against v2 after fixing global memory coalescing.

## matrix_mul_v4.cu

Register tiled outer product version.

Uses 16 by 16 thread blocks.

Each thread computes a 4 by 4 output tile.

Each block computes a 64 by 64 output tile.

Kernel name is MatrixMul_outter_product.

Good for reducing shared memory pressure and increasing math per thread.

## matrix_mul_v5.cu

Wider register tiled version based on v4.

Uses shared K tile size 32.

Uses 32 by 8 thread blocks.

Each thread computes a 4 by 8 output tile.

Each block computes a 128 by 64 output tile.

Keeps scalar memory access because the vector access experiment was slower.

Removes the arbitrary grid size cap.

Prints kernel GFLOP per second.

Good for improving shared memory reuse compared with v4.

## matrix_mul_v6.cu

Tensor core WMMA version.

Uses TF32 tensor core fragments.

Kernel name is MatrixMul_tensor_cores.

Each warp computes a 16 by 16 output tile.

Each block computes a 64 by 64 output tile.

Good for learning tensor core WMMA basics.

This simple tensor core version is slower than v5 because it does not yet reuse A and B tiles as effectively.
