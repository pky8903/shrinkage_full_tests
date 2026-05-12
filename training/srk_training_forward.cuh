#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  srk_training_forward.cuh
//  Batched forward operator (training-style):
//    R_b = ( conv(E_b, Gx) · dI_b/dx + conv(E_b, Gy) · dI_b/dy )   for b = 0..B-1
//
//  Math is identical to srk_forward.cuh (cufftDX OaS) but the implementation
//  uses cuFFT *batched* 2D R2C / C2R with FFT size = N_large (no tiling).
//  Suitable for moderate image sizes (≤ 512²) with a large batch B.
//
//  Memory layout — ArrayFire column-major (rows innermost):
//    E, I : [B × N × N]  with offset = r + c*N + b*N*N    (row r, col c, batch b)
//    R    : [B × W × W]  with offset = or + oc*W + b*W*W  (W = N − 2·erosion)
//    Gx_freq, Gy_freq : column-major complex [(N/2+1) × N]
//      (precomputed by srkPrecomputeKernelFreq with M = N)
//
//  Inside cuFFT, the same bytes are seen as a row-major (W=N, H=N) buffer; the
//  2D FFT is invariant to that relabelling.
// ─────────────────────────────────────────────────────────────────────────────

#include <cufft.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// ── Device helpers ───────────────────────────────────────────────────────────

// 2D fftshift per batch element.  Column-major in / out: data[r + c*N + b*N*N].
__global__ void srk_train_fftshift_2d_batched(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (r >= N || c >= N) return;
    int sr = (r + N / 2) % N;
    int sc = (c + N / 2) % N;
    out[sr + sc * N + b * N * N] = in[r + c * N + b * N * N];
}

// U_freq[b, k_r, c] = scale · E_freq[b, k_r, c] · G_freq[k_r, c]
// Complex column-major: freq buffer shape (half, N) with k_r innermost.
//   offset = k_r + c * half + b * half * N
__global__ void srk_train_multiply_broadcast(
    cufftComplex* __restrict__ U_freq,
    const cufftComplex* __restrict__ E_freq,
    const cufftComplex* __restrict__ G_freq,
    int N, int half, float scale)
{
    int k_r = blockIdx.x * blockDim.x + threadIdx.x;
    int c   = blockIdx.y;
    int b   = blockIdx.z;
    if (k_r >= half) return;
    int idx_b = k_r + c * half + b * half * N;
    int idx_g = k_r + c * half;
    cufftComplex e = E_freq[idx_b];
    cufftComplex g = G_freq[idx_g];
    U_freq[idx_b].x = scale * (e.x * g.x - e.y * g.y);
    U_freq[idx_b].y = scale * (e.x * g.y + e.y * g.x);
}

// Central-difference gradients (per batch), column-major [r + c*N + b*N*N].
// dI/dx = derivative along col axis; dI/dy = derivative along row axis.
__global__ void srk_train_grad_x_batched(const float* I, float* dIdx, int N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (r >= N || c >= N) return;
    int off = b * N * N;
    float left  = (c > 0)     ? I[off + r + (c - 1) * N] : 0.f;
    float right = (c < N - 1) ? I[off + r + (c + 1) * N] : 0.f;
    dIdx[off + r + c * N] = (right - left) * 0.5f;
}

__global__ void srk_train_grad_y_batched(const float* I, float* dIdy, int N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (r >= N || c >= N) return;
    int off = b * N * N;
    float up   = (r > 0)     ? I[off + (r - 1) + c * N] : 0.f;
    float down = (r < N - 1) ? I[off + (r + 1) + c * N] : 0.f;
    dIdy[off + r + c * N] = (down - up) * 0.5f;
}

// Combine using ux_full / uy_full of shape [B × N × N] (column-major).
//
// With column-major fftshift(E) as the FFT input, the IFFT output at buffer
// position p holds the linear conv at natural position (p + N/2) mod N.  We
// therefore read ux_full / uy_full at fftshifted indices, while dIdx / dIdy
// (computed directly from I) stay at natural indices.  Same logic as the
// column-major srk_fft_rows_c2r_valid in srk_forward.cuh.
__global__ void srk_train_combine_full(
    const float* ux_full, const float* uy_full,   // [B × N × N] (column-major)
    const float* dIdx,    const float* dIdy,      // [B × N × N] (column-major)
    float* R,                                     // [B × W × W] (column-major)
    int W, int N, int erosion)
{
    int or_ = blockIdx.x * blockDim.x + threadIdx.x;
    int oc  = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;
    if (or_ >= W || oc >= W) return;
    int gr = or_ + erosion, gc = oc + erosion;     // natural E coord
    int sr = (gr + N / 2) % N;
    int sc = (gc + N / 2) % N;                     // fftshifted buffer coord
    int gi_grad = gr + gc * N + b * N * N;
    int gi_conv = sr + sc * N + b * N * N;
    int oi      = or_ + oc * W + b * W * W;
    R[oi] = ux_full[gi_conv] * dIdx[gi_grad]
          + uy_full[gi_conv] * dIdy[gi_grad];
}

// ── Workspace ────────────────────────────────────────────────────────────────
struct SrkTrainWorkspace {
    int N = 0, B_max = 0, half = 0;
    cufftHandle plan_r2c = 0;
    cufftHandle plan_c2r = 0;

    float*        d_E_sh    = nullptr;   ///< [B × N × N]
    cufftComplex* d_E_freq  = nullptr;   ///< [B × N × half]
    cufftComplex* d_U_freq  = nullptr;   ///< scratch
    float*        d_ux_full = nullptr;   ///< [B × N × N]
    float*        d_uy_full = nullptr;   ///< [B × N × N]
    float*        d_dIdx    = nullptr;   ///< [B × N × N]
    float*        d_dIdy    = nullptr;   ///< [B × N × N]

    void allocate(int N_, int B_, cudaStream_t stream = nullptr) {
        N = N_; B_max = B_; half = N / 2 + 1;
        size_t real_bytes = sizeof(float)        * (size_t)B_max * N * N;
        size_t cplx_bytes = sizeof(cufftComplex) * (size_t)B_max * N * half;
        cudaMallocAsync(&d_E_sh,    real_bytes, stream);
        cudaMallocAsync(&d_E_freq,  cplx_bytes, stream);
        cudaMallocAsync(&d_U_freq,  cplx_bytes, stream);
        cudaMallocAsync(&d_ux_full, real_bytes, stream);
        cudaMallocAsync(&d_uy_full, real_bytes, stream);
        cudaMallocAsync(&d_dIdx,    real_bytes, stream);
        cudaMallocAsync(&d_dIdy,    real_bytes, stream);

        // cuFFT sees our column-major data as a transposed row-major (N, N).
        // The 2D FFT is identical numerically.  For square N we don't need
        // any stride trickery; just declare n = {N, N}.
        int n[2] = {N, N};
        int idist_r = N * N;
        int odist_c = N * half;
        cufftPlanMany(&plan_r2c, /*rank=*/2, n,
                      /*inembed=*/nullptr, /*istride=*/1, /*idist=*/idist_r,
                      /*onembed=*/nullptr, /*ostride=*/1, /*odist=*/odist_c,
                      CUFFT_R2C, B_max);
        cufftPlanMany(&plan_c2r, 2, n,
                      nullptr, 1, odist_c,
                      nullptr, 1, idist_r,
                      CUFFT_C2R, B_max);
        cufftSetStream(plan_r2c, stream);
        cufftSetStream(plan_c2r, stream);
    }

    void release(cudaStream_t stream = nullptr) {
        if (plan_r2c) { cufftDestroy(plan_r2c); plan_r2c = 0; }
        if (plan_c2r) { cufftDestroy(plan_c2r); plan_c2r = 0; }
        if (d_E_sh)    cudaFreeAsync(d_E_sh,    stream);
        if (d_E_freq)  cudaFreeAsync(d_E_freq,  stream);
        if (d_U_freq)  cudaFreeAsync(d_U_freq,  stream);
        if (d_ux_full) cudaFreeAsync(d_ux_full, stream);
        if (d_uy_full) cudaFreeAsync(d_uy_full, stream);
        if (d_dIdx)    cudaFreeAsync(d_dIdx,    stream);
        if (d_dIdy)    cudaFreeAsync(d_dIdy,    stream);
    }
};

// ── Entry point ──────────────────────────────────────────────────────────────
//   B must equal ws.B_max (cuFFT plan is built for fixed batch).
inline void srkTrainingForward(
    const float* d_E_batch,             // [B × N × N] (column-major)
    const float* d_I_batch,             // [B × N × N] (column-major)
    const cufftComplex* d_Gx_freq,      // [(N/2+1) × N] column-major
    const cufftComplex* d_Gy_freq,
    int N, int W, int erosion, float dx,
    int B,
    cudaStream_t stream,
    float* d_R_batch,                   // [B × W × W] (column-major)
    SrkTrainWorkspace& ws)
{
    assert(N == ws.N && B == ws.B_max);
    const int   half  = ws.half;
    const float scale = dx * dx / (float(N) * float(N));

    cufftSetStream(ws.plan_r2c, stream);
    cufftSetStream(ws.plan_c2r, stream);

    // 1) fftshift E → d_E_sh
    {
        dim3 blk(16, 16, 1);
        dim3 grd((N + 15) / 16, (N + 15) / 16, B);
        srk_train_fftshift_2d_batched<<<grd, blk, 0, stream>>>(
            d_E_batch, ws.d_E_sh, N);
    }

    // 2) Batched 2D R2C FFT
    cufftExecR2C(ws.plan_r2c, ws.d_E_sh, ws.d_E_freq);

    // 3) Pointwise multiply with Gx, IFFT → ux_full
    {
        dim3 blk(64, 1, 1);
        dim3 grd((half + 63) / 64, N, B);
        srk_train_multiply_broadcast<<<grd, blk, 0, stream>>>(
            ws.d_U_freq, ws.d_E_freq, d_Gx_freq, N, half, scale);
    }
    cufftExecC2R(ws.plan_c2r, ws.d_U_freq, ws.d_ux_full);

    // 4) Pointwise multiply with Gy, IFFT → uy_full
    {
        dim3 blk(64, 1, 1);
        dim3 grd((half + 63) / 64, N, B);
        srk_train_multiply_broadcast<<<grd, blk, 0, stream>>>(
            ws.d_U_freq, ws.d_E_freq, d_Gy_freq, N, half, scale);
    }
    cufftExecC2R(ws.plan_c2r, ws.d_U_freq, ws.d_uy_full);

    // 5) Gradients of I
    {
        dim3 blk(16, 16, 1);
        dim3 grd((N + 15) / 16, (N + 15) / 16, B);
        srk_train_grad_x_batched<<<grd, blk, 0, stream>>>(d_I_batch, ws.d_dIdx, N);
        srk_train_grad_y_batched<<<grd, blk, 0, stream>>>(d_I_batch, ws.d_dIdy, N);
    }

    // 6) Combine
    {
        dim3 blk(16, 16, 1);
        dim3 grd((W + 15) / 16, (W + 15) / 16, B);
        srk_train_combine_full<<<grd, blk, 0, stream>>>(
            ws.d_ux_full, ws.d_uy_full,
            ws.d_dIdx,    ws.d_dIdy,
            d_R_batch,
            W, N, erosion);
    }
    cudaStreamSynchronize(stream);
}
