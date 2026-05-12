#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  srk_cudnn_forward.cuh
//  Spatial-domain convolution version of the SRK forward, via cuDNN.
//
//    R_b = ( conv(E_b, Gx) · dI_b/dx + conv(E_b, Gy) · dI_b/dy )
//
//  Conventions in *our* API: ArrayFire-style column-major (rows innermost).
//  cuDNN's NCHW expects row-major (cols innermost), so we transpose buffers
//  on the way in and out — that fully isolates the column-major convention
//  to our internal layer.
//
//  Implementation:
//    • cuDNN convolution in CUDNN_CONVOLUTION mode (true convolution; kernel
//      flipped internally).
//    • Pad = 0, stride = 1 → output W = N − KW + 1 = N − 2·erosion (when KW is
//      odd and erosion = (KW−1)/2).
//    • Convolution alpha = dx²  (continuous-conv discretization factor).
// ─────────────────────────────────────────────────────────────────────────────

#include "srk_training_forward.cuh"   // shares column-major gradient kernels

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDNN_CHECK(call)                                                      \
    do {                                                                       \
        cudnnStatus_t _s = (call);                                             \
        if (_s != CUDNN_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuDNN error %d at %s:%d\n",                  \
                         (int)_s, __FILE__, __LINE__);                         \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// ── Column-major ↔ row-major transposes (per batch) ─────────────────────────
// src is column-major [B × H × W] with src[r + c*H + b*H*W];
// dst is row-major   [B × H × W] with dst[r*W + c + b*H*W].
__global__ void srk_cm_to_rm_batched(
    const float* __restrict__ src, float* __restrict__ dst,
    int H, int W)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (r >= H || c >= W) return;
    dst[r * W + c + b * H * W] = src[r + c * H + b * H * W];
}

__global__ void srk_rm_to_cm_batched(
    const float* __restrict__ src, float* __restrict__ dst,
    int H, int W)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (r >= H || c >= W) return;
    dst[r + c * H + b * H * W] = src[r * W + c + b * H * W];
}

// Same transpose for a single 2D kernel (no batch).
__global__ void srk_cm_to_rm_2d(
    const float* __restrict__ src, float* __restrict__ dst,
    int H, int W)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= H || c >= W) return;
    dst[r * W + c] = src[r + c * H];
}

// Combine with ux / uy [B × W × W] column-major, dIdx / dIdy [B × N × N] cm.
__global__ void srk_cudnn_combine(
    const float* ux,   const float* uy,        // [B × W × W] (column-major)
    const float* dIdx, const float* dIdy,      // [B × N × N] (column-major)
    float* R,                                  // [B × W × W] (column-major)
    int W, int N, int erosion)
{
    int or_ = blockIdx.x * blockDim.x + threadIdx.x;
    int oc  = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;
    if (or_ >= W || oc >= W) return;
    int gr = or_ + erosion, gc = oc + erosion;
    int gi = gr + gc * N + b * N * N;
    int wi = or_ + oc * W + b * W * W;
    R[wi] = ux[wi] * dIdx[gi] + uy[wi] * dIdy[gi];
}

struct SrkCudnnWorkspace {
    cudnnHandle_t                handle    = nullptr;
    cudnnTensorDescriptor_t      in_desc   = nullptr;
    cudnnTensorDescriptor_t      out_desc  = nullptr;
    cudnnFilterDescriptor_t      filt_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t    algo;
    void*                        d_ws_buf  = nullptr;
    size_t                       ws_bytes  = 0;

    float* d_E_rm    = nullptr;   ///< [B × N × N] row-major (transposed input)
    float* d_Gx_rm   = nullptr;   ///< [KH × KW]   row-major filter
    float* d_Gy_rm   = nullptr;   ///< [KH × KW]   row-major filter
    float* d_ux_rm   = nullptr;   ///< [B × W × W] row-major (cuDNN output)
    float* d_uy_rm   = nullptr;   ///< [B × W × W] row-major (cuDNN output)
    float* d_ux_cm   = nullptr;   ///< [B × W × W] column-major (transposed back)
    float* d_uy_cm   = nullptr;
    float* d_dIdx    = nullptr;   ///< [B × N × N] column-major
    float* d_dIdy    = nullptr;   ///< [B × N × N] column-major

    int B = 0, N = 0, W = 0, KW = 0, KH = 0;

    void allocate(int B_, int N_, int KW_, int KH_, int W_, cudaStream_t stream = nullptr) {
        B = B_; N = N_; KW = KW_; KH = KH_; W = W_;

        CUDNN_CHECK(cudnnCreate(&handle));
        cudnnSetStream(handle, stream);

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, 1, N, N));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, 1, W, W));

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filt_desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, KH, KW));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc,
            /*pad_h=*/0, /*pad_w=*/0,
            /*u=*/1,     /*v=*/1,
            /*dilation_h=*/1, /*dilation_w=*/1,
            CUDNN_CONVOLUTION,             // true convolution (flips kernel)
            CUDNN_DATA_FLOAT));

        // Sanity: cuDNN-computed output dims must equal (B, 1, W, W).
        int nOut, cOut, hOut, wOut;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
            conv_desc, in_desc, filt_desc, &nOut, &cOut, &hOut, &wOut));
        if (nOut != B || cOut != 1 || hOut != W || wOut != W) {
            std::fprintf(stderr,
                "[cuDNN] output dim mismatch: got (%d,%d,%d,%d), expected (%d,1,%d,%d)\n",
                nOut, cOut, hOut, wOut, B, W, W);
            std::abort();
        }

        // Pick an algorithm via heuristic.
        int returned = 0;
        cudnnConvolutionFwdAlgoPerf_t perf[8];
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle, in_desc, filt_desc, conv_desc, out_desc,
            /*requestedAlgoCount=*/8, &returned, perf));
        algo = perf[0].algo;

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_bytes));
        if (ws_bytes > 0) cudaMalloc(&d_ws_buf, ws_bytes);

        cudaMallocAsync(&d_E_rm,  sizeof(float) * (size_t)B * N * N, stream);
        cudaMallocAsync(&d_Gx_rm, sizeof(float) * (size_t)KH * KW,   stream);
        cudaMallocAsync(&d_Gy_rm, sizeof(float) * (size_t)KH * KW,   stream);
        cudaMallocAsync(&d_ux_rm, sizeof(float) * (size_t)B * W * W, stream);
        cudaMallocAsync(&d_uy_rm, sizeof(float) * (size_t)B * W * W, stream);
        cudaMallocAsync(&d_ux_cm, sizeof(float) * (size_t)B * W * W, stream);
        cudaMallocAsync(&d_uy_cm, sizeof(float) * (size_t)B * W * W, stream);
        cudaMallocAsync(&d_dIdx,  sizeof(float) * (size_t)B * N * N, stream);
        cudaMallocAsync(&d_dIdy,  sizeof(float) * (size_t)B * N * N, stream);
    }

    void release(cudaStream_t stream = nullptr) {
        if (d_ws_buf) cudaFree(d_ws_buf);
        if (d_E_rm)   cudaFreeAsync(d_E_rm,  stream);
        if (d_Gx_rm)  cudaFreeAsync(d_Gx_rm, stream);
        if (d_Gy_rm)  cudaFreeAsync(d_Gy_rm, stream);
        if (d_ux_rm)  cudaFreeAsync(d_ux_rm, stream);
        if (d_uy_rm)  cudaFreeAsync(d_uy_rm, stream);
        if (d_ux_cm)  cudaFreeAsync(d_ux_cm, stream);
        if (d_uy_cm)  cudaFreeAsync(d_uy_cm, stream);
        if (d_dIdx)   cudaFreeAsync(d_dIdx,  stream);
        if (d_dIdy)   cudaFreeAsync(d_dIdy,  stream);
        if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
        if (filt_desc) cudnnDestroyFilterDescriptor(filt_desc);
        if (in_desc)   cudnnDestroyTensorDescriptor(in_desc);
        if (out_desc)  cudnnDestroyTensorDescriptor(out_desc);
        if (handle)    cudnnDestroy(handle);
    }
};

inline void srkCudnnForward(
    const float* d_E_batch,             // [B × N × N] column-major
    const float* d_I_batch,             // [B × N × N] column-major
    const float* d_Gx_spatial,          // [KH × KW]   column-major
    const float* d_Gy_spatial,
    int N, int W, int erosion, float dx,
    int B,
    cudaStream_t stream,
    float* d_R_batch,                   // [B × W × W] column-major
    SrkCudnnWorkspace& ws)
{
    (void)N; (void)W;
    const float alpha = dx * dx;
    const float beta  = 0.f;

    cudnnSetStream(ws.handle, stream);

    // Transpose E and filters into row-major scratch for cuDNN.
    {
        dim3 blk(16, 16, 1);
        dim3 grd((ws.N + 15) / 16, (ws.N + 15) / 16, B);
        srk_cm_to_rm_batched<<<grd, blk, 0, stream>>>(d_E_batch, ws.d_E_rm, ws.N, ws.N);
    }
    {
        dim3 blk(16, 16);
        dim3 grd((ws.KH + 15) / 16, (ws.KW + 15) / 16);
        srk_cm_to_rm_2d<<<grd, blk, 0, stream>>>(d_Gx_spatial, ws.d_Gx_rm, ws.KH, ws.KW);
        srk_cm_to_rm_2d<<<grd, blk, 0, stream>>>(d_Gy_spatial, ws.d_Gy_rm, ws.KH, ws.KW);
    }

    CUDNN_CHECK(cudnnConvolutionForward(
        ws.handle, &alpha,
        ws.in_desc,   ws.d_E_rm,
        ws.filt_desc, ws.d_Gx_rm,
        ws.conv_desc, ws.algo,
        ws.d_ws_buf,  ws.ws_bytes,
        &beta,
        ws.out_desc,  ws.d_ux_rm));

    CUDNN_CHECK(cudnnConvolutionForward(
        ws.handle, &alpha,
        ws.in_desc,   ws.d_E_rm,
        ws.filt_desc, ws.d_Gy_rm,
        ws.conv_desc, ws.algo,
        ws.d_ws_buf,  ws.ws_bytes,
        &beta,
        ws.out_desc,  ws.d_uy_rm));

    // Transpose row-major cuDNN outputs back to column-major.
    {
        dim3 blk(16, 16, 1);
        dim3 grd((ws.W + 15) / 16, (ws.W + 15) / 16, B);
        srk_rm_to_cm_batched<<<grd, blk, 0, stream>>>(ws.d_ux_rm, ws.d_ux_cm, ws.W, ws.W);
        srk_rm_to_cm_batched<<<grd, blk, 0, stream>>>(ws.d_uy_rm, ws.d_uy_cm, ws.W, ws.W);
    }

    // Gradients of I (column-major).
    {
        dim3 blk(16, 16, 1);
        dim3 grd((ws.N + 15) / 16, (ws.N + 15) / 16, B);
        srk_train_grad_x_batched<<<grd, blk, 0, stream>>>(d_I_batch, ws.d_dIdx, ws.N);
        srk_train_grad_y_batched<<<grd, blk, 0, stream>>>(d_I_batch, ws.d_dIdy, ws.N);
    }

    // Combine in column-major.
    {
        dim3 blk(16, 16, 1);
        dim3 grd((ws.W + 15) / 16, (ws.W + 15) / 16, B);
        srk_cudnn_combine<<<grd, blk, 0, stream>>>(
            ws.d_ux_cm, ws.d_uy_cm,
            ws.d_dIdx, ws.d_dIdy,
            d_R_batch,
            ws.W, ws.N, erosion);
    }
    cudaStreamSynchronize(stream);
}
