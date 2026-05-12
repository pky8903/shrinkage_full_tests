#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  srk_tangent_adjoint.cuh
//  Tangent-adjoint (mixed second-order) derivative for the shrinkage operator
//
//   h = c · ( conv(Gx, g)·fx + conv(Gy, g)·fy )      (forward)
//
//  Given base state (f, g), upstream adjoint λ_h, and tangent perturbations
//  (df, dg, dλ_h), the tangent-adjoint computes (dλ_f, dλ_g):
//
//   dλ_f = -c·∂x( dλ_h·conv(Gx,g) + λ_h·conv(Gx,dg) )
//          -c·∂y( dλ_h·conv(Gy,g) + λ_h·conv(Gy,dg) )
//
//        = D_x^T( dλ_h·u_x  + λ_h·du_x )
//          + D_y^T( dλ_h·u_y + λ_h·du_y )
//        = adjointWrtI(dλ_h, u_x,  u_y )            ← term A
//          + adjointWrtI(λ_h,  du_x, du_y)            ← term B
//
//   dλ_g = c·conv(G̃x, dλ_h·fx + λ_h·dfx)
//          + c·conv(G̃y, dλ_h·fy + λ_h·dfy)
//        = adjointWrtE(dλ_h, f,  Gx_freq, Gy_freq) ← term C
//          + adjointWrtE(λ_h,  df, Gx_freq, Gy_freq) ← term D
//
//  where u_x = c·conv(Gx,g), du_x = c·conv(Gx,dg), and similarly for y.
//  Both adjointWrtI / adjointWrtE come from srk_adjoint.cuh; we compute each
//  term into a scratch buffer and add.
// ─────────────────────────────────────────────────────────────────────────────

#include "srk_adjoint.cuh"

// Elementwise: dst[i] += src[i]  (size: n)
__global__ void srk_add_inplace(float* __restrict__ dst,
                                const float* __restrict__ src,
                                size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}

// ─────────────────────────────────────────────────────────────────────────────
//  SrkTanAdjWorkspace<Arch, M, ...>
//  Pre-allocated scratch for the tangent-adjoint operator.
//
//  Reuses SrkConvWorkspace (for the four conv(*, *) calls on g and dg) and
//  SrkAdjWorkspace (for the two adjointWrtE calls).  Adds W×W scratch for
//  the four conv outputs and N×N scratch buffers for the second-term
//  accumulation (dλ_f' and dλ_g').
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,
         unsigned int FPB_COL = FPB>
struct SrkTanAdjWorkspace {
    SrkConvWorkspace<Arch, M, EPT, FPB, FPB_COL> conv_ws;
    SrkAdjWorkspace <Arch, M, EPT, FPB, FPB_COL> adj_ws;

    // forward-style W×W conv outputs (c-scaled)
    float* d_u_x_g  = nullptr;
    float* d_u_y_g  = nullptr;
    float* d_u_x_dg = nullptr;
    float* d_u_y_dg = nullptr;

    // N×N scratch for second-term accumulation
    float* d_dLam_f_tmp = nullptr;
    float* d_dLam_g_tmp = nullptr;

    void allocate(unsigned N_large, unsigned W, cudaStream_t stream = nullptr) {
        conv_ws.allocate(N_large, W);
        adj_ws .allocate(N_large);
        cudaMallocAsync(&d_u_x_g,       sizeof(float) * W * W,             stream);
        cudaMallocAsync(&d_u_y_g,       sizeof(float) * W * W,             stream);
        cudaMallocAsync(&d_u_x_dg,      sizeof(float) * W * W,             stream);
        cudaMallocAsync(&d_u_y_dg,      sizeof(float) * W * W,             stream);
        cudaMallocAsync(&d_dLam_f_tmp,  sizeof(float) * N_large * N_large, stream);
        cudaMallocAsync(&d_dLam_g_tmp,  sizeof(float) * N_large * N_large, stream);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  shrinkageTangentAdjointOaS<Arch, M, ...>
//
//  Inputs (all device pointers):
//    d_f_pad     [N×N]   base state f
//    d_g_pad     [N×N]   base state g
//    d_df_pad    [N×N]   tangent direction df
//    d_dg_pad    [N×N]   tangent direction dg
//    d_lam_h     [W×W]   base adjoint λ_h
//    d_dlam_h    [W×W]   tangent direction of adjoint dλ_h
//    d_Gx_freq, d_Gy_freq : kernel frequencies (M × M/2+1)
//
//  Outputs (device, both required):
//    d_dLam_f    [N×N]   dλ_f
//    d_dLam_g    [N×N]   dλ_g
//
//  Scratch:
//    ws        : pre-allocated SrkTanAdjWorkspace
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,
         unsigned int FPB_COL = FPB>
void shrinkageTangentAdjointOaS(
    const float*        d_f_pad,
    const float*        d_g_pad,
    const float*        d_df_pad,
    const float*        d_dg_pad,
    const float*        d_lam_h,
    const float*        d_dlam_h,
    const cufftComplex* d_Gx_freq,
    const cufftComplex* d_Gy_freq,
    int N_large, int W, int erosion,
    float dx,
    cudaStream_t stream,
    float* d_dLam_f,   // [N × N]
    float* d_dLam_g,   // [N × N]
    SrkTanAdjWorkspace<Arch, M, EPT, FPB, FPB_COL>& ws)
{
    // ── A: u_x = c·conv(Gx, g),   u_y = c·conv(Gy, g)
    // ── A':du_x = c·conv(Gx, dg), du_y = c·conv(Gy, dg)
    srkConvOaS<Arch, M, EPT, FPB, FPB_COL>(
        d_g_pad,  d_Gx_freq, N_large, W, erosion, dx, stream, ws.d_u_x_g,  ws.conv_ws);
    srkConvOaS<Arch, M, EPT, FPB, FPB_COL>(
        d_g_pad,  d_Gy_freq, N_large, W, erosion, dx, stream, ws.d_u_y_g,  ws.conv_ws);
    srkConvOaS<Arch, M, EPT, FPB, FPB_COL>(
        d_dg_pad, d_Gx_freq, N_large, W, erosion, dx, stream, ws.d_u_x_dg, ws.conv_ws);
    srkConvOaS<Arch, M, EPT, FPB, FPB_COL>(
        d_dg_pad, d_Gy_freq, N_large, W, erosion, dx, stream, ws.d_u_y_dg, ws.conv_ws);

    // ── dλ_f  = adjointWrtI(dλ_h, u_x,  u_y) + adjointWrtI(λ_h, du_x, du_y) ──
    adjointWrtI(d_dlam_h, ws.d_u_x_g,  ws.d_u_y_g,
                W, N_large, erosion, stream, d_dLam_f);
    adjointWrtI(d_lam_h,  ws.d_u_x_dg, ws.d_u_y_dg,
                W, N_large, erosion, stream, ws.d_dLam_f_tmp);
    {
        const size_t n = (size_t)N_large * N_large;
        const int blk = 256;
        const int grd = (int)((n + blk - 1) / blk);
        srk_add_inplace<<<grd, blk, 0, stream>>>(d_dLam_f, ws.d_dLam_f_tmp, n);
    }

    // ── dλ_g  = adjointWrtE(dλ_h, f, Gx, Gy) + adjointWrtE(λ_h, df, Gx, Gy) ──
    adjointWrtE<Arch, M, EPT, FPB, FPB_COL>(
        d_dlam_h, d_f_pad, d_Gx_freq, d_Gy_freq,
        N_large, W, erosion, dx, stream, d_dLam_g, ws.adj_ws);
    adjointWrtE<Arch, M, EPT, FPB, FPB_COL>(
        d_lam_h, d_df_pad, d_Gx_freq, d_Gy_freq,
        N_large, W, erosion, dx, stream, ws.d_dLam_g_tmp, ws.adj_ws);
    {
        const size_t n = (size_t)N_large * N_large;
        const int blk = 256;
        const int grd = (int)((n + blk - 1) / blk);
        srk_add_inplace<<<grd, blk, 0, stream>>>(d_dLam_g, ws.d_dLam_g_tmp, n);
    }

    cudaStreamSynchronize(stream);
}
