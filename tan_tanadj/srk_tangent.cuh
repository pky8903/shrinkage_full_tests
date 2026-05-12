#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  srk_tangent.cuh
//  Tangent (forward-mode) derivative for the shrinkage operator
//
//   h = c · ( conv(Gx, g)·fx + conv(Gy, g)·fy )
//
//  Given perturbations (df, dg), the tangent derivative is
//
//   dh = c · ( conv(Gx, dg)·fx + conv(Gy, dg)·fy
//             + conv(Gx, g)·dfx + conv(Gy, g)·dfy )
//
//  where fx = ∂f/∂x, fy = ∂f/∂y, dfx = ∂(df)/∂x, dfy = ∂(df)/∂y
//  (central difference on the padded N×N grid).
//
//  Implementation reuses srkConvOaS<Arch,M,...> from srk_forward.cuh for all
//  four conv(*, *) terms — cufftDX overlap-and-save tiled 2D convolution.
// ─────────────────────────────────────────────────────────────────────────────

#include "srk_forward.cuh"

// ─────────────────────────────────────────────────────────────────────────────
//  Tangent combine: dh[oy,ox] =  u_x_dg · fx + u_y_dg · fy
//                              + u_x_g  · dfx + u_y_g  · dfy
//  evaluated at the valid region (oy,ox in [0,W)).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void srk_tangent_combine(
    const float* __restrict__ u_x_g,    // [W × W]  c·conv(Gx, g)
    const float* __restrict__ u_y_g,    // [W × W]  c·conv(Gy, g)
    const float* __restrict__ u_x_dg,   // [W × W]  c·conv(Gx, dg)
    const float* __restrict__ u_y_dg,   // [W × W]  c·conv(Gy, dg)
    const float* __restrict__ fx,       // [N × N]  ∂f/∂x
    const float* __restrict__ fy,       // [N × N]  ∂f/∂y
    const float* __restrict__ dfx,      // [N × N]  ∂(df)/∂x
    const float* __restrict__ dfy,      // [N × N]  ∂(df)/∂y
    float*       __restrict__ dh,       // [W × W]  output
    int W, int N_large, int erosion)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W || oy >= W) return;
    int gx = ox + erosion, gy = oy + erosion;

    float v =
        u_x_dg[ox*W + oy] * fx[gx*N_large + gy]
      + u_y_dg[ox*W + oy] * fy[gx*N_large + gy]
      + u_x_g [ox*W + oy] * dfx[gx*N_large + gy]
      + u_y_g [ox*W + oy] * dfy[gx*N_large + gy];   // col-major

    dh[ox*W + oy] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
//  SrkTanWorkspace<Arch, M, EPT, FPB, FPB_COL>
//  Pre-allocated scratch for the tangent kernel.
//  Aggregates a SrkConvWorkspace (reused for all four conv calls) plus
//  W×W scratch for u_x_g, u_y_g, u_x_dg, u_y_dg and N×N scratch for
//  dfx, dfy (fx, fy are reused from the conv workspace's d_dIdx/d_dIdy).
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,
         unsigned int FPB_COL = FPB>
struct SrkTanWorkspace {
    SrkConvWorkspace<Arch, M, EPT, FPB, FPB_COL> conv_ws;

    float* d_u_x_g  = nullptr;   // [W × W]
    float* d_u_y_g  = nullptr;   // [W × W]
    float* d_u_x_dg = nullptr;   // [W × W]
    float* d_u_y_dg = nullptr;   // [W × W]
    float* d_dfx    = nullptr;   // [N_large × N_large]
    float* d_dfy    = nullptr;   // [N_large × N_large]

    void allocate(unsigned N_large, unsigned W, cudaStream_t stream = nullptr) {
        conv_ws.allocate(N_large, W);
        cudaMallocAsync(&d_u_x_g,  sizeof(float) * W * W,                 stream);
        cudaMallocAsync(&d_u_y_g,  sizeof(float) * W * W,                 stream);
        cudaMallocAsync(&d_u_x_dg, sizeof(float) * W * W,                 stream);
        cudaMallocAsync(&d_u_y_dg, sizeof(float) * W * W,                 stream);
        cudaMallocAsync(&d_dfx,    sizeof(float) * N_large * N_large,     stream);
        cudaMallocAsync(&d_dfy,    sizeof(float) * N_large * N_large,     stream);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  shrinkageTangentOaS<Arch, M, ...>
//
//  Computes the analytical tangent derivative dh given perturbations (df, dg)
//  using cufftDX overlap-and-save block FFTs for all four conv(*, *) terms.
//
//  Inputs:
//    d_f_pad   [N×N]   base state f                (padded)
//    d_g_pad   [N×N]   base state g                (padded)
//    d_df_pad  [N×N]   perturbation direction df   (padded)
//    d_dg_pad  [N×N]   perturbation direction dg   (padded)
//    d_Gx_freq / d_Gy_freq : precomputed kernel frequencies (M × M/2+1)
//
//  Output:
//    d_dh      [W×W]   tangent derivative at the valid region
//
//  Scratch:
//    ws        : pre-allocated SrkTanWorkspace (no allocation inside)
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,
         unsigned int FPB_COL = FPB>
void shrinkageTangentOaS(
    const float*        d_f_pad,
    const float*        d_g_pad,
    const float*        d_df_pad,
    const float*        d_dg_pad,
    const cufftComplex* d_Gx_freq,
    const cufftComplex* d_Gy_freq,
    int N_large, int W, int erosion,
    float dx,
    cudaStream_t stream,
    float* d_dh,
    SrkTanWorkspace<Arch, M, EPT, FPB, FPB_COL>& ws)
{
    // c-scaled convolutions (c = dx²/M² is baked into srkConvOaS).
    srkConvOaS<Arch, M, EPT, FPB, FPB_COL>(
        d_g_pad,  d_Gx_freq, N_large, W, erosion, dx, stream, ws.d_u_x_g,  ws.conv_ws);
    srkConvOaS<Arch, M, EPT, FPB, FPB_COL>(
        d_g_pad,  d_Gy_freq, N_large, W, erosion, dx, stream, ws.d_u_y_g,  ws.conv_ws);
    srkConvOaS<Arch, M, EPT, FPB, FPB_COL>(
        d_dg_pad, d_Gx_freq, N_large, W, erosion, dx, stream, ws.d_u_x_dg, ws.conv_ws);
    srkConvOaS<Arch, M, EPT, FPB, FPB_COL>(
        d_dg_pad, d_Gy_freq, N_large, W, erosion, dx, stream, ws.d_u_y_dg, ws.conv_ws);

    // Central differences for fx/fy and dfx/dfy on the full N×N grid.
    dim3 gblk(16, 16);
    dim3 ggrd((N_large+15)/16, (N_large+15)/16);
    srk_grad_x<<<ggrd, gblk, 0, stream>>>(d_f_pad,  ws.conv_ws.d_dIdx, N_large);
    srk_grad_y<<<ggrd, gblk, 0, stream>>>(d_f_pad,  ws.conv_ws.d_dIdy, N_large);
    srk_grad_x<<<ggrd, gblk, 0, stream>>>(d_df_pad, ws.d_dfx,          N_large);
    srk_grad_y<<<ggrd, gblk, 0, stream>>>(d_df_pad, ws.d_dfy,          N_large);

    // Combine into dh at valid pixels.
    dim3 rblk(16, 16), rgrd((W+15)/16, (W+15)/16);
    srk_tangent_combine<<<rgrd, rblk, 0, stream>>>(
        ws.d_u_x_g, ws.d_u_y_g, ws.d_u_x_dg, ws.d_u_y_dg,
        ws.conv_ws.d_dIdx, ws.conv_ws.d_dIdy,
        ws.d_dfx, ws.d_dfy,
        d_dh, W, N_large, erosion);

    cudaStreamSynchronize(stream);
}
