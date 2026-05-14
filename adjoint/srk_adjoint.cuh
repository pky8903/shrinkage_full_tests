#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  srk_adjoint.cuh
//  Adjoint operators for R = scale*(conv(E,Gx)*dI/dx + conv(E,Gy)*dI/dy)
//
//  adjointWrtI:
//    δL/δI[r,c] = -[D_x^T(ux·v) + D_y^T(uy·v)][r-e,c-e]
//    (adjoint of central difference = negative central difference)
//    Input:  v[W×W], ux[W×W], uy[W×W]
//    Output: dLdI_pad[N_large×N_large]
//
//  adjointWrtE (OaS, AC-FUSE):
//    δL/δE = scale*(conv(Gx, dI/dx·v) + conv(Gy, dI/dy·v))
//    Input:  v[W×W], I_pad[N_large×N_large], Gx_freq, Gy_freq
//    Output: dLdE_pad[N_large×N_large]
//    The x-direction kernel fuses both Gx and Gy contributions (AC-FUSE).
// ─────────────────────────────────────────────────────────────────────────────

#include "srk_forward.cuh"

// ─────────────────────────────────────────────────────────────────────────────
//  adjointWrtI
// ─────────────────────────────────────────────────────────────────────────────
__global__ void adjointWrtI_kernel(
    const float* __restrict__ v,    // [W × W]
    const float* __restrict__ ux,   // [W × W]  = scale*conv(E,Gx)
    const float* __restrict__ uy,   // [W × W]  = scale*conv(E,Gy)
    float*       __restrict__ dLdI, // [N_large × N_large]
    int W, int N_large, int erosion)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;  // global col
    int gy = blockIdx.y * blockDim.y + threadIdx.y;  // global row
    if (gx >= N_large || gy >= N_large) { return; }

    // Map to output-domain coordinates
    int ox = gx - erosion;
    int oy = gy - erosion;

    // w_x(r,c) = ux[r,c] * v[r,c]  (returns 0 outside [0,W) range)
    // col-major: buf[c * W + r]
    auto wx = [&](int r, int c) -> float {
        if (r < 0 || r >= W || c < 0 || c >= W) return 0.f;
        return ux[c * W + r] * v[c * W + r];
    };
    auto wy = [&](int r, int c) -> float {
        if (r < 0 || r >= W || c < 0 || c >= W) return 0.f;
        return uy[c * W + r] * v[c * W + r];
    };

    // Adjoint of D_x (central diff in x):  D_x^T(w)[r,c] = (w[r,c-1] - w[r,c+1])/2
    // Adjoint of D_y (central diff in y):  D_y^T(w)[r,c] = (w[r-1,c] - w[r+1,c])/2
    // dL/dI = D_x^T(wx) + D_y^T(wy)
    float val =
        (wx(oy, ox-1) - wx(oy, ox+1)) * 0.5f +
        (wy(oy-1, ox) - wy(oy+1, ox)) * 0.5f;

    dLdI[gx * N_large + gy] = val;   // col-major
}

inline void adjointWrtI(
    const float* d_v,       // [W × W]
    const float* d_ux,      // [W × W]
    const float* d_uy,      // [W × W]
    int W, int N_large, int erosion,
    cudaStream_t stream,
    float* d_dLdI)          // [N_large × N_large]
{
    cudaMemsetAsync(d_dLdI, 0, sizeof(float)*N_large*N_large, stream);
    dim3 blk(16, 16);
    dim3 grd((N_large+15)/16, (N_large+15)/16);
    adjointWrtI_kernel<<<grd, blk, 0, stream>>>(
        d_v, d_ux, d_uy, d_dLdI, W, N_large, erosion);
    cudaStreamSynchronize(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Fused x-direction kernel for adjointWrtE (AC-FUSE)
//  Processes one x_freq band per thread block.
//  Phase 1: FFT_y(Hx_y) * Gx * scale → accumulator
//  Phase 2: FFT_y(Hy_y) * Gy * scale → add to accumulator
//  Phase 3: IFFT_y(accumulator) → output
// ─────────────────────────────────────────────────────────────────────────────
template<class FFTF, class FFTI,
         unsigned int ColStride,
         unsigned int NumCols,
         class CT = typename FFTF::value_type>
__launch_bounds__(FFTF::max_threads_per_block) __global__
void srk_adj_fused_cols(
    const CT* __restrict__ Hx_y,
    const CT* __restrict__ Hy_y,
    const CT* __restrict__ Gx_freq,
    const CT* __restrict__ Gy_freq,
    CT*       __restrict__ output,
    typename FFTF::workspace_type wsf,
    typename FFTI::workspace_type wsi,
    typename CT::value_type scale)
{
    using RT = typename CT::value_type;
    CT td[FFTF::storage_size];
    CT acc[FFTF::storage_size];
    for (unsigned int i = 0; i < FFTF::storage_size; ++i) acc[i] = CT{RT(0), RT(0)};

    const unsigned int bid    = blockIdx.x * FFTF::ffts_per_block + threadIdx.y;
    const unsigned int stride = ColStride * FFTF::stride;
    const CT sc{scale, RT(0)};

    extern __shared__ __align__(alignof(float4)) CT smem[];

    // ── Phase 1: FFT_y(Hx_y) * Gx → accumulator ──────────────────────────
    {
        unsigned int idx = bid + threadIdx.x * ColStride;
        for (unsigned int i = 0; i < FFTF::elements_per_thread; ++i) {
            if (i * FFTF::stride + threadIdx.x < cufftdx::size_of<FFTF>::value)
                if (bid < NumCols) td[i] = Hx_y[idx];
            idx += stride;
        }
    }
    FFTF().execute(td, smem, wsf);
    {
        unsigned int idx = bid + threadIdx.x * ColStride;
        for (unsigned int i = 0; i < FFTF::elements_per_thread; ++i) {
            if (i * FFTF::stride + threadIdx.x < cufftdx::size_of<FFTF>::value) {
                if (bid < NumCols)
                    acc[i] = acc[i] + td[i] * (sc * Gx_freq[idx]);
            }
            idx += stride;
        }
    }

    // ── Phase 2: FFT_y(Hy_y) * Gy → add to accumulator ───────────────────
    {
        unsigned int idx = bid + threadIdx.x * ColStride;
        for (unsigned int i = 0; i < FFTF::elements_per_thread; ++i) {
            if (i * FFTF::stride + threadIdx.x < cufftdx::size_of<FFTF>::value)
                if (bid < NumCols) td[i] = Hy_y[idx];
            idx += stride;
        }
    }
    FFTF().execute(td, smem, wsf);
    {
        unsigned int idx = bid + threadIdx.x * ColStride;
        for (unsigned int i = 0; i < FFTF::elements_per_thread; ++i) {
            if (i * FFTF::stride + threadIdx.x < cufftdx::size_of<FFTF>::value) {
                if (bid < NumCols)
                    acc[i] = acc[i] + td[i] * (sc * Gy_freq[idx]);
            }
            idx += stride;
        }
    }

    // ── Phase 3: IFFT_y(accumulator) → output ─────────────────────────────
    FFTI().execute(acc, smem, wsi);
    {
        unsigned int idx = bid + threadIdx.x * ColStride;
        for (unsigned int i = 0; i < FFTI::elements_per_thread; ++i) {
            if (i * FFTI::stride + threadIdx.x < cufftdx::size_of<FFTI>::value) {
                if (bid < NumCols) output[idx] = acc[i];
            }
            idx += stride;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Load adjoint tile: h_val = v[out] * dIdx[global] (or dIdy) at valid pixels
//  Stores at fftshift-compatible position for subsequent FFT_x pipeline.
// ─────────────────────────────────────────────────────────────────────────────
template<bool isX>   // isX=true → multiply by dIdx, false → dIdy
__global__ void srk_adj_load_tile(
    const float* __restrict__ v,
    const float* __restrict__ dIdx_or_dIdy,
    float* __restrict__ buf,
    int W, int N_large, int erosion,
    unsigned int M, unsigned int tile_stride,
    unsigned int tid_x, unsigned int tid_y)
{
    // Iterate over "shifted" tile coordinates (sx, sy) ∈ [0,M)
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= (int)M || sy >= (int)M) return;

    int gx = (int)(tile_stride * tid_x) + sx;
    int gy = (int)(tile_stride * tid_y) + sy;

    float h_val = 0.f;
    bool valid_local  = sx >= erosion && sx < (int)M - erosion
                     && sy >= erosion && sy < (int)M - erosion;
    bool valid_global = gx >= erosion && gx < N_large - erosion
                     && gy >= erosion && gy < N_large - erosion;
    if (valid_local && valid_global) {
        int ox = gx - erosion, oy = gy - erosion;
        h_val = v[ox * W + oy] * dIdx_or_dIdy[gx * N_large + gy];   // col-major
    }

    // Store at (inverse-fftshift) position to match forward fftshift convention
    int lx = (sx + M/2) % M;
    int ly = (sy + M/2) % M;
    buf[lx * M + ly] = h_val;   // col-major
}

// Atomic-add real tile back to dLdE_pad (with fftshift unrolling)
__global__ void srk_adj_store_atomic(
    const float* __restrict__ src,
    float*       __restrict__ dLdE,
    int N_large,
    unsigned int M, unsigned int tile_stride,
    unsigned int tid_x, unsigned int tid_y)
{
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= (int)M || ly >= (int)M) return;

    float val = src[lx * M + ly];   // col-major

    // Recover shifted coords (inverse of the load's store step)
    int sx = (lx + M/2) % M;
    int sy = (ly + M/2) % M;

    int gx = (int)(tile_stride * tid_x) + sx;
    int gy = (int)(tile_stride * tid_y) + sy;

    if (gx >= 0 && gx < N_large && gy >= 0 && gy < N_large)
        dLdE[gx * N_large + gy] += val;   // col-major
}

// ─────────────────────────────────────────────────────────────────────────────
//  SrkAdjWorkspace<Arch, M, EPT, FPB>
//  Pre-allocated buffers and cufftDX workspaces for adjointWrtE / adjointSRK.
//  Allocate once before timing loops; pass by reference to the hot functions.
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,   // FPB for row-direction FFTs
         unsigned int FPB_COL = FPB> // FPB for col-direction FFTs (C2C fwd/inv)
struct SrkAdjWorkspace {
    using PT       = float;
    using CT       = cufftdx::complex<PT>;
    using fft_base = decltype(cufftdx::Block() + cufftdx::Precision<PT>() + cufftdx::SM<Arch>());
    using FFT_R2C  = decltype(fft_base() + cufftdx::Type<cufftdx::fft_type::r2c>()
                              + cufftdx::Size<M>() + cufftdx::ElementsPerThread<EPT>()
                              + cufftdx::FFTsPerBlock<FPB>());
    using FFT_C2R  = decltype(fft_base() + cufftdx::Type<cufftdx::fft_type::c2r>()
                              + cufftdx::Size<M>() + cufftdx::ElementsPerThread<EPT>()
                              + cufftdx::FFTsPerBlock<FPB>());
    using FFT_FWD  = decltype(fft_base() + cufftdx::Type<cufftdx::fft_type::c2c>()
                              + cufftdx::Direction<cufftdx::fft_direction::forward>()
                              + cufftdx::Size<M>() + cufftdx::ElementsPerThread<EPT>()
                              + cufftdx::FFTsPerBlock<FPB_COL>());
    using FFT_INV  = decltype(fft_base() + cufftdx::Type<cufftdx::fft_type::c2c>()
                              + cufftdx::Direction<cufftdx::fft_direction::inverse>()
                              + cufftdx::Size<M>() + cufftdx::ElementsPerThread<EPT>()
                              + cufftdx::FFTsPerBlock<FPB_COL>());
    static constexpr unsigned int HALF = FFT_R2C::output_length;

    float* d_dIdx     = nullptr;   // [N_large × N_large]
    float* d_dIdy     = nullptr;   // [N_large × N_large]
    float* d_buf_x    = nullptr;   // [M × M]
    float* d_buf_y    = nullptr;   // [M × M]
    float* d_real_out = nullptr;   // [M × M]
    CT*    d_hx       = nullptr;   // [M × HALF]
    CT*    d_hy       = nullptr;   // [M × HALF]
    CT*    d_combined = nullptr;   // [M × HALF]
    float* d_scratch  = nullptr;   // [N_large × N_large] — discard output of adjointWrtI/E

    typename FFT_R2C::workspace_type ws_r2c;
    typename FFT_C2R::workspace_type ws_c2r;
    typename FFT_FWD::workspace_type ws_fwd;
    typename FFT_INV::workspace_type ws_inv;

    void allocate(unsigned N_large, cudaStream_t stream = nullptr) {
        cudaMallocAsync(&d_dIdx,     sizeof(float) * N_large * N_large,    stream);
        cudaMallocAsync(&d_dIdy,     sizeof(float) * N_large * N_large,    stream);
        cudaMallocAsync(&d_buf_x,    sizeof(float) * M * M,                stream);
        cudaMallocAsync(&d_buf_y,    sizeof(float) * M * M,                stream);
        cudaMallocAsync(&d_real_out, sizeof(float) * M * M,                stream);
        cudaMallocAsync(&d_hx,       sizeof(CT) * M * HALF,                stream);
        cudaMallocAsync(&d_hy,       sizeof(CT) * M * HALF,                stream);
        cudaMallocAsync(&d_combined, sizeof(CT) * M * HALF,                stream);
        cudaMallocAsync(&d_scratch,  sizeof(float) * N_large * N_large,    stream);
        ws_r2c = cufftdx::make_workspace<FFT_R2C>(stream);
        ws_c2r = cufftdx::make_workspace<FFT_C2R>(stream);
        ws_fwd = cufftdx::make_workspace<FFT_FWD>(stream);
        ws_inv = cufftdx::make_workspace<FFT_INV>(stream);
        cudaFuncSetAttribute(srk_fft_rows<FFT_R2C, float, CT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FFT_R2C::shared_memory_size);
        cudaFuncSetAttribute(srk_fft_rows<FFT_C2R, CT, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FFT_C2R::shared_memory_size);
        cudaFuncSetAttribute(
            (srk_adj_fused_cols<FFT_FWD, FFT_INV, HALF, HALF, CT>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * FFT_FWD::shared_memory_size);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  adjointWrtE<Arch, M>
//  OaS adjoint:  dLdE_pad[N_large×N_large] = scale*(conv(v*dIdx,Gx) + conv(v*dIdy,Gy))
//  ws: pre-allocated workspace (no allocation inside this function)
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,
         unsigned int FPB_COL = FPB>
void adjointWrtE(
    const float*        d_v,         // [W × W]
    const float*        d_I_pad,     // [N_large × N_large]
    const cufftComplex* d_Gx_freq,   // [M × (M/2+1)]
    const cufftComplex* d_Gy_freq,   // [M × (M/2+1)]
    int N_large, int W, int erosion,
    float dx,
    cudaStream_t stream,
    float* d_dLdE_pad,               // [N_large × N_large]
    SrkAdjWorkspace<Arch, M, EPT, FPB, FPB_COL>& ws)
{
    using WS      = SrkAdjWorkspace<Arch, M, EPT, FPB, FPB_COL>;
    using CT      = typename WS::CT;
    using FFT_R2C = typename WS::FFT_R2C;
    using FFT_C2R = typename WS::FFT_C2R;
    using FFT_FWD = typename WS::FFT_FWD;
    using FFT_INV = typename WS::FFT_INV;
    constexpr unsigned int HALF = WS::HALF;

    const int  tile_stride = M - 2 * erosion;
    const int  num_tiles   = (W + tile_stride - 1) / tile_stride;
    const float scale      = dx * dx / (float(M) * float(M));
    const unsigned int fused_smem = 2 * FFT_FWD::shared_memory_size;

    dim3 ggrd((N_large+15)/16,(N_large+15)/16), gblk(16,16);
    srk_grad_x<<<ggrd, gblk, 0, stream>>>(d_I_pad, ws.d_dIdx, N_large);
    srk_grad_y<<<ggrd, gblk, 0, stream>>>(d_I_pad, ws.d_dIdy, N_large);

    cudaMemsetAsync(d_dLdE_pad, 0, sizeof(float)*N_large*N_large, stream);

    const dim3 tblk(32, 32), tgrd((M+31)/32, (M+31)/32);
    const dim3 rows_blk_r2c = FFT_R2C::block_dim;
    const dim3 rows_blk_c2r = FFT_C2R::block_dim;
    const dim3 rows_grd((M + FPB - 1) / FPB);
    const dim3 cols_blk = FFT_FWD::block_dim;
    const dim3 cols_grd((HALF + FPB_COL - 1) / FPB_COL);

    for (int ty = 0; ty < num_tiles; ++ty) {
        for (int tx = 0; tx < num_tiles; ++tx) {
            cudaMemsetAsync(ws.d_buf_x, 0, sizeof(float)*M*M, stream);
            cudaMemsetAsync(ws.d_buf_y, 0, sizeof(float)*M*M, stream);

            srk_adj_load_tile<true><<<tgrd, tblk, 0, stream>>>(
                d_v, ws.d_dIdx, ws.d_buf_x,
                W, N_large, erosion, M, (unsigned)tile_stride,
                (unsigned)tx, (unsigned)ty);

            srk_adj_load_tile<false><<<tgrd, tblk, 0, stream>>>(
                d_v, ws.d_dIdy, ws.d_buf_y,
                W, N_large, erosion, M, (unsigned)tile_stride,
                (unsigned)tx, (unsigned)ty);

            srk_fft_rows<FFT_R2C, float, CT><<<rows_grd, rows_blk_r2c,
                FFT_R2C::shared_memory_size, stream>>>(ws.d_buf_x, ws.d_hx, ws.ws_r2c);

            srk_fft_rows<FFT_R2C, float, CT><<<rows_grd, rows_blk_r2c,
                FFT_R2C::shared_memory_size, stream>>>(ws.d_buf_y, ws.d_hy, ws.ws_r2c);

            srk_adj_fused_cols<FFT_FWD, FFT_INV, HALF, HALF, CT><<<cols_grd, cols_blk,
                fused_smem, stream>>>(
                ws.d_hx, ws.d_hy,
                reinterpret_cast<const CT*>(d_Gx_freq),
                reinterpret_cast<const CT*>(d_Gy_freq),
                ws.d_combined,
                ws.ws_fwd, ws.ws_inv, scale);

            srk_fft_rows<FFT_C2R, CT, float><<<rows_grd, rows_blk_c2r,
                FFT_C2R::shared_memory_size, stream>>>(ws.d_combined, ws.d_real_out, ws.ws_c2r);

            srk_adj_store_atomic<<<tgrd, tblk, 0, stream>>>(
                ws.d_real_out, d_dLdE_pad,
                N_large, M, (unsigned)tile_stride,
                (unsigned)tx, (unsigned)ty);
        }
    }
    cudaStreamSynchronize(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
//  adjointSRK<Arch, M>  — fused adjoint for both inputs simultaneously
//
//  Given v = dL/dR, computes both:
//    dL/dI[N×N]  via adjointWrtI:  D_x^T(ux·v) + D_y^T(uy·v)
//    dL/dE[N×N]  via adjointWrtE:  conv(v·dIdx, Gx) + conv(v·dIdy, Gy)
//
//  Inputs:
//    d_v        [W × W]          — upstream gradient dL/dR
//    d_ux       [W × W]          — scale*conv(E, Gx) from forward pass
//    d_uy       [W × W]          — scale*conv(E, Gy) from forward pass
//    d_I_pad    [N_large × N_large] — padded image (for central-diff)
//    d_Gx_freq  [M × (M/2+1)]   — frequency-domain Gx kernel
//    d_Gy_freq  [M × (M/2+1)]   — frequency-domain Gy kernel
//  Outputs:
//    d_dLdI     [N_large × N_large]
//    d_dLdE     [N_large × N_large]
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,
         unsigned int FPB_COL = FPB>
void adjointSRK(
    const float*        d_v,
    const float*        d_ux,
    const float*        d_uy,
    const float*        d_I_pad,
    const cufftComplex* d_Gx_freq,
    const cufftComplex* d_Gy_freq,
    int N_large, int W, int erosion,
    float dx,
    cudaStream_t stream,
    float* d_dLdI,
    float* d_dLdE,
    SrkAdjWorkspace<Arch, M, EPT, FPB, FPB_COL>& ws)
{
    adjointWrtI(d_v, d_ux, d_uy, W, N_large, erosion, stream, d_dLdI);
    adjointWrtE<Arch, M, EPT, FPB, FPB_COL>(
        d_v, d_I_pad, d_Gx_freq, d_Gy_freq,
        N_large, W, erosion, dx, stream, d_dLdE, ws);
}
