#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  srk_forward.cuh
//  Forward operator: R = scale * (conv(E,Gx) * dI/dx + conv(E,Gy) * dI/dy)
//
//  Implements:
//    srkPrecomputeKernelFreq  – fftshift + 2D R2C FFT of a real kernel
//    srkConvOaS<Arch,M>       – OaS conv of E with one freq-domain kernel
//    srkForwardOaS<Arch,M>    – full forward operator (conv + gradient multiply)
//
//  Data layout conventions:
//    • Real arrays are col-major (ArrayFire / Fortran):
//        arr[col * H + row]    (row varies fastest, then col, then channel, then batch)
//    • Block-FFT kernels operate on M contiguous elements (=one column in col-major).
//    • Intermediate freq buffers are M × (M/2+1) complex; we treat them as opaque
//      and never spatially index into them outside the FFT kernels themselves.
// ─────────────────────────────────────────────────────────────────────────────

#include <cufft.h>
#include <cuda_runtime.h>
#include <cufftdx.hpp>
#include <nvtx3/nvToolsExt.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
//  Utility kernels (shared with adjoint)
// ─────────────────────────────────────────────────────────────────────────────

// Load tile from full input with simultaneous 2D fftshift → buffer
template<typename T>
__global__ void srk_loadFFTshiftedTile(
    const T* __restrict__ input,
    T* __restrict__ buffer,
    unsigned int full_N,
    unsigned int M,
    unsigned int tile_stride,
    unsigned int tid_x, unsigned int tid_y)
{
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= (int)M || ly >= (int)M) return;

    int gx = (int)(tile_stride * tid_x) + lx;
    int gy = (int)(tile_stride * tid_y) + ly;

    T val = T(0);
    if (gx >= 0 && gx < (int)full_N && gy >= 0 && gy < (int)full_N)
        val = input[gx * full_N + gy];      // col-major: [col * H + row]

    int sx = (lx + M / 2) % M;
    int sy = (ly + M / 2) % M;
    buffer[sx * M + sy] = val;              // col-major
}

// ─────────────────────────────────────────────────────────────────────────────
//  cufftDX inner kernels (mirror of ffts.cuh, self-contained here)
// ─────────────────────────────────────────────────────────────────────────────

// FFT_x along rows: R2C or C2R (one FFT per row; rows indexed by global_fft_id)
template<class FFT, class InT, class OutT>
__launch_bounds__(FFT::max_threads_per_block) __global__
void srk_fft_rows(const InT* input, OutT* output,
                  typename FFT::workspace_type ws)
{
    using CT = typename FFT::value_type;
    CT td[FFT::storage_size];

    const unsigned int fid = blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    const unsigned int in_off  = FFT::input_length  * fid;
    const unsigned int out_off = FFT::output_length * fid;
    constexpr unsigned int s   = FFT::stride;

    unsigned int idx = in_off + threadIdx.x;
    for (unsigned int i = 0; i < FFT::input_ept; ++i) {
        if (i * s + threadIdx.x < FFT::input_length)
            reinterpret_cast<InT*>(td)[i] = reinterpret_cast<const InT*>(input)[idx];
        idx += s;
    }

    extern __shared__ __align__(alignof(float4)) CT smem[];
    FFT().execute(td, smem, ws);

    idx = out_off + threadIdx.x;
    for (unsigned int i = 0; i < FFT::output_ept; ++i) {
        if (i * s + threadIdx.x < FFT::output_length)
            reinterpret_cast<OutT*>(output)[idx] = reinterpret_cast<const OutT*>(td)[i];
        idx += s;
    }
}

// FFT_y along columns + pointwise multiply + IFFT_y (fused, single launch)
// Input layout: complex[row * (M/2+1) + x_freq]
// bid = x_freq index; elements at bid, bid+(M/2+1), bid+2*(M/2+1), ... for varying row
template<class FFTF, class FFTI,
         unsigned int ColStride,   // = M/2+1
         unsigned int NumCols,     // = M/2+1 (valid range for bid)
         class CT = typename FFTF::value_type>
__launch_bounds__(FFTF::max_threads_per_block) __global__
void srk_conv_cols(
    const CT* __restrict__ input,
    const CT* __restrict__ kernel,
    CT* __restrict__ output,
    typename FFTF::workspace_type wsf,
    typename FFTI::workspace_type wsi,
    typename CT::value_type scale)
{
    using RT = typename CT::value_type;
    CT td[FFTF::storage_size];

    const unsigned int bid    = blockIdx.x * FFTF::ffts_per_block + threadIdx.y;
    const unsigned int stride = ColStride * FFTF::stride;
    const CT sc{scale, RT(0)};

    unsigned int idx = bid + threadIdx.x * ColStride;
    for (unsigned int i = 0; i < FFTF::elements_per_thread; ++i) {
        if (i * FFTF::stride + threadIdx.x < cufftdx::size_of<FFTF>::value)
            if (bid < NumCols) td[i] = input[idx];
        idx += stride;
    }

    extern __shared__ __align__(alignof(float4)) CT smem[];
    FFTF().execute(td, smem, wsf);

    idx = bid + threadIdx.x * ColStride;
    for (unsigned int i = 0; i < FFTF::elements_per_thread; ++i) {
        if (i * FFTF::stride + threadIdx.x < cufftdx::size_of<FFTF>::value) {
            if (bid < NumCols) td[i] = td[i] * (sc * kernel[idx]);
        }
        idx += stride;
    }

    FFTI().execute(td, smem, wsi);

    idx = bid + threadIdx.x * ColStride;
    for (unsigned int i = 0; i < FFTI::elements_per_thread; ++i) {
        if (i * FFTI::stride + threadIdx.x < cufftdx::size_of<FFTI>::value) {
            if (bid < NumCols) output[idx] = td[i];
        }
        idx += stride;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  srkPrecomputeKernelFreq
//  Pads kernel G (kw×kh real, centered at center) into an M×M array,
//  applies 2D fftshift, then 2D R2C FFT.
//  Output: M*(M/2+1) cufftComplex values.
// ─────────────────────────────────────────────────────────────────────────────
inline void srkPrecomputeKernelFreq(
    const std::vector<float>& h_G,
    int kw, int kh,           // kernel spatial dimensions
    int M,                    // FFT tile size
    cufftComplex* d_G_freq,   // output: M*(M/2+1) complex
    cudaStream_t stream)
{
    // Zero-pad G into M×M, centered
    std::vector<float> h_pad(M * M, 0.f);
    int ox = (M - kw + 1) / 2;   // center G at col M/2 so fftshift maps it to 0
    int oy = (M - kh + 1) / 2;   // center G at row M/2 so fftshift maps it to 0
    for (int y = 0; y < kh; ++y)
        for (int x = 0; x < kw; ++x)
            h_pad[(ox + x) * M + (oy + y)] = h_G[x * kh + y];   // col-major

    // Upload padded kernel
    float* d_pad = nullptr;
    cudaMallocAsync(&d_pad, sizeof(float) * M * M, stream);
    cudaMemcpyAsync(d_pad, h_pad.data(), sizeof(float) * M * M,
                    cudaMemcpyHostToDevice, stream);

    // fftshift in-place (out-of-place via temp)
    float* d_shifted = nullptr;
    cudaMallocAsync(&d_shifted, sizeof(float) * M * M, stream);
    // 2D fftshift (host-side; need D→H to finish, then host compute, then H→D)
    {
        std::vector<float> h_tmp(M * M);
        cudaMemcpyAsync(h_tmp.data(), d_pad,
                        sizeof(float)*M*M, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);   // host must read h_tmp now
        std::vector<float> h_sh(M * M);
        for (int y = 0; y < M; ++y)
            for (int x = 0; x < M; ++x)
                h_sh[((x+M/2)%M)*M + ((y+M/2)%M)] = h_tmp[x*M + y];   // col-major
        cudaMemcpyAsync(d_shifted, h_sh.data(),
                        sizeof(float)*M*M, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);   // h_sh going out of scope; ensure upload done
    }

    // 2D R2C FFT
    cufftHandle plan;
    cufftPlan2d(&plan, M, M, CUFFT_R2C);
    cufftSetStream(plan, stream);
    cufftExecR2C(plan, d_shifted, d_G_freq);
    cufftDestroy(plan);

    cudaFreeAsync(d_pad, stream);
    cudaFreeAsync(d_shifted, stream);
    cudaStreamSynchronize(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
//  srk_fft_rows_c2r_valid
//  Fused IFFT_x (C2R, row-wise) + fftshift-undo + valid-tile accumulation.
//  Adapted from fft_2d_kernel_tile_out_y in 10.SRSRK_forward_cufftDX_2d_conv_tiled_.
//  Replaces srk_fft_rows<FFT_C2R> + srk_store_valid_tile (two separate launches).
// ─────────────────────────────────────────────────────────────────────────────
template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
void srk_fft_rows_c2r_valid(
    const cufftdx::complex<float>* input_1,   // [M × (M/2+1)] complex (IFFT input)
    float*                         output,     // [W × W]        accumulator
    typename FFT::workspace_type   workspace,
    const unsigned int total_input_size,       // N_large
    const unsigned int erosion,
    const unsigned int tid_x,
    const unsigned int tid_y)
{
    using complex_type = typename FFT::value_type;
    using real_type    = typename complex_type::value_type;

    complex_type thread_data[FFT::storage_size];

    const unsigned int local_fft_id  = threadIdx.y;
    const unsigned int global_fft_id = (blockIdx.x * FFT::ffts_per_block) + local_fft_id;

    // Load data from global memory to registers
    const unsigned int offset = FFT::input_length * global_fft_id;
    constexpr unsigned int stride = FFT::stride;
    unsigned int index = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::input_ept; i++) {
        if ((i * stride + threadIdx.x) < FFT::input_length) {
            reinterpret_cast<complex_type*>(thread_data)[i] =
                reinterpret_cast<const complex_type*>(input_1)[index];
            index += stride;
        }
    }

    // Execute IFFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_memory[];
    FFT().execute(thread_data, shared_memory, workspace);

    // Save valid results — fused fftshift-undo + valid-zone check + atomicAdd
    const unsigned int tile_stride = FFT::output_length - 2 * erosion;
    const unsigned int out_offset  = FFT::output_length * global_fft_id;
    index = out_offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::output_ept; i++) {
        if ((i * stride + threadIdx.x) < FFT::output_length) {
            // col-major: FFT operates on contiguous M elements = one column.
            //   global_fft_id = column index (x), intra-FFT = row index (y).
            unsigned int in_tile_y    = index % FFT::output_length;
            unsigned int in_tile_x    = index / FFT::output_length;
            unsigned int shifted_tile_x = (in_tile_x + FFT::output_length / 2) % FFT::output_length;
            unsigned int shifted_tile_y = (in_tile_y + FFT::output_length / 2) % FFT::output_length;

            if (shifted_tile_x >= erosion && shifted_tile_x < FFT::output_length - erosion
             && shifted_tile_y >= erosion && shifted_tile_y < FFT::output_length - erosion)
            {
                unsigned int global_x = tile_stride * tid_x + shifted_tile_x;
                unsigned int global_y = tile_stride * tid_y + shifted_tile_y;

                if (global_x >= erosion && global_x < total_input_size - erosion
                 && global_y >= erosion && global_y < total_input_size - erosion)
                {
                    unsigned int out_x     = global_x - erosion;
                    unsigned int out_y     = global_y - erosion;
                    unsigned int W         = total_input_size - 2 * erosion;
                    unsigned int out_index = out_x * W + out_y;   // col-major: col*H+row

                    output[out_index] =
                              reinterpret_cast<const real_type*>(thread_data)[i];
                }
            }
            index += stride;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SrkConvWorkspace<Arch, M, EPT, FPB>
//  Pre-allocated buffers and cufftDX workspaces for srkConvOaS / srkForwardOaS.
//  Allocate once before timing loops; pass by reference to the hot functions.
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,   // FPB for row-direction (R2C/C2R)
         unsigned int FPB_COL = FPB> // FPB for col-direction (C2C fwd/inv)
struct SrkConvWorkspace {
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

    // Per-tile scratch (reused each tile)
    float* d_real_buf  = nullptr;   // [M × M]
    CT*    d_cplx_buf  = nullptr;   // [M × HALF]
    // srkForwardOaS scratch
    float* d_dIdx      = nullptr;   // [N_large × N_large]
    float* d_dIdy      = nullptr;   // [N_large × N_large]
    float* d_ux_tmp    = nullptr;   // [W × W] — only used if ux/uy not provided
    float* d_uy_tmp    = nullptr;   // [W × W]

    typename FFT_R2C::workspace_type ws_r2c;
    typename FFT_C2R::workspace_type ws_c2r;
    typename FFT_FWD::workspace_type ws_fwd;
    typename FFT_INV::workspace_type ws_inv;

    void allocate(unsigned N_large, unsigned W, cudaStream_t stream = nullptr) {
        cudaMallocAsync(&d_real_buf, sizeof(float) * M * M,                 stream);
        cudaMallocAsync(&d_cplx_buf, sizeof(CT) * M * HALF,                 stream);
        cudaMallocAsync(&d_dIdx,     sizeof(float) * N_large * N_large,     stream);
        cudaMallocAsync(&d_dIdy,     sizeof(float) * N_large * N_large,     stream);
        cudaMallocAsync(&d_ux_tmp,   sizeof(float) * W * W,                 stream);
        cudaMallocAsync(&d_uy_tmp,   sizeof(float) * W * W,                 stream);
        ws_r2c = cufftdx::make_workspace<FFT_R2C>(stream);
        ws_c2r = cufftdx::make_workspace<FFT_C2R>(stream);
        ws_fwd = cufftdx::make_workspace<FFT_FWD>(stream);
        ws_inv = cufftdx::make_workspace<FFT_INV>(stream);
        cudaFuncSetAttribute(srk_fft_rows<FFT_R2C, float, CT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FFT_R2C::shared_memory_size);
        cudaFuncSetAttribute(srk_fft_rows_c2r_valid<FFT_C2R>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FFT_C2R::shared_memory_size);
        cudaFuncSetAttribute(
            (srk_conv_cols<FFT_FWD, FFT_INV, HALF, HALF, CT>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * FFT_FWD::shared_memory_size);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  srkConvOaS<Arch, M, EPT, FPB>
//  Convolves E_pad[N_large×N_large] with pre-computed G_freq using OaS.
//  Output: out[W×W] = scale * conv(E_pad, G)  (valid region only)
//  scale = dx*dx / (M*M)  (baked in, same as forward reference code)
//  ws: pre-allocated workspace (no allocation inside this function)
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch,
         unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,
         unsigned int FPB_COL = FPB>
void srkConvOaS(
    const float* d_E_pad,          // [N_large × N_large]
    const cufftComplex* d_G_freq,  // [M × (M/2+1)] complex
    int N_large,                   // padded image size
    int W,                         // output (valid) size = N_large - 2*erosion
    int erosion,
    float dx,
    cudaStream_t stream,
    float* d_out,                  // [W × W]
    SrkConvWorkspace<Arch, M, EPT, FPB, FPB_COL>& ws)
{
    using WS      = SrkConvWorkspace<Arch, M, EPT, FPB, FPB_COL>;
    using CT      = typename WS::CT;
    using FFT_R2C = typename WS::FFT_R2C;
    using FFT_C2R = typename WS::FFT_C2R;
    using FFT_FWD = typename WS::FFT_FWD;
    using FFT_INV = typename WS::FFT_INV;
    constexpr unsigned int HALF = WS::HALF;

    const int  tile_stride = M - 2 * erosion;
    const int  num_tiles   = (W + tile_stride - 1) / tile_stride;
    const float scale      = dx * dx / (float(M) * float(M));
    const unsigned int cols_smem = 2 * FFT_FWD::shared_memory_size;

    cudaMemsetAsync(d_out, 0, sizeof(float) * W * W, stream);

    const dim3 tile_blk(32, 32);
    const dim3 tile_grd((M+31)/32, (M+31)/32);
    const dim3 rows_grd((M + FPB - 1) / FPB);
    const dim3 rows_blk = FFT_R2C::block_dim;
    const dim3 cols_grd((HALF + FPB_COL - 1) / FPB_COL);
    const dim3 cols_blk = FFT_FWD::block_dim;

    for (int ty = 0; ty < num_tiles; ++ty) {
        for (int tx = 0; tx < num_tiles; ++tx) {
            cudaMemsetAsync(ws.d_real_buf, 0, sizeof(float) * M * M, stream);

            srk_loadFFTshiftedTile<float><<<tile_grd, tile_blk, 0, stream>>>(
                d_E_pad, ws.d_real_buf,
                (unsigned)N_large, (unsigned)M, (unsigned)tile_stride,
                (unsigned)tx, (unsigned)ty);

            srk_fft_rows<FFT_R2C, float, CT><<<rows_grd, rows_blk,
                FFT_R2C::shared_memory_size, stream>>>(
                ws.d_real_buf,
                reinterpret_cast<CT*>(ws.d_cplx_buf),
                ws.ws_r2c);

            srk_conv_cols<FFT_FWD, FFT_INV, HALF, HALF, CT><<<cols_grd, cols_blk,
                          cols_smem, stream>>>(
                reinterpret_cast<CT*>(ws.d_cplx_buf),
                reinterpret_cast<const CT*>(d_G_freq),
                reinterpret_cast<CT*>(ws.d_cplx_buf),
                ws.ws_fwd, ws.ws_inv, scale);

            srk_fft_rows_c2r_valid<FFT_C2R><<<rows_grd, rows_blk,
                FFT_C2R::shared_memory_size, stream>>>(
                reinterpret_cast<const CT*>(ws.d_cplx_buf),
                d_out, ws.ws_c2r,
                (unsigned)N_large, (unsigned)erosion,
                (unsigned)tx, (unsigned)ty);
        }
    }
    cudaStreamSynchronize(stream);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Gradient helper: central difference kernels on N_large grid
// ─────────────────────────────────────────────────────────────────────────────
// col-major access:  buf[col * N + row]
__global__ void srk_grad_x(const float* I, float* dIdx, int N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= N || c >= N) return;
    float left  = (c > 0)     ? I[(c-1)*N + r] : 0.f;
    float right = (c < N-1)   ? I[(c+1)*N + r] : 0.f;
    dIdx[c*N + r] = (right - left) * 0.5f;
}

__global__ void srk_grad_y(const float* I, float* dIdy, int N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= N || c >= N) return;
    float up   = (r > 0)     ? I[c*N + (r-1)] : 0.f;
    float down = (r < N-1)   ? I[c*N + (r+1)] : 0.f;
    dIdy[c*N + r] = (down - up) * 0.5f;
}

// R[oy,ox] = ux[oy,ox]*dIdx[gy,gx] + uy[oy,ox]*dIdy[gy,gx]
__global__ void srk_fwd_combine(
    const float* ux, const float* uy,
    const float* dIdx, const float* dIdy,
    float* R,
    int W, int N_large, int erosion)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W || oy >= W) return;
    int gx = ox + erosion, gy = oy + erosion;
    R[ox*W + oy] = ux[ox*W + oy] * dIdx[gx*N_large + gy]
                 + uy[ox*W + oy] * dIdy[gx*N_large + gy];   // col-major
}

// ─────────────────────────────────────────────────────────────────────────────
//  srkForwardOaS<Arch, M>
//  Full forward:  R[W×W] = ux[W×W] * dI/dx[valid] + uy[W×W] * dI/dy[valid]
//  where ux = srkConvOaS(E, Gx), uy = srkConvOaS(E, Gy)  (both include scale)
//  d_ux and d_uy (optional): if non-null, returned for use in adjointWrtI
// ─────────────────────────────────────────────────────────────────────────────
template<unsigned int Arch, unsigned int M,
         unsigned int EPT     = 16,
         unsigned int FPB     = 8,
         unsigned int FPB_COL = FPB>
void srkForwardOaS(
    const float*        d_E_pad,    // [N_large × N_large]
    const float*        d_I_pad,    // [N_large × N_large]
    const cufftComplex* d_Gx_freq,  // [M × (M/2+1)]
    const cufftComplex* d_Gy_freq,  // [M × (M/2+1)]
    int N_large, int W, int erosion,
    float dx,
    cudaStream_t stream,
    float* d_R,                     // output [W × W]
    SrkConvWorkspace<Arch, M, EPT, FPB, FPB_COL>& ws,
    float* d_ux = nullptr,          // optional [W × W] (uses ws.d_ux_tmp if null)
    float* d_uy = nullptr)          // optional [W × W] (uses ws.d_uy_tmp if null)
{
    if (!d_ux) d_ux = ws.d_ux_tmp;
    if (!d_uy) d_uy = ws.d_uy_tmp;

    srkConvOaS<Arch, M, EPT, FPB>(d_E_pad, d_Gx_freq, N_large, W, erosion, dx, stream, d_ux, ws);
    srkConvOaS<Arch, M, EPT, FPB>(d_E_pad, d_Gy_freq, N_large, W, erosion, dx, stream, d_uy, ws);

    dim3 gblk(16, 16);
    dim3 ggrd((N_large+15)/16, (N_large+15)/16);
    srk_grad_x<<<ggrd, gblk, 0, stream>>>(d_I_pad, ws.d_dIdx, N_large);
    srk_grad_y<<<ggrd, gblk, 0, stream>>>(d_I_pad, ws.d_dIdy, N_large);

    dim3 rblk(16, 16), rgrd((W+15)/16, (W+15)/16);
    srk_fwd_combine<<<rgrd, rblk, 0, stream>>>(
        d_ux, d_uy, ws.d_dIdx, ws.d_dIdy, d_R, W, N_large, erosion);

    cudaStreamSynchronize(stream);
}

