// main.cu — shrinkage_adjoint_test
//
// Task 1: Validate srkForwardOaS against a CPU direct-convolution reference.
//         Pass criterion: relative max error < 1e-3 (float32 FFT precision).
//
// Task 2: AdjointTester for adjointWrtI and adjointWrtE.
//         Three seeds each; pass criterion: rel_err_max < 1e-4.
//
// Forward operator:
//   R[W×W] = scale * (conv(E,Gx)*dI/dx + conv(E,Gy)*dI/dy)
//   where scale = dx² / M²  (absorbed into srkConvOaS)
// ─────────────────────────────────────────────────────────────────

#include "srk_adjoint.cuh"
#include <stTestCore/st_test_core.h>
#include <patternUtil/test_pattern.h>

#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <filesystem>

using namespace st::util;

// ── Compile-time parameters (template / architecture constants) ───
static constexpr unsigned ARCH    = 860;
static constexpr unsigned M_TILE  = 1024;
static constexpr unsigned M_TILE2 = 2048;  // larger tile for big-image test
static constexpr float    DX      = 1.0f;
static constexpr float    KRN_C_W = 3.0f;  // Gaussian truncation: R_w = c_w·sigma
static constexpr int      FREQ_N  = M_TILE * (M_TILE / 2 + 1);  // complex elems

// Runtime-configurable values are parsed from argv in main().
static float get_arg_float(int argc, char** argv, const char* key, float def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key) return std::stof(argv[i + 1]);
    return def;
}
static int get_arg_int(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == key) return std::stoi(argv[i + 1]);
    return def;
}

// ── Gaussian kernel (normalized, sum=1) ──────────────────────────
// Col-major layout: g[col * kh + row], i.e. g[x*kh + y]
static std::vector<float> make_gaussian_2d(int kw, int kh, float sigma) {
    std::vector<float> g(kw * kh, 0.f);
    int cx = kw / 2, cy = kh / 2;
    float sum = 0.f;
    for (int y = 0; y < kh; ++y)
        for (int x = 0; x < kw; ++x) {
            float dx = float(x - cx), dy = float(y - cy);
            float v = std::exp(-(dx*dx + dy*dy) / (2.f * sigma * sigma));
            g[x*kh + y] = v;
            sum += v;
        }
    for (auto& v : g) v /= sum;
    return g;
}

// ── CPU reference: direct 2D linear convolution at valid positions ─
// Col-major layout throughout: arr[col * nrows + row]
// Returns result[W×W] = scale * sum_{ky,kx} E_pad[gy-ky, gx-kx] * G[ky+cy, kx+cx]
static std::vector<float> cpu_conv_valid(
    const std::vector<float>& h_E, int N,
    const std::vector<float>& h_G, int kw, int kh,
    float scale, int W, int erosion)
{
    std::vector<float> out(W * W, 0.f);
    int cx = kw / 2, cy = kh / 2;
    for (int oy = 0; oy < W; ++oy)
        for (int ox = 0; ox < W; ++ox) {
            int gy = oy + erosion, gx = ox + erosion;
            float sum = 0.f;
            for (int ky = -cy; ky <= cy; ++ky)
                for (int kx = -cx; kx <= cx; ++kx) {
                    int y = gy - ky, x = gx - kx;
                    if (y >= 0 && y < N && x >= 0 && x < N)
                        sum += h_E[x*N + y] * h_G[(kx+cx)*kh + (ky+cy)];
                }
            out[ox*W + oy] = scale * sum;
        }
    return out;
}

// ── Tiny kernel: R = ux * D_x(I) + uy * D_y(I)  at valid positions ─
// Col-major: arr[col * nrows + row]
__global__ void g_fwd_wrtI_kernel(
    const float* d_I, const float* d_ux, const float* d_uy,
    float* d_R, int W, int N, int e)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W || oy >= W) return;
    int gx = ox + e, gy = oy + e;
    float dix = 0.f, diy = 0.f;
    if (gx > 0 && gx < N-1) dix = (d_I[(gx+1)*N + gy] - d_I[(gx-1)*N + gy]) * 0.5f;
    if (gy > 0 && gy < N-1) diy = (d_I[gx*N + (gy+1)] - d_I[gx*N + (gy-1)]) * 0.5f;
    d_R[ox*W + oy] = d_ux[ox*W + oy] * dix + d_uy[ox*W + oy] * diy;
}

// ── Per-(M, FPB_COL) workspace context (templated so multiple configs coexist) ──
template<unsigned int M, unsigned int FPB_COL = 8>
struct SrkTestCtx {
    static inline SrkConvWorkspace<ARCH, M, 16, 8, FPB_COL>* fwd_ws = nullptr;
    static inline SrkAdjWorkspace <ARCH, M, 16, 8, FPB_COL>* adj_ws = nullptr;
};

// ── Wrapper functions for AdjointTester ──────────────────────────
// wrtI wrappers: no FFT, so M-independent and shared across all configs.

// Forward wrt I:  (I_pad, ux_fixed, uy_fixed) → R
static void srk_fwd_wrtI_fn(
    float* d_R,        // [W×W]         output
    float* d_I_pad,    // [N×N]         perturbed input
    int*   p_N, int* p_W, int* p_e,
    float* d_ux,       // [W×W]         fixed param
    float* d_uy)       // [W×W]         fixed param
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    dim3 blk(16, 16), grd((Wv+15)/16, (Wv+15)/16);
    g_fwd_wrtI_kernel<<<grd, blk>>>(d_I_pad, d_ux, d_uy, d_R, Wv, N, e);
    cudaDeviceSynchronize();
}

// Adjoint wrt I: dL/dI = D_x^T(ux·v) + D_y^T(uy·v)  — trivial kernel, no FFT
static void srk_adj_wrtI_fn(
    float* d_dLdI,     // [N×N]         output
    float* d_v,        // [W×W]         dqdout
    int*   p_N, int* p_W, int* p_e,
    float* d_ux,       // [W×W]         fixed
    float* d_uy,       // [W×W]         fixed
    float* /*d_I_pad*/, float* /*d_Gx_f*/, float* /*d_Gy_f*/, float* /*p_dx*/)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    adjointWrtI(d_v, d_ux, d_uy, Wv, N, e, nullptr, d_dLdI);
}

constexpr AdaptedFunc fwd_wrtI_adapted =
    FuncAdapter<srk_fwd_wrtI_fn,
        float*, float*, int*, int*, int*, float*, float*>::call;

constexpr AdaptedFunc adj_wrtI_adapted =
    FuncAdapter<srk_adj_wrtI_fn,
        float*, float*, int*, int*, int*, float*, float*, float*, float*, float*, float*>::call;

// wrtE wrappers: OaS FFT. Templated on (M, FPB_COL) so multiple configs coexist.

template<unsigned int M, unsigned int FPB_COL = 8>
static void srk_fwd_wrtE_fn(
    float* d_R,          // [W×W]        output
    float* d_E_pad,      // [N×N]        perturbed input
    float* d_I_pad,      // [N×N]        fixed
    float* d_Gx_f,       // [FREQ_N*2 floats]  fixed
    float* d_Gy_f,       // [FREQ_N*2 floats]  fixed
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    float dx = *p_dx;
    srkForwardOaS<ARCH, M, 16, 8, FPB_COL>(
        d_E_pad, d_I_pad,
        reinterpret_cast<const cufftComplex*>(d_Gx_f),
        reinterpret_cast<const cufftComplex*>(d_Gy_f),
        N, Wv, e, dx, /*stream=*/nullptr, d_R, *SrkTestCtx<M, FPB_COL>::fwd_ws);
}

template<unsigned int M, unsigned int FPB_COL = 8>
static void srk_adj_wrtE_fn(
    float* d_dLdE,       // [N×N]        output
    float* d_v,          // [W×W]        dqdout
    float* d_I_pad,      // [N×N]        fixed
    float* d_Gx_f,       // fixed
    float* d_Gy_f,       // fixed
    int* p_N, int* p_W, int* p_e, float* p_dx,
    float* /*d_ux*/, float* /*d_uy*/)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    float dx = *p_dx;
    adjointWrtE<ARCH, M, 16, 8, FPB_COL>(
        d_v, d_I_pad,
        reinterpret_cast<const cufftComplex*>(d_Gx_f),
        reinterpret_cast<const cufftComplex*>(d_Gy_f),
        N, Wv, e, dx, nullptr, d_dLdE, *SrkTestCtx<M, FPB_COL>::adj_ws);
}

template<unsigned int M, unsigned int FPB_COL = 8>
constexpr AdaptedFunc fwd_wrtE_adapted_v =
    FuncAdapter<srk_fwd_wrtE_fn<M, FPB_COL>,
        float*, float*, float*, float*, float*, int*, int*, int*, float*>::call;

template<unsigned int M, unsigned int FPB_COL = 8>
constexpr AdaptedFunc adj_wrtE_adapted_v =
    FuncAdapter<srk_adj_wrtE_fn<M, FPB_COL>,
        float*, float*, float*, float*, float*, int*, int*, int*, float*, float*, float*>::call;

// ── CPU brute-force adjoint wrt E ─────────────────────────────────
// dLdE[row_E, col_E] = sum_{oy,ox} v[oy,ox] *
//   (Gx[(oy+e-row_E)+cy, (ox+e-col_E)+cx] * dIdx[oy+e,ox+e] +
//    Gy[...]                              * dIdy[oy+e,ox+e])
static std::vector<float> cpu_adj_wrtE(
    const std::vector<float>& h_v, int W,
    const std::vector<float>& h_Gx, const std::vector<float>& h_Gy,
    int kw, int kh,
    const std::vector<float>& h_I, int N, int erosion)
{
    int cx = kw/2, cy = kh/2;
    // Compute dIdx, dIdy from h_I  (col-major: arr[col*N + row])
    std::vector<float> dIdx(N*N, 0.f), dIdy(N*N, 0.f);
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            float l = (c>0)   ? h_I[(c-1)*N + r] : 0.f;
            float ri = (c<N-1) ? h_I[(c+1)*N + r] : 0.f;
            dIdx[c*N + r] = (ri - l) * 0.5f;
            float u = (r>0)   ? h_I[c*N + (r-1)] : 0.f;
            float d = (r<N-1) ? h_I[c*N + (r+1)] : 0.f;
            dIdy[c*N + r] = (d - u) * 0.5f;
        }
    }
    std::vector<float> out(N*N, 0.f);
    for (int row_E = 0; row_E < N; ++row_E) {
        for (int col_E = 0; col_E < N; ++col_E) {
            float sum = 0.f;
            for (int oy = 0; oy < W; ++oy) {
                int gy = oy + erosion;
                int ky = gy - row_E;   // row offset: gy = row_E + ky
                if (ky < -cy || ky > cy) continue;
                for (int ox = 0; ox < W; ++ox) {
                    int gx = ox + erosion;
                    int kx = gx - col_E;
                    if (kx < -cx || kx > cx) continue;
                    float gx_v = h_Gx[(kx+cx)*kh + (ky+cy)];
                    float gy_v = h_Gy[(kx+cx)*kh + (ky+cy)];
                    sum += h_v[ox*W + oy] * (gx_v * dIdx[gx*N + gy] + gy_v * dIdy[gx*N + gy]);
                }
            }
            out[col_E*N + row_E] = sum;
        }
    }
    return out;
}

// ── main ──────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    const float SIGMA   = get_arg_float(argc, argv, "--sigma",  5.f);
    const int   W_OUT   = get_arg_int  (argc, argv, "--w_out",  256);
    const float R_W     = KRN_C_W * SIGMA;
    const int   EROSION = (int)(R_W / DX) + 1;
    const int   KW      = 2 * (int)(R_W / DX) + 1;
    const int   KH      = KW;
    const int   N_LARGE = W_OUT + 2 * EROSION;
    printf("sigma=%.1f  w_out=%d  R_w=%.1f  kw=%d  erosion=%d  N_large=%d\n",
           SIGMA, W_OUT, R_W, KW, EROSION, N_LARGE);
    cudaSetDevice(0);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);

    // Smooth random images via Gaussian process (length scale = 32 px).
    // GpGenerator internally pads to power-of-2 so any N_LARGE works.
    auto h_E = generate_pattern(TestPatternConfig::gaussian_process(N_LARGE, 32.f, /*seed=*/42));
    auto h_I = generate_pattern(TestPatternConfig::gaussian_process(N_LARGE, 32.f, /*seed=*/43));

    // Gaussian kernels (even, normalized)
    auto h_Gx = make_gaussian_2d(KW, KH, SIGMA);
    auto h_Gy = make_gaussian_2d(KW, KH, SIGMA);  // same isotropic Gaussian

    // Upload images
    float *d_E = nullptr, *d_I = nullptr;
    cudaMalloc(&d_E, sizeof(float) * N_LARGE * N_LARGE);
    cudaMalloc(&d_I, sizeof(float) * N_LARGE * N_LARGE);
    cudaMemcpy(d_E, h_E.data(), sizeof(float)*N_LARGE*N_LARGE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, h_I.data(), sizeof(float)*N_LARGE*N_LARGE, cudaMemcpyHostToDevice);

    // Precompute kernel frequency representations (M_TILE × (M_TILE/2+1) complex)
    cufftComplex *d_Gx_freq = nullptr, *d_Gy_freq = nullptr;
    cudaMalloc(&d_Gx_freq, sizeof(cufftComplex) * FREQ_N);
    cudaMalloc(&d_Gy_freq, sizeof(cufftComplex) * FREQ_N);
    srkPrecomputeKernelFreq(h_Gx, KW, KH, M_TILE, d_Gx_freq, /*stream=*/nullptr);
    srkPrecomputeKernelFreq(h_Gy, KW, KH, M_TILE, d_Gy_freq, /*stream=*/nullptr);

    // Download freq buffers for AdjointTester init_fn
    std::vector<float> h_Gx_freq(FREQ_N * 2), h_Gy_freq(FREQ_N * 2);
    cudaMemcpy(h_Gx_freq.data(), d_Gx_freq, sizeof(float)*FREQ_N*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Gy_freq.data(), d_Gy_freq, sizeof(float)*FREQ_N*2, cudaMemcpyDeviceToHost);

    // ─────────────────────────────────────────────────────────────
    //  Allocate workspaces (once; reused by all timed functions)
    // ─────────────────────────────────────────────────────────────
    static SrkConvWorkspace<ARCH, M_TILE> fwd_ws;
    static SrkAdjWorkspace <ARCH, M_TILE> adj_ws;
    fwd_ws.allocate(N_LARGE, W_OUT);
    adj_ws.allocate(N_LARGE);
    SrkTestCtx<M_TILE>::fwd_ws = &fwd_ws;
    SrkTestCtx<M_TILE>::adj_ws = &adj_ws;

    // ─────────────────────────────────────────────────────────────
    //  Task 1: Forward validation
    //  Compare srkForwardOaS output vs CPU direct convolution reference.
    // ─────────────────────────────────────────────────────────────
    printf("\n=== Task 1: Forward Validation ===\n");
    bool task1_pass = false;
    {
        // CPU reference:
        //   ux_ref = dx² * direct_conv(E_pad, Gx)   (scale=DX*DX)
        //   uy_ref = dx² * direct_conv(E_pad, Gy)
        //   R_ref = ux_ref * dIdx + uy_ref * dIdy
        //
        // Note: srkConvOaS uses scale = dx²/M² but the M² from the unnormalized
        // cufftDX 2D IFFT cancels, giving the same dx² * direct_conv result.
        const float cpu_scale = DX * DX;
        auto h_ux_ref = cpu_conv_valid(h_E, N_LARGE, h_Gx, KW, KH, cpu_scale, W_OUT, EROSION);
        auto h_uy_ref = cpu_conv_valid(h_E, N_LARGE, h_Gy, KW, KH, cpu_scale, W_OUT, EROSION);

        std::vector<float> h_R_ref(W_OUT * W_OUT, 0.f);
        for (int oy = 0; oy < W_OUT; ++oy) {
            for (int ox = 0; ox < W_OUT; ++ox) {
                int gx = ox + EROSION, gy = oy + EROSION;
                float dix = 0.f, diy = 0.f;
                if (gx > 0 && gx < N_LARGE-1)
                    dix = (h_I[(gx+1)*N_LARGE + gy] - h_I[(gx-1)*N_LARGE + gy]) * 0.5f;
                if (gy > 0 && gy < N_LARGE-1)
                    diy = (h_I[gx*N_LARGE + (gy+1)] - h_I[gx*N_LARGE + (gy-1)]) * 0.5f;
                h_R_ref[ox*W_OUT + oy] = h_ux_ref[ox*W_OUT + oy] * dix
                                       + h_uy_ref[ox*W_OUT + oy] * diy;
            }
        }

        // GPU forward
        float* d_R = nullptr;
        cudaMalloc(&d_R, sizeof(float) * W_OUT * W_OUT);

        srkForwardOaS<ARCH, M_TILE>(
            d_E, d_I, d_Gx_freq, d_Gy_freq,
            N_LARGE, W_OUT, EROSION, DX, /*stream=*/nullptr, d_R, fwd_ws);

        std::vector<float> h_R(W_OUT * W_OUT);
        cudaMemcpy(h_R.data(), d_R, sizeof(float)*W_OUT*W_OUT, cudaMemcpyDeviceToHost);
        cudaFree(d_R);

        // Metrics
        float max_abs = 0.f, max_ref = 0.f;
        float E_abs = 0.f, I_abs = 0.f, R_abs = 0.f;
        for (int i = 0; i < N_LARGE * N_LARGE; ++i) {
            E_abs = std::max(E_abs, std::abs(h_E[i]));
            I_abs = std::max(I_abs, std::abs(h_I[i]));
        }
        for (int i = 0; i < W_OUT * W_OUT; ++i) {
            max_abs = std::max(max_abs, std::abs(h_R[i] - h_R_ref[i]));
            max_ref = std::max(max_ref, std::abs(h_R_ref[i]));
            R_abs   = std::max(R_abs,   std::abs(h_R[i]));
        }
        float rel = (max_ref > 0.f) ? max_abs / max_ref : 0.f;
        task1_pass = (rel < 1e-3f);
        printf("  Input scale: E_max=%.3e  I_max=%.3e  R_max=%.3e\n",
               E_abs, I_abs, R_abs);
        printf("  max_abs_err=%.2e  max_ref=%.2e  rel=%.2e  → %s\n",
               max_abs, max_ref, rel, task1_pass ? "PASS" : "FAIL");

        // Dump inputs and forward output
        std::filesystem::create_directories("dump/forward");
        dump_buffer("dump/forward/E_pad.txt",
            (void*)h_E.data(), MemSpace::HOST, DType::FLOAT32,
            Dims::make_2d(N_LARGE, N_LARGE), "E_pad");
        dump_buffer("dump/forward/I_pad.txt",
            (void*)h_I.data(), MemSpace::HOST, DType::FLOAT32,
            Dims::make_2d(N_LARGE, N_LARGE), "I_pad");
        dump_buffer("dump/forward/R_gpu.txt",
            (void*)h_R.data(), MemSpace::HOST, DType::FLOAT32,
            Dims::make_2d(W_OUT, W_OUT), "R_gpu");
    }

    // ─────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────
    //  Precompute ux, uy from E (fixed params for adjointWrtI test)
    // ─────────────────────────────────────────────────────────────
    float *d_ux = nullptr, *d_uy = nullptr;
    cudaMalloc(&d_ux, sizeof(float) * W_OUT * W_OUT);
    cudaMalloc(&d_uy, sizeof(float) * W_OUT * W_OUT);
    srkConvOaS<ARCH, M_TILE>(d_E, d_Gx_freq, N_LARGE, W_OUT, EROSION, DX, nullptr, d_ux, fwd_ws);
    srkConvOaS<ARCH, M_TILE>(d_E, d_Gy_freq, N_LARGE, W_OUT, EROSION, DX, nullptr, d_uy, fwd_ws);

    std::vector<float> h_ux(W_OUT * W_OUT), h_uy(W_OUT * W_OUT);
    cudaMemcpy(h_ux.data(), d_ux, sizeof(float)*W_OUT*W_OUT, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), d_uy, sizeof(float)*W_OUT*W_OUT, cudaMemcpyDeviceToHost);

    // ─────────────────────────────────────────────────────────────
    //  Task 2: AdjointTester
    // ─────────────────────────────────────────────────────────────
    // (Diagnostic block removed after validation)
    if (false) {
        // Random v (W×W)
        std::vector<float> h_v(W_OUT * W_OUT);
        for (auto& x : h_v) x = nd(rng);

        // CPU adjoint
        auto h_dLdE_cpu = cpu_adj_wrtE(h_v, W_OUT, h_Gx, h_Gy, KW, KH, h_I, N_LARGE, EROSION);

        // GPU adjoint
        float *d_v_diag = nullptr, *d_dLdE_gpu = nullptr;
        cudaMalloc(&d_v_diag,   sizeof(float)*W_OUT*W_OUT);
        cudaMalloc(&d_dLdE_gpu, sizeof(float)*N_LARGE*N_LARGE);
        cudaMemcpy(d_v_diag, h_v.data(), sizeof(float)*W_OUT*W_OUT, cudaMemcpyHostToDevice);
        adjointWrtE<ARCH, M_TILE>(d_v_diag, d_I, d_Gx_freq, d_Gy_freq,
                                   N_LARGE, W_OUT, EROSION, DX, nullptr, d_dLdE_gpu, adj_ws);
        std::vector<float> h_dLdE_gpu(N_LARGE * N_LARGE);
        cudaMemcpy(h_dLdE_gpu.data(), d_dLdE_gpu, sizeof(float)*N_LARGE*N_LARGE, cudaMemcpyDeviceToHost);
        cudaFree(d_v_diag);
        cudaFree(d_dLdE_gpu);

        // Compare  (col-major: arr[col*N + row])
        float max_rel = 0.f;
        int worst_r = 0, worst_c = 0;
        for (int r = 0; r < N_LARGE; ++r) {
            for (int c = 0; c < N_LARGE; ++c) {
                float cpu_v = h_dLdE_cpu[c*N_LARGE+r];
                float gpu_v = h_dLdE_gpu[c*N_LARGE+r];
                float err = std::abs(cpu_v - gpu_v);
                float denom = std::max(std::abs(cpu_v), std::abs(gpu_v));
                float rel = (denom > 1e-9f) ? err/denom : 0.f;
                if (rel > max_rel) { max_rel = rel; worst_r = r; worst_c = c; }
            }
        }
        printf("\n[Diagnostic] CPU vs GPU adjointWrtE: max_rel=%.4e at (%d,%d)\n", max_rel, worst_r, worst_c);
        printf("  cpu=%.6e  gpu=%.6e\n",
               h_dLdE_cpu[worst_c*N_LARGE+worst_r],
               h_dLdE_gpu[worst_c*N_LARGE+worst_r]);

        // Direct numerical derivative check: perturb E at (row=192, col=2) manually
        {
            int test_row = 192, test_col = 2;
            float eps = 1e-3f;

            // Perturb E: E_plus  (col-major: arr[col*N + row])
            std::vector<float> h_E_plus = h_E, h_E_minus = h_E;
            h_E_plus [test_col*N_LARGE+test_row] += eps;
            h_E_minus[test_col*N_LARGE+test_row] -= eps;

            auto run_fwd = [&](const std::vector<float>& h_Epert) {
                float* d_Ep = nullptr, *d_Rp = nullptr;
                cudaMalloc(&d_Ep, sizeof(float)*N_LARGE*N_LARGE);
                cudaMalloc(&d_Rp, sizeof(float)*W_OUT*W_OUT);
                cudaMemcpy(d_Ep, h_Epert.data(), sizeof(float)*N_LARGE*N_LARGE, cudaMemcpyHostToDevice);
                srkForwardOaS<ARCH, M_TILE>(d_Ep, d_I, d_Gx_freq, d_Gy_freq,
                                             N_LARGE, W_OUT, EROSION, DX, nullptr, d_Rp, fwd_ws);
                std::vector<float> h_Rp(W_OUT*W_OUT);
                cudaMemcpy(h_Rp.data(), d_Rp, sizeof(float)*W_OUT*W_OUT, cudaMemcpyDeviceToHost);
                cudaFree(d_Ep); cudaFree(d_Rp);
                return h_Rp;
            };

            auto h_Rp = run_fwd(h_E_plus);
            auto h_Rm = run_fwd(h_E_minus);

            // Numerical dot product: v · (R_plus - R_minus) / (2*eps)
            float num_dot = 0.f;
            for (int i = 0; i < W_OUT*W_OUT; ++i)
                num_dot += h_v[i] * (h_Rp[i] - h_Rm[i]) / (2.f*eps);

            // CPU brute-force forward for the same perturbation (col-major)
            std::vector<float> h_dIdx_f(N_LARGE*N_LARGE, 0.f), h_dIdy_f(N_LARGE*N_LARGE, 0.f);
            for (int r = 0; r < N_LARGE; ++r) for (int c = 0; c < N_LARGE; ++c) {
                h_dIdx_f[c*N_LARGE+r] = ((c>0     ? h_I[(c-1)*N_LARGE+r] : 0.f) - (c<N_LARGE-1 ? h_I[(c+1)*N_LARGE+r] : 0.f)) * (-0.5f);
                h_dIdy_f[c*N_LARGE+r] = ((r>0     ? h_I[c*N_LARGE+(r-1)] : 0.f) - (r<N_LARGE-1 ? h_I[c*N_LARGE+(r+1)] : 0.f)) * (-0.5f);
            }
            // d(ux[oy,ox])/dE[test_row,test_col] = Gx[(oy+e-test_row)+cy, (ox+e-test_col)+cx]
            float cpu_num_dot = 0.f;
            {
                int cx = KW/2, cy = KH/2;
                for (int oy = 0; oy < W_OUT; ++oy) {
                    int ky = (oy+EROSION) - test_row;
                    if (ky < -cy || ky > cy) continue;
                    for (int ox = 0; ox < W_OUT; ++ox) {
                        int kx = (ox+EROSION) - test_col;
                        if (kx < -cx || kx > cx) continue;
                        float gv = h_Gx[(kx+cx)*KH+(ky+cy)];
                        float dix = (h_dIdx_f[(ox+EROSION)*N_LARGE+(oy+EROSION)]);
                        float diy = (h_dIdy_f[(ox+EROSION)*N_LARGE+(oy+EROSION)]);
                        // Forward: R[oy,ox] changes by gv*(dix+diy) per unit E perturbation
                        cpu_num_dot += h_v[ox*W_OUT+oy] * gv * (dix + diy);
                    }
                }
            }

            float ana_val = h_dLdE_gpu[test_col*N_LARGE+test_row];
            float cpu_adj_val = h_dLdE_cpu[test_col*N_LARGE+test_row];
            printf("[Diagnostic] Manual num-diff at E[%d,%d]:\n", test_row, test_col);
            printf("  GPU num  = %.6e  (from perturbing GPU fwd)\n", num_dot);
            printf("  CPU num  = %.6e  (brute-force chain rule)\n", cpu_num_dot);
            printf("  GPU adj  = %.6e  (adjointWrtE output)\n", ana_val);
            printf("  CPU adj  = %.6e  (cpu_adj_wrtE output)\n", cpu_adj_val);

            // Per-position breakdown: show R difference at each affected output (col-major)
            {
                int cx = KW/2, cy = KH/2;
                printf("[Diagnostic] Per-position breakdown (oy,ox, gpu_diff, chain_rule_diff):\n");
                float sum_gpu = 0.f, sum_chain = 0.f;
                for (int oy = 0; oy < W_OUT; ++oy) {
                    int ky = (oy+EROSION) - test_row;
                    if (ky < -cy || ky > cy) continue;
                    for (int ox = 0; ox < W_OUT; ++ox) {
                        int kx = (ox+EROSION) - test_col;
                        if (kx < -cx || kx > cx) continue;
                        float gpu_diff = (h_Rp[ox*W_OUT+oy] - h_Rm[ox*W_OUT+oy]) / (2.f*eps);
                        float gv = h_Gx[(kx+cx)*KH+(ky+cy)];
                        float dix = h_dIdx_f[(ox+EROSION)*N_LARGE+(oy+EROSION)];
                        float diy = h_dIdy_f[(ox+EROSION)*N_LARGE+(oy+EROSION)];
                        float chain_diff = gv * (dix + diy);
                        printf("  (%d,%d) gpu=%.4e chain=%.4e ratio=%.3f gv=%.4e dIdx=%.4e dIdy=%.4e\n",
                               oy, ox, gpu_diff, chain_diff,
                               (std::abs(chain_diff)>1e-10f ? gpu_diff/chain_diff : 0.f),
                               gv, dix, diy);
                        sum_gpu   += h_v[ox*W_OUT+oy] * gpu_diff;
                        sum_chain += h_v[ox*W_OUT+oy] * chain_diff;
                    }
                }
                printf("  sum*v: gpu=%.6e  chain=%.6e\n", sum_gpu, sum_chain);
            }
            // Show top leakage positions (where Rp-Rm is nonzero outside expected support)
            {
                int cx = KW/2, cy = KH/2;
                printf("[Diagnostic] Leakage at unexpected positions (top 20):\n");
                std::vector<std::pair<float,int>> leaks;
                for (int oy = 0; oy < W_OUT; ++oy) {
                    int ky = (oy+EROSION) - test_row;
                    for (int ox = 0; ox < W_OUT; ++ox) {
                        int kx = (ox+EROSION) - test_col;
                        bool expected = (ky >= -cy && ky <= cy && kx >= -cx && kx <= cx);
                        float diff = (h_Rp[ox*W_OUT+oy] - h_Rm[ox*W_OUT+oy]) / (2.f*eps);
                        if (!expected && std::abs(diff) > 1e-9f)
                            leaks.push_back({std::abs(diff), ox*W_OUT+oy});
                    }
                }
                std::sort(leaks.begin(), leaks.end(), [](auto& a, auto& b){ return a.first > b.first; });
                for (int i = 0; i < std::min((int)leaks.size(), 20); ++i) {
                    int idx = leaks[i].second;
                    int ox = idx/W_OUT, oy = idx%W_OUT;
                    float diff = (h_Rp[ox*W_OUT+oy] - h_Rm[ox*W_OUT+oy]) / (2.f*eps);
                    printf("  (%d,%d) diff=%.4e  v=%.4e  contribution=%.4e\n",
                           oy, ox, diff, h_v[ox*W_OUT+oy], h_v[ox*W_OUT+oy]*diff);
                }
            }
        }
    }

    printf("\n=== Task 2: Adjoint Tests (small, N=286, M=1024) ===\n");

    int   p_N  = N_LARGE;
    int   p_W  = W_OUT;
    int   p_e  = EROSION;
    float p_dx = DX;

    AdjointTester tester;       // small tests
    AdjointTester tester_big;   // large tests, run separately after small

    // ── 3a: adjointWrtI  (3 seeds) ────────────────────────────────
    // F_I: I_pad[N×N] → R[W×W] = ux * D_x(I) + uy * D_y(I)
    // F_I^T: v[W×W]   → dL/dI[N×N]
    for (int seed = 1; seed <= 3; ++seed) {
        std::vector<ArgDescriptor> fwd_args = {
            // idx 0: output R
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "R"),
            // idx 1: perturbed input I_pad
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "I_pad"),
            // idx 2,3,4: scalar params
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "erosion"),
            // idx 5,6: fixed ux, uy
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "ux"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "uy"),
        };

        // adj_args for fused adjointSRK (wrtI output):
        // idx 0: dLdI, idx 1: v, idx 2-4: scalars, idx 5-6: ux/uy,
        // idx 7: I_pad, idx 8-9: Gx/Gy_freq, idx 10: dx
        std::vector<ArgDescriptor> adj_args = {
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "dLdI"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "v"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "erosion"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "ux"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "uy"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "I_pad"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2), "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2), "Gy_freq"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1), "dx"),
        };

        auto wrtI_fwd_init = [&](void* ptr, const ArgDescriptor& /*desc*/, size_t idx) {
            switch (idx) {
                case 1: memcpy(ptr, h_I.data(),  sizeof(float)*N_LARGE*N_LARGE); break;
                case 2: memcpy(ptr, &p_N,  sizeof(int));   break;
                case 3: memcpy(ptr, &p_W,  sizeof(int));   break;
                case 4: memcpy(ptr, &p_e,  sizeof(int));   break;
                case 5: memcpy(ptr, h_ux.data(), sizeof(float)*W_OUT*W_OUT); break;
                case 6: memcpy(ptr, h_uy.data(), sizeof(float)*W_OUT*W_OUT); break;
                default: break;
            }
        };

        auto wrtI_adj_init = [&](void* ptr, const ArgDescriptor& /*desc*/, size_t idx) {
            switch (idx) {
                case 2:  memcpy(ptr, &p_N,  sizeof(int));   break;
                case 3:  memcpy(ptr, &p_W,  sizeof(int));   break;
                case 4:  memcpy(ptr, &p_e,  sizeof(int));   break;
                case 5:  memcpy(ptr, h_ux.data(), sizeof(float)*W_OUT*W_OUT); break;
                case 6:  memcpy(ptr, h_uy.data(), sizeof(float)*W_OUT*W_OUT); break;
                case 7:  memcpy(ptr, h_I.data(),  sizeof(float)*N_LARGE*N_LARGE); break;
                case 8:  memcpy(ptr, h_Gx_freq.data(), sizeof(float)*FREQ_N*2); break;
                case 9:  memcpy(ptr, h_Gy_freq.data(), sizeof(float)*FREQ_N*2); break;
                case 10: memcpy(ptr, &p_dx, sizeof(float)); break;
                default: break;
            }
        };

        tester.register_adjoint(
            "adjointWrtI seed=" + std::to_string(seed),
            fwd_wrtI_adapted,  fwd_args,
            /*fwd_out_idx=*/     0,
            adj_wrtI_adapted,  adj_args,
            /*adj_out_idx=*/     0,
            /*adj_dqdout_idx=*/  1,
            /*fwd_perturb_idx=*/ 1,
            /*adj_input_idx=*/   SIZE_MAX,
            AdjointConfig{ N_LARGE*N_LARGE, 0.1, 1e-5, 1e-3, (uint64_t)seed },
            {},              // PerfConfig
            wrtI_fwd_init,   // init_fn
            wrtI_adj_init,   // adj_init_fn
            {},              // GpConfig
            (seed == 1) ? "wrtI_seed1" : ""   // dump seed=1 only
        );
    }

    // ── 3b: adjointWrtE  (3 seeds) ────────────────────────────────
    // F_E: E_pad[N×N] → R[W×W]
    // F_E^T: v[W×W]   → dL/dE[N×N]
    for (int seed = 1; seed <= 3; ++seed) {
        std::vector<ArgDescriptor> fwd_args = {
            // idx 0: output R
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "R"),
            // idx 1: perturbed input E_pad
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "E_pad"),
            // idx 2: fixed I_pad
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "I_pad"),
            // idx 3,4: kernel freq buffers
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2), "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2), "Gy_freq"),
            // idx 5,6,7,8: scalars
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "erosion"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1), "dx"),
        };

        // adj_args for fused adjointSRK (wrtE output):
        // idx 0: dLdE, idx 1: v, idx 2: I_pad, idx 3-4: Gx/Gy_freq,
        // idx 5-7: scalars N/W/e, idx 8: dx, idx 9-10: ux/uy from fixed E
        std::vector<ArgDescriptor> adj_args = {
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "dLdE"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "v"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "I_pad"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2), "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2), "Gy_freq"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "erosion"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1), "dx"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "ux"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT), "uy"),
        };

        auto wrtE_fwd_init = [&](void* ptr, const ArgDescriptor& /*desc*/, size_t idx) {
            switch (idx) {
                case 1: memcpy(ptr, h_E.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
                case 2: memcpy(ptr, h_I.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
                case 3: memcpy(ptr, h_Gx_freq.data(), sizeof(float)*FREQ_N*2);        break;
                case 4: memcpy(ptr, h_Gy_freq.data(), sizeof(float)*FREQ_N*2);        break;
                case 5: memcpy(ptr, &p_N,  sizeof(int));   break;
                case 6: memcpy(ptr, &p_W,  sizeof(int));   break;
                case 7: memcpy(ptr, &p_e,  sizeof(int));   break;
                case 8: memcpy(ptr, &p_dx, sizeof(float)); break;
                default: break;
            }
        };

        auto wrtE_adj_init = [&](void* ptr, const ArgDescriptor& /*desc*/, size_t idx) {
            switch (idx) {
                case 2:  memcpy(ptr, h_I.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
                case 3:  memcpy(ptr, h_Gx_freq.data(), sizeof(float)*FREQ_N*2);        break;
                case 4:  memcpy(ptr, h_Gy_freq.data(), sizeof(float)*FREQ_N*2);        break;
                case 5:  memcpy(ptr, &p_N,  sizeof(int));   break;
                case 6:  memcpy(ptr, &p_W,  sizeof(int));   break;
                case 7:  memcpy(ptr, &p_e,  sizeof(int));   break;
                case 8:  memcpy(ptr, &p_dx, sizeof(float)); break;
                case 9:  memcpy(ptr, h_ux.data(), sizeof(float)*W_OUT*W_OUT); break;
                case 10: memcpy(ptr, h_uy.data(), sizeof(float)*W_OUT*W_OUT); break;
                default: break;
            }
        };

        tester.register_adjoint(
            "adjointWrtE seed=" + std::to_string(seed),
            fwd_wrtE_adapted_v<M_TILE>,    fwd_args,
            /*fwd_out_idx=*/     0,
            adj_wrtE_adapted_v<M_TILE>,    adj_args,
            /*adj_out_idx=*/     0,
            /*adj_dqdout_idx=*/  1,
            /*fwd_perturb_idx=*/ 1,
            /*adj_input_idx=*/   SIZE_MAX,
            AdjointConfig{ N_LARGE*N_LARGE, 0.5, 1e-5, 1e-3, (uint64_t)seed },
            {},              // PerfConfig
            wrtE_fwd_init,   // init_fn
            wrtE_adj_init,   // adj_init_fn
            {},              // GpConfig
            (seed == 1) ? "wrtE_seed1" : ""   // dump seed=1 only
        );
    }

    // Run small tests first, so their output comes before the big-test header
    auto results_small = tester.run_all();

    // ─────────────────────────────────────────────────────────────
    //  Task 2b: Big test  (N=8192, M=2048, point sampling)
    // ─────────────────────────────────────────────────────────────
    printf("\n=== Task 2b: Large adjoint test (N=8192, M=2048, point sampling) ===\n");

    const int N2       = get_arg_int(argc, argv, "--n_big", 8192);
    const int W_OUT2   = N2 - 2 * EROSION;
    constexpr int FREQ_N2     = M_TILE2 * (M_TILE2 / 2 + 1);
    constexpr int NUM_SAMPLES = 256;

    // Host GP-sampled E, I (length scale ~N2/30 ≈ 256 px for visual structure)
    auto h_E_2 = generate_pattern(TestPatternConfig::gaussian_process(N2, 256.f, /*seed=*/142));
    auto h_I_2 = generate_pattern(TestPatternConfig::gaussian_process(N2, 256.f, /*seed=*/143));

    // Upload E, I
    float *d_E_2 = nullptr, *d_I_2 = nullptr;
    cudaMalloc(&d_E_2, sizeof(float) * (size_t)N2 * N2);
    cudaMalloc(&d_I_2, sizeof(float) * (size_t)N2 * N2);
    cudaMemcpy(d_E_2, h_E_2.data(), sizeof(float)*(size_t)N2*N2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_I_2, h_I_2.data(), sizeof(float)*(size_t)N2*N2, cudaMemcpyHostToDevice);

    // Precompute Gx/Gy freq for M=2048  (same 31x31 Gaussian kernel)
    cufftComplex *d_Gx_freq_2 = nullptr, *d_Gy_freq_2 = nullptr;
    cudaMalloc(&d_Gx_freq_2, sizeof(cufftComplex) * FREQ_N2);
    cudaMalloc(&d_Gy_freq_2, sizeof(cufftComplex) * FREQ_N2);
    srkPrecomputeKernelFreq(h_Gx, KW, KH, M_TILE2, d_Gx_freq_2, nullptr);
    srkPrecomputeKernelFreq(h_Gy, KW, KH, M_TILE2, d_Gy_freq_2, nullptr);

    std::vector<float> h_Gx_freq_2(FREQ_N2 * 2), h_Gy_freq_2(FREQ_N2 * 2);
    cudaMemcpy(h_Gx_freq_2.data(), d_Gx_freq_2, sizeof(float)*FREQ_N2*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Gy_freq_2.data(), d_Gy_freq_2, sizeof(float)*FREQ_N2*2, cudaMemcpyDeviceToHost);

    // Allocate workspaces for M=2048 with FPB_COL=1 (row FPB=8, col FPB=1 per ref code).
    static constexpr unsigned FPB_COL_2 = 1;
    static SrkConvWorkspace<ARCH, M_TILE2, 16, 8, FPB_COL_2> fwd_ws_2;
    static SrkAdjWorkspace <ARCH, M_TILE2, 16, 8, FPB_COL_2> adj_ws_2;
    fwd_ws_2.allocate(N2, W_OUT2);
    adj_ws_2.allocate(N2);
    SrkTestCtx<M_TILE2, FPB_COL_2>::fwd_ws = &fwd_ws_2;
    SrkTestCtx<M_TILE2, FPB_COL_2>::adj_ws = &adj_ws_2;

    // Precompute ux_2, uy_2 from E_2 (for wrtI test — fixed params)
    float *d_ux_2 = nullptr, *d_uy_2 = nullptr;
    cudaMalloc(&d_ux_2, sizeof(float) * (size_t)W_OUT2 * W_OUT2);
    cudaMalloc(&d_uy_2, sizeof(float) * (size_t)W_OUT2 * W_OUT2);
    srkConvOaS<ARCH, M_TILE2, 16, 8, FPB_COL_2>(d_E_2, d_Gx_freq_2, N2, W_OUT2, EROSION, DX, nullptr, d_ux_2, fwd_ws_2);
    srkConvOaS<ARCH, M_TILE2, 16, 8, FPB_COL_2>(d_E_2, d_Gy_freq_2, N2, W_OUT2, EROSION, DX, nullptr, d_uy_2, fwd_ws_2);

    std::vector<float> h_ux_2((size_t)W_OUT2 * W_OUT2);
    std::vector<float> h_uy_2((size_t)W_OUT2 * W_OUT2);
    cudaMemcpy(h_ux_2.data(), d_ux_2, sizeof(float)*(size_t)W_OUT2*W_OUT2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy_2.data(), d_uy_2, sizeof(float)*(size_t)W_OUT2*W_OUT2, cudaMemcpyDeviceToHost);

    // Scalar params for big config
    int   p_N_2  = N2;
    int   p_W_2  = W_OUT2;
    int   p_e_2  = EROSION;
    float p_dx_2 = DX;

    // ── 3c-i: adjointWrtI (big, 1 seed, point sampling) ───────────
    {
        int seed = 1;
        std::vector<ArgDescriptor> fwd_args = {
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "R"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N2, N2), "I_pad"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "erosion"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "ux"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "uy"),
        };
        std::vector<ArgDescriptor> adj_args = {
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(N2, N2), "dLdI"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "v"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "erosion"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "ux"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "uy"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(N2, N2), "I_pad"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N2 * 2), "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N2 * 2), "Gy_freq"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1), "dx"),
        };

        auto wrtI_fwd_init = [&](void* ptr, const ArgDescriptor&, size_t idx) {
            switch (idx) {
                case 1: memcpy(ptr, h_I_2.data(),  sizeof(float)*(size_t)N2*N2); break;
                case 2: memcpy(ptr, &p_N_2,  sizeof(int));   break;
                case 3: memcpy(ptr, &p_W_2,  sizeof(int));   break;
                case 4: memcpy(ptr, &p_e_2,  sizeof(int));   break;
                case 5: memcpy(ptr, h_ux_2.data(), sizeof(float)*(size_t)W_OUT2*W_OUT2); break;
                case 6: memcpy(ptr, h_uy_2.data(), sizeof(float)*(size_t)W_OUT2*W_OUT2); break;
                default: break;
            }
        };
        auto wrtI_adj_init = [&](void* ptr, const ArgDescriptor&, size_t idx) {
            switch (idx) {
                case 2:  memcpy(ptr, &p_N_2,  sizeof(int));   break;
                case 3:  memcpy(ptr, &p_W_2,  sizeof(int));   break;
                case 4:  memcpy(ptr, &p_e_2,  sizeof(int));   break;
                case 5:  memcpy(ptr, h_ux_2.data(), sizeof(float)*(size_t)W_OUT2*W_OUT2); break;
                case 6:  memcpy(ptr, h_uy_2.data(), sizeof(float)*(size_t)W_OUT2*W_OUT2); break;
                case 7:  memcpy(ptr, h_I_2.data(),  sizeof(float)*(size_t)N2*N2); break;
                case 8:  memcpy(ptr, h_Gx_freq_2.data(), sizeof(float)*FREQ_N2*2); break;
                case 9:  memcpy(ptr, h_Gy_freq_2.data(), sizeof(float)*FREQ_N2*2); break;
                case 10: memcpy(ptr, &p_dx_2, sizeof(float)); break;
                default: break;
            }
        };

        tester_big.register_adjoint(
            "adjointWrtI BIG seed=" + std::to_string(seed),
            fwd_wrtI_adapted, fwd_args, /*fwd_out_idx=*/0,
            adj_wrtI_adapted, adj_args, /*adj_out_idx=*/0,
            /*adj_dqdout_idx=*/1, /*fwd_perturb_idx=*/1, /*adj_input_idx=*/SIZE_MAX,
            AdjointConfig{ NUM_SAMPLES, 0.1, 1e-5, 1e-3, (uint64_t)(seed+100) },
            {}, wrtI_fwd_init, wrtI_adj_init, {}, ""
        );
    }

    // ── 3c-ii: adjointWrtE (big, 1 seed, point sampling) ──────────
    {
        int seed = 1;
        std::vector<ArgDescriptor> fwd_args = {
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "R"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N2, N2), "E_pad"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(N2, N2), "I_pad"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N2 * 2), "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N2 * 2), "Gy_freq"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "erosion"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1), "dx"),
        };
        std::vector<ArgDescriptor> adj_args = {
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(N2, N2), "dLdE"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "v"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(N2, N2), "I_pad"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N2 * 2), "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N2 * 2), "Gy_freq"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1), "erosion"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1), "dx"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "ux"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_2d(W_OUT2, W_OUT2), "uy"),
        };

        auto wrtE_fwd_init = [&](void* ptr, const ArgDescriptor&, size_t idx) {
            switch (idx) {
                case 1: memcpy(ptr, h_E_2.data(),       sizeof(float)*(size_t)N2*N2); break;
                case 2: memcpy(ptr, h_I_2.data(),       sizeof(float)*(size_t)N2*N2); break;
                case 3: memcpy(ptr, h_Gx_freq_2.data(), sizeof(float)*FREQ_N2*2);        break;
                case 4: memcpy(ptr, h_Gy_freq_2.data(), sizeof(float)*FREQ_N2*2);        break;
                case 5: memcpy(ptr, &p_N_2,  sizeof(int));   break;
                case 6: memcpy(ptr, &p_W_2,  sizeof(int));   break;
                case 7: memcpy(ptr, &p_e_2,  sizeof(int));   break;
                case 8: memcpy(ptr, &p_dx_2, sizeof(float)); break;
                default: break;
            }
        };
        auto wrtE_adj_init = [&](void* ptr, const ArgDescriptor&, size_t idx) {
            switch (idx) {
                case 2:  memcpy(ptr, h_I_2.data(),       sizeof(float)*(size_t)N2*N2); break;
                case 3:  memcpy(ptr, h_Gx_freq_2.data(), sizeof(float)*FREQ_N2*2);        break;
                case 4:  memcpy(ptr, h_Gy_freq_2.data(), sizeof(float)*FREQ_N2*2);        break;
                case 5:  memcpy(ptr, &p_N_2,  sizeof(int));   break;
                case 6:  memcpy(ptr, &p_W_2,  sizeof(int));   break;
                case 7:  memcpy(ptr, &p_e_2,  sizeof(int));   break;
                case 8:  memcpy(ptr, &p_dx_2, sizeof(float)); break;
                case 9:  memcpy(ptr, h_ux_2.data(), sizeof(float)*(size_t)W_OUT2*W_OUT2); break;
                case 10: memcpy(ptr, h_uy_2.data(), sizeof(float)*(size_t)W_OUT2*W_OUT2); break;
                default: break;
            }
        };

        tester_big.register_adjoint(
            "adjointWrtE BIG seed=" + std::to_string(seed),
            fwd_wrtE_adapted_v<M_TILE2, FPB_COL_2>, fwd_args, /*fwd_out_idx=*/0,
            adj_wrtE_adapted_v<M_TILE2, FPB_COL_2>, adj_args, /*adj_out_idx=*/0,
            /*adj_dqdout_idx=*/1, /*fwd_perturb_idx=*/1, /*adj_input_idx=*/SIZE_MAX,
            // Big-test tolerance: abs bumped to 1e-4 (float32 FFT accum error scales with image size)
            AdjointConfig{ NUM_SAMPLES, 0.5, 1e-4, 1e-3, (uint64_t)(seed+100) },
            {}, wrtE_fwd_init, wrtE_adj_init, {}, ""
        );
    }

    auto results_big = tester_big.run_all();

    // ── Summary ───────────────────────────────────────────────────
    printf("\n=== Summary ===\n");
    printf("  Task 1 forward validation: %s\n", task1_pass ? "PASS" : "FAIL");
    printf("  Task 2 adjoint tests (small + big):\n");

    bool task3_pass = true;
    size_t idx = 0;
    for (auto& r : results_small) {
        printf("    [%zu] %s  rel_err_max=%.2e\n",
               idx++, r.passed ? "PASS" : "FAIL", r.rel_error_max);
        if (!r.passed) task3_pass = false;
    }
    for (auto& r : results_big) {
        printf("    [%zu] %s  rel_err_max=%.2e   (BIG)\n",
               idx++, r.passed ? "PASS" : "FAIL", r.rel_error_max);
        if (!r.passed) task3_pass = false;
    }

    bool all_pass = task1_pass && task3_pass;
    printf("\n  Overall: %s\n", all_pass ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_E);
    cudaFree(d_I);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_Gx_freq);
    cudaFree(d_Gy_freq);

    return all_pass ? 0 : 1;
}
