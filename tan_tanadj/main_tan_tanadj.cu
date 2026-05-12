// main.cu — shrinkage_tan_tanadj_test
//
// Validates the analytical tangent-derivative and tangent-adjoint-derivative
// kernels of the shrinkage operator
//
//   h = c · ( conv(Gx, g)·fx + conv(Gy, g)·fy )
//
// against numerical central-difference references.
//
// Pass criterion (PRD §6):
//   ∃ ε ∈ [1e-4, 1e-1] such that for all pixels,
//     |analytical - numerical(ε)| ≤ max(tol_abs, tol_rel · max(|ana|, |num|))
//   with tol_abs = 1e-5, tol_rel = 1e-3.
// ─────────────────────────────────────────────────────────────────

#include "srk_tangent.cuh"
#include "srk_tangent_adjoint.cuh"
#include <stTestCore/st_test_core.h>
#include <patternUtil/test_pattern.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <filesystem>

using namespace st::util;

// ── Compile-time parameters (template / architecture constants) ───
static constexpr unsigned ARCH      = 860;
static constexpr unsigned M_TILE    = 1024;
static constexpr float    DX        = 1.0f;
static constexpr float    SRSRK_C_W = 3.0f;  // Blackman radius = c_w · sigma
static constexpr int      FREQ_N    = M_TILE * (M_TILE / 2 + 1);

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

static float blackman_radial(float r, float R_w) {
    if (r < 0.f || r > R_w) return 0.f;
    const float t = r / R_w;
    return 0.42f + 0.5f * std::cos((float)M_PI * t)
                + 0.08f * std::cos(2.f * (float)M_PI * t);
}

// Modified Bessel I1(x), |x|<=3.75 polynomial (A&S 9.8.3)
static float bessel_I1_small(float x) {
    const float y = (x / 3.75f) * (x / 3.75f);
    return x * (0.5f + y * (0.87890594f +
                  y * (0.51498869f +
                  y * (0.15084934f +
                  y * (0.02658733f +
                  y * (0.00301532f + y * 0.00032411f))))));
}

// Modified Bessel K1(x): A&S 9.8.7 (x<=2) and 9.8.8 (x>2). ~1e-7 accurate.
static float bessel_K1(float x) {
    if (x <= 0.f) return 1e30f;
    if (x <= 2.f) {
        const float y = x * x * 0.25f;
        const float I1 = bessel_I1_small(x);
        const float poly =
            1.f + y * (0.15443144f -
                  y * (0.67278579f +
                  y * (0.18156897f +
                  y * (0.01919402f +
                  y * (0.00110404f + y * 0.00004686f)))));
        return std::log(x * 0.5f) * I1 + poly / x;
    }
    const float y = 2.f / x;
    const float poly =
        1.25331414f + y * (0.23498619f +
                      y * (-0.03655620f +
                      y * (0.01504268f +
                      y * (-0.00780353f +
                      y * (0.00325614f - y * 0.00068245f)))));
    return std::exp(-x) / std::sqrt(x) * poly;
}

// Generate (Kx, Ky) of size kw×kh, dx-spaced, centred at (kw/2, kh/2).
// Returns a pair via out-params.  Pixels with r > R_w or r ≈ 0 are zeroed.
static void make_srsrk_kernel_2d(std::vector<float>& Kx, std::vector<float>& Ky,
                                 int kw, int kh, float dx, float sigma) {
    const float gamma = 1.f / sigma;
    const float R_w   = SRSRK_C_W / gamma;          // = c_w · σ
    const float eps   = 1e-6f * dx;
    const float coeff = gamma / (2.f * (float)M_PI);
    Kx.assign((size_t)kw * kh, 0.f);
    Ky.assign((size_t)kw * kh, 0.f);
    const int cx = kw / 2, cy = kh / 2;
    for (int j = 0; j < kh; ++j) {
        for (int i = 0; i < kw; ++i) {
            const int idx = i * kh + j;          // col-major: col*H + row
            const float x = (i - cx) * dx;
            const float y = (j - cy) * dx;
            const float r = std::sqrt(x*x + y*y);
            if (r < eps || r > R_w) continue;
            const float w  = blackman_radial(r, R_w);
            const float K1 = bessel_K1(gamma * r);
            const float base = coeff * K1 * w;
            Kx[idx] = base * (x / r);
            Ky[idx] = base * (y / r);
        }
    }
}

// ── Per-M workspace context ───────────────────────────────────────
template<unsigned int M, unsigned int FPB_COL = 8>
struct ShrCtx {
    static inline SrkConvWorkspace  <ARCH, M, 16, 8, FPB_COL>* conv_ws    = nullptr;
    static inline SrkAdjWorkspace   <ARCH, M, 16, 8, FPB_COL>* adj_ws     = nullptr;
    static inline SrkTanWorkspace   <ARCH, M, 16, 8, FPB_COL>* tan_ws     = nullptr;
    static inline SrkTanAdjWorkspace<ARCH, M, 16, 8, FPB_COL>* tan_adj_ws = nullptr;
    // Scratch buffers for adjointWrtI (needs precomputed ux, uy from g).
    static inline float* d_ux_scratch = nullptr;
    static inline float* d_uy_scratch = nullptr;
};

// ─────────────────────────────────────────────────────────────────
//  Forward wrapper: (R, f, g, Gx, Gy, N, W, e, dx) → R
// ─────────────────────────────────────────────────────────────────
template<unsigned int M, unsigned int FPB_COL = 8>
static void srk_fwd_fn(
    float* d_R, float* d_f, float* d_g,
    float* d_Gx_f, float* d_Gy_f,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    float dx = *p_dx;
    srkForwardOaS<ARCH, M, 16, 8, FPB_COL>(
        d_g, d_f,
        reinterpret_cast<const cufftComplex*>(d_Gx_f),
        reinterpret_cast<const cufftComplex*>(d_Gy_f),
        N, Wv, e, dx, /*stream=*/nullptr, d_R, *ShrCtx<M, FPB_COL>::conv_ws);
}

// ─────────────────────────────────────────────────────────────────
//  Tangent wrapper: (dh, f, g, df, dg, Gx, Gy, N, W, e, dx) → dh
// ─────────────────────────────────────────────────────────────────
template<unsigned int M, unsigned int FPB_COL = 8>
static void srk_tan_fn(
    float* d_dh, float* d_f, float* d_g, float* d_df, float* d_dg,
    float* d_Gx_f, float* d_Gy_f,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    float dx = *p_dx;
    shrinkageTangentOaS<ARCH, M, 16, 8, FPB_COL>(
        d_f, d_g, d_df, d_dg,
        reinterpret_cast<const cufftComplex*>(d_Gx_f),
        reinterpret_cast<const cufftComplex*>(d_Gy_f),
        N, Wv, e, dx, nullptr, d_dh, *ShrCtx<M, FPB_COL>::tan_ws);
}

// ─────────────────────────────────────────────────────────────────
//  Adjoint wrappers
//  λ_f(f, g, λ_h) = adjointWrtI(λ_h, ux, uy) where ux,uy = c·conv(*,g)
//  λ_g(f, g, λ_h) = adjointWrtE(λ_h, f, Gx, Gy)
//
//  Args (mirror tan-adj layout): (out, f, g, λ_h, Gx, Gy, N, W, e, dx)
//  - λ_f variant: ignores f (no dependence); recomputes ux,uy from g.
//  - λ_g variant: ignores g (no dependence).
// ─────────────────────────────────────────────────────────────────
template<unsigned int M, unsigned int FPB_COL = 8>
static void srk_adj_lambda_f_fn(
    float* d_lam_f,            // [N × N]
    float* /*d_f*/,             // unused (λ_f independent of f)
    float* d_g,                // [N × N]
    float* d_lam_h,            // [W × W]
    float* d_Gx_f, float* d_Gy_f,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    float dx = *p_dx;
    auto& cws = *ShrCtx<M, FPB_COL>::conv_ws;
    // Compute ux = c·conv(Gx, g), uy = c·conv(Gy, g)
    srkConvOaS<ARCH, M, 16, 8, FPB_COL>(
        d_g, reinterpret_cast<const cufftComplex*>(d_Gx_f),
        N, Wv, e, dx, nullptr,
        ShrCtx<M, FPB_COL>::d_ux_scratch, cws);
    srkConvOaS<ARCH, M, 16, 8, FPB_COL>(
        d_g, reinterpret_cast<const cufftComplex*>(d_Gy_f),
        N, Wv, e, dx, nullptr,
        ShrCtx<M, FPB_COL>::d_uy_scratch, cws);
    adjointWrtI(d_lam_h,
                ShrCtx<M, FPB_COL>::d_ux_scratch,
                ShrCtx<M, FPB_COL>::d_uy_scratch,
                Wv, N, e, nullptr, d_lam_f);
}

template<unsigned int M, unsigned int FPB_COL = 8>
static void srk_adj_lambda_g_fn(
    float* d_lam_g,            // [N × N]
    float* d_f,                // [N × N]
    float* /*d_g*/,             // unused (λ_g independent of g)
    float* d_lam_h,            // [W × W]
    float* d_Gx_f, float* d_Gy_f,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    float dx = *p_dx;
    adjointWrtE<ARCH, M, 16, 8, FPB_COL>(
        d_lam_h, d_f,
        reinterpret_cast<const cufftComplex*>(d_Gx_f),
        reinterpret_cast<const cufftComplex*>(d_Gy_f),
        N, Wv, e, dx, nullptr, d_lam_g, *ShrCtx<M, FPB_COL>::adj_ws);
}

// ─────────────────────────────────────────────────────────────────
//  Tangent-adjoint wrappers
//  Args: (out, f, g, λ_h, df, dg, dλ_h, Gx, Gy, N, W, e, dx) → dλ_*
//  d_dLam_f and d_dLam_g are produced together by the fused kernel; each
//  wrapper exposes one and writes the other into a scratch dummy buffer.
// ─────────────────────────────────────────────────────────────────
template<unsigned int M, unsigned int FPB_COL = 8>
struct ShrTanAdjDummy {
    static inline float* d_dummy_NN = nullptr;   ///< [N × N] scratch for unused output
};

template<unsigned int M, unsigned int FPB_COL = 8>
static void srk_tan_adj_dLam_f_fn(
    float* d_dLam_f,
    float* d_f, float* d_g,
    float* d_lam_h,
    float* d_df, float* d_dg, float* d_dlam_h,
    float* d_Gx_f, float* d_Gy_f,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    float dx = *p_dx;
    auto& ws = *ShrCtx<M, FPB_COL>::tan_adj_ws;
    shrinkageTangentAdjointOaS<ARCH, M, 16, 8, FPB_COL>(
        d_f, d_g, d_df, d_dg, d_lam_h, d_dlam_h,
        reinterpret_cast<const cufftComplex*>(d_Gx_f),
        reinterpret_cast<const cufftComplex*>(d_Gy_f),
        N, Wv, e, dx, nullptr,
        d_dLam_f, ShrTanAdjDummy<M, FPB_COL>::d_dummy_NN,
        ws);
}

template<unsigned int M, unsigned int FPB_COL = 8>
static void srk_tan_adj_dLam_g_fn(
    float* d_dLam_g,
    float* d_f, float* d_g,
    float* d_lam_h,
    float* d_df, float* d_dg, float* d_dlam_h,
    float* d_Gx_f, float* d_Gy_f,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    int N = *p_N, Wv = *p_W, e = *p_e;
    float dx = *p_dx;
    auto& ws = *ShrCtx<M, FPB_COL>::tan_adj_ws;
    shrinkageTangentAdjointOaS<ARCH, M, 16, 8, FPB_COL>(
        d_f, d_g, d_df, d_dg, d_lam_h, d_dlam_h,
        reinterpret_cast<const cufftComplex*>(d_Gx_f),
        reinterpret_cast<const cufftComplex*>(d_Gy_f),
        N, Wv, e, dx, nullptr,
        ShrTanAdjDummy<M, FPB_COL>::d_dummy_NN, d_dLam_g,
        ws);
}

// ─────────────────────────────────────────────────────────────────
//  FuncAdapter instantiations
// ─────────────────────────────────────────────────────────────────
template<unsigned M, unsigned FPB_COL = 8>
constexpr AdaptedFunc fwd_adapt = FuncAdapter<srk_fwd_fn<M, FPB_COL>,
    float*, float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

template<unsigned M, unsigned FPB_COL = 8>
constexpr AdaptedFunc tan_adapt = FuncAdapter<srk_tan_fn<M, FPB_COL>,
    float*, float*, float*, float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

template<unsigned M, unsigned FPB_COL = 8>
constexpr AdaptedFunc adj_lam_f_adapt = FuncAdapter<srk_adj_lambda_f_fn<M, FPB_COL>,
    float*, float*, float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

template<unsigned M, unsigned FPB_COL = 8>
constexpr AdaptedFunc adj_lam_g_adapt = FuncAdapter<srk_adj_lambda_g_fn<M, FPB_COL>,
    float*, float*, float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

template<unsigned M, unsigned FPB_COL = 8>
constexpr AdaptedFunc tan_adj_lam_f_adapt = FuncAdapter<srk_tan_adj_dLam_f_fn<M, FPB_COL>,
    float*, float*, float*, float*, float*, float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

template<unsigned M, unsigned FPB_COL = 8>
constexpr AdaptedFunc tan_adj_lam_g_adapt = FuncAdapter<srk_tan_adj_dLam_g_fn<M, FPB_COL>,
    float*, float*, float*, float*, float*, float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

// ─────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    const float SIGMA   = get_arg_float(argc, argv, "--sigma",  5.f);
    const int   W_OUT   = get_arg_int  (argc, argv, "--w_out",  256);
    const float R_W     = SRSRK_C_W * SIGMA;
    const int   EROSION = (int)(R_W / DX) + 1;
    const int   KW      = 2 * (int)(R_W / DX) + 1;
    const int   KH      = KW;
    const int   N_LARGE = W_OUT + 2 * EROSION;
    printf("sigma=%.1f  w_out=%d  R_w=%.1f  kw=%d  erosion=%d  N_large=%d\n",
           SIGMA, W_OUT, R_W, KW, EROSION, N_LARGE);
    cudaSetDevice(0);

    // ── Generate base state (f, g, λ_h) — fixed test fixture ───────
    auto h_f  = generate_pattern(TestPatternConfig::gaussian_process(N_LARGE, 32.f, 42));
    auto h_g  = generate_pattern(TestPatternConfig::gaussian_process(N_LARGE, 32.f, 43));
    std::vector<float> h_lam_h((size_t)W_OUT * W_OUT);
    {
        std::mt19937 rng(123);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : h_lam_h)  x = nd(rng);
    }
    std::vector<float> h_zero_NN((size_t)N_LARGE * N_LARGE, 0.f);
    std::vector<float> h_zero_WW((size_t)W_OUT * W_OUT, 0.f);

    // ── Generate N_DIRS direction sets (df_d, dg_d, dλ_h_d) for j-sweep ──
    // PRD §6.1 / §6.2 ask "∀j" — direction-axis coverage.  We register each
    // mode N_DIRS times with independent random directions.
    constexpr int N_DIRS = 3;
    std::vector<std::vector<float>> h_dfs    (N_DIRS);
    std::vector<std::vector<float>> h_dgs    (N_DIRS);
    std::vector<std::vector<float>> h_dlam_hs(N_DIRS);
    for (int d = 0; d < N_DIRS; ++d) {
        h_dfs[d] = generate_pattern(
            TestPatternConfig::gaussian_process(N_LARGE, 32.f, /*seed=*/100 + 3*d + 0));
        h_dgs[d] = generate_pattern(
            TestPatternConfig::gaussian_process(N_LARGE, 32.f, /*seed=*/100 + 3*d + 1));
        h_dlam_hs[d].resize((size_t)W_OUT * W_OUT);
        std::mt19937 r(200 + d);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : h_dlam_hs[d]) x = nd(r);
    }

    // Gaussian kernels and their frequency representations
    // SRSRK Green's tensor kernel (Bessel K1 · radial Blackman).
    // σ_k = 5 with dx=1 gives R_w = c_w·σ = 15, which fits KW=KH=31 (cx=cy=15).
    std::vector<float> h_Gx, h_Gy;
    make_srsrk_kernel_2d(h_Gx, h_Gy, KW, KH, DX, SIGMA);

    cufftComplex *d_Gx_freq = nullptr, *d_Gy_freq = nullptr;
    cudaMallocAsync(&d_Gx_freq, sizeof(cufftComplex) * FREQ_N, /*stream=*/nullptr);
    cudaMallocAsync(&d_Gy_freq, sizeof(cufftComplex) * FREQ_N, /*stream=*/nullptr);
    srkPrecomputeKernelFreq(h_Gx, KW, KH, M_TILE, d_Gx_freq, nullptr);
    srkPrecomputeKernelFreq(h_Gy, KW, KH, M_TILE, d_Gy_freq, nullptr);

    std::vector<float> h_Gx_freq(FREQ_N * 2), h_Gy_freq(FREQ_N * 2);
    cudaMemcpyAsync(h_Gx_freq.data(), d_Gx_freq, sizeof(float)*FREQ_N*2,
                    cudaMemcpyDeviceToHost, /*stream=*/nullptr);
    cudaMemcpyAsync(h_Gy_freq.data(), d_Gy_freq, sizeof(float)*FREQ_N*2,
                    cudaMemcpyDeviceToHost, /*stream=*/nullptr);
    cudaStreamSynchronize(nullptr);   // init_fn lambdas below will read these

    // ── Allocate workspaces (once) ────────────────────────────────
    static SrkConvWorkspace  <ARCH, M_TILE> conv_ws;
    static SrkAdjWorkspace   <ARCH, M_TILE> adj_ws;
    static SrkTanWorkspace   <ARCH, M_TILE> tan_ws;
    static SrkTanAdjWorkspace<ARCH, M_TILE> tan_adj_ws;
    conv_ws.allocate(N_LARGE, W_OUT);
    adj_ws .allocate(N_LARGE);
    tan_ws .allocate(N_LARGE, W_OUT);
    tan_adj_ws.allocate(N_LARGE, W_OUT);
    ShrCtx<M_TILE>::conv_ws    = &conv_ws;
    ShrCtx<M_TILE>::adj_ws     = &adj_ws;
    ShrCtx<M_TILE>::tan_ws     = &tan_ws;
    ShrCtx<M_TILE>::tan_adj_ws = &tan_adj_ws;
    cudaMallocAsync(&ShrCtx<M_TILE>::d_ux_scratch,    sizeof(float) * W_OUT * W_OUT,     /*stream=*/nullptr);
    cudaMallocAsync(&ShrCtx<M_TILE>::d_uy_scratch,    sizeof(float) * W_OUT * W_OUT,     /*stream=*/nullptr);
    cudaMallocAsync(&ShrTanAdjDummy<M_TILE>::d_dummy_NN, sizeof(float) * N_LARGE * N_LARGE, /*stream=*/nullptr);

    // ── Scalar host params (shared across registrations) ─────────
    int   p_N  = N_LARGE;
    int   p_W  = W_OUT;
    int   p_e  = EROSION;
    float p_dx = DX;

    // ─────────────────────────────────────────────────────────────
    // Helpers to build argument descriptor lists.
    // ─────────────────────────────────────────────────────────────
    auto make_fwd_args = [&]() {
        return std::vector<ArgDescriptor>{
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT),     "h"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "f"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "g"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2),       "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2),       "Gy_freq"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "erosion"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1),                "dx"),
        };
    };
    auto make_tan_args = [&]() {
        return std::vector<ArgDescriptor>{
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT),     "dh"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "f"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "g"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "df"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "dg"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2),       "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2),       "Gy_freq"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "erosion"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1),                "dx"),
        };
    };
    auto make_adj_args = [&](const std::string& out_name, int out_dim_n) {
        return std::vector<ArgDescriptor>{
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(out_dim_n, out_dim_n), out_name),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE),     "f"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE),     "g"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT),         "lam_h"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2),           "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2),           "Gy_freq"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                    "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                    "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                    "erosion"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1),                    "dx"),
        };
    };
    auto make_tan_adj_args = [&](const std::string& out_name) {
        return std::vector<ArgDescriptor>{
            ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), out_name),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "f"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "g"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(W_OUT,   W_OUT),   "lam_h"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "df"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE), "dg"),
            ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(W_OUT,   W_OUT),   "dlam_h"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2),       "Gx_freq"),
            ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N * 2),       "Gy_freq"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "N_large"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "W"),
            ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                "erosion"),
            ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1),                "dx"),
        };
    };

    // ─────────────────────────────────────────────────────────────
    //  PART 1 — TANGENT DERIVATIVE TESTS
    // ─────────────────────────────────────────────────────────────
    printf("\n=== Part 1: Tangent Derivative Tests ===\n");
    TangentTester tan_tester;

    // Common forward-input init.
    auto fwd_init = [&](void* ptr, const ArgDescriptor&, size_t idx) {
        switch (idx) {
            case 1: memcpy(ptr, h_f.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
            case 2: memcpy(ptr, h_g.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
            case 3: memcpy(ptr, h_Gx_freq.data(), sizeof(float)*FREQ_N*2);        break;
            case 4: memcpy(ptr, h_Gy_freq.data(), sizeof(float)*FREQ_N*2);        break;
            case 5: memcpy(ptr, &p_N,  sizeof(int));   break;
            case 6: memcpy(ptr, &p_W,  sizeof(int));   break;
            case 7: memcpy(ptr, &p_e,  sizeof(int));   break;
            case 8: memcpy(ptr, &p_dx, sizeof(float)); break;
            default: break;
        }
    };

    TangentConfig tcfg;
    tcfg.num_samples = -1;     // full: compare all 256×256 = 65536 output pixels

    // Build a tangent init lambda for (mode, direction-index `dir`).
    //   mode 'A': df=h_dfs[dir], dg=0
    //   mode 'B': df=0,          dg=h_dgs[dir]
    //   mode 'C': df=h_dfs[dir], dg=h_dgs[dir]
    auto make_tan_init = [&](char mode, int dir) {
        return [&, mode, dir](void* ptr, const ArgDescriptor&, size_t idx) {
            // tan_args: 0:dh, 1:f, 2:g, 3:df, 4:dg, 5:Gx, 6:Gy, 7:N, 8:W, 9:e, 10:dx
            const float* df_src = (mode == 'B')                 ? h_zero_NN.data() : h_dfs[dir].data();
            const float* dg_src = (mode == 'A')                 ? h_zero_NN.data() : h_dgs[dir].data();
            switch (idx) {
                case 1:  memcpy(ptr, h_f.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
                case 2:  memcpy(ptr, h_g.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
                case 3:  memcpy(ptr, df_src,           sizeof(float)*N_LARGE*N_LARGE); break;
                case 4:  memcpy(ptr, dg_src,           sizeof(float)*N_LARGE*N_LARGE); break;
                case 5:  memcpy(ptr, h_Gx_freq.data(), sizeof(float)*FREQ_N*2);        break;
                case 6:  memcpy(ptr, h_Gy_freq.data(), sizeof(float)*FREQ_N*2);        break;
                case 7:  memcpy(ptr, &p_N,  sizeof(int));   break;
                case 8:  memcpy(ptr, &p_W,  sizeof(int));   break;
                case 9:  memcpy(ptr, &p_e,  sizeof(int));   break;
                case 10: memcpy(ptr, &p_dx, sizeof(float)); break;
                default: break;
            }
        };
    };

    for (int dir = 0; dir < N_DIRS; ++dir) {
        const std::string ds = " dir=" + std::to_string(dir);

        // Mode A: df-only
        tan_tester.register_tangent(
            "tangent ModeA df-only" + ds,
            fwd_adapt<M_TILE>, make_fwd_args(), /*fwd_out_idx=*/0,
            tan_adapt<M_TILE>, make_tan_args(), /*tan_out_idx=*/0,
            std::vector<PerturbSpec>{ {1, 3} },
            tcfg, {}, fwd_init, make_tan_init('A', dir), {},
            (dir == 0) ? "tangent_modeA" : "");

        // Mode B: dg-only
        tan_tester.register_tangent(
            "tangent ModeB dg-only" + ds,
            fwd_adapt<M_TILE>, make_fwd_args(), 0,
            tan_adapt<M_TILE>, make_tan_args(), 0,
            std::vector<PerturbSpec>{ {2, 4} },
            tcfg, {}, fwd_init, make_tan_init('B', dir), {},
            (dir == 0) ? "tangent_modeB" : "");

        // Mode C: full
        tan_tester.register_tangent(
            "tangent ModeC full" + ds,
            fwd_adapt<M_TILE>, make_fwd_args(), 0,
            tan_adapt<M_TILE>, make_tan_args(), 0,
            std::vector<PerturbSpec>{ {1, 3}, {2, 4} },
            tcfg, {}, fwd_init, make_tan_init('C', dir), {},
            (dir == 0) ? "tangent_modeC" : "");
    }

    auto tan_results = tan_tester.run_all();

    // ─────────────────────────────────────────────────────────────
    //  PART 2 — TANGENT-ADJOINT DERIVATIVE TESTS
    // ─────────────────────────────────────────────────────────────
    printf("\n=== Part 2: Tangent-Adjoint Derivative Tests ===\n");
    TangentAdjointTester tanadj_tester;

    // adj_args layout: 0:out, 1:f, 2:g, 3:lam_h, 4:Gx, 5:Gy, 6:N, 7:W, 8:e, 9:dx
    auto adj_init_f = [&](void* ptr, const ArgDescriptor&, size_t idx) {
        switch (idx) {
            case 1: memcpy(ptr, h_f.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
            case 2: memcpy(ptr, h_g.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
            case 3: memcpy(ptr, h_lam_h.data(),   sizeof(float)*W_OUT*W_OUT);     break;
            case 4: memcpy(ptr, h_Gx_freq.data(), sizeof(float)*FREQ_N*2);        break;
            case 5: memcpy(ptr, h_Gy_freq.data(), sizeof(float)*FREQ_N*2);        break;
            case 6: memcpy(ptr, &p_N,  sizeof(int));   break;
            case 7: memcpy(ptr, &p_W,  sizeof(int));   break;
            case 8: memcpy(ptr, &p_e,  sizeof(int));   break;
            case 9: memcpy(ptr, &p_dx, sizeof(float)); break;
            default: break;
        }
    };

    // tan_adj_args layout:
    //  0:out, 1:f, 2:g, 3:lam_h, 4:df, 5:dg, 6:dlam_h, 7:Gx, 8:Gy, 9:N, 10:W, 11:e, 12:dx
    auto make_tanadj_init = [&](bool use_df, bool use_dg, bool use_dlam_h, int dir) {
        return [&, use_df, use_dg, use_dlam_h, dir](void* ptr, const ArgDescriptor&, size_t idx) {
            switch (idx) {
                case 1:  memcpy(ptr, h_f.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
                case 2:  memcpy(ptr, h_g.data(),       sizeof(float)*N_LARGE*N_LARGE); break;
                case 3:  memcpy(ptr, h_lam_h.data(),   sizeof(float)*W_OUT*W_OUT);     break;
                case 4:  memcpy(ptr, use_df    ? h_dfs[dir].data()     : h_zero_NN.data(),
                                    sizeof(float)*N_LARGE*N_LARGE); break;
                case 5:  memcpy(ptr, use_dg    ? h_dgs[dir].data()     : h_zero_NN.data(),
                                    sizeof(float)*N_LARGE*N_LARGE); break;
                case 6:  memcpy(ptr, use_dlam_h ? h_dlam_hs[dir].data() : h_zero_WW.data(),
                                    sizeof(float)*W_OUT*W_OUT); break;
                case 7:  memcpy(ptr, h_Gx_freq.data(), sizeof(float)*FREQ_N*2); break;
                case 8:  memcpy(ptr, h_Gy_freq.data(), sizeof(float)*FREQ_N*2); break;
                case 9:  memcpy(ptr, &p_N,  sizeof(int));   break;
                case 10: memcpy(ptr, &p_W,  sizeof(int));   break;
                case 11: memcpy(ptr, &p_e,  sizeof(int));   break;
                case 12: memcpy(ptr, &p_dx, sizeof(float)); break;
                default: break;
            }
        };
    };

    TangentAdjointConfig tacfg;
    tacfg.num_samples = -1;

    for (int dir = 0; dir < N_DIRS; ++dir) {
        const std::string ds = " dir=" + std::to_string(dir);

        // ── dλ_f cases ─────────────────────────────────────────────
        // Mode D: dλ_h only
        tanadj_tester.register_tangent_adjoint(
            "tan-adj dLam_f ModeD dlam_h-only" + ds,
            adj_lam_f_adapt<M_TILE>,    make_adj_args("lam_f", N_LARGE), 0,
            tan_adj_lam_f_adapt<M_TILE>, make_tan_adj_args("dLam_f"),    0,
            std::vector<PerturbSpec>{ {3, 6} },
            tacfg, {}, adj_init_f, make_tanadj_init(false, false, true, dir), {},
            (dir == 0) ? "tanadj_dLam_f_modeD" : "");

        // Mode E: dg-only
        tanadj_tester.register_tangent_adjoint(
            "tan-adj dLam_f ModeE dg-only" + ds,
            adj_lam_f_adapt<M_TILE>,    make_adj_args("lam_f", N_LARGE), 0,
            tan_adj_lam_f_adapt<M_TILE>, make_tan_adj_args("dLam_f"),    0,
            std::vector<PerturbSpec>{ {2, 5} },
            tacfg, {}, adj_init_f, make_tanadj_init(false, true, false, dir), {},
            (dir == 0) ? "tanadj_dLam_f_modeE" : "");

        // Mode G: full
        tanadj_tester.register_tangent_adjoint(
            "tan-adj dLam_f ModeG full" + ds,
            adj_lam_f_adapt<M_TILE>,    make_adj_args("lam_f", N_LARGE), 0,
            tan_adj_lam_f_adapt<M_TILE>, make_tan_adj_args("dLam_f"),    0,
            std::vector<PerturbSpec>{ {1, 4}, {2, 5}, {3, 6} },
            tacfg, {}, adj_init_f, make_tanadj_init(true, true, true, dir), {},
            (dir == 0) ? "tanadj_dLam_f_modeG" : "");

        // ── dλ_g cases ─────────────────────────────────────────────
        // Mode D: dλ_h only
        tanadj_tester.register_tangent_adjoint(
            "tan-adj dLam_g ModeD dlam_h-only" + ds,
            adj_lam_g_adapt<M_TILE>,    make_adj_args("lam_g", N_LARGE), 0,
            tan_adj_lam_g_adapt<M_TILE>, make_tan_adj_args("dLam_g"),    0,
            std::vector<PerturbSpec>{ {3, 6} },
            tacfg, {}, adj_init_f, make_tanadj_init(false, false, true, dir), {},
            (dir == 0) ? "tanadj_dLam_g_modeD" : "");

        // Mode F: df-only
        tanadj_tester.register_tangent_adjoint(
            "tan-adj dLam_g ModeF df-only" + ds,
            adj_lam_g_adapt<M_TILE>,    make_adj_args("lam_g", N_LARGE), 0,
            tan_adj_lam_g_adapt<M_TILE>, make_tan_adj_args("dLam_g"),    0,
            std::vector<PerturbSpec>{ {1, 4} },
            tacfg, {}, adj_init_f, make_tanadj_init(true, false, false, dir), {},
            (dir == 0) ? "tanadj_dLam_g_modeF" : "");

        // Mode G: full
        tanadj_tester.register_tangent_adjoint(
            "tan-adj dLam_g ModeG full" + ds,
            adj_lam_g_adapt<M_TILE>,    make_adj_args("lam_g", N_LARGE), 0,
            tan_adj_lam_g_adapt<M_TILE>, make_tan_adj_args("dLam_g"),    0,
            std::vector<PerturbSpec>{ {1, 4}, {2, 5}, {3, 6} },
            tacfg, {}, adj_init_f, make_tanadj_init(true, true, true, dir), {},
            (dir == 0) ? "tanadj_dLam_g_modeG" : "");
    }

    auto tanadj_results = tanadj_tester.run_all();

    // ─────────────────────────────────────────────────────────────
    //  Summary
    // ─────────────────────────────────────────────────────────────
    printf("\n=== Summary ===\n");
    printf("  N_DIRS = %d   (direction-axis sweep per mode)\n", N_DIRS);
    bool all_pass = true;
    int n_tan_pass = 0, n_tanadj_pass = 0;

    const char* tan_modes[]    = {"ModeA(df)  ", "ModeB(dg)  ", "ModeC(full)"};
    const char* tanadj_modes[] = {"dLam_f ModeD(dlh)  ", "dLam_f ModeE(dg)   ", "dLam_f ModeG(full) ",
                                  "dLam_g ModeD(dlh)  ", "dLam_g ModeF(df)   ", "dLam_g ModeG(full) "};

    printf("  Tangent derivative tests:\n");
    for (size_t i = 0; i < tan_results.size(); ++i) {
        int dir   = (int)(i / 3);
        int mode  = (int)(i % 3);
        auto& r = tan_results[i];
        printf("    [%2zu] dir=%d %s  %s  best_eps=%.2e  abs_max=%.2e  rel_max=%.2e\n",
               i, dir, tan_modes[mode],
               r.passed ? "PASS" : "FAIL",
               r.best_epsilon, r.best_abs_error_max, r.best_rel_error_max);
        if (r.passed) ++n_tan_pass; else all_pass = false;
    }
    printf("  Tangent-adjoint derivative tests:\n");
    for (size_t i = 0; i < tanadj_results.size(); ++i) {
        int dir  = (int)(i / 6);
        int mode = (int)(i % 6);
        auto& r = tanadj_results[i];
        printf("    [%2zu] dir=%d %s  %s  best_eps=%.2e  abs_max=%.2e  rel_max=%.2e\n",
               i, dir, tanadj_modes[mode],
               r.passed ? "PASS" : "FAIL",
               r.best_epsilon, r.best_abs_error_max, r.best_rel_error_max);
        if (r.passed) ++n_tanadj_pass; else all_pass = false;
    }
    printf("\n  Tangent:        %d / %zu PASS\n", n_tan_pass,    tan_results.size());
    printf("  Tangent-adjoint: %d / %zu PASS\n", n_tanadj_pass, tanadj_results.size());
    printf("  Overall: %s\n", all_pass ? "PASS" : "FAIL");

    // Cleanup
    cudaFreeAsync(d_Gx_freq,                          /*stream=*/nullptr);
    cudaFreeAsync(d_Gy_freq,                          /*stream=*/nullptr);
    cudaFreeAsync(ShrCtx<M_TILE>::d_ux_scratch,       /*stream=*/nullptr);
    cudaFreeAsync(ShrCtx<M_TILE>::d_uy_scratch,       /*stream=*/nullptr);
    cudaFreeAsync(ShrTanAdjDummy<M_TILE>::d_dummy_NN, /*stream=*/nullptr);

    return all_pass ? 0 : 1;
}
