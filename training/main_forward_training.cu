// main.cu — shrinkage_forward_training_test
//
// Compares three implementations of the SRK forward operator:
//
//     R = ( conv(E, Gx) · dI/dx + conv(E, Gy) · dI/dy )
//
//   (1) srkForwardOaS       — cufftDX OaS, M_TILE-sized tiles  (reference)
//   (2) srkTrainingForward  — cuFFT batched, FFT size = N_large
//   (3) srkCudnnForward     — cuDNN spatial convolution
//
// At batch B = 1, all three should agree to within FP precision because they
// implement the same mathematical operator.  Tests are driven by FuncTester
// (stLitho/util/examples/src/func_test.cu pattern).
//
// G_x, G_y use the SRSRK Green's tensor:
//     K_x = γ/(2π) · (x/r) · K1(γr) · w(r)   (modified Bessel × Blackman window)
// ─────────────────────────────────────────────────────────────────

#include "srk_forward.cuh"
#include "srk_training_forward.cuh"
#include "srk_cudnn_forward.cuh"

#include <stTestCore/st_test_core.h>
#include <patternUtil/test_pattern.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

using namespace st::util;

// ── Compile-time parameters (template / architecture constants) ───
static constexpr unsigned ARCH      = 860;
static constexpr unsigned M_TILE    = 1024;      // for srkForwardOaS
static constexpr float    DX        = 1.0f;
static constexpr float    SRSRK_C_W = 3.0f;
// N1, N2 drive template instantiation (Ctx<N,FREQ_N>) — keep compile-time.
static constexpr int      N1        = 256;
static constexpr int      N2        = 512;
static constexpr int      FREQ_N_M  = M_TILE * (M_TILE / 2 + 1);
static constexpr int      FREQ_N_1  = N1     * (N1     / 2 + 1);
static constexpr int      FREQ_N_2  = N2     * (N2     / 2 + 1);

// Runtime-configurable values are parsed from argv in main().
// wrap_fwd_* reads B_TEST and wrap_big_* reads B_BIG via these (set in main).
static int s_B_TEST = 1;
static int s_B_BIG  = 3000;

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

// ── SRSRK Green's tensor kernel (Bessel K1 × radial Blackman) ─────
// Copied verbatim from shrinkage_tan_tanadj_test/main.cu.

static float blackman_radial(float r, float R_w) {
    if (r < 0.f || r > R_w) return 0.f;
    const float t = r / R_w;
    return 0.42f + 0.5f * std::cos((float)M_PI * t)
                + 0.08f * std::cos(2.f * (float)M_PI * t);
}

static float bessel_I1_small(float x) {
    const float y = (x / 3.75f) * (x / 3.75f);
    return x * (0.5f + y * (0.87890594f +
                  y * (0.51498869f +
                  y * (0.15084934f +
                  y * (0.02658733f +
                  y * (0.00301532f + y * 0.00032411f))))));
}

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

// Column-major kernel buffers: K[row=j + col=i*kh].  j indexes vertical (y),
// i indexes horizontal (x).  Kx is anti-symmetric in x (i), Ky in y (j).
static void make_srsrk_kernel_2d(std::vector<float>& Kx, std::vector<float>& Ky,
                                 int kw, int kh, float dx, float sigma) {
    const float gamma = 1.f / sigma;
    const float R_w   = SRSRK_C_W / gamma;          // = c_w · σ
    const float eps   = 1e-6f * dx;
    const float coeff = gamma / (2.f * (float)M_PI);
    Kx.assign((size_t)kw * kh, 0.f);
    Ky.assign((size_t)kw * kh, 0.f);
    const int cx = kw / 2, cy = kh / 2;
    for (int i = 0; i < kw; ++i) {
        for (int j = 0; j < kh; ++j) {
            const int idx = j + i * kh;             // column-major
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

// ─────────────────────────────────────────────────────────────────
//  Per-(N, FreqN) test context — workspaces live for the program's
//  lifetime, shared by all wrappers compiled for that N.
// ─────────────────────────────────────────────────────────────────
template<int N_LARGE, int FREQ_N_NS>
struct Ctx {
    static inline SrkConvWorkspace<ARCH, M_TILE>* conv_ws  = nullptr;
    static inline SrkTrainWorkspace*              train_ws = nullptr;
    static inline SrkCudnnWorkspace*              cudnn_ws = nullptr;
};

// ─────────────────────────────────────────────────────────────────
//  Wrappers
//
//  Test 1 arg layout (srk_forward  vs  srk_training_forward):
//    0:R, 1:E, 2:I,
//    3:Gx_freq_M, 4:Gy_freq_M,      (used by srk_forward)
//    5:Gx_freq_N, 6:Gy_freq_N,      (used by srk_training_forward)
//    7:N, 8:W, 9:erosion, 10:dx
//
//  Test 2 arg layout (srk_forward  vs  srk_cudnn_forward):
//    0:R, 1:E, 2:I,
//    3:Gx_freq_M, 4:Gy_freq_M,      (used by srk_forward)
//    5:Gx_spatial, 6:Gy_spatial,    (used by srk_cudnn_forward)
//    7:N, 8:W, 9:erosion, 10:dx
// ─────────────────────────────────────────────────────────────────

template<int N_LARGE, int FREQ_N_NS>
static void wrap_fwd_oas(
    float* d_R, float* d_E, float* d_I,
    float* d_Gx_M, float* d_Gy_M,
    float* /*d_Gx_alt*/, float* /*d_Gy_alt*/,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    srkForwardOaS<ARCH, M_TILE>(
        d_E, d_I,
        reinterpret_cast<const cufftComplex*>(d_Gx_M),
        reinterpret_cast<const cufftComplex*>(d_Gy_M),
        *p_N, *p_W, *p_e, *p_dx,
        /*stream=*/nullptr, d_R,
        *Ctx<N_LARGE, FREQ_N_NS>::conv_ws);
}

template<int N_LARGE, int FREQ_N_NS>
static void wrap_fwd_training(
    float* d_R, float* d_E, float* d_I,
    float* /*d_Gx_M*/, float* /*d_Gy_M*/,
    float* d_Gx_N, float* d_Gy_N,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    srkTrainingForward(
        d_E, d_I,
        reinterpret_cast<const cufftComplex*>(d_Gx_N),
        reinterpret_cast<const cufftComplex*>(d_Gy_N),
        *p_N, *p_W, *p_e, *p_dx, /*B=*/s_B_TEST,
        /*stream=*/nullptr, d_R,
        *Ctx<N_LARGE, FREQ_N_NS>::train_ws);
}

template<int N_LARGE, int FREQ_N_NS>
static void wrap_fwd_cudnn(
    float* d_R, float* d_E, float* d_I,
    float* /*d_Gx_M*/, float* /*d_Gy_M*/,
    float* d_Gx_sp, float* d_Gy_sp,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    srkCudnnForward(
        d_E, d_I, d_Gx_sp, d_Gy_sp,
        *p_N, *p_W, *p_e, *p_dx, /*B=*/s_B_TEST,
        /*stream=*/nullptr, d_R,
        *Ctx<N_LARGE, FREQ_N_NS>::cudnn_ws);
}

template<int N_LARGE, int FREQ_N_NS>
constexpr AdaptedFunc fwd_oas_adapt = FuncAdapter<
    wrap_fwd_oas<N_LARGE, FREQ_N_NS>,
    float*, float*, float*,
    float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

template<int N_LARGE, int FREQ_N_NS>
constexpr AdaptedFunc fwd_training_adapt = FuncAdapter<
    wrap_fwd_training<N_LARGE, FREQ_N_NS>,
    float*, float*, float*,
    float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

template<int N_LARGE, int FREQ_N_NS>
constexpr AdaptedFunc fwd_cudnn_adapt = FuncAdapter<
    wrap_fwd_cudnn<N_LARGE, FREQ_N_NS>,
    float*, float*, float*,
    float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

// ─────────────────────────────────────────────────────────────────
//  Big-batch wrappers (srk_training vs srk_cudnn at B = B_BIG).
//  Uses dedicated workspaces (big_train_ws, big_cudnn_ws) — distinct
//  from the B=1 workspaces above.
//  Args layout (shared between the two wrappers):
//    0:R, 1:E, 2:I,
//    3:Gx_freq_N, 4:Gy_freq_N,        (used by srk_training only)
//    5:Gx_spatial, 6:Gy_spatial,      (used by srk_cudnn only)
//    7:N, 8:W, 9:erosion, 10:dx
// ─────────────────────────────────────────────────────────────────
static SrkTrainWorkspace big_train_ws;
static SrkCudnnWorkspace big_cudnn_ws;

static void wrap_big_training(
    float* d_R, float* d_E, float* d_I,
    float* d_Gx_N, float* d_Gy_N,
    float* /*d_Gx_sp*/, float* /*d_Gy_sp*/,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    srkTrainingForward(
        d_E, d_I,
        reinterpret_cast<const cufftComplex*>(d_Gx_N),
        reinterpret_cast<const cufftComplex*>(d_Gy_N),
        *p_N, *p_W, *p_e, *p_dx, /*B=*/s_B_BIG,
        /*stream=*/nullptr, d_R, big_train_ws);
}

static void wrap_big_cudnn(
    float* d_R, float* d_E, float* d_I,
    float* /*d_Gx_N*/, float* /*d_Gy_N*/,
    float* d_Gx_sp, float* d_Gy_sp,
    int* p_N, int* p_W, int* p_e, float* p_dx)
{
    srkCudnnForward(
        d_E, d_I, d_Gx_sp, d_Gy_sp,
        *p_N, *p_W, *p_e, *p_dx, /*B=*/s_B_BIG,
        /*stream=*/nullptr, d_R, big_cudnn_ws);
}

constexpr AdaptedFunc big_training_adapt = FuncAdapter<wrap_big_training,
    float*, float*, float*, float*, float*, float*, float*,
    int*, int*, int*, float*>::call;
constexpr AdaptedFunc big_cudnn_adapt = FuncAdapter<wrap_big_cudnn,
    float*, float*, float*, float*, float*, float*, float*,
    int*, int*, int*, float*>::call;

// ─────────────────────────────────────────────────────────────────
//  Test registration helper.
//  Registers two FuncTester pairs:
//    (A) srk_forward vs srk_training_forward
//    (B) srk_forward vs srk_cudnn_forward
// ─────────────────────────────────────────────────────────────────
template<int N_LARGE, int FREQ_N_NS>
static void register_size(
    int                                W_OUT,
    int                                kw,
    int                                kh,
    FuncTester&                        tester,
    const std::vector<float>&          h_E,
    const std::vector<float>&          h_I,
    const std::vector<float>&          h_Gx_freq_M,
    const std::vector<float>&          h_Gy_freq_M,
    const std::vector<float>&          h_Gx_freq_N,
    const std::vector<float>&          h_Gy_freq_N,
    const std::vector<float>&          h_Gx_spatial,
    const std::vector<float>&          h_Gy_spatial,
    int  p_N, int  p_W, int  p_e, float p_dx,
    double tol_train, double tol_cudnn,
    const std::string&                 size_tag)
{
    auto args_train = std::vector<ArgDescriptor>{
        ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT),       "R"),
        ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE),   "E"),
        ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE),   "I"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N_M * 2),       "Gx_freq_M"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N_M * 2),       "Gy_freq_M"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N_NS * 2),      "Gx_freq_N"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N_NS * 2),      "Gy_freq_N"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                  "N"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                  "W"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                  "erosion"),
        ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1),                  "dx"),
    };

    auto args_cudnn = std::vector<ArgDescriptor>{
        ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_OUT, W_OUT),       "R"),
        ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE),   "E"),
        ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_LARGE, N_LARGE),   "I"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N_M * 2),       "Gx_freq_M"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N_M * 2),       "Gy_freq_M"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(kw * kh),            "Gx_spatial"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(kw * kh),            "Gy_spatial"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                  "N"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                  "W"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                  "erosion"),
        ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1),                  "dx"),
    };

    auto init_train = [=, &h_E, &h_I,
                       &h_Gx_freq_M, &h_Gy_freq_M,
                       &h_Gx_freq_N, &h_Gy_freq_N]
        (void* ptr, const ArgDescriptor&, size_t idx) {
        switch (idx) {
            case 1:  memcpy(ptr, h_E.data(),         sizeof(float) * N_LARGE * N_LARGE); break;
            case 2:  memcpy(ptr, h_I.data(),         sizeof(float) * N_LARGE * N_LARGE); break;
            case 3:  memcpy(ptr, h_Gx_freq_M.data(), sizeof(float) * FREQ_N_M * 2);      break;
            case 4:  memcpy(ptr, h_Gy_freq_M.data(), sizeof(float) * FREQ_N_M * 2);      break;
            case 5:  memcpy(ptr, h_Gx_freq_N.data(), sizeof(float) * FREQ_N_NS * 2);     break;
            case 6:  memcpy(ptr, h_Gy_freq_N.data(), sizeof(float) * FREQ_N_NS * 2);     break;
            case 7:  { int v = p_N;  memcpy(ptr, &v, sizeof(int));   break; }
            case 8:  { int v = p_W;  memcpy(ptr, &v, sizeof(int));   break; }
            case 9:  { int v = p_e;  memcpy(ptr, &v, sizeof(int));   break; }
            case 10: { float v = p_dx; memcpy(ptr, &v, sizeof(float)); break; }
            default: break;
        }
    };

    auto init_cudnn = [=, &h_E, &h_I,
                       &h_Gx_freq_M, &h_Gy_freq_M,
                       &h_Gx_spatial, &h_Gy_spatial]
        (void* ptr, const ArgDescriptor&, size_t idx) {
        switch (idx) {
            case 1:  memcpy(ptr, h_E.data(),          sizeof(float) * N_LARGE * N_LARGE); break;
            case 2:  memcpy(ptr, h_I.data(),          sizeof(float) * N_LARGE * N_LARGE); break;
            case 3:  memcpy(ptr, h_Gx_freq_M.data(),  sizeof(float) * FREQ_N_M * 2);      break;
            case 4:  memcpy(ptr, h_Gy_freq_M.data(),  sizeof(float) * FREQ_N_M * 2);      break;
            case 5:  memcpy(ptr, h_Gx_spatial.data(), sizeof(float) * kw * kh);           break;
            case 6:  memcpy(ptr, h_Gy_spatial.data(), sizeof(float) * kw * kh);           break;
            case 7:  { int v = p_N;  memcpy(ptr, &v, sizeof(int));   break; }
            case 8:  { int v = p_W;  memcpy(ptr, &v, sizeof(int));   break; }
            case 9:  { int v = p_e;  memcpy(ptr, &v, sizeof(int));   break; }
            case 10: { float v = p_dx; memcpy(ptr, &v, sizeof(float)); break; }
            default: break;
        }
    };

    PerfConfig perf;
    perf.measure_latency  = true;
    perf.warmup_runs      = 3;
    perf.bench_runs       = 10;
    perf.measure_memory   = false;

    tester.register_pair(
        "[" + size_tag + "] srk_forward vs srk_training (B=1)",
        fwd_oas_adapt<N_LARGE, FREQ_N_NS>,
        fwd_training_adapt<N_LARGE, FREQ_N_NS>,
        args_train,
        tol_train,
        init_train,
        perf);

    tester.register_pair(
        "[" + size_tag + "] srk_forward vs srk_cudnn (B=1)",
        fwd_oas_adapt<N_LARGE, FREQ_N_NS>,
        fwd_cudnn_adapt<N_LARGE, FREQ_N_NS>,
        args_cudnn,
        tol_cudnn,
        init_cudnn,
        perf);
}

// ─────────────────────────────────────────────────────────────────
//  Helper: precompute G-frequency at a given FFT size.
//  Returns host-side flat float buffer of length M*(M/2+1)*2.
// ─────────────────────────────────────────────────────────────────
static std::vector<float> precompute_G_freq_host(
    const std::vector<float>& h_G, int kw, int kh, int M)
{
    const int freq_n = M * (M / 2 + 1);
    cufftComplex* d_G_freq = nullptr;
    cudaMalloc(&d_G_freq, sizeof(cufftComplex) * freq_n);
    srkPrecomputeKernelFreq(h_G, kw, kh, M, d_G_freq, /*stream=*/nullptr);
    std::vector<float> h_G_freq(freq_n * 2);
    cudaMemcpy(h_G_freq.data(), d_G_freq,
               sizeof(float) * freq_n * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_G_freq);
    return h_G_freq;
}

// ─────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // ── Runtime parameters ─────────────────────────────────────────
    // Small correctness test (N1=256, N2=512 are compile-time template constants)
    const float SIGMA   = get_arg_float(argc, argv, "--sigma",     5.f);
    const int   B_TEST  = get_arg_int  (argc, argv, "--b_test",    1);
    // Big batched test
    const float SIGMA_BIG = get_arg_float(argc, argv, "--sigma_big", 10.f);
    const int   N_BIG     = get_arg_int  (argc, argv, "--n_big",     192);
    const int   B_BIG     = get_arg_int  (argc, argv, "--b_big",     3000);
    s_B_TEST = B_TEST;  // picked up by wrap_fwd_* static functions
    s_B_BIG  = B_BIG;   // picked up by wrap_big_* static functions

    const float R_W       = SRSRK_C_W * SIGMA;
    const int   EROSION   = (int)(R_W / DX);   // = (KW-1)/2; matches cuDNN valid conv
    const int   KW        = 2 * (int)(R_W / DX) + 1;
    const int   KH        = KW;
    const int   W1        = N1 - 2 * EROSION;  // = N1 - KW + 1
    const int   W2        = N2 - 2 * EROSION;

    const float R_W_BIG     = SRSRK_C_W * SIGMA_BIG;
    const int   EROSION_BIG = (int)(R_W_BIG / DX);   // = (KW_BIG-1)/2
    const int   KW_BIG      = 2 * (int)(R_W_BIG / DX) + 1;
    const int   KH_BIG      = KW_BIG;
    const int   W_BIG       = N_BIG - 2 * EROSION_BIG;
    const int   FREQ_N_BIG  = N_BIG * (N_BIG / 2 + 1);

    printf("Small test: sigma=%.1f  kw=%d  erosion=%d  N1=%d/W1=%d  N2=%d/W2=%d  B=%d\n",
           SIGMA, KW, EROSION, N1, W1, N2, W2, B_TEST);
    printf("Big   test: sigma=%.1f  kw=%d  erosion=%d  N=%d  W=%d  B=%d\n",
           SIGMA_BIG, KW_BIG, EROSION_BIG, N_BIG, W_BIG, B_BIG);

    cudaSetDevice(0);
    std::srand(42);

    // ── Kernels (Bessel K1 × Blackman) ────────────────────────────
    std::vector<float> h_Gx, h_Gy;
    make_srsrk_kernel_2d(h_Gx, h_Gy, KW, KH, DX, SIGMA);

    // Precompute G_freq at all required FFT sizes (host-side buffers).
    auto h_Gx_freq_M  = precompute_G_freq_host(h_Gx, KW, KH, M_TILE);
    auto h_Gy_freq_M  = precompute_G_freq_host(h_Gy, KW, KH, M_TILE);
    auto h_Gx_freq_N1 = precompute_G_freq_host(h_Gx, KW, KH, N1);
    auto h_Gy_freq_N1 = precompute_G_freq_host(h_Gy, KW, KH, N1);
    auto h_Gx_freq_N2 = precompute_G_freq_host(h_Gx, KW, KH, N2);
    auto h_Gy_freq_N2 = precompute_G_freq_host(h_Gy, KW, KH, N2);

    // ── Fixed-seed inputs E and I (Gaussian-process samples) ───────
    auto h_E_1 = generate_pattern(TestPatternConfig::gaussian_process(N1, 32.f, 42));
    auto h_I_1 = generate_pattern(TestPatternConfig::gaussian_process(N1, 32.f, 43));
    auto h_E_2 = generate_pattern(TestPatternConfig::gaussian_process(N2, 48.f, 44));
    auto h_I_2 = generate_pattern(TestPatternConfig::gaussian_process(N2, 48.f, 45));

    // ── Workspaces (one set per N) ─────────────────────────────────
    static SrkConvWorkspace<ARCH, M_TILE> conv_ws_1, conv_ws_2;
    static SrkTrainWorkspace              train_ws_1, train_ws_2;
    static SrkCudnnWorkspace              cudnn_ws_1, cudnn_ws_2;

    conv_ws_1.allocate(N1, W1);
    conv_ws_2.allocate(N2, W2);
    train_ws_1.allocate(N1, /*B_max=*/B_TEST);
    train_ws_2.allocate(N2, /*B_max=*/B_TEST);
    cudnn_ws_1.allocate(/*B=*/B_TEST, N1, KW, KH, W1);
    cudnn_ws_2.allocate(/*B=*/B_TEST, N2, KW, KH, W2);

    Ctx<N1, FREQ_N_1>::conv_ws  = &conv_ws_1;
    Ctx<N1, FREQ_N_1>::train_ws = &train_ws_1;
    Ctx<N1, FREQ_N_1>::cudnn_ws = &cudnn_ws_1;
    Ctx<N2, FREQ_N_2>::conv_ws  = &conv_ws_2;
    Ctx<N2, FREQ_N_2>::train_ws = &train_ws_2;
    Ctx<N2, FREQ_N_2>::cudnn_ws = &cudnn_ws_2;

    // ── Run the comparison tests ───────────────────────────────────
    FuncTester tester;

    // Tolerances: FFT vs FFT (different sizes) is essentially numerical FP
    // roundoff at the conv stage, then a multiply.  cuDNN vs FFT compares
    // GEMM-style spatial conv against FFT and can drift slightly more,
    // especially for the larger 512² case.
    register_size<N1, FREQ_N_1>(
        W1, KW, KH, tester,
        h_E_1, h_I_1,
        h_Gx_freq_M,  h_Gy_freq_M,
        h_Gx_freq_N1, h_Gy_freq_N1,
        h_Gx,         h_Gy,
        N1, W1, EROSION, DX,
        /*tol_train=*/ 5e-4f,
        /*tol_cudnn=*/ 1e-3f,
        "N=256");

    register_size<N2, FREQ_N_2>(
        W2, KW, KH, tester,
        h_E_2, h_I_2,
        h_Gx_freq_M,  h_Gy_freq_M,
        h_Gx_freq_N2, h_Gy_freq_N2,
        h_Gx,         h_Gy,
        N2, W2, EROSION, DX,
        /*tol_train=*/ 1e-3f,
        /*tol_cudnn=*/ 2e-3f,
        "N=512");

    // ── Big batched test: srk_training vs srk_cudnn ─────────────────
    // B=3000, KW=63 — runtime + memory comparison only.  Same input shape
    // (column-major [B × N × N]); R is [B × W × W].
    std::vector<float> h_Gx_BIG, h_Gy_BIG;
    make_srsrk_kernel_2d(h_Gx_BIG, h_Gy_BIG, KW_BIG, KH_BIG, DX, SIGMA_BIG);
    auto h_Gx_freq_BIG = precompute_G_freq_host(h_Gx_BIG, KW_BIG, KH_BIG, N_BIG);
    auto h_Gy_freq_BIG = precompute_G_freq_host(h_Gy_BIG, KW_BIG, KH_BIG, N_BIG);

    // Single shared (E_base, I_base) tiled across all B_BIG batches.  Cheap
    // init that still exercises both kernels at the right shape; we don't
    // need distinct samples since the per-test comparison is runtime/memory,
    // not statistical (the per-batch correctness comparison has already been
    // covered by the B=1 tests above).
    auto h_E_base = generate_pattern(
        TestPatternConfig::gaussian_process(N_BIG, 24.f, 60));
    auto h_I_base = generate_pattern(
        TestPatternConfig::gaussian_process(N_BIG, 24.f, 61));

    // Measure workspace footprint via cudaMemGetInfo — MemoryTracker
    // wouldn't see this because the workspaces live across all calls.
    size_t free_before = 0, free_after = 0, total_gpu = 0;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free_before, &total_gpu);
    big_train_ws.allocate(N_BIG, /*B_max=*/B_BIG);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free_after, &total_gpu);
    size_t big_train_bytes = free_before - free_after;

    cudaMemGetInfo(&free_before, &total_gpu);
    big_cudnn_ws.allocate(/*B=*/B_BIG, N_BIG, KW_BIG, KH_BIG, W_BIG);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free_after, &total_gpu);
    size_t big_cudnn_bytes = free_before - free_after;

    int   p_N_BIG  = N_BIG;
    int   p_W_BIG  = W_BIG;
    int   p_e_BIG  = EROSION_BIG;
    float p_dx_BIG = DX;

    auto args_big = std::vector<ArgDescriptor>{
        ArgDescriptor::output_device(DType::FLOAT32, Dims::make_2d(W_BIG * W_BIG, B_BIG),       "R"),
        ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_BIG * N_BIG, B_BIG),       "E"),
        ArgDescriptor::input_device (DType::FLOAT32, Dims::make_2d(N_BIG * N_BIG, B_BIG),       "I"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N_BIG * 2),             "Gx_freq_N"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(FREQ_N_BIG * 2),             "Gy_freq_N"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(KW_BIG * KH_BIG),            "Gx_spatial"),
        ArgDescriptor::param_device (DType::FLOAT32, Dims::make_1d(KW_BIG * KH_BIG),            "Gy_spatial"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                          "N"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                          "W"),
        ArgDescriptor::param_host   (DType::INT32,   Dims::make_1d(1),                          "erosion"),
        ArgDescriptor::param_host   (DType::FLOAT32, Dims::make_1d(1),                          "dx"),
    };

    auto init_big = [&, p_N_BIG, p_W_BIG, p_e_BIG, p_dx_BIG]
        (void* ptr, const ArgDescriptor&, size_t idx) {
        const size_t img_bytes = sizeof(float) * (size_t)N_BIG * N_BIG;
        switch (idx) {
            case 1:
                for (int b = 0; b < B_BIG; ++b)
                    memcpy(static_cast<float*>(ptr) + (size_t)b * N_BIG * N_BIG,
                           h_E_base.data(), img_bytes);
                break;
            case 2:
                for (int b = 0; b < B_BIG; ++b)
                    memcpy(static_cast<float*>(ptr) + (size_t)b * N_BIG * N_BIG,
                           h_I_base.data(), img_bytes);
                break;
            case 3:  memcpy(ptr, h_Gx_freq_BIG.data(), sizeof(float) * FREQ_N_BIG * 2); break;
            case 4:  memcpy(ptr, h_Gy_freq_BIG.data(), sizeof(float) * FREQ_N_BIG * 2); break;
            case 5:  memcpy(ptr, h_Gx_BIG.data(),      sizeof(float) * KW_BIG * KH_BIG); break;
            case 6:  memcpy(ptr, h_Gy_BIG.data(),      sizeof(float) * KW_BIG * KH_BIG); break;
            case 7:  { int v = p_N_BIG;  memcpy(ptr, &v, sizeof(int));   break; }
            case 8:  { int v = p_W_BIG;  memcpy(ptr, &v, sizeof(int));   break; }
            case 9:  { int v = p_e_BIG;  memcpy(ptr, &v, sizeof(int));   break; }
            case 10: { float v = p_dx_BIG; memcpy(ptr, &v, sizeof(float)); break; }
            default: break;
        }
    };

    PerfConfig perf_big;
    perf_big.measure_latency  = true;
    perf_big.warmup_runs      = 1;     // fewer runs — each call is heavy
    perf_big.bench_runs       = 3;
    perf_big.measure_memory   = true;
    perf_big.poll_interval_ms = 1;

    tester.register_pair(
        "[B=3000 N=192 KW=63] srk_training vs srk_cudnn",
        big_training_adapt, big_cudnn_adapt,
        args_big,
        /*tolerance=*/ 5e-3f,
        init_big,
        perf_big);

    printf("\n=== shrinkage_forward_training_test ===\n\n");
    auto results = tester.run_all();

    // ── Summary ────────────────────────────────────────────────────
    printf("\n=== Summary ===\n");
    int pass = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        bool ok = results[i].accuracy.passed;
        if (ok) ++pass;
        printf("  [%zu] max_err=%.3e  l2=%.3e  %s\n",
               i, results[i].accuracy.max_error,
               results[i].accuracy.l2_error,
               ok ? "PASS" : "FAIL");
    }
    printf("  %d / %zu pass\n", pass, results.size());

    // Big-test workspace sizes (queried via cudaMemGetInfo at allocate time).
    auto fmt_bytes = [](size_t b) -> std::string {
        char buf[64];
        if (b >= (1ull<<30)) snprintf(buf, sizeof(buf), "%.2f GB", b / double(1ull<<30));
        else if (b >= (1<<20)) snprintf(buf, sizeof(buf), "%.2f MB", b / double(1<<20));
        else                   snprintf(buf, sizeof(buf), "%zu B", b);
        return buf;
    };
    printf("\n--- Big-test workspace footprint (B=%d, N=%d, KW=%d) ---\n",
           B_BIG, N_BIG, KW_BIG);
    printf("  srk_training pre-allocated workspace : %s\n",
           fmt_bytes(big_train_bytes).c_str());
    printf("  srk_cudnn    pre-allocated workspace : %s\n",
           fmt_bytes(big_cudnn_bytes).c_str());

    // Cleanup
    cudnn_ws_1.release();  cudnn_ws_2.release();  big_cudnn_ws.release();
    train_ws_1.release();  train_ws_2.release();  big_train_ws.release();

    return (pass == (int)results.size()) ? 0 : 1;
}
