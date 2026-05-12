// main_cd_training.cu
// CD-accuracy test for srkTrainingForward.
//
//  [Ref]  op_ref()  — srkTrainingForward, FFT path (srk_training_forward.cuh)
//  [Test] op_test() — PLACEHOLDER; replace the body with your accelerated
//                     srkTrainingForward and recompile. See the comment block
//                     inside op_test() for the exact contract.
//
//  Section A  FuncTester pixel-accuracy check (random input, B patterns).
//  Section B  CD sweep on LNS_VERTICAL   patterns (center-row CD).
//  Section C  CD sweep on LNS_HORIZONTAL patterns (center-col CD).
//
//  Build: 'cd_training_test' target in CMakeLists.txt.
//  Run:   ./build/cd_training_test [--sigma 20.0] [--N 256] [--B 30]
//                                  [--lns_sig 8.0] [--lns_period 0.0]

#include "srk_training_forward.cuh"
#include "srk_forward.cuh"

#include <stTestCore/func_tester.h>
#include <stTestCore/adapter.h>
#include <stTestCore/dump.h>
#include <patternUtil/test_pattern.h>
#include <metricUtil/cd_measure_util.cuh>

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>

using namespace st::util;

static constexpr float DX_PX     = 1.0f;  // pixel spacing in pixel coords (kernel internal)
static constexpr float SRSRK_C_W = 3.0f;

// ── Kernel generation ────────────────────────────────────────────────────────
static float blackman_radial(float r, float Rw) {
    if (r<0.f||r>Rw) return 0.f;
    float t=r/Rw;
    return 0.42f+0.5f*cosf((float)M_PI*t)+0.08f*cosf(2.f*(float)M_PI*t);
}
static float bessel_I1_small(float x) {
    float y=(x/3.75f)*(x/3.75f);
    return x*(0.5f+y*(0.87890594f+y*(0.51498869f+y*(0.15084934f+
               y*(0.02658733f+y*(0.00301532f+y*0.00032411f))))));
}
static float bessel_K1(float x) {
    if (x<=0.f) return 1e30f;
    if (x<=2.f) {
        float y=x*x*0.25f, I1=bessel_I1_small(x);
        float p=1.f+y*(0.15443144f-y*(0.67278579f+y*(0.18156897f+
                  y*(0.01919402f+y*(0.00110404f+y*0.00004686f)))));
        return std::log(x*0.5f)*I1+p/x;
    }
    float y=2.f/x;
    float p=1.25331414f+y*(0.23498619f+y*(-0.03655620f+y*(0.01504268f+
              y*(-0.00780353f+y*(0.00325614f-y*0.00068245f)))));
    return std::exp(-x)/std::sqrt(x)*p;
}
static void make_kernel(std::vector<float>& Kx, std::vector<float>& Ky,
                        int kw, float dx, float sigma) {
    float gamma=1.f/sigma, Rw=SRSRK_C_W/gamma;
    float coeff=gamma/(2.f*(float)M_PI), eps=1e-6f*dx;
    int cx=kw/2;
    Kx.assign((size_t)kw*kw,0.f); Ky.assign((size_t)kw*kw,0.f);
    for (int i=0;i<kw;++i)
        for (int j=0;j<kw;++j) {
            float x=(i-cx)*dx, y=(j-cx)*dx, r=std::sqrt(x*x+y*y);
            if (r<eps||r>Rw) continue;
            float base=coeff*bessel_K1(gamma*r)*blackman_radial(r,Rw);
            Kx[j+i*kw]=base*(x/r);
            Ky[j+i*kw]=base*(y/r);
        }
}

// ── Global operator context ──────────────────────────────────────────────────
struct OpCtx {
    int N=0, W=0, EROSION=0, B=0;
    float dx=1.f;
    cufftComplex *d_Gx_freq=nullptr, *d_Gy_freq=nullptr;
    float *d_R_ref=nullptr, *d_R_test=nullptr;   // [B×W×W] col-major
    SrkTrainWorkspace ref_ws, test_ws;
    cudaStream_t stream=nullptr;
};
static OpCtx g_ctx;

// ── Op wrappers ──────────────────────────────────────────────────────────────
// Both take (const float* d_E [B×N×N], float* d_R [B×W×W]), col-major.

static void op_ref(const float* d_E, float* d_R) {
    auto& c = g_ctx;
    srkTrainingForward(d_E, d_E,
        c.d_Gx_freq, c.d_Gy_freq,
        c.N, c.W, c.EROSION, c.dx, c.B, c.stream,
        d_R, c.ref_ws);
}

static void op_test(const float* d_E, float* d_R) {
    // ── DEVELOPER: replace this block with your accelerated implementation ──
    //
    // Contract:
    //   d_E [in]  : [B × N × N] col-major float32, device pointer
    //                 E and I are the same (E=I=d_E).
    //   d_R [out] : [B × W × W] col-major float32, device pointer
    //                 R[b] = (conv(E_b,Gx)·∂E_b/∂x + conv(E_b,Gy)·∂E_b/∂y)
    //   B, N, W, EROSION, dx: read from g_ctx.
    //   Kernel freq buffers: g_ctx.d_Gx_freq, g_ctx.d_Gy_freq  [N×(N/2+1)].
    //   Workspace: g_ctx.test_ws — pre-allocated, same size as ref_ws.
    //   Stream: g_ctx.stream.
    //
    // Example (FFT path, identical to ref):
    //   srkTrainingForward(d_E, d_E,
    //       g_ctx.d_Gx_freq, g_ctx.d_Gy_freq,
    //       g_ctx.N, g_ctx.W, g_ctx.EROSION, g_ctx.dx,
    //       g_ctx.B, g_ctx.stream, d_R, g_ctx.test_ws);
    // ───────────────────────────────────────────────────────────────────────
    op_ref(d_E, d_R);  // placeholder: mirrors reference exactly
}

constexpr AdaptedFunc OP_REF  = FuncAdapter<op_ref,  const float*, float*>::call;
constexpr AdaptedFunc OP_TEST = FuncAdapter<op_test, const float*, float*>::call;

// ── GPU helpers ──────────────────────────────────────────────────────────────

// Crop center W×W from E[B×N×N] col-major → E_crop[B×W×W] col-major.
__global__ void crop_center(const float* E, float* E_crop,
                             int N, int W, int erosion) {
    int r = blockIdx.x*blockDim.x+threadIdx.x;  // row in W×W
    int c = blockIdx.y*blockDim.y+threadIdx.y;  // col in W×W
    int b = blockIdx.z;
    if (r>=W || c>=W) return;
    int gr=r+erosion, gc=c+erosion;
    E_crop[r+c*W+(size_t)b*W*W] = E[gr+gc*N+(size_t)b*N*N];
}

// shrink[b,r,c] = E_crop[b,r,c] + coeff * R[b,r,c]  (all col-major B×W×W).
__global__ void apply_shrinkage(const float* E_crop, const float* R,
                                float* shrink, float coeff, int W) {
    int r = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;
    int b = blockIdx.z;
    if (r>=W || c>=W) return;
    size_t i = r+c*W+(size_t)b*W*W;
    shrink[i] = E_crop[i] + coeff * R[i];
}

// Extract center row (row W/2) from [B×W×W] col-major → [B×W] row-major.
__global__ void extract_center_row(const float* src, float* dst, int W) {
    int oc = blockIdx.x*blockDim.x+threadIdx.x;
    int b  = blockIdx.y;
    if (oc>=W) return;
    dst[b*W+oc] = src[(W/2)+oc*W+(size_t)b*W*W];
}

// Extract center col (col W/2) from [B×W×W] col-major → [B×W] row-major.
__global__ void extract_center_col(const float* src, float* dst, int W) {
    int or_ = blockIdx.x*blockDim.x+threadIdx.x;
    int b   = blockIdx.y;
    if (or_>=W) return;
    dst[b*W+or_] = src[or_+(W/2)*W+(size_t)b*W*W];
}

// Upload B col-major N×N host images → device [B×N×N].
static void upload_patterns(float* d_E,
                             const std::vector<std::vector<float>>& imgs,
                             int N, cudaStream_t stream) {
    for (int b=0; b<(int)imgs.size(); ++b)
        cudaMemcpyAsync(d_E+(size_t)b*N*N, imgs[b].data(),
                        sizeof(float)*N*N, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}

// ── Dump helpers ─────────────────────────────────────────────────────────────
static void mkdir_p(const std::string& p) {
    for(size_t i=1;i<=p.size();++i)
        if(i==p.size()||p[i]=='/')  mkdir(p.substr(0,i).c_str(),0755);
}
static void dump_image(const std::string& path, float* d_ptr, int W,
                       const std::string& name) {
    dump_buffer(path, d_ptr, MemSpace::DEVICE,
                DType::FLOAT32, Dims::make_2d(W, W), name);
}

// ── CD sweep helper ──────────────────────────────────────────────────────────
struct CDSweepResult {
    float mean_err=0.f, max_err=0.f, rms_err=0.f;
    int   n=0;
};

static CDSweepResult run_cd_sweep(
    float* d_E,                    // [B×N×N] pre-filled on device
    float* d_R_ref_out,            // [B×W×W]
    float* d_R_test_out,           // [B×W×W]
    float* d_E_crop,               // [B×W×W] scratch (center crop of E)
    float* d_shrink,               // [B×W×W] scratch (E_crop + c*R)
    float* d_sig_ref,              // [B×W] scratch
    float* d_sig_test,             // [B×W] scratch
    int B, int N, int W, int erosion,
    float coeff_c,                 // shrinkage coefficient
    float dx_nm,                   // physical pixel pitch [nm] for unit conversion
    bool center_row,               // true=extract row, false=extract col
    const char* label,
    cudaStream_t stream,
    const std::string& dump_dir,
    int dump_max)
{
    // Crop center W×W from E (shared for both ref and test).
    { dim3 blk(16,16), grd((W+15)/16,(W+15)/16,B);
      crop_center<<<grd,blk,0,stream>>>(d_E, d_E_crop, N, W, erosion); }

    op_ref (d_E, d_R_ref_out);
    op_test(d_E, d_R_test_out);

    // shrinkage ref: signal = E_crop + coeff * R_ref
    { dim3 blk(16,16), grd((W+15)/16,(W+15)/16,B);
      apply_shrinkage<<<grd,blk,0,stream>>>(d_E_crop, d_R_ref_out,  d_shrink, coeff_c, W);
      cudaStreamSynchronize(stream); }

    // dump E_crop and shrink_ref for first dump_max batches
    if (!dump_dir.empty()) {
        mkdir_p(dump_dir);
        const int nd = std::min(dump_max, B);
        for (int b=0; b<nd; ++b) {
            char buf[256];
            snprintf(buf,sizeof(buf),"%s/b%d_E.txt",      dump_dir.c_str(), b);
            dump_image(buf, d_E_crop + (size_t)b*W*W, W, "E_crop");
            snprintf(buf,sizeof(buf),"%s/b%d_shrink_ref.txt", dump_dir.c_str(), b);
            dump_image(buf, d_shrink  + (size_t)b*W*W, W, "shrink_ref");
        }
    }

    { dim3 blk1(128), grd1((W+127)/128,B);
      if (center_row) extract_center_row<<<grd1,blk1,0,stream>>>(d_shrink, d_sig_ref, W);
      else            extract_center_col<<<grd1,blk1,0,stream>>>(d_shrink, d_sig_ref, W); }

    // shrinkage test: signal = E_crop + coeff * R_test
    { dim3 blk(16,16), grd((W+15)/16,(W+15)/16,B);
      apply_shrinkage<<<grd,blk,0,stream>>>(d_E_crop, d_R_test_out, d_shrink, coeff_c, W);
      cudaStreamSynchronize(stream); }

    // dump shrink_test for first dump_max batches
    if (!dump_dir.empty()) {
        const int nd = std::min(dump_max, B);
        for (int b=0; b<nd; ++b) {
            char buf[256];
            snprintf(buf,sizeof(buf),"%s/b%d_shrink_test.txt", dump_dir.c_str(), b);
            dump_image(buf, d_shrink + (size_t)b*W*W, W, "shrink_test");
        }
    }

    { dim3 blk1(128), grd1((W+127)/128,B);
      if (center_row) extract_center_row<<<grd1,blk1,0,stream>>>(d_shrink, d_sig_test, W);
      else            extract_center_col<<<grd1,blk1,0,stream>>>(d_shrink, d_sig_test, W); }

    cudaStreamSynchronize(stream);

    auto cd_ref  = MeasureCD_Batched(d_sig_ref,  B, W, 16, 8, stream);
    auto cd_test = MeasureCD_Batched(d_sig_test, B, W, 16, 8, stream);

    CDSweepResult res; res.n=B;
    printf("\n  %s\n", label);
    printf("  %-4s  %12s  %12s  %12s\n","pat","CD_ref[nm]","CD_test[nm]","|diff|[nm]");
    for (int b=0; b<B; ++b) {
        float r=cd_ref.cd[b].cd_px * dx_nm;
        float t=cd_test.cd[b].cd_px * dx_nm;
        float d=std::fabs(r-t);
        res.mean_err += d; res.max_err = std::max(res.max_err,d); res.rms_err += d*d;
        printf("  %-4d  %12.4f  %12.4f  %12.4f\n", b, r, t, d);
    }
    res.mean_err /= B; res.rms_err = std::sqrt(res.rms_err/B);
    printf("  mean|diff|=%.4f  max|diff|=%.4f  rms|diff|=%.4f  [nm]\n",
           res.mean_err, res.max_err, res.rms_err);
    return res;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
static float get_arg_float(int argc,char**argv,const char*key,float def) {
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])==key) return std::stof(argv[i+1]);
    return def;
}
static int get_arg_int(int argc,char**argv,const char*key,int def) {
    for(int i=1;i+1<argc;++i) if(std::string(argv[i])==key) return std::stoi(argv[i+1]);
    return def;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    const float SIGMA      = get_arg_float(argc,argv,"--sigma",    20.0f);
    const int   N          = get_arg_int  (argc,argv,"--N",        256);
    const int   B          = get_arg_int  (argc,argv,"--B",        30);
    const float LNS_SIG    = get_arg_float(argc,argv,"--lns_sig",  8.0f);
    const float LNS_PERIOD = get_arg_float(argc,argv,"--lns_period",0.0f);
    const float COEFF_C    = get_arg_float(argc,argv,"--coeff_c",  0.1f);
    const int   GP_ITERS   = get_arg_int  (argc,argv,"--gp_iters", 10);

    const float DX_NM      = get_arg_float(argc,argv,"--dx",        10.0f);
    const int   KW         = 2*(int)(SRSRK_C_W*SIGMA/DX_PX)+1;
    const int   EROSION    = (KW-1)/2;
    const int   W          = N - 2*EROSION;

    if (W<=0) {
        fprintf(stderr,"Erosion=%d too large for N=%d (KW=%d). "
                       "Increase N or reduce sigma.\n", EROSION, N, KW);
        return 1;
    }

    printf("sigma=%.1f px  KW=%d  erosion=%d  N=%d  W=%d  B=%d  dx=%.1f nm/px\n",
           SIGMA, KW, EROSION, N, W, B, DX_NM);
    printf("LNS: sigma_px=%.1f  period_px=%.1f (0=auto)  coeff_c=%.3f\n\n",
           LNS_SIG, LNS_PERIOD, COEFF_C);

    cudaSetDevice(0);
    cudaStream_t stream; cudaStreamCreate(&stream);

    // ── Build frequency-domain kernels ────────────────────────────────────
    std::vector<float> h_Kx, h_Ky;
    make_kernel(h_Kx, h_Ky, KW, DX_PX, SIGMA);

    const int FREQ_N = N*(N/2+1);
    cufftComplex *d_Gx_freq=nullptr, *d_Gy_freq=nullptr;
    cudaMalloc(&d_Gx_freq, sizeof(cufftComplex)*FREQ_N);
    cudaMalloc(&d_Gy_freq, sizeof(cufftComplex)*FREQ_N);
    srkPrecomputeKernelFreq(h_Kx, KW, KW, N, d_Gx_freq, stream);
    srkPrecomputeKernelFreq(h_Ky, KW, KW, N, d_Gy_freq, stream);

    // ── Populate global context ───────────────────────────────────────────
    g_ctx.N=N; g_ctx.W=W; g_ctx.EROSION=EROSION; g_ctx.B=B;
    g_ctx.dx=DX_PX; g_ctx.stream=stream;
    g_ctx.d_Gx_freq=d_Gx_freq; g_ctx.d_Gy_freq=d_Gy_freq;

    cudaMalloc(&g_ctx.d_R_ref,  sizeof(float)*(size_t)B*W*W);
    cudaMalloc(&g_ctx.d_R_test, sizeof(float)*(size_t)B*W*W);
    g_ctx.ref_ws.allocate(N, B, stream);
    g_ctx.test_ws.allocate(N, B, stream);
    cudaStreamSynchronize(stream);

    // ── Section A: FuncTester pixel accuracy (random input) ──────────────
    printf("══════════════════════════════════════════════════\n");
    printf("  Section A  Pixel accuracy (random input, GP mode)\n");
    printf("══════════════════════════════════════════════════\n");
    {
        FuncTester tester;
        GpConfig gp;
        gp.enabled    = true;
        gp.cov        = Covariance2D::isotropic(static_cast<float>(EROSION)/2.f);
        gp.iterations = GP_ITERS;
        gp.input_arg  = 0;

        tester.register_pair(
            "op_ref vs op_test  (N=" + std::to_string(N) + ", B=" +
                std::to_string(B) + ", GP random)",
            OP_REF, OP_TEST,
            {
                ArgDescriptor::input_device (DType::FLOAT32,
                    Dims::make_1d((size_t)B*N*N), "E"),
                ArgDescriptor::output_device(DType::FLOAT32,
                    Dims::make_1d((size_t)B*W*W), "R"),
            },
            /*tolerance=*/-1.0,
            /*init_fn=*/nullptr,
            /*perf=*/{.measure_latency=true, .warmup_runs=2, .bench_runs=5,
                      .measure_memory=false},
            gp,
            /*dump_dir=*/"");
        tester.run_all();
    }

    // ── Section B & C: CD sweeps ──────────────────────────────────────────
    printf("\n══════════════════════════════════════════════════\n");
    printf("  Section B+C  CD sweeps (B=%d LNS patterns)\n", B);
    printf("══════════════════════════════════════════════════\n");

    // Allocate shared device buffers for the CD sections.
    float *d_E=nullptr, *d_E_crop=nullptr, *d_shrink=nullptr;
    float *d_sig_ref=nullptr, *d_sig_test=nullptr;
    cudaMalloc(&d_E,        sizeof(float)*(size_t)B*N*N);
    cudaMalloc(&d_E_crop,   sizeof(float)*(size_t)B*W*W);
    cudaMalloc(&d_shrink,   sizeof(float)*(size_t)B*W*W);
    cudaMalloc(&d_sig_ref,  sizeof(float)*(size_t)B*W);
    cudaMalloc(&d_sig_test, sizeof(float)*(size_t)B*W);

    // Generate B LNS patterns with varying sigma (sweep).
    auto make_lns_patterns = [&](bool vertical) {
        std::vector<std::vector<float>> imgs(B);
        const float sig_lo=LNS_SIG*0.7f, sig_hi=LNS_SIG*1.4f;
        for (int b=0;b<B;++b) {
            float sig = (B>1) ? sig_lo+(sig_hi-sig_lo)*b/(B-1) : LNS_SIG;
            TestPatternConfig cfg = vertical
                ? TestPatternConfig::lns_vertical  (N, sig, LNS_PERIOD)
                : TestPatternConfig::lns_horizontal(N, sig, LNS_PERIOD);
            imgs[b] = generate_pattern(cfg);
        }
        return imgs;
    };

    const int DUMP_MAX = get_arg_int(argc, argv, "--dump_max", 4);

    // Section B: LNS_VERTICAL → measure horizontal CD (center row).
    {
        auto imgs = make_lns_patterns(/*vertical=*/true);
        upload_patterns(d_E, imgs, N, stream);
        run_cd_sweep(d_E, g_ctx.d_R_ref, g_ctx.d_R_test,
                     d_E_crop, d_shrink, d_sig_ref, d_sig_test,
                     B, N, W, EROSION, COEFF_C, DX_NM, /*center_row=*/true,
                     "LNS_VERTICAL — center-row CD  [signal = E + c·R]",
                     stream, "dump_cd/lns_vert", DUMP_MAX);
    }

    // Section C: LNS_HORIZONTAL → measure vertical CD (center column).
    {
        auto imgs = make_lns_patterns(/*vertical=*/false);
        upload_patterns(d_E, imgs, N, stream);
        run_cd_sweep(d_E, g_ctx.d_R_ref, g_ctx.d_R_test,
                     d_E_crop, d_shrink, d_sig_ref, d_sig_test,
                     B, N, W, EROSION, COEFF_C, DX_NM, /*center_row=*/false,
                     "LNS_HORIZONTAL — center-col CD  [signal = E + c·R]",
                     stream, "dump_cd/lns_horiz", DUMP_MAX);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    cudaFree(d_E); cudaFree(d_E_crop); cudaFree(d_shrink);
    cudaFree(d_sig_ref); cudaFree(d_sig_test);
    g_ctx.ref_ws.release(stream);
    g_ctx.test_ws.release(stream);
    cudaFree(d_Gx_freq); cudaFree(d_Gy_freq);
    cudaFree(g_ctx.d_R_ref); cudaFree(g_ctx.d_R_test);
    cudaStreamDestroy(stream);
    return 0;
}
