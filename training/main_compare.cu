// main_compare.cu
// FFT (srkTrainingForward, full 256×256) vs Strip cuDNN (121×254 → 1×134)
// Vertical L&S input; compare center horizontal line; B=3000 timing+memory.
//
// Strip rationale: for a single horizontal output row at N/2, cuDNN only
// needs the (2·erosion+1) input rows the kernel touches, not the full image.
//   strip height SH = KH = 2·EROSION_CNN+1 = 121
//   strip width  NW = N - 2·CROP = 254       (1-px column crop for alignment)
//   cuDNN output: 1 × W                       (exactly one row)

#include "srk_training_forward.cuh"   // FFT path + srk_train_grad_*
#include "srk_cudnn_forward.cuh"      // CUDNN_CHECK, srk_cm_to_rm_2d
#include "srk_forward.cuh"            // srkPrecomputeKernelFreq

#include <patternUtil/test_pattern.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <sys/stat.h>

using namespace st::util;

static constexpr float DX        = 1.0f;
static constexpr float SRSRK_C_W = 3.0f;

// ── Kernel: Bessel K1 × Blackman ─────────────────────────────────────────
static float blackman_radial(float r, float Rw) {
    if (r < 0.f || r > Rw) return 0.f;
    float t = r / Rw;
    return 0.42f + 0.5f*cosf((float)M_PI*t) + 0.08f*cosf(2.f*(float)M_PI*t);
}
static float bessel_I1_small(float x) {
    float y = (x/3.75f)*(x/3.75f);
    return x*(0.5f+y*(0.87890594f+y*(0.51498869f+y*(0.15084934f+
               y*(0.02658733f+y*(0.00301532f+y*0.00032411f))))));
}
static float bessel_K1(float x) {
    if (x <= 0.f) return 1e30f;
    if (x <= 2.f) {
        float y = x*x*0.25f, I1 = bessel_I1_small(x);
        float p = 1.f+y*(0.15443144f-y*(0.67278579f+y*(0.18156897f+
                  y*(0.01919402f+y*(0.00110404f+y*0.00004686f)))));
        return std::log(x*0.5f)*I1 + p/x;
    }
    float y = 2.f/x;
    float p = 1.25331414f+y*(0.23498619f+y*(-0.03655620f+y*(0.01504268f+
              y*(-0.00780353f+y*(0.00325614f-y*0.00068245f)))));
    return std::exp(-x)/std::sqrt(x)*p;
}
static void make_kernel(std::vector<float>& Kx, std::vector<float>& Ky,
                        int kw, int kh, float dx, float sigma) {
    float gamma = 1.f/sigma, Rw = SRSRK_C_W/gamma;
    float coeff = gamma/(2.f*(float)M_PI), eps = 1e-6f*dx;
    int cx = kw/2, cy = kh/2;
    Kx.assign((size_t)kw*kh, 0.f);
    Ky.assign((size_t)kw*kh, 0.f);
    for (int i = 0; i < kw; ++i)
        for (int j = 0; j < kh; ++j) {
            float x=(i-cx)*dx, y=(j-cy)*dx, r=std::sqrt(x*x+y*y);
            if (r<eps || r>Rw) continue;
            float base = coeff*bessel_K1(gamma*r)*blackman_radial(r,Rw);
            Kx[j+i*kh] = base*(x/r);
            Ky[j+i*kh] = base*(y/r);
        }
}

// ── Strip extraction ──────────────────────────────────────────────────────
// [B × N × N] col-major → [B × SH × NW] col-major
// Rows [row_start, row_start+SH), cols [col_crop, col_crop+NW)
__global__ void srk_extract_strip(
    const float* __restrict__ full,
    float* __restrict__       strip,
    int N, int SH, int NW, int row_start, int col_crop)
{
    int sr = blockIdx.x*blockDim.x + threadIdx.x;
    int sc = blockIdx.y*blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (sr >= SH || sc >= NW) return;
    strip[sr + sc*SH + b*SH*NW] =
        full[(sr+row_start) + (sc+col_crop)*N + b*N*N];
}

// ── Transpose: col-major [B×H×W] → row-major [B×H×W] ────────────────────
__global__ void srk_cm_to_rm_hw(const float* src, float* dst, int H, int W) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    int c=blockIdx.y*blockDim.y+threadIdx.y;
    int b=blockIdx.z;
    if (r>=H || c>=W) return;
    dst[r*W+c+b*H*W] = src[r+c*H+b*H*W];
}

// ── Strip gradient (non-square col-major [B×H×W]) ─────────────────────────
__global__ void srk_strip_grad_x(const float* I, float* out, int H, int W) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    int c=blockIdx.y*blockDim.y+threadIdx.y;
    int b=blockIdx.z;
    if (r>=H || c>=W) return;
    int off=b*H*W;
    float l=(c>0)   ? I[off+r+(c-1)*H] : 0.f;
    float ri=(c<W-1)? I[off+r+(c+1)*H] : 0.f;
    out[off+r+c*H] = (ri-l)*0.5f;
}
__global__ void srk_strip_grad_y(const float* I, float* out, int H, int W) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    int c=blockIdx.y*blockDim.y+threadIdx.y;
    int b=blockIdx.z;
    if (r>=H || c>=W) return;
    int off=b*H*W;
    float u=(r>0)   ? I[off+(r-1)+c*H] : 0.f;
    float d=(r<H-1) ? I[off+(r+1)+c*H] : 0.f;
    out[off+r+c*H] = (d-u)*0.5f;
}

// ── Strip combine (1-row output per batch) ────────────────────────────────
// ux_rm/uy_rm: [B×W] (cuDNN output, alpha=dx² already applied)
// dIdx/dIdy:   [B×SH×NW] col-major strip gradients
// R:           [B×W] single-row result
__global__ void srk_strip_combine(
    const float* ux, const float* uy,
    const float* dIdx, const float* dIdy,
    float* R,
    int W, int SH, int NW, int e)   // e = EROSION_CNN
{
    int oc=blockIdx.x*blockDim.x+threadIdx.x;
    int b =blockIdx.y;
    if (oc>=W) return;
    int gi = e + (oc+e)*SH + b*SH*NW;   // grad at (row=e, col=oc+e) in strip
    int oi = oc + b*W;
    R[oi] = ux[oi]*dIdx[gi] + uy[oi]*dIdy[gi];
}

// ── Strip cuDNN workspace ─────────────────────────────────────────────────
struct StripCudnnWs {
    cudnnHandle_t                handle    = nullptr;
    cudnnTensorDescriptor_t      in_desc   = nullptr;  // B×1×SH×NW
    cudnnTensorDescriptor_t      out_desc  = nullptr;  // B×1×1×W
    cudnnFilterDescriptor_t      filt_desc = nullptr;  // 1×1×KH×KW
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t    algo      = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    void*  d_algo_ws   = nullptr;
    size_t algo_ws_bytes = 0;

    float* d_E_rm  = nullptr;  // [B×SH×NW] row-major
    float* d_Gx_rm = nullptr;  // [KH×KW]
    float* d_Gy_rm = nullptr;
    float* d_ux_rm = nullptr;  // [B×W] (1-row cuDNN output, rm==cm)
    float* d_uy_rm = nullptr;
    float* d_dIdx  = nullptr;  // [B×SH×NW] col-major strip gradients
    float* d_dIdy  = nullptr;

    int B, SH, NW, KH, KW, W;

    void allocate(int B_, int SH_, int NW_, int KH_, int KW_, int W_,
                  cudaStream_t stream = nullptr) {
        B=B_; SH=SH_; NW=NW_; KH=KH_; KW=KW_; W=W_;
        CUDNN_CHECK(cudnnCreate(&handle));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, 1, SH, NW));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, 1, 1, W));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filt_desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, KH, KW));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc, 0,0, 1,1, 1,1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

        int nO,cO,hO,wO;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
            conv_desc, in_desc, filt_desc, &nO, &cO, &hO, &wO));
        if (nO!=B || cO!=1 || hO!=1 || wO!=W) {
            fprintf(stderr,"cuDNN out dim mismatch: got (%d,%d,%d,%d), want (%d,1,1,%d)\n",
                    nO,cO,hO,wO, B,W);
            std::abort();
        }

        int ret=0; cudnnConvolutionFwdAlgoPerf_t perfs[8];
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle, in_desc, filt_desc, conv_desc, out_desc, 8, &ret, perfs));
        algo = perfs[0].algo;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle, in_desc, filt_desc, conv_desc, out_desc, algo, &algo_ws_bytes));
        if (algo_ws_bytes > 0) cudaMalloc(&d_algo_ws, algo_ws_bytes);

        cudaMallocAsync(&d_E_rm,  sizeof(float)*(size_t)B*SH*NW, stream);
        cudaMallocAsync(&d_Gx_rm, sizeof(float)*(size_t)KH*KW,   stream);
        cudaMallocAsync(&d_Gy_rm, sizeof(float)*(size_t)KH*KW,   stream);
        cudaMallocAsync(&d_ux_rm, sizeof(float)*(size_t)B*W,     stream);
        cudaMallocAsync(&d_uy_rm, sizeof(float)*(size_t)B*W,     stream);
        cudaMallocAsync(&d_dIdx,  sizeof(float)*(size_t)B*SH*NW, stream);
        cudaMallocAsync(&d_dIdy,  sizeof(float)*(size_t)B*SH*NW, stream);
    }

    void release(cudaStream_t stream = nullptr) {
        if (d_algo_ws) { cudaFree(d_algo_ws); d_algo_ws=nullptr; }
        if (d_E_rm)   cudaFreeAsync(d_E_rm,  stream);
        if (d_Gx_rm)  cudaFreeAsync(d_Gx_rm, stream);
        if (d_Gy_rm)  cudaFreeAsync(d_Gy_rm, stream);
        if (d_ux_rm)  cudaFreeAsync(d_ux_rm, stream);
        if (d_uy_rm)  cudaFreeAsync(d_uy_rm, stream);
        if (d_dIdx)   cudaFreeAsync(d_dIdx,  stream);
        if (d_dIdy)   cudaFreeAsync(d_dIdy,  stream);
        if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
        if (filt_desc) cudnnDestroyFilterDescriptor(filt_desc);
        if (out_desc)  cudnnDestroyTensorDescriptor(out_desc);
        if (in_desc)   cudnnDestroyTensorDescriptor(in_desc);
        if (handle)    cudnnDestroy(handle);
    }
};

// ── Strip cuDNN forward ───────────────────────────────────────────────────
static void stripCudnnForward(
    const float* d_strip,      // [B×SH×NW] col-major (E and I, same here)
    float* d_R,                // [B×W]  single-row output
    StripCudnnWs& ws,
    int erosion_cnn,
    cudaStream_t stream)
{
    const float alpha=DX*DX, beta=0.f;
    cudnnSetStream(ws.handle, stream);

    // E: col-major → row-major
    { dim3 blk(16,16), grd((ws.SH+15)/16,(ws.NW+15)/16,ws.B);
      srk_cm_to_rm_hw<<<grd,blk,0,stream>>>(d_strip, ws.d_E_rm, ws.SH, ws.NW); }

    // conv Gx
    CUDNN_CHECK(cudnnConvolutionForward(ws.handle, &alpha,
        ws.in_desc, ws.d_E_rm, ws.filt_desc, ws.d_Gx_rm,
        ws.conv_desc, ws.algo, ws.d_algo_ws, ws.algo_ws_bytes,
        &beta, ws.out_desc, ws.d_ux_rm));

    // conv Gy
    CUDNN_CHECK(cudnnConvolutionForward(ws.handle, &alpha,
        ws.in_desc, ws.d_E_rm, ws.filt_desc, ws.d_Gy_rm,
        ws.conv_desc, ws.algo, ws.d_algo_ws, ws.algo_ws_bytes,
        &beta, ws.out_desc, ws.d_uy_rm));

    // strip gradients
    { dim3 blk(16,16), grd((ws.SH+15)/16,(ws.NW+15)/16,ws.B);
      srk_strip_grad_x<<<grd,blk,0,stream>>>(d_strip, ws.d_dIdx, ws.SH, ws.NW);
      srk_strip_grad_y<<<grd,blk,0,stream>>>(d_strip, ws.d_dIdy, ws.SH, ws.NW); }

    // combine
    { dim3 blk(128), grd((ws.W+127)/128, ws.B);
      srk_strip_combine<<<grd,blk,0,stream>>>(
          ws.d_ux_rm, ws.d_uy_rm, ws.d_dIdx, ws.d_dIdy,
          d_R, ws.W, ws.SH, ws.NW, erosion_cnn); }

    cudaStreamSynchronize(stream);
}

// ── Helpers ───────────────────────────────────────────────────────────────
struct Timer {
    cudaEvent_t s_, e_;
    Timer()  { cudaEventCreate(&s_); cudaEventCreate(&e_); }
    ~Timer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }
    void  start() { cudaEventRecord(s_); }
    float stop_ms() {
        cudaEventRecord(e_); cudaEventSynchronize(e_);
        float ms; cudaEventElapsedTime(&ms, s_, e_); return ms;
    }
};
static void mkdir_p(const std::string& path) {
    for (size_t i=1; i<=path.size(); ++i)
        if (i==path.size() || path[i]=='/')
            mkdir(path.substr(0,i).c_str(), 0755);
}
static std::string fmt_bytes(size_t b) {
    char buf[64];
    if      (b>=(1ull<<30)) snprintf(buf,sizeof(buf),"%.2f GB",b/double(1ull<<30));
    else if (b>=(1<<20))    snprintf(buf,sizeof(buf),"%.2f MB",b/double(1<<20));
    else                    snprintf(buf,sizeof(buf),"%zu B",  b);
    return buf;
}

// ─────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    float sigma   = 20.0f;
    float lns_sig = 8.0f;
    int   B       = 3000;
    int   NWARM   = 2;
    int   NRUNS   = 5;
    for (int i=1;i+1<argc;++i) {
        if (std::string(argv[i])=="--sigma")   sigma   = std::stof(argv[i+1]);
        if (std::string(argv[i])=="--b")       B       = std::stoi(argv[i+1]);
        if (std::string(argv[i])=="--nruns")   NRUNS   = std::stoi(argv[i+1]);
        if (std::string(argv[i])=="--lns_sig") lns_sig = std::stof(argv[i+1]);
    }

    const int   N           = 256;
    const float R_W         = SRSRK_C_W * sigma;
    const int   KW          = 2*(int)(R_W/DX) + 1;
    const int   KH          = KW;
    const int   EROSION     = (KW-1)/2;              // = 60; same for both paths
    const int   N_CNN       = N;                     // no crop needed
    const int   SH          = KH;                    // strip height = 121
    const int   W           = N - 2*EROSION;         // 136
    const int   ROW_STRIP   = N/2 - EROSION;         // start row in full img: 68
    // cuDNN output size check: SH - KH + 1 = 1, N_CNN - KW + 1 = W
    const int   ROW_OUT     = N/2 - EROSION;         // center row in output: 68 (both paths)

    printf("sigma=%.1f  KW=%d  R_w=%.0f\n", sigma, KW, R_W);
    printf("FFT   : N=%d  erosion=%d  output %d×%d\n", N, EROSION, W, W);
    printf("Strip : SH=%d  NW=%d  erosion=%d  output 1×%d\n",
           SH, N_CNN, EROSION, W);
    printf("Comparing: FFT row %d  vs  Strip row 0  (both = input row N/2=%d)\n",
           ROW_OUT, N/2);
    printf("B=%d  NRUNS=%d\n\n", B, NRUNS);

    cudaSetDevice(0);

    // ── Pattern + kernel ──────────────────────────────────────────────────
    auto h_img = generate_pattern(TestPatternConfig::lns_vertical(N, lns_sig));

    std::vector<float> h_Gx, h_Gy;
    make_kernel(h_Gx, h_Gy, KW, KH, DX, sigma);

    // ── Persistent device buffers ─────────────────────────────────────────
    float *d_E_full  = nullptr;   // [B × N × N] col-major
    float *d_E_strip = nullptr;   // [B × SH × NW] col-major
    float *d_R_fft   = nullptr;   // [B × W × W] col-major
    float *d_R_cnn   = nullptr;   // [B × W]      single-row
    cudaMalloc(&d_E_full,  sizeof(float)*(size_t)B*N*N);
    cudaMalloc(&d_E_strip, sizeof(float)*(size_t)B*SH*N_CNN);
    cudaMalloc(&d_R_fft,   sizeof(float)*(size_t)B*W*W);
    cudaMalloc(&d_R_cnn,   sizeof(float)*(size_t)B*W);

    // Fill d_E_full (same image for every batch)
    for (int b=0; b<B; ++b)
        cudaMemcpy(d_E_full+(size_t)b*N*N, h_img.data(),
                   sizeof(float)*N*N, cudaMemcpyHostToDevice);

    // Extract strip: rows [ROW_STRIP, ROW_STRIP+SH), cols [0, NW)
    { dim3 blk(16,16), grd((SH+15)/16,(N_CNN+15)/16,B);
      srk_extract_strip<<<grd,blk>>>(
          d_E_full, d_E_strip, N, SH, N_CNN, ROW_STRIP, /*col_crop=*/0); }
    cudaDeviceSynchronize();

    // Freq-domain kernels for FFT path
    const int FREQ_N = N*(N/2+1);
    cufftComplex *d_Gx_freq=nullptr, *d_Gy_freq=nullptr;
    cudaMalloc(&d_Gx_freq, sizeof(cufftComplex)*FREQ_N);
    cudaMalloc(&d_Gy_freq, sizeof(cufftComplex)*FREQ_N);
    srkPrecomputeKernelFreq(h_Gx, KW, KH, N, d_Gx_freq, nullptr);
    srkPrecomputeKernelFreq(h_Gy, KW, KH, N, d_Gy_freq, nullptr);

    // Spatial kernels for cuDNN (row-major, uploaded via the strip ws)
    float *d_Gx_sp=nullptr, *d_Gy_sp=nullptr;
    cudaMalloc(&d_Gx_sp, sizeof(float)*KW*KH);
    cudaMalloc(&d_Gy_sp, sizeof(float)*KW*KH);
    cudaMemcpy(d_Gx_sp, h_Gx.data(), sizeof(float)*KW*KH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gy_sp, h_Gy.data(), sizeof(float)*KW*KH, cudaMemcpyHostToDevice);

    // ──────────────────────────────────────────────────────────────────────
    //  FFT path
    // ──────────────────────────────────────────────────────────────────────
    size_t free0, free1, total_gpu;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free0, &total_gpu);

    SrkTrainWorkspace fft_ws;
    fft_ws.allocate(N, B);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free1, &total_gpu);
    size_t fft_ws_bytes = free0 - free1;

    for (int i=0;i<NWARM;++i)
        srkTrainingForward(d_E_full, d_E_full,
            d_Gx_freq, d_Gy_freq,
            N, W, EROSION, DX, B, nullptr, d_R_fft, fft_ws);

    Timer fft_timer; fft_timer.start();
    for (int i=0;i<NRUNS;++i)
        srkTrainingForward(d_E_full, d_E_full,
            d_Gx_freq, d_Gy_freq,
            N, W, EROSION, DX, B, nullptr, d_R_fft, fft_ws);
    float fft_ms = fft_timer.stop_ms() / NRUNS;

    fft_ws.release();

    // ──────────────────────────────────────────────────────────────────────
    //  Strip cuDNN path
    // ──────────────────────────────────────────────────────────────────────
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free0, &total_gpu);

    StripCudnnWs cnn_ws;
    cnn_ws.allocate(B, SH, N_CNN, KH, KW, W);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free1, &total_gpu);
    size_t cnn_ws_bytes = free0 - free1;

    // Upload spatial kernels (row-major) into strip workspace
    { dim3 blk(16,16), grd((KH+15)/16,(KW+15)/16);
      srk_cm_to_rm_2d<<<grd,blk>>>(d_Gx_sp, cnn_ws.d_Gx_rm, KH, KW);
      srk_cm_to_rm_2d<<<grd,blk>>>(d_Gy_sp, cnn_ws.d_Gy_rm, KH, KW); }
    cudaDeviceSynchronize();

    for (int i=0;i<NWARM;++i)
        stripCudnnForward(d_E_strip, d_R_cnn, cnn_ws, EROSION, nullptr);

    Timer cnn_timer; cnn_timer.start();
    for (int i=0;i<NRUNS;++i)
        stripCudnnForward(d_E_strip, d_R_cnn, cnn_ws, EROSION, nullptr);
    float cnn_ms = cnn_timer.stop_ms() / NRUNS;

    cnn_ws.release();

    // ──────────────────────────────────────────────────────────────────────
    //  Signal comparison (batch 0)
    // ──────────────────────────────────────────────────────────────────────
    // FFT batch0 center row: col-major [W×W], row ROW_OUT, col oc
    //   offset = ROW_OUT + oc*W
    // Strip cuDNN batch0 single row: [W], col oc  → offset = oc
    std::vector<float> h_row_fft(W), h_row_cnn(W);
    for (int oc=0; oc<W; ++oc) {
        cudaMemcpy(&h_row_fft[oc],
                   d_R_fft + ROW_OUT + oc*W,
                   sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_row_cnn[oc],
                   d_R_cnn + oc,
                   sizeof(float), cudaMemcpyDeviceToHost);
    }

    float max_diff=0.f, rms=0.f;
    for (int oc=0; oc<W; ++oc) {
        float d = std::fabs(h_row_fft[oc] - h_row_cnn[oc]);
        max_diff = std::max(max_diff, d);
        rms += d*d;
    }
    rms = std::sqrt(rms/W);

    printf("col  R_fft[row%d,:]     R_strip_cudnn[row0,:]  |diff|\n", ROW_OUT);
    printf("────────────────────────────────────────────────────────\n");
    for (int oc=0; oc<W; ++oc) {
        bool print = (oc<5 || oc>=W-3 || (oc>=W/2-2 && oc<=W/2+2));
        if (print)
            printf("%3d  %+.7f     %+.7f      %.2e\n",
                   oc, h_row_fft[oc], h_row_cnn[oc],
                   std::fabs(h_row_fft[oc]-h_row_cnn[oc]));
        else if (oc==5) printf("  ...\n");
    }

    // ── Dump signals for notebook ─────────────────────────────────────────
    {
        mkdir_p("dump_compare");
        FILE* f = fopen("dump_compare/compare_signal.txt", "w");
        if (f) {
            fprintf(f, "# sigma=%.1f  KW=%d  erosion=%d  W=%d  B=%d\n",
                    sigma, KW, EROSION, W, B);
            fprintf(f, "# fft_ms=%.4f  cnn_ms=%.4f\n", fft_ms, cnn_ms);
            fprintf(f, "# fft_ws_bytes=%zu  cnn_ws_bytes=%zu\n",
                    fft_ws_bytes, cnn_ws_bytes);
            fprintf(f, "# col  R_fft  R_cnn  diff\n");
            for (int oc=0; oc<W; ++oc)
                fprintf(f, "%d %.8g %.8g %.8g\n",
                        oc, h_row_fft[oc], h_row_cnn[oc],
                        h_row_fft[oc]-h_row_cnn[oc]);
            fclose(f);
            printf("Dumped signal to dump_compare/compare_signal.txt\n");
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  Max |diff|  : %.3e\n", max_diff);
    printf("  RMS  diff   : %.3e\n", rms);
    printf("\n");
    printf("  %-30s %10s %10s\n", "", "FFT", "Strip cuDNN");
    printf("  %-30s %10s %10s\n", "Input size", "256×256", "121×256");
    printf("  %-30s %10d %10d\n", "Filter KW", KW, KW);
    printf("  %-30s %10d %10d\n", "Erosion", EROSION, EROSION);
    printf("  %-30s %10s %10s\n", "Output per batch",
           (std::to_string(W)+"×"+std::to_string(W)).c_str(),
           ("1×"+std::to_string(W)).c_str());
    printf("  %-30s %10d %10d\n", "Batch B", B, B);
    printf("  %-30s %10s %10s\n", "Workspace",
           fmt_bytes(fft_ws_bytes).c_str(), fmt_bytes(cnn_ws_bytes).c_str());
    printf("  %-30s %10.2f %10.2f  ms/call\n",
           "Latency (avg)", fft_ms, cnn_ms);
    printf("══════════════════════════════════════════════════════\n");

    // Cleanup
    cudaFree(d_E_full); cudaFree(d_E_strip);
    cudaFree(d_R_fft);  cudaFree(d_R_cnn);
    cudaFree(d_Gx_sp);  cudaFree(d_Gy_sp);
    cudaFree(d_Gx_freq); cudaFree(d_Gy_freq);
    return 0;
}
