// batch_demo.cu — quick visual sanity check of srkTrainingForward on B=4.
//
// Generates 4 independent (E_b, I_b) inputs at N=256, runs the batched
// forward pass once, and dumps E, I, R per batch to dump_batch/ as simple
// row-major text files.  Open plot_batch_demo.ipynb to visualize.
//
// Not a correctness test — outputs are visually inspected.
// ─────────────────────────────────────────────────────────────────

#include "srk_training_forward.cuh"
#include "srk_forward.cuh"   // for srkPrecomputeKernelFreq

#include <patternUtil/test_pattern.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <sys/stat.h>

using namespace st::util;

static constexpr float DX        = 1.0f;   // pixel spacing (not exposed as arg)
static constexpr float SRSRK_C_W = 3.0f;  // Blackman radius = c_w · sigma

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
// Column-major kernel buffers (see main.cu for details).
static void make_srsrk_kernel_2d(std::vector<float>& Kx, std::vector<float>& Ky,
                                 int kw, int kh, float dx, float sigma) {
    const float gamma = 1.f / sigma;
    const float R_w   = SRSRK_C_W / gamma;
    const float eps   = 1e-6f * dx;
    const float coeff = gamma / (2.f * (float)M_PI);
    Kx.assign((size_t)kw * kh, 0.f);
    Ky.assign((size_t)kw * kh, 0.f);
    const int cx = kw / 2, cy = kh / 2;
    for (int i = 0; i < kw; ++i) {
        for (int j = 0; j < kh; ++j) {
            const int idx = j + i * kh;
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

// ── Dump a column-major image as one line per row of pixels ────────
//   # name: <name>
//   # shape: <H> <W>           (rows, cols)
//   p(0,0) p(0,1) ... p(0,W-1)
//   p(1,0) p(1,1) ...
//   ...
//   Memory access is column-major (data[r + c*H]); the file format is the
//   plain visual layout so np.loadtxt yields shape (H, W) with arr[r, c].
static void dump_image_colmajor(const std::string& path,
                                const float* data, int H, int W,
                                const std::string& name)
{
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "[dump] cannot open '%s'\n", path.c_str()); return; }
    fprintf(f, "# name: %s\n# shape: %d %d\n", name.c_str(), H, W);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            fprintf(f, c + 1 < W ? "%.8g " : "%.8g\n", data[r + c * H]);
        }
    }
    fclose(f);
}

static void mkdir_p(const std::string& path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            std::string sub = path.substr(0, i);
            mkdir(sub.c_str(), 0755);
        }
    }
}

int main(int argc, char** argv) {
    const float SIGMA   = get_arg_float(argc, argv, "--sigma", 5.f);
    const int   N       = get_arg_int  (argc, argv, "--n",     256);
    const int   B       = get_arg_int  (argc, argv, "--b",     4);
    const float R_W     = SRSRK_C_W * SIGMA;
    const int   EROSION = (int)(R_W / DX) + 1;
    const int   KW      = 2 * (int)(R_W / DX) + 1;
    const int   KH      = KW;
    const int   W       = N - 2 * EROSION;
    printf("sigma=%.1f  N=%d  B=%d  R_w=%.1f  kw=%d  erosion=%d  W=%d\n",
           SIGMA, N, B, R_W, KW, EROSION, W);
    cudaSetDevice(0);

    // ── Kernels + freq precomputation (at FFT size = N) ────────────
    std::vector<float> h_Gx, h_Gy;
    make_srsrk_kernel_2d(h_Gx, h_Gy, KW, KH, DX, SIGMA);

    const int FREQ_N = N * (N / 2 + 1);
    cufftComplex *d_Gx_freq = nullptr, *d_Gy_freq = nullptr;
    cudaMalloc(&d_Gx_freq, sizeof(cufftComplex) * FREQ_N);
    cudaMalloc(&d_Gy_freq, sizeof(cufftComplex) * FREQ_N);
    srkPrecomputeKernelFreq(h_Gx, KW, KH, N, d_Gx_freq, /*stream=*/nullptr);
    srkPrecomputeKernelFreq(h_Gy, KW, KH, N, d_Gy_freq, /*stream=*/nullptr);

    // ── Build B inputs on host; per batch use the same image for E and I ──
    // b=0,1: vertical L&S  (sigma=8,6 px);  b=2,3: horizontal L&S (sigma=8,6 px)
    std::vector<float> h_E(B * N * N), h_I(B * N * N);
    static const float lns_sigmas[] = {8.f, 6.f, 8.f, 6.f};
    for (int b = 0; b < B; ++b) {
        const float sig = lns_sigmas[b % 4];
        auto cfg = (b < 2)
            ? TestPatternConfig::lns_vertical  (N, sig)
            : TestPatternConfig::lns_horizontal(N, sig);
        auto img = generate_pattern(cfg);
        std::memcpy(h_E.data() + (size_t)b * N * N, img.data(), sizeof(float) * N * N);
        std::memcpy(h_I.data() + (size_t)b * N * N, img.data(), sizeof(float) * N * N);
    }

    // ── Device buffers & workspace ─────────────────────────────────
    float *d_E = nullptr, *d_I = nullptr, *d_R = nullptr;
    cudaMalloc(&d_E, sizeof(float) * B * N * N);
    cudaMalloc(&d_I, sizeof(float) * B * N * N);
    cudaMalloc(&d_R, sizeof(float) * B * W * W);
    cudaMemcpy(d_E, h_E.data(), sizeof(float) * B * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, h_I.data(), sizeof(float) * B * N * N, cudaMemcpyHostToDevice);

    SrkTrainWorkspace ws;
    ws.allocate(N, /*B_max=*/B);

    // ── Run the batched forward (single call, B=4) ─────────────────
    srkTrainingForward(
        d_E, d_I, d_Gx_freq, d_Gy_freq,
        N, W, EROSION, DX, B, /*stream=*/nullptr,
        d_R, ws);

    // ── Pull results back and dump ─────────────────────────────────
    std::vector<float> h_R(B * W * W);
    cudaMemcpy(h_R.data(), d_R, sizeof(float) * B * W * W, cudaMemcpyDeviceToHost);

    const std::string dir = "dump_batch";
    mkdir_p(dir);
    for (int b = 0; b < B; ++b) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s/b%d_E.txt", dir.c_str(), b);
        dump_image_colmajor(buf, h_E.data() + (size_t)b * N * N, N, N, "E_b" + std::to_string(b));
        snprintf(buf, sizeof(buf), "%s/b%d_I.txt", dir.c_str(), b);
        dump_image_colmajor(buf, h_I.data() + (size_t)b * N * N, N, N, "I_b" + std::to_string(b));
        snprintf(buf, sizeof(buf), "%s/b%d_R.txt", dir.c_str(), b);
        dump_image_colmajor(buf, h_R.data() + (size_t)b * W * W, W, W, "R_b" + std::to_string(b));
    }
    printf("Dumped %d batches to %s/  (E: %dx%d, I: %dx%d, R: %dx%d)\n",
           B, dir.c_str(), N, N, N, N, W, W);

    // Cleanup
    ws.release();
    cudaFree(d_E); cudaFree(d_I); cudaFree(d_R);
    cudaFree(d_Gx_freq); cudaFree(d_Gy_freq);
    return 0;
}
