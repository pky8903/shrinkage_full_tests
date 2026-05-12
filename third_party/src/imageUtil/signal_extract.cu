// signal_extract.cu
//
// CPU + GPU implementations for 1-D signal extraction from an N×N image.
//
// Memory convention throughout: col-major  image[col * N + row]

#include <imageUtil/signal_extract.cuh>

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace st { namespace util {

// ─── helpers ──────────────────────────────────────────────────────────────

static float sinc_norm(float x)
{
    // Normalised sinc: sinc(x) = sin(π x) / (π x)
    if (std::abs(x) < 1e-7f) return 1.0f;
    const float px = static_cast<float>(M_PI) * x;
    return std::sin(px) / px;
}

static float blackman_win(float n, int A)
{
    // Blackman window over support [-A, A]
    const float alpha = 0.16f;
    const float a0 = (1.0f - alpha) * 0.5f;
    const float a1 = 0.5f;
    const float a2 = alpha * 0.5f;
    const float pi = static_cast<float>(M_PI);
    return a0 - a1 * std::cos(pi * (n + A) / A)
               + a2 * std::cos(2.0f * pi * (n + A) / A);
}

// Evaluate sinc×Blackman kernel at fractional position x (centred at 0)
// with half-width A lobes.  Returns 0 outside [-A, A].
static float sinc_blackman(float x, int A)
{
    if (std::abs(x) >= static_cast<float>(A)) return 0.0f;
    return sinc_norm(x) * blackman_win(x, A);
}

// Bilinear-sinc sample of col-major image at (col_f, row_f).
// Uses sinc×Blackman with A lobes in both directions.
static float sample_image(const float* image, int N,
                           float col_f, float row_f, int A)
{
    const int ic0 = static_cast<int>(std::floor(col_f));
    const int ir0 = static_cast<int>(std::floor(row_f));

    float val = 0.0f;
    float wsum = 0.0f;

    for (int dc = -(A - 1); dc <= A; ++dc) {
        const int c = ic0 + dc;
        if (c < 0 || c >= N) continue;
        const float wc = sinc_blackman(col_f - c, A);

        for (int dr = -(A - 1); dr <= A; ++dr) {
            const int r = ir0 + dr;
            if (r < 0 || r >= N) continue;
            const float wr = sinc_blackman(row_f - r, A);
            const float w  = wc * wr;
            val  += w * image[c * N + r];
            wsum += w;
        }
    }
    return (std::abs(wsum) > 1e-12f) ? val / wsum : 0.0f;
}

// ─── CPU implementation ───────────────────────────────────────────────────

std::vector<float> extract_signal_cpu(const float* image, int N,
                                      const SignalExtractConfig& cfg)
{
    const int erosion = cfg.erosion;
    const int L       = signal_extract_length(N, erosion);
    if (L <= 0)
        throw std::invalid_argument("signal_extract: erosion too large");

    std::vector<float> out(L);
    const int A = cfg.sinc_lobes;

    switch (cfg.direction) {
    case SignalDirection::VERTICAL:
        // column = N/2, row = erosion … N-erosion-1
        for (int i = 0; i < L; ++i)
            out[i] = image[(N / 2) * N + (erosion + i)];
        break;

    case SignalDirection::HORIZONTAL:
        // row = N/2, col = erosion … N-erosion-1
        for (int i = 0; i < L; ++i)
            out[i] = image[(erosion + i) * N + (N / 2)];
        break;

    case SignalDirection::ANYANGLE: {
        // Line passes through centre (N/2, N/2) at angle_deg CCW from horizontal.
        // angle_deg = 0  → horizontal scan (col varies, row fixed)
        // angle_deg = 90 → vertical   scan (row varies, col fixed)
        const float angle_rad = cfg.angle_deg * static_cast<float>(M_PI) / 180.0f;
        const float cos_a = std::cos(angle_rad);
        const float sin_a = std::sin(angle_rad);
        const float cx    = static_cast<float>(N) / 2.0f;
        const float cy    = static_cast<float>(N) / 2.0f;

        // Parameter t runs from -(N/2 - erosion) to (N/2 - erosion) - 1
        const float t0 = static_cast<float>(erosion) - static_cast<float>(N) / 2.0f;

        for (int i = 0; i < L; ++i) {
            const float t   = t0 + static_cast<float>(i);
            const float col_f = cx + t * cos_a;
            const float row_f = cy + t * sin_a;
            out[i] = sample_image(image, N, col_f, row_f, A);
        }
        break;
    }
    }

    return out;
}

// ─── GPU kernels ──────────────────────────────────────────────────────────

__global__ static void kernel_extract_vertical(
    const float* __restrict__ image, int N, int erosion, int L,
    float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;
    out[i] = image[(N / 2) * N + (erosion + i)];
}

__global__ static void kernel_extract_horizontal(
    const float* __restrict__ image, int N, int erosion, int L,
    float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;
    out[i] = image[(erosion + i) * N + (N / 2)];
}

// Device sinc×Blackman helper
__device__ static float d_sinc_norm(float x)
{
    if (fabsf(x) < 1e-7f) return 1.0f;
    const float px = float(M_PI) * x;
    return sinf(px) / px;
}

__device__ static float d_blackman_win(float n, int A)
{
    const float alpha = 0.16f;
    const float a0 = (1.0f - alpha) * 0.5f;
    const float a1 = 0.5f;
    const float a2 = alpha * 0.5f;
    const float pi = float(M_PI);
    return a0 - a1 * cosf(pi * (n + A) / A)
               + a2 * cosf(2.0f * pi * (n + A) / A);
}

__device__ static float d_sinc_blackman(float x, int A)
{
    if (fabsf(x) >= (float)A) return 0.0f;
    return d_sinc_norm(x) * d_blackman_win(x, A);
}

__global__ static void kernel_extract_anyangle(
    const float* __restrict__ image, int N, int erosion, int L,
    float cos_a, float sin_a, int A,
    float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;

    const float cx  = (float)N / 2.0f;
    const float cy  = (float)N / 2.0f;
    const float t   = (float)(erosion - N / 2) + (float)i;
    const float col_f = cx + t * cos_a;
    const float row_f = cy + t * sin_a;

    const int ic0 = (int)floorf(col_f);
    const int ir0 = (int)floorf(row_f);

    float val  = 0.0f;
    float wsum = 0.0f;

    for (int dc = -(A - 1); dc <= A; ++dc) {
        const int c = ic0 + dc;
        if (c < 0 || c >= N) continue;
        const float wc = d_sinc_blackman(col_f - (float)c, A);

        for (int dr = -(A - 1); dr <= A; ++dr) {
            const int r = ir0 + dr;
            if (r < 0 || r >= N) continue;
            const float wr = d_sinc_blackman(row_f - (float)r, A);
            const float w  = wc * wr;
            val  += w * image[c * N + r];
            wsum += w;
        }
    }

    out[i] = (fabsf(wsum) > 1e-12f) ? val / wsum : 0.0f;
}

// ─── GPU API ──────────────────────────────────────────────────────────────

float* extract_signal_gpu(const float* d_image, int N,
                          const SignalExtractConfig& cfg,
                          cudaStream_t stream)
{
    const int erosion = cfg.erosion;
    const int L       = signal_extract_length(N, erosion);
    if (L <= 0)
        throw std::invalid_argument("signal_extract: erosion too large");

    float* d_out = nullptr;
    cudaMalloc(&d_out, (size_t)L * sizeof(float));

    const int block = 256;
    const int grid  = (L + block - 1) / block;

    switch (cfg.direction) {
    case SignalDirection::VERTICAL:
        kernel_extract_vertical<<<grid, block, 0, stream>>>(
            d_image, N, erosion, L, d_out);
        break;

    case SignalDirection::HORIZONTAL:
        kernel_extract_horizontal<<<grid, block, 0, stream>>>(
            d_image, N, erosion, L, d_out);
        break;

    case SignalDirection::ANYANGLE: {
        const float angle_rad = cfg.angle_deg * float(M_PI) / 180.0f;
        const float cos_a     = std::cos(angle_rad);
        const float sin_a     = std::sin(angle_rad);
        kernel_extract_anyangle<<<grid, block, 0, stream>>>(
            d_image, N, erosion, L, cos_a, sin_a, cfg.sinc_lobes, d_out);
        break;
    }
    }

    return d_out;
}

} } // namespace st::util
