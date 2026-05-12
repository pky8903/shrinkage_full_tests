#include "metricUtil/cd_measure_util.cuh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace st { namespace util {

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr float kPi  = 3.14159265358979323846f;
static constexpr float k2Pi = 6.28318530717958647692f;
static constexpr float k4Pi = 12.5663706143591729539f;
static constexpr float kInf = 3.402823466e+38f;

// ---------------------------------------------------------------------------
// Device helper: sinc x Blackman
// ---------------------------------------------------------------------------

__device__ __forceinline__
float sinc_blackman(float t, int A)
{
    if (fabsf(t) > float(A) + 1e-6f) return 0.f;

    float s = (fabsf(t) < 1e-6f)
              ? 1.f
              : __sinf(kPi * t) / (kPi * t);

    float u  = (t + float(A)) / float(2 * A);
    float bw = 0.42f
             - 0.5f  * __cosf(k2Pi * u)
             + 0.08f * __cosf(k4Pi * u);
    return s * bw;
}

// ---------------------------------------------------------------------------
// Kernel: build polyphase coefficient table
//
// Grid : (ceil(L / 256),)
// Block: (256,)
// Output: coeff[r, n]  row-major [L, 2A+1], normalised so sum(row) = 1
// ---------------------------------------------------------------------------

__global__ void build_polyphase_table_kernel(
    float* __restrict__ coeff,
    int L, int A)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= L) return;

    const int   taps  = 2 * A + 1;
    const float phase = float(r) / float(L);
    float* h = coeff + r * taps;

    float sum = 0.f;
    for (int n = -A; n <= A; ++n) {
        float w = sinc_blackman(float(n) - phase, A);
        h[n + A] = w;
        sum += w;
    }

    const float inv = (fabsf(sum) > 1e-20f) ? 1.f / sum : 1.f;
    for (int i = 0; i < taps; ++i)
        h[i] *= inv;
}

// ---------------------------------------------------------------------------
// Kernel: batched upsample using prebuilt polyphase table
//
// Grid : (ceil(N*L / 256), B)
// Block: (256, 1)
// ---------------------------------------------------------------------------

__global__ void upsample_batched_kernel(
    const float* __restrict__ in,      // [B, N]
    const float* __restrict__ coeff,   // [L, 2A+1]
    float* __restrict__       out,     // [B, N*L]
    int N, int L, int A, int B)
{
    const int b = blockIdx.y;
    if (b >= B) return;

    const size_t Nout = (size_t)N * L;
    const size_t m    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= Nout) return;

    const int taps = 2 * A + 1;
    const int r    = (int)(m % (size_t)L);
    const int k    = (int)(m / (size_t)L);

    const float* sig = in    + (size_t)b * N;
    const float* h   = coeff + (size_t)r * taps;
    float* dst       = out   + (size_t)b * Nout;

    float acc = 0.f;
    for (int i = 0; i < taps; ++i) {
        const int idx = k + i - A;
        if ((unsigned)idx < (unsigned)N)
            acc += sig[idx] * h[i];
    }
    dst[m] = acc;
}

// ---------------------------------------------------------------------------
// Kernel: detect zero-crossings (batched)
//
// Grid : (ceil(Nedge / 256), B)
// Block: (256, 1)
// ---------------------------------------------------------------------------

__global__ void detect_crossings_batched_kernel(
    const float* __restrict__ d_up,    // [B, Nup]
    int*   __restrict__       flags,   // [B, Nedge]  1 if crossing, else 0
    float* __restrict__       x_frac,  // [B, Nedge]  sub-sample position
    int Nup, int Nedge, int B)
{
    const int b = blockIdx.y;
    if (b >= B) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nedge) return;

    const float* sig = d_up   + (size_t)b * Nup;
    int*         fl  = flags  + (size_t)b * Nedge;
    float*       xf  = x_frac + (size_t)b * Nedge;

    const float y0 = sig[i];
    const float y1 = sig[i + 1];

    int   flag = 0;
    float x    = 0.f;

    if ((y0 <= 0.f && y1 > 0.f) || (y0 > 0.f && y1 <= 0.f)) {
        float den = y1 - y0;
        x    = (fabsf(den) < 1e-30f) ? float(i) : float(i) - y0 / den;
        flag = 1;
    }

    fl[i] = flag;
    xf[i] = x;
}

// ---------------------------------------------------------------------------
// Kernel: scatter crossing positions into compact arrays
//
// Grid : (ceil(Nedge / 256), B)
// Block: (256, 1)
// ---------------------------------------------------------------------------

__global__ void scatter_crossings_batched_kernel(
    const int*   __restrict__ flags,
    const int*   __restrict__ offs,
    const float* __restrict__ x_frac,
    float* __restrict__       ev_x,    // [B, Nedge]  compact output
    int Nedge, int B)
{
    const int b = blockIdx.y;
    if (b >= B) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nedge) return;

    const size_t base = (size_t)b * Nedge;
    if (!flags[base + i]) return;

    ev_x[base + offs[base + i]] = x_frac[base + i];
}

// ---------------------------------------------------------------------------
// Kernel: find nearest crossing to centre on each side  (one block per signal)
//
// Grid : (B,)
// Block: (red_threads,)   must be power-of-two
// Smem : 2 * red_threads * sizeof(float)
// ---------------------------------------------------------------------------

__global__ void find_nearest_crossings_kernel(
    const float* __restrict__ ev_x,      // [B, Nedge]
    const int*   __restrict__ M_arr,     // [B]  number of crossings per signal
    float* __restrict__       out_left,  // [B]
    float* __restrict__       out_right, // [B]
    float centre_s, int Nedge, int B)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    const int    M  = M_arr[b];
    const float* ex = ev_x + (size_t)b * Nedge;

    extern __shared__ float smem[];
    float* s_left  = smem;
    float* s_right = smem + blockDim.x;

    float my_left  = -kInf;
    float my_right =  kInf;

    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        float x = ex[j];
        if (x < centre_s) { if (x > my_left)  my_left  = x; }
        else               { if (x < my_right) my_right = x; }
    }

    s_left [threadIdx.x] = my_left;
    s_right[threadIdx.x] = my_right;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_left [threadIdx.x + stride] > s_left [threadIdx.x])
                s_left [threadIdx.x] = s_left [threadIdx.x + stride];
            if (s_right[threadIdx.x + stride] < s_right[threadIdx.x])
                s_right[threadIdx.x] = s_right[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_left [b] = s_left [0];
        out_right[b] = s_right[0];
    }
}

// ---------------------------------------------------------------------------
// Kernel: convert upsampled coordinates to pixel units and compute CD
//
// Grid : (ceil(B / 256),)
// Block: (256,)
// ---------------------------------------------------------------------------

__global__ void convert_to_pixels_kernel(
    const float* __restrict__ out_left,
    const float* __restrict__ out_right,
    float* __restrict__       left_px,
    float* __restrict__       right_px,
    float* __restrict__       cd_px,
    float invL, int B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float lx = out_left [b] * invL;
    const float rx = out_right[b] * invL;
    left_px [b] = lx;
    right_px[b] = rx;
    cd_px   [b] = rx - lx;
}

// =============================================================================
// Public API implementation
// =============================================================================

float* BuildPolyphaseTable(int L, int A, cudaStream_t stream)
{
    const int taps = 2 * A + 1;
    float* d_coeff = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_coeff, (size_t)L * taps * sizeof(float), stream));

    const int threads = 256;
    const int blocks  = (L + threads - 1) / threads;
    build_polyphase_table_kernel<<<blocks, threads, 0, stream>>>(d_coeff, L, A);
    CUDA_CHECK(cudaGetLastError());

    return d_coeff;
}

float* UpsampleBatched(
    const float* d_in,
    const float* d_coeff,
    int B, int N, int L, int A,
    cudaStream_t stream)
{
    const size_t Nout    = (size_t)N * L;
    const int    threads = 256;
    const int    bx      = (int)((Nout + threads - 1) / threads);
    const dim3   grid(bx, B);

    float* d_out = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_out, (size_t)B * Nout * sizeof(float), stream));

    upsample_batched_kernel<<<grid, threads, 0, stream>>>(
        d_in, d_coeff, d_out, N, L, A, B);
    CUDA_CHECK(cudaGetLastError());

    return d_out;
}

BatchCDResult MeasureCD_Batched(
    const float* d_in,
    const float* d_coeff,
    const float* d_up,
    int B, int N,
    int L, int A,
    cudaStream_t stream)
{
    if (B <= 0 || N <= 0 || L <= 0 || A <= 0) return {};

    const size_t Nup    = (size_t)N * L;
    const size_t Nedge  = Nup - 1;
    const int    thds   = 256;
    const dim3   grid_e((int)((Nedge + thds - 1) / thds), B);

    float* d_coeff_owned = nullptr;
    if (!d_coeff) {
        d_coeff_owned = BuildPolyphaseTable(L, A, stream);
        d_coeff       = d_coeff_owned;
    }

    float* d_up_owned = nullptr;
    if (!d_up) {
        d_up_owned = UpsampleBatched(d_in, d_coeff, B, N, L, A, stream);
        d_up       = d_up_owned;
    }

    int*   d_flags     = nullptr;
    float* d_x_frac    = nullptr;
    int*   d_offs      = nullptr;
    float* d_ev_x      = nullptr;
    int*   d_M         = nullptr;
    float* d_out_left  = nullptr;
    float* d_out_right = nullptr;
    float* d_left_px   = nullptr;
    float* d_right_px  = nullptr;
    float* d_cd_px     = nullptr;

    CUDA_CHECK(cudaMallocAsync(&d_flags,     (size_t)B * Nedge * sizeof(int),   stream));
    CUDA_CHECK(cudaMallocAsync(&d_x_frac,    (size_t)B * Nedge * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_offs,      (size_t)B * Nedge * sizeof(int),   stream));
    CUDA_CHECK(cudaMallocAsync(&d_ev_x,      (size_t)B * Nedge * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_M,         (size_t)B         * sizeof(int),   stream));
    CUDA_CHECK(cudaMallocAsync(&d_out_left,  (size_t)B         * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_out_right, (size_t)B         * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_left_px,   (size_t)B         * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_right_px,  (size_t)B         * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_cd_px,     (size_t)B         * sizeof(float), stream));

    detect_crossings_batched_kernel<<<grid_e, thds, 0, stream>>>(
        d_up, d_flags, d_x_frac, (int)Nup, (int)Nedge, B);
    CUDA_CHECK(cudaGetLastError());

    for (int b = 0; b < B; ++b) {
        thrust::device_ptr<int> fp(d_flags + (size_t)b * Nedge);
        thrust::device_ptr<int> op(d_offs  + (size_t)b * Nedge);
        thrust::exclusive_scan(thrust::cuda::par.on(stream), fp, fp + Nedge, op);
    }

    scatter_crossings_batched_kernel<<<grid_e, thds, 0, stream>>>(
        d_flags, d_offs, d_x_frac, d_ev_x, (int)Nedge, B);
    CUDA_CHECK(cudaGetLastError());

    {
        const int Nedge_i = (int)Nedge;
        thrust::device_ptr<int> flags_ptr(d_flags);
        thrust::device_ptr<int> offs_ptr (d_offs);
        thrust::device_ptr<int> M_ptr    (d_M);
        thrust::transform(
            thrust::cuda::par.on(stream),
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(B),
            M_ptr,
            [=] __device__ (int b) {
                const int last = b * Nedge_i + (Nedge_i - 1);
                return offs_ptr[last] + flags_ptr[last];
            });
    }

    const float  centre_s    = 0.5f * float(Nup - 1);
    const int    red_threads = 128;
    const size_t smem_bytes  = 2 * red_threads * sizeof(float);

    find_nearest_crossings_kernel<<<B, red_threads, smem_bytes, stream>>>(
        d_ev_x, d_M, d_out_left, d_out_right,
        centre_s, (int)Nedge, B);
    CUDA_CHECK(cudaGetLastError());

    {
        const float invL = 1.0f / float(L);
        const int   bx   = (B + thds - 1) / thds;
        convert_to_pixels_kernel<<<bx, thds, 0, stream>>>(
            d_out_left, d_out_right,
            d_left_px, d_right_px, d_cd_px,
            invL, B);
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<float> h_left(B), h_right(B), h_cd(B);
    CUDA_CHECK(cudaMemcpyAsync(h_left.data(),  d_left_px,  B*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_right.data(), d_right_px, B*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_cd.data(),    d_cd_px,    B*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_flags,     stream));
    CUDA_CHECK(cudaFreeAsync(d_x_frac,    stream));
    CUDA_CHECK(cudaFreeAsync(d_offs,      stream));
    CUDA_CHECK(cudaFreeAsync(d_ev_x,      stream));
    CUDA_CHECK(cudaFreeAsync(d_M,         stream));
    CUDA_CHECK(cudaFreeAsync(d_out_left,  stream));
    CUDA_CHECK(cudaFreeAsync(d_out_right, stream));
    CUDA_CHECK(cudaFreeAsync(d_left_px,   stream));
    CUDA_CHECK(cudaFreeAsync(d_right_px,  stream));
    CUDA_CHECK(cudaFreeAsync(d_cd_px,     stream));
    if (d_up_owned)    CUDA_CHECK(cudaFreeAsync(d_up_owned,    stream));
    if (d_coeff_owned) CUDA_CHECK(cudaFreeAsync(d_coeff_owned, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    BatchCDResult result;
    result.B = B;
    result.cd.resize(B);
    for (int b = 0; b < B; ++b) {
        result.cd[b] = { h_left[b], h_right[b], h_cd[b] };
        if (h_left[b] == -kInf || h_right[b] == kInf)
            fprintf(stderr, "warning: signal %d missing crossing on %s side\n",
                b, h_left[b] == -kInf ? "left" : "right");
    }
    return result;
}

BatchCDResult MeasureCD_Batched(
    const float* d_in,
    int B, int N, int L, int A,
    cudaStream_t stream)
{
    return MeasureCD_Batched(d_in, nullptr, nullptr, B, N, L, A, stream);
}

} } // namespace st::util
