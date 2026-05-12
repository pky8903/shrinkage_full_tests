// image_crop.cu
//
// Center-crop an N×N image (col-major) to (crop_w × crop_h).
//
// Alignment: image centre (N/2, N/2) → output centre (crop_w/2, crop_h/2).
// Output layout: col-major  out[col * crop_h + row]

#include <imageUtil/image_crop.cuh>

#include <cuda_runtime.h>
#include <stdexcept>

namespace st { namespace util {

// ─── CPU implementation ───────────────────────────────────────────────────

std::vector<float> crop_center_cpu(const float* image, int N,
                                   int crop_w, int crop_h)
{
    if (crop_w > N || crop_h > N)
        throw std::invalid_argument("image_crop: crop size exceeds image size");

    // top-left corner of the crop region in the source image
    const int col0 = N / 2 - crop_w / 2;
    const int row0 = N / 2 - crop_h / 2;

    std::vector<float> out((size_t)crop_w * crop_h);

    for (int c = 0; c < crop_w; ++c) {
        const int src_c = col0 + c;
        for (int r = 0; r < crop_h; ++r) {
            const int src_r = row0 + r;
            out[c * crop_h + r] = image[src_c * N + src_r];
        }
    }

    return out;
}

// ─── GPU kernel ───────────────────────────────────────────────────────────

__global__ static void kernel_crop_center(
    const float* __restrict__ src, int N,
    int crop_w, int crop_h, int col0, int row0,
    float* __restrict__ dst)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= crop_w || r >= crop_h) return;

    dst[c * crop_h + r] = src[(col0 + c) * N + (row0 + r)];
}

// ─── GPU API ──────────────────────────────────────────────────────────────

float* crop_center_gpu(const float* d_image, int N,
                       int crop_w, int crop_h,
                       cudaStream_t stream)
{
    if (crop_w > N || crop_h > N)
        throw std::invalid_argument("image_crop: crop size exceeds image size");

    const int col0 = N / 2 - crop_w / 2;
    const int row0 = N / 2 - crop_h / 2;

    float* d_out = nullptr;
    cudaMalloc(&d_out, (size_t)crop_w * crop_h * sizeof(float));

    const dim3 block(16, 16);
    const dim3 grid((crop_w + block.x - 1) / block.x,
                    (crop_h + block.y - 1) / block.y);

    kernel_crop_center<<<grid, block, 0, stream>>>(
        d_image, N, crop_w, crop_h, col0, row0, d_out);

    return d_out;
}

} } // namespace st::util
