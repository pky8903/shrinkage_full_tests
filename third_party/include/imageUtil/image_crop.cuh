// image_crop.cuh
//
// Center-crop an N×N image (col-major) to (crop_w × crop_h).
//
// The centres (N/2, N/2) and (crop_w/2, crop_h/2) are aligned.
// Output is col-major: out[col * crop_h + row], size = crop_w * crop_h.

#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace st { namespace util {

// CPU implementation – returns host vector of size crop_w * crop_h.
std::vector<float> crop_center_cpu(const float* image, int N,
                                   int crop_w, int crop_h);

// GPU implementation – allocates and returns device pointer; caller owns memory.
float* crop_center_gpu(const float* d_image, int N,
                       int crop_w, int crop_h,
                       cudaStream_t stream = nullptr);

} } // namespace st::util
