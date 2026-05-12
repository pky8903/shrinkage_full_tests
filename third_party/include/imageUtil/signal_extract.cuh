// signal_extract.cuh
//
// Extract a 1-D signal from an N×N image (col-major layout).
//
// Directions
//   VERTICAL   – column N/2, rows [erosion, N-erosion)
//   HORIZONTAL – row   N/2, cols [erosion, N-erosion)
//   ANYANGLE   – sinc×Blackman interpolation along the line passing through
//                (N/2, N/2) at the given angle; same [erosion, N-erosion) span
//
// Memory layout: col-major  image[col * N + row]
// Output:        std::vector<float> (CPU) or device float* (GPU), length = N - 2*erosion

#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace st { namespace util {

enum class SignalDirection { VERTICAL, HORIZONTAL, ANYANGLE };

struct SignalExtractConfig {
    SignalDirection direction  = SignalDirection::VERTICAL;
    int             erosion    = 0;
    float           angle_deg  = 0.0f;   // for ANYANGLE: degrees CCW from horizontal
    int             sinc_lobes = 4;      // Blackman-sinc half-width (number of lobes)
};

// Returns the number of output samples for given N and erosion.
inline int signal_extract_length(int N, int erosion)
{
    return N - 2 * erosion;
}

// CPU implementation – returns host vector of length signal_extract_length(N, erosion).
std::vector<float> extract_signal_cpu(const float* image, int N,
                                      const SignalExtractConfig& cfg);

// GPU implementation – allocates and returns device pointer; caller owns memory.
// Length = signal_extract_length(N, cfg.erosion)
float* extract_signal_gpu(const float* d_image, int N,
                          const SignalExtractConfig& cfg,
                          cudaStream_t stream = nullptr);

} } // namespace st::util
