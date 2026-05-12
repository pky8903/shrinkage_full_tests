#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace st { namespace util {

// =============================================================================
// Upsample API
// =============================================================================

// Build a polyphase sinc x Blackman coefficient table on the GPU.
//
//   L    : upsample factor  (e.g. 16)
//   A    : sinc half-width  (filter support = 2A+1 taps)
//
// Returns a device pointer to a [L, 2A+1] row-major float array.
// The caller is responsible for freeing it with cudaFreeAsync(ptr, stream).
float* BuildPolyphaseTable(int L, int A, cudaStream_t stream);

// Upsample a batch of B signals by factor L using a prebuilt polyphase table.
//
//   d_in    : device pointer [B, N]  row-major
//   d_coeff : device pointer [L, 2A+1]  from BuildPolyphaseTable()
//   B, N    : batch size and signal length
//   L, A    : must match the values used to build d_coeff
//
// Returns a device pointer to [B, N*L] row-major float array.
// The caller is responsible for freeing it with cudaFreeAsync(ptr, stream).
float* UpsampleBatched(
    const float* d_in,
    const float* d_coeff,
    int B, int N, int L, int A,
    cudaStream_t stream);

// =============================================================================
// CD measurement API
// =============================================================================

// CD measurement for a single signal.
// All positions are in original pixel units (before upsampling).
struct CDMeasurement {
    float left_edge_px;    // nearest crossing left  of centre [pixel]
    float right_edge_px;   // nearest crossing right of centre [pixel]
    float cd_px;           // right_edge - left_edge           [pixel]
};

// CD measurements for a batch of B signals.
struct BatchCDResult {
    int B = 0;
    std::vector<CDMeasurement> cd;   // [B]
};

// Measure CD for B signals of equal length N.
//
// Algorithm:
//   1. Build polyphase table for (L, A).
//   2. Upsample each signal by factor L.
//   3. Detect all zero-crossings (sub-pixel, linear interpolation).
//   4. Per signal, find:
//        left  edge = crossing with largest  position <  centre (N*L/2)
//        right edge = crossing with smallest position >= centre (N*L/2)
//   5. cd = right_edge - left_edge  (valid for both line and trench features).
//
//   d_in   : device pointer [B, N]  row-major
//   L      : upsample factor
//   A      : sinc half-width
//   stream : CUDA stream
BatchCDResult MeasureCD_Batched(
    const float* d_in,
    int B, int N,
    int L, int A,
    cudaStream_t stream);

// Measure CD using a prebuilt polyphase table and/or a precomputed upsampled
// signal. Useful when upsample output is needed for other purposes too.
//
//   d_coeff : device pointer [L, 2A+1]  from BuildPolyphaseTable()
//             pass nullptr to build internally
//   d_up    : device pointer [B, N*L]   from UpsampleBatched()
//             pass nullptr to compute internally
BatchCDResult MeasureCD_Batched(
    const float* d_in,
    const float* d_coeff,   // nullable
    const float* d_up,      // nullable
    int B, int N,
    int L, int A,
    cudaStream_t stream);

} } // namespace st::util
