#pragma once
#include "arg_descriptor.h"
#include "buffer.h"
#include <cmath>
#include <vector>
#include <limits>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace st { namespace util {

// ─────────────────────────────────────────────
//  Results
// ─────────────────────────────────────────────

struct PixelPos {
    size_t nx = 0;
    size_t ny = 0;
};

struct CompareResult {
    double   l2_error     = 0.0;  ///< sqrt( sum( diff^2 ) )
    double   max_error    = 0.0;  ///< max |ref - test|
    PixelPos max_pos;             ///< position of max error
    double   stdev        = 0.0;  ///< stdev of |ref - test|
    size_t   num_elements = 0;
    bool     passed       = false;
    double   tolerance    = -1.0; ///< if set, passed = max_error <= tolerance
};

// ─────────────────────────────────────────────
//  Internal: typed comparison on host memory
// ─────────────────────────────────────────────

template<typename T>
CompareResult compare_typed(const T* ref, const T* test,
                             const Dims& dims, double tolerance) {
    const size_t N = dims.total();
    double sum_sq  = 0.0;
    double sum_err = 0.0;
    double max_err = 0.0;
    PixelPos max_pos;

    for (size_t i = 0; i < N; ++i) {
        const double diff = std::abs(static_cast<double>(ref[i]) -
                                     static_cast<double>(test[i]));
        sum_err += diff;
        sum_sq  += diff * diff;
        if (diff > max_err) {
            max_err    = diff;
            max_pos.nx = dims.is_2d() ? (i / dims.h) : i;
            max_pos.ny = dims.is_2d() ? (i % dims.h) : 0;
        }
    }

    const double mean = sum_err / static_cast<double>(N);
    double var = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double diff = std::abs(static_cast<double>(ref[i]) -
                                     static_cast<double>(test[i]));
        var += (diff - mean) * (diff - mean);
    }

    CompareResult r;
    r.l2_error     = std::sqrt(sum_sq);
    r.max_error    = max_err;
    r.max_pos      = max_pos;
    r.stdev        = std::sqrt(var / static_cast<double>(N));
    r.num_elements = N;
    r.tolerance    = tolerance;
    r.passed       = (tolerance >= 0.0) ? (max_err <= tolerance) : true;
    return r;
}

// ─────────────────────────────────────────────
//  Public: dtype-dispatched comparison
//  Handles device→host copy internally
// ─────────────────────────────────────────────

inline CompareResult compare(
    const void* ref_ptr,  MemSpace ref_mem,
    const void* test_ptr, MemSpace test_mem,
    DType dtype, const Dims& dims,
    double tolerance = -1.0)
{
    const size_t bytes = dims.total() * dtype_size(dtype);

    std::vector<uint8_t> h_ref(bytes), h_test(bytes);
    copy_to_host(h_ref.data(),  ref_ptr,  bytes, ref_mem);
    copy_to_host(h_test.data(), test_ptr, bytes, test_mem);

    const void* r = h_ref.data();
    const void* t = h_test.data();

    switch (dtype) {
        case DType::FLOAT16: {
            const size_t n = dims.total();
            std::vector<float> fr(n), ft(n);
            for (size_t i = 0; i < n; ++i) {
                uint16_t rh, th;
                memcpy(&rh, (const uint8_t*)r + i*2, 2);
                memcpy(&th, (const uint8_t*)t + i*2, 2);
                fr[i] = f16_to_f32(rh);
                ft[i] = f16_to_f32(th);
            }
            return compare_typed(fr.data(), ft.data(), dims, tolerance);
        }
        case DType::FLOAT32: return compare_typed((const float*)   r, (const float*)   t, dims, tolerance);
        case DType::FLOAT64: return compare_typed((const double*)  r, (const double*)  t, dims, tolerance);
        case DType::INT32:   return compare_typed((const int32_t*) r, (const int32_t*) t, dims, tolerance);
        case DType::INT16:   return compare_typed((const int16_t*) r, (const int16_t*) t, dims, tolerance);
        case DType::INT8:    return compare_typed((const int8_t*)  r, (const int8_t*)  t, dims, tolerance);
        case DType::UINT32:  return compare_typed((const uint32_t*)r, (const uint32_t*)t, dims, tolerance);
        case DType::UINT16:  return compare_typed((const uint16_t*)r, (const uint16_t*)t, dims, tolerance);
        case DType::UINT8:   return compare_typed((const uint8_t*) r, (const uint8_t*) t, dims, tolerance);
    }
    throw std::invalid_argument("Unknown DType in compare()");
}

// ─────────────────────────────────────────────
//  Pretty printer
// ─────────────────────────────────────────────

inline void print_result(const CompareResult& r,
                          const std::string& label = "") {
    std::cout << "─────────────────────────────────────\n";
    if (!label.empty())
        std::cout << "  Test : " << label << "\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "  L2   error  : " << r.l2_error  << "\n";
    std::cout << "  Max  error  : " << r.max_error;
    if (r.num_elements > 0) {
        if (r.max_pos.ny > 0 || r.num_elements > r.max_pos.nx + 1)
            std::cout << "  @ (" << r.max_pos.nx << ", " << r.max_pos.ny << ")";
        else
            std::cout << "  @ [" << r.max_pos.nx << "]";
    }
    std::cout << "\n";
    std::cout << "  Stdev       : " << r.stdev << "\n";
    std::cout << "  Elements    : " << r.num_elements << "\n";
    if (r.tolerance >= 0.0)
        std::cout << "  Result      : " << (r.passed ? "PASS v" : "FAIL x")
                  << "  (tol=" << r.tolerance << ")\n";
    std::cout << "─────────────────────────────────────\n";
}

} } // namespace st::util
