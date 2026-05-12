#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>

#ifdef API_TESTER_CUDA
#  include <cuda_runtime.h>
#endif

namespace st { namespace util {

struct LatencyResult {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    int    warmup = 0;
    int    runs   = 0;
};

inline LatencyResult measure_latency(
    std::function<void()> func,
    int warmup = 3,
    int runs   = 10)
{
#ifndef API_TESTER_CUDA
    throw std::runtime_error(
        "measure_latency requires API_TESTER_CUDA. "
        "Pass -DAPI_TESTER_CUDA=ON to CMake.");
#else
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int i = 0; i < warmup; ++i)
        func();
    cudaDeviceSynchronize();

    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(ev_start);
        func();
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        times.push_back(static_cast<double>(ms));
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    LatencyResult r;
    r.warmup  = warmup;
    r.runs    = runs;
    r.avg_ms  = std::accumulate(times.begin(), times.end(), 0.0) / runs;
    r.min_ms  = *std::min_element(times.begin(), times.end());
    r.max_ms  = *std::max_element(times.begin(), times.end());
    return r;
#endif
}

} } // namespace st::util
