#pragma once
#include <atomic>
#include <thread>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <cstddef>

#ifdef API_TESTER_CUDA
#  include <cuda_runtime.h>
#endif

namespace st { namespace util {

struct MemoryResult {
    size_t baseline_bytes  = 0;
    size_t peak_used_bytes = 0;
    size_t workspace_bytes = 0;   ///< peak_used - baseline_used
    int    poll_interval_ms = 1;
    int    sample_count     = 0;
};

// Spawns a background polling thread (cudaMemGetInfo every poll_interval_ms).
// workspace_bytes = max memory allocated by func beyond baseline.
// NOTE: alloc+free within one poll interval may be missed.

class MemoryTracker {
public:
    explicit MemoryTracker(int poll_interval_ms = 1)
        : poll_interval_ms_(poll_interval_ms) {}

    MemoryResult measure(std::function<void()> func) {
#ifndef API_TESTER_CUDA
        throw std::runtime_error(
            "MemoryTracker requires API_TESTER_CUDA. "
            "Pass -DAPI_TESTER_CUDA=ON to CMake.");
#else
        cudaDeviceSynchronize();
        size_t free_before = 0, total = 0;
        cudaMemGetInfo(&free_before, &total);
        const size_t used_before = total - free_before;

        std::atomic<bool>   running{true};
        std::atomic<size_t> peak_used{used_before};
        std::atomic<int>    sample_count{0};

        std::thread poller([&]() {
            while (running.load(std::memory_order_relaxed)) {
                size_t free_now = 0, total_now = 0;
                cudaMemGetInfo(&free_now, &total_now);
                size_t used_now = total_now - free_now;

                size_t cur = peak_used.load(std::memory_order_relaxed);
                while (used_now > cur &&
                       !peak_used.compare_exchange_weak(
                           cur, used_now,
                           std::memory_order_relaxed)) {}

                sample_count.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(poll_interval_ms_));
            }
        });

        func();
        cudaDeviceSynchronize();

        running.store(false, std::memory_order_relaxed);
        poller.join();

        // final sample after join
        {
            size_t free_now = 0, total_now = 0;
            cudaMemGetInfo(&free_now, &total_now);
            size_t used_now = total_now - free_now;
            size_t cur = peak_used.load(std::memory_order_relaxed);
            while (used_now > cur &&
                   !peak_used.compare_exchange_weak(
                       cur, used_now, std::memory_order_relaxed)) {}
        }

        MemoryResult r;
        r.baseline_bytes   = used_before;
        r.peak_used_bytes  = peak_used.load();
        r.workspace_bytes  = (peak_used.load() > used_before)
                             ? peak_used.load() - used_before : 0;
        r.poll_interval_ms = poll_interval_ms_;
        r.sample_count     = sample_count.load();
        return r;
#endif
    }

private:
    int poll_interval_ms_;
};

} } // namespace st::util
