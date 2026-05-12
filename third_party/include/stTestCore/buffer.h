#pragma once
#include "arg_descriptor.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <memory>

#ifdef API_TESTER_CUDA
#  include <cuda_runtime.h>
#endif

namespace st { namespace util {

// ─────────────────────────────────────────────
//  Buffer: RAII wrapper for host or device mem
// ─────────────────────────────────────────────

struct Buffer {
    void*     ptr      = nullptr;
    MemSpace  mem      = MemSpace::HOST;
    size_t    bytes    = 0;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& o) noexcept
        : ptr(o.ptr), mem(o.mem), bytes(o.bytes) {
        o.ptr = nullptr;
    }
    Buffer& operator=(Buffer&& o) noexcept {
        free_mem();
        ptr = o.ptr; mem = o.mem; bytes = o.bytes;
        o.ptr = nullptr;
        return *this;
    }

    ~Buffer() { free_mem(); }

    bool valid() const { return ptr != nullptr; }

private:
    void free_mem() {
        if (!ptr) return;
        if (mem == MemSpace::HOST) {
            std::free(ptr);
        } else {
#ifdef API_TESTER_CUDA
            cudaFree(ptr);
#endif
        }
        ptr = nullptr;
    }
};

// ─────────────────────────────────────────────
//  Allocate buffer from descriptor
// ─────────────────────────────────────────────

inline Buffer alloc_buffer(const ArgDescriptor& desc) {
    Buffer buf;
    buf.mem   = desc.mem;
    buf.bytes = desc.dims.total() * dtype_size(desc.dtype);

    if (desc.mem == MemSpace::HOST) {
        buf.ptr = std::malloc(buf.bytes);
        if (!buf.ptr) throw std::bad_alloc();
        std::memset(buf.ptr, 0, buf.bytes);
    } else {
#ifdef API_TESTER_CUDA
        cudaError_t err = cudaMalloc(&buf.ptr, buf.bytes);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        cudaMemset(buf.ptr, 0, buf.bytes);
#else
        throw std::runtime_error(
            "MemSpace::DEVICE requested but API_TESTER_CUDA not defined. "
            "Add -DAPI_TESTER_CUDA to your build.");
#endif
    }
    return buf;
}

// ─────────────────────────────────────────────
//  Copy device → host (no-op if already host)
// ─────────────────────────────────────────────

inline void copy_to_host(void* dst, const void* src,
                          size_t bytes, MemSpace src_mem) {
    if (src_mem == MemSpace::HOST) {
        std::memcpy(dst, src, bytes);
    } else {
#ifdef API_TESTER_CUDA
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
}

} } // namespace st::util
