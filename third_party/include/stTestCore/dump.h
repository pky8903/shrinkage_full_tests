#pragma once
#include "arg_descriptor.h"
#include "buffer.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdint>

namespace st { namespace util {

// ─────────────────────────────────────────────────────────────
//  dump_buffer
//
//  Writes a buffer to a text file in a format compatible with
//  numpy.loadtxt / numpy.genfromtxt.
//
//  File format:
//    # name: <name>       (optional)
//    # dtype: <dtype>
//    # shape: W H
//    v00 v01 ... v0(W-1)
//    v10 v11 ...
//    ...
//
//  Usage:
//    dump_buffer("out/x.txt", ptr, MemSpace::HOST,
//                DType::FLOAT32, Dims::make_2d(2,2), "x");
//
//  Python read:
//    import numpy as np
//    x = np.loadtxt("out/x.txt")   # skips #-comment lines automatically
// ─────────────────────────────────────────────────────────────

inline void dump_buffer(
    const std::string& filename,
    void*              ptr,
    MemSpace           mem,
    DType              dtype,
    Dims               dims,
    const std::string& name = "")
{
    size_t n   = dims.total();
    size_t esz = dtype_size(dtype);
    std::vector<uint8_t> raw(n * esz);
    copy_to_host(raw.data(), ptr, n * esz, mem);

    FILE* f = fopen(filename.c_str(), "w");
    if (!f) { fprintf(stderr, "[dump] cannot open '%s'\n", filename.c_str()); return; }

    if (!name.empty())     fprintf(f, "# name: %s\n",       name.c_str());
    fprintf(f, "# dtype: %s\n",  dtype_name(dtype).c_str());
    fprintf(f, "# shape: %zu %zu\n", dims.w, dims.h);

    size_t W = dims.w, H = dims.h;
    for (size_t row = 0; row < H; ++row) {
        for (size_t col = 0; col < W; ++col) {
            size_t idx = col * H + row;
            double v = 0.0;
            switch (dtype) {
                case DType::FLOAT16: { uint16_t x; memcpy(&x, raw.data()+idx*esz, esz); v=f16_to_f32(x); break; }
                case DType::FLOAT32: { float    x; memcpy(&x, raw.data()+idx*esz, esz); v=x; break; }
                case DType::FLOAT64: { double   x; memcpy(&x, raw.data()+idx*esz, esz); v=x; break; }
                case DType::INT32:   { int32_t  x; memcpy(&x, raw.data()+idx*esz, esz); v=x; break; }
                case DType::INT16:   { int16_t  x; memcpy(&x, raw.data()+idx*esz, esz); v=x; break; }
                case DType::INT8:    { int8_t   x; memcpy(&x, raw.data()+idx*esz, esz); v=x; break; }
                case DType::UINT32:  { uint32_t x; memcpy(&x, raw.data()+idx*esz, esz); v=x; break; }
                case DType::UINT16:  { uint16_t x; memcpy(&x, raw.data()+idx*esz, esz); v=x; break; }
                case DType::UINT8:   { uint8_t  x; memcpy(&x, raw.data()+idx*esz, esz); v=x; break; }
            }
            if (col + 1 < W) fprintf(f, "%.8g ", v);
            else              fprintf(f, "%.8g\n", v);
        }
    }
    fclose(f);
    printf("  [dump] '%s'  (%zux%zu %s)\n",
           filename.c_str(), W, H, dtype_name(dtype).c_str());
}

// ─────────────────────────────────────────────────────────────
//  dump_adjoint_pixels
//
//  Dumps per-pixel adjoint gradient comparison to a text file.
//
//  File format:
//    # label: <label>      (optional)
//    # nx ny analytical numerical rel_error abs_error
//    0 0 32.0 32.000008 2.5e-7 8e-6
//    ...
//
//  Templated to avoid circular dependency with adjoint_tester.h.
//  Expects AdjResult to have:
//    .pixels     — iterable of {.nx, .ny, .analytical, .numerical, .rel_error, .abs_error}
//    .num_samples — int
// ─────────────────────────────────────────────────────────────

template<typename AdjResult>
inline void dump_adjoint_pixels(
    const std::string& filename,
    const AdjResult&   r,
    const std::string& label = "")
{
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) { fprintf(stderr, "[dump] cannot open '%s'\n", filename.c_str()); return; }
    if (!label.empty()) fprintf(f, "# label: %s\n", label.c_str());
    fprintf(f, "# nx ny analytical numerical rel_error abs_error\n");
    for (auto& p : r.pixels)
        fprintf(f, "%zu %zu %.8g %.8g %.8g %.8g\n",
                p.nx, p.ny, p.analytical, p.numerical, p.rel_error, p.abs_error);
    fclose(f);
    printf("  [dump] '%s'  (%d pixels)\n", filename.c_str(), r.num_samples);
}

} } // namespace st::util
