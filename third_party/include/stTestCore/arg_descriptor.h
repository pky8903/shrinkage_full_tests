#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>

namespace st { namespace util {

// ─────────────────────────────────────────────
//  Enums
// ─────────────────────────────────────────────

enum class ParamRole { INPUT, OUTPUT, PARAM };

enum class DType {
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT32,
    INT16,
    INT8,
    UINT32,
    UINT16,
    UINT8,
};

// ─────────────────────────────────────────────
//  float16 ↔ float32 (pure C++, no CUDA)
// ─────────────────────────────────────────────
inline float f16_to_f32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1f;
    int mant = h & 0x3ff;
    float result;
    if (exp == 0) {
        result = ldexpf((float)mant, -24);          // subnormal / zero
    } else if (exp == 31) {
        uint32_t bits = ((uint32_t)sign << 31) | 0x7f800000u | ((uint32_t)mant << 13);
        memcpy(&result, &bits, 4);
        return result;
    } else {
        result = ldexpf((float)(mant + 1024), exp - 25);
    }
    return sign ? -result : result;
}

inline uint16_t f32_to_f16(float v) {
    uint32_t f; memcpy(&f, &v, 4);
    int sign     = (f >> 31) & 1;
    int exp_f32  = (f >> 23) & 0xff;
    int exp_f16  = exp_f32 - 127 + 15;
    uint32_t mant = f & 0x7fffff;
    if (exp_f16 <= 0)  return (uint16_t)(sign << 15);
    if (exp_f16 >= 31) return (uint16_t)((sign << 15) | 0x7c00);
    return (uint16_t)((sign << 15) | (exp_f16 << 10) | (mant >> 13));
}

enum class MemSpace { HOST, DEVICE };

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FLOAT16: return 2;
        case DType::FLOAT32: return sizeof(float);
        case DType::FLOAT64: return sizeof(double);
        case DType::INT32:   return sizeof(int32_t);
        case DType::INT16:   return sizeof(int16_t);
        case DType::INT8:    return sizeof(int8_t);
        case DType::UINT32:  return sizeof(uint32_t);
        case DType::UINT16:  return sizeof(uint16_t);
        case DType::UINT8:   return sizeof(uint8_t);
    }
    throw std::invalid_argument("Unknown DType");
}

inline std::string dtype_name(DType dt) {
    switch (dt) {
        case DType::FLOAT16: return "float16";
        case DType::FLOAT32: return "float32";
        case DType::FLOAT64: return "float64";
        case DType::INT32:   return "int32";
        case DType::INT16:   return "int16";
        case DType::INT8:    return "int8";
        case DType::UINT32:  return "uint32";
        case DType::UINT16:  return "uint16";
        case DType::UINT8:   return "uint8";
    }
    return "unknown";
}

// ─────────────────────────────────────────────
//  Dims: 1D or 2D
// ─────────────────────────────────────────────

struct Dims {
    size_t w = 0;   ///< width (1D: total elements)
    size_t h = 1;   ///< height (1D: always 1)

    static Dims make_1d(size_t n)           { return {n, 1}; }
    static Dims make_2d(size_t w, size_t h) { return {w, h}; }

    bool   is_2d()  const { return h > 1; }
    size_t total()  const { return w * h; }
};

// ─────────────────────────────────────────────
//  ArgDescriptor: one argument's metadata
// ─────────────────────────────────────────────

struct ArgDescriptor {
    ParamRole   role;
    DType       dtype;
    MemSpace    mem;
    Dims        dims;
    std::string name; ///< optional, for logging

    static ArgDescriptor input_host  (DType dt, Dims d, std::string n = "") { return {ParamRole::INPUT,  dt, MemSpace::HOST,   d, n}; }
    static ArgDescriptor input_device(DType dt, Dims d, std::string n = "") { return {ParamRole::INPUT,  dt, MemSpace::DEVICE, d, n}; }
    static ArgDescriptor output_host  (DType dt, Dims d, std::string n = "") { return {ParamRole::OUTPUT, dt, MemSpace::HOST,   d, n}; }
    static ArgDescriptor output_device(DType dt, Dims d, std::string n = "") { return {ParamRole::OUTPUT, dt, MemSpace::DEVICE, d, n}; }
    static ArgDescriptor param_host   (DType dt, Dims d, std::string n = "") { return {ParamRole::PARAM,  dt, MemSpace::HOST,   d, n}; }
    static ArgDescriptor param_device (DType dt, Dims d, std::string n = "") { return {ParamRole::PARAM,  dt, MemSpace::DEVICE, d, n}; }
};

} } // namespace st::util
