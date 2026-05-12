#pragma once
#include "arg_descriptor.h"
#include "buffer.h"
#include "adapter.h"
#include "comparator.h"
#include "func_tester.h"   // for BufferInitFn, PerfConfig, GpConfig
#include "dump.h"
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

#ifdef API_TESTER_CUDA
#  include <cuda_runtime.h>
#endif

namespace st { namespace util {

// ─────────────────────────────────────────────────────────────
//  AdjointConfig
// ─────────────────────────────────────────────────────────────

struct AdjointConfig {
    int      num_samples = 256;   ///< pixels to check; -1 = full W×H
    double   epsilon     = 1e-3;  ///< perturbation size for central difference
    double   tol_abs     = 1e-5;  ///< absolute error floor: pass if |a-n| <= tol_abs
    double   tol_rel     = 1e-3;  ///< relative tolerance: pass if |a-n| <= tol_rel * max(|a|,|n|)
    uint64_t seed        = 0;     ///< RNG seed (0 = random)
    ///< criterion: |a-n| <= max(tol_abs, tol_rel * max(|a|,|n|))
};

// ─────────────────────────────────────────────────────────────
//  AdjointPixelResult / AdjointResult
// ─────────────────────────────────────────────────────────────

struct AdjointPixelResult {
    size_t nx = 0, ny = 0;
    double analytical = 0.0;
    double numerical  = 0.0;
    double rel_error  = 0.0;
    double abs_error  = 0.0;
};

struct AdjointResult {
    std::vector<AdjointPixelResult> pixels;
    double rel_error_mean  = 0.0;
    double rel_error_max   = 0.0;
    double rel_error_stdev = 0.0;
    size_t worst_nx = 0, worst_ny = 0;
    double abs_error_mean  = 0.0;
    double abs_error_max   = 0.0;
    double abs_error_stdev = 0.0;
    int    num_samples  = 0;
    bool   passed       = false;
    double tol_rel      = -1.0;
    double tol_abs      =  0.0;
    // perf (only populated when PerfConfig::measure_latency/memory is set)
    bool          has_perf   = false;
    LatencyResult fwd_latency;
    LatencyResult adj_latency;
    MemoryResult  fwd_memory;
    MemoryResult  adj_memory;
};

// ─────────────────────────────────────────────────────────────
//  AdjointCase
//
//  forward  : func_forward(input1, input2, output, param1, param2)
//  adjoint1 : func_adjoint_input1(dqdout, input1, input2, dqdinput1, param1, param2)
//
//  Indices you must specify:
//    fwd_out_idx      : which fwd arg is the OUTPUT  (e.g. 2)
//    adj_out_idx      : which adj arg is dqdinput    (e.g. 3)
//    adj_dqdout_idx   : which adj arg is dqd(output) (e.g. 0)
//    fwd_perturb_idx  : which fwd INPUT to perturb   (e.g. 0 = input1)
//    adj_input_idx    : corresponding position in adj_args (e.g. 1)
// ─────────────────────────────────────────────────────────────

struct AdjointCase {
    std::string label;
    AdaptedFunc                fwd_func;
    std::vector<ArgDescriptor> fwd_args;
    size_t                     fwd_out_idx;
    AdaptedFunc                adj_func;
    std::vector<ArgDescriptor> adj_args;
    size_t                     adj_out_idx;
    size_t                     adj_dqdout_idx;
    size_t                     fwd_perturb_idx;
    size_t                     adj_input_idx;
    AdjointConfig              cfg;
    PerfConfig                 perf;
    GpConfig                   gp;
    BufferInitFn               init_fn;
    BufferInitFn               adj_init_fn;  ///< if set, used to init all adj buffers except out/dqdout
    std::string                dump_dir;  ///< if non-empty, dump buffers/pixels to this directory
};

// ─────────────────────────────────────────────────────────────
//  AdjointTester
//
//  Verification method:
//    q(A) = Σ B²    where B = forward(A)
//
//    Analytical:
//      dq/dA = adjoint(dq/dB = 2B, A, ...)
//
//    Numerical (central difference):
//      dq/dA[i,j] ≈ (q(A + ε·e_ij) - q(A - ε·e_ij)) / (2ε)
//
//    Metric (mixed absolute/relative):
//      scale[i,j]   = max(|analytical|, |numerical|)
//      err_tol[i,j] = max(tol_abs, tol_rel * scale[i,j])
//      pass[i,j]    = |analytical - numerical| <= err_tol[i,j]
// ─────────────────────────────────────────────────────────────

class AdjointTester {
public:

    void register_adjoint(
        std::string                label,
        AdaptedFunc                fwd_func,
        std::vector<ArgDescriptor> fwd_args,
        size_t                     fwd_out_idx,
        AdaptedFunc                adj_func,
        std::vector<ArgDescriptor> adj_args,
        size_t                     adj_out_idx,
        size_t                     adj_dqdout_idx,
        size_t                     fwd_perturb_idx,
        size_t                     adj_input_idx,
        AdjointConfig              cfg          = {},
        PerfConfig                 perf         = {},
        BufferInitFn               init_fn      = nullptr,
        BufferInitFn               adj_init_fn  = nullptr,
        GpConfig                   gp           = {},
        std::string                dump_dir     = "")
    {
        // default: enable GP for adjoint input generation when no init_fn given
        if (!init_fn && !gp.enabled) {
            gp.enabled = true;
        }
        AdjointCase ac;
        ac.label            = std::move(label);
        ac.fwd_func         = fwd_func;
        ac.fwd_args         = std::move(fwd_args);
        ac.fwd_out_idx      = fwd_out_idx;
        ac.adj_func         = adj_func;
        ac.adj_args         = std::move(adj_args);
        ac.adj_out_idx      = adj_out_idx;
        ac.adj_dqdout_idx   = adj_dqdout_idx;
        ac.fwd_perturb_idx  = fwd_perturb_idx;
        ac.adj_input_idx    = adj_input_idx;
        ac.cfg              = cfg;
        ac.perf             = perf;
        ac.gp               = gp;
        ac.init_fn          = std::move(init_fn);
        ac.adj_init_fn      = std::move(adj_init_fn);
        ac.dump_dir         = std::move(dump_dir);
        cases_.push_back(std::move(ac));
    }

    std::vector<AdjointResult> run_all() {
        std::vector<AdjointResult> results;
        results.reserve(cases_.size());
        for (auto& c : cases_)
            results.push_back(run_one(c));
        return results;
    }

    AdjointResult run(size_t idx) {
        if (idx >= cases_.size())
            throw std::out_of_range("AdjointCase index out of range");
        return run_one(cases_[idx]);
    }

private:

    std::vector<float> to_host_float(void* ptr, MemSpace mem,
                                      size_t n_elems, DType dtype) {
        size_t bytes = n_elems * dtype_size(dtype);
        std::vector<uint8_t> raw(bytes);
        copy_to_host(raw.data(), ptr, bytes, mem);
        std::vector<float> out(n_elems);
        for (size_t i = 0; i < n_elems; ++i) {
            switch (dtype) {
                case DType::FLOAT16: { uint16_t x; memcpy(&x,raw.data()+i*2,2); out[i]=f16_to_f32(x); break; }
                case DType::FLOAT32: { float   x; memcpy(&x,raw.data()+i*4,4); out[i]=x; break; }
                case DType::FLOAT64: { double  x; memcpy(&x,raw.data()+i*8,8); out[i]=(float)x; break; }
                case DType::INT32:   { int32_t x; memcpy(&x,raw.data()+i*4,4); out[i]=(float)x; break; }
                case DType::INT16:   { int16_t x; memcpy(&x,raw.data()+i*2,2); out[i]=(float)x; break; }
                case DType::INT8:    { int8_t  x; memcpy(&x,raw.data()+i*1,1); out[i]=(float)x; break; }
                default:             { out[i]=0.f; break; }
            }
        }
        return out;
    }

    void set_element_host(void* ptr, size_t idx, float value, DType dtype) {
        uint8_t* b = static_cast<uint8_t*>(ptr);
        size_t esz = dtype_size(dtype);
        switch (dtype) {
            case DType::FLOAT16: { uint16_t x=f32_to_f16(value); memcpy(b+idx*esz,&x,esz); break; }
            case DType::FLOAT32: { float   x=(float)value;   memcpy(b+idx*esz,&x,esz); break; }
            case DType::FLOAT64: { double  x=(double)value;  memcpy(b+idx*esz,&x,esz); break; }
            case DType::INT32:   { int32_t x=(int32_t)value; memcpy(b+idx*esz,&x,esz); break; }
            case DType::INT16:   { int16_t x=(int16_t)value; memcpy(b+idx*esz,&x,esz); break; }
            default: break;
        }
    }

    float get_element_host(const void* ptr, size_t idx, DType dtype) {
        const uint8_t* b = static_cast<const uint8_t*>(ptr);
        size_t esz = dtype_size(dtype);
        switch (dtype) {
            case DType::FLOAT16: { uint16_t x; memcpy(&x,b+idx*esz,esz); return f16_to_f32(x); }
            case DType::FLOAT32: { float   x; memcpy(&x,b+idx*esz,esz); return x; }
            case DType::FLOAT64: { double  x; memcpy(&x,b+idx*esz,esz); return (float)x; }
            case DType::INT32:   { int32_t x; memcpy(&x,b+idx*esz,esz); return (float)x; }
            case DType::INT16:   { int16_t x; memcpy(&x,b+idx*esz,esz); return (float)x; }
            default: return 0.f;
        }
    }

    double compute_q(const std::vector<float>& B) {
        double q = 0.0;
        for (float v : B) q += (double)v * v;
        return q;
    }

    void fill_dqdB(void* ptr, MemSpace mem,
                   const std::vector<float>& B, size_t n, DType dtype) {
        std::vector<uint8_t> host(n * dtype_size(dtype));
        for (size_t i = 0; i < n; ++i)
            set_element_host(host.data(), i, 2.f * B[i], dtype);
        if (mem == MemSpace::HOST) {
            memcpy(ptr, host.data(), host.size());
        } else {
#ifdef API_TESTER_CUDA
            cudaMemcpy(ptr, host.data(), host.size(), cudaMemcpyHostToDevice);
#endif
        }
    }

    void perturb_element(Buffer& buf, const ArgDescriptor& desc,
                         size_t linear_idx, float delta) {
        if (desc.mem == MemSpace::HOST) {
            float orig = get_element_host(buf.ptr, linear_idx, desc.dtype);
            set_element_host(buf.ptr, linear_idx, orig + delta, desc.dtype);
        } else {
#ifdef API_TESTER_CUDA
            size_t esz = dtype_size(desc.dtype);
            std::vector<uint8_t> tmp(esz);
            cudaMemcpy(tmp.data(),
                       static_cast<uint8_t*>(buf.ptr) + linear_idx*esz,
                       esz, cudaMemcpyDeviceToHost);
            float orig = get_element_host(tmp.data(), 0, desc.dtype);
            set_element_host(tmp.data(), 0, orig + delta, desc.dtype);
            cudaMemcpy(static_cast<uint8_t*>(buf.ptr) + linear_idx*esz,
                       tmp.data(), esz, cudaMemcpyHostToDevice);
#endif
        }
    }

    struct CallBuffers {
        std::vector<Buffer> bufs;
        std::vector<void*>  args;
    };

    CallBuffers alloc_call(const std::vector<ArgDescriptor>& descs) {
        CallBuffers cb;
        cb.bufs.resize(descs.size());
        cb.args.resize(descs.size());
        for (size_t i = 0; i < descs.size(); ++i) {
            cb.bufs[i] = alloc_buffer(descs[i]);
            cb.args[i] = cb.bufs[i].ptr;
        }
        return cb;
    }

    void zero_buffer(Buffer& buf) {
        if (!buf.ptr) return;
        if (buf.mem == MemSpace::HOST) memset(buf.ptr, 0, buf.bytes);
#ifdef API_TESTER_CUDA
        else cudaMemset(buf.ptr, 0, buf.bytes);
#endif
    }

    void copy_buffer(Buffer& dst, const Buffer& src) {
        if (dst.mem == MemSpace::HOST && src.mem == MemSpace::HOST)
            memcpy(dst.ptr, src.ptr, src.bytes);
#ifdef API_TESTER_CUDA
        else if (dst.mem == MemSpace::DEVICE && src.mem == MemSpace::DEVICE)
            cudaMemcpy(dst.ptr, src.ptr, src.bytes, cudaMemcpyDeviceToDevice);
        else if (dst.mem == MemSpace::DEVICE && src.mem == MemSpace::HOST)
            cudaMemcpy(dst.ptr, src.ptr, src.bytes, cudaMemcpyHostToDevice);
        else
            cudaMemcpy(dst.ptr, src.ptr, src.bytes, cudaMemcpyDeviceToHost);
#endif
    }

    AdjointResult run_one(AdjointCase& tc) {
        const auto& cfg          = tc.cfg;
        const auto& perturb_desc = tc.fwd_args[tc.fwd_perturb_idx];
        const auto& fwd_out_desc = tc.fwd_args[tc.fwd_out_idx];
        const size_t N_in  = perturb_desc.dims.total();
        const size_t N_out = fwd_out_desc.dims.total();

        printf("\n[AdjointTester] %s\n", tc.label.c_str());

        // ── 1. Alloc + init forward buffers ──────────────────────
        CallBuffers fwd = alloc_call(tc.fwd_args);
        std::mt19937_64 rng(cfg.seed == 0 ? std::random_device{}() : cfg.seed);
        std::normal_distribution<float> normal(0.f, 1.f);

        auto is_pow2 = [](size_t x){ return x > 0 && (x & (x - 1)) == 0; };

        for (size_t i = 0; i < tc.fwd_args.size(); ++i) {
            if (tc.fwd_args[i].role == ParamRole::OUTPUT) continue;
            ArgDescriptor hd = tc.fwd_args[i];
            hd.mem = MemSpace::HOST;
            Buffer htmp = alloc_buffer(hd);
            if (tc.init_fn) {
                tc.init_fn(htmp.ptr, hd, i);
            } else if (tc.fwd_args[i].role == ParamRole::INPUT) {
                size_t n = hd.dims.total();
                int W = static_cast<int>(hd.dims.w);
                int H = static_cast<int>(hd.dims.h);
                bool used_gp = false;
                if (tc.gp.enabled && is_pow2(static_cast<size_t>(W))
                                  && is_pow2(static_cast<size_t>(H))) {
                    uint64_t gseed = tc.gp.seed ? tc.gp.seed
                                                : (cfg.seed + static_cast<uint64_t>(i) + 1);
                    GpGenerator gen(tc.gp.cov, W, H, gseed);
                    auto sample = gen.generate();
                    size_t copy_n = std::min(n, sample.size());
                    for (size_t j = 0; j < copy_n; ++j)
                        set_element_host(htmp.ptr, j, sample[j], hd.dtype);
                    printf("  [GP] arg[%zu] \"%s\": %dx%d GP sample\n",
                           i, hd.name.c_str(), W, H);
                    used_gp = true;
                }
                if (!used_gp) {
                    for (size_t j = 0; j < n; ++j)
                        set_element_host(htmp.ptr, j, normal(rng), hd.dtype);
                    if (tc.gp.enabled)
                        printf("  [GP] arg[%zu] \"%s\": dims not power-of-2 (%dx%d), fallback N(0,1)\n",
                               i, hd.name.c_str(), W, H);
                }
                // PARAMs stay zero — provide init_fn for params
            }
            copy_buffer(fwd.bufs[i], htmp);
        }

        // ── 2. Baseline forward ───────────────────────────────────
        zero_buffer(fwd.bufs[tc.fwd_out_idx]);
        tc.fwd_func(fwd.args.data());
        auto B_base = to_host_float(fwd.args[tc.fwd_out_idx],
                                     fwd_out_desc.mem, N_out, fwd_out_desc.dtype);

        // ── Dump fwd input / output (if requested) ────────────────
        if (!tc.dump_dir.empty()) {
            std::string safe = tc.label;
            for (char& c : safe) if (!isalnum(c) && c!='_' && c!='-') c='_';
            mkdir_p("dump/" + tc.dump_dir);
            const auto& in_desc  = tc.fwd_args[tc.fwd_perturb_idx];
            const std::string in_name  = in_desc.name.empty()      ? "input"  : in_desc.name;
            const std::string out_name = fwd_out_desc.name.empty() ? "output" : fwd_out_desc.name;
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_" + in_name + ".txt",
                        fwd.args[tc.fwd_perturb_idx],
                        in_desc.mem, in_desc.dtype, in_desc.dims, in_name);
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_" + out_name + ".txt",
                        fwd.args[tc.fwd_out_idx],
                        fwd_out_desc.mem, fwd_out_desc.dtype, fwd_out_desc.dims, out_name);
        }

        // ── 3. Alloc adjoint buffers + fill dqdB = 2B ────────────
        CallBuffers adj = alloc_call(tc.adj_args);
        fill_dqdB(adj.args[tc.adj_dqdout_idx],
                  tc.adj_args[tc.adj_dqdout_idx].mem,
                  B_base, N_out, tc.adj_args[tc.adj_dqdout_idx].dtype);

        // ── 3b. Initialize remaining adjoint buffers ─────────────
        if (tc.adj_init_fn) {
            // adj_init_fn handles all adj buffers except output and dqdout
            for (size_t i = 0; i < tc.adj_args.size(); ++i) {
                if (i == tc.adj_out_idx || i == tc.adj_dqdout_idx) continue;
                ArgDescriptor hd = tc.adj_args[i];
                hd.mem = MemSpace::HOST;
                Buffer htmp = alloc_buffer(hd);
                tc.adj_init_fn(htmp.ptr, tc.adj_args[i], i);
                copy_buffer(adj.bufs[i], htmp);
            }
        } else {
            // Default: copy fwd input → adj input, then copy PARAMs by name/order
            if (tc.adj_input_idx != SIZE_MAX) {
                copy_buffer(adj.bufs[tc.adj_input_idx], fwd.bufs[tc.fwd_perturb_idx]);
            }
            // Copy PARAMs: name match first, fall back to declaration order.
            std::vector<bool> adj_matched(tc.adj_args.size(), false);
            for (size_t fi = 0; fi < tc.fwd_args.size(); ++fi) {
                if (tc.fwd_args[fi].role != ParamRole::PARAM) continue;
                const std::string& fname = tc.fwd_args[fi].name;

                bool found = false;
                // 1. name match
                if (!fname.empty()) {
                    for (size_t ai = 0; ai < tc.adj_args.size(); ++ai) {
                        if (adj_matched[ai]) continue;
                        if (ai == tc.adj_out_idx || ai == tc.adj_dqdout_idx) continue;
                        if (tc.adj_args[ai].role != ParamRole::PARAM) continue;
                        if (tc.adj_args[ai].name == fname) {
                            copy_buffer(adj.bufs[ai], fwd.bufs[fi]);
                            adj_matched[ai] = true;
                            found = true;
                            break;
                        }
                    }
                }
                // 2. order fallback
                if (!found) {
                    for (size_t ai = 0; ai < tc.adj_args.size(); ++ai) {
                        if (adj_matched[ai]) continue;
                        if (ai == tc.adj_out_idx || ai == tc.adj_dqdout_idx) continue;
                        if (tc.adj_args[ai].role != ParamRole::PARAM) continue;
                        copy_buffer(adj.bufs[ai], fwd.bufs[fi]);
                        adj_matched[ai] = true;
                        break;
                    }
                }
            }
        }

        // ── 4. Run adjoint → analytical gradient ─────────────────
        zero_buffer(adj.bufs[tc.adj_out_idx]);
        tc.adj_func(adj.args.data());
        auto dqdinput_analytical = to_host_float(
            adj.args[tc.adj_out_idx],
            tc.adj_args[tc.adj_out_idx].mem,
            N_in, tc.adj_args[tc.adj_out_idx].dtype);

        // ── Dump analytical gradient (if requested) ───────────────
        if (!tc.dump_dir.empty()) {
            std::string safe = tc.label;
            for (char& c : safe) if (!isalnum(c) && c!='_' && c!='-') c='_';
            const auto& in_desc   = tc.fwd_args[tc.fwd_perturb_idx];
            const std::string in_name  = in_desc.name.empty() ? "input" : in_desc.name;
            const auto& grad_desc = tc.adj_args[tc.adj_out_idx];
            const std::string grad_label = "dqd" + in_name + "_analytical";
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_" + grad_label + ".txt",
                        adj.args[tc.adj_out_idx],
                        grad_desc.mem, grad_desc.dtype, grad_desc.dims, grad_label);
        }

        // ── 5. Numerical gradient (central difference) ────────────
        std::vector<size_t> sample_indices;
        if (cfg.num_samples < 0 || cfg.num_samples >= static_cast<int>(N_in)) {
            sample_indices.resize(N_in);
            std::iota(sample_indices.begin(), sample_indices.end(), 0);
            printf("  Mode: full (%zu pixels)\n", N_in);
        } else {
            std::vector<size_t> all(N_in);
            std::iota(all.begin(), all.end(), 0);
            std::shuffle(all.begin(), all.end(), rng);
            sample_indices.assign(all.begin(), all.begin() + cfg.num_samples);
            printf("  Mode: random sampling (%d / %zu pixels)\n",
                   cfg.num_samples, N_in);
        }

        CallBuffers fwd_p = alloc_call(tc.fwd_args);
        CallBuffers fwd_m = alloc_call(tc.fwd_args);
        for (size_t i = 0; i < tc.fwd_args.size(); ++i) {
            copy_buffer(fwd_p.bufs[i], fwd.bufs[i]);
            copy_buffer(fwd_m.bufs[i], fwd.bufs[i]);
        }

        std::vector<AdjointPixelResult> pixel_results;
        pixel_results.reserve(sample_indices.size());

        const double eps       = cfg.epsilon;
        const double inv_2eps  = 1.0 / (2.0 * eps);
        const size_t H_perturb = perturb_desc.dims.h;
        size_t progress_step  = std::max(size_t(1), sample_indices.size() / 10);

        printf("  Running central difference (epsilon=%.2e)...\n", eps);

        for (size_t s = 0; s < sample_indices.size(); ++s) {
            size_t lin = sample_indices[s];

            // A + eps*e_ij
            copy_buffer(fwd_p.bufs[tc.fwd_perturb_idx], fwd.bufs[tc.fwd_perturb_idx]);
            perturb_element(fwd_p.bufs[tc.fwd_perturb_idx], perturb_desc, lin, (float)eps);
            zero_buffer(fwd_p.bufs[tc.fwd_out_idx]);
            tc.fwd_func(fwd_p.args.data());
            double q_plus = compute_q(to_host_float(
                fwd_p.args[tc.fwd_out_idx], fwd_out_desc.mem, N_out, fwd_out_desc.dtype));

            // A - eps*e_ij
            copy_buffer(fwd_m.bufs[tc.fwd_perturb_idx], fwd.bufs[tc.fwd_perturb_idx]);
            perturb_element(fwd_m.bufs[tc.fwd_perturb_idx], perturb_desc, lin, -(float)eps);
            zero_buffer(fwd_m.bufs[tc.fwd_out_idx]);
            tc.fwd_func(fwd_m.args.data());
            double q_minus = compute_q(to_host_float(
                fwd_m.args[tc.fwd_out_idx], fwd_out_desc.mem, N_out, fwd_out_desc.dtype));

            double numerical  = (q_plus - q_minus) * inv_2eps;
            double analytical = static_cast<double>(dqdinput_analytical[lin]);
            double abs_err    = std::abs(analytical - numerical);
            double scale      = std::max(std::abs(analytical), std::abs(numerical));
            double rel_err    = abs_err / (scale + cfg.tol_abs);

            AdjointPixelResult pr;
            pr.nx         = perturb_desc.dims.is_2d() ? (lin / H_perturb) : lin;
            pr.ny         = perturb_desc.dims.is_2d() ? (lin % H_perturb) : 0;
            pr.analytical = analytical;
            pr.numerical  = numerical;
            pr.abs_error  = abs_err;
            pr.rel_error  = rel_err;
            pixel_results.push_back(pr);

            if ((s + 1) % progress_step == 0 || s + 1 == sample_indices.size())
                printf("    %5zu / %5zu  done\r", s + 1, sample_indices.size());
        }
        printf("\n");

        // ── 6. Aggregate ──────────────────────────────────────────
        AdjointResult result;
        result.pixels      = pixel_results;
        result.num_samples = static_cast<int>(pixel_results.size());
        result.tol_rel     = cfg.tol_rel;
        result.tol_abs     = cfg.tol_abs;

        std::vector<double> rel_errs, abs_errs;
        for (auto& p : pixel_results) {
            rel_errs.push_back(p.rel_error);
            abs_errs.push_back(p.abs_error);
        }

        auto stat = [](const std::vector<double>& v,
                       double& mean, double& mx, double& sd) {
            const int n = static_cast<int>(v.size());
            mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
            mx   = *std::max_element(v.begin(), v.end());
            double var = 0.0;
            for (double x : v) var += (x - mean) * (x - mean);
            sd = std::sqrt(var / n);
        };

        // FIX: separate stdev fields for rel and abs errors
        stat(rel_errs, result.rel_error_mean, result.rel_error_max, result.rel_error_stdev);
        stat(abs_errs, result.abs_error_mean, result.abs_error_max, result.abs_error_stdev);

        size_t worst = static_cast<size_t>(
            std::max_element(rel_errs.begin(), rel_errs.end()) - rel_errs.begin());
        result.worst_nx = pixel_results[worst].nx;
        result.worst_ny = pixel_results[worst].ny;
        result.passed   = std::all_of(pixel_results.begin(), pixel_results.end(),
            [&](const AdjointPixelResult& p) {
                double sc = std::max(std::abs(p.analytical), std::abs(p.numerical));
                return p.abs_error <= std::max(cfg.tol_abs, cfg.tol_rel * sc);
            });

        // ── Dump pixel table (if requested) ───────────────────────
        if (!tc.dump_dir.empty()) {
            std::string safe = tc.label;
            for (char& c : safe) if (!isalnum(c) && c!='_' && c!='-') c='_';
            dump_adjoint_pixels("dump/" + tc.dump_dir + "/" + safe + "_pixels.txt",
                                result, tc.label);
        }

        // ── 7. Perf (latency + workspace) ────────────────────────
        // Runs forward and adjoint independently after verification.
        // Input/output buffers are already allocated → workspace only.
        if (tc.perf.measure_latency) {
            result.has_perf = true;
            result.fwd_latency = measure_latency([&]{
                zero_buffer(fwd.bufs[tc.fwd_out_idx]);
                tc.fwd_func(fwd.args.data());
            }, tc.perf.warmup_runs, tc.perf.bench_runs);
            result.adj_latency = measure_latency([&]{
                zero_buffer(adj.bufs[tc.adj_out_idx]);
                tc.adj_func(adj.args.data());
            }, tc.perf.warmup_runs, tc.perf.bench_runs);
        }

        if (tc.perf.measure_memory) {
            result.has_perf = true;
            MemoryTracker tracker(tc.perf.poll_interval_ms);
            result.fwd_memory = tracker.measure([&]{
                zero_buffer(fwd.bufs[tc.fwd_out_idx]);
                tc.fwd_func(fwd.args.data());
            });
            result.adj_memory = tracker.measure([&]{
                zero_buffer(adj.bufs[tc.adj_out_idx]);
                tc.adj_func(adj.args.data());
            });
        }

        print_result(result, tc);
        return result;
    }

    void print_result(const AdjointResult& r, const AdjointCase& tc) {
        std::cout << "═════════════════════════════════════════════\n";
        std::cout << "  Adjoint Test : " << tc.label << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Samples      : " << r.num_samples << "\n";
        std::cout << "  epsilon      : " << tc.cfg.epsilon << "\n";
        // Per-pixel table for small cases (≤64 samples)
        if (r.num_samples <= 64) {
            std::cout << "  ── Per-pixel gradient ──\n";
            std::cout << "    " << std::setw(6) << "nx"
                      << std::setw(6) << "ny"
                      << std::setw(16) << "analytical"
                      << std::setw(16) << "numerical"
                      << std::setw(14) << "rel_err"
                      << std::setw(14) << "abs_err" << "\n";
            for (auto& p : r.pixels) {
                std::cout << "    " << std::setw(6) << p.nx
                          << std::setw(6) << p.ny
                          << std::setw(16) << p.analytical
                          << std::setw(16) << p.numerical
                          << std::setw(14) << p.rel_error
                          << std::setw(14) << p.abs_error << "\n";
            }
        }
        std::cout << "  ── Relative error ──\n";
        std::cout << "  mean   : " << r.rel_error_mean  << "\n";
        std::cout << "  max    : " << r.rel_error_max
                  << "  @ (" << r.worst_nx << ", " << r.worst_ny << ")\n";
        std::cout << "  stdev  : " << r.rel_error_stdev << "\n";
        std::cout << "  ── Absolute error ──\n";
        std::cout << "  mean   : " << r.abs_error_mean << "\n";
        std::cout << "  max    : " << r.abs_error_max  << "\n";
        std::cout << "  stdev  : " << r.abs_error_stdev << "\n";
        if (tc.cfg.tol_rel >= 0.0)
            std::cout << "  Result : "
                      << (r.passed ? "PASS v" : "FAIL x")
                      << "  (tol_rel=" << tc.cfg.tol_rel
                      << ", tol_abs=" << tc.cfg.tol_abs << ")\n";

        if (r.has_perf && tc.perf.measure_latency) {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  ── Latency (warmup=" << r.fwd_latency.warmup
                      << ", runs=" << r.fwd_latency.runs << ") ──\n";
            std::cout << "  fwd  avg=" << r.fwd_latency.avg_ms << "ms"
                      << "  min=" << r.fwd_latency.min_ms << "ms"
                      << "  max=" << r.fwd_latency.max_ms << "ms\n";
            std::cout << "  adj  avg=" << r.adj_latency.avg_ms << "ms"
                      << "  min=" << r.adj_latency.min_ms << "ms"
                      << "  max=" << r.adj_latency.max_ms << "ms\n";
        }
        if (r.has_perf && tc.perf.measure_memory) {
            std::cout << "  ── Workspace Memory (poll=" << r.fwd_memory.poll_interval_ms << "ms) ──\n";
            std::cout << "  fwd  peak workspace : " << format_bytes(r.fwd_memory.workspace_bytes) << "\n";
            std::cout << "  adj  peak workspace : " << format_bytes(r.adj_memory.workspace_bytes) << "\n";
            std::cout << "  (alloc+free within poll interval may be missed)\n";
        }
        std::cout << "═════════════════════════════════════════════\n";
    }

    static void mkdir_p(const std::string& path) {
        for (size_t i = 1; i <= path.size(); ++i) {
            if (i == path.size() || path[i] == '/') {
                std::string sub = path.substr(0, i);
                mkdir(sub.c_str(), 0755);  // ignore error (already exists is ok)
            }
        }
    }

    std::vector<AdjointCase> cases_;
};

} } // namespace st::util
