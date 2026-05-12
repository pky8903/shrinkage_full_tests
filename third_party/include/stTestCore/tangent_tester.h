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
//  TangentConfig
//
//  Validation criterion (per PRD §6.2):
//    pass if ∃ ε in `epsilons` such that ∀ pixels (i,j),
//      |J1_ij(ε) - J2_ij| ≤ max(tol_abs, tol_rel * max(|J1_ij(ε)|, |J2_ij|))
//
//  where
//    J1(ε) = (fwd(state + ε·dir) - fwd(state - ε·dir)) / (2ε)   numerical
//    J2    = tan(state, dir)                                    analytical
// ─────────────────────────────────────────────────────────────

inline std::vector<double> default_epsilon_sweep() {
    return {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1};
}

struct TangentConfig {
    int                  num_samples = -1;             ///< pixels to check; -1 = full W×H
    std::vector<double>  epsilons    = default_epsilon_sweep();
    double               tol_abs     = 1e-5;
    double               tol_rel     = 1e-3;
    uint64_t             seed        = 0;
};

// ─────────────────────────────────────────────────────────────
//  PerturbSpec
//
//  Pairs a forward-input slot to perturb with the tangent-arg
//  slot holding the perturbation direction.
// ─────────────────────────────────────────────────────────────

struct PerturbSpec {
    size_t fwd_idx;       ///< index into fwd_args being perturbed
    size_t tan_dir_idx;   ///< index in tan_args holding the perturbation direction
};

// ─────────────────────────────────────────────────────────────
//  TangentPixelResult / TangentEpsilonResult / TangentResult
// ─────────────────────────────────────────────────────────────

struct TangentPixelResult {
    size_t nx = 0, ny = 0;
    double analytical = 0.0;
    double numerical  = 0.0;
    double rel_error  = 0.0;
    double abs_error  = 0.0;
};

struct TangentEpsilonResult {
    double                          epsilon = 0.0;
    std::vector<TangentPixelResult> pixels;
    int                             num_samples = 0;   ///< == pixels.size() (for dump_adjoint_pixels)
    double                          rel_error_max  = 0.0;
    double                          abs_error_max  = 0.0;
    double                          rel_error_mean = 0.0;
    double                          abs_error_mean = 0.0;
    size_t                          worst_nx = 0, worst_ny = 0;
    bool                            passed   = false;
    int                             num_violations = 0;
};

struct TangentResult {
    std::vector<TangentEpsilonResult> per_epsilon;
    bool   passed       = false;
    double best_epsilon = 0.0;
    double best_rel_error_max = 0.0;
    double best_abs_error_max = 0.0;
    size_t best_worst_nx = 0, best_worst_ny = 0;
    int    best_num_violations = 0;
    int    num_samples = 0;
    double tol_rel = -1.0;
    double tol_abs = 0.0;
    // perf
    bool          has_perf = false;
    LatencyResult fwd_latency;
    LatencyResult tan_latency;
    MemoryResult  fwd_memory;
    MemoryResult  tan_memory;
};

// ─────────────────────────────────────────────────────────────
//  TangentCase
//
//  forward : fwd_func(args...)        producing fwd_args[fwd_out_idx] = h
//  tangent : tan_func(args...)        producing tan_args[tan_out_idx] = dh
//
//  perturb_specs : list of {fwd_idx, tan_dir_idx} pairs identifying
//                  which forward INPUTs to perturb and the
//                  corresponding direction buffer in tan_args.
//                  For mode-style tests (df-only, dg-only, full),
//                  user controls this by including / omitting specs
//                  and / or zeroing direction buffers via tan_init_fn.
// ─────────────────────────────────────────────────────────────

struct TangentCase {
    std::string                label;
    AdaptedFunc                fwd_func;
    std::vector<ArgDescriptor> fwd_args;
    size_t                     fwd_out_idx;
    AdaptedFunc                tan_func;
    std::vector<ArgDescriptor> tan_args;
    size_t                     tan_out_idx;
    std::vector<PerturbSpec>   perturb_specs;
    TangentConfig              cfg;
    PerfConfig                 perf;
    GpConfig                   gp;
    BufferInitFn               init_fn;     ///< fwd buffer init
    BufferInitFn               tan_init_fn; ///< tan buffer init (state + directions), excludes output
    std::string                dump_dir;
};

// ─────────────────────────────────────────────────────────────
//  TangentTester
// ─────────────────────────────────────────────────────────────

class TangentTester {
public:

    void register_tangent(
        std::string                label,
        AdaptedFunc                fwd_func,
        std::vector<ArgDescriptor> fwd_args,
        size_t                     fwd_out_idx,
        AdaptedFunc                tan_func,
        std::vector<ArgDescriptor> tan_args,
        size_t                     tan_out_idx,
        std::vector<PerturbSpec>   perturb_specs,
        TangentConfig              cfg          = {},
        PerfConfig                 perf         = {},
        BufferInitFn               init_fn      = nullptr,
        BufferInitFn               tan_init_fn  = nullptr,
        GpConfig                   gp           = {},
        std::string                dump_dir     = "")
    {
        if (!init_fn && !gp.enabled) gp.enabled = true;
        TangentCase tc;
        tc.label          = std::move(label);
        tc.fwd_func       = fwd_func;
        tc.fwd_args       = std::move(fwd_args);
        tc.fwd_out_idx    = fwd_out_idx;
        tc.tan_func       = tan_func;
        tc.tan_args       = std::move(tan_args);
        tc.tan_out_idx    = tan_out_idx;
        tc.perturb_specs  = std::move(perturb_specs);
        tc.cfg            = cfg;
        tc.perf           = perf;
        tc.gp             = gp;
        tc.init_fn        = std::move(init_fn);
        tc.tan_init_fn    = std::move(tan_init_fn);
        tc.dump_dir       = std::move(dump_dir);
        cases_.push_back(std::move(tc));
    }

    std::vector<TangentResult> run_all() {
        std::vector<TangentResult> results;
        results.reserve(cases_.size());
        for (auto& c : cases_) results.push_back(run_one(c));
        return results;
    }

    TangentResult run(size_t idx) {
        if (idx >= cases_.size())
            throw std::out_of_range("TangentCase index out of range");
        return run_one(cases_[idx]);
    }

private:

    // ── byte-level helpers (same as AdjointTester) ────────────
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
            case DType::FLOAT32: { float   x=(float)value;       memcpy(b+idx*esz,&x,esz); break; }
            case DType::FLOAT64: { double  x=(double)value;      memcpy(b+idx*esz,&x,esz); break; }
            default: break;
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

    // dst <- src + scale * dir   (typed; element-wise)
    void axpy_buffer(Buffer& dst, const Buffer& src, const Buffer& dir,
                     double scale, DType dtype, size_t n_elems) {
        size_t bytes = n_elems * dtype_size(dtype);
        std::vector<uint8_t> hsrc(bytes), hdir(bytes), hdst(bytes);
        copy_to_host(hsrc.data(), src.ptr, bytes, src.mem);
        copy_to_host(hdir.data(), dir.ptr, bytes, dir.mem);
        for (size_t i = 0; i < n_elems; ++i) {
            float s = 0.f, d = 0.f;
            switch (dtype) {
                case DType::FLOAT16: { uint16_t x; memcpy(&x,hsrc.data()+i*2,2); s=f16_to_f32(x);
                                       uint16_t y; memcpy(&y,hdir.data()+i*2,2); d=f16_to_f32(y); break; }
                case DType::FLOAT32: { float x; memcpy(&x,hsrc.data()+i*4,4); s=x;
                                       float y; memcpy(&y,hdir.data()+i*4,4); d=y; break; }
                case DType::FLOAT64: { double x; memcpy(&x,hsrc.data()+i*8,8); s=(float)x;
                                       double y; memcpy(&y,hdir.data()+i*8,8); d=(float)y; break; }
                default: break;
            }
            set_element_host(hdst.data(), i, (float)(s + scale * d), dtype);
        }
        if (dst.mem == MemSpace::HOST) {
            memcpy(dst.ptr, hdst.data(), bytes);
        } else {
#ifdef API_TESTER_CUDA
            cudaMemcpy(dst.ptr, hdst.data(), bytes, cudaMemcpyHostToDevice);
#endif
        }
    }

    // Initialize all non-output fwd buffers via init_fn or random N(0,1) (with optional GP).
    void init_fwd_buffers(CallBuffers& fwd, const TangentCase& tc, std::mt19937_64& rng) {
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
                if (tc.gp.enabled && is_pow2(W) && is_pow2(H)) {
                    uint64_t gseed = tc.gp.seed ? tc.gp.seed : (tc.cfg.seed + (uint64_t)i + 1);
                    GpGenerator gen(tc.gp.cov, W, H, gseed);
                    auto sample = gen.generate();
                    size_t copy_n = std::min(n, sample.size());
                    for (size_t j = 0; j < copy_n; ++j)
                        set_element_host(htmp.ptr, j, sample[j], hd.dtype);
                    used_gp = true;
                }
                if (!used_gp) {
                    for (size_t j = 0; j < n; ++j)
                        set_element_host(htmp.ptr, j, normal(rng), hd.dtype);
                }
            }
            copy_buffer(fwd.bufs[i], htmp);
        }
    }

    // Initialize all non-output tan buffers from tan_init_fn (or auto-copy from fwd by name).
    void init_tan_buffers(CallBuffers& tan, CallBuffers& fwd, const TangentCase& tc,
                          std::mt19937_64& rng) {
        std::normal_distribution<float> normal(0.f, 1.f);

        if (tc.tan_init_fn) {
            for (size_t i = 0; i < tc.tan_args.size(); ++i) {
                if (i == tc.tan_out_idx) continue;
                ArgDescriptor hd = tc.tan_args[i];
                hd.mem = MemSpace::HOST;
                Buffer htmp = alloc_buffer(hd);
                tc.tan_init_fn(htmp.ptr, tc.tan_args[i], i);
                copy_buffer(tan.bufs[i], htmp);
            }
            return;
        }

        // Auto-copy: for each non-output tan slot, copy from fwd by name match.
        // Direction slots (those listed in perturb_specs) are filled with random N(0,1).
        std::vector<bool> is_dir(tc.tan_args.size(), false);
        for (auto& s : tc.perturb_specs) is_dir[s.tan_dir_idx] = true;

        for (size_t i = 0; i < tc.tan_args.size(); ++i) {
            if (i == tc.tan_out_idx) continue;
            if (is_dir[i]) {
                ArgDescriptor hd = tc.tan_args[i];
                hd.mem = MemSpace::HOST;
                Buffer htmp = alloc_buffer(hd);
                size_t n = hd.dims.total();
                for (size_t j = 0; j < n; ++j)
                    set_element_host(htmp.ptr, j, normal(rng), hd.dtype);
                copy_buffer(tan.bufs[i], htmp);
                continue;
            }
            const std::string& tname = tc.tan_args[i].name;
            bool matched = false;
            if (!tname.empty()) {
                for (size_t fi = 0; fi < tc.fwd_args.size(); ++fi) {
                    if (fi == tc.fwd_out_idx) continue;
                    if (tc.fwd_args[fi].name == tname) {
                        copy_buffer(tan.bufs[i], fwd.bufs[fi]);
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) zero_buffer(tan.bufs[i]);
        }
    }

    TangentResult run_one(TangentCase& tc) {
        const auto& cfg          = tc.cfg;
        const auto& tan_out_desc = tc.tan_args[tc.tan_out_idx];
        const auto& fwd_out_desc = tc.fwd_args[tc.fwd_out_idx];
        const size_t N_out = fwd_out_desc.dims.total();

        printf("\n[TangentTester] %s\n", tc.label.c_str());
        printf("  Epsilon sweep: %zu values in [%.1e, %.1e]\n",
               cfg.epsilons.size(),
               cfg.epsilons.empty() ? 0.0 : cfg.epsilons.front(),
               cfg.epsilons.empty() ? 0.0 : cfg.epsilons.back());

        std::mt19937_64 rng(cfg.seed == 0 ? std::random_device{}() : cfg.seed);

        // ── 1. Alloc + init forward buffers ──────────────────────
        CallBuffers fwd = alloc_call(tc.fwd_args);
        init_fwd_buffers(fwd, tc, rng);

        // ── 2. Alloc + init tangent buffers ──────────────────────
        CallBuffers tan = alloc_call(tc.tan_args);
        init_tan_buffers(tan, fwd, tc, rng);

        // ── 3. Run analytical tangent: dh_ana = tan_func(...) ──────
        zero_buffer(tan.bufs[tc.tan_out_idx]);
        tc.tan_func(tan.args.data());
        auto dh_ana = to_host_float(tan.args[tc.tan_out_idx],
                                     tan_out_desc.mem, N_out, tan_out_desc.dtype);

        // ── Dump (analytical tangent + state + directions) ────────
        if (!tc.dump_dir.empty()) {
            std::string safe = tc.label;
            for (char& c : safe) if (!isalnum(c) && c!='_' && c!='-') c='_';
            mkdir_p("dump/" + tc.dump_dir);
            const std::string ana_name = "dh_analytical";
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_" + ana_name + ".txt",
                        tan.args[tc.tan_out_idx],
                        tan_out_desc.mem, tan_out_desc.dtype, tan_out_desc.dims, ana_name);
            for (auto& s : tc.perturb_specs) {
                const auto& d = tc.tan_args[s.tan_dir_idx];
                const std::string dn = d.name.empty() ? ("dir" + std::to_string(s.tan_dir_idx)) : d.name;
                dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_" + dn + ".txt",
                            tan.args[s.tan_dir_idx],
                            d.mem, d.dtype, d.dims, dn);
            }
        }

        // ── 4. Sample indices ─────────────────────────────────────
        std::vector<size_t> sample_indices;
        if (cfg.num_samples < 0 || cfg.num_samples >= (int)N_out) {
            sample_indices.resize(N_out);
            std::iota(sample_indices.begin(), sample_indices.end(), 0);
            printf("  Mode: full (%zu pixels)\n", N_out);
        } else {
            std::vector<size_t> all(N_out);
            std::iota(all.begin(), all.end(), 0);
            std::shuffle(all.begin(), all.end(), rng);
            sample_indices.assign(all.begin(), all.begin() + cfg.num_samples);
            printf("  Mode: random sampling (%d / %zu pixels)\n",
                   cfg.num_samples, N_out);
        }

        // ── 5. Allocate perturbed forward buffer ──────────────────
        CallBuffers fwd_p = alloc_call(tc.fwd_args);
        CallBuffers fwd_m = alloc_call(tc.fwd_args);
        // Initial copy: same as fwd
        for (size_t i = 0; i < tc.fwd_args.size(); ++i) {
            copy_buffer(fwd_p.bufs[i], fwd.bufs[i]);
            copy_buffer(fwd_m.bufs[i], fwd.bufs[i]);
        }

        const size_t H_out = fwd_out_desc.dims.h;

        // ── 6. Epsilon sweep ──────────────────────────────────────
        TangentResult result;
        result.tol_rel = cfg.tol_rel;
        result.tol_abs = cfg.tol_abs;
        result.num_samples = (int)sample_indices.size();
        result.per_epsilon.reserve(cfg.epsilons.size());

        for (double eps : cfg.epsilons) {
            // Rebuild perturbed buffers: fwd_p[idx] = fwd[idx] + eps * dir
            for (auto& s : tc.perturb_specs) {
                const auto& fd = tc.fwd_args[s.fwd_idx];
                const auto& td = tc.tan_args[s.tan_dir_idx];
                if (fd.dims.total() != td.dims.total() || fd.dtype != td.dtype) {
                    throw std::runtime_error(
                        "PerturbSpec mismatch: fwd_args[" + std::to_string(s.fwd_idx) +
                        "] vs tan_args[" + std::to_string(s.tan_dir_idx) + "]");
                }
                axpy_buffer(fwd_p.bufs[s.fwd_idx], fwd.bufs[s.fwd_idx],
                            tan.bufs[s.tan_dir_idx], +eps, fd.dtype, fd.dims.total());
                axpy_buffer(fwd_m.bufs[s.fwd_idx], fwd.bufs[s.fwd_idx],
                            tan.bufs[s.tan_dir_idx], -eps, fd.dtype, fd.dims.total());
            }
            // Run forward at +eps and -eps
            zero_buffer(fwd_p.bufs[tc.fwd_out_idx]);
            tc.fwd_func(fwd_p.args.data());
            auto h_plus = to_host_float(fwd_p.args[tc.fwd_out_idx],
                                         fwd_out_desc.mem, N_out, fwd_out_desc.dtype);

            zero_buffer(fwd_m.bufs[tc.fwd_out_idx]);
            tc.fwd_func(fwd_m.args.data());
            auto h_minus = to_host_float(fwd_m.args[tc.fwd_out_idx],
                                          fwd_out_desc.mem, N_out, fwd_out_desc.dtype);

            const double inv_2eps = 1.0 / (2.0 * eps);
            TangentEpsilonResult er;
            er.epsilon = eps;
            er.pixels.reserve(sample_indices.size());

            double sum_rel = 0.0, sum_abs = 0.0;
            for (size_t lin : sample_indices) {
                double num = ((double)h_plus[lin] - (double)h_minus[lin]) * inv_2eps;
                double ana = (double)dh_ana[lin];
                double abs_err = std::abs(ana - num);
                double scale = std::max(std::abs(ana), std::abs(num));
                double rel_err = abs_err / (scale + cfg.tol_abs);

                TangentPixelResult pr;
                pr.nx = fwd_out_desc.dims.is_2d() ? (lin / H_out) : lin;
                pr.ny = fwd_out_desc.dims.is_2d() ? (lin % H_out) : 0;
                pr.analytical = ana;
                pr.numerical  = num;
                pr.abs_error  = abs_err;
                pr.rel_error  = rel_err;
                er.pixels.push_back(pr);
                sum_rel += rel_err;
                sum_abs += abs_err;

                if (abs_err > std::max(cfg.tol_abs, cfg.tol_rel * scale))
                    er.num_violations++;
            }

            // aggregate per-epsilon
            int nps = (int)er.pixels.size();
            er.num_samples = nps;
            er.abs_error_mean = sum_abs / std::max(1, nps);
            er.rel_error_mean = sum_rel / std::max(1, nps);
            size_t worst = 0;
            double worst_rel = -1.0;
            for (size_t k = 0; k < er.pixels.size(); ++k) {
                if (er.pixels[k].rel_error > worst_rel) {
                    worst_rel = er.pixels[k].rel_error;
                    worst = k;
                }
                if (er.pixels[k].abs_error > er.abs_error_max)
                    er.abs_error_max = er.pixels[k].abs_error;
                if (er.pixels[k].rel_error > er.rel_error_max)
                    er.rel_error_max = er.pixels[k].rel_error;
            }
            er.worst_nx = er.pixels[worst].nx;
            er.worst_ny = er.pixels[worst].ny;
            er.passed   = (er.num_violations == 0);

            printf("  eps=%.2e  abs_max=%.3e  rel_max=%.3e  viol=%d/%d  %s\n",
                   eps, er.abs_error_max, er.rel_error_max,
                   er.num_violations, nps,
                   er.passed ? "PASS" : "fail");

            result.per_epsilon.push_back(std::move(er));
        }

        // ── 7. Aggregate: pick best epsilon (smallest violations, smallest abs_err) ──
        size_t best = 0;
        bool any_pass = false;
        for (size_t k = 0; k < result.per_epsilon.size(); ++k) {
            const auto& er = result.per_epsilon[k];
            if (er.passed) {
                if (!any_pass || er.abs_error_max < result.per_epsilon[best].abs_error_max) {
                    best = k; any_pass = true;
                }
            } else if (!any_pass) {
                if (er.num_violations < result.per_epsilon[best].num_violations ||
                    (er.num_violations == result.per_epsilon[best].num_violations &&
                     er.abs_error_max < result.per_epsilon[best].abs_error_max)) {
                    best = k;
                }
            }
        }
        result.passed             = any_pass;
        result.best_epsilon       = result.per_epsilon[best].epsilon;
        result.best_abs_error_max = result.per_epsilon[best].abs_error_max;
        result.best_rel_error_max = result.per_epsilon[best].rel_error_max;
        result.best_worst_nx      = result.per_epsilon[best].worst_nx;
        result.best_worst_ny      = result.per_epsilon[best].worst_ny;
        result.best_num_violations = result.per_epsilon[best].num_violations;

        // ── Dump (numerical + difference @ best epsilon) ──────────
        if (!tc.dump_dir.empty()) {
            std::string safe = tc.label;
            for (char& c : safe) if (!isalnum(c) && c!='_' && c!='-') c='_';
            // Regenerate numerical at best epsilon and dump as image + pixel table.
            double eps = result.best_epsilon;
            for (auto& s : tc.perturb_specs) {
                const auto& fd = tc.fwd_args[s.fwd_idx];
                axpy_buffer(fwd_p.bufs[s.fwd_idx], fwd.bufs[s.fwd_idx],
                            tan.bufs[s.tan_dir_idx], +eps, fd.dtype, fd.dims.total());
                axpy_buffer(fwd_m.bufs[s.fwd_idx], fwd.bufs[s.fwd_idx],
                            tan.bufs[s.tan_dir_idx], -eps, fd.dtype, fd.dims.total());
            }
            zero_buffer(fwd_p.bufs[tc.fwd_out_idx]);
            tc.fwd_func(fwd_p.args.data());
            zero_buffer(fwd_m.bufs[tc.fwd_out_idx]);
            tc.fwd_func(fwd_m.args.data());
            auto h_plus  = to_host_float(fwd_p.args[tc.fwd_out_idx], fwd_out_desc.mem, N_out, fwd_out_desc.dtype);
            auto h_minus = to_host_float(fwd_m.args[tc.fwd_out_idx], fwd_out_desc.mem, N_out, fwd_out_desc.dtype);
            std::vector<float> dh_num(N_out), dh_diff(N_out);
            for (size_t i = 0; i < N_out; ++i) {
                dh_num[i] = (float)(((double)h_plus[i] - (double)h_minus[i]) / (2.0 * eps));
                dh_diff[i] = (float)((double)dh_ana[i] - (double)dh_num[i]);
            }
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_dh_numerical.txt",
                        (void*)dh_num.data(), MemSpace::HOST, DType::FLOAT32,
                        tan_out_desc.dims, "dh_numerical");
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_dh_diff.txt",
                        (void*)dh_diff.data(), MemSpace::HOST, DType::FLOAT32,
                        tan_out_desc.dims, "dh_diff");
            // Per-pixel CSV at best epsilon (uses adjoint-pixels compatible schema).
            dump_adjoint_pixels("dump/" + tc.dump_dir + "/" + safe + "_pixels.txt",
                                result.per_epsilon[best], tc.label);
        }

        // ── 8. Perf (latency + workspace) ─────────────────────────
        if (tc.perf.measure_latency) {
            result.has_perf = true;
            result.fwd_latency = measure_latency([&]{
                zero_buffer(fwd.bufs[tc.fwd_out_idx]);
                tc.fwd_func(fwd.args.data());
            }, tc.perf.warmup_runs, tc.perf.bench_runs);
            result.tan_latency = measure_latency([&]{
                zero_buffer(tan.bufs[tc.tan_out_idx]);
                tc.tan_func(tan.args.data());
            }, tc.perf.warmup_runs, tc.perf.bench_runs);
        }
        if (tc.perf.measure_memory) {
            result.has_perf = true;
            MemoryTracker tracker(tc.perf.poll_interval_ms);
            result.fwd_memory = tracker.measure([&]{
                zero_buffer(fwd.bufs[tc.fwd_out_idx]);
                tc.fwd_func(fwd.args.data());
            });
            result.tan_memory = tracker.measure([&]{
                zero_buffer(tan.bufs[tc.tan_out_idx]);
                tc.tan_func(tan.args.data());
            });
        }

        print_result(result, tc);
        return result;
    }

    void print_result(const TangentResult& r, const TangentCase& tc) {
        std::cout << "═════════════════════════════════════════════\n";
        std::cout << "  Tangent Test : " << tc.label << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Samples         : " << r.num_samples << "\n";
        std::cout << "  Epsilon range   : " << std::scientific
                  << (tc.cfg.epsilons.empty() ? 0.0 : tc.cfg.epsilons.front()) << " ~ "
                  << (tc.cfg.epsilons.empty() ? 0.0 : tc.cfg.epsilons.back()) << "\n";
        std::cout << "  Best epsilon    : " << std::scientific << r.best_epsilon << "\n";
        std::cout << "  Best abs_max    : " << std::scientific << r.best_abs_error_max << "\n";
        std::cout << "  Best rel_max    : " << std::scientific << r.best_rel_error_max
                  << "  @ (" << r.best_worst_nx << ", " << r.best_worst_ny << ")\n";
        std::cout << "  Best violations : " << r.best_num_violations << " / " << r.num_samples << "\n";
        if (tc.cfg.tol_rel >= 0.0)
            std::cout << "  Result          : "
                      << (r.passed ? "PASS v" : "FAIL x")
                      << "  (tol_rel=" << std::scientific << tc.cfg.tol_rel
                      << ", tol_abs=" << tc.cfg.tol_abs << ")\n";

        if (r.has_perf && tc.perf.measure_latency) {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  ── Latency (warmup=" << r.fwd_latency.warmup
                      << ", runs=" << r.fwd_latency.runs << ") ──\n";
            std::cout << "  fwd  avg=" << r.fwd_latency.avg_ms << "ms"
                      << "  min=" << r.fwd_latency.min_ms << "ms"
                      << "  max=" << r.fwd_latency.max_ms << "ms\n";
            std::cout << "  tan  avg=" << r.tan_latency.avg_ms << "ms"
                      << "  min=" << r.tan_latency.min_ms << "ms"
                      << "  max=" << r.tan_latency.max_ms << "ms\n";
        }
        if (r.has_perf && tc.perf.measure_memory) {
            std::cout << "  ── Workspace Memory (poll=" << r.fwd_memory.poll_interval_ms << "ms) ──\n";
            std::cout << "  fwd  peak workspace : " << format_bytes(r.fwd_memory.workspace_bytes) << "\n";
            std::cout << "  tan  peak workspace : " << format_bytes(r.tan_memory.workspace_bytes) << "\n";
        }
        std::cout << "═════════════════════════════════════════════\n";
    }

    static void mkdir_p(const std::string& path) {
        for (size_t i = 1; i <= path.size(); ++i) {
            if (i == path.size() || path[i] == '/') {
                std::string sub = path.substr(0, i);
                mkdir(sub.c_str(), 0755);
            }
        }
    }

    std::vector<TangentCase> cases_;
};

} } // namespace st::util
