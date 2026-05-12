#pragma once
#include "arg_descriptor.h"
#include "buffer.h"
#include "adapter.h"
#include "comparator.h"
#include "func_tester.h"
#include "dump.h"
#include "tangent_tester.h"   // for PerturbSpec, default_epsilon_sweep, TangentPixelResult shape
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
//  TangentAdjointConfig
//
//  Validation criterion (per PRD §6.1):
//    pass if ∃ ε in `epsilons` such that ∀ (α,β,j),
//      |H1_αβ(d_j)(ε) - H2_αβ(d_j)| ≤ max(tol_abs, tol_rel·max(|H1|,|H2|))
//
//  where for one direction set d = (df, dg, dλ_h):
//    H1(d)(ε) = (adj(state + ε·d) - adj(state - ε·d)) / (2ε)   numerical
//    H2(d)    = tan_adj(state, d)                              analytical
// ─────────────────────────────────────────────────────────────

struct TangentAdjointConfig {
    int                  num_samples = -1;
    std::vector<double>  epsilons    = default_epsilon_sweep();
    double               tol_abs     = 1e-5;
    double               tol_rel     = 1e-3;
    uint64_t             seed        = 0;
};

// Reuse PerturbSpec from tangent_tester.h.
// Here fwd_idx refers to a slot in adj_args (the upstream operator);
// tan_dir_idx refers to a slot in tan_adj_args holding the corresponding
// direction (df / dg / dλ_h).
//
// Naming kept generic for symmetry; we alias for readability.
using TangentAdjPerturbSpec = PerturbSpec;

struct TanAdjPixelResult {
    size_t nx = 0, ny = 0;
    double analytical = 0.0;
    double numerical  = 0.0;
    double rel_error  = 0.0;
    double abs_error  = 0.0;
};

struct TanAdjEpsilonResult {
    double                         epsilon = 0.0;
    std::vector<TanAdjPixelResult> pixels;
    int                            num_samples = 0;   ///< == pixels.size() (for dump_adjoint_pixels)
    double                         rel_error_max  = 0.0;
    double                         abs_error_max  = 0.0;
    double                         rel_error_mean = 0.0;
    double                         abs_error_mean = 0.0;
    size_t                         worst_nx = 0, worst_ny = 0;
    bool                           passed   = false;
    int                            num_violations = 0;
};

struct TangentAdjointResult {
    std::vector<TanAdjEpsilonResult> per_epsilon;
    bool   passed       = false;
    double best_epsilon = 0.0;
    double best_rel_error_max = 0.0;
    double best_abs_error_max = 0.0;
    size_t best_worst_nx = 0, best_worst_ny = 0;
    int    best_num_violations = 0;
    int    num_samples = 0;
    double tol_rel = -1.0;
    double tol_abs = 0.0;
    bool          has_perf = false;
    LatencyResult adj_latency;
    LatencyResult tan_adj_latency;
    MemoryResult  adj_memory;
    MemoryResult  tan_adj_memory;
};

// ─────────────────────────────────────────────────────────────
//  TangentAdjointCase
//
//  Tests that the analytical tangent-adjoint kernel agrees with
//  the central-difference of the adjoint kernel along a chosen
//  set of perturbation directions.
//
//  adj_func    : (f, g, λ_h, ...) → λ_f or λ_g   (a single adjoint output)
//  tan_adj_func: (f, g, λ_h, df, dg, dλ_h, ...) → dλ_f or dλ_g
//
//  - adj_out_idx     : which adj_args slot holds λ_? (the output being differentiated)
//  - tan_adj_out_idx : which tan_adj_args slot holds dλ_? (the analytical answer)
//  - perturb_specs   : (adj_idx, tan_adj_dir_idx) pairs — perturb adj inputs in directions
//                      stored at tan_adj_args[tan_adj_dir_idx].
// ─────────────────────────────────────────────────────────────

struct TangentAdjointCase {
    std::string                label;
    AdaptedFunc                adj_func;
    std::vector<ArgDescriptor> adj_args;
    size_t                     adj_out_idx;
    AdaptedFunc                tan_adj_func;
    std::vector<ArgDescriptor> tan_adj_args;
    size_t                     tan_adj_out_idx;
    std::vector<PerturbSpec>   perturb_specs;
    TangentAdjointConfig       cfg;
    PerfConfig                 perf;
    GpConfig                   gp;
    BufferInitFn               adj_init_fn;
    BufferInitFn               tan_adj_init_fn;
    std::string                dump_dir;
};

// ─────────────────────────────────────────────────────────────
//  TangentAdjointTester
// ─────────────────────────────────────────────────────────────

class TangentAdjointTester {
public:

    void register_tangent_adjoint(
        std::string                label,
        AdaptedFunc                adj_func,
        std::vector<ArgDescriptor> adj_args,
        size_t                     adj_out_idx,
        AdaptedFunc                tan_adj_func,
        std::vector<ArgDescriptor> tan_adj_args,
        size_t                     tan_adj_out_idx,
        std::vector<PerturbSpec>   perturb_specs,
        TangentAdjointConfig       cfg              = {},
        PerfConfig                 perf             = {},
        BufferInitFn               adj_init_fn      = nullptr,
        BufferInitFn               tan_adj_init_fn  = nullptr,
        GpConfig                   gp               = {},
        std::string                dump_dir         = "")
    {
        if (!adj_init_fn && !gp.enabled) gp.enabled = true;
        TangentAdjointCase tc;
        tc.label            = std::move(label);
        tc.adj_func         = adj_func;
        tc.adj_args         = std::move(adj_args);
        tc.adj_out_idx      = adj_out_idx;
        tc.tan_adj_func     = tan_adj_func;
        tc.tan_adj_args     = std::move(tan_adj_args);
        tc.tan_adj_out_idx  = tan_adj_out_idx;
        tc.perturb_specs    = std::move(perturb_specs);
        tc.cfg              = cfg;
        tc.perf             = perf;
        tc.gp               = gp;
        tc.adj_init_fn      = std::move(adj_init_fn);
        tc.tan_adj_init_fn  = std::move(tan_adj_init_fn);
        tc.dump_dir         = std::move(dump_dir);
        cases_.push_back(std::move(tc));
    }

    std::vector<TangentAdjointResult> run_all() {
        std::vector<TangentAdjointResult> results;
        results.reserve(cases_.size());
        for (auto& c : cases_) results.push_back(run_one(c));
        return results;
    }

    TangentAdjointResult run(size_t idx) {
        if (idx >= cases_.size())
            throw std::out_of_range("TangentAdjointCase index out of range");
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
        if (dst.mem == MemSpace::HOST) memcpy(dst.ptr, hdst.data(), bytes);
        else {
#ifdef API_TESTER_CUDA
            cudaMemcpy(dst.ptr, hdst.data(), bytes, cudaMemcpyHostToDevice);
#endif
        }
    }

    void init_adj_buffers(CallBuffers& adj, const TangentAdjointCase& tc,
                          std::mt19937_64& rng) {
        std::normal_distribution<float> normal(0.f, 1.f);
        auto is_pow2 = [](size_t x){ return x > 0 && (x & (x - 1)) == 0; };

        for (size_t i = 0; i < tc.adj_args.size(); ++i) {
            if (tc.adj_args[i].role == ParamRole::OUTPUT) continue;
            ArgDescriptor hd = tc.adj_args[i];
            hd.mem = MemSpace::HOST;
            Buffer htmp = alloc_buffer(hd);
            if (tc.adj_init_fn) {
                tc.adj_init_fn(htmp.ptr, hd, i);
            } else if (tc.adj_args[i].role == ParamRole::INPUT) {
                size_t n = hd.dims.total();
                int W = (int)hd.dims.w, H = (int)hd.dims.h;
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
            copy_buffer(adj.bufs[i], htmp);
        }
    }

    void init_tan_adj_buffers(CallBuffers& tan_adj, CallBuffers& adj,
                              const TangentAdjointCase& tc, std::mt19937_64& rng) {
        std::normal_distribution<float> normal(0.f, 1.f);

        if (tc.tan_adj_init_fn) {
            for (size_t i = 0; i < tc.tan_adj_args.size(); ++i) {
                if (i == tc.tan_adj_out_idx) continue;
                ArgDescriptor hd = tc.tan_adj_args[i];
                hd.mem = MemSpace::HOST;
                Buffer htmp = alloc_buffer(hd);
                tc.tan_adj_init_fn(htmp.ptr, tc.tan_adj_args[i], i);
                copy_buffer(tan_adj.bufs[i], htmp);
            }
            return;
        }
        // Auto-copy: copy by name match; fill direction buffers randomly.
        std::vector<bool> is_dir(tc.tan_adj_args.size(), false);
        for (auto& s : tc.perturb_specs) is_dir[s.tan_dir_idx] = true;

        for (size_t i = 0; i < tc.tan_adj_args.size(); ++i) {
            if (i == tc.tan_adj_out_idx) continue;
            if (is_dir[i]) {
                ArgDescriptor hd = tc.tan_adj_args[i];
                hd.mem = MemSpace::HOST;
                Buffer htmp = alloc_buffer(hd);
                size_t n = hd.dims.total();
                for (size_t j = 0; j < n; ++j)
                    set_element_host(htmp.ptr, j, normal(rng), hd.dtype);
                copy_buffer(tan_adj.bufs[i], htmp);
                continue;
            }
            const std::string& tname = tc.tan_adj_args[i].name;
            bool matched = false;
            if (!tname.empty()) {
                for (size_t fi = 0; fi < tc.adj_args.size(); ++fi) {
                    if (fi == tc.adj_out_idx) continue;
                    if (tc.adj_args[fi].name == tname) {
                        copy_buffer(tan_adj.bufs[i], adj.bufs[fi]);
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) zero_buffer(tan_adj.bufs[i]);
        }
    }

    TangentAdjointResult run_one(TangentAdjointCase& tc) {
        const auto& cfg          = tc.cfg;
        const auto& tan_out_desc = tc.tan_adj_args[tc.tan_adj_out_idx];
        const auto& adj_out_desc = tc.adj_args[tc.adj_out_idx];
        const size_t N_out = adj_out_desc.dims.total();
        if (tan_out_desc.dims.total() != N_out)
            throw std::runtime_error(
                "TangentAdjointCase '" + tc.label +
                "': adj_out and tan_adj_out dims differ.");

        printf("\n[TangentAdjointTester] %s\n", tc.label.c_str());

        std::mt19937_64 rng(cfg.seed == 0 ? std::random_device{}() : cfg.seed);

        // 1. Alloc + init adj buffers
        CallBuffers adj = alloc_call(tc.adj_args);
        init_adj_buffers(adj, tc, rng);

        // 2. Alloc + init tan_adj buffers
        CallBuffers tan_adj = alloc_call(tc.tan_adj_args);
        init_tan_adj_buffers(tan_adj, adj, tc, rng);

        // 3. Analytical tangent-adjoint: dλ_ana
        zero_buffer(tan_adj.bufs[tc.tan_adj_out_idx]);
        tc.tan_adj_func(tan_adj.args.data());
        auto dlam_ana = to_host_float(tan_adj.args[tc.tan_adj_out_idx],
                                       tan_out_desc.mem, N_out, tan_out_desc.dtype);

        if (!tc.dump_dir.empty()) {
            std::string safe = tc.label;
            for (char& c : safe) if (!isalnum(c) && c!='_' && c!='-') c='_';
            mkdir_p("dump/" + tc.dump_dir);
            const std::string ana_name = "dlambda_analytical";
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_" + ana_name + ".txt",
                        tan_adj.args[tc.tan_adj_out_idx],
                        tan_out_desc.mem, tan_out_desc.dtype, tan_out_desc.dims, ana_name);
            for (auto& s : tc.perturb_specs) {
                const auto& d = tc.tan_adj_args[s.tan_dir_idx];
                const std::string dn = d.name.empty() ? ("dir" + std::to_string(s.tan_dir_idx)) : d.name;
                dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_" + dn + ".txt",
                            tan_adj.args[s.tan_dir_idx],
                            d.mem, d.dtype, d.dims, dn);
            }
        }

        // 4. Sample indices
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

        // 5. Perturbed adjoint buffers
        CallBuffers adj_p = alloc_call(tc.adj_args);
        CallBuffers adj_m = alloc_call(tc.adj_args);
        for (size_t i = 0; i < tc.adj_args.size(); ++i) {
            copy_buffer(adj_p.bufs[i], adj.bufs[i]);
            copy_buffer(adj_m.bufs[i], adj.bufs[i]);
        }

        const size_t H_out = adj_out_desc.dims.h;

        // 6. Epsilon sweep
        TangentAdjointResult result;
        result.tol_rel = cfg.tol_rel;
        result.tol_abs = cfg.tol_abs;
        result.num_samples = (int)sample_indices.size();
        result.per_epsilon.reserve(cfg.epsilons.size());

        for (double eps : cfg.epsilons) {
            for (auto& s : tc.perturb_specs) {
                const auto& fd = tc.adj_args[s.fwd_idx];
                const auto& td = tc.tan_adj_args[s.tan_dir_idx];
                if (fd.dims.total() != td.dims.total() || fd.dtype != td.dtype) {
                    throw std::runtime_error(
                        "PerturbSpec mismatch: adj_args[" + std::to_string(s.fwd_idx) +
                        "] vs tan_adj_args[" + std::to_string(s.tan_dir_idx) + "]");
                }
                axpy_buffer(adj_p.bufs[s.fwd_idx], adj.bufs[s.fwd_idx],
                            tan_adj.bufs[s.tan_dir_idx], +eps, fd.dtype, fd.dims.total());
                axpy_buffer(adj_m.bufs[s.fwd_idx], adj.bufs[s.fwd_idx],
                            tan_adj.bufs[s.tan_dir_idx], -eps, fd.dtype, fd.dims.total());
            }
            zero_buffer(adj_p.bufs[tc.adj_out_idx]);
            tc.adj_func(adj_p.args.data());
            auto lam_plus = to_host_float(adj_p.args[tc.adj_out_idx],
                                           adj_out_desc.mem, N_out, adj_out_desc.dtype);
            zero_buffer(adj_m.bufs[tc.adj_out_idx]);
            tc.adj_func(adj_m.args.data());
            auto lam_minus = to_host_float(adj_m.args[tc.adj_out_idx],
                                            adj_out_desc.mem, N_out, adj_out_desc.dtype);

            const double inv_2eps = 1.0 / (2.0 * eps);
            TanAdjEpsilonResult er;
            er.epsilon = eps;
            er.pixels.reserve(sample_indices.size());

            double sum_rel = 0.0, sum_abs = 0.0;
            for (size_t lin : sample_indices) {
                double num = ((double)lam_plus[lin] - (double)lam_minus[lin]) * inv_2eps;
                double ana = (double)dlam_ana[lin];
                double abs_err = std::abs(ana - num);
                double scale = std::max(std::abs(ana), std::abs(num));
                double rel_err = abs_err / (scale + cfg.tol_abs);

                TanAdjPixelResult pr;
                pr.nx = adj_out_desc.dims.is_2d() ? (lin / H_out) : lin;
                pr.ny = adj_out_desc.dims.is_2d() ? (lin % H_out) : 0;
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

        // 7. Aggregate: pick best epsilon
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
        result.passed              = any_pass;
        result.best_epsilon        = result.per_epsilon[best].epsilon;
        result.best_abs_error_max  = result.per_epsilon[best].abs_error_max;
        result.best_rel_error_max  = result.per_epsilon[best].rel_error_max;
        result.best_worst_nx       = result.per_epsilon[best].worst_nx;
        result.best_worst_ny       = result.per_epsilon[best].worst_ny;
        result.best_num_violations = result.per_epsilon[best].num_violations;

        // 8. Dump numerical + diff at best epsilon
        if (!tc.dump_dir.empty()) {
            std::string safe = tc.label;
            for (char& c : safe) if (!isalnum(c) && c!='_' && c!='-') c='_';
            double eps = result.best_epsilon;
            for (auto& s : tc.perturb_specs) {
                const auto& fd = tc.adj_args[s.fwd_idx];
                axpy_buffer(adj_p.bufs[s.fwd_idx], adj.bufs[s.fwd_idx],
                            tan_adj.bufs[s.tan_dir_idx], +eps, fd.dtype, fd.dims.total());
                axpy_buffer(adj_m.bufs[s.fwd_idx], adj.bufs[s.fwd_idx],
                            tan_adj.bufs[s.tan_dir_idx], -eps, fd.dtype, fd.dims.total());
            }
            zero_buffer(adj_p.bufs[tc.adj_out_idx]); tc.adj_func(adj_p.args.data());
            zero_buffer(adj_m.bufs[tc.adj_out_idx]); tc.adj_func(adj_m.args.data());
            auto lp = to_host_float(adj_p.args[tc.adj_out_idx], adj_out_desc.mem, N_out, adj_out_desc.dtype);
            auto lm = to_host_float(adj_m.args[tc.adj_out_idx], adj_out_desc.mem, N_out, adj_out_desc.dtype);
            std::vector<float> dlam_num(N_out), dlam_diff(N_out);
            for (size_t i = 0; i < N_out; ++i) {
                dlam_num[i]  = (float)(((double)lp[i] - (double)lm[i]) / (2.0 * eps));
                dlam_diff[i] = (float)((double)dlam_ana[i] - (double)dlam_num[i]);
            }
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_dlambda_numerical.txt",
                        (void*)dlam_num.data(), MemSpace::HOST, DType::FLOAT32,
                        tan_out_desc.dims, "dlambda_numerical");
            dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_dlambda_diff.txt",
                        (void*)dlam_diff.data(), MemSpace::HOST, DType::FLOAT32,
                        tan_out_desc.dims, "dlambda_diff");
            dump_adjoint_pixels("dump/" + tc.dump_dir + "/" + safe + "_pixels.txt",
                                result.per_epsilon[best], tc.label);
        }

        // 9. Perf
        if (tc.perf.measure_latency) {
            result.has_perf = true;
            result.adj_latency = measure_latency([&]{
                zero_buffer(adj.bufs[tc.adj_out_idx]);
                tc.adj_func(adj.args.data());
            }, tc.perf.warmup_runs, tc.perf.bench_runs);
            result.tan_adj_latency = measure_latency([&]{
                zero_buffer(tan_adj.bufs[tc.tan_adj_out_idx]);
                tc.tan_adj_func(tan_adj.args.data());
            }, tc.perf.warmup_runs, tc.perf.bench_runs);
        }
        if (tc.perf.measure_memory) {
            result.has_perf = true;
            MemoryTracker tracker(tc.perf.poll_interval_ms);
            result.adj_memory = tracker.measure([&]{
                zero_buffer(adj.bufs[tc.adj_out_idx]);
                tc.adj_func(adj.args.data());
            });
            result.tan_adj_memory = tracker.measure([&]{
                zero_buffer(tan_adj.bufs[tc.tan_adj_out_idx]);
                tc.tan_adj_func(tan_adj.args.data());
            });
        }

        print_result(result, tc);
        return result;
    }

    void print_result(const TangentAdjointResult& r, const TangentAdjointCase& tc) {
        std::cout << "═════════════════════════════════════════════\n";
        std::cout << "  Tangent-Adjoint Test : " << tc.label << "\n";
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
            std::cout << "  ── Latency (warmup=" << r.adj_latency.warmup
                      << ", runs=" << r.adj_latency.runs << ") ──\n";
            std::cout << "  adj      avg=" << r.adj_latency.avg_ms << "ms"
                      << "  min=" << r.adj_latency.min_ms << "ms"
                      << "  max=" << r.adj_latency.max_ms << "ms\n";
            std::cout << "  tan_adj  avg=" << r.tan_adj_latency.avg_ms << "ms"
                      << "  min=" << r.tan_adj_latency.min_ms << "ms"
                      << "  max=" << r.tan_adj_latency.max_ms << "ms\n";
        }
        if (r.has_perf && tc.perf.measure_memory) {
            std::cout << "  ── Workspace Memory (poll=" << r.adj_memory.poll_interval_ms << "ms) ──\n";
            std::cout << "  adj      peak workspace : " << format_bytes(r.adj_memory.workspace_bytes) << "\n";
            std::cout << "  tan_adj  peak workspace : " << format_bytes(r.tan_adj_memory.workspace_bytes) << "\n";
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

    std::vector<TangentAdjointCase> cases_;
};

} } // namespace st::util
