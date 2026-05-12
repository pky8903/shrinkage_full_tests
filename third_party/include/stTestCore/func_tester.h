#pragma once
#include "arg_descriptor.h"
#include "buffer.h"
#include "comparator.h"
#include "adapter.h"
#include "latency.h"
#include "memory_tracker.h"
#include "gp_generator.h"
#include "dump.h"
#include <vector>
#include <string>
#include <functional>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

namespace st { namespace util {

// ─────────────────────────────────────────────────────────
//  BufferInitFn
//  Signature: (void* ptr, const ArgDescriptor& desc, size_t arg_idx)
// ─────────────────────────────────────────────────────────

using BufferInitFn = std::function<void(void*, const ArgDescriptor&, size_t)>;

inline BufferInitFn init_random(float lo = 0.0f, float hi = 1.0f) {
    return [lo, hi](void* ptr, const ArgDescriptor& desc, size_t) {
        const size_t N   = desc.dims.total();
        const size_t esz = dtype_size(desc.dtype);
        auto* b = static_cast<uint8_t*>(ptr);
        for (size_t i = 0; i < N; ++i) {
            float v = lo + (hi - lo) *
                      (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            switch (desc.dtype) {
                case DType::FLOAT16: { uint16_t x=f32_to_f16(v);      memcpy(b+i*esz,&x,esz); break; }
                case DType::FLOAT32: { float    x=v;                memcpy(b+i*esz,&x,esz); break; }
                case DType::FLOAT64: { double   x=(double)v;        memcpy(b+i*esz,&x,esz); break; }
                case DType::INT32:   { int32_t  x=(int32_t)(v*1000);  memcpy(b+i*esz,&x,esz); break; }
                case DType::INT16:   { int16_t  x=(int16_t)(v*1000);  memcpy(b+i*esz,&x,esz); break; }
                case DType::INT8:    { int8_t   x=(int8_t)(v*100);    memcpy(b+i*esz,&x,esz); break; }
                case DType::UINT32:  { uint32_t x=(uint32_t)(v*1000); memcpy(b+i*esz,&x,esz); break; }
                case DType::UINT16:  { uint16_t x=(uint16_t)(v*1000); memcpy(b+i*esz,&x,esz); break; }
                case DType::UINT8:   { uint8_t  x=(uint8_t)(v*255);   memcpy(b+i*esz,&x,esz); break; }
            }
        }
    };
}

inline BufferInitFn init_constant(float value) {
    return [value](void* ptr, const ArgDescriptor& desc, size_t) {
        const size_t N   = desc.dims.total();
        const size_t esz = dtype_size(desc.dtype);
        auto* b = static_cast<uint8_t*>(ptr);
        for (size_t i = 0; i < N; ++i) {
            switch (desc.dtype) {
                case DType::FLOAT16: { uint16_t x=f32_to_f16(value); memcpy(b+i*esz,&x,esz); break; }
                case DType::FLOAT32: { float    x=value;           memcpy(b+i*esz,&x,esz); break; }
                case DType::FLOAT64: { double   x=(double)value;   memcpy(b+i*esz,&x,esz); break; }
                case DType::INT32:   { int32_t  x=(int32_t)value;  memcpy(b+i*esz,&x,esz); break; }
                case DType::INT16:   { int16_t  x=(int16_t)value;  memcpy(b+i*esz,&x,esz); break; }
                case DType::INT8:    { int8_t   x=(int8_t)value;   memcpy(b+i*esz,&x,esz); break; }
                case DType::UINT32:  { uint32_t x=(uint32_t)value; memcpy(b+i*esz,&x,esz); break; }
                case DType::UINT16:  { uint16_t x=(uint16_t)value; memcpy(b+i*esz,&x,esz); break; }
                case DType::UINT8:   { uint8_t  x=(uint8_t)value;  memcpy(b+i*esz,&x,esz); break; }
            }
        }
    };
}

// ─────────────────────────────────────────────────────────
//  PerfConfig
// ─────────────────────────────────────────────────────────

struct PerfConfig {
    bool measure_latency  = true;
    int  warmup_runs      = 3;
    int  bench_runs       = 10;
    bool measure_memory   = true;
    int  poll_interval_ms = 1;
};

// ─────────────────────────────────────────────────────────
//  GpConfig
// ─────────────────────────────────────────────────────────

struct GpConfig {
    bool         enabled    = false;
    Covariance2D cov        = Covariance2D::isotropic(8.0);  ///< ell=8 grid points
    int          iterations = 10;
    uint64_t     seed       = 0;
    size_t       input_arg  = 0;   ///< which arg index receives GP samples
};

// ─────────────────────────────────────────────────────────
//  IterResult / AggregateResult / FullResult
// ─────────────────────────────────────────────────────────

struct IterResult {
    int           iteration = 0;
    CompareResult accuracy;
};

struct AggregateResult {
    int    n = 0;
    double l2_mean  = 0.0, l2_max  = 0.0, l2_stdev = 0.0;
    double max_err_mean  = 0.0, max_err_max  = 0.0, max_err_stdev = 0.0;
    double stdev_mean    = 0.0, stdev_max    = 0.0, stdev_stdev   = 0.0;
    PixelPos max_err_worst_pos;
    int    pass_count = 0;
    double pass_rate  = 0.0;
};

struct FullResult {
    CompareResult           accuracy;         // single-run mode
    std::vector<IterResult> iter_results;     // GP mode
    AggregateResult         aggregate;        // GP mode
    bool          has_perf     = false;
    LatencyResult ref_latency;
    LatencyResult test_latency;
    MemoryResult  ref_memory;
    MemoryResult  test_memory;
};

// ─────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────

inline std::string format_bytes(size_t bytes) {
    char buf[64];
    if      (bytes >= 1024ULL*1024*1024) snprintf(buf,sizeof(buf),"%.2f GB",bytes/(1024.0*1024*1024));
    else if (bytes >= 1024*1024)         snprintf(buf,sizeof(buf),"%.2f MB",bytes/(1024.0*1024));
    else if (bytes >= 1024)              snprintf(buf,sizeof(buf),"%.2f KB",bytes/1024.0);
    else                                 snprintf(buf,sizeof(buf),"%zu B",bytes);
    return buf;
}

inline AggregateResult aggregate_iters(
    const std::vector<IterResult>& iters, double tolerance)
{
    const int n = static_cast<int>(iters.size());
    AggregateResult r; r.n = n;
    if (n == 0) return r;

    std::vector<double> l2s(n), maxs(n), stds(n);
    for (int i = 0; i < n; ++i) {
        l2s[i]  = iters[i].accuracy.l2_error;
        maxs[i] = iters[i].accuracy.max_error;
        stds[i] = iters[i].accuracy.stdev;
    }

    auto stats = [&](const std::vector<double>& v,
                     double& mean, double& mx, double& sd) {
        mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
        mx   = *std::max_element(v.begin(), v.end());
        double var = 0.0;
        for (double x : v) var += (x - mean) * (x - mean);
        sd = std::sqrt(var / n);
    };

    stats(l2s,  r.l2_mean,      r.l2_max,      r.l2_stdev);
    stats(maxs, r.max_err_mean, r.max_err_max,  r.max_err_stdev);
    stats(stds, r.stdev_mean,   r.stdev_max,    r.stdev_stdev);

    int worst_idx = static_cast<int>(
        std::max_element(maxs.begin(), maxs.end()) - maxs.begin());
    r.max_err_worst_pos = iters[worst_idx].accuracy.max_pos;

    if (tolerance >= 0.0) {
        for (auto& it : iters)
            if (it.accuracy.passed) r.pass_count++;
        r.pass_rate = static_cast<double>(r.pass_count) / n;
    }

    return r;
}

// ─────────────────────────────────────────────────────────
//  TestCase
// ─────────────────────────────────────────────────────────

struct TestCase {
    std::string                label;
    AdaptedFunc                ref_func;
    AdaptedFunc                test_func;
    std::vector<ArgDescriptor> args;
    double                     tolerance = -1.0;
    BufferInitFn               init_fn;
    PerfConfig                 perf;
    GpConfig                   gp;
    std::string                dump_dir;  ///< if non-empty, dump input/output buffers here
};

// ─────────────────────────────────────────────────────────
//  FuncTester
// ─────────────────────────────────────────────────────────

class FuncTester {
public:

    void register_pair(
        std::string                label,
        AdaptedFunc                ref_func,
        AdaptedFunc                test_func,
        std::vector<ArgDescriptor> args,
        double                     tolerance = -1.0,
        BufferInitFn               init_fn   = nullptr,
        PerfConfig                 perf      = {},
        GpConfig                   gp        = {},
        std::string                dump_dir  = "")
    {
        if (!init_fn) init_fn = init_random(0.0f, 1.0f);
        cases_.push_back({std::move(label), ref_func, test_func,
                          std::move(args), tolerance,
                          std::move(init_fn), perf, gp, std::move(dump_dir)});
    }

    std::vector<FullResult> run_all() {
        std::vector<FullResult> results;
        results.reserve(cases_.size());
        for (auto& tc : cases_)
            results.push_back(run_one(tc));
        return results;
    }

    FullResult run(size_t idx) {
        if (idx >= cases_.size())
            throw std::out_of_range("TestCase index out of range");
        return run_one(cases_[idx]);
    }

private:

    struct Buffers {
        std::vector<Buffer> shared;
        std::vector<Buffer> ref_out;
        std::vector<Buffer> test_out;
        std::vector<void*>  ref_args;
        std::vector<void*>  test_args;
    };

    Buffers alloc_and_init(const TestCase& tc) {
        const size_t N = tc.args.size();
        Buffers b;
        b.shared.resize(N); b.ref_out.resize(N); b.test_out.resize(N);
        b.ref_args.resize(N, nullptr); b.test_args.resize(N, nullptr);

        for (size_t i = 0; i < N; ++i) {
            const auto& desc = tc.args[i];
            if (desc.role == ParamRole::OUTPUT) {
                b.ref_out[i]   = alloc_buffer(desc);
                b.test_out[i]  = alloc_buffer(desc);
                b.ref_args[i]  = b.ref_out[i].ptr;
                b.test_args[i] = b.test_out[i].ptr;
            } else {
                b.shared[i] = alloc_buffer(desc);
                if (tc.init_fn) {
                    if (desc.mem == MemSpace::HOST) {
                        tc.init_fn(b.shared[i].ptr, desc, i);
                    } else {
#ifdef API_TESTER_CUDA
                        ArgDescriptor hd = desc; hd.mem = MemSpace::HOST;
                        Buffer tmp = alloc_buffer(hd);
                        tc.init_fn(tmp.ptr, hd, i);
                        cudaMemcpy(b.shared[i].ptr, tmp.ptr,
                            desc.dims.total()*dtype_size(desc.dtype),
                            cudaMemcpyHostToDevice);
#endif
                    }
                }
                b.ref_args[i]  = b.shared[i].ptr;
                b.test_args[i] = b.shared[i].ptr;
            }
        }
        return b;
    }

    void reset_outputs(Buffers& b, const TestCase& tc) {
        for (size_t i = 0; i < tc.args.size(); ++i) {
            if (tc.args[i].role != ParamRole::OUTPUT) continue;
            auto reset = [](Buffer& buf) {
                if (!buf.ptr) return;
                if (buf.mem == MemSpace::HOST) memset(buf.ptr, 0, buf.bytes);
#ifdef API_TESTER_CUDA
                else cudaMemset(buf.ptr, 0, buf.bytes);
#endif
            };
            reset(b.ref_out[i]); reset(b.test_out[i]);
        }
    }

    void upload_gp_sample(Buffers& b, const TestCase& tc,
                           const std::vector<float>& sample) {
        const size_t idx   = tc.gp.input_arg;
        const auto& desc   = tc.args[idx];
        const size_t bytes = desc.dims.total() * dtype_size(desc.dtype);
        if (desc.mem == MemSpace::HOST) {
            memcpy(b.shared[idx].ptr, sample.data(), bytes);
        } else {
#ifdef API_TESTER_CUDA
            cudaMemcpy(b.shared[idx].ptr, sample.data(),
                       bytes, cudaMemcpyHostToDevice);
#endif
        }
    }

    static void mkdir_p(const std::string& path) {
        for (size_t i = 1; i <= path.size(); ++i) {
            if (i == path.size() || path[i] == '/') {
                std::string sub = path.substr(0, i);
                mkdir(sub.c_str(), 0755);
            }
        }
    }

    CompareResult compare_output(const Buffers& b, const TestCase& tc) {
        for (size_t i = 0; i < tc.args.size(); ++i) {
            if (tc.args[i].role != ParamRole::OUTPUT) continue;
            return compare(
                b.ref_args[i],  tc.args[i].mem,
                b.test_args[i], tc.args[i].mem,
                tc.args[i].dtype, tc.args[i].dims, tc.tolerance);
        }
        throw std::runtime_error(
            "TestCase '" + tc.label + "' has no OUTPUT descriptor.");
    }

    FullResult run_one(const TestCase& tc) {
        FullResult result;
        Buffers b = alloc_and_init(tc);

        auto ref_call  = [&]() { tc.ref_func(b.ref_args.data()); };
        auto test_call = [&]() { tc.test_func(b.test_args.data()); };

        if (tc.gp.enabled) {
            const auto& gcfg = tc.gp;
            // FIX: use input_arg dims (not output dims) for GP generator size
            const auto& in_desc = tc.args[gcfg.input_arg];

            GpGenerator gen(gcfg.cov,
                            static_cast<int>(in_desc.dims.w),
                            static_cast<int>(in_desc.dims.h),
                            gcfg.seed);

            printf("  [GP] Generating %d samples (%dx%d)...\n",
                   gcfg.iterations,
                   static_cast<int>(in_desc.dims.w),
                   static_cast<int>(in_desc.dims.h));

            result.iter_results.reserve(gcfg.iterations);
            for (int it = 0; it < gcfg.iterations; ++it) {
                auto sample = gen.generate();
                upload_gp_sample(b, tc, sample);
                reset_outputs(b, tc);
                ref_call();
                test_call();

                IterResult ir;
                ir.iteration = it;
                ir.accuracy  = compare_output(b, tc);
                result.iter_results.push_back(ir);

                printf("    iter %3d/%d  L2=%.4f  Max=%.4f  Stdev=%.4f\n",
                       it + 1, gcfg.iterations,
                       ir.accuracy.l2_error,
                       ir.accuracy.max_error,
                       ir.accuracy.stdev);
            }
            result.aggregate = aggregate_iters(result.iter_results, tc.tolerance);

        } else {
            reset_outputs(b, tc);
            ref_call();
            test_call();
            result.accuracy = compare_output(b, tc);
        }

        // ── Dump input / ref output / test output (if requested) ──
        if (!tc.dump_dir.empty()) {
            std::string safe = tc.label;
            for (char& c : safe) if (!isalnum(c) && c!='_' && c!='-') c = '_';
            mkdir_p("dump/" + tc.dump_dir);
            for (size_t i = 0; i < tc.args.size(); ++i) {
                const auto& desc = tc.args[i];
                const std::string name = desc.name.empty()
                    ? ("arg" + std::to_string(i)) : desc.name;
                if (desc.role == ParamRole::INPUT) {
                    dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_" + name + ".txt",
                                b.shared[i].ptr, desc.mem, desc.dtype, desc.dims, name);
                } else if (desc.role == ParamRole::OUTPUT) {
                    dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_ref_" + name + ".txt",
                                b.ref_args[i], desc.mem, desc.dtype, desc.dims, "ref_" + name);
                    dump_buffer("dump/" + tc.dump_dir + "/" + safe + "_test_" + name + ".txt",
                                b.test_args[i], desc.mem, desc.dtype, desc.dims, "test_" + name);
                }
            }
        }

        if (tc.perf.measure_latency) {
            result.has_perf = true;
            result.ref_latency = measure_latency(
                [&](){ reset_outputs(b,tc); ref_call(); },
                tc.perf.warmup_runs, tc.perf.bench_runs);
            result.test_latency = measure_latency(
                [&](){ reset_outputs(b,tc); test_call(); },
                tc.perf.warmup_runs, tc.perf.bench_runs);
        }

        if (tc.perf.measure_memory) {
            result.has_perf = true;
            MemoryTracker tracker(tc.perf.poll_interval_ms);
            reset_outputs(b, tc);
            result.ref_memory  = tracker.measure([&](){ ref_call(); });
            reset_outputs(b, tc);
            result.test_memory = tracker.measure([&](){ test_call(); });
        }

        print_full_result(result, tc);
        return result;
    }

    void print_full_result(const FullResult& r, const TestCase& tc) {
        std::cout << "═════════════════════════════════════════════\n";
        std::cout << "  Test : " << tc.label << "\n";
        std::cout << std::fixed << std::setprecision(6);

        if (tc.gp.enabled && !r.iter_results.empty()) {
            const auto& ag = r.aggregate;
            std::cout << "  ── Accuracy over " << ag.n << " GP iterations ──\n";
            std::cout << "                   mean        max       stdev\n";
            std::cout << "  L2  error  :  " << std::setw(10) << ag.l2_mean   << "  " << std::setw(10) << ag.l2_max    << "  " << std::setw(10) << ag.l2_stdev  << "\n";
            std::cout << "  Max error  :  " << std::setw(10) << ag.max_err_mean  << "  " << std::setw(10) << ag.max_err_max   << "  " << std::setw(10) << ag.max_err_stdev << "\n";
            std::cout << "  Stdev      :  " << std::setw(10) << ag.stdev_mean  << "  " << std::setw(10) << ag.stdev_max   << "  " << std::setw(10) << ag.stdev_stdev << "\n";
            std::cout << "  Worst max error @ (" << ag.max_err_worst_pos.nx << ", " << ag.max_err_worst_pos.ny << ")\n";
            if (tc.tolerance >= 0.0)
                std::cout << "  Pass rate  : " << ag.pass_count << " / " << ag.n
                          << "  (" << std::setprecision(1) << ag.pass_rate * 100.0 << "%)\n";
        } else {
            const auto& a = r.accuracy;
            std::cout << "  ── Accuracy ──\n";
            std::cout << "  L2  error : " << a.l2_error << "\n";
            std::cout << "  Max error : " << a.max_error;
            if (a.max_pos.ny > 0) std::cout << "  @ (" << a.max_pos.nx << ", " << a.max_pos.ny << ")";
            else                  std::cout << "  @ [" << a.max_pos.nx << "]";
            std::cout << "\n  Stdev     : " << a.stdev << "\n";
            if (a.tolerance >= 0.0)
                std::cout << "  Result    : " << (a.passed ? "PASS v" : "FAIL x")
                          << "  (tol=" << a.tolerance << ")\n";
        }

        if (r.has_perf && tc.perf.measure_latency) {
            const auto& rl = r.ref_latency;
            const auto& tl = r.test_latency;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  ── Latency (warmup=" << rl.warmup << ", runs=" << rl.runs << ") ──\n";
            std::cout << "  ref   avg=" << rl.avg_ms << "ms  min=" << rl.min_ms << "ms  max=" << rl.max_ms << "ms\n";
            std::cout << "  test  avg=" << tl.avg_ms << "ms  min=" << tl.min_ms << "ms  max=" << tl.max_ms << "ms\n";
            if (rl.avg_ms > 0.0)
                std::cout << "  speedup : " << std::setprecision(2) << rl.avg_ms / tl.avg_ms << "x\n";
        }

        if (r.has_perf && tc.perf.measure_memory) {
            const auto& rm = r.ref_memory;
            const auto& tm = r.test_memory;
            std::cout << "  ── Workspace Memory (poll=" << rm.poll_interval_ms << "ms) ──\n";
            std::cout << "  ref   peak workspace : " << format_bytes(rm.workspace_bytes) << "\n";
            std::cout << "  test  peak workspace : " << format_bytes(tm.workspace_bytes);
            if (rm.workspace_bytes > 0) {
                double pct = 100.0*(double)tm.workspace_bytes/(double)rm.workspace_bytes - 100.0;
                char buf[32]; snprintf(buf,sizeof(buf),"  (%+.1f%%)",pct);
                std::cout << buf;
            }
            std::cout << "\n  (alloc+free within poll interval may be missed)\n";
        }

        std::cout << "═════════════════════════════════════════════\n";
    }

    std::vector<TestCase> cases_;
};

} } // namespace st::util
