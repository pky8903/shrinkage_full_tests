#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  gtest_adjoint.h — GoogleTest helpers for AdjointTester
//
//  Provides:
//    check_adjoint(...)         one-shot free function (skips AdjointTester boilerplate)
//    expect_adjoint_valid(...)  EXPECT_TRUE wrapper with structured failure message
//    assert_adjoint_valid(...)  ASSERT_TRUE variant (stops test on failure)
//    ST_EXPECT_ADJOINT(...)     convenience macro (non-fatal)
//    ST_ASSERT_ADJOINT(...)     convenience macro (fatal)
//
//  Typical usage in a gtest TEST() body:
//
//    #include <stTestCore/gtest_adjoint.h>   // includes adjoint_tester.h + gtest
//
//    TEST(MyKernelTest, AdjointCorrectness) {
//        auto fwd_args = std::vector<ArgDescriptor>{ ... };
//        auto adj_args = std::vector<ArgDescriptor>{ ... };
//        ST_EXPECT_ADJOINT("my_kernel", fwd_a, fwd_args, 1,
//                          adj_a,       adj_args, 2, 0, 0, 1);
//    }
//
//  Link requirements (CMakeLists):
//    target_link_libraries(my_test PRIVATE
//        stTestCore::stTestCore GTest::gtest_main CUDA::cudart)
// ─────────────────────────────────────────────────────────────────────────────

#include "adjoint_tester.h"
#include <gtest/gtest.h>

namespace st { namespace util {

// ─────────────────────────────────────────────────────────────────────────────
//  check_adjoint — single-shot adjoint verification without AdjointTester setup
//
//  Parameters mirror AdjointTester::register_adjoint exactly.
//  Returns the full AdjointResult so the caller can inspect error statistics
//  before (or instead of) asserting on result.passed.
//
//  Example:
//    AdjointResult r = check_adjoint("label", fwd, fwd_args, 1,
//                                             adj, adj_args, 2, 0, 0, 1, cfg);
//    EXPECT_LT(r.rel_error_max, 1e-4);
//    EXPECT_TRUE(r.passed);
// ─────────────────────────────────────────────────────────────────────────────
inline AdjointResult check_adjoint(
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
    AdjointConfig              cfg         = {},
    PerfConfig                 perf        = {},
    BufferInitFn               init_fn     = nullptr,
    BufferInitFn               adj_init_fn = nullptr,
    GpConfig                   gp          = {},
    std::string                dump_dir    = "")
{
    AdjointTester tester;
    tester.register_adjoint(
        std::move(label),
        fwd_func, std::move(fwd_args), fwd_out_idx,
        adj_func, std::move(adj_args), adj_out_idx, adj_dqdout_idx,
        fwd_perturb_idx, adj_input_idx,
        cfg, perf,
        std::move(init_fn), std::move(adj_init_fn),
        gp, std::move(dump_dir));
    return tester.run(0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  expect_adjoint_valid / assert_adjoint_valid
//
//  Wrap an AdjointResult in a gtest EXPECT_TRUE / ASSERT_TRUE with a
//  structured failure message that includes the key error statistics.
//
//  Failure message format:
//    [my_kernel] FAIL  rel_max=1.23e-02  rel_mean=4.56e-03  abs_max=7.89e-04
//               worst_px=(12,34)  tol_rel=1e-03  tol_abs=1e-05
// ─────────────────────────────────────────────────────────────────────────────
inline void expect_adjoint_valid(const AdjointResult& r,
                                  const std::string&   label = "")
{
    EXPECT_TRUE(r.passed)
        << "[" << label << "] FAIL"
        << "  rel_max="  << r.rel_error_max
        << "  rel_mean=" << r.rel_error_mean
        << "  abs_max="  << r.abs_error_max
        << "\n         worst_px=(" << r.worst_nx << "," << r.worst_ny << ")"
        << "  tol_rel=" << r.tol_rel
        << "  tol_abs=" << r.tol_abs;
}

inline void assert_adjoint_valid(const AdjointResult& r,
                                  const std::string&   label = "")
{
    ASSERT_TRUE(r.passed)
        << "[" << label << "] FAIL"
        << "  rel_max="  << r.rel_error_max
        << "  rel_mean=" << r.rel_error_mean
        << "  abs_max="  << r.abs_error_max
        << "\n         worst_px=(" << r.worst_nx << "," << r.worst_ny << ")"
        << "  tol_rel=" << r.tol_rel
        << "  tol_abs=" << r.tol_abs;
}

}} // namespace st::util

// ─────────────────────────────────────────────────────────────────────────────
//  ST_EXPECT_ADJOINT — non-fatal adjoint check macro
//  ST_ASSERT_ADJOINT — fatal adjoint check macro (stops the test on failure)
//
//  Positional parameters:
//    label    (const char*)            — test identification string
//    fwd      (AdaptedFunc)            — forward function
//    fwd_args (std::vector<ArgDescriptor>) — forward arg descriptors (named var)
//    fwd_out  (size_t)                 — fwd_out_idx
//    adj      (AdaptedFunc)            — adjoint function
//    adj_args (std::vector<ArgDescriptor>) — adjoint arg descriptors (named var)
//    adj_out  (size_t)                 — adj_out_idx
//    dqdout   (size_t)                 — adj_dqdout_idx
//    perturb  (size_t)                 — fwd_perturb_idx
//    adj_in   (size_t)                 — adj_input_idx
//    ...                               — optional: AdjointConfig, PerfConfig,
//                                         BufferInitFn, BufferInitFn, GpConfig,
//                                         dump_dir  (all positional, same order
//                                         as check_adjoint / register_adjoint)
//
//  Note: fwd_args and adj_args must be named local variables (not brace-init
//  literals) to avoid comma ambiguity in the macro expansion.
//
//  Example:
//    auto fwd_args = std::vector<ArgDescriptor>{ ... };
//    auto adj_args = std::vector<ArgDescriptor>{ ... };
//    ST_EXPECT_ADJOINT("my_kernel", fwd_a, fwd_args, 1,
//                      adj_a,       adj_args, 2, 0, 0, 1,
//                      AdjointConfig{256, 1e-3, 1e-5, 1e-3, 42});
// ─────────────────────────────────────────────────────────────────────────────

#define ST_EXPECT_ADJOINT(label, fwd, fwd_args, fwd_out,       \
                          adj, adj_args, adj_out, dqdout,       \
                          perturb, adj_in, ...)                 \
    ::st::util::expect_adjoint_valid(                           \
        ::st::util::check_adjoint(                              \
            label, fwd, fwd_args, fwd_out,                      \
            adj, adj_args, adj_out, dqdout,                     \
            perturb, adj_in, ##__VA_ARGS__),                    \
        label)

#define ST_ASSERT_ADJOINT(label, fwd, fwd_args, fwd_out,       \
                          adj, adj_args, adj_out, dqdout,       \
                          perturb, adj_in, ...)                 \
    ::st::util::assert_adjoint_valid(                           \
        ::st::util::check_adjoint(                              \
            label, fwd, fwd_args, fwd_out,                      \
            adj, adj_args, adj_out, dqdout,                     \
            perturb, adj_in, ##__VA_ARGS__),                    \
        label)
