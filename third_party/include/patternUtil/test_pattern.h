// test_pattern.h
//
// Uniform test-image generator for stTest examples.
//
// All outputs: N×N col-major float32 host vector, image[col*N + row].
// Signals are zero-centred at feature edges (background ≈ −0.5, peak ≈ +0.5)
// so they are directly usable with MeasureCD_Batched (zero-crossing = edge).

#pragma once

#include <vector>

namespace st { namespace util {

// ---------------------------------------------------------------------------
enum class PatternType {
    LNS_VERTICAL,    ///< Gaussian pulse train — vertical lines   (varies in col)
    LNS_HORIZONTAL,  ///< Gaussian pulse train — horizontal lines (varies in row)
    STAGGERED_2D,    ///< Staggered 2-D dot array: small isotropic Gaussian blobs
                     ///<   on a brick lattice; odd rows offset by pitch_x/2
    GAUSSIAN_PROCESS ///< Isotropic GP with Gaussian covariance (uses GpGenerator)
};

// ---------------------------------------------------------------------------
struct TestPatternConfig {
    PatternType type = PatternType::LNS_VERTICAL;
    int         N    = 256;          ///< Image is N×N pixels

    // ── Line & Space (LNS_VERTICAL / LNS_HORIZONTAL) ──────────────
    float pulse_sigma_px = 8.0f;    ///< Gaussian pulse σ [px]
    float period_px      = 0.0f;    ///< Line pitch [px]; 0 → auto = 4·FWHM
    int   num_nbrs       = 4;       ///< Neighbour lines each side of centre

    // ── Staggered 2-D dot array (STAGGERED_2D) ─────────────────────
    float blob_sigma_px = 3.0f;     ///< Blob σ [px] (isotropic 2-D Gaussian)
    float pitch_x_px    = 20.0f;    ///< Horizontal pitch [px]
    float pitch_y_px    = 20.0f;    ///< Vertical pitch [px]

    // ── Gaussian Process (GAUSSIAN_PROCESS) ────────────────────────
    float    gp_length_scale_px = 32.0f;  ///< Isotropic covariance length scale [px]
    unsigned gp_seed            = 42;

    // ── Convenience factories ──────────────────────────────────────
    static TestPatternConfig lns_vertical(
        int N, float sigma_px, float period_px = 0.f, int num_nbrs = 4);

    static TestPatternConfig lns_horizontal(
        int N, float sigma_px, float period_px = 0.f, int num_nbrs = 4);

    static TestPatternConfig staggered_2d(
        int N, float blob_sigma_px,
        float pitch_x_px = 20.f, float pitch_y_px = 20.f);

    static TestPatternConfig gaussian_process(
        int N, float length_scale_px, unsigned seed = 42);
};

// ---------------------------------------------------------------------------
/// Generate an N×N test image (col-major, host) according to cfg.
std::vector<float> generate_pattern(const TestPatternConfig& cfg);

}} // namespace st::util
