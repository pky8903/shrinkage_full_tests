// test_pattern.cu
//
// Implementation of test-image generators.
// Compiled as CUDA because GAUSSIAN_PROCESS uses GpGenerator (GPU-backed).

#include <patternUtil/test_pattern.h>
#include <stTestCore/st_test_core.h>   // GpGenerator, Covariance2D

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace st { namespace util {

// ── Factory methods ───────────────────────────────────────────────────────

TestPatternConfig TestPatternConfig::lns_vertical(
    int N, float sigma_px, float period_px, int num_nbrs)
{
    TestPatternConfig c;
    c.type           = PatternType::LNS_VERTICAL;
    c.N              = N;
    c.pulse_sigma_px = sigma_px;
    c.period_px      = period_px;
    c.num_nbrs       = num_nbrs;
    return c;
}

TestPatternConfig TestPatternConfig::lns_horizontal(
    int N, float sigma_px, float period_px, int num_nbrs)
{
    TestPatternConfig c;
    c.type           = PatternType::LNS_HORIZONTAL;
    c.N              = N;
    c.pulse_sigma_px = sigma_px;
    c.period_px      = period_px;
    c.num_nbrs       = num_nbrs;
    return c;
}

TestPatternConfig TestPatternConfig::staggered_2d(
    int N, float blob_sigma_px, float pitch_x_px, float pitch_y_px)
{
    TestPatternConfig c;
    c.type          = PatternType::STAGGERED_2D;
    c.N             = N;
    c.blob_sigma_px = blob_sigma_px;
    c.pitch_x_px    = pitch_x_px;
    c.pitch_y_px    = pitch_y_px;
    return c;
}

TestPatternConfig TestPatternConfig::gaussian_process(
    int N, float length_scale_px, unsigned seed)
{
    TestPatternConfig c;
    c.type               = PatternType::GAUSSIAN_PROCESS;
    c.N                  = N;
    c.gp_length_scale_px = length_scale_px;
    c.gp_seed            = seed;
    return c;
}

// ── Internal generators ───────────────────────────────────────────────────

// Evaluate Gaussian pulse train at position x, centred at `centre`.
static float lns_value(float x, float centre, float sigma, float period, int nbrs)
{
    float val = 0.f;
    for (int k = -nbrs; k <= nbrs; ++k) {
        float dx = x - (centre + k * period);
        val += expf(-dx * dx / (2.f * sigma * sigma));
    }
    return val - 0.5f;   // zero crossing at half-peak = feature edge
}

static std::vector<float> gen_lns_vertical(const TestPatternConfig& cfg)
{
    const int   N      = cfg.N;
    const float sigma  = cfg.pulse_sigma_px;
    const float fwhm   = 2.f * sqrtf(2.f * logf(2.f)) * sigma;
    const float period = (cfg.period_px > 0.f) ? cfg.period_px : 4.f * fwhm;
    const float centre = 0.5f * (N - 1);

    std::vector<float> img((size_t)N * N);
    for (int c = 0; c < N; ++c) {
        const float val = lns_value((float)c, centre, sigma, period, cfg.num_nbrs);
        for (int r = 0; r < N; ++r)
            img[c * N + r] = val;   // uniform in row direction
    }
    return img;
}

static std::vector<float> gen_lns_horizontal(const TestPatternConfig& cfg)
{
    const int   N      = cfg.N;
    const float sigma  = cfg.pulse_sigma_px;
    const float fwhm   = 2.f * sqrtf(2.f * logf(2.f)) * sigma;
    const float period = (cfg.period_px > 0.f) ? cfg.period_px : 4.f * fwhm;
    const float centre = 0.5f * (N - 1);

    std::vector<float> img((size_t)N * N);
    for (int r = 0; r < N; ++r) {
        const float val = lns_value((float)r, centre, sigma, period, cfg.num_nbrs);
        for (int c = 0; c < N; ++c)
            img[c * N + r] = val;   // uniform in col direction
    }
    return img;
}

static std::vector<float> gen_staggered_2d(const TestPatternConfig& cfg)
{
    const int   N      = cfg.N;
    const float sigma  = cfg.blob_sigma_px;
    const float px     = cfg.pitch_x_px;
    const float py     = cfg.pitch_y_px;
    const float cx     = 0.5f * (N - 1);
    const float cy     = 0.5f * (N - 1);
    const float inv2s2 = 1.f / (2.f * sigma * sigma);

    // Cover 1.5× image radius in each direction
    const int ni = static_cast<int>(std::ceil(1.5f * N / px)) + 1;
    const int nj = static_cast<int>(std::ceil(1.5f * N / py)) + 1;

    std::vector<float> img((size_t)N * N, 0.f);

    for (int j = -nj; j <= nj; ++j) {
        const float row_j   = cy + j * py;
        // Odd rows (by absolute index) are shifted by half pitch_x (brick stagger)
        const float x_shift = (std::abs(j) % 2) ? px * 0.5f : 0.f;

        for (int i = -ni; i <= ni; ++i) {
            const float col_i = cx + i * px + x_shift;

            // Only iterate over pixels within 4σ of this blob
            const int rmin = std::max(0, (int)std::floor(row_j - 4.f * sigma));
            const int rmax = std::min(N - 1, (int)std::ceil(row_j  + 4.f * sigma));
            const int cmin = std::max(0, (int)std::floor(col_i - 4.f * sigma));
            const int cmax = std::min(N - 1, (int)std::ceil(col_i  + 4.f * sigma));

            for (int c = cmin; c <= cmax; ++c) {
                const float dc = c - col_i;
                for (int r = rmin; r <= rmax; ++r) {
                    const float dr = r - row_j;
                    img[c * N + r] += expf(-(dc * dc + dr * dr) * inv2s2);
                }
            }
        }
    }

    // Subtract 0.5 → zero crossing at feature edges, background ≈ −0.5
    for (auto& v : img) v -= 0.5f;
    return img;
}

static std::vector<float> gen_gaussian_process(const TestPatternConfig& cfg)
{
    GpGenerator gp(Covariance2D::isotropic(cfg.gp_length_scale_px),
                   cfg.N, cfg.N, cfg.gp_seed);
    return gp.generate();
}

// ── Public API ────────────────────────────────────────────────────────────

std::vector<float> generate_pattern(const TestPatternConfig& cfg)
{
    switch (cfg.type) {
    case PatternType::LNS_VERTICAL:    return gen_lns_vertical(cfg);
    case PatternType::LNS_HORIZONTAL:  return gen_lns_horizontal(cfg);
    case PatternType::STAGGERED_2D:    return gen_staggered_2d(cfg);
    case PatternType::GAUSSIAN_PROCESS: return gen_gaussian_process(cfg);
    default:
        throw std::invalid_argument("generate_pattern: unknown PatternType");
    }
}

}} // namespace st::util
