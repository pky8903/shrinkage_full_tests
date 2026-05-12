#pragma once
#include "arg_descriptor.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace st { namespace util {

// ─────────────────────────────────────────────────────────────
//  Covariance2D
//
//  Spatial Gaussian covariance kernel:
//
//    C(dx, dy) = σ² exp(−½ Q(dx, dy))
//
//  where
//
//    Q(dx, dy) = [(dx/ℓx)² − 2ρ(dx/ℓx)(dy/ℓy) + (dy/ℓy)²] / (1 − ρ²)
//
//  Parameters:
//    sigma : amplitude (field std dev ≈ sigma)
//    ell_x : correlation length in x direction (grid points)
//    ell_y : correlation length in y direction (grid points)
//    rho   : cross-correlation coefficient, −1 < rho < 1
//
//  For isotropic fields use ell_x == ell_y, rho == 0.
// ─────────────────────────────────────────────────────────────

struct Covariance2D {
    double sigma = 1.0;   ///< amplitude scale
    double ell_x = 16.0;  ///< correlation length in x (grid points)
    double ell_y = 16.0;  ///< correlation length in y (grid points)
    double rho   = 0.0;   ///< cross-correlation coefficient, |rho| < 1

    bool is_valid() const {
        return sigma > 0.0 && ell_x > 0.0 && ell_y > 0.0
            && rho > -1.0 && rho < 1.0;
    }

    /// Evaluate covariance at spatial displacement (dx, dy) in grid units.
    double eval(double dx, double dy) const {
        const double u = dx / ell_x, v = dy / ell_y;
        const double q = (u*u - 2.0*rho*u*v + v*v) / (1.0 - rho*rho);
        return sigma * sigma * std::exp(-0.5 * q);
    }

    /// Isotropic: C(r) = σ² exp(−r²/(2ℓ²))
    static Covariance2D isotropic(double ell, double sigma = 1.0) {
        return {sigma, ell, ell, 0.0};
    }

    /// Anisotropic: independent length scales per axis, optional cross-correlation.
    static Covariance2D anisotropic(double ell_x, double ell_y,
                                     double rho = 0.0, double sigma = 1.0) {
        return {sigma, ell_x, ell_y, rho};
    }
};

// ─────────────────────────────────────────────────────────────
//  GpGenerator
//
//  Generates 2D Gaussian Random Fields via the spectral (FFT) method,
//  which is mathematically equivalent to direct sampling from N(0, Σ)
//  when Σ is a BCCB (block-circulant with circulant blocks) matrix —
//  i.e., a periodic stationary covariance (see §6, §8 of reference doc).
//
//  Algorithm:
//    1. Evaluate the spatial covariance kernel c on the W×H grid
//       with periodic (nearest-image) wrapping.
//    2. Compute the power spectrum: Ŝ = FFT2D(c).
//       For a valid stationary covariance, Ŝ is real and non-negative.
//    3. Precompute amplitude filter: H[k] = sqrt(max(0, Re(Ŝ[k]))).
//    4. For each sample:
//         a. Generate white noise w ~ N(0,1).
//         b. ŵ = FFT2D(w)
//         c. x̂[k] = H[k] · ŵ[k]
//         d. x = Re(IFFT2D(x̂))
//       The resulting field has covariance Cov(x[i], x[j]) = c[i−j].
//
//  Equivalence justification (Cov(x) = Σ):
//    x = IFFT(H · FFT(w))  with Cov(w) = I
//    → Cov(x[i,j]) = IFFT(|H|²)[i−j] = IFFT(Ŝ)[i−j] = c[i−j]
//
//  Arbitrary (W, H) are supported: the internal Radix-2 FFT operates on
//  padded dimensions (Wp, Hp) = next-power-of-2 ≥ (W, H), and the final
//  output is cropped to the requested W × H region. The cropped field is
//  still a valid stationary GP sample with the requested covariance.
// ─────────────────────────────────────────────────────────────

class GpGenerator {
public:
    GpGenerator(Covariance2D cov, int width, int height, uint64_t seed = 0)
        : cov_(cov), W_(width), H_(height)
    {
        if (!cov.is_valid())
            throw std::invalid_argument(
                "Covariance2D is not valid (check sigma>0, ell>0, |rho|<1).");
        if (W_ <= 0 || H_ <= 0)
            throw std::invalid_argument("Width and height must be positive");

        // Pad to next power-of-2 for the internal Radix-2 FFT
        Wp_ = next_pow2(W_);
        Hp_ = next_pow2(H_);

        rng_.seed(seed == 0
            ? std::random_device{}()
            : static_cast<std::mt19937_64::result_type>(seed));

        precompute_filter();
    }

    /// Returns float32 field, col-major, size W×H.
    /// Element at spatial (col, row) is at out[col * H_ + row].
    /// The field has covariance Cov(x[i], x[j]) = C(i−j) where C is the
    /// Gaussian kernel defined by the Covariance2D parameters.
    std::vector<float> generate() {
        const int Np = Wp_ * Hp_;
        std::normal_distribution<double> normal(0.0, 1.0);

        // White noise w ~ N(0,1) over padded grid
        std::vector<std::complex<double>> freq(Np);
        for (int i = 0; i < Np; ++i)
            freq[i] = { normal(rng_), 0.0 };

        // ŵ = FFT2D(w)
        fft2d(freq, Wp_, Hp_, false);

        // x̂[k] = H[k] · ŵ[k]
        for (int i = 0; i < Np; ++i)
            freq[i] *= H_k_[i];

        // x = Re(IFFT2D(x̂))
        fft2d(freq, Wp_, Hp_, true);

        // Crop from padded row-major to col-major W_×H_ (top-left region).
        // freq[r * Wp_ + c].real() → out[c * H_ + r], for r<H_, c<W_.
        std::vector<float> out((size_t)W_ * H_);
        for (int r = 0; r < H_; ++r)
            for (int c = 0; c < W_; ++c)
                out[c * H_ + r] = static_cast<float>(freq[r * Wp_ + c].real());

        return out;
    }

    int width()  const { return W_; }
    int height() const { return H_; }

private:
    static int next_pow2(int n) {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

public:

private:

    // ─── Filter precomputation ──────────────────────────────────────────────
    //
    //  Correct FFT-based GP sampling requires:
    //    H[k] = sqrt(Ŝ[k]) = sqrt(FFT(c)[k])
    //
    //  where c is the spatial covariance kernel evaluated on the grid.
    //
    //  Derivation (non-unitary FFT, IFFT divides by N):
    //    Cov(x[i], x[j]) = IFFT(|H|²)[i−j] = IFFT(Ŝ)[i−j] = c[i−j]
    //
    //  The covariance kernel is evaluated with periodic (nearest-image)
    //  wrapping to produce the BCCB structure required for exact equivalence.
    // ───────────────────────────────────────────────────────────────────────

    void precompute_filter() {
        const int Np = Wp_ * Hp_;
        H_k_.resize(Np);

        // Step 1: Evaluate spatial covariance kernel with periodic wrapping
        // on the padded grid (Wp_ × Hp_).
        std::vector<std::complex<double>> c_grid(Np);
        for (int r = 0; r < Hp_; ++r) {
            double dy = (r <= Hp_/2) ? static_cast<double>(r)
                                     : static_cast<double>(r - Hp_);
            for (int c = 0; c < Wp_; ++c) {
                double dx = (c <= Wp_/2) ? static_cast<double>(c)
                                         : static_cast<double>(c - Wp_);
                c_grid[r * Wp_ + c] = { cov_.eval(dx, dy), 0.0 };
            }
        }

        // Step 2: Power spectrum Ŝ = FFT2D(c).
        fft2d(c_grid, Wp_, Hp_, false);

        // Step 3: Amplitude filter H[k] = sqrt(max(0, Re(Ŝ[k]))).
        for (int i = 0; i < Np; ++i)
            H_k_[i] = std::sqrt(std::max(0.0, c_grid[i].real()));
    }

    // ─── Radix-2 Cooley-Tukey FFT ──────────────────────────────────────────

    static void fft1d(std::vector<std::complex<double>>& a, bool inverse) {
        const int n = static_cast<int>(a.size());
        if (n == 1) return;

        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }

        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2.0 * M_PI / len * (inverse ? 1 : -1);
            std::complex<double> wlen(std::cos(ang), std::sin(ang));
            for (int i = 0; i < n; i += len) {
                std::complex<double> w(1.0, 0.0);
                for (int j = 0; j < len / 2; ++j) {
                    auto u = a[i + j];
                    auto v = a[i + j + len/2] * w;
                    a[i + j]         = u + v;
                    a[i + j + len/2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (inverse)
            for (auto& x : a) x /= n;
    }

    void fft2d(std::vector<std::complex<double>>& data,
               int w, int h, bool inverse) {
        std::vector<std::complex<double>> row_buf(w), col_buf(h);

        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) row_buf[c] = data[r * w + c];
            fft1d(row_buf, inverse);
            for (int c = 0; c < w; ++c) data[r * w + c] = row_buf[c];
        }
        for (int c = 0; c < w; ++c) {
            for (int r = 0; r < h; ++r) col_buf[r] = data[r * w + c];
            fft1d(col_buf, inverse);
            for (int r = 0; r < h; ++r) data[r * w + c] = col_buf[r];
        }
    }

    Covariance2D cov_;
    int W_, H_;     // requested output dims
    int Wp_, Hp_;   // padded (power-of-2) dims used internally
    std::mt19937_64 rng_;
    std::vector<double> H_k_;
};

} } // namespace st::util
