#!/usr/bin/env bash
# ── run_forward_training_test.sh ──────────────────────────────────
# Compares three SRK forward implementations:
#   (1) srkForwardOaS       — cufftDX OaS tiled (reference)
#   (2) srkTrainingForward  — cuFFT batched N-sized FFT
#   (3) srkCudnnForward     — cuDNN spatial convolution
#
# Small test: B=1, N=256 and N=512 (compile-time template constants).
# Big test  : large batch at user-specified N and B.
# ─────────────────────────────────────────────────────────────────

# ── Small correctness test ────────────────────────────────────────
# NOTE: N1=256 and N2=512 are compile-time constants (template params).
#       To change them, edit main_forward_training.cu and rebuild.
SIGMA=5.0       # kernel sigma for small test (pixels)
B_TEST=1        # batch size for small test

# ── Big batched test (srk_training vs srk_cudnn) ──────────────────
SIGMA_BIG=10.0  # kernel sigma for big test (pixels)
N_BIG=192       # padded image size for big test (N × N)
B_BIG=3000      # batch count for big test

# Derived — small test (shown for reference; computed inside binary):
#   c_w     = 3.0  (Blackman window factor, hardcoded)
#   R_w     = c_w * SIGMA                    = 15 px
#   erosion = floor(R_w)                     = 15   (= (kw-1)/2, matches cuDNN valid conv)
#   kw      = 2 * floor(R_w) + 1            = 31
#   W1      = 256 - 2*erosion = 256-kw+1   = 226
#   W2      = 512 - 2*erosion = 512-kw+1   = 482
#
# Derived — big test:
#   R_w_big = c_w * SIGMA_BIG               = 30 px
#   erosion_big = floor(R_w_big)            = 30
#   kw_big  = 2 * floor(R_w_big) + 1       = 61
#   W_big   = N_BIG - 2*erosion_big        = 132
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/build/shrinkage_forward_training_test" \
    --sigma     "$SIGMA"     \
    --b_test    "$B_TEST"    \
    --sigma_big "$SIGMA_BIG" \
    --n_big     "$N_BIG"     \
    --b_big     "$B_BIG"
