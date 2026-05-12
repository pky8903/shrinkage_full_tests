#!/usr/bin/env bash
# ── run_adjoint_test.sh ───────────────────────────────────────────
# Validates srkForwardOaS against a CPU reference (Task 1), then
# runs AdjointTester for adjointWrtI and adjointWrtE (Task 2).
# Also runs a large-image test at N_BIG × N_BIG with M=2048 tiles.
# ─────────────────────────────────────────────────────────────────

# ── Tunable parameters ────────────────────────────────────────────
SIGMA=5.0     # Gaussian kernel sigma (pixels)
W_OUT=256     # valid output size (W × W); padded N = W_OUT + 2*erosion
N_BIG=8192    # padded image size for the large-N adjoint test

# Derived (shown for reference; computed inside the binary):
#   c_w     = 3.0  (Gaussian truncation factor, hardcoded)
#   R_w     = c_w * SIGMA                    = 15 px
#   erosion = floor(R_w) + 1                 = 16
#   kw      = 2 * floor(R_w) + 1            = 31
#   N_large = W_OUT + 2 * erosion           = 288
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/build/shrinkage_adjoint_test" \
    --sigma "$SIGMA"  \
    --w_out "$W_OUT"  \
    --n_big "$N_BIG"
