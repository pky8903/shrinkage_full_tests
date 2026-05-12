#!/usr/bin/env bash
# ── run_tan_tanadj_test.sh ────────────────────────────────────────
# Validates the tangent-derivative and tangent-adjoint-derivative of
# the SRK shrinkage operator against numerical central-difference.
# ─────────────────────────────────────────────────────────────────

# ── Tunable parameters ────────────────────────────────────────────
SIGMA=5.0     # SRSRK kernel sigma (pixels); drives Blackman window radius
W_OUT=256     # valid output size (W × W); padded N = W_OUT + 2*erosion

# Derived (shown for reference; computed inside the binary):
#   c_w     = 3.0  (Blackman window factor, hardcoded)
#   R_w     = c_w * SIGMA                    = 15 px
#   erosion = floor(R_w) + 1                 = 16
#   kw      = 2 * floor(R_w) + 1            = 31
#   N_large = W_OUT + 2 * erosion           = 288
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/build/shrinkage_tan_tanadj_test" \
    --sigma "$SIGMA"  \
    --w_out "$W_OUT"
