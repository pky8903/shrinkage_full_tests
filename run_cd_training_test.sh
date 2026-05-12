#!/usr/bin/env bash
# ── run_cd_training_test.sh ───────────────────────────────────────
# CD-accuracy validation for srkTrainingForward.
#
# Plug in your accelerated implementation:
#   1. Open training/main_cd_training.cu and replace op_test().
#   2. Rebuild:  bash build.sh
#   3. Run this script.
#
# Three sections:
#   A — Pixel accuracy vs reference (GP random inputs, GP_ITERS iterations)
#   B — CD sweep on LNS vertical patterns;   measures CD(E + c·R)
#   C — CD sweep on LNS horizontal patterns; measures CD(E + c·R)
#
# Pass criteria:
#   Section A : max|diff| = 0  (float32 bit-exact) or < tol_abs in fp16 mode
#   Sections B/C: max CD diff < 0.01 px across all batch patterns
#   Speedup   : ≥ 2× vs reference (reported in Section A latency block)
#   Memory    : peak GPU memory unchanged
# ─────────────────────────────────────────────────────────────────

# ── Tunable parameters ────────────────────────────────────────────
SIGMA=20.0        # SRSRK kernel sigma (pixels); drives filter width
N=256             # padded input image size (N × N)
B=30              # number of patterns per CD sweep
LNS_SIG=8.0       # LNS edge sigma (pixels)
LNS_PERIOD=0.0    # LNS period (0 = auto: one full cycle across W)
COEFF_C=0.1       # shrinkage coefficient c  (signal = E + c·R)
GP_ITERS=10       # GP random iterations for Section A
DUMP_MAX=4        # number of batches to dump as images (0 = no dump)

# Derived (shown for reference; computed inside the binary):
#   c_w     = 3.0  (Blackman window factor, hardcoded)
#   KW      = 2 * floor(c_w * SIGMA / dx) + 1  = 121  (at SIGMA=20, dx=1)
#   erosion = (KW - 1) / 2                      = 60
#   W       = N - 2 * erosion                   = 136
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/build/cd_training_test" \
    --sigma      "$SIGMA"      \
    --N          "$N"          \
    --B          "$B"          \
    --lns_sig    "$LNS_SIG"    \
    --lns_period "$LNS_PERIOD" \
    --coeff_c    "$COEFF_C"    \
    --gp_iters   "$GP_ITERS"   \
    --dump_max   "$DUMP_MAX"
