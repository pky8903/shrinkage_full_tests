#!/usr/bin/env bash
# ── run_batch_demo.sh ─────────────────────────────────────────────
# Visual sanity check of srkTrainingForward on a small batch.
# Dumps E, I, R per batch to dump_batch/ for notebook inspection.
# Open notebooks/plot_batch_demo.ipynb after running.
# ─────────────────────────────────────────────────────────────────

# ── Tunable parameters ────────────────────────────────────────────
SIGMA=20.0     # kernel sigma (pixels); drives filter size
N=256         # padded image size (N × N)
B=4           # batch count

# Derived (shown for reference; computed inside the binary):
#   c_w     = 3.0  (Blackman window factor, hardcoded)
#   R_w     = c_w * SIGMA                    = 60 px   (sigma=20)
#   erosion = floor(R_w) + 1                 = 61
#   kw      = 2 * floor(R_w) + 1            = 121
#   W_valid = N - 2 * erosion               = 134
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/build/batch_demo" \
    --sigma "$SIGMA" \
    --n     "$N"     \
    --b     "$B"
