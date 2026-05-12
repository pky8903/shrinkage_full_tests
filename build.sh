#!/usr/bin/env bash
# Build all four shrinkage tests:
#   shrinkage_adjoint_test            — Task 1 forward + adjoint validation
#   shrinkage_tan_tanadj_test         — Tangent + tangent-adjoint validation (27 cases)
#   shrinkage_forward_training_test   — cuFFTDX vs cuDNN forward correctness
#   batch_demo                        — cuDNN batched forward demo
#
# Usage:
#   bash build.sh                                          # default sm 86
#   bash build.sh -DCMAKE_CUDA_ARCHITECTURES=80             # override SM
#   bash build.sh -DCUFFTDX_DIR=/path/to/mathdx ...          # override deps
#
# Each executable lands in build/.  Run individually:
#   ./build/shrinkage_adjoint_test
#   ./build/shrinkage_tan_tanadj_test
#   ./build/shrinkage_forward_training_test
#   ./build/batch_demo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    "$@"

cmake --build "$BUILD_DIR" -- -j"$(nproc)"

echo
echo "Build complete.  Executables:"
for tgt in shrinkage_adjoint_test shrinkage_tan_tanadj_test shrinkage_forward_training_test batch_demo; do
    [ -x "$BUILD_DIR/$tgt" ] && echo "  $BUILD_DIR/$tgt"
done
