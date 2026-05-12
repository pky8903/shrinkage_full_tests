## Acceleration targets

The table below lists the functions a developer must accelerate, the parameter ranges of interest, and the associated test binary.

| Function | Image size (N×N) | Filter width KW = 2·erosion+1 | Batch B | Test binary | Pass criteria |
|---|---|---|---|---|---|
| `srkTrainingForward` | 256, 512 | 31 – 121 | (typical case) 5 000 | `cd_training_test` | max CD diff < 0.01 nm (pixel: 4, 10 nm) · speedup ≥ 2× · peak mem unchanged |
| `adjointSRK` | 1k, 2k, 4k, 8k, 16k, 32k | 31 – 121 | 1 | `shrinkage_adjoint_test` | fp32: tol_rel = 1e-03, tol_abs = 1e-05; fp16: tol_rel = 1e-02, tol_abs = 1e-03 (ε ≤ 1e-01) · speedup ≥ 2× · peak mem unchanged |
| `shrinkageTangentOaS` | 1k, 2k, 4k, 8k, 16k, 32k | 31 – 121 | 1 | `shrinkage_tan_tanadj_test` | fp32: tol_rel = 1e-03, tol_abs = 1e-05; fp16: tol_rel = 1e-02, tol_abs = 1e-03 (ε ≤ 1e-01) · speedup ≥ 2× · peak mem unchange |
| `shrinkageTangentAdjointOaS` | 1k, 2k, 4k, 8k, 16k, 32k | 31 – 121 | 1 | `shrinkage_tan_tanadj_test` | fp32: tol_rel = 1e-03, tol_abs = 1e-05; fp16: tol_rel = 1e-02, tol_abs = 1e-03 (ε ≤ 1e-01) · speedup ≥ 2× · peak mem unchange |

**Derived parameters** (given KW):
- `erosion = (KW − 1) / 2`  (e.g. KW=121 → erosion=60)
- `output size W = N − 2·erosion`
- `erosion = (input size − output size) / 2`

## How to plug in your accelerated `srkTrainingForward`

`cd_training_test` (`training/main_cd_training.cu`) is the entry point for CD-accuracy validation.

1. Open `training/main_cd_training.cu` and locate `op_test()`.
2. Replace the placeholder body with your accelerated implementation. The function contract is documented in the comment block inside `op_test()`:
   - Input  `d_E`  : `[B × N × N]` col-major float32, device pointer (E = I = same array)
   - Output `d_R`  : `[B × W × W]` col-major float32, device pointer
   - Parameters `N`, `W`, `EROSION`, `dx`, `B`, stream, and pre-computed kernel freq buffers are all available via `g_ctx`.
3. Rebuild and run:
   ```bash
   bash build.sh
   bash run_cd_training_test.sh
   ```
4. The binary runs three sections:
   - **Section A** — Pixel-level accuracy vs reference (GP random inputs, 10 iterations).
   - **Section B** — CD sweep on LNS vertical patterns; measures `CD(E + c·R)`.
   - **Section C** — CD sweep on LNS horizontal patterns; measures `CD(E + c·R)`.

   **Pass**: Section A `max|diff| = 0` (float32 bit-exact); Sections B/C `max CD diff < 0.01 px`; speedup ≥ 2× (latency block in Section A); peak GPU memory unchanged.

---

# shrinkage_full_tests

Four CUDA test executables for the shrinkage operator `h = c·(conv(Gx,g)·∂f/∂x + conv(Gy,g)·∂f/∂y)`.

## Build

```bash
bash build.sh                                        # default sm_86
bash build.sh -DCMAKE_CUDA_ARCHITECTURES=80          # override SM
```

Executables land in `build/`.

## Executables

### `shrinkage_adjoint_test`

**Source:** `main_adjoint.cu`

Validates that the cufftDX OaS forward operator and its adjoint satisfy `<Av, w> = <v, A*w>`.
Runs pixel-by-pixel unit-vector perturbations; passes when every pixel's analytic vs. numerical gradient agrees within tolerance.

- Kernels: `srk_forward.cuh`, `srk_adjoint.cuh`
- No cuDNN required.

### `shrinkage_tan_tanadj_test`

**Source:** `main_tan_tanadj.cu`

Two tests in one binary:

1. **Tangent test** — validates the forward-mode derivative `dh` against central differences `(fwd(A+εd) - fwd(A-εd))/2ε` for 3 independent random directions × 9 mode combinations.
2. **Tangent-adjoint test** — validates the adjoint of the tangent operator against central differences of the adjoint output, for 3 directions × 6 mode combinations.

Kernel: `srk_tangent.cuh`, `srk_tangent_adjoint.cuh` (reuse `srk_forward.cuh`, `srk_adjoint.cuh`).
Green's tensor: SRSRK Bessel K1 × radial Blackman window — `Gx = (γ/2π)(x/r)K1(γr)w(r)`, `Gy = (γ/2π)(y/r)K1(γr)w(r)`.
No cuDNN required.

### `shrinkage_forward_training_test`

**Source:** `main_forward_training.cu`

Compares the cufftDX OaS forward (`srk_training_forward.cuh`) against a reference cuDNN convolution (`srk_cudnn_forward.cuh`) for correctness.
Requires cuDNN.

### `batch_demo`

**Source:** `main_batch_demo.cu`

Demonstrates batched cuDNN forward convolution with timing.
Requires cuDNN.

## Shared kernel files

| File | Description |
|---|---|
| `srk_forward.cuh` | cufftDX OaS tiled 2D convolution (forward) |
| `srk_adjoint.cuh` | Adjoint w.r.t. image f and filter g |
| `srk_tangent.cuh` | Forward-mode tangent derivative |
| `srk_tangent_adjoint.cuh` | Adjoint of the tangent operator |
| `srk_training_forward.cuh` | Training-time forward (cufftDX) |
| `srk_cudnn_forward.cuh` | cuDNN reference forward |

All spatial 2D buffers use **col-major layout** (`buf[col*H + row]`), consistent with ArrayFire / patternUtil / `dump.h`.

## Notebooks

- `plot_adjoints.ipynb` — visualise adjoint test dumps
- `plot_tan_tanadj.ipynb` — visualise tangent / tangent-adjoint dumps
- `plot_batch_demo.ipynb` — visualise batch demo outputs
- `plot_compare_demo.ipynb` — signal + performance comparison: FFT vs Strip cuDNN
- `plot_cd_training.ipynb` — visualise `dump_cd/` images (E, E+c·R ref, E+c·R test, diff)
