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
# shrinkage_full_tests
