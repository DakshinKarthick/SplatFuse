# SplatFuse implementation record

This is the durable, code-by-code record of how SplatFuse turns the original
GaussianSplat foundation into a dual-pipeline WebGPU research renderer. It also
records what was verified locally and what still requires the CUDA/HIP-capable
computer mentioned in the project brief.

## 1. Outcome and scope

SplatFuse keeps the completed capture/COLMAP work, the CUDA forward-pass
blueprint, the PLY loader, and the original WebGL shader work. The new active
viewer is native WebGPU and implements:

1. a unified GPU storage layout;
2. GPU projection, covariance, conic generation, culling, and frame reduction;
3. a portable WGSL bitonic key/value sorter shared by both paths;
4. globally sorted instanced quads with hardware blending;
5. count, exclusive scan, duplicate, tile sort, and range passes;
6. a 16×16 cooperative compute rasterizer with early-ray termination;
7. timestamp-query telemetry plus deterministic benchmark controls;
8. an analytical/empirical crossover model;
9. a ten-frame device calibration and adaptive per-frame selector.

The original repository was not modified. This work lives in the separate
`SplatFuse` repository and is split across exactly ten progress commits.

## 2. Ten stages and commits

| Stage | Commit subject | Deliverable |
|---:|---|---|
| 1 | `stage 1: establish SplatFuse WebGPU foundation` | Renamed fork, WebGPU adapter/device/canvas bootstrap |
| 2 | `stage 2: add unified aligned GPU scene storage` | 64-byte splat records and shared buffers |
| 3 | `stage 3: implement GPU projection culling and tile counts` | Covariance projection, conics, bounds, visible reduction |
| 4 | `stage 4: add shared WGSL GPU key value sort` | Portable 64-bit-pair bitonic sort |
| 5 | `stage 5: render globally sorted WebGPU quads` | Pipeline A and premultiplied hardware blending |
| 6 | `stage 6: scan and bin splats into screen tiles` | Prefix scan and tile-key duplication |
| 7 | `stage 7: complete compute tile rasterizer with early exit` | Pipeline B, storage texture, presentation |
| 8 | `stage 8: instrument passes and add benchmark matrix` | Timestamp queries, workload generator, CSV/JSON |
| 9 | `stage 9: derive empirical crossover cost model` | Regression, boundary, model fitter |
| 10 | `stage 10: integrate adaptive renderer and document implementation` | Ten-frame calibration, live selection, this record |

Use `git log --reverse --oneline` to show the project progressing in this order.

## 3. Repository and runtime entry points

### `viewer/src/main.js`

This is the browser entry point. It performs the following work in order:

1. sizes the canvas in physical pixels, capped at device pixel ratio 2;
2. requests WebGPU while loading a binary PLY in parallel;
3. falls back to a deterministic 50k synthetic scene when no local PLY exists;
4. optionally applies the `?flip=1` 180-degree X transform to centers and quaternions;
5. robustly frames on sampled 5th–95th percentiles so distant floaters do not
   collapse the useful scene;
6. uses Three.js only for `PerspectiveCamera`, matrix math, and `OrbitControls`—
   Three.js no longer owns rendering;
7. calls `SplatFuseRenderer.render()` on each animation frame;
8. displays measured time, active path, visible count, mean radius, duplication,
   calibration state, predictions, and overflow.

Query controls are intentionally reproducible:

```text
?scene=/scenes/mypic1.ply&flip=1
?synthetic=250000
?pipeline=auto|quad|tile
?maxDup=8&k=16
```

### `viewer/src/webgpu/GpuContext.js`

`GpuContext.create()` requests a high-performance adapter, enables
`timestamp-query` only when advertised, and requests the adapter's available
`maxBufferSize` and `maxStorageBufferBindingSize`. It configures the canvas with
the browser's preferred format and reports device loss. Timestamp denial is not
fatal: telemetry falls back to CPU/queue completion time.

### `viewer/src/SplatLoader.js`

The retained loader parses binary little-endian PLY by property name rather than
assuming a fixed record order. It decodes:

- `scale_i` with `exp(raw)`;
- `opacity` with `1 / (1 + exp(-raw))`;
- DC spherical harmonics with `0.5 + 0.28209479 * f_dc_i`;
- quaternion, position, scale, RGB, and opacity into typed arrays.

The SplatFuse change checks `header + count × stride` before reading, producing a
clear truncated-file error instead of an out-of-range `DataView` failure.

## 4. Shared GPU data contract

### `layout.js`

`packSplats()` changes the loader's structure-of-arrays output into four aligned
`vec4<f32>` values per Gaussian:

| Byte | WGSL field | Meaning |
|---:|---|---|
| 0 | `centerOpacity` | XYZ center and activated opacity |
| 16 | `scale` | activated XYZ scale plus padding |
| 32 | `rotation` | normalized in shader; file order WXYZ |
| 48 | `color` | decoded RGB plus padding |

The stride is 64 bytes. This layout avoids `vec3` alignment surprises and is read
unchanged by both paths.

Projection writes another 64-byte record:

| Field | Contents |
|---|---|
| `centerRadius` | pixel center, view-space Z, 3-sigma radius |
| `conicOpacity` | inverse covariance `(a,b,c)` and opacity |
| `color` | RGB |
| `tileBounds` | inclusive min and exclusive max tile coordinates |

### `SceneBuffers.js`

This class owns the one scene upload and all per-Gaussian shared products:
`splats`, `projected`, `tileCounts`, `tileOffsets`, `activeIds`, global sort keys
and values, frame statistics, and indirect draw arguments. It checks the
adapter's storage-binding limit before allocating. `sortCapacity` is the next
power of two because the portable bitonic network needs a power-of-two domain;
unused records receive the all-ones sentinel.

## 5. Projection and reduction

### `shaders/projection.wgsl`

`resetFrame` clears sort sentinels, counts, four atomic statistics, and the
indirect draw command. `projectAndCull` then maps one invocation to one Gaussian.

For each Gaussian it:

1. transforms the center into view and clip space;
2. rejects centers behind the camera;
3. rotates the three scaled principal axes by the normalized quaternion;
4. transforms each axis differential through the view and perspective Jacobian;
5. builds the 2D covariance
   `Σ₂D = d0·d0ᵀ + d1·d1ᵀ + d2·d2ᵀ + 0.3I`;
6. rejects a non-positive determinant;
7. finds the largest eigenvalue and uses `ceil(3 sqrt(λmax))` for the footprint;
8. inverts the 2×2 covariance into conic `(c,-b,a)/det`;
9. calculates clamped 16×16 tile bounds and touched-tile count;
10. atomically appends a visible ID, a far-to-near global depth key, radius sum,
    duplicate sum, and indirect instance count.

Float depth is converted to an order-preserving unsigned integer by flipping the
sign bit for positive values and complementing negatives. This avoids invalid
numeric comparison of raw IEEE-754 bits.

### `ProjectionPipeline.js`

The JavaScript wrapper owns the 224-byte camera uniform: view, projection,
view-projection, viewport and reciprocal viewport, splat count, sort capacity,
and tile grid. It dispatches reset and projection as separately timed passes.

## 6. Shared GPU sort

### `sortPlan.js`, `GpuBitonicSort.js`, and `shaders/bitonic-sort.wgsl`

The brief allows radix or bitonic sorting. SplatFuse chooses bitonic sorting for
the portable baseline because it needs no subgroups and has deterministic
cross-workgroup behavior. A key is `vec2<u32>`:

- high word: primary key (`0` globally, tile ID for binned work);
- low word: order-preserving depth;
- value: Gaussian ID.

The JavaScript plan generates every `(k,j)` compare/merge stage. Each stage has a
256-byte dynamically aligned uniform record, so all dispatches may remain in one
compute pass without the common error where repeated `queue.writeBuffer` calls
make every dispatch see only the final parameters.

Global keys use normal ordered depth (far-to-near). Tile keys complement ordered
depth (near-to-far), while keeping tile ID in the high word. The all-ones key
sorts unused capacity to the end.

This is fully GPU-resident and removes `sortWorker.js` from the active path. It is
also intentionally the clearest optimization target: bitonic complexity is
`O(M log²M)`; a production 4/8-bit radix or Onesweep implementation should replace
it after the portable measurements establish the baseline.

## 7. Pipeline A: global hardware quads

### `QuadPipeline.js` and `shaders/quad.wgsl`

The quad path executes:

```text
projection/cull -> global GPU sort -> indirect instanced draw
```

There is no vertex buffer. Six corners are indexed from a WGSL constant and the
indirect instance count comes from projection. Each instance reads a sorted
Gaussian ID, expands a square by the projected 3-sigma radius, and passes pixel
delta plus conic data to the fragment shader.

The fragment exponent is:

```text
-0.5 * (a dx² + 2b dxdy + c dy²)
```

Fragments outside the ellipse or below `1/255` alpha are discarded. Output is
premultiplied `(RGB × alpha, alpha)`. The render pipeline configures
`src=one`, `dst=one-minus-src-alpha`, which implements:

```text
result = source + destination * (1 - sourceAlpha)
```

The far-to-near ordering is therefore correct for hardware "over" blending.

## 8. Pipeline B: compute tiles

### `GpuPrefixScan.js` and `shaders/prefix-scan.wgsl`

The first level performs a 256-element Blelloch exclusive scan in
`var<workgroup>` memory and writes one sum per block. JavaScript recursively
scans block sums until one block remains, then dispatches add-offset passes from
the top level back down. It uses no subgroup/warp extension.

### `TileBinningFrontend.js` and `shaders/tile-binning.wgsl`

Tile key capacity begins at `nextPowerOfTwo(N × maxDup)` and is capped by both
the adapter storage-binding limit and maximum legal dispatch size. Each frame:

1. all key/value slots become sentinels;
2. projected touched-tile counts are exclusively scanned;
3. each visible Gaussian loops over its `[min,max)` tile rectangle;
4. it writes `(tile ID, complemented depth)` and Gaussian ID at its scan offset;
5. writes beyond bounded capacity are skipped and atomically counted.

The selector forces quads after an overflow measurement, so capacity failure is
observable and safe rather than memory corruption.

### `TilePipeline.js` and `shaders/tile-raster.wgsl`

After shared GPU sorting, `detectRanges` compares adjacent high words and records
`[start,end)` for each tile. The raster dispatch uses workgroups of exactly
`16×16×1`; each invocation maps to one pixel in that tile.

Every 256-entry batch is cooperatively loaded into three workgroup arrays
(center, conic, color). Every live pixel then evaluates the batch front-to-back:

```text
C = C + color_i * alpha_i * T
T = T * (1 - alpha_i)
```

When `T < 0.0001`, that pixel stops evaluating Gaussian data. It must still pass
through workgroup barriers while other pixels finish; this avoids divergent
barrier deadlocks. The result is stored directly in an `rgba8unorm` storage
texture. `present.wgsl` displays that texture with a full-screen triangle.

## 9. Telemetry and benchmark matrix

### `Telemetry.js`

Each logical compute/render pass requests beginning and ending timestamp writes.
After the last pass the query set is resolved, copied to a tiny mapping buffer,
and converted from nanoseconds to milliseconds. The reported total spans the
first pass start through the final pass end. Labels include:

```text
projection.reset, projection.project-cull
quad.sort, quad.raster
tile.bin-reset, tile.prefix-scan, tile.prefix-add, tile.duplicate
tile.sort, tile.ranges, tile.raster, tile.present
```

If `timestamp-query` is absent, the same API waits for queue completion and
reports a CPU-observed total. This is suitable as a fallback but must not be
mixed with timestamp rows when fitting a publication model.

### `BenchmarkHarness.js`

`buildBenchmarkMatrix()` spans the requested independent variables:

- N: 50k, 100k, 250k, 500k, 1M, 2M, 5M;
- mean projected radius: 2, 8, 24, 64 px;
- duplication factor: 1, 4, 16;
- depth complexity: 2, 8, 32, 128;
- resolution: 720p, 1080p, 1440p, 4K.

`SyntheticSceneGenerator` is seeded and deterministic. `BenchmarkHarness`
performs warmups, records repeated samples for both paths, classifies a coarse
hardware tier, and exports JSON or properly escaped CSV. The callback-based
design lets a study choose the full Cartesian matrix or a smaller stratified
subset without changing renderer code.

## 10. Crossover model

### `CrossoverModel.js`

The quad feature vector is:

```text
[1, N log2(N), N r², pixels k]
```

The tile feature vector is:

```text
[1, (ND) log2(ND), tiles * min(k,64) * 256, ND]
```

These correspond to sort, footprint/fragment, saturated tile work, and
duplication costs. Features are scaled before ridge regression to avoid an
ill-conditioned normal equation. Separate coefficients are fitted per pipeline
and may be filtered by hardware profile.

`boundary()` returns:

```text
F(N,r,D,k,res,arch) = predictedQuadMs - predictedTileMs
```

`F = 0` is the crossover. Negative selects quads; positive selects tiles. It also
returns a normalized confidence. `evaluate()` reports mean absolute error, and
`scripts/fit-model.mjs` turns benchmark JSON into a versioned model JSON.

## 11. Adaptive hybrid selection

### `FrameStatsReader.js`

Projection already reduces visible count, fixed-point radius sum, duplicate
count, and overflow count into four atomics. A 16-byte asynchronous copy maps
those statistics without blocking the animation callback. The most recently
completed result drives the next frame, so selection is normally one GPU frame
behind camera motion instead of forcing a CPU/GPU synchronization bubble.

### `AdaptiveSelector.js`

The first ten scheduled frames alternate:

```text
quad, tile, quad, tile, quad, tile, quad, tile, quad, tile
```

For each path, measured/predicted ratios produce a robust median correction for
that device. After calibration, corrections adapt slowly with an EWMA to power
or thermal changes. Every frame evaluates both corrected costs. An 8% hysteresis
margin prevents flicker at the boundary. Any observed tile-capacity overflow
forces the safe quad path. `?pipeline=quad|tile` bypasses selection for controlled
experiments.

### `SplatFuseRenderer.js`

This is the orchestration point and proves the two paths share scene buffers:

```text
update camera
-> reset/project/reduce shared data
-> selector chooses quad OR tile
-> encode chosen sort+raster path
-> copy 16-byte stats + resolve timestamps
-> submit once
-> asynchronously update calibration/statistics
```

No scene buffer is reloaded and no PLY data returns to the CPU when paths switch.

## 11A. CUDA training application

`trainer/train.py` is now a complete COLMAP-to-PLY training application. It
discovers the Phase-0 reconstruction (preferring the largest model exported
directly into `sparse/`), parses TXT without extra bindings or binary through
`pycolmap`, validates registered images, corrects supported lens distortion,
and scales each intrinsic matrix with its prepared image.

Sparse points initialize world-space means, nearest-neighbor log-scales,
identity WXYZ quaternions, opacity logits, and spherical-harmonic coefficients.
The per-view loop calls the pinned `gsplat==1.5.3` differentiable CUDA
rasterizer, minimizes `0.8 L1 + 0.2 (1 - SSIM)`, steps named Adam optimizers,
decays the means learning rate, and performs gradient-driven clone/split plus
opacity/size pruning. A local workaround restores the opacity reset that
gsplat 1.5.3 skips because of its published operator-precedence typo.

Lifecycle behavior includes deterministic train/validation splitting, bounded
image caching, SH-degree scheduling, non-finite guards, a conservative Gaussian
count ceiling, validation PSNR/SSIM, preview images, JSONL metrics, atomic PLY
and checkpoint writes, safe-boundary SIGINT/SIGTERM checkpoints, full
optimizer/strategy and RNG resume state, dataset fingerprinting, and optional
publication into `viewer/public/scenes/`. CUDA OOM never serializes potentially
mixed in-flight state; recovery uses the newest prior completed checkpoint.

The CUDA boundary is deliberately explicit: training uses gsplat's maintained
forward/backward kernels. It does **not** use `renderer-cuda`, because that
target has no implemented backward kernel or PyTorch extension. CPU-safe
parser/math/lifecycle tests run on this computer; the full kernel path still
requires the planned NVIDIA acceptance run.

## 12. Tests and verification

From `viewer/`:

```powershell
npm test
npm run build
npm run smoke:webgpu
```

From the repository root, trainer checks that do not require NVIDIA are:

```powershell
python trainer/train.py --doctor
python trainer/train.py --self-test
python -m unittest trainer.test_train -v
```

The Node tests cover aligned packing, power-of-two capacities, bitonic stage and
key order, benchmark Cartesian control, deterministic synthetic data, CSV
capture, empirical fit reconstruction, signed boundaries, ten-frame alternation,
and overflow fallback. The smoke test starts Vite and runs the quad, tile, and
adaptive modes in a headless Chrome WebGPU context while rejecting validation,
shader, pipeline, bind-group, or device-loss errors.

Browser/GPU verification should use:

```text
?synthetic=50000&pipeline=quad
?synthetic=50000&pipeline=tile
?synthetic=50000&pipeline=auto
```

Inspect the HUD and browser console for validation errors, then repeat on each
target hardware tier. Timestamp availability must be recorded with every row.

## 13. Known limits and honest research notes

1. **CUDA/HIP was not run here.** The gsplat-backed trainer is implemented but
   its full raster/backward path still needs the documented NVIDIA acceptance
   run. The separate `renderer-cuda` blueprint remains a forward-only
   algorithmic reference and is not the trainer backend.
2. **Bitonic is the portable baseline, not the final throughput winner.** It is
   fully GPU-side and fairer than the old CPU worker, but its `O(M log²M)` cost is
   expected to be visible at multi-million capacity. A radix implementation is a
   clear follow-on experiment.
3. **Tile storage is deliberately bounded.** `maxDup` is a research/control knob.
   Overflow is measured and causes adaptive fallback; increase it only when the
   adapter has enough storage.
4. **Very large scenes remain adapter-limited.** A 5M scene needs about 320 MB for
   each 64-byte scene/projected binding. Hardware exposing a lower WebGPU storage
   binding limit needs chunked projection, which is outside this ten-stage pass.
5. **Per-frame selection uses completed prior-frame statistics.** This is the
   practical no-stall design; an instantaneous same-command-buffer branch is not
   available in core WebGPU.
6. **Timestamp security policy varies by browser.** Unsupported adapters still
   run, but publication data should use explicit GPU timestamps and store browser,
   driver, adapter, power state, and timestamp source beside each sample.
7. **The default analytical coefficients are priors.** Defensible crossover plots
   require captured benchmark rows per device, model fitting, held-out error, and
   confidence intervals.

## 14. Publication workflow

1. Fix browser version, power mode, and scene seed.
2. Run at least three warmups and ten timestamped samples per workload/path.
3. Save adapter metadata, timestamp source, all independent variables, pass map,
   total time, and overflow count.
4. Fit one device/profile at a time with `npm run model:fit`.
5. Hold out workloads and report MAE, not only training fit.
6. Plot `F=0` slices over `(N,r)`, `(D,k)`, and resolution.
7. Compare adaptive choices against an oracle that takes the measured minimum.
8. Report calibration overhead and selector regret in addition to average FPS.

That workflow separates an implemented renderer from a defensible research
claim: the code supplies both pipelines and measurement machinery, while the
cross-device experiments supply the evidence.
