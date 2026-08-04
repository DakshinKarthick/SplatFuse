# SplatFuse

SplatFuse is an adaptive native-WebGPU renderer for 3D Gaussian splats. One
aligned GPU scene feeds two complete paths:

- globally depth-sorted instanced quads with hardware alpha blending;
- 16×16 compute tiles with local ranges and early-ray termination.

Per-pass timestamp telemetry and a device-calibrated crossover model select the
path expected to be faster for the current visible count, screen radius,
duplication factor, depth complexity, and resolution.

## Run

```powershell
cd viewer
npm install
npm run dev
```

The viewer loads `/scenes/mypic1.ply`; when it is absent it runs a deterministic
50k-splat synthetic scene. Useful query parameters:

- `?scene=/scenes/name.ply&flip=1`
- `?synthetic=250000`
- `?pipeline=auto`, `?pipeline=quad`, or `?pipeline=tile`
- `?maxDup=8&k=16` for tile capacity and assumed depth complexity

Run `npm test`, `npm run build`, and `npm run smoke:webgpu` from `viewer/`. The
smoke test executes quad, tile, and auto modes in Chrome. Fit a captured benchmark
JSON with `npm run model:fit -- benchmark.json model.json`.

## Project map

| Area | Location | Purpose |
|---|---|---|
| WebGPU renderer | `viewer/src/webgpu/` | Both GPU pipelines, telemetry, model, switcher |
| Web viewer | `viewer/src/main.js` | Scene load, camera, controls, live HUD |
| CUDA reference | `renderer-cuda/` | Original algorithmic forward-pass blueprint |
| Capture | `phase0-capture/` | Completed COLMAP workflow |
| Training | `trainer/` | Existing training foundation |
| Full implementation notes | `IMPLEMENTATION.md` | Ten stages, data layouts, pass-by-pass explanation |

CUDA and HIP are intentionally not validated on this machine. WebGPU is the
active implementation target; see `IMPLEMENTATION.md` for tested scope and known
limits.
