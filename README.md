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
| Training | `trainer/` | COLMAP-to-PLY CUDA trainer using differentiable gsplat |
| Full implementation notes | `IMPLEMENTATION.md` | Ten stages, data layouts, pass-by-pass explanation |

## Train a scene

Training requires an NVIDIA CUDA computer. This AMD development PC can validate
the trainer and a dataset without starting a GPU run:

```powershell
python trainer/train.py --doctor
python trainer/train.py --self-test
python -m unittest trainer.test_train -v
```

On the NVIDIA computer, install the pinned stack described in
`trainer/README.md`, then run:

```bash
python trainer/train.py --data /data/scene \
  --output-dir trainer/runs/scene \
  --viewer-ply viewer/public/scenes/scene.ply \
  --iterations 30000 --downscale 2
```

The trainer uses gsplat's differentiable CUDA rasterizer/backward pass. The
repository's educational `renderer-cuda/` target is still a standalone,
unvalidated forward prototype; its `backward.cu` and PyTorch extension are not
implemented. CUDA and HIP therefore remain unvalidated on this machine. See
`IMPLEMENTATION.md` and `trainer/README.md` for the exact tested boundary.
