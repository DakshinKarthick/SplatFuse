# SplatFuse — Adaptive WebGPU Gaussian Splatting

SplatFuse shares one GPU scene representation between a globally sorted
hardware-quad pipeline and a CUDA-style compute-tile pipeline. It measures both
paths and selects the faster one for the current device and frame.

The original capture, training, WebGL, and CUDA reference work is retained as the
foundation. Active WebGPU development lives in `viewer/src/webgpu/`; the complete
ten-stage implementation record lives in `IMPLEMENTATION.md`.

Photos in → 3D scene out. Take ~30–100 overlapping photos of a scene, solve where
each camera was, fill the space with thousands of fuzzy colored blobs (3D Gaussians),
and tweak every blob with gradient descent until the blobs — seen from your photo
angles — match your photos. The result is a photoreal 3D scene you can fly through in
real time. No mesh, no modeling.

> This is a **learning build**. Each phase is done by hand so you understand the
> algorithm, not just a library. The scaffold is stubs only — the code is yours to write.

## The four phases

| Phase | Folder | Language | Status |
|-------|--------|----------|--------|
| 0 · Capture a scene + run the reference tool | [`phase0-capture/`](phase0-capture/) | Python (COLMAP + nerfstudio) | ☐ |
| 1 · Your own Three.js viewer | [`viewer/`](viewer/) | JS + vanilla Three.js + Vite | ☐ |
| 2 · Your own CUDA renderer (the core) | [`renderer-cuda/`](renderer-cuda/) | C++ / CUDA | ☐ |
| 3 · Train it end-to-end | [`trainer/`](trainer/) | Python + PyTorch | ☐ |

See [`TODO.md`](TODO.md) for the task breakdown **and the exact course topic / video to
watch for each step**.

## GPU note (this machine = AMD RX 7600)

CUDA is NVIDIA-only and **won't run locally**, and **ROCm-in-WSL is confirmed impossible**
for the RX 7600 (the WSL runtime never enumerates the GPU). So: **Phase 1 runs locally**,
and **Phases 0/2/3 run on a cloud NVIDIA GPU** — see [`cloud/`](cloud/). Details and the
"do not retry ROCm-in-WSL" note are in [`TODO.md`](TODO.md).

## Why these languages (and not Java)

Gaussian splatting is inherently multi-language — that's the point of the exercise:

- **Python** for Phase 0 (COLMAP/nerfstudio) and Phase 3 (PyTorch training loop) —
  reuses your Deep Learning course directly.
- **JavaScript + vanilla Three.js** for Phase 1 — your home turf, already `[DONE]` in
  Three.js Journey. Vanilla (not React/R3F) keeps the depth-sort + custom shaders clean.
- **C++/CUDA** for Phase 2 — the impressive core. (Plain C won't do: CUDA is C++.)
- **Java** has no place in this domain — skip it.

## Directory layout

```
GaussianSplat/
├── README.md
├── TODO.md                  ← start here: tasks + exact study references
├── data/                    ← your captured photos + COLMAP output   (gitignored)
├── scenes/                  ← trained .ply / .splat outputs           (gitignored)
├── phase0-capture/          ← Phase 0: capture notes + reference-tool scripts
├── viewer/                  ← Phase 1: your Three.js splat viewer
│   └── src/
│       ├── main.js
│       ├── SplatLoader.js       ← parse .ply/.splat into typed arrays
│       ├── SplatMaterial.js     ← billboard + soft-dot fragment shader
│       ├── sortWorker.js        ← depth sort off the main thread
│       └── shaders/*.glsl
├── renderer-cuda/           ← Phase 2: from-scratch forward + backward passes
│   └── src/{main.cpp, forward.cu, backward.cu}
└── trainer/                 ← Phase 3: render → compare → nudge → densify/prune
    └── train.py
```

## How to run (once you've filled in the stubs)

Each phase runs independently — see the README in each folder. Quick reference:

- **Phase 1 viewer:** `cd viewer && npm install && npm run dev` → open the printed localhost URL.
- **Phase 2 renderer:** `cd renderer-cuda && cmake -B build && cmake --build build` (needs CUDA Toolkit + a CMake generator).
- **Phase 3 trainer:** `cd trainer && python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && python train.py`.

## README tip for later

Lead the final README with a short **GIF of you flying through your own captured scene**,
then the one-liner "photos in → 3D scene out," then the architecture. Recruiters watch
the GIF before they read anything.
