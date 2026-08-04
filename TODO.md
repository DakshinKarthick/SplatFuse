# TODO — GaussianSplat build plan

Work top-to-bottom. Each task lists **what to build** and the **exact study reference**
(course + topic/video). References map to your `Skill.txt` archive:

- **DL** = Deep Learning: Beginner to Advanced `[DONE]`
- **TJ** = Three.js Journey (lesson numbers) — Basics/Classic/Advanced/Shaders `[DONE]`, Extra/R3F `[TODO]`
- **CUDA** = freeCodeCamp CUDA Course, Elliot Arledge (single YouTube video; chapter timestamps in Skill.txt)
- **EXT** = not covered by your courses → external resource named inline

Legend: ☐ not started · ◐ in progress · ☑ done

---

## Phase 0 — Capture a scene & run the reference tool  *(a weekend)*
Goal: see the whole pipeline end-to-end before touching internals.

- ☐ Capture 30–100 heavily-overlapping photos of a texture-rich subject (plant, cluttered desk, statue).
- ☐ Install a reference implementation and train once → get a finished `.ply`.
  - Recommended: **nerfstudio**, splat method **`splatfacto`** (friendlier than the original INRIA repo).
  - **EXT — no course covers this.** Refer to: nerfstudio docs (`docs.nerf.studio`, "splatfacto") and the COLMAP docs.
- ☐ Understand what COLMAP does: **structure-from-motion** → camera pose per photo + a sparse 3D point cloud.
  - **EXT.** COLMAP tutorial docs. (Concept only — you consume its output in later phases.)
- ☐ Open your trained scene in the reference viewer and fly around it.

**Milestone:** you know exactly what goes in (photos) and comes out (`.ply` of blobs) before writing any internals.

---

## Phase 1 — Your own Three.js viewer  *(your home turf)*
Goal: render a `.ply`/`.splat` yourself, no off-the-shelf splat library. Everything here is already `[DONE]` in your TJ course.

- ☐ **Parse the file** (`viewer/src/SplatLoader.js`): each blob = position (xyz), scale, rotation (quaternion), color, opacity. Read into typed arrays.
  - **EXT** for the exact 3DGS `.ply` field layout (incl. spherical-harmonics color) — see the INRIA "3D Gaussian Splatting" paper / `.ply` header.
  - **TJ 47 — Intro and loading progress** `[TODO]` for a load bar while parsing.
- ☐ **Draw each blob as a camera-facing billboard** (`viewer/src/SplatMaterial.js`, `shaders/splat.vert.glsl`).
  - **TJ 17 — Particles** (billboarded points, per-particle attributes).
  - **TJ 40 — Particles Morphing** / **TJ 39 — Particles Cursor Animation** (passing big per-instance attribute buffers).
- ☐ **Fade each dot from the center** in the fragment shader so it's a soft fuzzy splat, not a hard square (`shaders/splat.frag.glsl`).
  - **TJ 27 — Shaders** and **TJ 28 — Shader patterns** (radial gradient / distance-to-center).
  - **TJ 31 — Modified materials** (injecting shader code / custom blending).
- ☐ **Depth-sort blobs far→near every camera move**, in a Web Worker so the page never freezes (`viewer/src/sortWorker.js`).
  - **TJ 46 — Performance tips** `[TODO]` — do this lesson; it covers moving heavy work off the main thread.
  - Later GPU-sort upgrade: **TJ 41 — GPGPU Flow Field Particles** (compute on the GPU).
- ☐ **Organize the viewer** as it grows: **TJ 26 — Code structuring for bigger projects**.

**Milestone:** your own viewer flying through your captured scene. Already an interview-worthy demo.

---

## Phase 2 — Your own CUDA renderer  *(the impressive core)*
Goal: rebuild the GPU rasterizer the reference tool hides. **Not covered by TJ/DL — this is the CUDA course.**

> **GPU route: CLOUD NVIDIA.** This machine has an AMD RX 7600 (gfx1102). CUDA can't run
> on it, and **ROCm-in-WSL was tried and confirmed impossible** for this card (2026-07-03):
> the WSL ROCm runtime installs and loads but never enumerates gfx1102 as a GPU agent —
> `rocminfo` sees only the CPU, and `HSA_OVERRIDE_GFX_VERSION` can't fix a GPU that isn't
> listed. **Do not retry ROCm-in-WSL.** Run Phases 0/2/3 on a cloud NVIDIA GPU — see
> [`cloud/README.md`](cloud/README.md). (If you ever want local ROCm, it needs *native/dual-boot
> Linux*, not WSL.) An Ubuntu 22.04 WSL distro from the attempt is kept for general Linux use.

Prereqs (do before writing kernels — on the **cloud box**, not locally):
- ☐ **CUDA Ch 2 — CUDA Setup** (toolkit, drivers).
- ☐ **CUDA Ch 3 — C/C++ Review** (pointers, memory, structs).
- ☐ **CUDA Ch 4 — Gentle Intro to GPUs** (CPU vs GPU, why it parallelizes).

Forward pass (3D blobs → 2D image), each step its own kernel:
- ☐ Project each 3D Gaussian to 2D screen space.
- ☐ Split the image into 16×16 tiles; bin which blobs land in which tile.
- ☐ Sort blobs by depth within each tile.
- ☐ For each pixel, alpha-blend its tile's blobs front-to-back.
  - **CUDA Ch 5 — Writing Your First Kernels** (threads/blocks/grids, the full memory model — this is the core hands-on chapter).
  - **CUDA Ch 6 — CUDA API** (memory management, kernel launch, error handling).

Backward pass (what lets the scene learn):
- ☐ For every blob, compute how a tiny change to its position/scale/color/opacity changes the final image → the gradients.
  - Gradient/backprop intuition: **DL — Training through Backpropagation** and **DL — Autograd in PyTorch**.

Optimization (after it works):
- ☐ Shared-memory tiling + a faster GPU sort.
  - **CUDA Ch 7 — Optimizing Matrix Multiplication** (cache-tiled / shared-memory technique — applies directly to the tile rasterizer).

**Milestone:** a from-scratch splat renderer. Forward-only is already strong; forward+backward is the standout.

---

## Phase 3 — Train it & go deeper  *(stretch)*
Goal: the full loop — render → compare to photo → nudge every blob to shrink the error → densify/prune.

- ☐ Loss = difference between render and the real photo; backprop to blob params.
  - **DL — Gradient Descent (Theoretical Foundation, PyTorch Implementation)**; **DL — Model Optimization**.
- ☐ Optimize with **Adam**.
  - **DL — Adam Optimizer** and **DL — Gradient Descent with Momentum**.
- ☐ Drive it from PyTorch tensors / the training loop.
  - **DL — PyTorch Tensor Basics**, **DL — Autograd in PyTorch**.
- ☐ Wire your CUDA forward/backward into PyTorch as a custom op.
  - **CUDA Ch 9 — PyTorch Extensions (CUDA)**.
- ☐ **Densify & prune**: clone blobs where the scene is still blurry, delete useless/transparent ones.
  - **EXT** — INRIA 3DGS paper, "Adaptive Density Control" section (no course covers this specific heuristic).
- ☐ Pick ONE direction to go deep: faster (shared-mem tiling, better sort) · bigger scenes (streaming, LOD) · a feature (object cut-out/edit, compression).

**Milestone:** your own end-to-end trainer — you've rebuilt Gaussian splatting from scratch.

---

## Coverage gaps (things your courses don't teach — plan to use EXT)
- COLMAP / structure-from-motion (Phase 0) — external docs.
- nerfstudio / splatfacto (Phase 0) — external docs.
- 3DGS `.ply` field layout + spherical-harmonics color (Phase 1) — INRIA paper / file header.
- Densify-and-prune heuristic (Phase 3) — INRIA paper.
- Everything CUDA (Phase 2–3) — the freeCodeCamp CUDA course you already queued in `Skill.txt` (still `[TODO]` — start it before Phase 2).
