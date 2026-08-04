# Phase 0 — Capture a scene & run the reference tool

Goal: run the whole pipeline end-to-end with an existing tool, so you know what goes
in (a video / photos) and what comes out (camera poses + a sparse point cloud) before
writing any internals.

This folder holds the **fully-automated COLMAP camera-tracking workflow** from
[Polyfjord's tutorial](https://www.youtube.com/watch?v=xx85eyN1Xc0): drop a video in a
folder, double-click one `.bat`, and out comes a COLMAP sparse reconstruction (one
camera per frame + a 3D point cloud) that you import straight into Blender. That COLMAP
output — camera intrinsics/extrinsics + sparse cloud — is exactly the **structure-from-motion
input** a Gaussian-splat trainer consumes later, so this doubles as the SfM half of Phase 0.

---

## What runs where (this machine = AMD RX 7600)

COLMAP ships in two Windows builds. The **CUDA** build does GPU SIFT on NVIDIA only; the
**no-CUDA** build does SIFT on the CPU / OpenGL, which works on **any** GPU including AMD.
Since this box is an AMD RX 7600, the installed build here is **`colmap-x64-windows-nocuda`
3.12.3**, and it's been verified to run locally. Reconstruction (the `mapper`) is CPU-bound
in both builds. On an NVIDIA machine, swap in the `-cuda` build for much faster feature
extraction (see recreation commands). Unlike Gaussian *training* (Phases 2/3, which is
CUDA-only → cloud), this SfM step runs fine locally on AMD.

---

## Folder layout

The `.bat` hard-codes these sibling folder names (it resolves the parent of its own
folder, then looks for `01 COLMAP`, `02 VIDEOS`, `03 FFMPEG`, `04 SCENES`). Keep the
names **exactly** as below.

```
phase0-capture/
├── 01 COLMAP/     ← COLMAP 3.12.3 no-CUDA binaries        (installed; gitignored — large)
├── 02 VIDEOS/     ← YOU: drop your .mp4/.mov here          (gitignored)
├── 03 FFMPEG/     ← YOU: ffmpeg.exe or bin\ffmpeg.exe here (gitignored)
├── 04 SCENES/     ← OUTPUT: one sub-folder per video       (gitignored)
├── 05 SCRIPTS/
│   ├── batch_reconstruct.bat        ← Polyfjord's script (+ Qt/CPU-match fixes); extracts EVERY frame
│   └── batch_reconstruct_fast.bat   ← RECOMMENDED: decimates to N fps (knob at top), prints the best model
├── blender-addon/
│   └── photogrammetry_importer.zip  ← SBCV importer, install into Blender as-is
└── README.md
```

Only the two folders with source-controllable content (`05 SCRIPTS`, `blender-addon`)
are committed. `01/02/03/04` are gitignored (COLMAP binaries are large; videos/output are
yours) — **recreate them with the commands below**.

---

## What's already installed (done for you)

- **`01 COLMAP/`** — COLMAP 3.12.3 no-CUDA, extracted (`01 COLMAP\bin\colmap.exe`).
- **`05 SCRIPTS/batch_reconstruct.bat`** — Polyfjord's batch reconstruction script, verbatim.
- **`blender-addon/photogrammetry_importer.zip`** — SBCV Blender importer `v2026.02.16`.

## What you still provide

- **A video** in `02 VIDEOS/` (turn OFF image stabilization when filming; enable lens
  correction — see the tutorial's closing tips).
- **FFmpeg** (static build) in `03 FFMPEG/` — `ffmpeg.exe` at the root or under `bin\`.
- **Blender** — install the addon zip (Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk,
  pick the zip **without unzipping it**). Use the **OpenGL** backend, not Vulkan.

---

## How to run

1. Put a video in `02 VIDEOS/` and FFmpeg in `03 FFMPEG/`.
2. Double-click **`05 SCRIPTS/batch_reconstruct_fast.bat`** (Windows may warn about an
   untrusted script — allow it). It extracts frames at `FPS` fps (knob at the top of the
   file, default **6**), runs COLMAP feature extraction → sequential matching → sparse
   mapping, writes everything to `04 SCENES/<video-name>/`, and **prints the path of the
   best sub-model to import**. Already-processed videos are skipped on re-runs.
   *(The original `batch_reconstruct.bat` is kept alongside — same pipeline but extracts
   every frame, so matching is far slower on long clips.)*
3. Output per video: `04 SCENES/<name>/{images/, sparse/, database.db}` plus a TXT export.
4. In Blender: **File ▸ Import ▸ COLMAP model/workspace**, point at
   `04 SCENES/<name>/`, tick **Suppress distortion warnings**, import. You get one camera
   per frame + an animated fly-through camera + the point cloud. (Tick **Add points as
   mesh object** if you want editable geometry instead of just a point cloud.)

> **Re-running a failed scene:** the script skips any video whose `04 SCENES/<name>/`
> folder already exists. If a run failed partway, **delete that scene folder** before
> re-running, or it'll be skipped.

---

## Two fixes applied to Polyfjord's script for this machine

The upstream script assumes an NVIDIA box; two edits (both commented `(LOCAL EDIT)`
in `batch_reconstruct.bat`) make it work here. Both failures are *silent-ish* — COLMAP
prints an error but the frame-extract step still succeeds, so you only find out when the
`mapper` reports **"No images with matches found in the database."**

1. **`QT_PLUGIN_PATH`** — COLMAP 3.12.3 keeps its Qt platform plugin (`qwindows.dll`)
   in `01 COLMAP\plugins`. Without pointing Qt there, `feature_extractor` and
   `sequential_matcher` die with *"Could not find the Qt platform plugin windows"* and
   write nothing. The script now sets `QT_PLUGIN_PATH`, mirroring the official `COLMAP.bat`.
2. **`--SiftMatching.use_gpu 0` (CPU matching)** — this box has a **Parsec Virtual Display
   Adapter**, and COLMAP's OpenGL SIFT *matcher* binds to it and crashes with *"Not enough
   GPU memory to match N features"*. Forcing CPU matching is slower but reliable. (Feature
   *extraction* stays on the GPU — it works fine.) On a healthy NVIDIA machine you can drop
   this flag for GPU-speed matching.

Verified end-to-end on this machine (405-frame decimated drone clip): GPU extraction ≈ 0.4 min,
CPU matching ≈ 10 min (the long pole), mapper a few min → a 140-image / 43k-point model.

---

## Gotchas seen in practice

- **Import fails with "Invalid colmap model / workspace":** the `mapper` never produced a
  model — `04 SCENES/<name>/sparse/` is empty or has no numbered sub-folder. Check the
  terminal for an earlier COLMAP error (usually the two above) and re-run.
- **Fast/forward-moving footage fragments into several models** (`sparse/0`, `sparse/1`, …).
  COLMAP makes a *separate* model per chunk it can't connect — normal for FPV/drone flights
  that never revisit an area. `batch_reconstruct_fast.bat` auto-picks the biggest chunk and
  exports it as TXT into `sparse/` root, so you just import the **scene folder**. To inspect
  a *different* segment instead, import that segment's folder directly (e.g. `…\sparse\1`)
  in **COLMAP model** mode and set the image path to the scene's `images\`.
  Fewer, slower, overlap-rich shots → one connected model.
- **"Invalid colmap model / workspace" even though `sparse/0` exists:** the SBCV importer's
  *workspace* mode looks for `cameras/images/points3D` **directly in `sparse/`**, not in
  `sparse/0`. Either import `…\sparse\0` directly (model mode, set image path), or run
  `colmap model_converter --input_path …\sparse\0 --output_path …\sparse --output_type TXT`
  to drop the files into `sparse/` (the fast script does this for you).
- **CPU matching is slow on long sequences.** ~2000 frames ≈ 30 min+ just to match. Extract
  fewer frames (see the ffmpeg `-r` tip above) — a few hundred is plenty for a good track.

---

## Recreate on another PC

Everything gitignored (COLMAP binaries) is re-downloadable. Run from **PowerShell** with
this folder as the working directory. This mirrors exactly what was done here.

```powershell
# --- folder structure + Polyfjord batch script -------------------------------
$dirs = '01 COLMAP','02 VIDEOS','03 FFMPEG','04 SCENES','05 SCRIPTS','blender-addon'
$dirs | ForEach-Object { New-Item -ItemType Directory -Force $_ | Out-Null }

Invoke-WebRequest `
  'https://gist.githubusercontent.com/polyfjord/4ed7e8988bdb9674145f1c270440200d/raw' `
  -OutFile '05 SCRIPTS\batch_reconstruct.bat'

# --- COLMAP 3.12.3 -----------------------------------------------------------
# AMD / no NVIDIA GPU  → no-CUDA build (installed on this machine):
Invoke-WebRequest `
  'https://github.com/colmap/colmap/releases/download/3.12.3/colmap-x64-windows-nocuda.zip' `
  -OutFile 'colmap.zip'
# NVIDIA GPU → use the CUDA build instead (comment out the line above, use this):
# Invoke-WebRequest `
#   'https://github.com/colmap/colmap/releases/download/3.12.3/colmap-x64-windows-cuda.zip' `
#   -OutFile 'colmap.zip'
Expand-Archive 'colmap.zip' -DestinationPath '01 COLMAP' -Force   # → 01 COLMAP\bin\colmap.exe
Remove-Item 'colmap.zip'

# --- SBCV Blender photogrammetry importer ------------------------------------
Invoke-WebRequest `
  'https://github.com/SBCV/Blender-Addon-Photogrammetry-Importer/releases/download/v2026.02.16/photogrammetry_importer.zip' `
  -OutFile 'blender-addon\photogrammetry_importer.zip'

# --- sanity check ------------------------------------------------------------
& '01 COLMAP\bin\colmap.exe' --help    # should print: COLMAP 3.12.3 …
```

Then supply your own video (`02 VIDEOS/`) and FFmpeg (`03 FFMPEG/`), install the Blender
addon, and run the `.bat` as above.

---

## Sources

- COLMAP 3.12.3 — <https://github.com/colmap/colmap/releases/tag/3.12.3>
- Batch script — <https://gist.github.com/polyfjord/4ed7e8988bdb9674145f1c270440200d>
- Blender importer — <https://github.com/SBCV/Blender-Addon-Photogrammetry-Importer>
- Tutorial — <https://www.youtube.com/watch?v=xx85eyN1Xc0>

## Later: the Gaussian-splat path

The above stops at COLMAP SfM (poses + sparse cloud) for camera tracking. To get a trained
`.ply` **of Gaussians** for the Phase 1 viewer, feed COLMAP's output into nerfstudio's
`splatfacto` (`docs.nerf.studio`) — that training step is CUDA-only, so it runs on a cloud
NVIDIA GPU (see [`../cloud/`](../cloud/)), not on this AMD box. See [`../TODO.md`](../TODO.md).
