# CONTEXT — working notes for GaussianSplat

Durable, non-obvious context for anyone (human or agent) picking this up. Code structure
lives in the code; this file holds the *why*, the hardware quirks, and hard-won gotchas.

## The machine (matters a lot here)

- **GPU: AMD Radeon RX 7600** (gfx1102). CUDA is NVIDIA-only → **won't run locally**.
  ROCm-in-WSL was tried and confirmed impossible for this card (see `TODO.md`). So:
  Phase 1 (Three.js viewer) runs locally; Phases 2/3 (CUDA renderer + trainer) run on a
  **cloud NVIDIA GPU** (see `cloud/`).
- **A "Parsec Virtual Display Adapter" is also present.** This bites OpenGL apps: they may
  bind their GL context to the virtual adapter (no real VRAM) instead of the RX 7600.
  It's the root cause of the COLMAP matcher failure below.
- Platform: Windows 11, PowerShell primary. FFmpeg is the winget Gyan.dev static build.

## Phase 0 status — DONE and working locally

Phase 0 here is the **Polyfjord automated COLMAP camera-tracking workflow**
(video: youtube.com/watch?v=xx85eyN1Xc0), which produces COLMAP SfM output (camera poses +
sparse point cloud) for import into Blender. That SfM output is also the input a Gaussian
trainer would consume later. Full setup + recreation commands: `phase0-capture/README.md`.

Installed & verified in `phase0-capture/`:
- `01 COLMAP/` — COLMAP **3.12.3 no-CUDA** (not the CUDA build — see below). Gitignored (large).
- `03 FFMPEG/ffmpeg.exe` — user's static build, copied in. Gitignored (large).
- `05 SCRIPTS/batch_reconstruct.bat` — Polyfjord's original (+ our fixes).
- `05 SCRIPTS/batch_reconstruct_fast.bat` — **recommended**; fps-decimated, import-ready output.
- `blender-addon/photogrammetry_importer.zip` — SBCV importer v2026.02.16.

End-to-end proven on an FPV drone clip → 140-camera / 43k-point model imported into Blender.

## COLMAP gotchas discovered here (all fixed in the scripts)

1. **Use the no-CUDA build, not CUDA.** On AMD, the CUDA build's GPU SIFT can't run. The
   no-CUDA build does SIFT via OpenGL, which works on the RX 7600. (Mapper is CPU either way.)
2. **`QT_PLUGIN_PATH` must point at `01 COLMAP\plugins`.** COLMAP 3.12.x keeps `qwindows.dll`
   there; without it, `feature_extractor`/`sequential_matcher` die with *"Could not find the
   Qt platform plugin windows"* and silently write nothing. Mirrors the official `COLMAP.bat`.
3. **Match on CPU: `--SiftMatching.use_gpu 0`.** The OpenGL SIFT *matcher* grabs the Parsec
   virtual adapter and crashes with *"Not enough GPU memory to match N features."* Feature
   *extraction* stays on GPU (fine); only matching needs CPU. Slower but reliable.
4. **Decimate frames.** Extracting every frame (Polyfjord's default) makes CPU matching
   crawl (2000+ frames ≈ 30 min+ just to match). `batch_reconstruct_fast.bat` extracts at
   `FPS` fps (default 6) — a few hundred frames is plenty for a good track.
5. **Blender import "Invalid colmap model / workspace".** The SBCV importer's *workspace*
   check looks for `cameras/images/points3D` **directly in `sparse/`**, NOT in `sparse/0`.
   Fix: `model_converter --input_path sparse\0 --output_path sparse --output_type TXT` (the
   fast script does this automatically for the best sub-model).
6. **Forward-moving footage fragments into `sparse/0,1,2,…`** — one model per chunk COLMAP
   can't connect. Normal for FPV/drone. The fast script auto-picks the biggest (largest
   `points3D.bin`) and exports it. Slower, overlap-rich, revisiting shots → one model.

## Testing note

To run a `.bat` for a real end-to-end test, run it from **PowerShell/Explorer**, not from
the Bash tool: launching `cmd` from Git Bash pollutes PATH with Unix `find.exe`, which
hijacks the script's `find /c /v ""` and produces garbage. Also `pause` blocks headless runs.
