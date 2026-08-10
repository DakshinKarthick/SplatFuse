# SplatFuse trainer

`train.py` is the end-to-end 3D Gaussian Splatting training application. It
loads the Phase-0 COLMAP reconstruction, initializes Gaussians from sparse
points, uses the differentiable `gsplat` CUDA rasterizer, optimizes L1 + SSIM,
performs adaptive clone/split/prune density control, and exports a binary PLY
that the SplatFuse WebGPU viewer can load.

The custom code under `../renderer-cuda/` is **not** the training backend. Its
forward path is an unvalidated prototype, `backward.cu` is still a stub, and no
PyTorch binding exists. The `gsplat` 1.5.3 public release supplies the
production CUDA forward/backward boundary; this folder owns the dataset,
optimization, checkpointing, evaluation, and export lifecycle. Wheel versions
with a PEP 440 local build suffix, such as `1.5.3+pt24cu124`, use that same
public release and are accepted.

## What works on each computer

| Computer | Commands | Purpose |
|---|---|---|
| This AMD/CPU PC | `--doctor`, `--inspect-data`, `--self-test`, unit tests | Validate code/data without training |
| NVIDIA CUDA PC | all commands, including training | Full differentiable training |
| Browser with WebGPU | `npm run dev` in `viewer/` | Display the exported PLY |

Real training fails early when CUDA is unavailable. There is no silent CPU
fallback that would take days or run out of memory.

## NVIDIA installation

Use Python 3.10 or 3.11 and a CUDA 12.x development image. The following stack
matches the project's historically proven torch/CUDA combination:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r trainer/requirements.txt

# gsplat may JIT-compile CUDA kernels on first import. Limit parallel jobs so a
# 16-32 GB cloud VM is not OOM-killed during compilation.
MAX_JOBS=3 python -c \
  "from importlib.metadata import version; from gsplat.cuda._backend import _C; print('gsplat ready:', version('gsplat'))"
python trainer/train.py --doctor
```

The doctor must report `cuda_available: true`, an NVIDIA device, and
`ready_for_cuda_training: true`. Its `gsplat` value may be exactly `1.5.3` or a
local build of that release such as `1.5.3+pt24cu124`. The explicit import above
warms/compiles the backend with bounded build parallelism; the doctor then
independently checks dependency versions, CUDA visibility, and backend import.
Runtime readiness does not require `nvcc` after a prebuilt backend imports;
`cuda_development_tools_ready` reports `nvcc` plus `nvidia-smi` separately. The
reproducible cloud bootstrap deliberately requires both development tools.

## Dataset contract

Pass the scene root created by `phase0-capture`:

```text
scene/
├── images/
│   ├── frame_000001.jpg
│   └── ...
└── sparse/
    ├── cameras.txt or cameras.bin
    ├── images.txt or images.bin
    ├── points3D.txt or points3D.bin
    └── 0/, 1/, ...                 # optional raw disconnected models
```

The recommended capture script exports the largest numbered reconstruction
directly into `sparse/`. The trainer deliberately prefers those direct files;
if they are absent, it chooses the numbered model with the largest declared
point count (then registered-image count) and logs a warning. Use
`--colmap-model` to override discovery. TXT and BIN components are never mixed:
a complete BIN set wins, otherwise a complete TXT set is used consistently for
both loading and the resume fingerprint.

Text models work without `pycolmap`. Binary models use the pinned official
`pycolmap` binding. Registered image names may contain spaces or Windows
backslashes. Missing images, path traversal, invalid poses, and incomplete
models fail before GPU allocation.

COLMAP corner-coordinate intrinsics are scaled directly with the raster. The
trainer bridges the half-pixel convention only while calling OpenCV, and it
refuses aspect-ratio-changing crops/rotations. It also rejects georeferenced
world coordinates whose float32 spacing is too coarse for the reconstructed
scene; recenter such a model before CUDA training.

Supported camera models are `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`,
`RADIAL`, `OPENCV`, `FULL_OPENCV`, `SIMPLE_RADIAL_FISHEYE`,
`RADIAL_FISHEYE`, and `OPENCV_FISHEYE`. OpenCV creates a crop-safe undistorted
pinhole image and a matching intrinsic matrix before training. For another
camera model, first run COLMAP's `image_undistorter`.

## CPU-safe checks on this PC

```powershell
python trainer/train.py --doctor
python trainer/train.py --self-test
python trainer/train.py --data "D:\scene" --inspect-data --downscale 2
python -m unittest trainer.test_train -v
```

`--inspect-data` fingerprints the model and every registered image, decodes and
camera-corrects every frame, and reports the first prepared image and K matrix.

## Train

From the repository root on the NVIDIA machine:

```bash
python trainer/train.py \
  --data /data/my-scene \
  --output-dir trainer/runs/my-scene \
  --viewer-ply viewer/public/scenes/my-scene.ply \
  --iterations 30000 \
  --downscale 2
```

For a short integration run, use `--iterations 100 --refine-stop 0` after a
successful doctor. For the earlier 7,000-step cloud budget, use
`--iterations 7000 --refine-stop 7000`.

Important controls:

| Flag | Meaning |
|---|---|
| `--max-resolution` | Caps the longest prepared image side (default 1600) |
| `--downscale` | Integer resolution reduction after camera correction |
| `--sh-degree 0..3` | Maximum view-dependent color degree |
| `--background` | Random, white, or black training background |
| `--max-initial-points` | Bounded sparse-cloud initialization |
| `--max-gaussians` | Conservative growth guard and post-refinement ceiling |
| `--refine-*` | Density-control start/stop/frequency |
| `--checkpoint/eval/preview/export-every` | Artifact intervals; zero disables optional intervals |
| `--viewer-ply` | Publishes the final PLY directly into Vite's public scene folder |

Degree 0 exactly matches the in-repository viewers, which currently read only
`f_dc_*`. Degree 3 trains standard view-dependent 3DGS color and exports
`f_rest_*`, but the current WebGPU and custom CUDA viewers ignore those higher
bands. The PLY remains loadable; it simply appears with DC/base color there.

## Artifacts

```text
trainer/runs/my-scene/
├── config.json                 # effective config plus resume provenance
├── environment.json            # Python, packages, CUDA, GPU, command
├── metrics.jsonl               # per-step train + periodic validation data
├── resume-history.jsonl        # every continuation and metrics repair
├── summary.json                # successful completion summary
├── final.ply                   # viewer-compatible raw 3DGS parameters
├── checkpoints/
│   ├── step-001000.pt
│   ├── interrupted.pt          # safe boundary after first SIGINT/SIGTERM
│   └── final.pt
├── previews/                   # target | render PNG pairs
├── exports/                    # periodic PLY snapshots
└── history/                    # archived summaries when extending a run

trainer/runs/.my-scene.splatfuse-run.lock  # sibling lock, present only while owned
```

JSON, PLY, PNG, and checkpoint replacements use temporary sibling files and
atomic rename. `--overwrite` preserves the selected prior output in a
timestamped `.backup-*` directory. Outputs that overlap scene inputs or
protected project/root/home locations are rejected. A resume may append to its
own run; a branched resume into another output directory requires that
directory to be empty. The exclusive sibling `.splatfuse-run.lock` is acquired
before creation or overwrite, so two processes cannot move/write the same
artifact set; after a hard crash, verify that no trainer owns the directory
before removing a stale lock.

## Resume

```bash
python trainer/train.py \
  --data /data/my-scene \
  --resume trainer/runs/my-scene/checkpoints/step-005000.pt \
  --iterations 30000
```

Checkpoints contain dynamic Gaussian tensors, every Adam moment, scheduler and
density-strategy state, all random-number streams, completed step, split IDs,
configuration, and the scene fingerprint. The fingerprint covers the complete
COLMAP model and registered image bytes. A mismatch is rejected unless
`--allow-data-mismatch` is explicitly supplied.

Trajectory-defining values (learning rates, loss, SH, render mode, density
policy, split, and seed) are restored from the checkpoint automatically, so a
plain resume cannot silently fall back to parser defaults. Resource/output
controls such as `--downscale`, `--max-resolution`, `--max-gaussians`,
`--iterations`, and artifact intervals change only when that flag is explicitly
present on the resume command. If `--iterations` is extended, it changes the
stop step; the restored scheduler keeps its original saved gamma rather than
replanning the historical decay curve. `config.json` and
`resume-history.jsonl` record restored fields and explicit overrides.

Treat checkpoints as trusted local artifacts. They use PyTorch serialization to
preserve optimizer and Python/NumPy random state; do not pass an untrusted or
downloaded `.pt` file to `--resume` without verifying its source.

Only a fully completed optimization step is eligible for a checkpoint. On the
first `Ctrl+C`/SIGINT or SIGTERM, the trainer requests shutdown, finishes the
current iteration, and writes `interrupted.pt` at that safe boundary. A second
termination signal exits immediately; it does not claim that the in-flight
state is resumable. This two-stage behavior lets cloud shutdown hooks preserve
valid state without making an unresponsive process impossible to stop.

A CUDA out-of-memory exception can occur halfway through an optimizer or
density-control mutation, so the trainer deliberately does **not** serialize an
`oom.pt` from that potentially mixed state. Resume the newest earlier
`step-*.pt` checkpoint instead, then lower `--downscale`, `--max-resolution`, or
`--max-gaussians` as appropriate. If OOM recovery time matters, choose a shorter
`--checkpoint-every` interval and keep the run directory on persistent storage.
When a lower Gaussian cap is requested, the checkpoint and Adam/density arrays
are validated and pruned on CPU before allocating the restored CUDA tensors.

Resume is state-complete at the saved boundary: model tensors, optimizer and
strategy state, step number, and random streams are restored. It is not a
promise of bit-for-bit identity with an uninterrupted run. CUDA rasterization
uses parallel atomic operations whose accumulation order may differ across
runs, drivers, or GPUs, so small numerical divergence is expected.

## Training workflow

For each iteration:

1. deterministically sample a registered training frame;
2. load/cache its corrected RGB image and K matrix;
3. activate log-scales with `exp`, opacity logits with `sigmoid`, and progressively enable SH bands;
4. run gsplat projection, tile rasterization, and alpha compositing;
5. retain screen-mean gradients for density statistics;
6. minimize `0.8 × L1 + 0.2 × (1 - SSIM)` plus optional regularizers;
7. reject non-finite loss/gradients, then step six named Adam optimizers;
8. decay the position learning rate;
9. clone/split high-gradient splats and prune low-opacity/oversized splats;
10. enforce the Gaussian safety limit and write scheduled artifacts.

The `gsplat` 1.5.3 public release contains a precedence typo that prevents its
scheduled opacity reset. `train.py` explicitly applies the missing reset at the
configured interval and logs each workaround invocation, including for local
builds of that release.

## NVIDIA acceptance checklist

This repository cannot execute these checks on the AMD development PC. Before
calling a target machine production-ready, run:

1. doctor and first-time gsplat compilation;
2. the 100-step real-COLMAP smoke run;
3. send one SIGINT/SIGTERM, verify `interrupted.pt` names a completed step,
   then resume and check that the next loss is finite;
4. load `final.ply` through `viewer/`;
5. inspect previews and validation PSNR/SSIM;
6. force one controlled OOM, verify no `oom.pt` is created, and resume the newest prior scheduled checkpoint;
7. run a 7,000- or 30,000-step scene without OOM;
8. record GPU, driver, torch/CUDA, peak VRAM, resolution, and final Gaussian count.

Passing local CPU tests proves data/lifecycle logic, not NVIDIA kernel execution.
