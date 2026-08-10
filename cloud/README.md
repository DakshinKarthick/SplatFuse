# Cloud NVIDIA GPU — running the CUDA phases

Your local **RX 7600 is AMD**, and ROCm-in-WSL is confirmed a dead end for it
(the WSL ROCm runtime never enumerates gfx1102 — see `../TODO.md` GPU note). So
the CUDA-dependent work (Phase 0 reference training, Phase 2 renderer, Phase 3
trainer) runs on a **rented/loaned NVIDIA GPU**. Phase 1 (the Three.js viewer)
stays 100% local on your machine.

## Which service

| Service | Cost | Best for | Notes |
|---|---|---|---|
| **Google Colab** | Free tier (T4) / Pro | Phase 0 reference (nerfstudio), quick experiments | Easiest start; sessions are ephemeral — save outputs to Drive |
| **RunPod** | ~$0.20–0.40/hr | Phase 2 CUDA dev (nvcc, CMake), longer sessions | Persistent volume; SSH + VS Code Remote |
| **vast.ai** | ~$0.15–0.35/hr | Cheapest CUDA dev | Marketplace pricing; pick a recent CUDA image |
| **Lambda Cloud** | ~$0.50+/hr | Bigger training runs | Clean CUDA images |

Start with **Colab free** for Phase 0 (does it even train?), then move to **RunPod/vast.ai**
for the Phase 2 CUDA renderer where you need `nvcc` + a persistent dev box.

## Phase 0 on Colab (reference splat, fastest path to a .ply)

Ready-to-run notebook: [`phase0_nerfstudio_colab.ipynb`](phase0_nerfstudio_colab.ipynb) — upload it to
[colab.research.google.com](https://colab.research.google.com/) (or open from Drive) and run top to bottom.
It maps the "running nerfstudio" video steps onto a free T4:

1. New Colab notebook → Runtime → Change runtime type → **GPU (T4)**.
2. **Cell 1** — `pip install nerfstudio` (skips `tiny-cuda-nn`; `splatfacto` only needs `gsplat`).
3. **Cell 2** — `ns-download-data nerfstudio --capture-name poster` (the sample data + COLMAP poses).
4. **Cell 3** — `ns-train splatfacto` on the poster set (headless, TensorBoard).
5. **Cell 4** — `ns-export gaussian-splat` → download the trained `.ply` → drop into `../scenes/`
   locally → open in your Phase 1 viewer.

Swap the poster data for your own captured photos once the sample run works end-to-end.

## Phase 2/3 on RunPod or vast.ai (CUDA dev box)
1. Launch a pod with a recent **CUDA 12.x** image (PyTorch image is fine — brings toolkit + torch).
2. Connect via SSH or VS Code Remote-SSH.
3. Clone this repo (push it to GitHub first, or `scp` the folder up).
4. Build the renderer:
   ```bash
   cd renderer-cuda && cmake -B build && cmake --build build
   ```
5. Train:
   ```bash
   # Install the known-good CUDA torch wheel before the remaining pinned stack.
   pip install torch==2.4.1 torchvision==0.19.1 \
     --index-url https://download.pytorch.org/whl/cu124
   pip install -r trainer/requirements.txt
   MAX_JOBS=3 python -c \
     "from importlib.metadata import version; from gsplat.cuda._backend import _C; print('gsplat ready:', version('gsplat'))"
   python trainer/train.py --doctor
   python trainer/train.py --data /data/scene \
     --output-dir trainer/runs/scene \
     --viewer-ply viewer/public/scenes/scene.ply \
     --iterations 30000 --downscale 2
   ```
   The trainer accepts the `gsplat` 1.5.3 public release and local wheel builds
   such as `1.5.3+pt24cu124`. It uses gsplat for its differentiable CUDA
   forward/backward path; `renderer-cuda/src/backward.cu` is not connected to
   PyTorch. `MAX_JOBS=3` prevents first-build compiler workers from exhausting
   host RAM.

See `bootstrap.sh` for a one-shot dev-box setup after cloning the repository. It
stops unless Python is supported, the gsplat CUDA backend imports, the doctor
reports `ready_for_cuda_training: true`, and the CPU self-test passes. It sets
up the environment but does not start a billable long training run.

## Checkpoints, shutdown, and OOM recovery

Put the run directory on a persistent volume. Scheduled `step-*.pt` files and
`final.pt` contain the Gaussian tensors, all Adam/scheduler/density state, the
completed step, and the random streams required to continue from that boundary.
State restoration is complete, but gsplat's parallel CUDA atomic accumulation
is not promised to be bit-for-bit deterministic across runs, drivers, or GPUs.
Keep checkpoint permissions private to the run: PyTorch `.pt` files are trusted
serialization artifacts and must not be accepted from an untrusted source.

The first SIGINT (`Ctrl+C`) or SIGTERM requests a graceful stop. The trainer
finishes the in-flight iteration and atomically writes `interrupted.pt` only
after that step is complete. A second signal forces immediate exit and does not
write a checkpoint from partial state. This helps with providers that offer a
shutdown grace period; a hard eviction without notice can preserve only the
latest checkpoint already on the persistent volume.

Do not resume an `oom.pt`: the trainer intentionally does not create one,
because CUDA OOM may interrupt a multi-parameter update. Resume the newest prior
scheduled checkpoint and reduce the memory load, for example:

```bash
python trainer/train.py --data /data/scene \
  --resume trainer/runs/scene/checkpoints/step-005000.pt \
  --iterations 30000 --downscale 4 --max-gaussians 1000000
```

Use a shorter `--checkpoint-every` interval when the cost of repeating work is
more important than checkpoint I/O and storage.

## Reference material
- CUDA itself → freeCodeCamp CUDA Course (Elliot Arledge), still `[TODO]` in your Skill.txt.
  Ch 2 (Setup) applies to the **cloud box**, not your local machine.
- nerfstudio / splatfacto → `docs.nerf.studio` (EXT).
