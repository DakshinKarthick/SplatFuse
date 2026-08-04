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
   cd trainer && pip install -r requirements.txt && python train.py
   ```
   (On cloud NVIDIA, install the **CUDA** build of torch, not the ROCm one:
   `pip install torch` from the default index.)

See `bootstrap.sh` for a one-shot dev-box setup you can paste onto a fresh pod.

## Reference material
- CUDA itself → freeCodeCamp CUDA Course (Elliot Arledge), still `[TODO]` in your Skill.txt.
  Ch 2 (Setup) applies to the **cloud box**, not your local machine.
- nerfstudio / splatfacto → `docs.nerf.studio` (EXT).
