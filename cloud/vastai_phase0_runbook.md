# vast.ai runbook — Phase 0 splatfacto (the path that actually worked)

The Colab notebook ([`phase0_nerfstudio_colab.ipynb`](phase0_nerfstudio_colab.ipynb)) works but is
slow and flaky (gsplat OOM-compiles every session, sample data 404s). **This is the faster,
repeatable path used to produce [`../scenes/iceland_splat.ply`](../scenes/) on 2026-07-04.**

## Result achieved
RTX 3060, 7000 iterations in **~4 min**, 452k Gaussians, 112 MB `.ply`. Total wall time
including install/compile ≈ 35 min; cost < $0.10.

## 1. Rent the box (vast.ai)
- **GPU:** any single RTX 3060/4060/3090 — tiny scene, don't overpay (~$0.05–0.15/hr).
- **On-Demand** (not Interruptible). **~20–30 GB disk.**
- **Image:** a PyTorch template. NOTE: the one used shipped **Python 3.12 + torch 2.12/cu126**.
- Add your SSH pubkey (Account ▸ SSH Keys). Copy the **proxy** connect string
  (`ssh -p PORT root@sshN.vast.ai`) — the direct IP is often NAT-blocked.

## 2. The two gotchas (why the naive path fails)
1. **No prebuilt gsplat wheel for Python 3.12.** gsplat's wheels stop at cp311, so *any* py3.12
   box (Colab included) must JIT-compile gsplat's 26 CUDA kernels. Torch version is a red herring.
2. **The compile OOM-kills at default parallelism** (`ninja` launches ~nproc jobs, each 1.5–3 GB).
   Fix: throttle with **`MAX_JOBS=3`**. (This box had 31 GB RAM and *still* OOM'd at the default.)

We pin **torch 2.4.1+cu124** anyway — gsplat compiles more reliably against it than bleeding-edge
torch, and it matches the CUDA 12.4 driver.

## 3. Commands (run over SSH; venv is at /venv/main)
```bash
source /venv/main/bin/activate

# torch 2.4.1 (known-good for gsplat) + nerfstudio
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install nerfstudio            # pulls gsplat (source; compiles on first import)

# compile gsplat ONCE, throttled so it doesn't OOM (~15 min, then cached forever on this box)
MAX_JOBS=3 python -c "from gsplat.cuda._backend import _C; print('gsplat built')"

# upload your COLMAP scene (from a laptop:  scp -P PORT iceland_upload.zip root@sshN.vast.ai:/root/)
unzip -q iceland_upload.zip -d /root/     # -> /root/iceland/{images, sparse/0}

# train straight off the COLMAP model via the `colmap` dataparser (no ns-process-data,
# which needs a COLMAP binary; no transforms.json). Dataparser flags go AFTER `colmap`.
ns-train splatfacto --data /root/iceland --max-num-iterations 7000 --vis tensorboard \
    colmap --colmap-path sparse/0 --images-path images

# export the .ply
CFG=$(ls -t /root/outputs/iceland/splatfacto/*/config.yml | head -1)
ns-export gaussian-splat --load-config "$CFG" --output-dir /root/exports/iceland
```

## 4. Retrieve + tear down
```bash
# from your laptop:
scp -P PORT root@sshN.vast.ai:/root/exports/iceland/splat.ply ./scenes/<name>_splat.ply
```
Then **DESTROY the instance** on vast.ai (stops all billing). The gsplat compile cache dies with
it — fine, Phase 1 is local and Phase 2 rebuilds CUDA from scratch anyway.

## Input data note
`iceland_upload.zip` = the 405 frames + **largest** COLMAP sub-model (`sparse/0`, 140 imgs) from
`phase0-capture/04 SCENES/…Iceland…/`, staged locally. Forward-flying FPV footage → the splat
covers that canyon corridor and smears where views don't overlap. Good enough for the Phase 0
milestone; for a *pretty* splat, capture orbit-style footage that revisits surfaces.
