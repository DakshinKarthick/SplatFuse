# Phase 3 — Train it end-to-end

The loop: render → compare to photo → nudge every blob (Adam) → densify & prune.

## Run
```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python train.py
```

## ⚠️ AMD RX 7600
PyTorch's CUDA build won't use your GPU. Options: ROCm PyTorch under **WSL2/Linux**,
or a **cloud NVIDIA GPU**. See repo README for the full comparison.

## References
- Gradient descent → DL: *Gradient Descent (Theory + PyTorch)*, *Model Optimization*
- Adam → DL: *Adam Optimizer*, *Gradient Descent with Momentum*
- PyTorch → DL: *PyTorch Tensor Basics*, *Autograd in PyTorch*
- Wire your CUDA op into PyTorch → freeCodeCamp CUDA **Ch 9 (PyTorch Extensions)**
- Densify & prune → EXT: INRIA 3DGS paper, *Adaptive Density Control*
