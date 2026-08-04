"""
Phase 3 · training loop (STUB)

The core loop: render blobs -> compare render to the real photo -> nudge every
blob to shrink the error (Adam) -> periodically densify & prune. Once it trains,
you've rebuilt Gaussian splatting end-to-end.

Study refs:
  - Gradient descent           -> DL: Gradient Descent (Theory + PyTorch); Model Optimization
  - Adam                        -> DL: Adam Optimizer; Gradient Descent with Momentum
  - PyTorch loop / autograd     -> DL: PyTorch Tensor Basics; Autograd in PyTorch
  - CUDA op into PyTorch        -> freeCodeCamp CUDA Ch 9 (PyTorch Extensions)
  - Densify & prune heuristic   -> EXT: INRIA 3DGS paper, "Adaptive Density Control"
"""


def train():
    # TODO: load cameras + ground-truth photos (from Phase 0 / COLMAP)
    # TODO: initialize Gaussians from the COLMAP sparse point cloud
    # TODO: optimizer = Adam over blob params (position, scale, rotation, color, opacity)
    # TODO: for each iteration:
    #         render (Phase 2 forward) -> loss vs photo -> backward -> optimizer.step()
    #         every N iters: densify blurry regions, prune transparent/useless blobs
    # TODO: save the trained scene to ../scenes/<name>.ply
    raise NotImplementedError("train() is a stub — see TODO.md, Phase 3.")


if __name__ == "__main__":
    train()
