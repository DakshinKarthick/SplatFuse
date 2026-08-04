// Phase 2 · backward pass (STUB): gradients that let the scene learn.
// For every blob, how does a tiny change to its position / scale / color /
// opacity change the final image? Those gradients feed the Phase 3 optimizer.
// Ref: DL — Training through Backpropagation, Autograd in PyTorch (intuition).

#include <cuda_runtime.h>

// TODO __global__ void blendBackward(...)    // d(image)/d(blob params), reverse of blendTiles
// TODO __global__ void projectBackward(...)  // chain-rule back through the 2D projection

extern "C" void launch_backward(/* dL_dimage, saved forward state, grad outputs */) {
    // TODO: launch backward kernels; accumulate grads per blob
}
