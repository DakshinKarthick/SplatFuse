# 3D Gaussian Splatting CUDA Renderer

This repository implements a forward-pass renderer for 3D Gaussian Splatting using CUDA. It transforms 3D point clouds (splats) into a 2D image via tiled rasterization. This document provides a detailed, step-by-step breakdown of the process, designed for beginners to C++ and CUDA.

## Imports and Dependencies

### C++ Host Imports (`main.cpp`)
- `<cstdio>`, `<iostream>`, `<fstream>`, `<sstream>`: Used for standard input/output operations and reading/writing files (like parsing the `.ply` and saving `.ppm`).
- `<vector>`, `<string>`, `<unordered_map>`: Standard Template Library (STL) data structures used for dynamic arrays (storing Gaussians), string manipulation, and key-value mapping (for property offsets).
- `<cmath>`, `<cstring>`, `<algorithm>`: Math functions, memory manipulation (`memcpy`), and algorithms (`std::max`, `std::min`).
- `<cuda_runtime.h>`: The core CUDA runtime library that provides functions for memory management (`cudaMalloc`, `cudaMemcpy`) and error checking on the host side.

### CUDA Device Imports (`forward.cu`)
- `<cuda_runtime.h>`: Core CUDA API.
- `<device_launch_parameters.h>`: Defines built-in CUDA variables like `threadIdx`, `blockIdx`, and `blockDim` used to identify threads within the grid.
- `<cub/cub.cuh>`: CUB library headers. CUB provides high-performance parallel primitives like Prefix Sum (Scan) and Radix Sort, which are essential for grouping and ordering data on the GPU.
- `<cmath>`: Device-side math operations (`expf`, `sqrtf`).

## Essential CUDA Keywords
- `__global__`: Declares a function as a CUDA kernel. It is called from the host (CPU) and executed on the device (GPU) by many threads in parallel.
- `__device__`: Declares a function that is both called from and executed on the device (GPU).
- `cudaMalloc`: Allocates memory on the GPU (VRAM).
- `cudaMemcpy`: Copies memory between the Host and Device (e.g., `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`).
- `cudaFree`: Frees allocated memory on the GPU.
- `<<<blocks, threads>>>`: The execution configuration syntax used when launching a `__global__` kernel, specifying the number of thread blocks and threads per block.

---

## Pipeline Flow & Architecture

The renderer consists of a **Host program** (`main.cpp`) which manages data, and a **Device program** (`forward.cu`) which executes algorithms in parallel.

### 1. Data Loading & Preparation (Host - `main.cpp`)

#### 1.1. Parse PLY file & Extract Properties
We read a `.ply` binary file containing 3D Gaussians. We extract Position ($x, y, z$), Scale (Log-space), Rotation (Quaternion), Colors (Spherical Harmonics), and Opacity (Logit space).

#### 1.2. Activation Functions
Raw properties from the file need to be converted to physical values because they were optimized in an unbounded space during training.
- **Scale:** $S = e^{S_{raw}}$ (Ensures scale is strictly positive).
- **Opacity:** $\alpha = \sigma(O_{raw}) = \frac{1}{1 + e^{-O_{raw}}}$ (Sigmoid function binds opacity between 0 and 1).
- **Color:** $RGB = 0.5 + 0.28209 \times SH_{raw}$ (Converts the 0th degree Spherical Harmonics directly to RGB).
- **Rotation:** The quaternion is normalized to ensure it represents a valid 3D rotation.

#### 1.3. GPU Memory Allocation and Transfer
We allocate memory on the GPU using `cudaMalloc` and copy our converted arrays over using `cudaMemcpy`.
*Why?* The GPU cannot directly read the CPU's RAM. We must transfer data to the GPU's VRAM for fast, parallel processing.

---

### 2. GPU Rendering Pipeline (Device - `forward.cu`)

#### Step A: Projection Kernel (`projectGaussiansKernel`)
*Goal:* Map 3D Gaussians to the 2D image plane and determine their shape on the screen.
*Keyword:* `__global__`

1. **World to Camera Space**: 
   $t = W \cdot p$ (where $W$ is the View Matrix, $p$ is the 3D position).
   *Why?* We need to know where the point is relative to the camera's viewpoint.
2. **Camera to Screen Space (Perspective Projection)**: 
   $x_{screen} = f_x \frac{t_x}{t_z} + c_x$ 
   *Why?* This projects the 3D point onto a 2D plane, applying perspective (objects further away appear smaller due to division by $t_z$).
3. **3D Covariance Matrix**: 
   $\Sigma_{3D} = R S S^T R^T$ (R = Rotation Matrix, S = Scale Matrix)
   *Why?* Describes the volume and orientation of the 3D Gaussian ellipsoid.
4. **2D Screen Space Covariance**:
   $\Sigma_{2D} = J W \Sigma_{3D} W^T J^T$ 
   *Why?* Projects the 3D ellipsoid into a 2D ellipse on the screen. $J$ is the Jacobian of the perspective projection.
5. **Footprint Radius**: 
   We find the maximum eigenvalue of the 2D covariance to determine the radius. 
   $radius = \lceil 3 \sqrt{\lambda_{max}} \rceil$
   *Why?* 3 standard deviations cover 99.7% of the Gaussian. We only need to render within this bounding box.

#### Step B: Tile Overlap Counting (`countTilesKernel`)
*Goal:* Divide the screen into 16x16 pixel blocks (tiles). Calculate how many tiles each Gaussian touches.
*Why?* By splitting the screen into tiles, we can render each tile independently in parallel.

#### Step C: Prefix Sum (`cub::DeviceScan::ExclusiveSum`)
*Goal:* Calculate a running total of tile overlaps.
*Why?* If Gaussian 0 touches 3 tiles, and Gaussian 1 touches 2, we need an array of size 5 to store all intersections. The prefix sum gives us the exact starting index (offset) for each Gaussian to write its intersections into a flat array without overwriting each other.

#### Step D: Key-Value Generation (`writeDuplicateKeysKernel`)
*Goal:* For every tile a Gaussian overlaps, create a sorting key and value.
- **Key (64-bit)**: `[ 32-bit Tile ID | 32-bit Depth ]` (Depth is converted to an integer using `floatToOrderedInt`).
- **Value (32-bit)**: `[ Gaussian Index ]`
*Why?* We encode both the tile it belongs to and its distance from the camera into a single 64-bit number.

#### Step E: Radix Sort (`cub::DeviceRadixSort::SortPairs`)
*Goal:* Sort the 64-bit keys in parallel.
*Why?* Radix sort orders the arrays such that:
1. All instances belonging to the same Tile ID are grouped together.
2. Within the same Tile ID, they are ordered by depth (front-to-back). This is crucial for correct alpha blending.

#### Step F: Range Identification (`identifyTileRangesKernel`)
*Goal:* Scan the sorted keys to find where one Tile ID ends and the next begins.
*Why?* To give each pixel-rendering thread block the exact `[start_index, end_index]` of Gaussians it needs to process for its specific tile.

#### Step G: Rasterization and Blending (`blendTilesKernel`)
*Goal:* Compute the final color for every pixel.
*Configuration:* Launched with 2D thread blocks using `dim3(TILE_SIZE, TILE_SIZE)`, meaning one thread perfectly maps to one pixel in the tile.

1. **Evaluate Gaussian Density**:
   For a pixel $(px, py)$, calculate its distance from the Gaussian center:
   $$power = -\frac{1}{2} [dx, dy] \Sigma_{2D}^{-1} \begin{bmatrix} dx \\ dy \end{bmatrix}$$
   $$\alpha_{final} = \alpha_{base} \times e^{power}$$
   *Why?* This is the standard Gaussian probability density function. The color fades out exponentially the further you are from the center.
2. **Alpha Blending (Front-to-Back)**:
   $$C = C + c_i \alpha_{final} T$$
   $$T = T \times (1 - \alpha_{final})$$
   *Why?* We accumulate the color $c_i$. $T$ is Transmittance (starting at 1.0). As we add opaque layers, Transmittance drops, representing light being blocked.
3. **Early Ray Termination**:
   If $T < 0.0001$, we stop looping.
   *Why?* The pixel is completely covered. Processing Gaussians behind it would waste computations since they are blocked.

#### 3. Image Output
The final image array is copied back to the CPU (`cudaMemcpyDeviceToHost`) and saved as a `.ppm` image format. Finally, GPU memory is released (`cudaFree`).
