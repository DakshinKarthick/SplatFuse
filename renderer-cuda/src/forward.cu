#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cmath>

#define TILE_SIZE 16

// ----------------- MATH FUNCTIONS & CONSTANTS -----------------
// Computes the 2D screen-space covariance matrix from 3D scale and rotation for a Gaussian.
__device__ void computeCov2D(const float3 scale, const float4 rot, const float* viewMatrix, const float fx, const float fy, const float3 t, float* cov2D) {
    float r = rot.x, x = rot.y, y = rot.z, z = rot.w;
    float R[9] = { 1.f - 2.f*(y*y + z*z), 2.f*(x*y - r*z), 2.f*(x*z + r*y), 2.f*(x*y + r*z), 1.f - 2.f*(x*x + z*z), 2.f*(y*z - r*x), 2.f*(x*z - r*y), 2.f*(y*z + r*x), 1.f - 2.f*(x*x + y*y) };
    float M[9] = { R[0]*scale.x, R[1]*scale.y, R[2]*scale.z, R[3]*scale.x, R[4]*scale.y, R[5]*scale.z, R[6]*scale.x, R[7]*scale.y, R[8]*scale.z };
    float Sigma[6] = { M[0]*M[0] + M[1]*M[1] + M[2]*M[2], M[0]*M[3] + M[1]*M[4] + M[2]*M[5], M[0]*M[6] + M[1]*M[7] + M[2]*M[8], M[3]*M[3] + M[4]*M[4] + M[5]*M[5], M[3]*M[6] + M[4]*M[7] + M[5]*M[8], M[6]*M[6] + M[7]*M[7] + M[8]*M[8] };
    float W[9] = { viewMatrix[0], viewMatrix[1], viewMatrix[2], viewMatrix[4], viewMatrix[5], viewMatrix[6], viewMatrix[8], viewMatrix[9], viewMatrix[10] };
    float WS[9] = { W[0]*Sigma[0]+W[1]*Sigma[1]+W[2]*Sigma[2], W[0]*Sigma[1]+W[1]*Sigma[3]+W[2]*Sigma[4], W[0]*Sigma[2]+W[1]*Sigma[4]+W[2]*Sigma[5], W[3]*Sigma[0]+W[4]*Sigma[1]+W[5]*Sigma[2], W[3]*Sigma[1]+W[4]*Sigma[3]+W[5]*Sigma[4], W[3]*Sigma[2]+W[4]*Sigma[4]+W[5]*Sigma[5], W[6]*Sigma[0]+W[7]*Sigma[1]+W[8]*Sigma[2], W[6]*Sigma[1]+W[7]*Sigma[3]+W[8]*Sigma[4], W[6]*Sigma[2]+W[7]*Sigma[4]+W[8]*Sigma[5] };
    float Sigma_c[9] = { WS[0]*W[0]+WS[1]*W[1]+WS[2]*W[2], WS[0]*W[3]+WS[1]*W[4]+WS[2]*W[5], WS[0]*W[6]+WS[1]*W[7]+WS[2]*W[8], WS[3]*W[3]+WS[4]*W[4]+WS[5]*W[5], WS[3]*W[6]+WS[4]*W[7]+WS[5]*W[8], WS[6]*W[6]+WS[7]*W[7]+WS[8]*W[8] };
    float J[6] = { fx/t.z, 0.f, -(fx*t.x)/(t.z*t.z), 0.f, fy/t.z, -(fy*t.y)/(t.z*t.z) };
    float JS[6] = { J[0]*Sigma_c[0]+J[2]*Sigma_c[2], J[0]*Sigma_c[1]+J[2]*Sigma_c[5], J[0]*Sigma_c[2]+J[2]*Sigma_c[8], J[4]*Sigma_c[1]+J[5]*Sigma_c[2], J[4]*Sigma_c[4]+J[5]*Sigma_c[5], J[4]*Sigma_c[5]+J[5]*Sigma_c[8] };
    cov2D[0] = JS[0]*J[0] + JS[2]*J[2] + 0.3f;
    cov2D[1] = JS[1]*J[4] + JS[2]*J[5];
    cov2D[2] = JS[4]*J[4] + JS[5]*J[5] + 0.3f;
}

// Converts a float to an integer preserving sort order, useful for depth sorting.
__device__ uint32_t floatToOrderedInt(float val) {
    uint32_t intVal; memcpy(&intVal, &val, 4); return intVal;
}
// --------------------------------------------------------------

// Maps 3D Gaussians to the 2D image plane, calculating screen position, depth, and footprint radius.
__global__ void projectGaussiansKernel(int num_gaussians, const float3* positions, const float3* scales, const float4* rotations, const float* colors, const float* opacities, const float* viewMatrix, const float* projMatrix, int width, int height, float2* points_xy, float* points_depth, float3* points_cov2d, float* points_opacity_act, int* points_radii, bool* points_visible) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    float3 p = positions[idx];
    float3 t = { viewMatrix[0]*p.x + viewMatrix[1]*p.y + viewMatrix[2]*p.z + viewMatrix[3], viewMatrix[4]*p.x + viewMatrix[5]*p.y + viewMatrix[6]*p.z + viewMatrix[7], viewMatrix[8]*p.x + viewMatrix[9]*p.y + viewMatrix[10]*p.z + viewMatrix[11] };
    if (t.z < 0.01f) { points_visible[idx] = false; return; }
    
    float fx = projMatrix[0], fy = projMatrix[1], cx = projMatrix[2], cy = projMatrix[3];
    float2 xy = make_float2(fx * t.x / t.z + cx, fy * t.y / t.z + cy);
    if (xy.x < -100.f || xy.x > width + 100.f || xy.y < -100.f || xy.y > height + 100.f) { points_visible[idx] = false; return; }
    
    float cov2D[3]; computeCov2D(scales[idx], rotations[idx], viewMatrix, fx, fy, t, cov2D);
    float det = cov2D[0]*cov2D[2] - cov2D[1]*cov2D[1];
    float mid = 0.5f * (cov2D[0] + cov2D[2]);
    float lambda = mid + sqrtf(fmaxf(0.1f, mid*mid - det));
    
    points_xy[idx] = xy; points_depth[idx] = t.z;
    points_cov2d[idx] = make_float3(cov2D[0], cov2D[1], cov2D[2]);
    points_opacity_act[idx] = 1.0f / (1.0f + expf(-opacities[idx]));
    points_radii[idx] = (int)ceilf(3.f * sqrtf(lambda));
    points_visible[idx] = true;
}

// Calculates the bounding box of each Gaussian footprint and counts how many 16x16 tiles it overlaps.
__global__ void countTilesKernel(int num_gaussians, const float2* points_xy, const int* points_radii, const bool* points_visible, int width, int height, int* num_tiles_per_gaussian) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    if (!points_visible[idx]) { num_tiles_per_gaussian[idx] = 0; return; }
    float2 xy = points_xy[idx]; float r = points_radii[idx];
    int min_x = max(0, (int)((xy.x - r) / TILE_SIZE)), max_x = min((width + TILE_SIZE - 1) / TILE_SIZE - 1, (int)((xy.x + r) / TILE_SIZE));
    int min_y = max(0, (int)((xy.y - r) / TILE_SIZE)), max_y = min((height + TILE_SIZE - 1) / TILE_SIZE - 1, (int)((xy.y + r) / TILE_SIZE));
    num_tiles_per_gaussian[idx] = max(0, (max_x - min_x + 1) * (max_y - min_y + 1));
}

// Generates 64-bit sorting keys (Tile ID | Depth) for every tile a Gaussian overlaps.
__global__ void writeDuplicateKeysKernel(int num_gaussians, const float2* points_xy, const float* points_depth, const int* points_radii, const bool* points_visible, const int* offsets, int width, int height, uint64_t* tile_keys, int* tile_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians || !points_visible[idx]) return;
    float2 xy = points_xy[idx]; float r = points_radii[idx];
    int min_x = max(0, (int)((xy.x - r) / TILE_SIZE)), max_x = min((width + TILE_SIZE - 1) / TILE_SIZE - 1, (int)((xy.x + r) / TILE_SIZE));
    int min_y = max(0, (int)((xy.y - r) / TILE_SIZE)), max_y = min((height + TILE_SIZE - 1) / TILE_SIZE - 1, (int)((xy.y + r) / TILE_SIZE));
    int write_offset = offsets[idx]; int tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t depth_key = floatToOrderedInt(points_depth[idx]);
    for (int y = min_y; y <= max_y; ++y)
        for (int x = min_x; x <= max_x; ++x) {
            tile_keys[write_offset] = ((uint64_t)(y * tiles_x + x) << 32) | depth_key;
            tile_values[write_offset++] = idx;
        }
}

// Scans sorted keys to find the start and end indices of Gaussians belonging to each Tile ID.
__global__ void identifyTileRangesKernel(int num_items, const uint64_t* sorted_keys, int2* tile_ranges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;
    uint32_t tile_id = sorted_keys[idx] >> 32;
    if (idx == 0) tile_ranges[tile_id].x = 0;
    else {
        uint32_t prev = sorted_keys[idx - 1] >> 32;
        if (tile_id != prev) { tile_ranges[tile_id].x = idx; tile_ranges[prev].y = idx; }
    }
    if (idx == num_items - 1) tile_ranges[tile_id].y = num_items;
}

// Rasterizes and alpha-blends Gaussians front-to-back into the final image pixels for each tile.
__global__ void blendTilesKernel(int width, int height, const int2* tile_ranges, const int* sorted_values, const float2* points_xy, const float3* points_cov2d, const float* points_opacity, const float3* points_rgb, float* out_image) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    int tile_id = (py / TILE_SIZE) * ((width + TILE_SIZE - 1) / TILE_SIZE) + (px / TILE_SIZE);
    int2 range = tile_ranges[tile_id];
    float3 color = make_float3(0.f, 0.f, 0.f); float T = 1.0f;
    
    for (int i = range.x; i < range.y && T >= 0.0001f; ++i) {
        int g = sorted_values[i];
        float3 cov = points_cov2d[g]; float det = cov.x*cov.z - cov.y*cov.y;
        if (det < 1e-6f) continue;
        float inv = 1.f / det; float a = cov.z*inv, b = -cov.y*inv, c = cov.x*inv;
        float dx = px + 0.5f - points_xy[g].x, dy = py + 0.5f - points_xy[g].y;
        float power = -0.5f * (a*dx*dx + 2.f*b*dx*dy + c*dy*dy);
        if (power > 0.f) continue;
        float alpha = points_opacity[g] * expf(power);
        if (alpha < 1.f/255.f) continue;
        color.x += points_rgb[g].x * alpha * T; color.y += points_rgb[g].y * alpha * T; color.z += points_rgb[g].z * alpha * T;
        T *= (1.f - alpha);
    }
    int img_idx = (py * width + px) * 3;
    out_image[img_idx] = color.x; out_image[img_idx+1] = color.y; out_image[img_idx+2] = color.z;
}

// Orchestrates the entire GPU rendering pipeline: memory allocation, kernel launches, and sorting.
extern "C" void launch_forward(int num_gaussians, const float* d_positions, const float* d_scales, const float* d_rotations, const float* d_colors, const float* d_opacities, const float* d_viewMatrix, const float* d_projMatrix, float* d_out_image, int width, int height) {
    float2* d_xy; float* d_depth; float3* d_cov; float* d_op; int* d_radii; bool* d_vis;
    cudaMalloc(&d_xy, num_gaussians*sizeof(float2)); cudaMalloc(&d_depth, num_gaussians*sizeof(float));
    cudaMalloc(&d_cov, num_gaussians*sizeof(float3)); cudaMalloc(&d_op, num_gaussians*sizeof(float));
    cudaMalloc(&d_radii, num_gaussians*sizeof(int)); cudaMalloc(&d_vis, num_gaussians*sizeof(bool));

    int bs = 256, gs = (num_gaussians + bs - 1) / bs;
    projectGaussiansKernel<<<gs, bs>>>(num_gaussians, (float3*)d_positions, (float3*)d_scales, (float4*)d_rotations, d_colors, d_opacities, d_viewMatrix, d_projMatrix, width, height, d_xy, d_depth, d_cov, d_op, d_radii, d_vis);

    int* d_tiles; cudaMalloc(&d_tiles, num_gaussians*sizeof(int));
    countTilesKernel<<<gs, bs>>>(num_gaussians, d_xy, d_radii, d_vis, width, height, d_tiles);

    int* d_offsets; cudaMalloc(&d_offsets, num_gaussians*sizeof(int));
    void* d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_tiles, d_offsets, num_gaussians);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_tiles, d_offsets, num_gaussians);
    cudaFree(d_temp);

    int last_off, last_cnt;
    cudaMemcpy(&last_off, d_offsets + num_gaussians - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_cnt, d_tiles + num_gaussians - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int num_items = last_off + last_cnt;

    uint64_t *d_keys_in, *d_keys_out; int *d_vals_in, *d_vals_out;
    cudaMalloc(&d_keys_in, num_items*8); cudaMalloc(&d_keys_out, num_items*8);
    cudaMalloc(&d_vals_in, num_items*4); cudaMalloc(&d_vals_out, num_items*4);

    writeDuplicateKeysKernel<<<gs, bs>>>(num_gaussians, d_xy, d_depth, d_radii, d_vis, d_offsets, width, height, d_keys_in, d_vals_in);

    d_temp = nullptr; temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, num_items);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, num_items);
    cudaFree(d_temp);

    int num_tiles = ((width+TILE_SIZE-1)/TILE_SIZE) * ((height+TILE_SIZE-1)/TILE_SIZE);
    int2* d_ranges; cudaMalloc(&d_ranges, num_tiles*sizeof(int2)); cudaMemset(d_ranges, 0, num_tiles*sizeof(int2));
    if (num_items > 0) identifyTileRangesKernel<<<(num_items + bs - 1)/bs, bs>>>(num_items, d_keys_out, d_ranges);

    blendTilesKernel<<<dim3((width+TILE_SIZE-1)/TILE_SIZE, (height+TILE_SIZE-1)/TILE_SIZE), dim3(TILE_SIZE, TILE_SIZE)>>>(width, height, d_ranges, d_vals_out, d_xy, d_cov, d_op, (float3*)d_colors, d_out_image);
    cudaDeviceSynchronize();

    cudaFree(d_xy); cudaFree(d_depth); cudaFree(d_cov); cudaFree(d_op); cudaFree(d_radii); cudaFree(d_vis);
    cudaFree(d_tiles); cudaFree(d_offsets); cudaFree(d_keys_in); cudaFree(d_keys_out); cudaFree(d_vals_in); cudaFree(d_vals_out); cudaFree(d_ranges);
}
