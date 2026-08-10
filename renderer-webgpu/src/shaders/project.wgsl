struct Uniforms {
    viewMatrix: mat4x4<f32>,
    projMatrix: mat4x4<f32>,
    cameraPos: vec3<f32>,
    screenWidth: u32,
    screenHeight: u32,
    focal_x: f32,
    focal_y: f32,
    tan_fovx: f32,
    tan_fovy: f32,
    scale_modifier: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Input buffers
@group(1) @binding(0) var<storage, read> positions: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read> scales: array<vec3<f32>>;
@group(1) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read> colors: array<vec3<f32>>;
@group(1) @binding(4) var<storage, read> opacities: array<f32>;

struct Splat2D {
    xy: vec2<f32>,
    conic: vec3<f32>,
    color: vec4<f32>,
    depth: f32,
    bounds: vec2<u32>, // Packed min/max tile bounds (since we only need it for debugging, but actually we don't even need bounds here if we emit keys directly)
};

@group(2) @binding(0) var<storage, read_write> splats2D: array<Splat2D>;
@group(2) @binding(1) var<storage, read_write> instanceKeys: array<u32>;
@group(2) @binding(2) var<storage, read_write> instanceValues: array<u32>;
@group(2) @binding(3) var<storage, read_write> globalInstanceCount: atomic<u32>;

fn computeCov3D(scale: vec3<f32>, rot: vec4<f32>, modifier: f32) -> array<f32, 6> {
    let s = scale * modifier;
    let S = mat3x3<f32>(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
    let r = rot.x; let x = rot.y; let y = rot.z; let z = rot.w;
    let R = mat3x3<f32>(
        1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - r*z), 2.0 * (x*z + r*y),
        2.0 * (x*y + r*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - r*x),
        2.0 * (x*z - r*y), 2.0 * (y*z + r*x), 1.0 - 2.0 * (x*x + y*y)
    );
    let M = R * S;
    let Sigma = M * transpose(M);
    return array<f32, 6>(Sigma[0][0], Sigma[0][1], Sigma[0][2], Sigma[1][1], Sigma[1][2], Sigma[2][2]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&positions)) { return; }

    let p_orig = positions[idx];
    let p_view = uniforms.viewMatrix * vec4<f32>(p_orig, 1.0);
    
    // Near plane culling
    if (p_view.z >= -0.2) {
        splats2D[idx].bounds = vec2<u32>(0u, 0u);
        return;
    }

    let p_proj = uniforms.projMatrix * p_view;
    let p_ndc = p_proj.xyz / p_proj.w;
    let w = f32(uniforms.screenWidth);
    let h = f32(uniforms.screenHeight);
    let screen_pos = vec2<f32>((p_ndc.x + 1.0) * w * 0.5, (1.0 - p_ndc.y) * h * 0.5);

    let cov3D = computeCov3D(scales[idx], rotations[idx], uniforms.scale_modifier);
    let tx = p_view.x; let ty = p_view.y; let tz = p_view.z;
    let tz2 = tz * tz;
    let J = mat3x3<f32>(
        uniforms.focal_x / tz, 0.0, -(uniforms.focal_x * tx) / tz2,
        0.0, uniforms.focal_y / tz, -(uniforms.focal_y * ty) / tz2,
        0.0, 0.0, 0.0
    );
    let W = mat3x3<f32>(
        uniforms.viewMatrix[0][0], uniforms.viewMatrix[1][0], uniforms.viewMatrix[2][0],
        uniforms.viewMatrix[0][1], uniforms.viewMatrix[1][1], uniforms.viewMatrix[2][1],
        uniforms.viewMatrix[0][2], uniforms.viewMatrix[1][2], uniforms.viewMatrix[2][2]
    );
    let T = J * W;
    let Vrk = mat3x3<f32>(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]);
    let cov2D = T * Vrk * transpose(T);

    var cov2D_00 = cov2D[0][0] + 0.3;
    var cov2D_11 = cov2D[1][1] + 0.3;
    var cov2D_01 = cov2D[0][1];

    let det = cov2D_00 * cov2D_11 - cov2D_01 * cov2D_01;
    if (det == 0.0) { return; }
    let inv_det = 1.0 / det;
    let conic = vec3<f32>(cov2D_11 * inv_det, -cov2D_01 * inv_det, cov2D_00 * inv_det);
    let radius = ceil(3.0 * sqrt(max(cov2D_00, cov2D_11)));
    
    let min_x = u32(max(0.0, floor((screen_pos.x - radius) / 16.0)));
    let min_y = u32(max(0.0, floor((screen_pos.y - radius) / 16.0)));
    let max_x = u32(min(ceil(w / 16.0) - 1.0, floor((screen_pos.x + radius) / 16.0)));
    let max_y = u32(min(ceil(h / 16.0) - 1.0, floor((screen_pos.y + radius) / 16.0)));

    splats2D[idx].xy = screen_pos;
    splats2D[idx].conic = conic;
    splats2D[idx].color = vec4<f32>(colors[idx], opacities[idx]);
    splats2D[idx].depth = p_view.z;
    
    // Bounds can be packed or ignored, let's just pack them into vec2 for minimal storage
    splats2D[idx].bounds = vec2<u32>((min_x & 0xFFFFu) | (min_y << 16u), (max_x & 0xFFFFu) | (max_y << 16u));

    let tiles_x = u32(ceil(w / 16.0));
    
    // Normalize depth to 16 bits. p_view.z is negative (looking down -Z in standard gl-matrix). 
    // Assuming near=0.1, far=100. Let's remap z to 0-65535, ascending.
    // wait, if Z is negative, smaller (more negative) is further away.
    // We want front-to-back, so we want largest Z first?
    // -0.1 is front, -100 is back.
    // To sort front-to-back, we want -0.1 to be 0, and -100 to be 65535.
    // let's compute distance = -p_view.z.
    let dist = -p_view.z; 
    let norm_depth = u32(clamp((dist - 0.1) / 100.0, 0.0, 1.0) * 65535.0);
    
    let tiles_touched = (max_x - min_x + 1u) * (max_y - min_y + 1u);
    if (tiles_touched > 0u) {
        let offset = atomicAdd(&globalInstanceCount, tiles_touched);
        let max_capacity = arrayLength(&instanceKeys);
        if (offset + tiles_touched <= max_capacity) {
            var curr = offset;
            for (var y = min_y; y <= max_y; y++) {
                for (var x = min_x; x <= max_x; x++) {
                    let tile_id = y * tiles_x + x;
                    let key = (tile_id << 16u) | (norm_depth & 0xFFFFu);
                    instanceKeys[curr] = key;
                    instanceValues[curr] = idx;
                    curr++;
                }
            }
        }
    }
}
