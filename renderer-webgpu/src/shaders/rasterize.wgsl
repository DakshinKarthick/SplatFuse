struct Uniforms {
    screenWidth: u32,
    screenHeight: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct Splat2D {
    xy: vec2<f32>,
    conic: vec3<f32>,
    color: vec4<f32>,
    depth: f32,
    bounds: vec4<u32>,
};

@group(1) @binding(0) var<storage, read> splats2D: array<Splat2D>;
@group(1) @binding(1) var<storage, read> tileSplatIndices: array<u32>; // Pre-sorted indices for this tile
@group(1) @binding(2) var<storage, read> tileOffsets: array<u32>; // Start index and count per tile

@group(2) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) tile_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let px = global_id.x;
    let py = global_id.y;
    
    if (px >= uniforms.screenWidth || py >= uniforms.screenHeight) { return; }

    let tiles_x = (uniforms.screenWidth + 15u) / 16u;
    let tile_idx = tile_id.y * tiles_x + tile_id.x;

    // To implement real tile-based rasterization:
    // We would fetch the start and end of the sorted splats for this tile.
    // For scaffolding, assume tileOffsets[tile_idx] stores the start and tileOffsets[tile_idx + 1] is end.
    let start_idx = tileOffsets[tile_idx];
    let end_idx = tileOffsets[tile_idx + 1u];

    var final_color = vec3<f32>(0.0, 0.0, 0.0);
    var transmittance = 1.0;
    
    let pixel_coord = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

    for (var i = start_idx; i < end_idx; i = i + 1u) {
        if (transmittance < 0.01) { break; }
        
        let s_idx = tileSplatIndices[i];
        let splat = splats2D[s_idx];

        let d = pixel_coord - splat.xy;
        let power = -0.5 * (splat.conic.x * d.x * d.x + splat.conic.z * d.y * d.y) - splat.conic.y * d.x * d.y;
        
        if (power > 0.0) { continue; }
        
        let alpha = min(0.99, splat.color.a * exp(power));
        if (alpha < 1.0 / 255.0) { continue; }

        let test_t = transmittance * (1.0 - alpha);
        
        final_color += splat.color.rgb * alpha * transmittance;
        transmittance = test_t;
    }

    final_color += vec3<f32>(0.1, 0.1, 0.1) * transmittance; // Background color

    textureStore(outputTexture, vec2<i32>(i32(px), i32(py)), vec4<f32>(final_color, 1.0));
}
