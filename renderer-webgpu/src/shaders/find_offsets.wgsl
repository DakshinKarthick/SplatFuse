struct Uniforms {
    numInstances: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> tileOffsets: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.numInstances) { return; }

    let key = keys[idx];
    let tile_id = key >> 16u;

    if (idx == 0u) {
        atomicMin(&tileOffsets[tile_id * 2u], idx);
        atomicMax(&tileOffsets[tile_id * 2u + 1u], idx + 1u);
    } else {
        let prev_key = keys[idx - 1u];
        let prev_tile_id = prev_key >> 16u;
        
        if (tile_id != prev_tile_id) {
            // First element of the new tile
            atomicMin(&tileOffsets[tile_id * 2u], idx);
            // Last element of the previous tile (idx is the end bound)
            atomicMax(&tileOffsets[prev_tile_id * 2u + 1u], idx);
        }
    }
    
    // Handle the last element
    if (idx == uniforms.numInstances - 1u) {
        atomicMax(&tileOffsets[tile_id * 2u + 1u], idx + 1u);
    }
}
