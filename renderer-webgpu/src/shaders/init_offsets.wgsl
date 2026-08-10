struct Uniforms {
    numTiles: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> tileOffsets: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.numTiles) { return; }
    
    // tileOffsets stores [start_0, end_0, start_1, end_1, ...]
    atomicStore(&tileOffsets[idx * 2u], 0xFFFFFFFFu);
    atomicStore(&tileOffsets[idx * 2u + 1u], 0u);
}
