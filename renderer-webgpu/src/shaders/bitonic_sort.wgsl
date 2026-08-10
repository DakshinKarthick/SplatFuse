struct Uniforms {
    j: u32,
    k: u32,
    numInstances: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= uniforms.numInstances) { return; }

    let j = i ^ uniforms.j;
    
    // Only process each pair once
    if (j > i && j < uniforms.numInstances) {
        let key_i = keys[i];
        let key_j = keys[j];
        
        let up = (i & uniforms.k) == 0u;
        
        if (up) {
            if (key_i > key_j) {
                keys[i] = key_j;
                keys[j] = key_i;
                let tmp = values[i];
                values[i] = values[j];
                values[j] = tmp;
            }
        } else {
            if (key_i < key_j) {
                keys[i] = key_j;
                keys[j] = key_i;
                let tmp = values[i];
                values[i] = values[j];
                values[j] = tmp;
            }
        }
    }
}
