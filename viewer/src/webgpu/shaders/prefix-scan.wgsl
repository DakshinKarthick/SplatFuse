struct Parameters {
  length: u32,
  _padding: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read_write> inputData: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputData: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;

var<workgroup> scratch: array<u32, 256>;

@compute @workgroup_size(256)
fn scanBlocks(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(global_invocation_id) global: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let lane = local.x;
  scratch[lane] = select(0u, inputData[global.x], global.x < params.length);

  var offset = 1u;
  for (var active = 128u; active > 0u; active >>= 1u) {
    workgroupBarrier();
    if (lane < active) {
      let left = offset * (2u * lane + 1u) - 1u;
      let right = offset * (2u * lane + 2u) - 1u;
      scratch[right] += scratch[left];
    }
    offset <<= 1u;
  }
  workgroupBarrier();

  if (lane == 0u) {
    blockSums[group.x] = scratch[255];
    scratch[255] = 0u;
  }

  for (var active = 1u; active < 256u; active <<= 1u) {
    offset >>= 1u;
    workgroupBarrier();
    if (lane < active) {
      let left = offset * (2u * lane + 1u) - 1u;
      let right = offset * (2u * lane + 2u) - 1u;
      let value = scratch[left];
      scratch[left] = scratch[right];
      scratch[right] += value;
    }
  }
  workgroupBarrier();
  if (global.x < params.length) outputData[global.x] = scratch[lane];
}

@compute @workgroup_size(256)
fn addBlockOffsets(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.length) { return; }
  outputData[gid.x] += inputData[gid.x / 256u];
}
