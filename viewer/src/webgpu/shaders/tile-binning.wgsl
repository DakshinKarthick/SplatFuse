const INVALID: u32 = 0xffffffffu;

struct Camera {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
  viewport: vec4<f32>,
  scene: vec4<u32>,
};
struct TileParameters { capacity: u32, _padding: vec3<u32> };
struct Projected {
  centerRadius: vec4<f32>,
  conicOpacity: vec4<f32>,
  color: vec4<f32>,
  tileBounds: vec4<u32>,
};
struct FrameStats {
  visibleCount: atomic<u32>, radiusFixedSum: atomic<u32>,
  duplicateCount: atomic<u32>, overflowCount: atomic<u32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: TileParameters;
@group(0) @binding(2) var<storage, read> projected: array<Projected>;
@group(0) @binding(3) var<storage, read> tileCounts: array<u32>;
@group(0) @binding(4) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> values: array<u32>;
@group(0) @binding(7) var<storage, read_write> stats: FrameStats;

@compute @workgroup_size(256)
fn resetBins(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.capacity) { return; }
  keys[gid.x] = vec2<u32>(INVALID);
  values[gid.x] = INVALID;
}

fn orderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn duplicateIntoTiles(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= camera.scene.x || tileCounts[id] == 0u) { return; }
  let item = projected[id];
  let bounds = item.tileBounds;
  let depthNearFirst = ~orderedFloat(item.centerRadius.z);
  var destination = tileOffsets[id];
  for (var tileY = bounds.y; tileY < bounds.w; tileY++) {
    for (var tileX = bounds.x; tileX < bounds.z; tileX++) {
      if (destination < params.capacity) {
        keys[destination] = vec2<u32>(depthNearFirst, tileY * camera.scene.z + tileX);
        values[destination] = id;
      } else {
        atomicAdd(&stats.overflowCount, 1u);
      }
      destination++;
    }
  }
}
