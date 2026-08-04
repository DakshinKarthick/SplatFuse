const TILE_SIZE: u32 = 16u;
const INVALID: u32 = 0xffffffffu;

struct Camera {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
  viewport: vec4<f32>, // width, height, 1/width, 1/height
  scene: vec4<u32>,    // splat count, sort capacity, tiles x, tiles y
};

struct Splat {
  centerOpacity: vec4<f32>,
  scale: vec4<f32>,
  rotation: vec4<f32>, // w, x, y, z
  color: vec4<f32>,
};

struct Projected {
  centerRadius: vec4<f32>, // pixel x, pixel y, view z, 3-sigma radius
  conicOpacity: vec4<f32>, // inverse covariance a, b, c, opacity
  color: vec4<f32>,
  tileBounds: vec4<u32>,   // inclusive min, exclusive max
};

struct FrameStats {
  visibleCount: atomic<u32>,
  radiusFixedSum: atomic<u32>,
  duplicateCount: atomic<u32>,
  overflowCount: atomic<u32>,
};

struct DrawIndirect {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read_write> projected: array<Projected>;
@group(0) @binding(3) var<storage, read_write> tileCounts: array<u32>;
@group(0) @binding(4) var<storage, read_write> activeIds: array<u32>;
@group(0) @binding(5) var<storage, read_write> globalKeys: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> globalValues: array<u32>;
@group(0) @binding(7) var<storage, read_write> stats: FrameStats;
@group(0) @binding(8) var<storage, read_write> draw: DrawIndirect;

fn orderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn rotateByQuaternion(v: vec3<f32>, rawQ: vec4<f32>) -> vec3<f32> {
  let q = rawQ / max(length(rawQ), 1e-8);
  let xyz = q.yzw;
  return v + 2.0 * cross(xyz, cross(xyz, v) + q.x * v);
}

fn screenDifferential(axisWorld: vec3<f32>, viewPosition: vec3<f32>) -> vec2<f32> {
  let axisView = (camera.view * vec4<f32>(axisWorld, 0.0)).xyz;
  let invZ2 = 1.0 / max(viewPosition.z * viewPosition.z, 1e-8);
  let fx = camera.projection[0][0] * camera.viewport.x * 0.5;
  let fy = camera.projection[1][1] * camera.viewport.y * 0.5;
  return vec2<f32>(
    fx * (-viewPosition.z * axisView.x + viewPosition.x * axisView.z) * invZ2,
    fy * ( viewPosition.z * axisView.y - viewPosition.y * axisView.z) * invZ2,
  );
}

@compute @workgroup_size(256)
fn resetFrame(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < camera.scene.y) {
    globalKeys[i] = vec2<u32>(INVALID);
    globalValues[i] = INVALID;
  }
  if (i < camera.scene.x) {
    tileCounts[i] = 0u;
  }
  if (i == 0u) {
    atomicStore(&stats.visibleCount, 0u);
    atomicStore(&stats.radiusFixedSum, 0u);
    atomicStore(&stats.duplicateCount, 0u);
    atomicStore(&stats.overflowCount, 0u);
    draw.vertexCount = 6u;
    atomicStore(&draw.instanceCount, 0u);
    draw.firstVertex = 0u;
    draw.firstInstance = 0u;
  }
}

@compute @workgroup_size(256)
fn projectAndCull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= camera.scene.x) { return; }

  let splat = splats[id];
  let world = vec4<f32>(splat.centerOpacity.xyz, 1.0);
  let view4 = camera.view * world;
  let clip = camera.viewProjection * world;
  if (clip.w <= 0.0 || view4.z >= -1e-4) {
    tileCounts[id] = 0u;
    return;
  }

  let ndc = clip.xyz / clip.w;
  let pixel = vec2<f32>(
    (ndc.x * 0.5 + 0.5) * camera.viewport.x,
    (0.5 - ndc.y * 0.5) * camera.viewport.y,
  );

  let axis0 = rotateByQuaternion(vec3<f32>(splat.scale.x, 0.0, 0.0), splat.rotation);
  let axis1 = rotateByQuaternion(vec3<f32>(0.0, splat.scale.y, 0.0), splat.rotation);
  let axis2 = rotateByQuaternion(vec3<f32>(0.0, 0.0, splat.scale.z), splat.rotation);
  let d0 = screenDifferential(axis0, view4.xyz);
  let d1 = screenDifferential(axis1, view4.xyz);
  let d2 = screenDifferential(axis2, view4.xyz);
  let covA = dot(vec3<f32>(d0.x, d1.x, d2.x), vec3<f32>(d0.x, d1.x, d2.x)) + 0.3;
  let covB = dot(vec3<f32>(d0.x, d1.x, d2.x), vec3<f32>(d0.y, d1.y, d2.y));
  let covC = dot(vec3<f32>(d0.y, d1.y, d2.y), vec3<f32>(d0.y, d1.y, d2.y)) + 0.3;
  let determinant = covA * covC - covB * covB;
  if (determinant <= 1e-8) {
    tileCounts[id] = 0u;
    return;
  }

  let midpoint = 0.5 * (covA + covC);
  let lambdaMax = midpoint + sqrt(max(0.0, midpoint * midpoint - determinant));
  let radius = max(1.5, ceil(3.0 * sqrt(lambdaMax)));
  let minPixel = clamp(floor(pixel - vec2<f32>(radius)), vec2<f32>(0.0), camera.viewport.xy);
  let maxPixel = clamp(ceil(pixel + vec2<f32>(radius)), vec2<f32>(0.0), camera.viewport.xy);
  if (maxPixel.x <= minPixel.x || maxPixel.y <= minPixel.y) {
    tileCounts[id] = 0u;
    return;
  }

  let minTile = vec2<u32>(minPixel) / TILE_SIZE;
  let maxTile = min(vec2<u32>(ceil(maxPixel / f32(TILE_SIZE))), camera.scene.zw);
  let touched = (maxTile.x - minTile.x) * (maxTile.y - minTile.y);
  let inverse = vec3<f32>(covC, -covB, covA) / determinant;
  projected[id].centerRadius = vec4<f32>(pixel, view4.z, radius);
  projected[id].conicOpacity = vec4<f32>(inverse, splat.centerOpacity.w);
  projected[id].color = splat.color;
  projected[id].tileBounds = vec4<u32>(minTile, maxTile);
  tileCounts[id] = touched;

  let slot = atomicAdd(&stats.visibleCount, 1u);
  atomicAdd(&stats.radiusFixedSum, u32(min(radius * 256.0, 4294967040.0)));
  atomicAdd(&stats.duplicateCount, touched);
  if (slot < camera.scene.y) {
    activeIds[slot] = id;
    globalKeys[slot] = vec2<u32>(orderedFloat(view4.z), 0u); // far-to-near
    globalValues[slot] = id;
    atomicAdd(&draw.instanceCount, 1u);
  } else {
    atomicAdd(&stats.overflowCount, 1u);
  }
}
