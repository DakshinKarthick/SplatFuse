const INVALID: u32 = 0xffffffffu;
const BATCH_SIZE: u32 = 256u;

struct RasterParameters {
  width: u32,
  height: u32,
  tilesX: u32,
  tilesY: u32,
  keyCapacity: u32,
  _padding: vec3<u32>,
};
struct Projected {
  centerRadius: vec4<f32>,
  conicOpacity: vec4<f32>,
  color: vec4<f32>,
  tileBounds: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: RasterParameters;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> sortedValues: array<u32>;
@group(0) @binding(3) var<storage, read_write> tileRanges: array<vec2<u32>>;
@group(0) @binding(4) var outputImage: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read> sortedKeys: array<vec2<u32>>;

var<workgroup> sharedCenter: array<vec4<f32>, 256>;
var<workgroup> sharedConic: array<vec4<f32>, 256>;
var<workgroup> sharedColor: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn resetRanges(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tileCount = params.tilesX * params.tilesY;
  if (gid.x < tileCount) tileRanges[gid.x] = vec2<u32>(INVALID, 0u);
}

@compute @workgroup_size(256)
fn detectRanges(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.keyCapacity) { return; }
  let tile = sortedKeys[i].y;
  if (tile == INVALID || tile >= params.tilesX * params.tilesY) { return; }

  if (i == 0u || sortedKeys[i - 1u].y != tile) tileRanges[tile].x = i;
  if (i + 1u == params.keyCapacity || sortedKeys[i + 1u].y != tile) tileRanges[tile].y = i + 1u;
}

@compute @workgroup_size(16, 16, 1)
fn rasterizeTiles(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let lane = local.y * 16u + local.x;
  let pixel = group.xy * 16u + local.xy;
  let validPixel = pixel.x < params.width && pixel.y < params.height;
  let tile = group.y * params.tilesX + group.x;
  let range = tileRanges[tile];
  var color = vec3<f32>(0.0);
  var transmittance = 1.0;
  var terminated = !validPixel || range.x == INVALID;

  var batchStart = range.x;
  loop {
    if (range.x == INVALID || batchStart >= range.y) { break; }
    let source = batchStart + lane;
    if (source < range.y) {
      let item = projected[sortedValues[source]];
      sharedCenter[lane] = item.centerRadius;
      sharedConic[lane] = item.conicOpacity;
      sharedColor[lane] = item.color;
    }
    workgroupBarrier();

    let loaded = min(BATCH_SIZE, range.y - batchStart);
    if (!terminated) {
      for (var j = 0u; j < loaded; j++) {
        let delta = vec2<f32>(pixel) + vec2<f32>(0.5) - sharedCenter[j].xy;
        let conic = sharedConic[j];
        let exponent = -0.5 * (
          conic.x * delta.x * delta.x + 2.0 * conic.y * delta.x * delta.y + conic.z * delta.y * delta.y
        );
        if (exponent <= 0.0) {
          let alpha = min(0.99, conic.w * exp(exponent));
          if (alpha >= (1.0 / 255.0)) {
            color += sharedColor[j].xyz * alpha * transmittance;
            transmittance *= 1.0 - alpha;
            if (transmittance < 0.0001) {
              terminated = true;
              break;
            }
          }
        }
      }
    }
    workgroupBarrier();
    batchStart += BATCH_SIZE;
  }

  if (validPixel) textureStore(outputImage, vec2<i32>(pixel), vec4<f32>(color, 1.0 - transmittance));
}
