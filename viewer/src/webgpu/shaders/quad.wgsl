struct Camera {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
  viewport: vec4<f32>,
  scene: vec4<u32>,
};

struct Projected {
  centerRadius: vec4<f32>,
  conicOpacity: vec4<f32>,
  color: vec4<f32>,
  tileBounds: vec4<u32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) pixelDelta: vec2<f32>,
  @location(1) @interpolate(flat) conicOpacity: vec4<f32>,
  @location(2) @interpolate(flat) color: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> sortedIds: array<u32>;

const CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
);

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let splatId = sortedIds[instanceIndex];
  let item = projected[splatId];
  let delta = CORNERS[vertexIndex] * item.centerRadius.w;
  let pixel = item.centerRadius.xy + delta;
  let ndc = vec2<f32>(
    pixel.x * camera.viewport.z * 2.0 - 1.0,
    1.0 - pixel.y * camera.viewport.w * 2.0,
  );
  var out: VertexOutput;
  out.position = vec4<f32>(ndc, 0.0, 1.0);
  out.pixelDelta = delta;
  out.conicOpacity = item.conicOpacity;
  out.color = item.color.xyz;
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  let conic = in.conicOpacity.xyz;
  let d = in.pixelDelta;
  let exponent = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);
  if (exponent > 0.0) { discard; }
  let alpha = min(0.99, in.conicOpacity.w * exp(exponent));
  if (alpha < (1.0 / 255.0)) { discard; }
  return vec4<f32>(in.color * alpha, alpha);
}
