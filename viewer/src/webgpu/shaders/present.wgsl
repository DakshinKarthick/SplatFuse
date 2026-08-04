@group(0) @binding(0) var source: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
  );
  var out: VertexOutput;
  out.position = vec4<f32>(positions[index], 0.0, 1.0);
  out.uv = positions[index] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  let dimensions = vec2<f32>(textureDimensions(source));
  let coordinate = vec2<i32>(clamp(in.uv * dimensions, vec2<f32>(0.0), dimensions - 1.0));
  let value = textureLoad(source, coordinate, 0);
  return vec4<f32>(value.rgb, 1.0);
}
