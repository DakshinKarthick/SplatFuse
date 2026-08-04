struct SortParameters {
  j: u32,
  k: u32,
  length: u32,
  _padding: u32,
};

@group(0) @binding(0) var<uniform> params: SortParameters;
@group(0) @binding(1) var<storage, read_write> keys: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;

fn keyLess(a: vec2<u32>, b: vec2<u32>) -> bool {
  // High word is the primary key. Low word is depth or another sub-key.
  return (a.y < b.y) || (a.y == b.y && a.x < b.x);
}

@compute @workgroup_size(256)
fn bitonicStep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.length) { return; }
  let partner = i ^ params.j;
  if (partner <= i || partner >= params.length) { return; }

  let ascending = (i & params.k) == 0u;
  let a = keys[i];
  let b = keys[partner];
  let swap = select(keyLess(a, b), keyLess(b, a), ascending);
  if (swap) {
    keys[i] = b;
    keys[partner] = a;
    let value = values[i];
    values[i] = values[partner];
    values[partner] = value;
  }
}
