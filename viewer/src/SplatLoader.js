/**
 * Phase 1 · SplatLoader
 * ============================================================================
 * WHAT THIS DOES
 *   Reads a 3D Gaussian Splatting `.ply` file off the network and turns it into
 *   plain flat typed arrays the renderer can hand straight to the GPU.
 *
 * WHAT A 3DGS SCENE *IS*
 *   Not a mesh. It's a big list of "gaussians" — soft 3D blobs. Each blob has:
 *     - a position (x, y, z)           where it sits in space
 *     - a scale    (3 numbers)         how big it is along each of its own axes
 *     - a rotation (4-number quaternion) how it's oriented
 *     - a color    (spherical harmonics) its color
 *     - an opacity (1 number)          how see-through it is
 *   Render tens of thousands of these overlapping and you get a photoreal scene.
 *
 * .PLY FILE STRUCTURE
 *   A `.ply` is [ ASCII header text ] + [ raw binary blob of all vertices ].
 *   The header ends with the exact line `end_header\n`; everything after that
 *   byte is packed little-endian floats, `count` vertices back-to-back, each
 *   vertex being the same fixed set of properties in the order the header lists.
 *   So parsing = read header → learn the layout → slice the binary by that layout.
 * ============================================================================
 */

// Spherical-harmonics band-0 constant = sqrt(1 / (4*pi)).
// 3DGS stores color as spherical harmonics (SH): a base color plus terms that
// make the color shift with viewing angle (shiny/reflective look). The "DC" term
// (f_dc_*) is the base, view-independent color. Recovering RGB from the DC
// coefficient is:  rgb = 0.5 + SH_C0 * dc.  We ignore the higher bands (f_rest_*)
// because Phase 1 draws flat, non-view-dependent dots — that's Phase 2's job.
const SH_C0 = 0.28209479177387814

// Byte size of each .ply property type, so we can compute where each field sits.
const TYPE_SIZES = {
  float: 4, float32: 4, double: 8,
  uchar: 1, uint8: 1, int: 4, int32: 4, short: 2, ushort: 2,
}

// clamp a value into [0, 1] (colors can decode slightly out of range)
function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v
}

// Scan the raw bytes for the literal marker `end_header\n` and return the byte
// index just past it — i.e. where the binary vertex data begins. We scan bytes
// (not decoded text) because decoding the whole multi-hundred-MB file as text
// just to find the header would be wasteful and could choke on the binary tail.
function findHeaderEnd(bytes) {
  const marker = new TextEncoder().encode('end_header\n')
  for (let i = 0; i <= bytes.length - marker.length; i++) {
    let match = true
    for (let j = 0; j < marker.length; j++) {
      if (bytes[i + j] !== marker[j]) { match = false; break }
    }
    if (match) return i + marker.length
  }
  throw new Error('.ply end_header not found — not a valid ply file')
}

/**
 * Read the ASCII header and work out the memory layout of one vertex.
 * Returns:
 *   headerEnd — byte where binary data starts
 *   count     — number of vertices (gaussians)
 *   stride    — bytes per vertex (sum of all property sizes)
 *   offsets   — map of property name → its byte offset WITHIN a vertex record
 *
 * Looking properties up by NAME (via `offsets`) instead of assuming a fixed order
 * is what makes this robust: nerfstudio, the INRIA reference, and other tools all
 * emit slightly different property orderings and extra fields.
 */
function parseHeader(bytes) {
  const headerEnd = findHeaderEnd(bytes)
  const text = new TextDecoder('ascii').decode(bytes.subarray(0, headerEnd))
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)

  if (!lines[0].startsWith('ply')) throw new Error('not a .ply file')
  // We only handle little-endian binary (what every 3DGS trainer exports).
  const formatLine = lines.find((l) => l.startsWith('format'))
  if (!formatLine || !formatLine.includes('binary_little_endian')) {
    throw new Error(`unsupported ply format: ${formatLine}`)
  }

  let count = 0
  const props = [] // ordered list of { type, name }, in file order
  for (const line of lines) {
    if (line.startsWith('element vertex')) {
      // e.g. "element vertex 452598"  → how many gaussians
      count = parseInt(line.split(/\s+/).pop(), 10)
    } else if (line.startsWith('property') && count > 0 && props.length < 10000) {
      // e.g. "property float scale_0"  → parts = [property, float, scale_0]
      const parts = line.split(/\s+/)
      props.push({ type: parts[1], name: parts[2] })
    } else if (line.startsWith('element') && !line.includes('vertex')) {
      break // a second element (e.g. faces) starts — its props aren't ours
    }
  }
  if (count === 0) throw new Error('.ply has no vertex element')

  // Walk the properties in order, accumulating a running byte offset. After this
  // loop `cursor` equals the total bytes per vertex (the "stride").
  const offsets = {}
  let cursor = 0
  for (const p of props) {
    if (!(p.type in TYPE_SIZES)) throw new Error(`unsupported ply property type: ${p.type}`)
    offsets[p.name] = cursor
    cursor += TYPE_SIZES[p.type]
  }

  return { headerEnd, count, stride: cursor, offsets }
}

/**
 * Fetch + parse a splat `.ply`.
 * @param {string} url
 * @returns {Promise<{count:number, positions:Float32Array, scales:Float32Array,
 *                     rotations:Float32Array, colors:Float32Array, opacities:Float32Array}>}
 * The outputs are FLAT arrays (positions is x0,y0,z0, x1,y1,z1, …) because that's
 * the shape WebGL wants to upload as instanced attributes — no array-of-objects.
 */
export async function loadSplats(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status} ${res.statusText}`)
  const buffer = await res.arrayBuffer() // the whole file as raw bytes in memory
  const bytes = new Uint8Array(buffer)

  const { headerEnd, count, stride, offsets } = parseHeader(bytes)

  const requiredBytes = headerEnd + count * stride
  if (requiredBytes > buffer.byteLength) {
    throw new Error(`truncated ply: expected ${requiredBytes} bytes, received ${buffer.byteLength}`)
  }

  // Resolve the byte offset of every field we care about, once, up front.
  // `need` throws a clear error if the file is missing a field we assume exists.
  const need = (name) => {
    if (!(name in offsets)) throw new Error(`.ply is missing expected property "${name}"`)
    return offsets[name]
  }
  const oX = need('x'), oY = need('y'), oZ = need('z')
  const oOpacity = need('opacity')
  const oS0 = need('scale_0'), oS1 = need('scale_1'), oS2 = need('scale_2')
  const oR0 = need('rot_0'), oR1 = need('rot_1'), oR2 = need('rot_2'), oR3 = need('rot_3')
  const oDc0 = need('f_dc_0'), oDc1 = need('f_dc_1'), oDc2 = need('f_dc_2')

  // DataView lets us read a float at an arbitrary byte offset with a chosen
  // endianness. We anchor it at `headerEnd` so offset 0 = first vertex's first byte.
  const view = new DataView(buffer, headerEnd)

  // Pre-allocate the flat output arrays (3 per vertex for vec3s, 4 for the quat).
  const positions = new Float32Array(count * 3)
  const scales = new Float32Array(count * 3)
  const rotations = new Float32Array(count * 4)
  const colors = new Float32Array(count * 3)
  const opacities = new Float32Array(count)

  for (let i = 0; i < count; i++) {
    const base = i * stride // byte offset of vertex i's record
    // getFloat32(offset, littleEndian=true). base + oX lands exactly on this
    // vertex's `x` field because oX is x's offset within one record.
    positions[i * 3] = view.getFloat32(base + oX, true)
    positions[i * 3 + 1] = view.getFloat32(base + oY, true)
    positions[i * 3 + 2] = view.getFloat32(base + oZ, true)

    // Scales are stored as LOG-scale and opacity as LOGIT (inverse-sigmoid). Why?
    // During training these must be free to be any real number, but the real
    // values must stay positive (a size) / within 0..1 (an alpha). exp() and
    // sigmoid() are the "activation" functions that map the stored raw floats
    // back to real-world values. We undo them here so the renderer gets usable
    // numbers.  scale_real = exp(scale_raw)
    scales[i * 3] = Math.exp(view.getFloat32(base + oS0, true))
    scales[i * 3 + 1] = Math.exp(view.getFloat32(base + oS1, true))
    scales[i * 3 + 2] = Math.exp(view.getFloat32(base + oS2, true))

    // Rotation quaternion (w, x, y, z order in the file), loaded unchanged.
    // The active WebGPU projection shader normalizes WXYZ before rotating the
    // scaled covariance axes, preserving the raw file components at this boundary.
    rotations[i * 4] = view.getFloat32(base + oR0, true)
    rotations[i * 4 + 1] = view.getFloat32(base + oR1, true)
    rotations[i * 4 + 2] = view.getFloat32(base + oR2, true)
    rotations[i * 4 + 3] = view.getFloat32(base + oR3, true)

    // Color from the SH DC term:  rgb = 0.5 + SH_C0 * dc, clamped to [0,1].
    colors[i * 3] = clamp01(0.5 + SH_C0 * view.getFloat32(base + oDc0, true))
    colors[i * 3 + 1] = clamp01(0.5 + SH_C0 * view.getFloat32(base + oDc1, true))
    colors[i * 3 + 2] = clamp01(0.5 + SH_C0 * view.getFloat32(base + oDc2, true))

    // Opacity via sigmoid: alpha = 1 / (1 + e^-raw), mapping any real number into 0..1.
    opacities[i] = 1 / (1 + Math.exp(-view.getFloat32(base + oOpacity, true)))
  }

  return { count, positions, scales, rotations, colors, opacities }
}
