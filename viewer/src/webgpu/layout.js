export const SPLAT_FLOATS = 16
export const SPLAT_STRIDE = SPLAT_FLOATS * Float32Array.BYTES_PER_ELEMENT
export const PROJECTED_STRIDE = 64
export const TILE_SIZE = 16

export function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment
}

export function nextPowerOfTwo(value) {
  if (value <= 1) return 1
  return 2 ** Math.ceil(Math.log2(value))
}

export function previousPowerOfTwo(value) {
  if (value < 1) return 0
  return 2 ** Math.floor(Math.log2(value))
}

/**
 * Convert structure-of-arrays PLY output into a WGSL-friendly array-of-structs.
 * Every record is four vec4<f32>s (64 bytes): center/opacity, scale/pad,
 * quaternion, and RGB/pad. This satisfies storage-buffer alignment everywhere.
 */
export function packSplats({ count, positions, scales, rotations, colors, opacities }) {
  const expected = [
    ['positions', positions, count * 3],
    ['scales', scales, count * 3],
    ['rotations', rotations, count * 4],
    ['colors', colors, count * 3],
    ['opacities', opacities, count],
  ]
  for (const [name, array, length] of expected) {
    if (!(array instanceof Float32Array) || array.length !== length) {
      throw new TypeError(`${name} must be Float32Array(${length})`)
    }
  }

  const packed = new Float32Array(count * SPLAT_FLOATS)
  for (let i = 0; i < count; i++) {
    const dst = i * SPLAT_FLOATS
    const i3 = i * 3
    const i4 = i * 4
    packed[dst] = positions[i3]
    packed[dst + 1] = positions[i3 + 1]
    packed[dst + 2] = positions[i3 + 2]
    packed[dst + 3] = opacities[i]
    packed[dst + 4] = scales[i3]
    packed[dst + 5] = scales[i3 + 1]
    packed[dst + 6] = scales[i3 + 2]
    packed[dst + 7] = 0
    packed[dst + 8] = rotations[i4]
    packed[dst + 9] = rotations[i4 + 1]
    packed[dst + 10] = rotations[i4 + 2]
    packed[dst + 11] = rotations[i4 + 3]
    packed[dst + 12] = colors[i3]
    packed[dst + 13] = colors[i3 + 1]
    packed[dst + 14] = colors[i3 + 2]
    packed[dst + 15] = 0
  }
  return packed
}
