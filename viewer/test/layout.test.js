import assert from 'node:assert/strict'
import test from 'node:test'
import { alignTo, nextPowerOfTwo, packSplats, previousPowerOfTwo, SPLAT_STRIDE } from '../src/webgpu/layout.js'

test('alignment and sort capacity helpers', () => {
  assert.equal(alignTo(65, 16), 80)
  assert.equal(nextPowerOfTwo(1), 1)
  assert.equal(nextPowerOfTwo(5_000_000), 8_388_608)
  assert.equal(previousPowerOfTwo(5_000_000), 4_194_304)
})

test('packs one aligned Gaussian record', () => {
  const packed = packSplats({
    count: 1,
    positions: new Float32Array([1, 2, 3]),
    scales: new Float32Array([4, 5, 6]),
    rotations: new Float32Array([1, 0, 0, 0]),
    colors: new Float32Array([0.2, 0.4, 0.6]),
    opacities: new Float32Array([0.8]),
  })
  assert.equal(packed.byteLength, SPLAT_STRIDE)
  assert.deepEqual(Array.from(packed.slice(0, 8)), [1, 2, 3, 0.800000011920929, 4, 5, 6, 0])
})
