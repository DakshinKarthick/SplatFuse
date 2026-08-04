import assert from 'node:assert/strict'
import test from 'node:test'
import { bitonicStages, compareKeyPair } from '../src/webgpu/sortPlan.js'

test('builds every bitonic merge stage', () => {
  assert.deepEqual(bitonicStages(8), [
    { j: 1, k: 2, length: 8 },
    { j: 2, k: 4, length: 8 },
    { j: 1, k: 4, length: 8 },
    { j: 4, k: 8, length: 8 },
    { j: 2, k: 8, length: 8 },
    { j: 1, k: 8, length: 8 },
  ])
  assert.throws(() => bitonicStages(7), /power of two/)
})

test('compares 64-bit pairs by high then low word', () => {
  const keys = [[9, 1], [2, 0], [1, 1], [0xffffffff, 0xffffffff]]
  keys.sort(compareKeyPair)
  assert.deepEqual(keys, [[2, 0], [1, 1], [9, 1], [0xffffffff, 0xffffffff]])
})
