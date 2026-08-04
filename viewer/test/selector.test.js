import assert from 'node:assert/strict'
import test from 'node:test'
import { AdaptiveSelector } from '../src/webgpu/AdaptiveSelector.js'

const workload = {
  gaussianCount: 100_000, meanRadius: 4, duplicationFactor: 2,
  depthComplexity: 8, width: 1280, height: 720, overflowCount: 0,
}

test('calibrates with an alternating ten-frame sweep', () => {
  const selector = new AdaptiveSelector()
  const scheduled = []
  for (let i = 0; i < 10; i++) {
    const pipeline = selector.choose(workload)
    scheduled.push(pipeline)
    selector.record(pipeline, { totalMs: pipeline === 'quad' ? 2 : 3 }, workload)
  }
  assert.deepEqual(scheduled, ['quad', 'tile', 'quad', 'tile', 'quad', 'tile', 'quad', 'tile', 'quad', 'tile'])
  assert.equal(selector.calibrated, true)
})

test('forces quad rendering after tile duplicate overflow', () => {
  const selector = new AdaptiveSelector({ calibrationFrames: 0 })
  assert.equal(selector.choose({ ...workload, overflowCount: 1 }), 'quad')
})
