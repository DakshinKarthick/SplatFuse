import assert from 'node:assert/strict'
import test from 'node:test'
import { BenchmarkHarness, buildBenchmarkMatrix, classifyHardware, SyntheticSceneGenerator } from '../src/webgpu/BenchmarkHarness.js'

test('benchmark controller spans every independent variable', () => {
  const cases = buildBenchmarkMatrix({ counts: [50_000], radii: [2, 24], duplication: [1], overdraw: [2, 32], resolutions: [[1280, 720]] })
  assert.equal(cases.length, 4)
  assert.equal(classifyHardware({ vendor: 'NVIDIA', device: 'RTX 4090' }), 'discrete')
})

test('synthetic scenes are deterministic and aligned', () => {
  const a = SyntheticSceneGenerator.generate(3)
  const b = SyntheticSceneGenerator.generate(3)
  assert.deepEqual(a.positions, b.positions)
  assert.equal(a.rotations.length, 12)
})

test('harness records both pipelines and exports CSV', async () => {
  const harness = new BenchmarkHarness({
    warmupFrames: 0,
    sampleFrames: 1,
    renderSample: async (_, pipeline) => ({ totalMs: pipeline === 'quad' ? 2 : 1, passes: {} }),
  })
  await harness.runCase({ gaussianCount: 10 })
  assert.equal(harness.rows.length, 2)
  assert.match(harness.toCSV(), /"pipeline"/)
})
