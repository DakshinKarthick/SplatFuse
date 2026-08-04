import assert from 'node:assert/strict'
import test from 'node:test'
import { CrossoverModel, quadFeatures, tileFeatures } from '../src/webgpu/CrossoverModel.js'

const workloads = Array.from({ length: 12 }, (_, i) => ({
  gaussianCount: 50_000 * (i + 1),
  meanRadius: 2 + (i % 4) * 7,
  duplicationFactor: 1 + (i % 3) * 3,
  depthComplexity: 2 + i * 3,
  width: i % 2 ? 1920 : 1280,
  height: i % 2 ? 1080 : 720,
}))

test('fits empirical coefficients and recreates measured costs', () => {
  const dot = (values, weights) => values.reduce((sum, value, i) => sum + value * weights[i], 0)
  const quadWeights = [0.2, 1e-8, 2e-8, 3e-9]
  const tileWeights = [0.3, 2e-8, 1e-9, 4e-8]
  const rows = workloads.flatMap((workload) => [
    { ...workload, pipeline: 'quad', totalMs: dot(quadFeatures(workload), quadWeights) },
    { ...workload, pipeline: 'tile', totalMs: dot(tileFeatures(workload), tileWeights) },
  ])
  const model = new CrossoverModel().fit(rows, { lambda: 1e-12 })
  assert.ok(model.evaluate(rows).meanAbsoluteErrorMs < 1e-4)
})

test('reports signed crossover and a selected pipeline', () => {
  const result = new CrossoverModel().boundary(workloads[0])
  assert.equal(result.differenceMs, result.quadMs - result.tileMs)
  assert.ok(['quad', 'tile'].includes(result.pipeline))
})
