import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { loadSplats } from './SplatLoader.js'
import { SyntheticSceneGenerator } from './webgpu/BenchmarkHarness.js'
import { GpuContext } from './webgpu/GpuContext.js'
import { SplatFuseRenderer } from './webgpu/SplatFuseRenderer.js'

const canvas = document.querySelector('canvas.webgl')
const status = document.querySelector('#status')
const query = new URLSearchParams(location.search)

function resize() {
  const scale = Math.min(devicePixelRatio, 2)
  canvas.width = Math.max(1, Math.floor(innerWidth * scale))
  canvas.height = Math.max(1, Math.floor(innerHeight * scale))
  canvas.style.width = `${innerWidth}px`
  canvas.style.height = `${innerHeight}px`
}

function robustFrame(positions, count) {
  const sampleCount = Math.min(50_000, count)
  const step = Math.max(1, Math.floor(count / sampleCount))
  const axes = [[], [], []]
  for (let i = 0; i < count; i += step) {
    for (let axis = 0; axis < 3; axis++) axes[axis].push(positions[i * 3 + axis])
  }
  const percentile = (values, fraction) => {
    values.sort((a, b) => a - b)
    return values[Math.min(values.length - 1, Math.floor(values.length * fraction))]
  }
  const low = axes.map((axis) => percentile(axis, 0.05))
  const high = axes.map((axis) => percentile(axis, 0.95))
  const center = new THREE.Vector3(...low.map((value, i) => (value + high[i]) * 0.5))
  const radius = Math.max(...high.map((value, i) => value - low[i])) * 0.5 || 1
  return { center, radius }
}

function applyXFlip(splats) {
  for (let i = 0; i < splats.count; i++) {
    splats.positions[i * 3 + 1] *= -1
    splats.positions[i * 3 + 2] *= -1
    const offset = i * 4
    const [w, x, y, z] = splats.rotations.subarray(offset, offset + 4)
    splats.rotations.set([-x, w, -z, y], offset) // (180° around X) * q
  }
}

async function loadScene() {
  const syntheticCount = Number(query.get('synthetic') || 0)
  if (syntheticCount > 0) {
    return { splats: SyntheticSceneGenerator.generate(syntheticCount, { radiusScale: 0.02 }), source: `synthetic:${syntheticCount}` }
  }
  const url = query.get('scene') || '/scenes/mypic1.ply'
  try {
    return { splats: await loadSplats(url), source: url }
  } catch (error) {
    console.warn(`${url} unavailable; using the deterministic 50k research scene`, error)
    return {
      splats: SyntheticSceneGenerator.generate(50_000, { radiusScale: 0.02 }),
      source: 'synthetic fallback (50k)',
    }
  }
}

async function start() {
  resize()
  status.textContent = 'SplatFuse · requesting WebGPU and loading scene…'
  const [gpu, loaded] = await Promise.all([GpuContext.create(canvas), loadScene()])
  const { splats, source } = loaded
  if (query.get('flip') === '1') applyXFlip(splats)

  const camera = new THREE.PerspectiveCamera(60, canvas.width / canvas.height, 0.001, 10_000)
  const controls = new OrbitControls(camera, canvas)
  controls.enableDamping = true
  const { center, radius } = robustFrame(splats.positions, splats.count)
  controls.target.copy(center)
  camera.position.copy(center).add(new THREE.Vector3(0, 0, radius * 2.2))
  camera.near = Math.max(0.001, radius * 0.002)
  camera.far = Math.max(100, radius * 100)
  camera.updateProjectionMatrix()
  controls.update()

  let lastMeasurement = null
  const renderer = new SplatFuseRenderer(gpu, splats, {
    mode: query.get('pipeline') || 'auto',
    maximumDuplication: Number(query.get('maxDup') || 8),
    depthComplexity: Number(query.get('k') || 16),
    onMeasurement: (value) => { lastMeasurement = value },
  })

  addEventListener('resize', () => {
    resize()
    gpu.configure()
    camera.aspect = canvas.width / canvas.height
    camera.updateProjectionMatrix()
  })

  function frame() {
    controls.update()
    camera.updateMatrixWorld(true)
    const current = renderer.render(camera)
    const stats = lastMeasurement?.workload || current.workload
    const selection = lastMeasurement?.selector || current.selector
    const measured = lastMeasurement?.measurement
    status.textContent = [
      `SplatFuse · ${current.pipeline.toUpperCase()} · ${source}`,
      `${splats.count.toLocaleString()} loaded · ${stats.gaussianCount.toLocaleString()} visible`,
      `radius ${stats.meanRadius.toFixed(2)} px · D ${stats.duplicationFactor.toFixed(2)} · k ${stats.depthComplexity}`,
      measured ? `${measured.totalMs.toFixed(3)} ms (${measured.source})` : 'collecting timing…',
      selection.calibrated
        ? `calibrated · predicted Q ${selection.predictedMs.quad.toFixed(2)} / T ${selection.predictedMs.tile.toFixed(2)} ms`
        : `calibrating ${selection.calibrationSamples}/10`,
      stats.overflowCount ? `tile overflow ${stats.overflowCount.toLocaleString()} · forcing quads` : '',
    ].filter(Boolean).join('\n')
    requestAnimationFrame(frame)
  }
  frame()
}

start().catch((error) => {
  console.error(error)
  status.style.color = '#fca5a5'
  status.textContent = `SplatFuse could not start\n${error.stack || error.message}`
})
