const DEFAULT_COUNTS = [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
const DEFAULT_RESOLUTIONS = [[1280, 720], [1920, 1080], [2560, 1440], [3840, 2160]]

export function buildBenchmarkMatrix({
  counts = DEFAULT_COUNTS,
  radii = [2, 8, 24, 64],
  duplication = [1, 4, 16],
  overdraw = [2, 8, 32, 128],
  resolutions = DEFAULT_RESOLUTIONS,
} = {}) {
  const matrix = []
  for (const gaussianCount of counts) {
    for (const meanRadius of radii) {
      for (const duplicationFactor of duplication) {
        for (const depthComplexity of overdraw) {
          for (const [width, height] of resolutions) {
            matrix.push({ gaussianCount, meanRadius, duplicationFactor, depthComplexity, width, height })
          }
        }
      }
    }
  }
  return matrix
}

export function classifyHardware(adapterInfo = {}) {
  const text = Object.values(adapterInfo).join(' ').toLowerCase()
  if (/apple|adreno|mali|powervr/.test(text)) return 'mobile-or-unified'
  if (/iris|uhd|apu|integrated/.test(text)) return 'integrated'
  if (/nvidia|radeon|geforce|rx\s|rtx\s|arc\s/.test(text)) return 'discrete'
  return 'unknown'
}

export class SyntheticSceneGenerator {
  static generate(count, { radiusScale = 1, depthLayers = 8, seed = 0x51a7f00d } = {}) {
    let state = seed >>> 0
    const random = () => ((state = (1664525 * state + 1013904223) >>> 0) / 0x100000000)
    const positions = new Float32Array(count * 3)
    const scales = new Float32Array(count * 3)
    const rotations = new Float32Array(count * 4)
    const colors = new Float32Array(count * 3)
    const opacities = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      const i3 = i * 3
      const i4 = i * 4
      positions.set([(random() - 0.5) * 4, (random() - 0.5) * 3, -1 - (i % depthLayers) * 0.08], i3)
      scales.set([radiusScale, radiusScale * (0.5 + random()), radiusScale], i3)
      rotations.set([1, 0, 0, 0], i4)
      colors.set([random(), random(), random()], i3)
      opacities[i] = 0.05 + random() * 0.5
    }
    return { count, positions, scales, rotations, colors, opacities }
  }
}

export class BenchmarkHarness {
  constructor({ renderSample, warmupFrames = 3, sampleFrames = 10, adapterInfo = {} }) {
    this.renderSample = renderSample
    this.warmupFrames = warmupFrames
    this.sampleFrames = sampleFrames
    this.adapterInfo = adapterInfo
    this.rows = []
  }

  async runCase(configuration, pipelines = ['quad', 'tile']) {
    for (const pipeline of pipelines) {
      for (let i = 0; i < this.warmupFrames; i++) await this.renderSample(configuration, pipeline, false)
      for (let i = 0; i < this.sampleFrames; i++) {
        const measurement = await this.renderSample(configuration, pipeline, true)
        this.rows.push({
          ...configuration,
          pipeline,
          hardwareProfile: classifyHardware(this.adapterInfo),
          sample: i,
          ...measurement,
          passes: JSON.stringify(measurement.passes || {}),
        })
      }
    }
    return this.rows
  }

  toJSON() {
    return JSON.stringify({ adapter: this.adapterInfo, rows: this.rows }, null, 2)
  }

  toCSV() {
    if (!this.rows.length) return ''
    const columns = [...new Set(this.rows.flatMap(Object.keys))]
    const escape = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`
    return [columns.map(escape).join(','), ...this.rows.map((row) => columns.map((key) => escape(row[key])).join(','))].join('\n')
  }
}
