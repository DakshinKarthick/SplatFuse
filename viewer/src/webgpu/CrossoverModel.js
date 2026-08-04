const LOG_FLOOR = 2

export function quadFeatures({ gaussianCount, meanRadius, depthComplexity, width, height }) {
  const n = Math.max(1, gaussianCount)
  const pixels = width * height
  return [1, n * Math.log2(Math.max(LOG_FLOOR, n)), n * meanRadius ** 2, pixels * depthComplexity]
}

export function tileFeatures({
  gaussianCount, duplicationFactor, depthComplexity, width, height, saturationDepth = 64,
}) {
  const duplicated = Math.max(1, gaussianCount * duplicationFactor)
  const tiles = Math.ceil(width / 16) * Math.ceil(height / 16)
  return [
    1,
    duplicated * Math.log2(Math.max(LOG_FLOOR, duplicated)),
    tiles * Math.min(depthComplexity, saturationDepth) * 256,
    duplicated,
  ]
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length
  const augmented = matrix.map((row, i) => [...row, vector[i]])
  for (let column = 0; column < n; column++) {
    let pivot = column
    for (let row = column + 1; row < n; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row
    }
    ;[augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]]
    const divisor = augmented[column][column]
    if (Math.abs(divisor) < 1e-12) throw new Error('Benchmark matrix is singular; collect more varied samples.')
    for (let j = column; j <= n; j++) augmented[column][j] /= divisor
    for (let row = 0; row < n; row++) {
      if (row === column) continue
      const factor = augmented[row][column]
      for (let j = column; j <= n; j++) augmented[row][j] -= factor * augmented[column][j]
    }
  }
  return augmented.map((row) => row[n])
}

function ridgeFit(rows, featureFunction, lambda) {
  const raw = rows.map(featureFunction)
  const scale = raw[0].map((_, column) => column === 0 ? 1 : Math.max(1, ...raw.map((x) => Math.abs(x[column]))))
  const x = raw.map((features) => features.map((value, column) => value / scale[column]))
  const size = x[0].length
  const xtx = Array.from({ length: size }, () => Array(size).fill(0))
  const xty = Array(size).fill(0)
  for (let row = 0; row < x.length; row++) {
    for (let i = 0; i < size; i++) {
      xty[i] += x[row][i] * rows[row].totalMs
      for (let j = 0; j < size; j++) xtx[i][j] += x[row][i] * x[row][j]
    }
  }
  for (let i = 1; i < size; i++) xtx[i][i] += lambda
  return { coefficients: solveLinearSystem(xtx, xty), scale }
}

function evaluateFit(fit, features) {
  return Math.max(0, features.reduce((sum, value, i) => sum + fit.coefficients[i] * value / fit.scale[i], 0))
}

const DEFAULT_FITS = {
  quad: { coefficients: [0.08, 3.0, 2.0, 1.5], scale: [1, 30_000_000, 20_000_000, 100_000_000] },
  tile: { coefficients: [0.12, 3.0, 2.0, 1.0], scale: [1, 60_000_000, 150_000_000, 5_000_000] },
}

/** Analytical feature model whose coefficients are fitted per hardware profile. */
export class CrossoverModel {
  constructor({ saturationDepth = 64, fits = {} } = {}) {
    this.saturationDepth = saturationDepth
    this.fits = { ...DEFAULT_FITS, ...fits }
  }

  fit(rows, { lambda = 1e-6, hardwareProfile } = {}) {
    const selected = hardwareProfile ? rows.filter((row) => row.hardwareProfile === hardwareProfile) : rows
    for (const pipeline of ['quad', 'tile']) {
      const samples = selected.filter((row) => row.pipeline === pipeline && Number.isFinite(row.totalMs))
      if (samples.length < 4) throw new Error(`At least four varied ${pipeline} measurements are required.`)
      const features = pipeline === 'quad'
        ? quadFeatures
        : (sample) => tileFeatures({ ...sample, saturationDepth: this.saturationDepth })
      this.fits[pipeline] = ridgeFit(samples, features, lambda)
    }
    return this
  }

  predict(workload) {
    const quadMs = evaluateFit(this.fits.quad, quadFeatures(workload))
    const tileMs = evaluateFit(this.fits.tile, tileFeatures({ ...workload, saturationDepth: this.saturationDepth }))
    return { quadMs, tileMs }
  }

  boundary(workload) {
    const { quadMs, tileMs } = this.predict(workload)
    const differenceMs = quadMs - tileMs // F(N,r,D,k,res,arch) = 0 is crossover
    const confidence = Math.min(1, Math.abs(differenceMs) / Math.max(0.1, Math.min(quadMs, tileMs)))
    return { quadMs, tileMs, differenceMs, confidence, pipeline: differenceMs <= 0 ? 'quad' : 'tile' }
  }

  evaluate(rows) {
    const errors = rows.map((row) => Math.abs(this.predict(row)[`${row.pipeline}Ms`] - row.totalMs))
    return { samples: errors.length, meanAbsoluteErrorMs: errors.reduce((a, b) => a + b, 0) / Math.max(1, errors.length) }
  }

  toJSON() {
    return { version: 1, saturationDepth: this.saturationDepth, fits: this.fits }
  }

  static fromJSON(value) {
    return new CrossoverModel(typeof value === 'string' ? JSON.parse(value) : value)
  }
}
