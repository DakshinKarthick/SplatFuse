import { CrossoverModel } from './CrossoverModel.js'

function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

/** Ten-frame device calibration followed by a hysteretic per-frame decision. */
export class AdaptiveSelector {
  constructor({ model = new CrossoverModel(), calibrationFrames = 10, hysteresis = 0.08, mode = 'auto' } = {}) {
    this.model = model
    this.calibrationFrames = calibrationFrames
    this.hysteresis = hysteresis
    this.mode = mode
    this.scheduledCalibrationFrames = 0
    this.samples = { quad: [], tile: [] }
    this.correction = { quad: 1, tile: 1 }
    this.current = 'quad'
  }

  get calibrated() {
    return this.samples.quad.length + this.samples.tile.length >= this.calibrationFrames
  }

  choose(workload) {
    if (this.mode === 'quad' || this.mode === 'tile') return this.mode
    if (workload.overflowCount > 0) return (this.current = 'quad')
    if (this.scheduledCalibrationFrames < this.calibrationFrames) {
      const pipeline = this.scheduledCalibrationFrames % 2 === 0 ? 'quad' : 'tile'
      this.scheduledCalibrationFrames++
      return (this.current = pipeline)
    }

    const prediction = this.model.boundary(workload)
    const cost = {
      quad: prediction.quadMs * this.correction.quad,
      tile: prediction.tileMs * this.correction.tile,
    }
    const candidate = cost.quad <= cost.tile ? 'quad' : 'tile'
    if (candidate !== this.current) {
      const improvement = (cost[this.current] - cost[candidate]) / Math.max(0.001, cost[this.current])
      if (improvement >= this.hysteresis) this.current = candidate
    }
    return this.current
  }

  record(pipeline, measurement, workload) {
    if (!measurement || !Number.isFinite(measurement.totalMs) || !['quad', 'tile'].includes(pipeline)) return
    const predicted = this.model.predict(workload)[`${pipeline}Ms`]
    const ratio = measurement.totalMs / Math.max(0.001, predicted)
    if (this.samples[pipeline].length < this.calibrationFrames / 2) {
      this.samples[pipeline].push(ratio)
      if (this.samples[pipeline].length === this.calibrationFrames / 2) {
        this.correction[pipeline] = median(this.samples[pipeline])
      }
    } else {
      // Slow EWMA adaptation tracks thermal/power changes without oscillation.
      this.correction[pipeline] = this.correction[pipeline] * 0.95 + ratio * 0.05
    }
  }

  snapshot(workload) {
    const prediction = this.model.predict(workload)
    return {
      mode: this.mode,
      current: this.current,
      calibrated: this.calibrated,
      calibrationSamples: this.samples.quad.length + this.samples.tile.length,
      correction: { ...this.correction },
      predictedMs: {
        quad: prediction.quadMs * this.correction.quad,
        tile: prediction.tileMs * this.correction.tile,
      },
    }
  }
}
