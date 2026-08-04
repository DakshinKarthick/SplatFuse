/** Per-pass WebGPU timestamp queries with a CPU total-time fallback. */
export class Telemetry {
  constructor(device, enabled, maxPasses = 32) {
    this.device = device
    this.enabled = enabled && device.features.has('timestamp-query')
    this.maxPasses = maxPasses
    if (this.enabled) {
      this.querySet = device.createQuerySet({ label: 'telemetry/timestamps', type: 'timestamp', count: maxPasses * 2 })
      this.resolveBuffer = device.createBuffer({
        label: 'telemetry/resolve',
        size: maxPasses * 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    }
  }

  beginFrame(metadata = {}) {
    this.labels = []
    this.cpuStart = performance.now()
    this.metadata = metadata
  }

  timestampWrites(label) {
    if (!this.enabled) return undefined
    if (this.labels.length >= this.maxPasses) throw new Error(`Telemetry pass limit (${this.maxPasses}) exceeded.`)
    const index = this.labels.length * 2
    this.labels.push(label)
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: index,
      endOfPassWriteIndex: index + 1,
    }
  }

  finishFrame(encoder) {
    const cpuStart = this.cpuStart
    const metadata = this.metadata
    const labels = [...this.labels]
    if (!this.enabled || labels.length === 0) {
      return {
        read: async () => {
          await this.device.queue.onSubmittedWorkDone()
          return { metadata, source: 'cpu', totalMs: performance.now() - cpuStart, passes: {} }
        },
      }
    }

    const bytes = labels.length * 16
    const readback = this.device.createBuffer({
      label: 'telemetry/readback', size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    encoder.resolveQuerySet(this.querySet, 0, labels.length * 2, this.resolveBuffer, 0)
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, readback, 0, bytes)
    const device = this.device
    return {
      read: async () => {
        await device.queue.onSubmittedWorkDone()
        await readback.mapAsync(GPUMapMode.READ)
        const values = new BigUint64Array(readback.getMappedRange().slice(0))
        readback.unmap()
        readback.destroy()
        const passes = {}
        for (let i = 0; i < labels.length; i++) {
          const milliseconds = Number(values[i * 2 + 1] - values[i * 2]) / 1e6
          passes[labels[i]] = (passes[labels[i]] || 0) + milliseconds
        }
        const totalMs = Number(values[values.length - 1] - values[0]) / 1e6
        return { metadata, source: 'gpu-timestamp', totalMs, passes }
      },
    }
  }
}

export function timedPassDescriptor(telemetry, label) {
  const timestampWrites = telemetry?.timestampWrites(label)
  return timestampWrites ? { label, timestampWrites } : { label }
}
