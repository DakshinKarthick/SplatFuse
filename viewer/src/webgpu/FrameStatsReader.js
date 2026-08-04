/** Copies the 16-byte projection reduction without stalling the animation loop. */
export class FrameStatsReader {
  constructor(device, statsBuffer) {
    this.device = device
    this.statsBuffer = statsBuffer
  }

  capture(encoder) {
    const readback = this.device.createBuffer({
      label: 'frame-stats/readback', size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    encoder.copyBufferToBuffer(this.statsBuffer, 0, readback, 0, 16)
    return {
      read: async () => {
        await this.device.queue.onSubmittedWorkDone()
        await readback.mapAsync(GPUMapMode.READ)
        const words = new Uint32Array(readback.getMappedRange().slice(0))
        readback.unmap()
        readback.destroy()
        const visibleCount = words[0]
        return {
          visibleCount,
          meanRadius: visibleCount ? words[1] / 256 / visibleCount : 0,
          duplicationFactor: visibleCount ? words[2] / visibleCount : 0,
          duplicateCount: words[2],
          overflowCount: words[3],
        }
      },
    }
  }
}
