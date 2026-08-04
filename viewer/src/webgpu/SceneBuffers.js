import { nextPowerOfTwo, packSplats, PROJECTED_STRIDE, SPLAT_STRIDE } from './layout.js'

const STORAGE_COPY = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC

function createBuffer(device, label, size, usage = STORAGE_COPY) {
  return device.createBuffer({ label, size: Math.max(4, size), usage })
}

export class SceneBuffers {
  static fromSplats(device, splats) {
    return new SceneBuffers(device, splats)
  }

  constructor(device, splats) {
    this.device = device
    this.count = splats.count
    this.sortCapacity = nextPowerOfTwo(Math.max(1, this.count))

    const sceneBytes = this.count * SPLAT_STRIDE
    const projectedBytes = this.count * PROJECTED_STRIDE
    const maxBinding = device.limits.maxStorageBufferBindingSize
    if (sceneBytes > maxBinding || projectedBytes > maxBinding) {
      throw new Error(
        `Scene needs ${Math.max(sceneBytes, projectedBytes).toLocaleString()} bytes in one storage ` +
        `binding; this adapter permits ${maxBinding.toLocaleString()}.`,
      )
    }

    this.splats = createBuffer(device, 'scene/splats', sceneBytes)
    device.queue.writeBuffer(this.splats, 0, packSplats(splats))

    this.projected = createBuffer(device, 'scene/projected', projectedBytes)
    this.tileCounts = createBuffer(device, 'scene/tile-counts', this.count * 4)
    this.tileOffsets = createBuffer(device, 'scene/tile-offsets', this.count * 4)
    this.activeIds = createBuffer(device, 'scene/active-ids', this.count * 4)
    this.globalKeys = createBuffer(device, 'quad/depth-keys', this.sortCapacity * 8)
    this.globalValues = createBuffer(device, 'quad/splat-ids', this.sortCapacity * 4)
    this.stats = createBuffer(device, 'scene/frame-stats', 16)
    this.indirectDraw = createBuffer(
      device,
      'quad/indirect-draw',
      16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    )
  }

  destroy() {
    for (const name of [
      'splats', 'projected', 'tileCounts', 'tileOffsets', 'activeIds',
      'globalKeys', 'globalValues', 'stats', 'indirectDraw',
    ]) this[name].destroy()
  }
}
