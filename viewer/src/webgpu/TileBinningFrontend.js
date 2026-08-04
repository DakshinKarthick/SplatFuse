import binningShader from './shaders/tile-binning.wgsl?raw'
import { GpuPrefixScan } from './GpuPrefixScan.js'
import { nextPowerOfTwo, previousPowerOfTwo } from './layout.js'

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

export class TileBinningFrontend {
  constructor(device, scene, cameraBuffer, maximumDuplication = 8) {
    this.device = device
    this.scene = scene
    const bindingLimit = Math.floor(device.limits.maxStorageBufferBindingSize / 8)
    const dispatchLimit = device.limits.maxComputeWorkgroupsPerDimension * 256
    const requested = nextPowerOfTwo(Math.max(1, scene.count * maximumDuplication))
    this.capacity = Math.min(requested, previousPowerOfTwo(Math.min(bindingLimit, dispatchLimit)))
    if (this.capacity < 1) throw new Error('Adapter cannot allocate a tile key buffer.')

    this.keys = device.createBuffer({ label: 'tile/keys', size: this.capacity * 8, usage: STORAGE })
    this.values = device.createBuffer({ label: 'tile/values', size: this.capacity * 4, usage: STORAGE })
    this.params = device.createBuffer({
      label: 'tile/binning-parameters', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.params, 0, new Uint32Array([this.capacity, 0, 0, 0]))
    this.scan = new GpuPrefixScan(device, scene.count, scene.tileCounts, scene.tileOffsets)

    const module = device.createShaderModule({ label: 'tile duplication WGSL', code: binningShader })
    this.resetPipeline = device.createComputePipeline({
      label: 'tile/reset-bins', layout: 'auto', compute: { module, entryPoint: 'resetBins' },
    })
    this.binPipeline = device.createComputePipeline({
      label: 'tile/duplicate-and-key', layout: 'auto', compute: { module, entryPoint: 'duplicateIntoTiles' },
    })
    this.resetBindings = device.createBindGroup({
      layout: this.resetPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.params } },
        { binding: 5, resource: { buffer: this.keys } },
        { binding: 6, resource: { buffer: this.values } },
      ],
    })
    this.binBindings = device.createBindGroup({
      layout: this.binPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: this.params } },
        { binding: 2, resource: { buffer: scene.projected } },
        { binding: 3, resource: { buffer: scene.tileCounts } },
        { binding: 4, resource: { buffer: scene.tileOffsets } },
        { binding: 5, resource: { buffer: this.keys } },
        { binding: 6, resource: { buffer: this.values } },
        { binding: 7, resource: { buffer: scene.stats } },
      ],
    })
  }

  encode(encoder) {
    let pass = encoder.beginComputePass({ label: 'reset tile key/value bins' })
    pass.setPipeline(this.resetPipeline)
    pass.setBindGroup(0, this.resetBindings)
    pass.dispatchWorkgroups(Math.ceil(this.capacity / 256))
    pass.end()

    this.scan.encode(encoder)

    pass = encoder.beginComputePass({ label: 'duplicate splats into touched tiles' })
    pass.setPipeline(this.binPipeline)
    pass.setBindGroup(0, this.binBindings)
    pass.dispatchWorkgroups(Math.ceil(this.scene.count / 256))
    pass.end()
  }
}
