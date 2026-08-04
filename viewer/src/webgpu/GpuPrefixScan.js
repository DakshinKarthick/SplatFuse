import scanShader from './shaders/prefix-scan.wgsl?raw'
import { timedPassDescriptor } from './Telemetry.js'

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

function buffer(device, label, words, usage = STORAGE) {
  return device.createBuffer({ label, size: Math.max(4, words * 4), usage })
}

/** Hierarchical Blelloch exclusive scan using only portable workgroup memory. */
export class GpuPrefixScan {
  constructor(device, length, input, output) {
    this.device = device
    this.length = length
    const module = device.createShaderModule({ label: 'exclusive prefix scan WGSL', code: scanShader })
    this.scanPipeline = device.createComputePipeline({
      label: 'prefix-scan/blocks', layout: 'auto', compute: { module, entryPoint: 'scanBlocks' },
    })
    this.addPipeline = device.createComputePipeline({
      label: 'prefix-scan/add-offsets', layout: 'auto', compute: { module, entryPoint: 'addBlockOffsets' },
    })

    this.levels = []
    let levelLength = length
    let levelInput = input
    let levelOutput = output
    while (true) {
      const blocks = Math.ceil(levelLength / 256)
      const sums = buffer(device, `prefix-scan/sums-${this.levels.length}`, blocks)
      const params = buffer(device, `prefix-scan/params-${this.levels.length}`, 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      device.queue.writeBuffer(params, 0, new Uint32Array([levelLength, 0, 0, 0]))
      const level = { length: levelLength, blocks, input: levelInput, output: levelOutput, sums, params }
      level.scanBindings = device.createBindGroup({
        label: `prefix-scan/level-${this.levels.length}`,
        layout: this.scanPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: levelInput } },
          { binding: 2, resource: { buffer: levelOutput } },
          { binding: 3, resource: { buffer: sums } },
        ],
      })
      this.levels.push(level)
      if (blocks <= 1) break
      levelLength = blocks
      levelInput = sums
      levelOutput = buffer(device, `prefix-scan/scanned-sums-${this.levels.length}`, blocks)
    }

    for (let i = 0; i < this.levels.length - 1; i++) {
      const level = this.levels[i]
      const scannedOffsets = this.levels[i + 1].output
      level.addBindings = device.createBindGroup({
        label: `prefix-scan/add-level-${i}`,
        layout: this.addPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: level.params } },
          { binding: 1, resource: { buffer: scannedOffsets } },
          { binding: 2, resource: { buffer: level.output } },
        ],
      })
    }
  }

  encode(encoder, telemetry) {
    let pass = encoder.beginComputePass(timedPassDescriptor(telemetry, 'tile.prefix-scan'))
    pass.setPipeline(this.scanPipeline)
    for (const level of this.levels) {
      pass.setBindGroup(0, level.scanBindings)
      pass.dispatchWorkgroups(level.blocks)
    }
    pass.end()

    if (this.levels.length > 1) {
      pass = encoder.beginComputePass(timedPassDescriptor(telemetry, 'tile.prefix-add'))
      pass.setPipeline(this.addPipeline)
      for (let i = this.levels.length - 2; i >= 0; i--) {
        const level = this.levels[i]
        pass.setBindGroup(0, level.addBindings)
        pass.dispatchWorkgroups(level.blocks)
      }
      pass.end()
    }
  }
}
