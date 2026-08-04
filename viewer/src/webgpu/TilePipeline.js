import rasterShader from './shaders/tile-raster.wgsl?raw'
import presentShader from './shaders/present.wgsl?raw'
import { GpuBitonicSort } from './GpuBitonicSort.js'
import { TileBinningFrontend } from './TileBinningFrontend.js'
import { TILE_SIZE } from './layout.js'
import { timedPassDescriptor } from './Telemetry.js'

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

/** CUDA-style bin, sort, range detection, and 16x16 compute raster pipeline. */
export class TilePipeline {
  constructor(device, format, scene, cameraBuffer, maximumDuplication = 8) {
    this.device = device
    this.scene = scene
    this.frontend = new TileBinningFrontend(device, scene, cameraBuffer, maximumDuplication)
    this.sorter = new GpuBitonicSort(device, this.frontend.capacity)
    this.sortBindings = this.sorter.bind(this.frontend.keys, this.frontend.values, 'tile/key-sort')
    this.params = device.createBuffer({
      label: 'tile/raster-parameters', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const module = device.createShaderModule({ label: 'compute tile rasterizer WGSL', code: rasterShader })
    this.resetPipeline = device.createComputePipeline({
      label: 'tile/reset-ranges', layout: 'auto', compute: { module, entryPoint: 'resetRanges' },
    })
    this.rangePipeline = device.createComputePipeline({
      label: 'tile/detect-ranges', layout: 'auto', compute: { module, entryPoint: 'detectRanges' },
    })
    this.rasterPipeline = device.createComputePipeline({
      label: 'tile/cooperative-raster', layout: 'auto', compute: { module, entryPoint: 'rasterizeTiles' },
    })

    const presentModule = device.createShaderModule({ label: 'tile output presentation WGSL', code: presentShader })
    this.presentPipeline = device.createRenderPipeline({
      label: 'tile/present', layout: 'auto',
      vertex: { module: presentModule, entryPoint: 'vertexMain' },
      fragment: { module: presentModule, entryPoint: 'fragmentMain', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })
  }

  resize(width, height) {
    if (width === this.width && height === this.height) return
    this.output?.destroy()
    this.ranges?.destroy()
    this.width = width
    this.height = height
    this.tilesX = Math.ceil(width / TILE_SIZE)
    this.tilesY = Math.ceil(height / TILE_SIZE)
    const tileCount = this.tilesX * this.tilesY
    this.ranges = this.device.createBuffer({ label: 'tile/ranges', size: tileCount * 8, usage: STORAGE })
    this.output = this.device.createTexture({
      label: 'tile/raster-output',
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      width, height, this.tilesX, this.tilesY, this.frontend.capacity, 0, 0, 0,
    ]))
    this.#createBindings()
  }

  #entries() {
    return [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.scene.projected } },
      { binding: 2, resource: { buffer: this.frontend.values } },
      { binding: 3, resource: { buffer: this.ranges } },
      { binding: 4, resource: this.output.createView() },
      { binding: 5, resource: { buffer: this.frontend.keys } },
    ]
  }

  #createBindings() {
    const entries = this.#entries()
    const used = (pipeline, bindings) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: entries.filter((entry) => bindings.includes(entry.binding)),
    })
    this.resetBindings = used(this.resetPipeline, [0, 3])
    this.rangeBindings = used(this.rangePipeline, [0, 3, 5])
    this.rasterBindings = used(this.rasterPipeline, [0, 1, 2, 3, 4])
    this.presentBindings = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.output.createView() }],
    })
  }

  encode(encoder, targetView, telemetry) {
    if (!this.output) throw new Error('TilePipeline.resize must be called before encode.')
    this.frontend.encode(encoder, telemetry)
    this.sorter.encode(encoder, this.sortBindings, 'tile.sort', telemetry)

    let pass = encoder.beginComputePass(timedPassDescriptor(telemetry, 'tile.ranges'))
    pass.setPipeline(this.resetPipeline)
    pass.setBindGroup(0, this.resetBindings)
    pass.dispatchWorkgroups(Math.ceil((this.tilesX * this.tilesY) / 256))
    pass.setPipeline(this.rangePipeline)
    pass.setBindGroup(0, this.rangeBindings)
    pass.dispatchWorkgroups(Math.ceil(this.frontend.capacity / 256))
    pass.end()

    pass = encoder.beginComputePass(timedPassDescriptor(telemetry, 'tile.raster'))
    pass.setPipeline(this.rasterPipeline)
    pass.setBindGroup(0, this.rasterBindings)
    pass.dispatchWorkgroups(this.tilesX, this.tilesY)
    pass.end()

    const present = encoder.beginRenderPass({
      ...timedPassDescriptor(telemetry, 'tile.present'),
      colorAttachments: [{
        view: targetView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store',
      }],
    })
    present.setPipeline(this.presentPipeline)
    present.setBindGroup(0, this.presentBindings)
    present.draw(3)
    present.end()
  }
}
