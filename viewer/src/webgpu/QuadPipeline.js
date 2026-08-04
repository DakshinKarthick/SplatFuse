import quadShader from './shaders/quad.wgsl?raw'
import { GpuBitonicSort } from './GpuBitonicSort.js'
import { timedPassDescriptor } from './Telemetry.js'

/** Globally depth-sorted instanced quads with premultiplied hardware blending. */
export class QuadPipeline {
  constructor(device, format, scene, cameraBuffer) {
    this.device = device
    this.scene = scene
    this.sorter = new GpuBitonicSort(device, scene.sortCapacity)
    this.sortBindings = this.sorter.bind(scene.globalKeys, scene.globalValues, 'quad/global-depth-sort')

    const module = device.createShaderModule({ label: 'instanced Gaussian quads WGSL', code: quadShader })
    this.pipeline = device.createRenderPipeline({
      label: 'globally sorted Gaussian quads',
      layout: 'auto',
      vertex: { module, entryPoint: 'vertexMain' },
      fragment: {
        module,
        entryPoint: 'fragmentMain',
        targets: [{
          format,
          blend: {
            color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
          writeMask: GPUColorWrite.ALL,
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    })
    this.bindGroup = device.createBindGroup({
      label: 'quad/render-bindings',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: scene.projected } },
        { binding: 2, resource: { buffer: scene.globalValues } },
      ],
    })
  }

  encodeSort(encoder, telemetry) {
    this.sorter.encode(encoder, this.sortBindings, 'quad.sort', telemetry)
  }

  encodeRender(encoder, targetView, clearValue = { r: 0, g: 0, b: 0, a: 1 }, telemetry) {
    const pass = encoder.beginRenderPass({
      ...timedPassDescriptor(telemetry, 'quad.raster'),
      colorAttachments: [{ view: targetView, clearValue, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.drawIndirect(this.scene.indirectDraw, 0)
    pass.end()
  }

  encode(encoder, targetView, clearValue, telemetry) {
    this.encodeSort(encoder, telemetry)
    this.encodeRender(encoder, targetView, clearValue, telemetry)
  }
}
