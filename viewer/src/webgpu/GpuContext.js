/** Owns the WebGPU adapter, device, canvas configuration, and loss handling. */
export class GpuContext {
  static async create(canvas) {
    if (!navigator.gpu) {
      throw new Error('WebGPU is unavailable. Use a current WebGPU-capable browser.')
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter was found.')

    const supportsTimestamps = adapter.features.has('timestamp-query')
    const requiredFeatures = supportsTimestamps ? ['timestamp-query'] : []
    const requiredLimits = {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    }
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits })
    const adapterInfo = adapter.info || {}

    const instance = new GpuContext(canvas, adapter, device, adapterInfo, supportsTimestamps)
    instance.configure()
    device.lost.then((info) => {
      console.error(`WebGPU device lost (${info.reason}): ${info.message}`)
    })
    return instance
  }

  constructor(canvas, adapter, device, adapterInfo, supportsTimestamps) {
    this.canvas = canvas
    this.adapter = adapter
    this.device = device
    this.adapterInfo = adapterInfo
    this.supportsTimestamps = supportsTimestamps
    this.context = canvas.getContext('webgpu')
    this.format = navigator.gpu.getPreferredCanvasFormat()
  }

  configure() {
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    })
  }

  clear(color) {
    const encoder = this.device.createCommandEncoder({ label: 'clear frame' })
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: color,
        loadOp: 'clear',
        storeOp: 'store',
      }],
    })
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }
}
