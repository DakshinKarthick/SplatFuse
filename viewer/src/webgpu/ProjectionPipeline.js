import projectionShader from './shaders/projection.wgsl?raw'
import { TILE_SIZE } from './layout.js'
import { timedPassDescriptor } from './Telemetry.js'

const CAMERA_BYTES = 224

export class ProjectionPipeline {
  constructor(device, scene) {
    this.device = device
    this.scene = scene
    this.camera = device.createBuffer({
      label: 'camera/frame-uniforms',
      size: CAMERA_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const module = device.createShaderModule({ label: 'projection/culling WGSL', code: projectionShader })
    this.resetPipeline = device.createComputePipeline({
      label: 'projection/reset',
      layout: 'auto',
      compute: { module, entryPoint: 'resetFrame' },
    })
    this.projectionPipeline = device.createComputePipeline({
      label: 'projection/project-and-cull',
      layout: 'auto',
      compute: { module, entryPoint: 'projectAndCull' },
    })
    this.resetBindGroup = this.#bind(this.resetPipeline)
    this.projectionBindGroup = this.#bind(this.projectionPipeline)
  }

  #bind(pipeline) {
    const s = this.scene
    return this.device.createBindGroup({
      label: `${pipeline.label}/bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera } },
        { binding: 1, resource: { buffer: s.splats } },
        { binding: 2, resource: { buffer: s.projected } },
        { binding: 3, resource: { buffer: s.tileCounts } },
        { binding: 4, resource: { buffer: s.activeIds } },
        { binding: 5, resource: { buffer: s.globalKeys } },
        { binding: 6, resource: { buffer: s.globalValues } },
        { binding: 7, resource: { buffer: s.stats } },
        { binding: 8, resource: { buffer: s.indirectDraw } },
      ],
    })
  }

  updateCamera({ view, projection, viewProjection, width, height }) {
    const raw = new ArrayBuffer(CAMERA_BYTES)
    const floats = new Float32Array(raw)
    floats.set(view, 0)
    floats.set(projection, 16)
    floats.set(viewProjection, 32)
    floats.set([width, height, 1 / width, 1 / height], 48)
    const integers = new Uint32Array(raw)
    integers.set([
      this.scene.count,
      this.scene.sortCapacity,
      Math.ceil(width / TILE_SIZE),
      Math.ceil(height / TILE_SIZE),
    ], 52)
    this.device.queue.writeBuffer(this.camera, 0, raw)
  }

  encode(encoder, telemetry) {
    let pass = encoder.beginComputePass(timedPassDescriptor(telemetry, 'projection.reset'))
    pass.setPipeline(this.resetPipeline)
    pass.setBindGroup(0, this.resetBindGroup)
    pass.dispatchWorkgroups(Math.ceil(Math.max(this.scene.count, this.scene.sortCapacity) / 256))
    pass.end()

    pass = encoder.beginComputePass(timedPassDescriptor(telemetry, 'projection.project-cull'))
    pass.setPipeline(this.projectionPipeline)
    pass.setBindGroup(0, this.projectionBindGroup)
    pass.dispatchWorkgroups(Math.ceil(this.scene.count / 256))
    pass.end()
  }
}
