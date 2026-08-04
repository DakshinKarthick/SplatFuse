import bitonicShader from './shaders/bitonic-sort.wgsl?raw'
import { bitonicStages } from './sortPlan.js'

/** Portable WGSL key/value sort; vec2 keys are ordered high-word then low-word. */
export class GpuBitonicSort {
  constructor(device, capacity) {
    this.device = device
    this.capacity = capacity
    this.stages = bitonicStages(capacity)
    this.parameterStride = device.limits.minUniformBufferOffsetAlignment

    const bindGroupLayout = device.createBindGroupLayout({
      label: 'bitonic-sort/bindings',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 },
        },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    const module = device.createShaderModule({ label: 'bitonic-sort WGSL', code: bitonicShader })
    this.pipeline = device.createComputePipeline({
      label: 'GPU bitonic key/value sort',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: 'bitonicStep' },
    })
    this.bindGroupLayout = bindGroupLayout
    this.parameters = device.createBuffer({
      label: 'bitonic-sort/stage-parameters',
      size: Math.max(16, this.stages.length * this.parameterStride),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const bytes = new ArrayBuffer(this.stages.length * this.parameterStride)
    const words = new Uint32Array(bytes)
    for (let i = 0; i < this.stages.length; i++) {
      const offset = (i * this.parameterStride) / 4
      const { j, k, length } = this.stages[i]
      words.set([j, k, length, 0], offset)
    }
    if (bytes.byteLength) device.queue.writeBuffer(this.parameters, 0, bytes)
  }

  bind(keys, values, label = 'sort data') {
    return this.device.createBindGroup({
      label,
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.parameters, size: 16 } },
        { binding: 1, resource: { buffer: keys } },
        { binding: 2, resource: { buffer: values } },
      ],
    })
  }

  encode(encoder, bindGroup, label = 'GPU bitonic sort') {
    if (this.capacity <= 1) return
    const pass = encoder.beginComputePass({ label })
    pass.setPipeline(this.pipeline)
    const workgroups = Math.ceil(this.capacity / 256)
    for (let i = 0; i < this.stages.length; i++) {
      pass.setBindGroup(0, bindGroup, [i * this.parameterStride])
      pass.dispatchWorkgroups(workgroups)
    }
    pass.end()
  }
}
