import { AdaptiveSelector } from './AdaptiveSelector.js'
import { FrameStatsReader } from './FrameStatsReader.js'
import { ProjectionPipeline } from './ProjectionPipeline.js'
import { QuadPipeline } from './QuadPipeline.js'
import { SceneBuffers } from './SceneBuffers.js'
import { Telemetry } from './Telemetry.js'
import { TilePipeline } from './TilePipeline.js'

export class SplatFuseRenderer {
  constructor(gpu, splats, { mode = 'auto', maximumDuplication = 8, depthComplexity = 16, onMeasurement } = {}) {
    this.gpu = gpu
    this.device = gpu.device
    this.scene = SceneBuffers.fromSplats(this.device, splats)
    this.projection = new ProjectionPipeline(this.device, this.scene)
    this.quad = new QuadPipeline(this.device, gpu.format, this.scene, this.projection.camera)
    this.tile = new TilePipeline(this.device, gpu.format, this.scene, this.projection.camera, maximumDuplication)
    this.telemetry = new Telemetry(this.device, gpu.supportsTimestamps)
    this.selector = new AdaptiveSelector({ mode })
    this.statsReader = new FrameStatsReader(this.device, this.scene.stats)
    this.depthComplexity = depthComplexity
    this.onMeasurement = onMeasurement
    this.latestStats = {
      visibleCount: splats.count,
      meanRadius: 8,
      duplicationFactor: 4,
      duplicateCount: splats.count * 4,
      overflowCount: 0,
    }
    this.frameIndex = 0
  }

  render(camera) {
    const width = this.gpu.canvas.width
    const height = this.gpu.canvas.height
    this.tile.resize(width, height)
    const workload = {
      gaussianCount: this.latestStats.visibleCount,
      meanRadius: this.latestStats.meanRadius,
      duplicationFactor: this.latestStats.duplicationFactor,
      depthComplexity: this.depthComplexity,
      overflowCount: this.latestStats.overflowCount,
      width,
      height,
    }
    const pipeline = this.selector.choose(workload)
    this.telemetry.beginFrame({ frame: this.frameIndex++, pipeline, ...workload })

    this.projection.updateCamera({
      view: camera.matrixWorldInverse.elements,
      projection: camera.projectionMatrix.elements,
      viewProjection: camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse).elements,
      width,
      height,
    })
    const encoder = this.device.createCommandEncoder({ label: `SplatFuse frame (${pipeline})` })
    this.projection.encode(encoder, this.telemetry)
    const target = this.gpu.context.getCurrentTexture().createView()
    if (pipeline === 'quad') this.quad.encode(encoder, target, undefined, this.telemetry)
    else this.tile.encode(encoder, target, this.telemetry)

    const statsToken = this.statsReader.capture(encoder)
    const timingToken = this.telemetry.finishFrame(encoder)
    this.device.queue.submit([encoder.finish()])

    statsToken.read().then((stats) => { this.latestStats = stats }).catch(console.error)
    timingToken.read().then((measurement) => {
      this.selector.record(pipeline, measurement, workload)
      this.onMeasurement?.({ pipeline, workload, measurement, selector: this.selector.snapshot(workload) })
    }).catch(console.error)
    return { pipeline, workload, selector: this.selector.snapshot(workload) }
  }
}
