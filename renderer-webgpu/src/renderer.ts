import type { SplatData } from './ply-loader';
import { Camera } from './camera';
import projectWgsl from './shaders/project.wgsl?raw';
import rasterizeWgsl from './shaders/rasterize.wgsl?raw';

export class WebGPURenderer {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private canvasFormat!: GPUTextureFormat;

  private splatData!: SplatData;
  private camera!: Camera;
  private canvas: HTMLCanvasElement;

  // Pipelines
  private projectPipeline!: GPUComputePipeline;
  // @ts-ignore
  private rasterizePipeline!: GPUComputePipeline;

  // Buffers
  private positionsBuffer!: GPUBuffer;
  private scalesBuffer!: GPUBuffer;
  private rotationsBuffer!: GPUBuffer;
  private colorsBuffer!: GPUBuffer;
  private opacitiesBuffer!: GPUBuffer;
  
  private splats2DBuffer!: GPUBuffer;
  private uniformsBuffer!: GPUBuffer;
  
  // @ts-ignore
  private tileOffsetsBuffer!: GPUBuffer;
  // @ts-ignore
  private tileSplatIndicesBuffer!: GPUBuffer;

  // Bind Groups
  private projectUniformBindGroup!: GPUBindGroup;
  private projectInputBindGroup!: GPUBindGroup;
  private projectOutputBindGroup!: GPUBindGroup;
  
  // @ts-ignore
  private rasterizeUniformBindGroup!: GPUBindGroup;
  // @ts-ignore
  private rasterizeInputBindGroup!: GPUBindGroup;
  // @ts-ignore
  private rasterizeOutputBindGroup!: GPUBindGroup;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  public async init() {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No appropriate GPUAdapter found');
    
    this.device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeInvocationsPerWorkgroup: 256,
        }
    });

    this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
    });

    this.camera = new Camera(this.canvas);
    await this.createPipelines();
  }

  private async createPipelines() {
    const projectModule = this.device.createShaderModule({ code: projectWgsl });
    this.projectPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: projectModule, entryPoint: 'main' }
    });

    const rasterizeModule = this.device.createShaderModule({ code: rasterizeWgsl });
    this.rasterizePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: rasterizeModule, entryPoint: 'main' }
    });
  }

  public setSplatData(data: SplatData) {
    this.splatData = data;
    const numSplats = data.vertexCount;

    const createBuffer = (dataArray: Float32Array) => {
      const buffer = this.device.createBuffer({
        size: dataArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      this.device.queue.writeBuffer(buffer, 0, dataArray);
      return buffer;
    };

    this.positionsBuffer = createBuffer(data.positions);
    this.scalesBuffer = createBuffer(data.scales);
    this.rotationsBuffer = createBuffer(data.rotations);
    this.colorsBuffer = createBuffer(data.colors);
    this.opacitiesBuffer = createBuffer(data.opacities);

    this.uniformsBuffer = this.device.createBuffer({
      size: 64 * 2 + 16 * 6, // mat4x4 * 2 + vec3 + params
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Each splat2D needs xy(2), conic(3), color(4), depth(1), bounds(4) = 14 floats -> 56 bytes per splat. Align to 64 bytes for layout.
    const splat2DSize = 16 * 4; // 64 bytes
    this.splats2DBuffer = this.device.createBuffer({
      size: numSplats * splat2DSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.projectUniformBindGroup = this.device.createBindGroup({
      layout: this.projectPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformsBuffer } }]
    });

    this.projectInputBindGroup = this.device.createBindGroup({
      layout: this.projectPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuffer } },
        { binding: 1, resource: { buffer: this.scalesBuffer } },
        { binding: 2, resource: { buffer: this.rotationsBuffer } },
        { binding: 3, resource: { buffer: this.colorsBuffer } },
        { binding: 4, resource: { buffer: this.opacitiesBuffer } },
      ]
    });

    this.projectOutputBindGroup = this.device.createBindGroup({
      layout: this.projectPipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: this.splats2DBuffer } },
        // Add tile counts if needed for sorting
      ]
    });
  }

  private updateUniforms() {
    const viewMatrix = this.camera.getViewMatrix();
    const projMatrix = this.camera.getProjectionMatrix();
    
    const w = this.canvas.width;
    const h = this.canvas.height;
    
    const fovY = this.camera.fov;
    const focal_y = h / (2 * Math.tan(fovY / 2));
    const focal_x = w / (2 * Math.tan(fovY / 2));

    const uniformsData = new ArrayBuffer(176);
    const floatView = new Float32Array(uniformsData);
    const uintView = new Uint32Array(uniformsData);

    floatView.set(viewMatrix, 0);
    floatView.set(projMatrix, 16);
    
    floatView[32] = this.camera.position[0];
    floatView[33] = this.camera.position[1];
    floatView[34] = this.camera.position[2];

    uintView[36] = w;
    uintView[37] = h;
    
    floatView[38] = focal_x;
    floatView[39] = focal_y;
    floatView[40] = Math.tan(fovY / 2);
    floatView[41] = Math.tan(fovY / 2);
    floatView[42] = 1.0; // scale_modifier

    this.device.queue.writeBuffer(this.uniformsBuffer, 0, uniformsData);
  }

  public render() {
    if (!this.splatData) return;

    this.updateUniforms();

    const commandEncoder = this.device.createCommandEncoder();

    // 1. Projection pass
    const projectPass = commandEncoder.beginComputePass();
    projectPass.setPipeline(this.projectPipeline);
    projectPass.setBindGroup(0, this.projectUniformBindGroup);
    projectPass.setBindGroup(1, this.projectInputBindGroup);
    projectPass.setBindGroup(2, this.projectOutputBindGroup);
    
    const workgroupCount = Math.ceil(this.splatData.vertexCount / 256);
    projectPass.dispatchWorkgroups(workgroupCount);
    projectPass.end();

    // TODO: 2. Tile binning and sorting
    // This requires reading splats2DBuffer, sorting by depth and tile, and populating tileOffsetsBuffer and tileSplatIndicesBuffer.
    // For now, this is skipped in the scaffolding until the sorting logic is implemented.

    // 3. Rasterization pass (Placeholder)
    // We would create a textureView from the canvas and bind it as a storage texture for the rasterizer
    // This requires setting up the rasterizeOutputBindGroup dynamically since the texture view changes per frame.
    
    /*
    const textureView = this.context.getCurrentTexture().createView();
    // Recreate bind group with new texture view...
    const rasterizePass = commandEncoder.beginComputePass();
    rasterizePass.setPipeline(this.rasterizePipeline);
    rasterizePass.setBindGroup(0, this.rasterizeUniformBindGroup);
    rasterizePass.setBindGroup(1, this.rasterizeInputBindGroup);
    rasterizePass.setBindGroup(2, this.rasterizeOutputBindGroup);
    
    const tilesX = Math.ceil(this.canvas.width / 16);
    const tilesY = Math.ceil(this.canvas.height / 16);
    rasterizePass.dispatchWorkgroups(tilesX, tilesY);
    rasterizePass.end();
    */

    this.device.queue.submit([commandEncoder.finish()]);
  }
}
