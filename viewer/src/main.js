import { GpuContext } from './webgpu/GpuContext.js'

const canvas = document.querySelector('canvas.webgl')
const status = document.querySelector('#status')

function resize() {
  const scale = Math.min(devicePixelRatio, 2)
  canvas.width = Math.max(1, Math.floor(innerWidth * scale))
  canvas.height = Math.max(1, Math.floor(innerHeight * scale))
  canvas.style.width = `${innerWidth}px`
  canvas.style.height = `${innerHeight}px`
}

async function start() {
  resize()
  const gpu = await GpuContext.create(canvas)
  status.textContent = [
    'SplatFuse · native WebGPU',
    `adapter: ${gpu.adapterInfo.description || gpu.adapterInfo.device || 'WebGPU adapter'}`,
    `timestamps: ${gpu.supportsTimestamps ? 'enabled' : 'unavailable'}`,
    'stage 1/10 · scaffold ready',
  ].join('\n')

  addEventListener('resize', () => {
    resize()
    gpu.configure()
  })

  function frame() {
    gpu.clear({ r: 0.012, g: 0.02, b: 0.04, a: 1 })
    requestAnimationFrame(frame)
  }
  frame()
}

start().catch((error) => {
  console.error(error)
  status.style.color = '#fca5a5'
  status.textContent = `SplatFuse could not start\n${error.message}`
})
