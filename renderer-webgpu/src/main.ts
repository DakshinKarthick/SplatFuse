import './style.css'
import { loadPLY } from './ply-loader';
import { WebGPURenderer } from './renderer';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div style="position: relative; width: 100vw; height: 100vh;">
    <canvas id="canvas" style="width: 100%; height: 100%; display: block;"></canvas>
    <div id="ui" style="position: absolute; top: 10px; left: 10px; color: white; font-family: monospace; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px;">
      <h2>WebGPU Tile Rasterizer</h2>
      <p>Waiting for PLY file...</p>
    </div>
  </div>
`

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
// Handle resize
const resize = () => {
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
};
window.addEventListener('resize', resize);
resize();

const renderer = new WebGPURenderer(canvas);
const ui = document.getElementById('ui')!;

async function init() {
  try {
    await renderer.init();
    ui.innerHTML += `<p>WebGPU Initialized.</p>`;
    
    // We expect the main agent to notify us with the .ply file URL.
    // The user can drop a file or we can provide a default path to test.
    // For now, we will wait for a file, or expose a load function on the window.
    (window as any).loadModel = async (url: string) => {
      ui.innerHTML += `<p>Loading ${url}...</p>`;
      const splatData = await loadPLY(url);
      renderer.setSplatData(splatData);
      ui.innerHTML += `<p>Loaded ${splatData.vertexCount} splats.</p>`;
    };

    ui.innerHTML += `<p>Use <code>window.loadModel('path/to.ply')</code> to load a splat.</p>`;

    const loop = () => {
      renderer.render();
      requestAnimationFrame(loop);
    };
    loop();

  } catch (e: any) {
    ui.innerHTML += `<p style="color: red;">Error: ${e.message}</p>`;
  }
}

init();
