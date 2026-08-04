import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

const OUT_DIR = 'E:\\Coding proj\\MoodMate\\MoodMate\\public\\static\\projects\\gaussiansplat';
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function run() {
  console.log('Starting Vite server...');
  const viteProcess = spawn('npm', ['run', 'dev'], {
    cwd: 'E:\\Coding proj\\MoodMate\\GaussianSplat\\viewer',
    shell: true,
  });

  viteProcess.stdout.on('data', (data) => console.log(`vite: ${data}`));
  viteProcess.stderr.on('data', (data) => console.error(`vite: ${data}`));

  await delay(5000); // Wait for vite to start

  console.log('Launching Puppeteer...');
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });

  const scenes = [
    { ply: 'iceland_splat.ply', name: 'gaussiansplat-iceland.png' },
    { ply: 'mypic1.ply', name: 'gaussiansplat-mypic1.png' },
    { ply: 'mypic2.ply', name: 'gaussiansplat-mypic2.png' }
  ];

  for (const scene of scenes) {
    console.log(`Loading ${scene.ply}...`);
    await page.goto(`http://localhost:5173/?scene=/scenes/${scene.ply}`);
    // Wait for the point cloud to load and render
    await delay(6000); 
    const outPath = path.join(OUT_DIR, scene.name);
    await page.screenshot({ path: outPath });
    console.log(`Saved screenshot to ${outPath}`);
  }

  await browser.close();
  viteProcess.kill();
  console.log('Done!');
}

run().catch(console.error);
