import fs from 'node:fs'
import puppeteer from 'puppeteer'
import { createServer } from 'vite'

const port = 4178
const chromeCandidates = [
  process.env.SPLATFUSE_CHROME,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
].filter(Boolean)
const executablePath = chromeCandidates.find((candidate) => fs.existsSync(candidate))
const server = await createServer({ server: { host: '127.0.0.1', port, strictPort: true } })
let browser

try {
  await server.listen()
  browser = await puppeteer.launch({
    headless: true,
    executablePath,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--disable-dawn-features=disallow_unsafe_apis'],
  })
  const page = await browser.newPage()
  await page.setViewport({ width: 640, height: 360, deviceScaleFactor: 1 })
  const errors = []
  page.on('pageerror', (error) => errors.push(error.stack || error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })

  for (const pipeline of ['quad', 'tile', 'auto']) {
    await page.goto(`http://127.0.0.1:${port}/?synthetic=1024&pipeline=${pipeline}&maxDup=4`, { waitUntil: 'networkidle0' })
    await page.waitForFunction(() => {
      const text = document.querySelector('#status')?.textContent || ''
      return text.includes('SplatFuse ·') && !text.includes('requesting WebGPU')
    }, { timeout: 20_000 })
    await new Promise((resolve) => setTimeout(resolve, 1500))
    const hud = await page.$eval('#status', (element) => element.textContent)
    if (/could not start/i.test(hud)) throw new Error(hud)
    console.log(`${pipeline}: ${hud.split('\n')[0]}`)
  }

  const validationErrors = errors.filter((message) => /validation|shader|pipeline|bindgroup|device lost/i.test(message))
  if (validationErrors.length) throw new Error(validationErrors.join('\n'))
  console.log('WebGPU smoke test passed for quad, tile, and auto modes.')
} finally {
  await browser?.close()
  await server.close()
}
