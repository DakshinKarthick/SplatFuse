import { readFile, writeFile } from 'node:fs/promises'
import { CrossoverModel } from '../src/webgpu/CrossoverModel.js'

const [, , inputPath, outputPath = 'splatfuse-model.json'] = process.argv
if (!inputPath) throw new Error('usage: npm run model:fit -- benchmark.json [model.json]')
const document = JSON.parse(await readFile(inputPath, 'utf8'))
const model = new CrossoverModel().fit(document.rows)
await writeFile(outputPath, `${JSON.stringify(model.toJSON(), null, 2)}\n`)
console.log(`wrote ${outputPath}`, model.evaluate(document.rows))
