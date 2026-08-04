# Gaussian Splat Viewer: `viewer/src` Complete Walkthrough

> A code-first explanation of how the browser viewer loads a binary PLY file, turns every Gaussian into GPU attributes, sorts transparent splats in a background worker, and renders soft billboards with GLSL shaders.

This guide matches the source as it exists on **2026-08-04**. Line numbers refer to:

- `viewer/src/main.js` (264 lines)
- `viewer/src/SplatLoader.js` (189 lines)
- `viewer/src/SplatMaterial.js` (54 lines)
- `viewer/src/sortWorker.js` (122 lines)
- `viewer/src/shaders/splat.vert.glsl` (72 lines)
- `viewer/src/shaders/splat.frag.glsl` (32 lines)

Blank lines and long comments are grouped with the code they describe. Every executable statement is covered in order.

---

## 1. The entire viewer in one picture

```text
Browser opens index.html
        |
        v
main.js creates scene, camera, controls, renderer and HUD
        |
        v
loadSplats(scene URL) ------------------------------+
        |                                           |
        v                                           v
fetch PLY bytes                              parse PLY header
        |                                           |
        +-------------------+-----------------------+
                            v
              decode flat typed arrays
              positions, scales, rotations,
              colors and opacities
                            |
                            v
main.js calculates radii and robust camera framing
                            |
                            v
creates one shared quad + one instance per splat
                            |
                            v
SplatMaterial connects attributes to the GLSL shaders
                            |
                            +-------------------------+
                            |                         |
                            v                         v
                 vertex shader runs          sortWorker receives
                 four times per splat         unsorted arrays
                            |                         |
                            v                         v
                 creates screen-facing       camera movement causes
                 quad in clip space           far-to-near counting sort
                            |                         |
                            v                         v
                 fragment shader creates     sorted arrays return to
                 soft Gaussian alpha          main.js and upload to GPU
                            |                         |
                            +------------+------------+
                                         v
                               renderer draws a frame
```

There are three different processors involved:

| Processor | Code | Responsibility |
|---|---|---|
| Browser main thread | `main.js`, `SplatLoader.js`, `SplatMaterial.js` | UI, loading, scene construction and GPU uploads |
| Browser worker thread | `sortWorker.js` | CPU depth sorting without freezing camera controls |
| GPU | `splat.vert.glsl`, `splat.frag.glsl` | Positioning quads and coloring millions of pixels |

---

## 2. Data carried for each splat

The PLY contains a list of Gaussians, not triangle-mesh vertices. The loader creates these arrays:

| Array | Values per splat | Example layout | Used by Phase 1? |
|---|---:|---|---|
| `positions` | 3 | `[x0,y0,z0, x1,y1,z1, ...]` | Yes |
| `scales` | 3 | `[sx0,sy0,sz0, sx1,sy1,sz1, ...]` | Yes, averaged into one radius |
| `rotations` | 4 | `[w0,x0,y0,z0, ...]` | Parsed, but not rendered yet |
| `colors` | 3 | `[r0,g0,b0, r1,g1,b1, ...]` | Yes |
| `opacities` | 1 | `[a0, a1, ...]` | Yes |

The arrays are **flat** because WebGL consumes contiguous numeric memory. A Python list of objects such as `[Splat(...), Splat(...)]` would be convenient for humans but inefficient for GPU upload.

For splat index `i`:

```js
const i3 = i * 3
const x = positions[i3]
const y = positions[i3 + 1]
const z = positions[i3 + 2]
const opacity = opacities[i]
```

Python equivalent:

```python
i3 = i * 3
x = positions[i3]
y = positions[i3 + 1]
z = positions[i3 + 2]
opacity = opacities[i]
```

---

## 3. JavaScript syntax used in this project

### `const` and `let`

```js
const count = 100   // the variable cannot be assigned a different value
let sorting = false // the variable can be reassigned
```

`const` does not make an object immutable. It only prevents rebinding the variable:

```js
const values = []
values.push(10) // valid: the same array is mutated
// values = []  // invalid: this would bind values to a new array
```

Python does not enforce this distinction. Uppercase names are only a convention for constants.

### Objects and property access

```js
const config = { count: 10, name: 'scene' }
config.count
config['count']
```

This is closest to a Python dictionary, although JavaScript objects also participate in its prototype system:

```python
config = {'count': 10, 'name': 'scene'}
config['count']
```

### Object destructuring

```js
const { count, positions } = splats
```

This takes the `count` and `positions` properties from `splats` and creates local variables with the same names. Rough Python equivalent:

```python
count = splats['count']
positions = splats['positions']
```

Renaming while destructuring:

```js
const { positions: p, scales: s } = e.data
```

means:

```python
p = e.data['positions']
s = e.data['scales']
```

### Arrow functions

```js
const square = (x) => x * x
```

Python equivalent:

```python
square = lambda x: x * x
```

With braces, an arrow function has a statement body:

```js
const square = (x) => {
  const result = x * x
  return result
}
```

Arrow functions are also heavily used as event callbacks:

```js
window.addEventListener('resize', () => {
  // called later, whenever the event occurs
})
```

### Template literals

```js
`loaded ${count} splats`
```

Backticks create a template string. `${expression}` inserts a value. Python equivalent:

```python
f'loaded {count} splats'
```

### Ternary expressions

```js
const color = failed ? 'red' : 'green'
```

Python equivalent:

```python
color = 'red' if failed else 'green'
```

The loader uses a nested ternary:

```js
return v < 0 ? 0 : v > 1 ? 1 : v
```

Read it as: return `0` if below zero; otherwise return `1` if above one; otherwise return `v`.

### Logical fallback

```js
const scene = params.get('scene') || '/scenes/mypic1.ply'
```

`||` returns the first truthy operand. An absent query parameter returns `null`, so the default path is selected.

```python
scene = params.get('scene') or '/scenes/mypic1.ply'
```

### Strict equality

```js
params.get('flip') === '1'
```

`===` compares value and type without automatic conversion. Prefer it over JavaScript's coercing `==`.

### Modules: `import` and `export`

```js
import { loadSplats } from './SplatLoader.js'
export function loadSplats(url) { /* ... */ }
```

`export` makes a value available to other modules. `import` retrieves the specifically named export.

```js
import * as THREE from 'three'
```

This imports all named Three.js exports into the namespace object `THREE`, leading to names such as `THREE.Scene`.

### `new`

```js
const scene = new THREE.Scene()
```

`new` constructs a class instance. Python calls the class directly:

```python
scene = Scene()
```

### Async functions and `await`

```js
async function init() {
  const splats = await loadSplats(SCENE_URL)
}
```

An `async` function always returns a `Promise`. `await` pauses only that async function until the promise settles; it does not block the browser's entire event loop.

Python has nearly identical syntax:

```python
async def init():
    splats = await load_splats(scene_url)
```

### Promises and `.catch()`

```js
init().catch((err) => {
  console.error(err)
})
```

This starts `init()` and registers an error callback. It plays the role of a surrounding asynchronous `try/except`.

### Typed arrays

```js
const positions = new Float32Array(count * 3)
const indices = new Uint16Array([0, 1, 2])
```

Unlike normal JavaScript arrays, typed arrays have a fixed numeric type and contiguous binary storage:

- `Float32Array`: 32-bit IEEE floating-point numbers, four bytes each.
- `Uint16Array`: unsigned 16-bit integers, two bytes each.
- `Uint32Array`: unsigned 32-bit integers, four bytes each.
- `Uint8Array`: raw unsigned bytes.

They are conceptually similar to NumPy arrays with `dtype=np.float32`, `np.uint16`, and so on.

### Short-circuit return

```js
if (!splatMaterial) return
```

`!value` means logical NOT. `null` is falsy, so the function returns early before the material exists.

### Increment operators

```js
counts[b]++
++sortCount
```

Both add one. The prefix form `++sortCount` evaluates to the new value, which matters when it is nested inside `String(...)`.

### Bitwise floor shortcut

```js
const b = value | 0
```

Bitwise operations convert the value to a signed 32-bit integer, truncating its fractional part. For the non-negative bucket value here, it acts like `Math.floor(value)`. It should not be used for arbitrary large or negative values without understanding the conversion.

---

## 4. `main.js`: browser entry point

`main.js` owns the application lifecycle. It creates browser and Three.js objects, asks the loader for data, constructs the instanced mesh, starts the sort worker, and continuously renders.

### Lines 1-15: purpose and imports

```js
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { loadSplats } from './SplatLoader.js'
import { createSplatMaterial } from './SplatMaterial.js'
```

- **Lines 1-10:** The JSDoc comment gives the five-step boot sequence. `/** ... */` is a documentation comment; it does not execute.
- **Line 12:** Makes the Three.js API available through the `THREE` namespace.
- **Line 13:** Imports the mouse/touch orbit controller from Three.js examples.
- **Line 14:** Imports this project's PLY loading function.
- **Line 15:** Imports the factory that connects the GLSL shaders to Three.js.

Imports run before the rest of this module, so all dependencies are ready when line 17 begins.

### Lines 17-24: scene URL and orientation

```js
const params = new URLSearchParams(location.search)
const SCENE_URL = params.get('scene') || '/scenes/mypic1.ply'
const FLIP = params.get('flip') === '1'
```

- **Line 19:** `location.search` is the query-string portion of the current browser URL, for example `?scene=/scenes/iceland_splat.ply&flip=1`. `URLSearchParams` parses it.
- **Line 20:** Reads the `scene` parameter. When it is missing, `null || default` selects `mypic1.ply`.
- **Line 24:** Produces an actual Boolean. Only the exact string `'1'` enables flipping.

Examples:

```text
http://localhost:5173/
http://localhost:5173/?scene=/scenes/iceland_splat.ply
http://localhost:5173/?scene=/scenes/mypic2.ply&flip=1
```

### Lines 26-44: debug HUD

```js
const hud = document.createElement('div')
hud.id = 'hud'
hud.style.cssText =
  'position:fixed;top:8px;left:8px;font:12px/1.4 monospace;color:#0f0;' +
  'background:rgba(0,0,0,.65);padding:6px 9px;white-space:pre;z-index:10;' +
  'pointer-events:none;max-width:90vw'
document.body.appendChild(hud)
const hudLines = {}
```

- **Line 29:** Asks the browser DOM to create a new `<div>` in memory.
- **Line 30:** Gives it the ID `hud`.
- **Lines 31-34:** Assigns inline CSS. Adjacent strings are joined with `+`; this is split across source lines only for readability.
- **Line 35:** Inserts the new element into the page's `<body>`, making it visible.
- **Line 36:** Creates an object that stores HUD values by key.

```js
function setHud(key, value) {
  hudLines[key] = value
  hud.textContent = Object.entries(hudLines).map(([k, v]) => `${k}: ${v}`).join('\n')
}
```

- **Line 37:** Declares a normal named function with parameters `key` and `value`.
- **Line 38:** Uses computed property access. If `key` is `'gpu'`, this is equivalent to `hudLines.gpu = value`.
- **Line 39:** Performs a pipeline:
  1. `Object.entries(hudLines)` turns the object into `[[key,value], ...]` pairs.
  2. `.map(([k,v]) => ...)` destructures each pair and formats it.
  3. `.join('\n')` joins all lines with newline characters.
  4. Assigning `textContent` safely replaces the visible plain text.

```js
function hudError(msg) {
  hud.style.color = '#f55'
  setHud('ERROR', msg)
}
```

- **Line 41:** Declares the error helper.
- **Line 42:** Changes the HUD from green to red.
- **Line 43:** Reuses `setHud` rather than duplicating formatting logic.

### Lines 46-63: Three.js foundation

```js
const canvas = document.querySelector('canvas.webgl')
const scene = new THREE.Scene()
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  1000,
)
scene.add(camera)
```

- **Line 47:** Finds the first `<canvas class="webgl">` already present in the HTML.
- **Line 48:** Creates the root Three.js scene graph container.
- **Line 52:** Creates a perspective camera with `fov=60 degrees`, current aspect ratio, near plane `0.01`, and far plane `1000`.
- **Line 53:** Adds the camera to the scene graph. This is not required for basic rendering, but makes its transforms part of the graph.

```js
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
```

- **Line 56:** Attaches pointer and wheel interaction on `canvas` to `camera`.
- **Line 57:** Enables inertial smoothing. Because damping is enabled, `controls.update()` must run every animation frame.

```js
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
```

- **Line 59:** Creates a WebGL renderer. `{ canvas, antialias: true }` uses object-property shorthand: `canvas` means `canvas: canvas`.
- **Line 60:** Sets the CSS/output size to the browser window.
- **Line 63:** Uses the smaller of the device pixel ratio and `2`. This avoids excessive pixel rendering on high-density displays.

### Lines 65-73: keep the shader viewport synchronized

```js
let splatMaterial = null
function updateViewportUniform() {
  if (!splatMaterial) return
  const size = new THREE.Vector2()
  renderer.getDrawingBufferSize(size)
  splatMaterial.uniforms.uViewport.value.copy(size)
}
```

- **Line 67:** The material does not exist during initial renderer setup, so the variable starts as `null`. It is `let` because line 188 assigns the material later.
- **Line 68:** Declares a reusable synchronization function.
- **Line 69:** Exits safely if called before material creation.
- **Line 70:** Allocates a two-component vector that Three.js can fill.
- **Line 71:** Retrieves the real drawing-buffer size in physical device pixels, not only CSS pixels.
- **Line 72:** Copies that vector into `uViewport`, a value read by the vertex shader.

The method uses `.copy(size)` instead of replacing `.value` so the existing Three.js uniform object remains intact.

### Lines 75-83: GPU name and initial status

```js
try {
  const gl = renderer.getContext()
  const dbg = gl.getExtension('WEBGL_debug_renderer_info')
  setHud('gpu', dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '(hidden)')
} catch (e) {
  setHud('gpu', 'query failed')
}
setHud('status', `loading ${SCENE_URL} ...`)
```

- **Line 76:** Starts a protected block because some browsers hide or reject debug GPU information.
- **Line 77:** Retrieves the underlying `WebGLRenderingContext`.
- **Line 78:** Requests an optional WebGL extension.
- **Line 79:** Uses a ternary: read the renderer name when the extension exists; otherwise show `'(hidden)'`.
- **Lines 80-82:** If any statement throws, record a nonfatal HUD message and continue.
- **Line 83:** Displays the selected URL using a template literal.

### Lines 85-93: resize event

```js
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  updateViewportUniform()
})
```

- **Line 87:** Registers an anonymous arrow function. The browser stores it and calls it after future resize events.
- **Line 88:** Updates the aspect ratio so the scene is not stretched.
- **Line 89:** Recalculates the camera projection matrix after changing camera parameters. Changing `aspect` alone is insufficient.
- **Line 90:** Resizes the renderer/canvas.
- **Line 91:** Reapplies the capped device pixel ratio.
- **Line 92:** Updates the shader's pixel-size calculation.

### Lines 95-122: robust scene framing

```js
function robustFraming(positions, count) {
  const sampleN = Math.min(50000, count)
  const step = Math.max(1, Math.floor(count / sampleN))
  const xs = [], ys = [], zs = []
```

- **Lines 95-102:** Explain why ordinary min/max bounds fail when training produces distant floater Gaussians.
- **Line 103:** Accepts the flat position array and number of splats.
- **Line 105:** Chooses at most 50,000 samples.
- **Line 106:** Computes how far to jump between samples. `Math.floor` produces an integer; `Math.max(1, ...)` prevents a zero step.
- **Line 107:** Creates three normal JavaScript arrays. Multiple declarations separated with commas share one `const` statement.

```js
  for (let i = 0; i < count; i += step) {
    xs.push(positions[i * 3])
    ys.push(positions[i * 3 + 1])
    zs.push(positions[i * 3 + 2])
  }
```

- **Line 108:** Starts at splat zero, continues while `i < count`, and advances by the calculated sample step.
- **Lines 109-111:** Extract the flat x/y/z entries and append them to separate arrays.

```js
  const pct = (arr, p) => {
    const a = Float32Array.from(arr).sort()
    return a[Math.min(a.length - 1, Math.floor((p / 100) * a.length))]
  }
```

- **Line 113:** Creates a local percentile helper. It closes over nothing except its own arguments.
- **Line 114:** Copies the normal array into a typed array and numerically sorts ascending. A normal JavaScript array's default sort is lexicographic, while a typed array's default is numeric.
- **Line 115:** Converts a percentage into an array index and caps it at the last valid element.

```js
  const lo = [pct(xs, 5), pct(ys, 5), pct(zs, 5)]
  const hi = [pct(xs, 95), pct(ys, 95), pct(zs, 95)]
  const center = new THREE.Vector3(
    (lo[0] + hi[0]) / 2,
    (lo[1] + hi[1]) / 2,
    (lo[2] + hi[2]) / 2,
  )
  const radius = Math.max(
    hi[0] - lo[0],
    hi[1] - lo[1],
    hi[2] - lo[2],
  ) / 2 || 1
  return { center, radius }
}
```

- **Lines 117-118:** Calculate the 5th and 95th percentile along every axis, excluding sparse extremes.
- **Line 119:** Uses the midpoint between percentile bounds as the center.
- **Line 120:** Uses half the longest robust dimension as radius. `|| 1` prevents a zero-sized scene.
- **Line 121:** Returns an object using property shorthand, equivalent to `{ center: center, radius: radius }`.

### Lines 124-138: initialize and calculate isotropic radius

```js
async function init() {
  const splats = await loadSplats(SCENE_URL)
  const { count, positions, scales, colors, opacities } = splats
  console.log(`loaded ${count.toLocaleString()} splats from ${SCENE_URL}`)
  setHud('splats', count.toLocaleString())
```

- **Line 126:** Declares the asynchronous initialization function.
- **Line 127:** Calls the loader and pauses this function until fetching and parsing finish.
- **Line 128:** Destructures the returned object. `rotations` is deliberately omitted because Phase 1 does not use it.
- **Line 129:** Logs a human-readable count. `toLocaleString()` adds locale-sensitive digit separators.
- **Line 130:** Shows the same count in the HUD.

```js
  const radii = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    radii[i] = (scales[i * 3] + scales[i * 3 + 1] + scales[i * 3 + 2]) / 3
  }
```

- **Line 135:** Allocates exactly one float per splat.
- **Line 136:** Visits every splat once.
- **Line 137:** Reads the three activated scale axes from the flat array and stores their arithmetic mean. This makes the Phase 1 splat circular rather than elliptical.

### Lines 140-145: the world group and optional flip

```js
const world = new THREE.Group()
if (FLIP) world.rotation.x = Math.PI
world.updateMatrixWorld(true)
scene.add(world)
```

- **Line 142:** Creates a parent transform for all splat-related objects.
- **Line 143:** When requested, rotates the parent 180 degrees around X. `Math.PI` radians equals 180 degrees.
- **Line 144:** Immediately computes the group's world matrix. The `true` argument forces descendants too.
- **Line 145:** Attaches the group to the scene.

Changing one parent matrix is cheaper and clearer than rewriting millions of positions.

### Lines 147-162: point the camera at real content

```js
const { center, radius } = robustFraming(positions, count)
const centerWorld = center.clone().applyMatrix4(world.matrixWorld)
controls.target.copy(centerWorld)
camera.position.copy(centerWorld).add(new THREE.Vector3(0, 0, radius * 2.0))
camera.near = Math.max(radius * 0.002, 0.001)
camera.far = radius * 100
camera.updateProjectionMatrix()
controls.update()
```

- **Line 150:** Destructures the framing result.
- **Line 151:** `clone()` prevents mutation of `center`; `applyMatrix4` converts the copy from local splat coordinates into the optionally flipped world coordinates.
- **Line 152:** Makes OrbitControls orbit around the content center.
- **Line 153:** Starts at the center and moves the camera two radii in positive Z.
- **Line 154:** Chooses a near plane proportional to scene size, but never below `0.001`.
- **Line 155:** Chooses a far plane large enough to contain outliers.
- **Line 156:** Rebuilds projection after changing near/far.
- **Line 157:** Immediately applies the new target and position to controls.
- **Line 158:** Formats center coordinates to two decimal places and reports whether flipping was used.
- **Line 162:** Adds red/green/blue X/Y/Z axes with a size proportional to the scene.

### Lines 164-171: one shared quad

```js
const quadPositions = new Float32Array([
  -1, -1, 0,
   1, -1, 0,
  -1,  1, 0,
   1,  1, 0,
])
const quadIndex = new Uint16Array([0, 1, 2, 2, 1, 3])
```

- **Line 170:** Defines four corners. Each corner contains x, y and z, so 12 numbers represent four vertices.
- **Line 171:** Defines two counter-clockwise triangles: vertices `(0,1,2)` and `(2,1,3)`.

The viewer does not create a separate quad array for every Gaussian. It stores this quad once and asks WebGL instancing to reuse it `count` times.

### Lines 173-185: instanced geometry and GPU attributes

```js
const geometry = new THREE.InstancedBufferGeometry()
geometry.instanceCount = count
geometry.setAttribute('position', new THREE.BufferAttribute(quadPositions, 3))
geometry.setIndex(new THREE.BufferAttribute(quadIndex, 1))
geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(positions, 3))
geometry.setAttribute('aScale', new THREE.InstancedBufferAttribute(radii, 1))
geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(colors, 3))
geometry.setAttribute('aOpacity', new THREE.InstancedBufferAttribute(opacities, 1))
```

- **Line 178:** Creates geometry capable of per-instance attributes.
- **Line 179:** Tells Three.js to draw one instance for every splat.
- **Line 180:** Registers the shared `position` attribute with three floats per vertex. Three.js automatically exposes this standard attribute to the vertex shader.
- **Line 181:** Registers one integer per triangle index.
- **Line 182:** Registers three floats per instance under the exact shader name `aCenter`.
- **Line 183:** Registers one float per instance as `aScale`.
- **Line 184:** Registers three floats per instance as `aColor`.
- **Line 185:** Registers one float per instance as `aOpacity`.

The names must match the GLSL declarations exactly:

```glsl
attribute vec3 aCenter;
attribute float aScale;
attribute vec3 aColor;
attribute float aOpacity;
```

### Lines 187-195: material plus mesh

```js
const material = createSplatMaterial()
splatMaterial = material
updateViewportUniform()
const mesh = new THREE.Mesh(geometry, material)
mesh.frustumCulled = false
world.add(mesh)
setHud('status', 'rendered - drag to orbit')
```

- **Line 187:** Calls the material factory, which compiles/configures the two shaders.
- **Line 188:** Stores it in the outer variable used by resizing.
- **Line 189:** Now that the material exists, fills its viewport uniform.
- **Line 190:** Combines geometry (data) and material (rendering behavior) into a drawable mesh.
- **Line 193:** Disables automatic frustum culling. Three.js sees only the tiny shared quad when calculating bounds, not the displaced instances, so normal culling could hide the whole scene incorrectly.
- **Line 194:** Makes the mesh a child of the optionally flipped group.
- **Line 195:** Updates the user-visible status.

### Lines 197-209: create and initialize the sort worker

```js
const sortWorker = new Worker(
  new URL('./sortWorker.js', import.meta.url),
  { type: 'module' },
)
sortWorker.onerror = (e) => hudError(`worker: ${e.message}`)
```

- **Line 201:** `new URL(relative, base)` gives Vite a worker dependency it can bundle. `import.meta.url` is this module's own URL. `{ type: 'module' }` enables module semantics in the worker.
- **Line 202:** Assigns an error callback. Worker failures become visible in the HUD.

```js
{
  const p = positions.slice()
  const s = radii.slice()
  const c = colors.slice()
  const o = opacities.slice()
  sortWorker.postMessage(
    { type: 'init', positions: p, scales: s, colors: c, opacities: o },
    [p.buffer, s.buffer, c.buffer, o.buffer],
  )
}
```

- **Line 203:** A bare block creates a temporary lexical scope for the short variable names.
- **Line 204:** `.slice()` makes typed-array copies. The geometry must retain its arrays while the worker also needs an original unsorted set.
- **Lines 205-208:** Sends a structured message plus a **transfer list**.
  - First argument: the data object the worker receives as `e.data`.
  - Second argument: transfers ownership of underlying `ArrayBuffer`s instead of cloning megabytes.
  - After posting, `p`, `s`, `c`, and `o` on the main thread are detached and unusable, but geometry's original arrays remain intact.

### Lines 211-226: sort backpressure and camera matrix

```js
let sorting = false
let dirty = false

function requestSort() {
  if (sorting) {
    dirty = true
    return
  }
  sorting = true
  dirty = false
```

- **Lines 214-215:** Track whether a sort is running and whether another is needed afterward.
- **Line 216:** Declares the callback used both initially and for control changes.
- **Line 217:** If a worker job is already active, remember that the view changed and return. This prevents a queue containing hundreds of stale camera matrices.
- **Lines 218-219:** Mark the new job active and clear the old dirty flag.

```js
  camera.updateMatrixWorld(true)
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert()
  const modelView = new THREE.Matrix4().multiplyMatrices(
    camera.matrixWorldInverse,
    mesh.matrixWorld,
  )
  sortWorker.postMessage({
    type: 'sort',
    viewMatrix: Array.from(modelView.elements),
  })
}
```

- **Line 222:** Forces current camera transforms after controls movement.
- **Line 223:** Copies and inverts the camera world transform to produce the view matrix.
- **Line 224:** Multiplies `view * model`. This includes the mesh/world flip and maps splat-local coordinates directly into camera-view coordinates.
- **Line 225:** Converts matrix typed storage into a structured-clone-friendly normal array and asks the worker to sort.

### Lines 228-244: receive sorted arrays and upload them

```js
let sortCount = 0
sortWorker.onmessage = (e) => {
  const { positions: p, scales: s, colors: c, opacities: o } = e.data
```

- **Line 230:** Initializes a debug counter.
- **Line 231:** Registers the callback invoked whenever the worker posts a result.
- **Line 232:** Destructures worker result properties and renames them to short local variables.

```js
geometry.attributes.aCenter.array.set(p)
geometry.attributes.aCenter.needsUpdate = true
geometry.attributes.aScale.array.set(s)
geometry.attributes.aScale.needsUpdate = true
geometry.attributes.aColor.array.set(c)
geometry.attributes.aColor.needsUpdate = true
geometry.attributes.aOpacity.array.set(o)
geometry.attributes.aOpacity.needsUpdate = true
```

- **Odd lines 233-239:** Typed-array `.set(source)` copies each returned sorted array into the attribute's existing CPU-side array.
- **Even lines 234-240:** `needsUpdate = true` tells Three.js to re-upload that array to its WebGL buffer before drawing.

```js
sorting = false
setHud('sorts', String(++sortCount))
if (dirty) requestSort()
```

- **Line 241:** The worker is free for another request.
- **Line 242:** Prefix increment updates the count first; `String` makes the HUD value explicit text.
- **Line 243:** If the camera moved while sorting, immediately use its newest matrix.

### Lines 246-264: initial sort, error handling and animation loop

```js
controls.addEventListener('change', requestSort)
requestSort()
```

- **Line 247:** Camera changes now request a depth sort.
- **Line 248:** Performs an initial sort even before the user moves.
- **Line 249:** Ends `init`.

```js
init().catch((err) => {
  console.error('failed to load splats:', err)
  hudError(err.message || String(err))
})
```

- **Line 252:** Calls `init` immediately and attaches a rejection handler.
- **Line 253:** Preserves a detailed developer error in browser DevTools.
- **Line 254:** Displays the normal error message, falling back to string conversion if `.message` is absent.

```js
function tick() {
  controls.update()
  renderer.render(scene, camera)
  requestAnimationFrame(tick)
}
tick()
```

- **Line 259:** Declares one frame of the animation loop.
- **Line 260:** Applies OrbitControls damping.
- **Line 261:** Draws the entire scene from the current camera.
- **Line 262:** Asks the browser to call `tick` again before the next repaint. Passing the function without parentheses schedules it; `tick()` would call it immediately.
- **Line 264:** Starts the first frame.

Notice that `tick()` begins immediately, even while `init()` is still fetching. Early frames simply render an empty scene and keep the browser responsive.

---

## 5. `SplatLoader.js`: binary PLY decoding

This module converts an HTTP response into typed arrays. It understands the file's property layout from its header instead of assuming one exporter-specific ordering.

### Lines 1-32: file model and SH constant

```js
const SH_C0 = 0.28209479177387814
```

- **Lines 1-24:** Document the PLY structure and per-Gaussian fields.
- **Lines 26-31:** Explain that Phase 1 uses only the zero-order, view-independent spherical harmonic coefficients.
- **Line 32:** Stores the constant used by `rgb = 0.5 + SH_C0 * dc`.

### Lines 34-43: property sizes and clamping

```js
const TYPE_SIZES = {
  float: 4,
  float32: 4,
  double: 8,
  uchar: 1,
  uint8: 1,
  int: 4,
  int32: 4,
  short: 2,
  ushort: 2,
}
```

- **Lines 35-38:** Map PLY type names to byte widths. Several aliases intentionally share sizes.

```js
function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v
}
```

- **Line 41:** Declares a small numeric helper.
- **Line 42:** Clamps below-range values to zero and above-range values to one.

Python equivalent:

```python
def clamp01(value):
    return max(0.0, min(1.0, value))
```

### Lines 45-59: locate the end of the ASCII header

```js
function findHeaderEnd(bytes) {
  const marker = new TextEncoder().encode('end_header\n')
```

- **Line 49:** Accepts a `Uint8Array` representing the entire file.
- **Line 50:** Converts the known ASCII marker into its byte representation.

```js
  for (let i = 0; i <= bytes.length - marker.length; i++) {
    let match = true
    for (let j = 0; j < marker.length; j++) {
      if (bytes[i + j] !== marker[j]) {
        match = false
        break
      }
    }
    if (match) return i + marker.length
  }
```

- **Line 51:** Tries every possible marker start. The final valid start is `bytes.length - marker.length`.
- **Line 52:** Initially assumes the marker matches at `i`.
- **Line 53:** Compares every marker byte.
- **Line 54:** On the first mismatch, marks failure and breaks only the inner loop.
- **Line 56:** If all bytes matched, returns the first binary-data index, immediately after the marker.

```js
throw new Error('.ply end_header not found - not a valid ply file')
```

- **Line 58:** If the whole file was searched without success, throws. This rejects `loadSplats` and eventually reaches `init().catch(...)` in `main.js`.

### Lines 61-76: decode header lines

```js
function parseHeader(bytes) {
  const headerEnd = findHeaderEnd(bytes)
  const text = new TextDecoder('ascii').decode(bytes.subarray(0, headerEnd))
  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
```

- **Line 73:** Starts the header parser.
- **Line 74:** Finds the exact boundary once.
- **Line 75:** Takes a no-copy byte view of only the header and decodes it as ASCII.
- **Line 76:** Splits on newline, trims whitespace from every line, and discards empty lines. `filter(Boolean)` keeps only truthy, nonempty strings.

### Lines 78-83: validate file type and endianness

```js
if (!lines[0].startsWith('ply')) throw new Error('not a .ply file')
const formatLine = lines.find((line) => line.startsWith('format'))
if (!formatLine || !formatLine.includes('binary_little_endian')) {
  throw new Error(`unsupported ply format: ${formatLine}`)
}
```

- **Line 78:** Requires the first meaningful header line to start with `ply`.
- **Line 80:** `.find(...)` returns the first format line or `undefined`.
- **Line 81:** Rejects a missing format line or anything except little-endian binary.
- **Line 82:** Includes the discovered value in the error for debugging.

### Lines 85-99: collect vertex property declarations

```js
let count = 0
const props = []
for (const line of lines) {
```

- **Line 85:** Zero means the vertex element has not yet been found.
- **Line 86:** Holds ordered objects shaped like `{ type, name }`.
- **Line 87:** `for...of` iterates the line values directly, similar to Python `for line in lines`.

```js
  if (line.startsWith('element vertex')) {
    count = parseInt(line.split(/\s+/).pop(), 10)
```

- **Line 88:** Detects a declaration such as `element vertex 452598`.
- **Line 90:** Splits on one-or-more whitespace characters (`/\s+/` is a regular expression), takes the last token with `.pop()`, and parses base ten.

```js
  } else if (line.startsWith('property') && count > 0 && props.length < 10000) {
    const parts = line.split(/\s+/)
    props.push({ type: parts[1], name: parts[2] })
```

- **Line 91:** Processes properties only after finding the vertex count. The 10,000 guard avoids unbounded malformed input.
- **Line 93:** Splits a line such as `property float scale_0`.
- **Line 94:** Adds `{ type: 'float', name: 'scale_0' }`, preserving file order.

```js
  } else if (line.startsWith('element') && !line.includes('vertex')) {
    break
  }
}
if (count === 0) throw new Error('.ply has no vertex element')
```

- **Lines 95-96:** Stops when a later element, such as faces, begins. Its properties are not vertex fields.
- **Line 99:** Rejects headers without a nonzero vertex count.

### Lines 101-116: calculate offsets and stride

```js
const offsets = {}
let cursor = 0
for (const p of props) {
  if (!(p.type in TYPE_SIZES)) {
    throw new Error(`unsupported ply property type: ${p.type}`)
  }
  offsets[p.name] = cursor
  cursor += TYPE_SIZES[p.type]
}
return { headerEnd, count, stride: cursor, offsets }
```

- **Line 103:** `offsets` will map names such as `x` to byte positions inside one vertex record.
- **Line 104:** `cursor` starts at byte zero of a record.
- **Line 105:** Visits properties in original file order.
- **Line 106:** The `in` operator checks whether the object contains that type name.
- **Line 107:** Fails clearly rather than calculating an invalid layout.
- **Line 109:** Records where this property begins.
- **Line 110:** Advances by the property's byte width.
- **Line 113:** Returns `cursor` under the renamed property `stride`. The final cursor is the byte width of one complete vertex.

Example:

```text
property float x        offset 0,  cursor becomes 4
property float y        offset 4,  cursor becomes 8
property float z        offset 8,  cursor becomes 12
property uchar opacity  offset 12, cursor becomes 13

stride = 13 bytes per vertex
```

### Lines 118-128: fetch the entire file

```js
export async function loadSplats(url) {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`failed to fetch ${url}: ${res.status} ${res.statusText}`)
  }
  const buffer = await res.arrayBuffer()
  const bytes = new Uint8Array(buffer)
  const { headerEnd, count, stride, offsets } = parseHeader(bytes)
```

- **Line 122:** Exports an async function so `main.js` can import it.
- **Line 123:** Starts an HTTP request. Vite serves paths under `viewer/public` at the web root.
- **Line 124:** `fetch` resolves even for HTTP 404/500 responses, so `.ok` must be checked explicitly.
- **Line 125:** Throws an error with URL, status code and status text.
- **Line 126:** Reads the whole response body as raw binary memory.
- **Line 127:** Creates a byte-level view over the same `ArrayBuffer`; this does not copy the file.
- **Line 128:** Parses and destructures its layout metadata.

### Lines 130-140: require fields and resolve offsets once

```js
const need = (name) => {
  if (!(name in offsets)) {
    throw new Error(`.ply is missing expected property "${name}"`)
  }
  return offsets[name]
}
```

- **Line 132:** Local helper captures `offsets` through a closure.
- **Line 133:** Rejects a file missing a field this renderer requires.
- **Line 134:** Returns the property's per-record byte offset.

```js
const oX = need('x'), oY = need('y'), oZ = need('z')
const oOpacity = need('opacity')
const oS0 = need('scale_0'), oS1 = need('scale_1'), oS2 = need('scale_2')
const oR0 = need('rot_0'), oR1 = need('rot_1'), oR2 = need('rot_2'), oR3 = need('rot_3')
const oDc0 = need('f_dc_0'), oDc1 = need('f_dc_1'), oDc2 = need('f_dc_2')
```

- **Lines 136-140:** Resolve all offsets before entering the large vertex loop. This avoids repeated object lookups and validation millions of times.
- The `o` prefix means **offset**, not opacity.

### Lines 142-151: create the binary reader and output arrays

```js
const view = new DataView(buffer, headerEnd)
```

- **Line 144:** Creates a flexible reader whose byte zero is the beginning of binary vertex data. `DataView` can read unaligned values and explicitly select little-endian order.

```js
const positions = new Float32Array(count * 3)
const scales = new Float32Array(count * 3)
const rotations = new Float32Array(count * 4)
const colors = new Float32Array(count * 3)
const opacities = new Float32Array(count)
```

- **Lines 147-151:** Preallocate exact sizes. Typed arrays cannot grow, and preallocation prevents millions of dynamic `push` operations.

### Lines 153-159: read positions

```js
for (let i = 0; i < count; i++) {
  const base = i * stride
  positions[i * 3] = view.getFloat32(base + oX, true)
  positions[i * 3 + 1] = view.getFloat32(base + oY, true)
  positions[i * 3 + 2] = view.getFloat32(base + oZ, true)
```

- **Line 153:** Visits each PLY vertex record.
- **Line 154:** Calculates the start of record `i` relative to the binary section.
- **Lines 157-159:** Read x/y/z from `record start + property offset`. The second argument `true` means little-endian.

Address formula:

```text
file field address = headerEnd + (i * stride) + propertyOffset
```

`DataView` already begins at `headerEnd`, so code supplies only the last two parts.

### Lines 161-169: activate logarithmic scales

```js
scales[i * 3] = Math.exp(view.getFloat32(base + oS0, true))
scales[i * 3 + 1] = Math.exp(view.getFloat32(base + oS1, true))
scales[i * 3 + 2] = Math.exp(view.getFloat32(base + oS2, true))
```

- **Lines 161-166:** Explain why training stores unconstrained logarithmic scale.
- **Lines 167-169:** Read each raw scale and apply `e^raw`, guaranteeing a positive physical size.

Python equivalent:

```python
real_scale = math.exp(raw_scale)
```

### Lines 171-177: copy quaternions

```js
rotations[i * 4] = view.getFloat32(base + oR0, true)
rotations[i * 4 + 1] = view.getFloat32(base + oR1, true)
rotations[i * 4 + 2] = view.getFloat32(base + oR2, true)
rotations[i * 4 + 3] = view.getFloat32(base + oR3, true)
```

- **Lines 174-177:** Copy four quaternion components in file order. They are retained for a later anisotropic renderer but are not passed to the current GPU geometry.

### Lines 179-182: decode base color

```js
colors[i * 3] = clamp01(0.5 + SH_C0 * view.getFloat32(base + oDc0, true))
colors[i * 3 + 1] = clamp01(0.5 + SH_C0 * view.getFloat32(base + oDc1, true))
colors[i * 3 + 2] = clamp01(0.5 + SH_C0 * view.getFloat32(base + oDc2, true))
```

- **Lines 180-182:** Read the three zero-order spherical-harmonic coefficients, transform each into RGB, and clamp it to the valid `[0,1]` range.

### Lines 184-189: activate opacity and return

```js
opacities[i] = 1 / (1 + Math.exp(-view.getFloat32(base + oOpacity, true)))
}

return { count, positions, scales, rotations, colors, opacities }
```

- **Line 185:** Reads an unconstrained logit and applies the sigmoid function, mapping every finite input into `(0,1)`.
- **Line 186:** Ends the per-record loop.
- **Line 188:** Returns one object containing metadata and every output array. Property shorthand is used for all six fields.
- **Line 189:** Ends the loader.

---

## 6. `SplatMaterial.js`: shader and blending configuration

The material is the bridge between Three.js and GLSL. It supplies shader source, uniform values, transparency state and blend equations.

### Lines 1-29: imports

```js
import * as THREE from 'three'
import vertexShader from './shaders/splat.vert.glsl?raw'
import fragmentShader from './shaders/splat.frag.glsl?raw'
```

- **Lines 1-23:** Explain premultiplied alpha and why depth writes are disabled.
- **Line 25:** Imports Three.js constants/classes.
- **Line 28:** Vite's `?raw` suffix imports the vertex shader file contents as a JavaScript string rather than as a URL.
- **Line 29:** Does the same for the fragment shader.

### Lines 31-41: create material and uniforms

```js
export function createSplatMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uViewport: { value: new THREE.Vector2(1, 1) },
      uMinPixels: { value: 1.5 },
    },
```

- **Line 31:** Exports a factory function rather than one global material instance.
- **Line 32:** Constructs and immediately returns `ShaderMaterial`.
- **Lines 33-34:** Object shorthand passes the imported shader strings under the keys Three.js expects.
- **Line 38:** Begins the uniform dictionary. Uniform names must match GLSL names.
- **Line 39:** Initializes the viewport safely; `main.js` replaces its contents with the real dimensions.
- **Line 40:** Sets the minimum rendered splat radius to 1.5 physical pixels.

Three.js uniforms use the wrapper shape `{ value: ... }` so it can track and upload changes.

### Lines 43-53: transparency and premultiplied blending

```js
transparent: true,
depthWrite: false,
depthTest: true,
side: THREE.DoubleSide,
blending: THREE.CustomBlending,
blendEquation: THREE.AddEquation,
blendSrc: THREE.OneFactor,
blendDst: THREE.OneMinusSrcAlphaFactor,
```

- **Line 43:** Places the material in Three.js's transparent rendering path and enables blending.
- **Line 44:** Stops splats from writing depth. Otherwise a nearer transparent splat could completely block farther contributions.
- **Line 45:** Still tests against depth written by opaque objects.
- **Line 46:** Disables face-direction culling for the billboard triangles.
- **Line 49:** Selects explicit custom blend factors.
- **Line 50:** Adds source and destination contributions.
- **Line 51:** Multiplies the premultiplied source color by one.
- **Line 52:** Keeps the destination in proportion to `1 - source alpha`.
- **Line 53:** Finishes the configuration object and constructor call.
- **Line 54:** Ends the factory.

Resulting color equation:

```text
output = sourcePremultipliedColor + destinationColor * (1 - sourceAlpha)
```

This requires far-to-near ordering, which is the worker's job.

---

## 7. `sortWorker.js`: background depth counting sort

A Web Worker has a separate global scope and event loop. It cannot access the DOM or Three.js objects from `main.js`; the two threads communicate only through messages.

### Lines 1-40: algorithm constants and persistent worker state

```js
const BUCKETS = 65536

let positions = null
let scales = null
let colors = null
let opacities = null
let count = 0
```

- **Lines 1-30:** Document why transparency requires sorting, why it runs off-thread, and why O(n) counting sort is used.
- **Line 32:** Uses 65,536 quantized depth bins (`2^16`).
- **Lines 36-40:** Declare state that survives between messages. It is filled once by the `init` message and reused for every camera sort.

These are unsorted originals. Each new result is derived from them, avoiding accumulated ordering errors.

### Lines 42-58: message dispatch

```js
self.onmessage = (e) => {
  const msg = e.data
```

- **Line 43:** `self` is the worker's global object, analogous to `window` on the main thread.
- **Line 44:** Extracts the structured message.

```js
if (msg.type === 'init') {
  positions = msg.positions
  scales = msg.scales
  colors = msg.colors
  opacities = msg.opacities
  count = positions.length / 3
  return
}
```

- **Line 48:** Dispatches the one-time initialization message by its string tag.
- **Lines 49-52:** Store transferred typed arrays in persistent variables.
- **Line 53:** Derives splat count because every position contains three floats.
- **Line 54:** Returns from the handler so initialization does not fall through into sorting.

```js
if (msg.type !== 'sort') return
const m = msg.viewMatrix
```

- **Line 57:** Ignores unknown future message types safely.
- **Line 58:** Stores the 16-element column-major model-view matrix under a short name.

### Lines 60-72: pass 1, calculate view-space depths

```js
const depth = new Float32Array(count)
let minD = Infinity
let maxD = -Infinity
```

- **Line 63:** Allocates one depth per splat for this sort.
- **Lines 64-65:** Initialize extrema so the first finite depth replaces both.

```js
for (let i = 0; i < count; i++) {
  const i3 = i * 3
  const d =
    m[2] * positions[i3] +
    m[6] * positions[i3 + 1] +
    m[10] * positions[i3 + 2] +
    m[14]
  depth[i] = d
  if (d < minD) minD = d
  if (d > maxD) maxD = d
}
```

- **Line 66:** Visits every splat.
- **Line 67:** Finds its first position component.
- **Line 68:** Computes only the z row of `modelView * vec4(x,y,z,1)`. The homogeneous coordinate `1` produces the translation term `m[14]`.
- **Line 69:** Saves depth for the bucket pass.
- **Lines 70-71:** Expand the observed depth range.

The camera looks toward negative view-space Z. More negative means farther away.

### Lines 74-85: pass 2, quantize and count buckets

```js
const range = maxD - minD || 1
const quantScale = (BUCKETS - 1) / range
const bucketOf = new Uint32Array(count)
const counts = new Uint32Array(BUCKETS)
```

- **Line 77:** Calculates depth range, using one if all depths are identical.
- **Line 78:** Produces a multiplier that maps `[minD,maxD]` into `[0,65535]`.
- **Line 79:** Allocates the chosen bucket number for every splat.
- **Line 80:** Allocates zero-initialized counts for every bucket.

```js
for (let i = 0; i < count; i++) {
  const b = ((depth[i] - minD) * quantScale) | 0
  bucketOf[i] = b
  counts[b]++
}
```

- **Line 81:** Visits all depth values.
- **Line 82:** Shifts minimum depth to zero, scales to bucket range, and truncates to integer using `| 0`.
- **Line 83:** Remembers which bucket this splat belongs to.
- **Line 84:** Adds one to that bucket's population.

### Lines 87-94: prefix sum into output starts

```js
const offsets = new Uint32Array(BUCKETS)
let acc = 0
for (let b = 0; b < BUCKETS; b++) {
  offsets[b] = acc
  acc += counts[b]
}
```

- **Line 89:** Allocates each bucket's output cursor.
- **Line 90:** `acc` tracks how many splats exist in all earlier buckets.
- **Line 91:** Walks buckets from far/negative to near/positive.
- **Line 92:** The current accumulated count is this bucket's first output index.
- **Line 93:** Advance past this bucket for the next iteration.

Small example:

```text
counts  = [2, 0, 3, 1]
offsets = [0, 2, 2, 5]

bucket 0 writes output positions 0..1
bucket 2 writes output positions 2..4
bucket 3 writes output position 5
```

### Lines 96-113: pass 3, scatter complete splat records

```js
const sortedPositions = new Float32Array(count * 3)
const sortedScales = new Float32Array(count)
const sortedColors = new Float32Array(count * 3)
const sortedOpacities = new Float32Array(count)
```

- **Lines 97-100:** Allocate result arrays in the same shapes expected by GPU attributes.

```js
for (let i = 0; i < count; i++) {
  const dst = offsets[bucketOf[i]]++
  const s3 = i * 3
  const d3 = dst * 3
```

- **Line 101:** Visits every original splat.
- **Line 102:** Reads this bucket's next free destination and then increments that bucket cursor. Postfix `++` evaluates to the old value before adding one.
- **Line 103:** Computes the source vec3 index.
- **Line 104:** Computes the destination vec3 index.

```js
sortedPositions[d3] = positions[s3]
sortedPositions[d3 + 1] = positions[s3 + 1]
sortedPositions[d3 + 2] = positions[s3 + 2]
sortedScales[dst] = scales[i]
sortedColors[d3] = colors[s3]
sortedColors[d3 + 1] = colors[s3 + 1]
sortedColors[d3 + 2] = colors[s3 + 2]
sortedOpacities[dst] = opacities[i]
```

- **Lines 105-112:** Copy every attribute of the same splat to the same sorted destination. Keeping attributes together is essential; sorting positions without colors would attach the wrong color to each position.

### Lines 115-122: transfer results to the main thread

```js
self.postMessage(
  {
    positions: sortedPositions,
    scales: sortedScales,
    colors: sortedColors,
    opacities: sortedOpacities,
  },
  [
    sortedPositions.buffer,
    sortedScales.buffer,
    sortedColors.buffer,
    sortedOpacities.buffer,
  ],
)
```

- **Line 118:** Sends a result through the worker's global `postMessage`.
- **Line 119:** Defines the object received by `sortWorker.onmessage` in `main.js`.
- **Line 120:** Transfers all four buffers without structured-clone copies.
- **Line 121:** Completes the call.
- **Line 122:** Ends the message handler.

After transfer, the four `sorted...` arrays are detached in the worker. This is safe because they are local to this sort; the persistent original arrays were not transferred back.

---

## 8. `splat.vert.glsl`: vertex shader

GLSL is C-like GPU shader code, not JavaScript. A vertex shader runs independently for each submitted vertex. Here there are four quad vertices per splat.

### GLSL syntax needed here

```glsl
attribute vec3 aCenter;
uniform vec2 uViewport;
varying vec3 vColor;
```

- `float`: one floating-point value.
- `vec2`, `vec3`, `vec4`: vectors containing 2, 3 or 4 floats.
- `mat4`: 4x4 matrix.
- `attribute`: input that can vary by vertex or instance.
- `uniform`: one value shared by the whole draw call.
- `varying`: vertex output interpolated across a triangle and received by the fragment shader.
- `.xy`: swizzle selecting x and y components.
- `vec4(aCenter, 1.0)`: construct a four-component vector from a `vec3` plus one float.

Three.js automatically declares standard inputs such as `position`, `modelViewMatrix` and `projectionMatrix` for `ShaderMaterial`.

### Lines 1-37: inputs, uniforms and outputs

```glsl
attribute vec3 aCenter;
attribute float aScale;
attribute vec3 aColor;
attribute float aOpacity;

uniform vec2 uViewport;
uniform float uMinPixels;

varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;
```

- **Lines 1-21:** Explain shader frequency and coordinate spaces.
- **Line 24:** Receives one center for the current instance.
- **Line 25:** Receives its isotropic world-space radius.
- **Line 26:** Receives its RGB color.
- **Line 27:** Receives its opacity.
- **Line 30:** Receives drawing-buffer width and height from `main.js`.
- **Line 31:** Receives the 1.5-pixel radius floor.
- **Lines 35-37:** Declare values that the rasterizer interpolates for fragments.

### Lines 39-45: forward per-instance appearance

```glsl
void main() {
  vQuad = position.xy;
  vColor = aColor;
  vOpacity = aOpacity;
```

- **Line 39:** Every GLSL shader begins execution in `main`; it returns no value (`void`).
- **Line 43:** Takes x/y from the current shared quad corner. This becomes the fragment shader's local coordinate.
- **Line 44:** Sends instance color toward the fragment shader.
- **Line 45:** Sends opacity too.

Although values are declared as varying, all four vertices of one instance receive identical color/opacity, so interpolation leaves those values unchanged across the quad.

### Lines 47-52: transform center into camera space

```glsl
vec4 cam = modelViewMatrix * vec4(aCenter, 1.0);
```

- **Line 52:** Converts a 3D point into homogeneous form with `w=1`, then applies model and camera transforms. The result is the Gaussian center in view space.

A position uses `w=1` so matrix translation applies. A direction would usually use `w=0`.

### Lines 54-64: enforce a minimum screen radius

```glsl
float w = max(-cam.z, 1.0e-6);
float focalY = projectionMatrix[1][1];
float pxRadius = aScale * focalY / w * (uViewport.y * 0.5);
float k = max(pxRadius, uMinPixels) / max(pxRadius, 1.0e-6);
```

- **Line 59:** In front of the camera, view-space z is negative, so `-cam.z` is positive depth. The epsilon avoids division by zero.
- **Line 60:** Reads the projection matrix's vertical focal-length factor.
- **Line 62:** Projects world radius into approximate physical pixels using perspective scaling.
- **Line 64:** Computes a dilation factor. If the splat is already large enough, numerator and denominator match, so `k=1`. If too small, `k>1` enlarges it.

Example:

```text
pxRadius = 0.5, uMinPixels = 1.5 -> k = 3.0
pxRadius = 4.0, uMinPixels = 1.5 -> k = 1.0
```

### Lines 66-72: expand the quad and project it

```glsl
cam.xy += vQuad * aScale * k;
gl_Position = projectionMatrix * cam;
}
```

- **Line 68:** Offsets the view-space center along camera-right and camera-up according to the current quad corner. Operating in view-space x/y automatically makes the quad face the camera.
- **Line 71:** Converts the final view-space corner into clip space. `gl_Position` is the required vertex shader output.
- **Line 72:** Ends the shader.

The GPU later divides clip x/y/z by clip w and rasterizes the two triangles.

---

## 9. `splat.frag.glsl`: fragment shader

A fragment shader runs for covered pixel samples after triangles are rasterized. It converts the hard square into a soft circular Gaussian approximation.

### Lines 1-14: interpolated inputs

```glsl
varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;
```

- **Lines 1-8:** Describe the shader's purpose.
- **Lines 12-14:** Must match the vertex shader's varying names and types exactly.

For `vQuad`, interpolation transforms corner values into a continuous coordinate field:

```text
top-left (-1,+1) -------- top-right (+1,+1)
          |       (0,0) center       |
bottom-left (-1,-1) ---- bottom-right (+1,-1)
```

### Lines 16-21: Gaussian-shaped alpha

```glsl
void main() {
  float a = vOpacity * exp(-4.0 * dot(vQuad, vQuad));
```

- **Line 16:** Starts per-fragment execution.
- **Line 21:** Calculates opacity in three steps:
  1. `dot(vQuad, vQuad)` equals `x*x + y*y`, the squared distance from center.
  2. `exp(-4*rSquared)` is 1 at the center and decays rapidly outward.
  3. Multiplying by instance opacity controls the entire splat's strength.

Squared distance avoids an expensive square root.

### Lines 23-25: discard negligible fragments

```glsl
if (a < 0.003) discard;
```

- **Line 25:** Ends this fragment without writing color/depth if its contribution is negligible. This removes square corners and reduces unnecessary blending.

### Lines 27-32: premultiplied output

```glsl
gl_FragColor = vec4(vColor * a, a);
}
```

- **Line 31:** Creates RGBA output. RGB is multiplied by alpha before output, so this is **premultiplied alpha**.
- **Line 32:** Ends the shader.

This output matches the material configuration:

```text
shader source RGB = vColor * a
blend source factor = ONE
blend destination factor = ONE_MINUS_SRC_ALPHA
```

---

## 10. Startup flow, in exact chronological order

1. The browser evaluates module imports.
2. `main.js` parses URL parameters.
3. It inserts the HUD into the DOM.
4. It creates scene, camera, controls and renderer.
5. It registers resize handling.
6. It calls `init()`; loading starts asynchronously.
7. It calls `tick()` immediately; empty frames can render while loading.
8. `fetch` downloads the entire PLY.
9. `parseHeader` discovers count, stride and property offsets.
10. `loadSplats` loops through binary records and fills typed arrays.
11. `init` receives the arrays and averages scales into radii.
12. It creates the optional orientation group.
13. It samples positions and frames the camera robustly.
14. It creates one four-vertex shared quad.
15. It attaches position/radius/color/opacity as per-instance attributes.
16. It creates `ShaderMaterial` and its uniforms/blending state.
17. It creates the mesh and adds it to the scene.
18. It copies original arrays for the worker and transfers the copies.
19. It registers sort callbacks.
20. It requests the initial far-to-near sort.
21. The worker calculates depths, buckets, prefix offsets and sorted arrays.
22. The worker transfers sorted arrays back.
23. `main.js` copies them into geometry attributes and marks GPU buffers dirty.
24. On the next render, Three.js uploads the new buffers.
25. The vertex shader expands each instance into a camera-facing quad.
26. The fragment shader fades each quad into a soft splat.
27. WebGL blends splats far-to-near.

---

## 11. What happens when the user drags the camera

```text
pointer input
    |
    v
OrbitControls changes camera
    |
    +--> change event --> requestSort()
    |                       |
    |                       +--> if busy: dirty = true, stop
    |                       |
    |                       +--> otherwise send newest model-view matrix
    |                                            |
    |                                            v
    |                                     worker counting sort
    |                                            |
    |                                            v
    |                                  transfer sorted arrays back
    |                                            |
    |                                            v
    |                                   GPU attributes updated
    |
    +--> tick() applies damping and renders continuously
```

Sorting and rendering are asynchronous. The viewer may render a few frames with the previous ordering while a sort is running, but the UI stays responsive. `dirty` guarantees one final sort using a newer camera after rapid movement.

---

## 12. Coordinate-space flow

For each splat, coordinates pass through these spaces:

```text
PLY/local position aCenter
        |
        | modelViewMatrix = cameraInverse * meshWorld
        v
view-space center cam
        |
        | add quad corner along view X/Y
        v
view-space billboard corner
        |
        | projectionMatrix
        v
clip-space gl_Position
        |
        | automatic perspective divide by w
        v
screen pixel
```

The worker uses the same `cameraInverse * meshWorld` matrix and reads only the resulting view-space z. This is why flipping the `world` group affects rendering and sorting consistently.

---

## 13. Why the worker sorts entire attribute arrays

It may seem cheaper to return only sorted indices:

```text
[2, 0, 1, ...]
```

However, WebGL instanced draw order follows the order of instance attribute entries. This viewer has no indirection attribute plus texture/storage lookup. Therefore it physically changes:

```text
position[old index] -> position[new index]
scale[old index]    -> scale[new index]
color[old index]    -> color[new index]
opacity[old index]  -> opacity[new index]
```

All attributes must move together to preserve each Gaussian's identity.

---

## 14. Memory lifecycle

The largest memory moments are important for large scenes:

1. `res.arrayBuffer()` holds the full PLY.
2. The loader allocates decoded positions, scales, rotations, colors and opacity.
3. `main.js` allocates radii.
4. It copies four arrays for worker initialization.
5. Transfer moves those copies to the worker without another copy.
6. Every sort allocates depth, bucket numbers, counts, offsets and four sorted output arrays.
7. Transfer moves sorted outputs to the main thread.
8. The main thread copies results into persistent geometry arrays.
9. Three.js uploads them into GPU buffers.

This explains why moving one-time PLY conversion into an offline Python tool would help: a compact runtime file could omit unused higher-order SH fields, normals and currently unused rotations, and could store already activated radii/colors/opacities.

It would **not** replace the worker or shader code, because those depend on the live browser camera and GPU.

---

## 15. Debugging map

| Symptom | First place to inspect | Likely reason |
|---|---|---|
| HUD says fetch failed | `SplatLoader.js` lines 123-126 | Wrong URL or asset not under `public` |
| PLY format error | `SplatLoader.js` lines 73-113 | Unsupported format/type/header |
| Missing property error | `SplatLoader.js` lines 132-140 | Exporter uses a different schema |
| Scene is upside down | `main.js` lines 19-24 and 142-145 | `?flip=1` missing or incorrectly present |
| Camera frames empty space | `main.js` lines 103-121 | Outliers or unsuitable percentile range |
| Whole mesh disappears | `main.js` line 193 | Frustum culling accidentally enabled |
| Hard square splats | fragment shader lines 21-25 | Gaussian alpha or discard is wrong |
| Incorrect transparency | material lines 43-52 or worker ordering | Blend factors and sort direction disagree |
| Stutter while dragging | worker and GPU upload path | Sort/all-buffer upload too expensive |
| Worker error in HUD | `main.js` lines 201-202 | Bundling error or exception in worker |
| Tiny splats vanish | vertex shader lines 59-68 | Viewport uniform or minimum-size math wrong |
| Colors attached to wrong points | worker lines 101-112 | Attribute arrays were not scattered together |

---

## 16. Recommended reading order

Read the files in data-flow order rather than alphabetical order:

1. `main.js` lines 1-130: understand app boot and the expected loader result.
2. `SplatLoader.js`: understand where typed arrays come from.
3. `main.js` lines 132-195: understand geometry and attributes.
4. `splat.vert.glsl`: see how attributes become screen-space quads.
5. `splat.frag.glsl`: see how quads become soft dots.
6. `SplatMaterial.js`: connect shader output to WebGL blending.
7. `main.js` lines 197-248 plus `sortWorker.js`: understand dynamic ordering.
8. `main.js` lines 251-264: finish with error handling and the render loop.

The shortest useful mental sentence is:

> Load flat arrays, reuse one quad for every splat, sort instances by camera depth, let the vertex shader place each quad, and let the fragment shader soften and blend it.

---

## 17. Glossary

| Term | Meaning in this viewer |
|---|---|
| Gaussian/splat | One soft colored 3D blob represented by center, size, color and opacity |
| Billboard | A flat quad oriented toward the camera |
| Instance | One reuse of shared geometry with different per-instance attributes |
| Attribute | GPU input that changes per vertex or per instance |
| Uniform | GPU input shared by an entire draw call |
| Varying | Vertex output interpolated for fragment shader inputs |
| Vertex shader | GPU program positioning every submitted vertex |
| Fragment shader | GPU program calculating output for covered pixels |
| View space | Coordinates relative to the camera at the origin |
| Clip space | Projection output before automatic perspective division |
| Stride | Bytes occupied by one complete PLY vertex record |
| Offset | Byte position of a property within one record |
| Typed array | Fixed-type contiguous JavaScript numeric memory |
| `ArrayBuffer` | Raw block of binary memory underlying typed views |
| Transfer list | Moves buffer ownership between threads without copying |
| Counting sort | Linear-time sort using quantized keys and bucket counts |
| Prefix sum | Running total used to find each bucket's starting position |
| Premultiplied alpha | RGB already multiplied by alpha before blending |
| Frustum culling | Skipping objects believed to be outside the camera view |
| Backpressure | Preventing stale sort requests from piling up |

