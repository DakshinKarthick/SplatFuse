/**
 * Phase 1 · sortWorker  (runs in a Web Worker — a background thread)
 * ============================================================================
 * WHY SORT AT ALL
 *   Splats are see-through, and our "over" blend math (see SplatMaterial.js) only
 *   composites correctly when they're drawn FAR → NEAR. As the camera moves, what
 *   counts as "far" changes, so the draw order has to be recomputed on every move.
 *
 * WHY A WORKER
 *   Sorting millions of items takes long enough to freeze the page if done on the
 *   main thread — the tab would stutter every time you drag. A Web Worker is a
 *   separate OS thread: main.js posts the camera here, we sort, we post the result
 *   back, and the UI never blocks. (TJ 46 — moving heavy work off the main thread.)
 *
 * WHY WE REORDER THE ACTUAL DATA (not just return indices)
 *   WebGL instancing has no "draw in this order" input — instances are drawn in
 *   the exact order their attribute buffers are laid out in memory. So to change
 *   draw order we must physically PERMUTE the per-splat arrays. We do that here
 *   so the main thread only has to upload the finished, correctly-ordered buffers.
 *
 * THE ALGORITHM: COUNTING SORT (O(n), not O(n log n))
 *   A comparator `array.sort((a,b)=>…)` on 8.8M items calls the JS comparator tens
 *   of millions of times → ~10s+, unusable. Counting sort avoids comparisons
 *   entirely: quantize each splat's depth into one of 65536 buckets, count how
 *   many land in each bucket, turn those counts into start offsets, then scatter
 *   each splat straight to its slot. Two linear passes → a few hundred ms.
 *   65536 buckets = one per ~0.0015% of the depth range, far finer than any
 *   blending artifact you could see. Later GPU-sort upgrade: TJ 41 (GPGPU).
 * ============================================================================
 */

const BUCKETS = 65536 // 2^16 depth buckets

// The worker keeps its OWN copy of the unsorted splat data (sent once via 'init'),
// so each 'sort' only needs the new camera matrix, not the whole scene again.
let positions = null // Float32Array(count*3), original order
let scales = null // Float32Array(count)
let colors = null // Float32Array(count*3)
let opacities = null // Float32Array(count)
let count = 0

// Messages arrive from main.js here. `e.data` is whatever it postMessage'd.
self.onmessage = (e) => {
  const msg = e.data

  // One-time setup: stash the unsorted arrays. (main.js TRANSFERS these buffers,
  // so ownership moves here with no copy — see the [buffers] arg on its postMessage.)
  if (msg.type === 'init') {
    positions = msg.positions
    scales = msg.scales
    colors = msg.colors
    opacities = msg.opacities
    count = positions.length / 3
    return
  }

  if (msg.type !== 'sort') return
  const m = msg.viewMatrix // a 16-element model-view matrix, column-major

  // --- pass 1: compute each splat's view-space depth, and the min/max range ---
  // We only need the z-coordinate after the view transform. For a column-major
  // matrix, the z of (x,y,z,1) is: m[2]*x + m[6]*y + m[10]*z + m[14].
  const depth = new Float32Array(count)
  let minD = Infinity
  let maxD = -Infinity
  for (let i = 0; i < count; i++) {
    const i3 = i * 3
    const d = m[2] * positions[i3] + m[6] * positions[i3 + 1] + m[10] * positions[i3 + 2] + m[14]
    depth[i] = d
    if (d < minD) minD = d
    if (d > maxD) maxD = d
  }

  // --- pass 2: bucket each splat + tally bucket sizes -------------------------
  // Ascending depth is the order we want: the camera looks down -Z, so the
  // most-NEGATIVE z is FARTHEST, and ascending z = far → near.
  const range = maxD - minD || 1
  const quantScale = (BUCKETS - 1) / range
  const bucketOf = new Uint32Array(count)   // which bucket each splat fell in
  const counts = new Uint32Array(BUCKETS)   // how many splats per bucket
  for (let i = 0; i < count; i++) {
    const b = ((depth[i] - minD) * quantScale) | 0 // |0 = floor to int
    bucketOf[i] = b
    counts[b]++
  }

  // --- prefix sum: turn per-bucket counts into per-bucket START offsets --------
  // offsets[b] = index in the output where bucket b's items begin.
  const offsets = new Uint32Array(BUCKETS)
  let acc = 0
  for (let b = 0; b < BUCKETS; b++) {
    offsets[b] = acc
    acc += counts[b]
  }

  // --- pass 3: scatter every splat's data into its sorted slot -----------------
  const sortedPositions = new Float32Array(count * 3)
  const sortedScales = new Float32Array(count)
  const sortedColors = new Float32Array(count * 3)
  const sortedOpacities = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    const dst = offsets[bucketOf[i]]++ // take this bucket's next free slot, then advance it
    const s3 = i * 3
    const d3 = dst * 3
    sortedPositions[d3] = positions[s3]
    sortedPositions[d3 + 1] = positions[s3 + 1]
    sortedPositions[d3 + 2] = positions[s3 + 2]
    sortedScales[dst] = scales[i]
    sortedColors[d3] = colors[s3]
    sortedColors[d3 + 1] = colors[s3 + 1]
    sortedColors[d3 + 2] = colors[s3 + 2]
    sortedOpacities[dst] = opacities[i]
  }

  // Post the sorted buffers back. The second arg is the TRANSFER list: instead of
  // copying these (megabytes) across the thread boundary, ownership is handed to
  // the main thread — near-instant, but these arrays become unusable here after.
  self.postMessage(
    { positions: sortedPositions, scales: sortedScales, colors: sortedColors, opacities: sortedOpacities },
    [sortedPositions.buffer, sortedScales.buffer, sortedColors.buffer, sortedOpacities.buffer],
  )
}
