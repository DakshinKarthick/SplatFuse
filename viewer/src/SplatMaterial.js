/**
 * Phase 1 · SplatMaterial
 * ============================================================================
 * A ShaderMaterial ties our two GLSL programs (vertex + fragment) to a set of
 * uniforms and GPU state flags. This factory just builds and configures it.
 *
 * THE BLENDING SETUP (the subtle part)
 *   Our fragment shader outputs PREMULTIPLIED alpha: gl_FragColor = (rgb*a, a).
 *   To composite many transparent splats correctly we want the standard "over"
 *   operator:   result = src + dst * (1 - srcAlpha).
 *   In WebGL terms that's:  blendSrc = ONE,  blendDst = ONE_MINUS_SRC_ALPHA.
 *   This math is only correct if splats are drawn BACK-TO-FRONT (far → near),
 *   which is exactly why sortWorker.js exists.
 *
 * WHY depthWrite:false
 *   Transparent things must NOT block each other in the depth buffer — a near
 *   splat writing depth would stop farther splats behind it from contributing.
 *   We still keep depthTest:true so splats are correctly hidden behind any
 *   opaque geometry (like the debug axes).
 *
 * Refs (Three.js Journey): TJ 27 (Shaders), TJ 28 (patterns), TJ 31 (materials).
 * ============================================================================
 */

import * as THREE from 'three'
// `?raw` (a Vite feature) imports the .glsl file as a plain string of source code,
// which is what ShaderMaterial wants.
import vertexShader from './shaders/splat.vert.glsl?raw'
import fragmentShader from './shaders/splat.frag.glsl?raw'

export function createSplatMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,

    // Uniforms = values shared by every vertex/fragment, editable from JS each
    // frame. main.js updates uViewport whenever the canvas resizes.
    uniforms: {
      uViewport: { value: new THREE.Vector2(1, 1) }, // device-pixel canvas size
      uMinPixels: { value: 1.5 },                    // splat on-screen size floor
    },

    transparent: true,   // put this in the transparent draw pass, enable blending
    depthWrite: false,   // don't let one splat occlude another in the depth buffer
    depthTest: true,     // but DO respect opaque geometry already drawn
    side: THREE.DoubleSide, // billboards always face us; never cull by triangle winding

    // Custom blend = the premultiplied "over" operator described above.
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,          // src (op) dst  where op = "+"
    blendSrc: THREE.OneFactor,                 // src * 1
    blendDst: THREE.OneMinusSrcAlphaFactor,    // dst * (1 - srcAlpha)
  })
}
