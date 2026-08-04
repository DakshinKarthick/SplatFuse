// ============================================================================
// Phase 1 · splat VERTEX shader
// ============================================================================
// Runs once per vertex, on the GPU. Our geometry is a single tiny square (a
// "quad", 4 corners) that is INSTANCED once per gaussian. So this shader runs
// 4 × (number of splats) times. Its whole job: take the shared quad and, for
// each instance, move + size it so it becomes a camera-facing card sitting where
// that gaussian lives — a "billboard".
//
// Coordinate spaces to keep straight (three.js gives us matrices between them):
//   local/model  → the raw quad corners (-1..1)
//   world        → after the object's transform
//   view (eye)   → after the camera's transform; camera at origin looking down -Z
//   clip         → after the projection; GPU divides by w to get the final screen pos
//
// three.js auto-declares these for a ShaderMaterial, so we don't redeclare them:
//   attribute vec3 position;      // the current quad corner, e.g. (-1,-1,0)
//   uniform mat4 modelViewMatrix; // model → view  (world transform + camera)
//   uniform mat4 projectionMatrix;// view  → clip
// Ref: TJ 17 (Particles), TJ 27 (Shaders).
// ============================================================================

// Per-INSTANCE attributes: one value per gaussian (set as InstancedBufferAttribute).
attribute vec3 aCenter;   // this gaussian's world-space center
attribute float aScale;   // this gaussian's (isotropic) radius in world units
attribute vec3 aColor;    // its RGB
attribute float aOpacity; // its alpha (0..1)

// Uniforms: same for every vertex this draw call.
uniform vec2 uViewport;   // drawing-buffer size in device pixels (w, h)
uniform float uMinPixels; // never let a splat get smaller than this on screen

// Varyings: values we compute here and the GPU interpolates across the quad,
// handing the blended result to the fragment shader for each pixel.
varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;       // which corner (-1..1) this is → the fragment uses it to fade from center

void main() {
  // `position` is the quad corner in local space: one of (±1, ±1, 0). We forward
  // it as vQuad so the fragment shader knows how far this pixel is from the
  // splat's center (0,0 = center, corners = ±1).
  vQuad = position.xy;
  vColor = aColor;
  vOpacity = aOpacity;

  // Put the gaussian's CENTER into view space. In view space the camera sits at
  // the origin looking down -Z, and the X/Y axes are exactly screen-right and
  // screen-up. That's the trick that makes billboarding trivial: to face a card
  // at the camera we just nudge it along view-space X/Y — no need to dig the
  // camera's right/up vectors out of a matrix ourselves.
  vec4 cam = modelViewMatrix * vec4(aCenter, 1.0);

  // ---- minimum on-screen size ---------------------------------------------
  // Gaussians are physically tiny (a few mm). From any distance most would
  // project to well under one pixel and vanish. So: figure out how many pixels
  // this splat's world radius WOULD cover at its current depth, and if that's
  // below uMinPixels, scale it up to that floor.
  float w = max(-cam.z, 1.0e-6);           // perspective depth (=clip w); -z because we look down -Z
  float focalY = projectionMatrix[1][1];   // vertical focal length = 1 / tan(fovY/2)
  // projected pixel radius = worldRadius * focal / depth * (halfViewportHeight)
  float pxRadius = aScale * focalY / w * (uViewport.y * 0.5);
  // dilation factor k ≥ 1: how much to enlarge so pxRadius reaches the floor.
  float k = max(pxRadius, uMinPixels) / max(pxRadius, 1.0e-6);

  // Offset the center along view X/Y by the (possibly dilated) corner to expand
  // the point into a screen-facing quad.
  cam.xy += vQuad * aScale * k;

  // Finally project view space → clip space. The GPU divides by w after this.
  gl_Position = projectionMatrix * cam;
}
