// ============================================================================
// Phase 1 · splat FRAGMENT shader
// ============================================================================
// Runs once per PIXEL that each quad covers. Its job: turn the hard square quad
// into a soft, round, fuzzy dot — a 2D approximation of a gaussian — and output
// a color + alpha that will blend with whatever is already behind it.
// Ref: TJ 28 (Shader patterns — distance to center), TJ 31 (Modified materials).
// ============================================================================

// Interpolated across the quad from the vertex shader. vQuad is (0,0) at the
// splat's center and reaches the unit circle / corners near the edges.
varying vec3 vColor;
varying float vOpacity;
varying vec2 vQuad;

void main() {
  // dot(vQuad, vQuad) = squared distance from the center (0 at center, grows out).
  // exp(-4 * r^2) is a Gaussian bump: 1.0 at the center, fading smoothly to ~0 at
  // the edges. Multiplying by the splat's opacity gives this pixel's alpha — so
  // the dot is bright/solid in the middle and feathers out, instead of a hard square.
  float a = vOpacity * exp(-4.0 * dot(vQuad, vQuad));

  // Fully-transparent-ish pixels contribute nothing but still cost blending, and
  // discarding them keeps the far corners of every quad from darkening the image.
  if (a < 0.003) discard;

  // PREMULTIPLIED alpha: we output color already multiplied by alpha (vColor * a),
  // and alpha in .a. This pairs with the material's blend mode ONE / ONE_MINUS_SRC_ALPHA
  // to do correct "over" compositing when thousands of these stack up. (It's also
  // why the splats MUST be drawn far→near — see sortWorker.js.)
  gl_FragColor = vec4(vColor * a, a);
}
