#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

// Non-indexed quads, 6 vertices each, no vertex attributes: quad and corner come from gl_VertexID.
// The mesh covers the whole image extent [-0.5, W-0.5] x [-0.5, H-0.5] (pixel centres at integers).
uniform sampler2D uDepth;   // R16F metres, NEAREST, read with texelFetch
uniform int uCols;
uniform int uRows;
uniform ivec2 uImgI;        // preview width, height (integers)
uniform vec2 uImg;          // the same as floats
uniform float uF;           // focal length in preview pixels
uniform mat3 uRt;           // R^T: the app uploads row-major R, which GLSL reads as its transpose
uniform vec3 uT;            // target camera position in the source frame [m]
uniform vec2 uClipScale;    // 2 * scale * f / canvas size (see gl/layout.ts clipScale)
uniform float uTau;         // cut threshold on max(z)/min(z) over the quad's four corners
uniform float uNear;
uniform float uFar;

out vec2 vUv;
flat out float vCut;

const ivec2 CORNER[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(0, 1), ivec2(1, 0), ivec2(1, 1));

// Depth texel for a mesh grid coordinate: clamp(floor(g * size / cols)), integer arithmetic (see gridTexel).
float zAt(ivec2 g) {
  ivec2 px = clamp((g * uImgI) / ivec2(uCols, uRows), ivec2(0), uImgI - 1);
  return texelFetch(uDepth, px, 0).r;
}

void main() {
  int q = gl_VertexID / 6;
  int k = gl_VertexID - q * 6;
  ivec2 qc = ivec2(q % uCols, q / uCols);
  ivec2 g = qc + CORNER[k];

  float z00 = zAt(qc);
  float z10 = zAt(qc + ivec2(1, 0));
  float z01 = zAt(qc + ivec2(0, 1));
  float z11 = zAt(qc + ivec2(1, 1));
  float zmax = max(max(z00, z10), max(z01, z11));
  float zmin = min(min(z00, z10), min(z01, z11));
  vCut = (zmax / zmin > uTau) ? 1.0 : 0.0;

  float z = zAt(g);
  vec2 edge = vec2(g) * uImg / vec2(uCols, uRows);   // position in edge coordinates: 0 .. W
  vec2 pix = edge - 0.5;                              // image pixel coordinates (centres at integers)
  vec2 c = (uImg - 1.0) * 0.5;
  vec3 P = vec3((pix - c) / uF * z, z);               // OpenCV: x right, y down, z forward
  vec3 Pc = uRt * (P - uT);

  vUv = edge / uImg;                                  // texture row 0 is the image's top row: no flip
  vec2 xy = uClipScale * vec2(Pc.x, -Pc.y);           // y down -> GL y up
  float zc = (uFar + uNear) / (uFar - uNear) * Pc.z - 2.0 * uFar * uNear / (uFar - uNear);
  gl_Position = vec4(xy, zc, Pc.z);
}
