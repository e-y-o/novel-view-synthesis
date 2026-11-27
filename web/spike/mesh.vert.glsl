#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

// Non-indexed quads: 6 vertices per quad, no vertex attributes. Quad and corner come from gl_VertexID.
uniform sampler2D uDepth;   // R16F metres, NEAREST, read with texelFetch
uniform int uCols;
uniform int uRows;
uniform vec2 uImg;          // source width, height in pixels
uniform float uF;           // focal length in source pixels
uniform mat3 uRt;           // R^T (uploaded row-major R, see main.ts): P_cam = uRt * (P - uT)
uniform vec3 uT;            // target camera position in the source frame
uniform float uTau;         // cut threshold on max(z)/min(z) over the quad's 4 corners
uniform float uAspect;      // canvasH / canvasW
uniform float uNear;
uniform float uFar;

out vec2 vUv;
flat out float vCut;

const ivec2 CORNER[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(0, 1), ivec2(1, 0), ivec2(1, 1));

float zAt(ivec2 g) {
  ivec2 px = ivec2(round(vec2(g) * (uImg - 1.0) / vec2(uCols, uRows)));
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
  vec2 pix = vec2(g) * (uImg - 1.0) / vec2(uCols, uRows);
  vec2 c = (uImg - 1.0) * 0.5;
  vec3 P = vec3((pix - c) / uF * z, z);          // OpenCV: x right, y down, z forward
  vec3 Pc = uRt * (P - uT);

  vUv = (pix + 0.5) / uImg;                      // image row 0 = top, texture row 0 = first uploaded row: no flip
  float halfH = uImg.y * 0.5;
  vec2 xy = vec2(Pc.x * uAspect, -Pc.y) * uF / halfH;   // y down -> GL y up
  float zc = (uFar + uNear) / (uFar - uNear) * Pc.z - 2.0 * uFar * uNear / (uFar - uNear);
  gl_Position = vec4(xy, zc, Pc.z);
}
