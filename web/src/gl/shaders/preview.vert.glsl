#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

// Non-indexed quads, 6 vertices each, no vertex attributes: quad and corner come from gl_VertexID.
// The mesh covers the whole image extent [-0.5, W-0.5] x [-0.5, H-0.5] (pixel centres at integers).
//
// Per quad the vertex shader decides how much of it to draw (vAlpha, flat):
//   mode 0 "tau":     legacy global test, cut when max(z)/min(z) over the corners exceeds uTau (T09).
//   mode 1 "stretch": adaptive. Measures how much the quad is stretched in the target view compared with
//                     what the same surface would cover there: for each quad edge, projected length divided by
//                     cell * (z / zt) (the cell size scaled by the change in screen scale of its endpoints).
//                     A smooth surface, however steep, stays near 1; a quad that spans a depth jump stretches
//                     as soon as the camera moves; at the identity pose every quad is exactly 1, so nothing
//                     is cut where the photo is shown as taken. alpha = 1 - smoothstep(fade, cut, stretch).
// Quads entirely inside the sky are never cut. Optionally (uSkyCut) quads straddling sky and non-sky are.
uniform sampler2D uDepth;   // R16F metres, NEAREST, read with texelFetch
uniform sampler2D uSky;     // R8, 255 = sky
uniform int uCols;
uniform int uRows;
uniform ivec2 uImgI;        // preview width, height (integers)
uniform vec2 uImg;          // the same as floats
uniform float uF;           // focal length in preview pixels
uniform mat3 uRt;           // R^T: the app uploads row-major R, which GLSL reads as its transpose
uniform vec3 uT;            // target camera position in the source frame [m]
uniform vec2 uClipScale;    // 2 * scale * f / canvas size (see gl/layout.ts clipScale)
uniform int uMode;          // 0 = tau, 1 = stretch
uniform float uTau;         // mode 0: cut threshold on max(z)/min(z)
uniform float uFade;        // mode 1: stretch where fading starts
uniform float uCutAt;       // mode 1: stretch where the quad is gone
uniform float uSmoothK;     // mode 1: threshold multiplier for quads on a continuous surface (>= 1)
uniform bool uFeather;      // mode 1: soft alpha between uFade and uCutAt (false: hard cut at uCutAt)
uniform bool uSkyCut;       // always cut quads that straddle sky and non-sky
uniform float uNear;
uniform float uFar;

out vec2 vUv;
flat out float vAlpha;

const ivec2 CORNER[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(0, 1), ivec2(1, 0), ivec2(1, 1));

// Depth/sky texel for a mesh grid coordinate: clamp(floor(g * size / cols)), integer arithmetic (see gridTexel).
ivec2 texelOf(ivec2 g) {
  return clamp((g * uImgI) / ivec2(uCols, uRows), ivec2(0), uImgI - 1);
}

// Camera-space position (target frame) of the mesh vertex g whose depth is z.
vec3 camPos(ivec2 g, float z) {
  vec2 pix = vec2(g) * uImg / vec2(uCols, uRows) - 0.5;   // image pixel coordinates (centres at integers)
  vec2 c = (uImg - 1.0) * 0.5;
  vec3 P = vec3((pix - c) / uF * z, z);                   // OpenCV: x right, y down, z forward
  return uRt * (P - uT);
}

// Projected length of the edge a-b in preview pixels over the length a surface would have there.
float edgeStretch(vec3 a, vec3 b, float za, float zb, float cell) {
  if (a.z < uNear || b.z < uNear) return 1.0e6;           // behind the camera
  float len = length(uF * (a.xy / a.z - b.xy / b.z));
  float expected = cell * 0.5 * (za / a.z + zb / b.z);
  return len / expected;
}

void main() {
  int q = gl_VertexID / 6;
  int k = gl_VertexID - q * 6;
  ivec2 qc = ivec2(q % uCols, q / uCols);
  ivec2 g = qc + CORNER[k];

  ivec2 g00 = qc;
  ivec2 g10 = qc + ivec2(1, 0);
  ivec2 g01 = qc + ivec2(0, 1);
  ivec2 g11 = qc + ivec2(1, 1);
  float z00 = texelFetch(uDepth, texelOf(g00), 0).r;
  float z10 = texelFetch(uDepth, texelOf(g10), 0).r;
  float z01 = texelFetch(uDepth, texelOf(g01), 0).r;
  float z11 = texelFetch(uDepth, texelOf(g11), 0).r;

  float alpha = 1.0;
  if (uMode == 0) {
    float zmax = max(max(z00, z10), max(z01, z11));
    float zmin = min(min(z00, z10), min(z01, z11));
    alpha = (zmax / zmin > uTau) ? 0.0 : 1.0;
  } else {
    float s00 = texelFetch(uSky, texelOf(g00), 0).r;
    float s10 = texelFetch(uSky, texelOf(g10), 0).r;
    float s01 = texelFetch(uSky, texelOf(g01), 0).r;
    float s11 = texelFetch(uSky, texelOf(g11), 0).r;
    float skyMax = max(max(s00, s10), max(s01, s11));
    float skyMin = min(min(s00, s10), min(s01, s11));
    if (skyMin > 0.5) {
      alpha = 1.0;                                         // fully inside the sky: a flat distant surface
    } else if (uSkyCut && skyMax > 0.5) {
      alpha = 0.0;                                         // sky boundary
    } else {
      float cell = uImg.x / float(uCols);
      vec3 p00 = camPos(g00, z00);
      vec3 p10 = camPos(g10, z10);
      vec3 p01 = camPos(g01, z01);
      vec3 p11 = camPos(g11, z11);
      float stretch = max(max(edgeStretch(p00, p10, z00, z10, cell), edgeStretch(p01, p11, z01, z11, cell)),
                          max(edgeStretch(p00, p01, z00, z01, cell), edgeStretch(p10, p11, z10, z11, cell)));
      // A surface with no depth jump may legitimately stretch a lot (the side of a pole seen at an angle);
      // a quad across a depth jump may not. Blend the thresholds by the quad's depth ratio.
      float zmax = max(max(z00, z10), max(z01, z11));
      float zmin = min(min(z00, z10), min(z01, z11));
      float k = mix(uSmoothK, 1.0, smoothstep(1.04, 1.2, zmax / zmin));
      float fadeAt = uFade * k;
      float cutAt = uCutAt * k;
      alpha = uFeather ? 1.0 - smoothstep(fadeAt, cutAt, stretch) : (stretch > cutAt ? 0.0 : 1.0);
    }
  }
  vAlpha = alpha;

  float z = texelFetch(uDepth, texelOf(g), 0).r;
  vec2 edge = vec2(g) * uImg / vec2(uCols, uRows);         // position in edge coordinates: 0 .. W
  vec3 Pc = camPos(g, z);
  vUv = edge / uImg;                                       // texture row 0 is the image's top row: no flip
  vec2 xy = uClipScale * vec2(Pc.x, -Pc.y);                // y down -> GL y up
  float zc = (uFar + uNear) / (uFar - uNear) * Pc.z - 2.0 * uFar * uNear / (uFar - uNear);
  gl_Position = vec4(xy, zc, Pc.z);
}
