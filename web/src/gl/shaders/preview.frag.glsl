#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uColor;
uniform sampler2D uSky;     // R8, 255 = sky
uniform bool uShowSky;
uniform int uPass;          // 0: opaque quads (alpha = 1); 1: feathered quads (0 < alpha < 1), blended
in vec2 vUv;
flat in float vAlpha;
out vec4 outColor;

void main() {
  if (vAlpha <= 0.002) discard;                                // cut: nothing to draw
  if (!uShowSky && texture(uSky, vUv).r > 0.5) discard;        // sky hidden: the background colour shows
  vec3 rgb = texture(uColor, vUv).rgb;
  if (uPass == 0) {
    if (vAlpha < 0.998) discard;                               // handled by the feather pass
    outColor = vec4(rgb, 1.0);
  } else {
    if (vAlpha >= 0.998) discard;                              // already drawn opaque
    outColor = vec4(rgb * vAlpha, vAlpha);                     // premultiplied: ONE, ONE_MINUS_SRC_ALPHA
  }
}
