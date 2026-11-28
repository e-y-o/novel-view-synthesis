#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uColor;
uniform sampler2D uSky;     // R8, 255 = sky
uniform bool uShowSky;
in vec2 vUv;
flat in float vCut;
out vec4 outColor;

void main() {
  if (vCut > 0.5) discard;                                    // torn edge: quad spans a large depth jump
  if (!uShowSky && texture(uSky, vUv).r > 0.5) discard;       // sky hidden: the background colour shows
  outColor = vec4(texture(uColor, vUv).rgb, 1.0);
}
