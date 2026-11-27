#version 300 es
precision highp float;

uniform sampler2D uColor;
in vec2 vUv;
flat in float vCut;
out vec4 outColor;

void main() {
  if (vCut > 0.5) discard;   // torn edge: quad spans a large depth jump
  outColor = vec4(texture(uColor, vUv).rgb, 1.0);
}
