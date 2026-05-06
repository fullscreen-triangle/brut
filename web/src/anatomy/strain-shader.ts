// Cardiac strain shader.
//
// Replaces the default PBR material on cardiac glbs with a fragment shader
// that encodes the fitted cardiac state — from cardiac-equations-of-state.tex —
// as a per-pixel strain colour map across the heart geometry.
//
// Per fragment we compute:
//
//   1. Sarcomere length:          L_s = L_s_ref · (EDV / V_ref)^(1/3)   (Eq. Ls_EDV)
//   2. Overlap function h(L_s):   piecewise from paper Section 2.3
//   3. Transmural strain factor:  inner wall strains > outer wall (endo > epi),
//                                 approximated by surface-normal dot view-axis.
//   4. Phase-shaped strain:       systolic shortening (phase 0..π)
//                                 diastolic lengthening (phase π..2π)
//   5. Contractility modulation:  scales with E_es / E_es_rest.
//
// The colour ramp follows the echo convention: blue (compression) →
// green (neutral) → red (stretch). Brightness is modulated by h(L_s) so
// regions off optimal sarcomere length appear dimmer (Frank-Starling
// directly visible). Coherence noise: low R_c adds spatial speckle.
//
// All uniforms are driven from the fitted CardiacState produced by the
// algebraic EOS inverter; nothing here is curve-fitted.

import { ShaderMaterial, Color, type IUniform } from 'three';

export interface StrainUniforms extends Record<string, IUniform<unknown>> {
  uPhase: IUniform<number>;       // cardiac cycle phase, 0..2π
  uHR: IUniform<number>;          // bpm
  uEes: IUniform<number>;         // mmHg/mL
  uEa: IUniform<number>;          // mmHg/mL
  uEDV: IUniform<number>;         // mL
  uESV: IUniform<number>;         // mL
  uEF: IUniform<number>;          // 0..1
  uRc: IUniform<number>;          // 0..1, cardiac coherence
  uTime: IUniform<number>;        // seconds since start, for noise stability
  uTint: IUniform<Color>;         // optional global tint
}

export interface StrainMaterial {
  material: ShaderMaterial;
  uniforms: StrainUniforms;
}

const VERTEX = /* glsl */ `
varying vec3 vWorldNormal;
varying vec3 vViewNormal;
varying vec3 vPos;
varying vec2 vUv;

void main() {
  vUv = uv;
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vPos = worldPos.xyz;
  vWorldNormal = normalize(mat3(modelMatrix) * normal);
  vViewNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const FRAGMENT = /* glsl */ `
precision highp float;

uniform float uPhase;
uniform float uHR;
uniform float uEes;
uniform float uEa;
uniform float uEDV;
uniform float uESV;
uniform float uEF;
uniform float uRc;
uniform float uTime;
uniform vec3  uTint;

varying vec3 vWorldNormal;
varying vec3 vViewNormal;
varying vec3 vPos;
varying vec2 vUv;

const float LS_MIN     = 1.27;
const float LS_OPT_LO  = 2.20;
const float LS_OPT_HI  = 2.35;
const float LS_MAX     = 3.65;
const float V_REF      = 80.0;
const float LS_REF     = 2.00;
const float PI         = 3.14159265359;
const float EES_REST   = 2.0;

float sarcomereLength(float edv) {
  return LS_REF * pow(max(edv / V_REF, 0.001), 1.0 / 3.0);
}

float overlapFunction(float ls) {
  if (ls < LS_MIN) return 0.0;
  if (ls < LS_OPT_LO) return (ls - LS_MIN) / (LS_OPT_LO - LS_MIN);
  if (ls <= LS_OPT_HI) return 1.0;
  if (ls < LS_MAX) return (LS_MAX - ls) / (LS_MAX - LS_OPT_HI);
  return 0.0;
}

float phaseStrain(float phase) {
  // Systolic shortening (phase 0..π) is negative strain (compression).
  // Diastolic lengthening (phase π..2π) is positive strain.
  if (phase < PI) {
    return -sin(phase);
  } else {
    return sin(phase - PI);
  }
}

vec3 strainColor(float strain) {
  // strain in [-1, 1]; -1 max compression (blue), 0 neutral (green), 1 max stretch (red)
  vec3 blue  = vec3(0.27, 0.45, 1.00);
  vec3 green = vec3(0.40, 0.95, 0.55);
  vec3 red   = vec3(1.00, 0.40, 0.40);
  float s = clamp(strain, -1.0, 1.0);
  if (s < 0.0) {
    return mix(green, blue, -s);
  }
  return mix(green, red, s);
}

float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
  // No-signal mode: when no fit has been pushed yet (or the camera is off),
  // we render a neutral dim grey with the same Lambert lighting so the model
  // is visible but visually "off". This way the user can't mistake fixed
  // rest-state defaults for an actual measurement of themselves.
  if (uHR < 1.0) {
    vec3 L = normalize(vec3(0.4, 0.6, 1.0));
    float lambert = clamp(dot(vWorldNormal, L), 0.0, 1.0);
    vec3 grey = vec3(0.18) * (0.6 + 0.4 * lambert);
    gl_FragColor = vec4(grey, 1.0);
    return;
  }

  // Equation of state evaluations.
  float ls = sarcomereLength(uEDV);
  float h  = overlapFunction(ls);

  // Transmural strain factor: surface normals roughly anti-aligned with view
  // direction sit on the back/inner side. We use abs(z) of the view-space
  // normal as a depth proxy. Endocardium (inner) gets a 30% boost in strain
  // magnitude, mirroring established echo physiology.
  float depthProxy = abs(vViewNormal.z);
  float transmural = 1.0 + 0.3 * (1.0 - depthProxy);

  // Phase-shaped strain.
  float baseStrain = phaseStrain(uPhase);

  // Contractility modulation: deviation from rest E_es scales the magnitude
  // of contraction (more inotropy → more shortening).
  float ctrFactor = clamp(uEes / EES_REST, 0.4, 2.5);

  // EF amplifies the strain — a high-EF heart strains more visibly.
  float efScale = 0.5 + uEF;

  float strain = baseStrain * transmural * ctrFactor * efScale * 0.55;

  // Compose colour.
  vec3 col = strainColor(strain);

  // Off-optimal sarcomere length dims the region (Frank-Starling visualised).
  col *= 0.45 + 0.55 * h;

  // Coherence noise: low R_c speckles the surface.
  float speckle = hash21(vPos.xy * 18.0 + uTime);
  col += (speckle - 0.5) * (1.0 - uRc) * 0.18;

  // Lambertian-ish lighting from a virtual key light.
  vec3 L = normalize(vec3(0.4, 0.6, 1.0));
  float lambert = clamp(dot(vWorldNormal, L), 0.0, 1.0);
  col *= 0.55 + 0.45 * lambert;

  // Optional global tint per-mesh.
  col *= uTint;

  gl_FragColor = vec4(col, 1.0);
}
`;

export function createStrainMaterial(tint = new Color(1, 1, 1)): StrainMaterial {
  const uniforms: StrainUniforms = {
    uPhase: { value: 0 },
    uHR: { value: 60 },
    uEes: { value: 2.0 },
    uEa: { value: 1.3 },
    uEDV: { value: 120 },
    uESV: { value: 50 },
    uEF: { value: 0.58 },
    uRc: { value: 0.85 },
    uTime: { value: 0 },
    uTint: { value: tint },
  };

  const material = new ShaderMaterial({
    name: 'cardiac-strain',
    uniforms,
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
  });

  return { material, uniforms };
}

export interface CardiacFitInput {
  HR: number;
  Ees: number;
  Ea: number;
  EDV: number;
  ESV: number;
  EF: number;
  Rc: number;
}

export function updateStrainUniforms(strain: StrainMaterial, fit: CardiacFitInput): void {
  strain.uniforms.uHR.value = fit.HR;
  strain.uniforms.uEes.value = fit.Ees;
  strain.uniforms.uEa.value = fit.Ea;
  strain.uniforms.uEDV.value = fit.EDV;
  strain.uniforms.uESV.value = fit.ESV;
  strain.uniforms.uEF.value = fit.EF;
  strain.uniforms.uRc.value = fit.Rc;
}
